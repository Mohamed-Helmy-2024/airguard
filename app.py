"""AirGuard backend (Phase 1) - simple single-file version.

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager, closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import httpx
import joblib
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

BASE = Path(__file__).resolve().parent          # the folder where app.py is
# Vercel deployments use temporary storage; persistent data uses Upstash.
DB_FILE = Path("/tmp/airguard.db") if os.getenv("VERCEL") else BASE / "airguard.db"
DEVICE_TIMEOUT_SECONDS = float(os.getenv("DEVICE_TIMEOUT_SECONDS", 50))

# Pico fields match the training column names.
PICO_FIELDS = ["rms", "p2p", "level_db", "freq_hz", "freq_reliable",
               "centroid_hz", "raw", "difference", "pressure_kpa"]

FEATURES = ["rms", "p2p", "level_db", "freq_hz", "freq_reliable", "centroid_hz",
            "raw", "difference", "pressure_kpa", "delta_pressure_kpa", "delta_freq_hz"]

# Preserve the model's training order.
MODEL_FEATURES = ["rms", "p2p", "level_db", "freq_hz", "centroid_hz",
                  "pressure_kpa", "delta_pressure_kpa"]

# Accepted normalized model labels.
LEAK_NAMES = {"leakage", "leak", "leaking"}
NORMAL_NAMES = {"not leakage", "no leakage", "non leakage", "not leak", "no leak", "normal"}


def clean(name) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def now_ts() -> str:
    """Current UTC time as ISO 8601 text."""
    return datetime.now(timezone.utc).strftime(TS_FORMAT)


def to_dt(ts: str) -> datetime:
    return datetime.strptime(ts, TS_FORMAT).replace(tzinfo=timezone.utc)


def db():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with closing(db()) as con, con:
        con.execute("""CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, device_id TEXT,
            rms REAL, p2p INTEGER, level_db REAL, freq_hz REAL, freq_reliable INTEGER,
            centroid_hz REAL, raw INTEGER, difference INTEGER, pressure_kpa REAL,
            delta_pressure_kpa REAL, delta_freq_hz REAL,
            prediction TEXT, is_test INTEGER, event_id INTEGER)""")
        con.execute("""CREATE TABLE IF NOT EXISTS leakage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_timestamp TEXT, end_timestamp TEXT, duration_seconds REAL,
            status TEXT, sample_count INTEGER DEFAULT 0, is_test INTEGER)""")
        con.execute("CREATE INDEX IF NOT EXISTS i_read_ts ON sensor_readings(timestamp)")


class Model:
    """Finds and loads the 3 trained files and checks them. Never makes up a prediction."""
    NAMES = {"svm": ["svc_model", "svm_model", "svm", "svc", "model"],
             "scaler": ["scaler", "standard_scaler"],
             "encoder": ["label_encoder", "encoder"]}

    def __init__(self):
        self.missing, self.error, self.found = ["not loaded yet"], None, {}
        self.order = list(MODEL_FEATURES)      # column order given to the scaler
        self.pipeline_has_scaler = False
        self.folders = [BASE / "models", Path.cwd() / "models", BASE]
        self.load()

    def load(self):
        """Looks in several folders. Can be called again without restarting the server."""
        found = {}
        for key, stems in self.NAMES.items():
            for folder in self.folders:
                for stem in stems:
                    for ext in (".pkl", ".joblib", ".sav"):
                        path = folder / (stem + ext)
                        if key not in found and path.exists():
                            found[key] = path
        self.found = {k: str(v) for k, v in found.items()}
        self.missing = [k for k in self.NAMES if k not in found]
        if self.missing:
            return
        try:
            self.svm = joblib.load(found["svm"])
            self.scaler = joblib.load(found["scaler"])
            self.encoder = joblib.load(found["encoder"])
            self.check()
            self.error = None
        except Exception as e:                      # version mismatch, wrong columns, wrong labels...
            self.error = repr(e)
            self.missing = ["load_error"]

    def check(self):
        """Stops the model from running with wrong columns, wrong order or unknown labels."""
        classes = [clean(c) for c in self.encoder.classes_]
        bad = [c for c in classes if c not in LEAK_NAMES | NORMAL_NAMES]
        if bad or not any(c in LEAK_NAMES for c in classes):
            raise ValueError(f"label_encoder classes {list(self.encoder.classes_)} are not understood. "
                             "Add the names to LEAK_NAMES / NORMAL_NAMES in app.py")
        names = getattr(self.scaler, "feature_names_in_", None)
        if names is not None:
            names = [str(n) for n in names]
            unknown = [n for n in names if n not in FEATURES]
            if unknown:
                raise ValueError(f"Scaler was trained on columns {unknown} that the server does not have")
            self.order = names
        else:
            count = getattr(self.scaler, "n_features_in_", len(MODEL_FEATURES))
            if count != len(MODEL_FEATURES):
                raise ValueError(f"Scaler expects {count} features but MODEL_FEATURES has {len(MODEL_FEATURES)}")
            self.order = list(MODEL_FEATURES)
        # The scaler and SVM must come from the same training run.
        svm_n = getattr(self.svm, "n_features_in_", None)
        if svm_n is not None and svm_n != len(self.order):
            raise ValueError(
                f"MISMATCH: scaler file {self.found['scaler']} has {len(self.order)} columns {self.order} "
                f"but SVM file {self.found['svm']} expects {svm_n}. They come from different training runs: "
                "re-run ALL cells of train_model.ipynb, copy the 3 new .pkl files into models/, delete old copies")
        # Avoid scaling twice when the pipeline already includes a scaler.
        steps = getattr(self.svm, "steps", [])
        self.pipeline_has_scaler = any("scaler" in type(s).__name__.lower() for _, s in steps)

    def predict(self, values: dict):
        """values has every stored feature; only self.order is used. Returns 'Leakage' / 'Not Leakage'."""
        x = pd.DataFrame([[values[f] for f in self.order]], columns=self.order)
        if not self.pipeline_has_scaler:
            x = self.scaler.transform(x)
        out = self.svm.predict(x)[0]
        name = out if isinstance(out, str) else self.encoder.inverse_transform([out])[0]
        name = clean(name)
        if name in LEAK_NAMES:
            return "Leakage"
        if name in NORMAL_NAMES:
            return "Not Leakage"
        raise ValueError(f"Unknown class name from model: {name!r}")

    def info(self) -> dict:
        """What the server really loaded - used to find integration mistakes."""
        if self.missing:
            return {"ready": False, "missing": self.missing, "error": self.error}
        sc = self.scaler
        mean = getattr(sc, "mean_", None)
        scale = getattr(sc, "scale_", None)
        return {
            "ready": True,
            "svm_type": type(self.svm).__name__,
            "svm_pipeline_already_scales": self.pipeline_has_scaler,
            "svm_classes": [str(c) for c in getattr(self.svm, "classes_", [])],
            "encoder_classes": [str(c) for c in self.encoder.classes_],
            "scaler_type": type(sc).__name__,
            "scaler_has_column_names": getattr(sc, "feature_names_in_", None) is not None,
            "feature_order_used": self.order,
            "scaler_mean": dict(zip(self.order, [float(v) for v in mean])) if mean is not None else None,
            "scaler_scale": dict(zip(self.order, [float(v) for v in scale])) if scale is not None else None,
        }


model = Model()

last = {"seen": None, "timestamp": None, "prediction": None, "reading": None, "note": None}
clients: set[WebSocket] = set()
pump = {"enabled": False}         # local fallback; Vercel uses shared Redis state
warmup_until = 0.0
KV_URL = os.getenv("UPSTASH_REDIS_REST_URL", "").rstrip("/")
KV_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
READINGS_KEY = "airguard:readings"
EVENTS_KEY = "airguard:events"
HISTORY_DAYS = 30


def shared_store_ready() -> bool:
    return bool(KV_URL and KV_TOKEN)


async def shared_get(key: str):
    if not shared_store_ready():
        return None
    url = KV_URL + "/get/" + quote(key, safe="")
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(url, headers={"Authorization": "Bearer " + KV_TOKEN})
        response.raise_for_status()
        return response.json().get("result")


async def shared_command(command: list):
    if not shared_store_ready():
        raise RuntimeError("Upstash Redis is not configured")
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.post(
            KV_URL,
            json=command,
            headers={"Authorization": "Bearer " + KV_TOKEN},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        return payload.get("result")


async def shared_set(key: str, value: str):
    if not shared_store_ready():
        return False
    if await shared_command(["SET", key, value]) != "OK":
        raise RuntimeError("Redis did not confirm SET")
    return True


async def shared_delete(key: str):
    return await shared_command(["DEL", key])


async def shared_zadd_recent(key: str, timestamp: str, member: str):
    score = to_dt(timestamp).timestamp()
    cutoff = datetime.now(timezone.utc).timestamp() - HISTORY_DAYS * 86400
    script = (
        "redis.call('ZADD', KEYS[1], ARGV[1], ARGV[2]); "
        "redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[3]); return 1"
    )
    return await shared_command(["EVAL", script, "1", key, str(score), member, str(cutoff)])


async def shared_zremove(key: str, member: str):
    return await shared_command(["ZREM", key, member])


async def shared_zrange(key: str, start: datetime, end: datetime, limit: int = 5000):
    result = await shared_command([
        "ZRANGEBYSCORE", key, str(start.timestamp()), str(end.timestamp()),
        "LIMIT", "0", str(max(1, min(limit, 5000))),
    ])
    return result or []


def is_connected() -> bool:
    return last["seen"] is not None and time.monotonic() - last["seen"] <= DEVICE_TIMEOUT_SECONDS


def get_state():
    """Returns (state, reason). Offline always wins over the last prediction."""
    if not pump["enabled"]:
        return "off", "system_off"
    if time.time() < warmup_until:
        return "normal", "ok"
    if last["seen"] is None:
        return "offline", "no_data_yet"
    if not is_connected():
        return "offline", "device_timeout"
    if last["prediction"] is None:
        return "offline", (last["note"] if last["note"] != "system_off" else None) or "no_prediction"
    return ("leakage" if last["prediction"] == "Leakage" else "normal"), "ok"


def event_dict(row) -> dict:
    d = dict(row)
    if d["status"] == "active":   # still open -> duration so far
        d["duration_seconds"] = (datetime.now(timezone.utc) - to_dt(d["start_timestamp"])).total_seconds()
    return d


def snapshot(kind: str) -> dict:
    state, reason = get_state()
    with closing(db()) as con:
        ev = con.execute("SELECT * FROM leakage_events WHERE status='active'").fetchone()
        reading_row = con.execute("SELECT * FROM sensor_readings ORDER BY id DESC LIMIT 1").fetchone()
    reading = dict(reading_row) if reading_row else last["reading"]
    reading_timestamp = reading["timestamp"] if reading else last["timestamp"]
    prediction = reading["prediction"] if reading else last["prediction"]
    return {
        "type": kind,
        "server_time": now_ts(),
        "state": state,
        "state_reason": reason,
        "device_connected": is_connected(),
        "pump_enabled": pump["enabled"],
        "last_reading_timestamp": reading_timestamp,
        "prediction": prediction,
        "reading": reading,
        "active_event": event_dict(ev) if ev else None,
    }


async def broadcast(message: dict):
    for ws in list(clients):
        try:
            await ws.send_json(message)
        except Exception:
            clients.discard(ws)


async def watchdog():
    """Every second: if the state changed (e.g. timeout), tell the clients."""
    previous = get_state()
    while True:
        await asyncio.sleep(1)
        current = get_state()
        if current != previous:
            previous = current
            await broadcast(snapshot("state_change"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    # Keep persisted readings offline until the device sends a new one.
    with closing(db()) as con:
        r = con.execute("SELECT * FROM sensor_readings ORDER BY id DESC LIMIT 1").fetchone()
    if r:
        last.update(timestamp=r["timestamp"], prediction=r["prediction"], reading=dict(r))
    task = asyncio.create_task(watchdog())
    yield
    task.cancel()


app = FastAPI(title="AirGuard API", version="1.1-phase1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ReadingIn(BaseModel):
    """What the Pico sends. No timestamp (server adds it), no deltas (server calculates them)."""
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "rms": 212.4, "p2p": 780, "level_db": 46.5, "freq_hz": 1243.7,
            "freq_reliable": 1, "centroid_hz": 1530.2, "raw": 4012345,
            "difference": 502460, "pressure_kpa": 4.123,
            "device_id": "pico2w-01", "is_test": True}})

    rms: float = Field(ge=0, le=32768, allow_inf_nan=False)
    p2p: int = Field(ge=0, le=65535)
    level_db: float = Field(ge=-20, le=200, allow_inf_nan=False)
    freq_hz: float = Field(ge=0, le=4000, allow_inf_nan=False)
    freq_reliable: int = Field(ge=0, le=1)
    centroid_hz: float = Field(ge=0, le=4000, allow_inf_nan=False)
    raw: int = Field(ge=-8388608, le=8388607)
    difference: int = Field(ge=-16777216, le=16777216)
    pressure_kpa: float = Field(ge=0, le=40, allow_inf_nan=False)
    device_id: str = Field("pico2w-01", max_length=64)
    is_test: bool = False   # set True for any manual / fake data


@app.post("/api/readings", status_code=201)
async def post_reading(data: ReadingIn):
    global warmup_until
    values = {f: getattr(data, f) for f in PICO_FIELDS}
    shared_previous = None
    shared_active_raw = None

    if shared_store_ready():
        try:
            stored_pump = await shared_get("airguard:pump_enabled")
            if stored_pump is not None:
                pump["enabled"] = str(stored_pump).lower() == "true"
            stored_warmup = await shared_get("airguard:warmup_until")
            warmup_until = float(stored_warmup) if stored_warmup is not None else 0.0
            latest_raw = await shared_get("airguard:latest_reading")
            if latest_raw:
                candidate = json.loads(latest_raw)
                if (datetime.now(timezone.utc) - to_dt(candidate["timestamp"])).total_seconds() <= DEVICE_TIMEOUT_SECONDS:
                    shared_previous = candidate
            shared_active_raw = await shared_get("airguard:active_event")
        except Exception as e:
            print("Shared state read error:", repr(e))
            raise HTTPException(503, "Could not read shared AirGuard state")

    # Calculate deltas from the previous recent reading.
    previous = shared_previous or (last["reading"] if is_connected() else None)
    if previous is not None:
        values["delta_pressure_kpa"] = values["pressure_kpa"] - previous["pressure_kpa"]
        values["delta_freq_hz"] = values["freq_hz"] - previous["freq_hz"]
    else:
        values["delta_pressure_kpa"] = values["delta_freq_hz"] = 0.0

    label, note = None, (None if pump["enabled"] else "system_off")
    if model.missing:
        model.load()                                # try again, no restart needed
    if note == "system_off":
        pass                                        # system OFF: no prediction
    elif time.time() < warmup_until:
        note = "startup_warmup"
    elif model.missing:
        note = "model_unavailable"
    else:
        try:
            label = model.predict(values)
            if values["pressure_kpa"] > 38.0:   
                label = "Not Leakage"
        except Exception as e:
            print("Prediction error:", repr(e))
            note = "prediction_error"

    ts = now_ts()
    reading_id = int(to_dt(ts).timestamp() * 1_000_000)
    shared_active = json.loads(shared_active_raw) if shared_active_raw else None
    shared_event_update = None
    shared_event_close = False
    event_id = None
    if shared_store_ready():
        if label == "Leakage":
            shared_active = shared_active or {
                "id": reading_id, "start_timestamp": ts, "end_timestamp": None,
                "duration_seconds": None, "status": "active", "sample_count": 0,
                "is_test": int(data.is_test),
            }
            shared_active["sample_count"] += 1
            event_id = shared_active["id"]
            shared_event_update = shared_active
        elif label == "Not Leakage" and shared_active is not None:
            shared_active["end_timestamp"] = ts
            shared_active["duration_seconds"] = (to_dt(ts) - to_dt(shared_active["start_timestamp"])).total_seconds()
            shared_active["status"] = "closed"
            shared_event_update = shared_active
            shared_event_close = True

    with closing(db()) as con, con:   # one transaction
        cur = con.execute(
            f"INSERT INTO sensor_readings (id, timestamp, device_id, {', '.join(FEATURES)},"
            f" prediction, is_test) VALUES (?, ?, ?, {', '.join('?' * len(FEATURES))}, ?, ?)",
            (reading_id, ts, data.device_id, *[values[f] for f in FEATURES], label, int(data.is_test)))

        if shared_store_ready():
            if event_id is not None:
                con.execute("UPDATE sensor_readings SET event_id=? WHERE id=?", (event_id, reading_id))
        else:
            active = con.execute("SELECT * FROM leakage_events WHERE status='active'").fetchone()
            if label == "Leakage":
                if active is None:
                    cur = con.execute(
                        "INSERT INTO leakage_events (start_timestamp, status, sample_count, is_test)"
                        " VALUES (?, 'active', 0, ?)", (ts, int(data.is_test)))
                    event_id = cur.lastrowid
                else:
                    event_id = active["id"]
                con.execute("UPDATE leakage_events SET sample_count = sample_count + 1 WHERE id=?", (event_id,))
                con.execute("UPDATE sensor_readings SET event_id=? WHERE id=?", (event_id, reading_id))
            elif label == "Not Leakage" and active is not None:
                seconds = (to_dt(ts) - to_dt(active["start_timestamp"])).total_seconds()
                con.execute("UPDATE leakage_events SET status='closed', end_timestamp=?, duration_seconds=?"
                            " WHERE id=?", (ts, seconds, active["id"]))

        row = con.execute("SELECT * FROM sensor_readings WHERE id=?", (reading_id,)).fetchone()

    row_data = dict(row)
    if shared_store_ready():
        row_data["id"] = reading_id
        row_data["event_id"] = event_id
        if shared_event_update is not None:
            if shared_active_raw:
                await shared_zremove(EVENTS_KEY, shared_active_raw)
            event_json = json.dumps(shared_event_update, separators=(",", ":"), sort_keys=True)
            await shared_command([
                "ZADD", EVENTS_KEY,
                str(to_dt(shared_event_update["start_timestamp"]).timestamp()), event_json,
            ])
            if shared_event_close:
                await shared_delete("airguard:active_event")
            else:
                await shared_set("airguard:active_event", event_json)
        row_json = json.dumps(row_data, separators=(",", ":"))
        await shared_zadd_recent(READINGS_KEY, ts, row_json)
        await shared_set("airguard:latest_reading", row_json)

    last.update(seen=time.monotonic(), timestamp=ts, prediction=label, reading=row_data, note=note)
    await broadcast(snapshot("reading"))
    state, reason = get_state()
    return {"success": True, "reading_id": reading_id, "timestamp": ts,
            "prediction": label, "system_state": state, "state_reason": reason,
            "pump_enabled": pump["enabled"]}


@app.post("/api/test-predict")
async def test_predict(data: dict[str, float]):
    """Checks the model with the 7 training columns (e.g. a row from your CSV):
    rms, p2p, level_db, freq_hz, centroid_hz, pressure_kpa, delta_pressure_kpa.
    Nothing is saved or broadcast. Use it to compare with the known label."""
    if model.missing:
        model.load()
    if model.missing:
        raise HTTPException(503, f"model not ready: {model.missing} {model.error or ''}")
    missing = [f for f in model.order if f not in data]
    if missing:
        raise HTTPException(422, f"missing values: {missing}")
    try:
        return {"prediction": model.predict(data), "is_test": True}
    except Exception as e:
        raise HTTPException(500, f"prediction failed: {e!r}")


@app.get("/api/model-info")
async def model_info():
    """Shows what the server loaded, and how far the LATEST live reading is from the
    training data (z-score = how many training standard deviations away)."""
    info = model.info()
    reading = last["reading"]
    if info.get("scaler_mean") and info.get("scaler_scale") and reading:
        z = {}
        for f in model.order:
            v, scale = reading.get(f), info["scaler_scale"][f]
            if v is not None and scale:
                z[f] = round((v - info["scaler_mean"][f]) / scale, 2)
        info["latest_reading_zscores"] = z
        info["features_far_from_training_(abs_z_over_4)"] = [f for f, v in z.items() if abs(v) > 4]
    return info


class PumpIn(BaseModel):
    on: bool


@app.get("/api/pump")
async def get_pump():
    if shared_store_ready():
        try:
            stored = await shared_get("airguard:pump_enabled")
            if stored is not None:
                pump["enabled"] = str(stored).lower() == "true"
        except Exception as e:
            print("Shared pump state read error:", repr(e))
            raise HTTPException(503, "Could not read the shared pump command")
    return {"pump_enabled": pump["enabled"], "shared_store_ready": shared_store_ready()}


@app.post("/api/pump")
async def set_pump(data: PumpIn):
    """Dashboard button. The Pico reads pump_enabled from its next POST /api/readings reply."""
    global warmup_until
    if shared_store_ready():
        try:
            stored = await shared_get("airguard:pump_enabled")
            was_enabled = str(stored).lower() == "true" if stored is not None else pump["enabled"]
            if data.on and not was_enabled:
                warmup_until = time.time() + 10.0
                await shared_set("airguard:warmup_until", str(warmup_until))
            elif not data.on:
                warmup_until = 0.0
                await shared_delete("airguard:warmup_until")
            await shared_set("airguard:pump_enabled", "true" if data.on else "false")
        except Exception as e:
            print("Shared pump state write error:", repr(e))
            raise HTTPException(503, "Could not save the pump command to shared storage")
    elif data.on and not pump["enabled"]:
        warmup_until = time.time() + 10.0
    elif not data.on:
        warmup_until = 0.0
    pump["enabled"] = data.on
    await broadcast(snapshot("state_change"))
    return {"pump_enabled": pump["enabled"]}


@app.get("/api/latest")
async def latest():
    global warmup_until
    result = snapshot("snapshot")
    if shared_store_ready():
        try:
            stored_warmup = await shared_get("airguard:warmup_until")
            warmup_until = float(stored_warmup) if stored_warmup is not None else 0.0
            pump_state = await shared_get("airguard:pump_enabled")
            if pump_state is not None:
                result["pump_enabled"] = str(pump_state).lower() == "true"
            stored = await shared_get("airguard:latest_reading")
            if stored:
                reading = json.loads(stored)
                result["reading"] = reading
                result["last_reading_timestamp"] = reading["timestamp"]
                result["prediction"] = reading.get("prediction")
                age = (datetime.now(timezone.utc) - to_dt(reading["timestamp"])).total_seconds()
                result["device_connected"] = age <= DEVICE_TIMEOUT_SECONDS
                active_raw = await shared_get("airguard:active_event")
                result["active_event"] = event_dict(json.loads(active_raw)) if active_raw else None
                if not result["pump_enabled"]:
                    result["state"], result["state_reason"] = "off", "system_off"
                elif time.time() < warmup_until:
                    result["state"], result["state_reason"] = "normal", "ok"
                elif not result["device_connected"]:
                    result["state"], result["state_reason"] = "offline", "device_timeout"
                elif result["prediction"] is None:
                    result["state"], result["state_reason"] = "offline", "no_prediction"
                else:
                    result["state"] = "leakage" if result["prediction"] == "Leakage" else "normal"
                    result["state_reason"] = "ok"
            if result["pump_enabled"] and time.time() < warmup_until:
                result["state"], result["state_reason"] = "normal", "ok"
        except Exception as e:
            print("Shared latest reading error:", repr(e))
            raise HTTPException(503, "Could not read the shared latest reading")
    return result


def time_range(range_, start, end):
    """Returns (start_text, end_text). Custom start/end win over range."""
    now = datetime.now(timezone.utc)
    if start or end:
        a = start or now - timedelta(days=30)
        b = end or now
    else:
        a = now - timedelta(days={"24h": 1, "7d": 7, "30d": 30}[range_])
        b = now
    a = a if a.tzinfo else a.replace(tzinfo=timezone.utc)   # no timezone -> assume UTC
    b = b if b.tzinfo else b.replace(tzinfo=timezone.utc)
    if a > b:
        raise HTTPException(422, "start must be before end")
    return a.astimezone(timezone.utc).strftime(TS_FORMAT), b.astimezone(timezone.utc).strftime(TS_FORMAT)


@app.get("/api/history")
async def history(range: str = "24h", start: datetime | None = None,
                  end: datetime | None = None, limit: int = 5000):
    if range not in ("24h", "7d", "30d"):
        raise HTTPException(422, "range must be 24h, 7d or 30d")
    a, b = time_range(range, start, end)
    if shared_store_ready():
        members = await shared_zrange(READINGS_KEY, to_dt(a), to_dt(b), limit)
        rows = [json.loads(member) for member in members]
    else:
        with closing(db()) as con:
            rows = con.execute("SELECT * FROM sensor_readings WHERE timestamp BETWEEN ? AND ?"
                               " ORDER BY timestamp LIMIT ?", (a, b, limit)).fetchall()
        rows = [dict(r) for r in rows]
    return {"start": a, "end": b, "count": len(rows), "readings": rows}


@app.get("/api/leakage-events")
async def leakage_events(range: str = "24h", start: datetime | None = None,
                         end: datetime | None = None, include_readings: bool = False):
    if range not in ("24h", "7d", "30d"):
        raise HTTPException(422, "range must be 24h, 7d or 30d")
    a, b = time_range(range, start, end)
    if shared_store_ready():
        members = await shared_zrange(
            EVENTS_KEY, datetime.fromtimestamp(0, timezone.utc), to_dt(b), 5000)
        stored_events = [json.loads(member) for member in members]
        events = [event for event in stored_events
                  if event["start_timestamp"] <= b
                  and (event.get("end_timestamp") is None or event["end_timestamp"] >= a)]
        active_raw = await shared_get("airguard:active_event")
        if active_raw:
            active = json.loads(active_raw)
            if active["start_timestamp"] <= b and active["start_timestamp"] not in {e["start_timestamp"] for e in events}:
                events.append(event_dict(active))
        events.sort(key=lambda event: event["start_timestamp"])
        if include_readings:
            sensor_members = await shared_zrange(READINGS_KEY, to_dt(a), to_dt(b), 5000)
            readings = [json.loads(member) for member in sensor_members]
            for event in events:
                event["readings"] = [r for r in readings if r.get("event_id") == event["id"]]
    else:
        with closing(db()) as con:
            rows = con.execute(
                "SELECT * FROM leakage_events WHERE start_timestamp <= ?"
                " AND (end_timestamp IS NULL OR end_timestamp >= ?) ORDER BY start_timestamp",
                (b, a)).fetchall()
            events = [event_dict(r) for r in rows]
            if include_readings:
                for e in events:
                    rs = con.execute("SELECT * FROM sensor_readings WHERE event_id=? ORDER BY timestamp",
                                     (e["id"],)).fetchall()
                    e["readings"] = [dict(r) for r in rs]
    return {"start": a, "end": b,
            "event_count": len(events),
            "total_duration_seconds": sum(e["duration_seconds"] or 0 for e in events),
            "events": events}


@app.get("/api/health")
async def health():
    try:
        with closing(db()) as con:
            con.execute("SELECT 1")
        database = "ok"
    except Exception:
        database = "error"
    state, reason = get_state()
    shared_ok = shared_store_ready()
    if shared_ok:
        try:
            await shared_get("airguard:pump_enabled")
        except Exception as e:
            shared_ok = False
            print("Shared store health error:", repr(e))
    return {"status": "ok" if database == "ok" else "degraded", "database": database,
            "system_state": state, "state_reason": reason,
            "device_timeout_seconds": DEVICE_TIMEOUT_SECONDS,
            "model_ready": not model.missing, "missing_model_files": model.missing,
            "model_error": model.error, "files_found": model.found,
            "shared_store_ready": shared_store_ready(),
            "shared_store_ok": shared_ok,
            "vercel_environment": bool(os.getenv("VERCEL")),
            "folders_searched": [str(f) for f in model.folders],
            "websocket_clients": len(clients)}


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        await ws.send_json(snapshot("snapshot"))
        while True:
            await ws.receive_text()   # just keeps the connection open
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)
