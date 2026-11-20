# Wiring: LM393 VCC/GND to pins 36/33, A0 to GP26, optional D0 to GP14.
# HX710B DOUT/SCK to GP2/GP3; pump to GP15 through a driver and flyback diode.

from machine import ADC, Pin
from time import ticks_us, ticks_add, ticks_diff, ticks_ms, sleep_ms, sleep_us
from array import array
import math
import json
import network

try:
    import urequests as requests
except ImportError:
    import requests

WIFI_SSID = "medo"
WIFI_PASSWORD = "medo21/10/2004"

# Keep this API host in sync with frontend/script.js.
API_URL = "https://brick-breeches-diploma.ngrok-free.dev/api/readings"
PUMP_URL = API_URL.rsplit("/", 1)[0] + "/pump"
DEVICE_ID = "pico2w-01"
SEND_INTERVAL_MS = 1000

FS = 8000
N = 400                      # 50 ms window
CAL_OFFSET = 0.0
M = 512                      # FFT size (zero-padded)
MIN_FREQ = 50
FREQ_MIN_RMS = 150           # below this RMS the frequency is unreliable

ZERO_OFFSET = 3509885
ADC_MAX = 8388607
MAX_PRESSURE_KPA = 40.0
ADC_RANGE = ADC_MAX - ZERO_OFFSET
PRESS_AVG = 4
SENSOR_TIMEOUT_MS = 3000     # pressure reading becomes unavailable after prolonged sensor loss

LOOP_DELAY_MS = 100

PERIOD = 1000000 // FS
adc = ADC(Pin(26))
d0 = Pin(14, Pin.IN)

DOUT = Pin(2, Pin.IN)
SCK = Pin(3, Pin.OUT)
pump = Pin(15, Pin.OUT)
pump.value(0)
SCK.value(0)

buf = array('H', [0] * N)

HALF = M // 2
DF = FS / M
KMIN = int(MIN_FREQ / DF) + 1
RE = array('f', [0.0] * M)
IM = array('f', [0.0] * M)
COS = array('f', [math.cos(-2 * math.pi * k / M) for k in range(HALF)])
SIN = array('f', [math.sin(-2 * math.pi * k / M) for k in range(HALF)])
WIN = array('f', [0.5 - 0.5 * math.cos(2 * math.pi * i / (N - 1))
                  for i in range(N)])
REV = [0] * M
_j = 0
for _i in range(1, M):
    _b = M >> 1
    while _j & _b:
        _j ^= _b
        _b >>= 1
    _j ^= _b
    REV[_i] = _j


def capture():
    """Fill buf with N microphone samples."""
    nxt = ticks_us()
    for i in range(N):
        while ticks_diff(nxt, ticks_us()) > 0:
            pass
        buf[i] = adc.read_u16()
        nxt = ticks_add(nxt, PERIOD)


def fft():
    for i in range(M):
        j = REV[i]
        if i < j:
            RE[i], RE[j] = RE[j], RE[i]
            IM[i], IM[j] = IM[j], IM[i]
    size = 2
    while size <= M:
        half = size >> 1
        step = M // size
        for i in range(0, M, size):
            for k in range(half):
                wr = COS[k * step]
                wi = SIN[k * step]
                u = i + k
                v = u + half
                tr = wr * RE[v] - wi * IM[v]
                ti = wr * IM[v] + wi * RE[v]
                RE[v] = RE[u] - tr
                IM[v] = IM[u] - ti
                RE[u] += tr
                IM[u] += ti
        size <<= 1


def frequency(mean):
    """Return (dominant frequency, spectral centroid) in Hz."""
    for i in range(N):
        RE[i] = (buf[i] - mean) * WIN[i]
        IM[i] = 0.0
    for i in range(N, M):
        RE[i] = 0.0
        IM[i] = 0.0
    fft()

    pk = 0.0
    kpk = KMIN
    tot = 0.0
    wsum = 0.0
    for k in range(KMIN, HALF):
        p = RE[k] * RE[k] + IM[k] * IM[k]
        m = math.sqrt(p)
        tot += m
        wsum += m * k
        if p > pk:
            pk = p
            kpk = k

    dom = kpk * DF
    if 1 < kpk < HALF - 1:               # parabolic interpolation
        a = RE[kpk - 1] ** 2 + IM[kpk - 1] ** 2
        c = RE[kpk + 1] ** 2 + IM[kpk + 1] ** 2
        if a > 0 and c > 0 and pk > 0:
            la = math.log(a)
            lb = math.log(pk)
            lc = math.log(c)
            den = la - 2 * lb + lc
            if den != 0:
                dom = (kpk + 0.5 * (la - lc) / den) * DF

    centroid = (wsum / tot) * DF if tot > 0 else 0.0
    return dom, centroid


def measure_sound():
    """Capture one window and return the 6 sound values."""
    capture()
    mean = sum(buf) / N

    sum_sq = 0.0
    low = 65535
    high = 0
    for x in buf:
        diff = x - mean
        sum_sq += diff * diff
        if x < low:
            low = x
        if x > high:
            high = x

    rms = math.sqrt(sum_sq / N)
    level_db = 20 * math.log10(max(rms, 1.0)) + CAL_OFFSET
    dom, centroid = frequency(mean)

    return {
        "rms": rms,
        "p2p": high - low,
        "level_db": level_db,
        "freq_hz": dom,
        "freq_reliable": 1 if rms >= FREQ_MIN_RMS else 0,
        "centroid_hz": centroid,
    }


press_hist = []
last_ok = ticks_ms()


def read_hx710b():
    """Return a signed 24-bit value, or None if no new data yet."""
    if DOUT.value() == 1:
        return None

    data = 0
    for i in range(24):
        SCK.value(1)
        sleep_us(1)
        data = (data << 1) | DOUT.value()
        SCK.value(0)
        sleep_us(1)

    for i in range(3):                   # 27 pulses select differential input at 40 Hz
        SCK.value(1)
        sleep_us(1)
        SCK.value(0)
        sleep_us(1)

    if data & 0x800000:
        data -= 0x1000000
    return data


def measure_pressure():
    """Return the 3 pressure values, or None if the sensor is not working."""
    global last_ok

    value = read_hx710b()
    if value is not None:
        press_hist.append(value)
        if len(press_hist) > PRESS_AVG:
            press_hist.pop(0)
        last_ok = ticks_ms()

    if not press_hist or ticks_diff(ticks_ms(), last_ok) > SENSOR_TIMEOUT_MS:
        return None

    raw = sum(press_hist) / len(press_hist)
    difference = raw - ZERO_OFFSET
    pressure = (difference / ADC_RANGE) * MAX_PRESSURE_KPA
    pressure = max(0.0, min(MAX_PRESSURE_KPA, pressure))

    return {
        "raw": int(raw),
        "difference": int(difference),
        "pressure_kpa": pressure,
    }


def control_pump(pressure_kpa):
    """Apply the dashboard's ON/OFF command."""
    if not pump_enabled:
        pump.value(0)
        return "OFF (dashboard)"
    pump.value(1)
    return "ON (dashboard)"


wlan = network.WLAN(network.STA_IF)
pump_enabled = False         # fail-safe default; dashboard must explicitly turn it on


def connect_wifi():
    """Connect to Wi-Fi (waits up to 15 s). Returns True if connected."""
    wlan.active(True)
    if wlan.isconnected():
        return True

    print("Connecting to Wi-Fi...")
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    for _ in range(15):
        if wlan.isconnected():
            print("Wi-Fi OK, IP:", wlan.ifconfig()[0])
            return True
        sleep_ms(1000)

    print("Wi-Fi FAILED")
    return False


def send_reading(sound, pressure):
    """Send one reading to the API. Never crashes the main loop."""
    global pump_enabled
    if not connect_wifi():
        return

    payload = {
        "rms": round(sound["rms"], 2),
        "p2p": sound["p2p"],
        "level_db": round(sound["level_db"], 2),
        "freq_hz": round(sound["freq_hz"], 2),
        "freq_reliable": sound["freq_reliable"],
        "centroid_hz": round(sound["centroid_hz"], 2),
        "raw": pressure["raw"],
        "difference": pressure["difference"],
        "pressure_kpa": round(pressure["pressure_kpa"], 3),
        "device_id": DEVICE_ID,
        "is_test": False,                # real hardware data
    }
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }

    response = None
    try:
        response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
        print("SEND", response.status_code, response.text)
        try:
            pump_enabled = response.json().get("pump_enabled", True)
        except Exception:
            pass                         # keep the last known state
        print("Pump enabled by dashboard:", pump_enabled)
    except Exception as e:
        print("SEND ERROR:", e)
    finally:
        if response:
            response.close()


def poll_pump_command():
    """Fetch the dashboard command even when the pressure sensor has failed."""
    global pump_enabled
    if not connect_wifi():
        return
    response = None
    try:
        response = requests.get(PUMP_URL, headers={"ngrok-skip-browser-warning": "true"})
        if response.status_code == 200:
            result = response.json()
            if result.get("shared_store_ready") is False:
                print("WARNING: no shared store; using this API instance's pump command")
            state = result.get("pump_enabled")
            if isinstance(state, bool):
                pump_enabled = state
                print("Dashboard pump command:", "ON" if state else "OFF")
            else:
                print("PUMP API ERROR: missing pump_enabled")
        else:
            print("PUMP API HTTP:", response.status_code)
    except Exception as e:
        print("PUMP API ERROR:", e)
    finally:
        if response:
            response.close()


print("AirGuard Pico started")
connect_wifi()
last_sent = ticks_ms()

while True:
    sound = measure_sound()
    pressure = measure_pressure()

    if ticks_diff(ticks_ms(), last_sent) >= SEND_INTERVAL_MS:
        poll_pump_command()
        if pressure is not None:
            send_reading(sound, pressure)
        last_sent = ticks_ms()

    print("RMS=%.1f p-p=%d dB=%.1f Freq=%.1f Hz Cen=%.1f Hz"
          % (sound["rms"], sound["p2p"], sound["level_db"],
             sound["freq_hz"], sound["centroid_hz"]))

    if pressure is None:
        pump_status = control_pump(None)
        print("HX710B ERROR: DOUT=%d; Pump=%s | GP15=%d" % (DOUT.value(), pump_status, pump.value()))
    else:
        pump_status = control_pump(pressure["pressure_kpa"])
        print("RAW:", pressure["raw"],
              "| DIFFERENCE:", pressure["difference"],
              "| PRESSURE:", round(pressure["pressure_kpa"], 3), "kPa",
              "| PUMP:", pump_status, "| GP15:", pump.value())

    sleep_ms(LOOP_DELAY_MS)
