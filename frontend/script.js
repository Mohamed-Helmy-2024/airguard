const API_BASE = "https://brick-breeches-diploma.ngrok-free.dev";

const $ = (id) => document.getElementById(id);

// Display server timestamps in Cairo time.
function cairo(ts) {
  if (!ts) return "-";
  const fixed = ts.replace(/(\.\d{3})\d*/, "$1");   // 6 -> 3 decimals (safe for all browsers)
  return new Date(fixed).toLocaleString("en-GB",
    { timeZone: "Africa/Cairo", dateStyle: "medium", timeStyle: "medium" });
}

function duration(s) {
  if (s == null) return "-";
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return (h ? h + "h " : "") + (h || m ? m + "m " : "") + (s % 60) + "s";
}

async function getJSON(path) {
  const res = await fetch(API_BASE + path, {
    cache: "no-store",
    headers: { "ngrok-skip-browser-warning": "true" },
  });
  if (!res.ok) throw new Error("Server returned " + res.status);
  return res.json();
}

function showPage(id) {
  document.querySelectorAll(".page").forEach((p) => p.classList.add("hidden"));
  $(id).classList.remove("hidden");
}

const STATES = { normal: "NORMAL", leakage: "LEAKAGE", offline: "OFFLINE", off: "SYSTEM OFF" };

function updateSystemState(state) {
  if (!STATES[state]) state = "offline";
  document.body.className = state;                       // changes the theme
  $("statusText").textContent = STATES[state];
  $("warning").classList.toggle("hidden", state !== "leakage");
}

const REASONS = {
  ok: "Device connected",
  device_timeout: "No data from the device (timeout)",
  no_data_yet: "No data received yet",
  model_unavailable: "SVM model files are missing on the server",
  prediction_error: "The model failed on the last reading (see server log)",
  no_prediction: "No prediction yet",
  system_off: "System is OFF (pump turned off)",
};

let pumpOn = false;

function showPump(on) {
  pumpOn = on;
  $("pumpState").textContent = "Pump: " + (on ? "ON" : "OFF");
  $("pumpState").classList.toggle("on", on);
  $("pumpState").classList.toggle("off", !on);
  $("pumpBtn").textContent = on ? "Turn pump OFF" : "Turn pump ON";
}

async function togglePump() {
  const button = $("pumpBtn");
  button.disabled = true;
  try {
    const res = await fetch(API_BASE + "/api/pump", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
      },
      body: JSON.stringify({ on: !pumpOn }),
    });
    if (res.status === 404 || res.status === 405)
      throw new Error("the server has no /api/pump - restart uvicorn with the new app.py");
    if (!res.ok) throw new Error("Server returned " + res.status);
    const state = await res.json();
    if (typeof state.pump_enabled !== "boolean") throw new Error("Server returned an invalid pump state");
    showPump(state.pump_enabled);
    loadLatest();                                        // update status / colour now, without waiting for the WebSocket
  } catch (e) {
    alert("Could not change the pump: " + e.message);
    loadLatest();
  } finally {
    button.disabled = false;
  }
}

const SENSORS = [
  ["rms", "RMS", ""], ["p2p", "Peak-to-peak", ""], ["level_db", "Sound level", "dB"],
  ["freq_hz", "Frequency", "Hz"], ["centroid_hz", "Spectral centroid", "Hz"],
  ["raw", "Pressure sensor raw", "counts"], ["difference", "Raw minus zero offset", "counts"],
];

function showReading(r) {
  $("testBadge").classList.toggle("hidden", !(r && r.is_test));
  $("sensors").innerHTML = SENSORS.map(([key, name, unit]) => {
    let value = "-";
    if (r && r[key] != null) {
      value = (Number.isInteger(r[key]) ? r[key] : r[key].toFixed(2)) + " " + unit;
      if (key === "freq_hz" && r.freq_reliable === 0) value += " (unreliable)";
    }
    return `<div class="item"><small>${name}</small><b>${value}</b></div>`;
  }).join("");

  // Gauge range is 0–40 kPa; SVG path length is about 283.
  const p = r && r.pressure_kpa != null ? r.pressure_kpa : null;
  const part = p == null ? 0 : Math.min(Math.max(p / 40, 0), 1);
  $("gaugeFill").style.strokeDasharray = (part * 283) + " 283";
  $("gaugeValue").textContent = p == null ? "--" : p.toFixed(2) + " kPa";
}

let knownEventId;   // undefined until the first message arrives

function showSnapshot(m) {
  if (m.pump_enabled !== undefined) showPump(m.pump_enabled);
  updateSystemState(m.state);
  $("stateReason").textContent = REASONS[m.state_reason] || m.state_reason;
  $("lastTime").textContent = cairo(m.last_reading_timestamp);

  $("prediction").textContent = m.prediction || "No prediction yet";
  $("predictionNote").textContent = (m.state === "offline" || m.state === "off") && m.prediction
    ? "Last known prediction (" + (m.state === "off" ? "system off" : "device offline") + ")" : "";

  showReading(m.reading);

  const ev = m.active_event;
  $("eventInfo").textContent = ev
    ? "Leakage event open since " + cairo(ev.start_timestamp) +
      (m.state === "offline" ? " (device offline - not assumed ended)" : "")
    : "";

  // popup only when a NEW event starts (not on the first load)
  if (knownEventId !== undefined && ev && ev.id !== knownEventId) {
    $("popupText").textContent = "Started at " + cairo(ev.start_timestamp);
    $("popup").classList.remove("hidden");
    loadEvents();
  }
  knownEventId = ev ? ev.id : null;

  addToCharts(m.reading);
}

async function loadLatest() {
  try {
    showSnapshot(await getJSON("/api/latest"));
    $("connection").textContent = "API: connected";
  } catch (e) {
    $("connection").textContent = "API error: " + e.message;
    console.error("Latest reading request failed:", e);
  }
}

async function loadHealth() {
  try {
    const h = await getJSON("/api/health");
    const warnings = [];
    if (!h.model_ready) warnings.push("SVM model not ready: " + (h.model_error || h.missing_model_files.join(", ")));
    if (h.vercel_environment && !h.shared_store_ready)
      warnings.push("Shared store missing: readings, history, and pump state may reset between Vercel instances.");
    else if (h.vercel_environment && !h.shared_store_ok)
      warnings.push("Shared store connection failed; check the Upstash URL/token in Vercel.");
    $("healthWarning").classList.toggle("hidden", warnings.length === 0);
    $("healthWarning").textContent = warnings.join(" | ");
  } catch (e) {
    $("healthWarning").classList.remove("hidden");
    $("healthWarning").textContent = "API health check failed: " + e.message;
  }
}

const CHARTS = {
  pressureChart: [["pressure_kpa", "Pressure (kPa)", "#2563eb"]],
  soundChart: [["level_db", "Sound level (dB)", "#f59e0b"]],
  freqChart: [["freq_hz", "Frequency (Hz)", "#7c3aed"], ["centroid_hz", "Centroid (Hz)", "#0d9488"]],
};
const charts = {};
let lastChartId = 0;

if (typeof Chart !== "undefined") {
  for (const id in CHARTS) {
    charts[id] = new Chart($(id), {
    type: "line",
    data: {
      labels: [],
      datasets: CHARTS[id].map(([key, label, color]) =>
        ({ label, data: [], borderColor: color, borderWidth: 2, pointRadius: 0 })),
    },
    options: { animation: false, maintainAspectRatio: false, scales: { x: { ticks: { maxTicksLimit: 6 } } } },
    });
  }
}

async function loadCharts() {
  const minutes = $("chartRange").value;
  const start = new Date(Date.now() - minutes * 60000).toISOString();
  try {
    const data = await getJSON("/api/history?limit=5000&start=" + encodeURIComponent(start));
    const rows = data.readings;
    for (const id in charts) {
      charts[id].data.labels = rows.map((r) => cairo(r.timestamp).split(", ")[1]);
      charts[id].data.datasets.forEach((d, i) => { d.data = rows.map((r) => r[CHARTS[id][i][0]]); });
      charts[id].update();
    }
    lastChartId = rows.length ? rows[rows.length - 1].id : 0;
  } catch (e) { console.log(e); }
}

function addToCharts(r) {
  if (!r || r.id <= lastChartId) return;
  lastChartId = r.id;
  for (const id in charts) {
    charts[id].data.labels.push(cairo(r.timestamp).split(", ")[1]);
    charts[id].data.datasets.forEach((d, i) => d.data.push(r[CHARTS[id][i][0]]));
    if (charts[id].data.labels.length > 5000) {
      charts[id].data.labels.shift();
      charts[id].data.datasets.forEach((d) => d.data.shift());
    }
    charts[id].update();
  }
}

function rangeQuery() {
  const range = $("range").value;
  if (range !== "custom") return "range=" + range;
  let q = "range=24h";                                   // ignored when start / end are given
  if ($("start").value) q += "&start=" + encodeURIComponent(new Date($("start").value).toISOString());
  if ($("end").value) q += "&end=" + encodeURIComponent(new Date($("end").value).toISOString());
  return q;
}

async function loadEvents() {
  $("customBox").classList.toggle("hidden", $("range").value !== "custom");
  try {
    const data = await getJSON("/api/leakage-events?" + rangeQuery());
    $("reportRange").textContent = `From ${cairo(data.start)} to ${cairo(data.end)} (Cairo time) - ` +
      `${data.event_count} events, total ${duration(data.total_duration_seconds)}`;
    $("eventsBody").innerHTML = data.events.length ? data.events.map((e) => `<tr>
        <td>${cairo(e.start_timestamp)}</td>
        <td>${e.end_timestamp ? cairo(e.end_timestamp) : "-"}</td>
        <td>${duration(e.duration_seconds)}</td>
        <td>${e.status === "active" ? "Active" : "Closed"}${e.is_test ? " (test)" : ""}</td></tr>`).join("")
      : `<tr><td colspan="4">No leakage events in this period.</td></tr>`;
  } catch (e) {
    $("eventsBody").innerHTML = `<tr><td colspan="4">Could not load events: ${e.message}</td></tr>`;
  }
}

let wsOpen = false;

function connectWS() {
  const ws = new WebSocket(API_BASE.replace(/^http/, "ws") + "/ws/live");
  ws.onopen = () => {
    wsOpen = true;
    $("connection").textContent = "Server: connected";
  };
  ws.onmessage = (e) => showSnapshot(JSON.parse(e.data));
  ws.onclose = () => {
    wsOpen = false;
    $("connection").textContent = "Server: disconnected - retrying...";
    setTimeout(connectWS, 3000);                         // reconnect
  };
  ws.onerror = () => ws.close();
}

const PEOPLE = [
  {
    name: "Mohamed Helmy",
    role: "Team member",
    theme: "Area to be added",
    description: "Description to be added.",
    initials: "MH",
    image: "profile-mohamed-helmy.jpg",
  },
  {
    name: "Mohamed Kroush",
    role: "Team member",
    theme: "Area to be added",
    description: "Description to be added.",
    initials: "MK",
    image: "profile-mohamed-kroush.jpg",
  },
  {
    name: "Mohamed",
    role: "Team member",
    theme: "Area to be added",
    description: "Description to be added.",
    initials: "M",
    image: "profile-mohamed.jpg",
  },
  {
    name: "Mohamed Helmi",
    role: "Team member",
    theme: "Area to be added",
    description: "Description to be added.",
    initials: "MH",
    image: "profile-mohamed-helmi-2.jpg",
  },
];

$("aboutCards").innerHTML = PEOPLE.map((member) => `
  <div class="flip" onclick="this.classList.toggle('flipped')"><div class="inner">
    <div class="front">
      <div class="avatar">${member.initials}<img src="${member.image}" alt="" onerror="this.remove()"></div>
      <b>${member.name}</b><span>${member.role}</span>
    </div>
    <div class="back"><p>${member.theme}: ${member.description}</p></div>
  </div></div>`).join("");

loadLatest();
loadHealth();
loadCharts();
loadEvents();
connectWS();
// REST polling keeps updates visible across Vercel function instances.
setInterval(loadLatest, 1000);
setInterval(loadEvents, 15000);
setInterval(loadHealth, 30000);
