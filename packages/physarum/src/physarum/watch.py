"""'mcm sweep watch <sweep-name>': a read-only localhost page for watching a
sweep converge. Deliberately decoupled from 'sweep optimize' -- it only reads
Trial manifests reishi.store already persists after every trial, so it can be
started, stopped, or pointed at a finished sweep independently of whichever
process is actually running study.optimize(). No websockets/SSE: the page
polls a plain JSON endpoint, which is enough for a human watching trials land
every few seconds and keeps this stdlib-only, matching reishi/physarum's
near-zero-dependency rule.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TypedDict

from reishi import store
from reishi.primitives import trial as trial_store

DEFAULT_PORT = 8765


class SweepTrialRow(TypedDict):
    trial: int
    status: str
    metrics: dict
    params: dict


def trials_for_sweep(
    sweep_name: str, started_at: str | None = None
) -> list[SweepTrialRow]:
    """Every Trial whose recipe is '<sweep_name>-t<N>' (objective.build_recipe's
    naming -- the '-s<seed>-<uuid>' suffix lives only on Trial.id, never on
    Trial.recipe), ascending by trial number.

    Trial numbers come from that recipe name, not a stored field -- it's all a
    Trial keeps of its sweep lineage.

    started_at (from the "sweeps" sidecar's most recent `sweep optimize`
    invocation) drops any Trial saved before it -- reishi's store never
    deletes, so re-running a sweep under the same name would otherwise mix
    this run's trials with every previous run's leftovers. Trial.created is
    an ISO-8601 UTC string, so plain string comparison orders correctly.
    """
    prefix = f"{sweep_name}-t"
    rows: list[SweepTrialRow] = []
    for t in trial_store.load_all():
        if not t.recipe.startswith(prefix):
            continue
        if started_at is not None and t.created < started_at:
            continue
        try:
            trial_number = int(t.recipe[len(prefix) :].split("-s", 1)[0])
        except ValueError:
            continue
        rows.append(
            {
                "trial": trial_number,
                "status": t.status,
                "metrics": t.metrics,
                "params": dict(t.spec.get("trainer", {})),
            }
        )
    rows.sort(key=lambda r: r["trial"])
    return rows


class SweepSidecar(TypedDict):
    n_trials: int | None
    started_at: str | None


def _sweep_sidecar(sweep_name: str) -> SweepSidecar:
    """The current run's bookkeeping, if `sweep optimize` has started (it
    writes this before the first trial) -- both fields fall back to None for
    a sweep watch started before that, or for a sweep that predates this
    sidecar existing."""
    try:
        m = store.load("sweeps", sweep_name)
    except FileNotFoundError:
        return {"n_trials": None, "started_at": None}
    n = m.get("n_trials")
    started_at = m.get("started_at")
    return {
        "n_trials": n if isinstance(n, int) else None,
        "started_at": started_at if isinstance(started_at, str) else None,
    }


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>mcm sweep watch</title>
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    background: #0e1116; color: #d8dee9; margin: 0; padding: 2rem;
  }
  h1 { font-size: 1.1rem; font-weight: 600; margin: 0 0 .25rem; color: #e8ecf1; }
  #subtitle { color: #7d8799; font-size: .85rem; margin-bottom: 1.5rem; }
  #staleness { color: #e0c57e; font-size: .8rem; margin-left: .75rem; display: none; }
  .stats { display: flex; gap: 2rem; margin-bottom: 1.25rem; }
  .stat .label { color: #7d8799; font-size: .7rem; text-transform: uppercase; letter-spacing: .05em; }
  .stat .value { font-size: 1.4rem; color: #7ee0c1; }
  .dashboard-grid { display: grid; grid-template-columns: 200px 1fr 200px; gap: 1rem; align-items: start; }
  .param-col { display: flex; flex-direction: column; gap: .6rem; }
  .param-tile { background: #131722; border: 1px solid #232a36; border-radius: 6px; padding: .5rem .6rem; }
  .param-tile canvas { width: 100%; height: 56px; display: block; }
  .plabel { font-size: .62rem; text-transform: uppercase; letter-spacing: .04em; color: #7d8799; }
  .plead { font-size: .75rem; color: #9db4e8; margin-top: .2rem; font-variant-numeric: tabular-nums; }
  @media (max-width: 900px) {
    .dashboard-grid { grid-template-columns: 1fr; }
    .param-col { flex-direction: row; flex-wrap: wrap; }
    .param-tile { flex: 1 1 140px; }
  }
  #chart { background: #131722; border: 1px solid #232a36; border-radius: 6px; width: 100%; height: 340px; }
  table { border-collapse: collapse; width: 100%; margin-top: 1.5rem; font-size: .8rem; }
  th, td { text-align: left; padding: .35rem .6rem; border-bottom: 1px solid #1d222c; white-space: nowrap; }
  th { color: #7d8799; font-weight: 500; text-transform: uppercase; font-size: .65rem; letter-spacing: .05em; }
  tr.best td { color: #7ee0c1; }
  tr.running td { color: #e0c57e; }
  tr.failed td { color: #e07e7e; }
  #empty { color: #7d8799; padding: 2rem 0; }
</style>
</head>
<body>
<h1 id="title">mcm sweep watch</h1>
<div id="subtitle">polling /trials.json every 1.5s -- Ctrl-C the server to stop<span id="staleness"></span></div>
<div class="dashboard-grid">
  <div id="params-left" class="param-col"></div>
  <div id="hero">
    <div class="stats">
      <div class="stat"><div class="label">Trials done</div><div class="value" id="stat-done">0/0</div></div>
      <div class="stat"><div class="label">Best value</div><div class="value" id="stat-best">-</div></div>
      <div class="stat"><div class="label">Latest value</div><div class="value" id="stat-latest">-</div></div>
    </div>
    <canvas id="chart" width="1100" height="340"></canvas>
  </div>
  <div id="params-right" class="param-col"></div>
</div>
<div id="empty" style="display:none">No trials yet for this sweep -- once 'mcm sweep optimize' starts saving trials, they'll appear here.</div>
<table id="tbl">
  <thead><tr><th>#</th><th>status</th><th>value</th><th>params</th></tr></thead>
  <tbody></tbody>
</table>
<script>
const METRIC_CANDIDATES = ["field_f1", "field_recall", "field_precision", "exact_match"];

function metricValue(m) {
  if (!m) return null;
  for (const k of METRIC_CANDIDATES) if (typeof m[k] === "number") return m[k];
  return null;
}

function draw(trials) {
  const canvas = document.getElementById("chart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, pad = 40;
  ctx.clearRect(0, 0, w, h);

  const done = trials.filter(t => t.status === "done" && metricValue(t.metrics) !== null);
  if (!done.length) return;

  // Scaled to the data's own range, not pinned to include 0 -- an f1 sweep
  // converging e.g. 0.70 -> 0.76 would otherwise be squeezed into a sliver at
  // the top of the canvas, hiding exactly the movement this chart is for.
  const values = done.map(t => metricValue(t.metrics));
  const rawMin = Math.min(...values), rawMax = Math.max(...values);
  const span = rawMax - rawMin;
  const padV = span > 0 ? span * 0.15 : Math.max(Math.abs(rawMax), 1) * 0.05;
  const minV = rawMin - padV;
  const maxV = rawMax + padV;
  const xFor = i => pad + (i / Math.max(1, done.length - 1)) * (w - 2 * pad);
  const yFor = v => h - pad - ((v - minV) / (maxV - minV || 1)) * (h - 2 * pad);

  // axes
  ctx.strokeStyle = "#232a36";
  ctx.beginPath();
  ctx.moveTo(pad, pad); ctx.lineTo(pad, h - pad); ctx.lineTo(w - pad, h - pad);
  ctx.stroke();
  ctx.fillStyle = "#7d8799";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText(maxV.toFixed(3), 4, pad + 4);
  ctx.fillText(minV.toFixed(3), 4, h - pad + 4);
  ctx.fillText("trial ->", w - pad - 36, h - pad + 16);

  // running best, stepped
  let best = -Infinity;
  ctx.strokeStyle = "#7ee0c1";
  ctx.lineWidth = 2;
  ctx.beginPath();
  done.forEach((t, i) => {
    best = Math.max(best, metricValue(t.metrics));
    const x = xFor(i), y = yFor(best);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // raw per-trial values
  ctx.fillStyle = "#5b8def";
  done.forEach((t, i) => {
    const x = xFor(i), y = yFor(metricValue(t.metrics));
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fill();
  });
}

function renderTable(trials) {
  const tbody = document.querySelector("#tbl tbody");
  tbody.innerHTML = "";
  const done = trials.filter(t => t.status === "done" && metricValue(t.metrics) !== null);
  const bestVal = done.length ? Math.max(...done.map(t => metricValue(t.metrics))) : null;
  [...trials].reverse().forEach(t => {
    const tr = document.createElement("tr");
    const v = metricValue(t.metrics);
    if (v !== null && v === bestVal) tr.className = "best";
    else if (t.status === "failed") tr.className = "failed";
    else if (t.status === "running" || t.status === "planned") tr.className = "running";
    tr.innerHTML = `<td>${t.trial}</td><td>${t.status}</td><td>${v === null ? "-" : v.toFixed(4)}</td>` +
      `<td>${Object.entries(t.params).map(([k, val]) => `${k}=${val}`).join(" ")}</td>`;
    tbody.appendChild(tr);
  });
}

// Which trainer keys are actually swept -- vs. fixed template config like
// eval_n -- isn't known here (this page never reads the sweep yaml, only the
// store), so infer it from the data itself: any key that takes more than one
// distinct value across trials seen so far is worth a tile.
function varyingParamKeys(trials) {
  const seen = {};
  trials.forEach(t => {
    for (const [k, v] of Object.entries(t.params || {})) {
      (seen[k] = seen[k] || new Set()).add(JSON.stringify(v));
    }
  });
  return Object.keys(seen).filter(k => seen[k].size > 1).sort();
}

function drawParamTile(canvas, trials, key, bestIdx) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, pad = 10;
  ctx.clearRect(0, 0, w, h);

  const points = trials.map((t, i) => ({ i, v: t.params ? t.params[key] : undefined }))
    .filter(d => d.v !== undefined);
  if (!points.length) return;

  const numeric = points.every(d => typeof d.v === "number");
  let yFor;
  if (numeric) {
    const vals = points.map(d => d.v);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const span = (hi - lo) || Math.max(Math.abs(hi), 1) * 0.1;
    yFor = v => h - pad - ((v - lo) / span) * (h - 2 * pad);
  } else {
    const cats = [...new Set(points.map(d => JSON.stringify(d.v)))];
    yFor = v => cats.length > 1
      ? h - pad - (cats.indexOf(JSON.stringify(v)) / (cats.length - 1)) * (h - 2 * pad)
      : h / 2;
  }
  const xFor = i => pad + (i / Math.max(1, trials.length - 1)) * (w - 2 * pad);

  ctx.strokeStyle = "#232a36";
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

  points.forEach(d => {
    const onBest = d.i === bestIdx;
    ctx.fillStyle = onBest ? "#7ee0c1" : "#3d4d6b";
    ctx.beginPath();
    ctx.arc(xFor(d.i), yFor(d.v), onBest ? 3.5 : 2.5, 0, 2 * Math.PI);
    ctx.fill();
  });
}

// Among the strongest ~30% of trials so far, which value most often shows up
// for this param -- a "converging toward" readout to pair with the trace dots.
function leadingValue(trials, key) {
  const done = trials.filter(t => t.status === "done" && metricValue(t.metrics) !== null && t.params && t.params[key] !== undefined);
  if (!done.length) return null;
  const topN = Math.max(1, Math.ceil(done.length * 0.3));
  const top = [...done].sort((a, b) => metricValue(b.metrics) - metricValue(a.metrics)).slice(0, topN);
  const counts = {};
  top.forEach(t => { const k = JSON.stringify(t.params[key]); counts[k] = (counts[k] || 0) + 1; });
  const [winner] = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return winner ? JSON.parse(winner[0]) : null;
}

let tileKeys = [];
function renderParamTiles(trials) {
  const keys = varyingParamKeys(trials);
  if (keys.join(",") !== tileKeys.join(",")) {
    tileKeys = keys;
    const left = document.getElementById("params-left");
    const right = document.getElementById("params-right");
    left.innerHTML = "";
    right.innerHTML = "";
    keys.forEach((k, i) => {
      const tile = document.createElement("div");
      tile.className = "param-tile";
      const label = k.replace(/^trainer\\./, "");
      tile.innerHTML = `<div class="plabel">${label}</div><canvas width="200" height="56"></canvas><div class="plead"></div>`;
      tile.dataset.key = k;
      (i % 2 === 0 ? left : right).appendChild(tile);
    });
  }

  const done = trials.filter(t => t.status === "done" && metricValue(t.metrics) !== null);
  const bestTrial = done.length ? done.reduce((a, b) => (metricValue(a.metrics) >= metricValue(b.metrics) ? a : b)) : null;
  const bestIdx = bestTrial ? trials.indexOf(bestTrial) : -1;

  document.querySelectorAll(".param-tile").forEach(tile => {
    const key = tile.dataset.key;
    drawParamTile(tile.querySelector("canvas"), trials, key, bestIdx);
    const lv = leadingValue(trials, key);
    tile.querySelector(".plead").textContent = lv === null ? "" : `-> ${lv}`;
  });
}

// n_trials comes from a sidecar `sweep optimize` writes before its first trial
// (watch.py's _sweep_total) -- fall back to counting trials seen so far if a
// watch was started before that file exists.
let lastSignature = null;
let lastChangeAt = Date.now();

function updateStaleness(trials) {
  const sig = JSON.stringify(trials.map(t => [t.trial, t.status, metricValue(t.metrics)]));
  if (sig !== lastSignature) {
    lastSignature = sig;
    lastChangeAt = Date.now();
  }
  const idleS = Math.round((Date.now() - lastChangeAt) / 1000);
  const el = document.getElementById("staleness");
  if (trials.length && idleS >= 10) {
    el.textContent = `-- no change in ${idleS}s, sweep may have finished or stalled`;
    el.style.display = "inline";
  } else {
    el.style.display = "none";
  }
}

async function tick() {
  try {
    const res = await fetch("/trials.json");
    const data = await res.json();
    document.getElementById("title").textContent = `mcm sweep watch -- ${data.sweep}`;
    const trials = data.trials;
    document.getElementById("empty").style.display = trials.length ? "none" : "block";
    updateStaleness(trials);

    const done = trials.filter(t => t.status === "done" && metricValue(t.metrics) !== null);
    const total = data.n_trials || trials.length;
    document.getElementById("stat-done").textContent = `${done.length}/${total}`;
    if (done.length) {
      const best = Math.max(...done.map(t => metricValue(t.metrics)));
      document.getElementById("stat-best").textContent = best.toFixed(4);
      document.getElementById("stat-latest").textContent = metricValue(done[done.length - 1].metrics).toFixed(4);
    }
    draw(trials);
    renderTable(trials);
    renderParamTiles(trials);
  } catch (e) {
    console.error(e);
  }
}

tick();
setInterval(tick, 1500);
</script>
</body>
</html>
"""


def _make_handler(sweep_name: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            pass  # per-request access logs are noise for a 1.5s polling loop

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/trials.json"):
                sidecar = _sweep_sidecar(sweep_name)
                payload = {
                    "sweep": sweep_name,
                    "n_trials": sidecar["n_trials"],
                    "trials": trials_for_sweep(sweep_name, sidecar["started_at"]),
                }
                self._send(200, json.dumps(payload).encode(), "application/json")
            elif self.path in ("/", ""):
                self._send(200, _PAGE.encode(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def serve(sweep_name: str, port: int = DEFAULT_PORT) -> None:
    # Threading, not plain HTTPServer: serve_forever() otherwise handles one
    # request at a time with no per-request timeout, so a single stalled
    # client (a half-open socket, a slow read) freezes every other poll --
    # indistinguishable from the sweep having converged.
    server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(sweep_name))
    print(
        f"[OK] watching sweep '{sweep_name}' -> http://127.0.0.1:{port}",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[OK] stopped watching", file=sys.stderr)
