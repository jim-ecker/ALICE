import json
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI()

_lock = threading.Lock()
_progress: dict[str, dict] = {}       # document_id -> {label, total, processed}
_triples: list[dict] = []             # rolling buffer, capped at _MAX_TRIPLES
_MAX_TRIPLES = 500

_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>ALICE — Extraction Progress</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0d1117; color: #e6edf3; font-family: ui-monospace, monospace; padding: 2rem; }
    h1 { color: #58a6ff; font-size: 1.5rem; margin-bottom: 0.25rem; }
    .subtitle { color: #8b949e; margin-bottom: 2rem; font-size: 0.9rem; }

    #summary { color: #8b949e; font-size: 0.9rem; margin-bottom: 1.5rem; padding: 0.75rem 1rem;
               background: #161b22; border-radius: 6px; border: 1px solid #21262d; }
    #summary span { color: #e6edf3; font-weight: bold; }

    h2 { color: #8b949e; font-size: 0.85rem; font-weight: normal; text-transform: uppercase;
         letter-spacing: 0.08em; margin-bottom: 0.75rem; }

    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th { text-align: left; color: #8b949e; font-weight: normal; padding: 0.5rem 1rem;
         border-bottom: 1px solid #21262d; }
    td { padding: 0.55rem 1rem; border-bottom: 1px solid #161b22; vertical-align: middle; }

    .doc-label { max-width: 360px; }
    .doc-label a { color: #58a6ff; text-decoration: none; white-space: nowrap; }
    .doc-label a:hover { text-decoration: underline; }
    .doc-title { color: #8b949e; font-size: 0.8rem; margin-top: 2px; }
    .progress-wrap { background: #21262d; border-radius: 4px; height: 8px; }
    .progress-bar { background: #238636; border-radius: 4px; height: 8px; transition: width 0.5s ease; }
    tr.complete .progress-bar { background: #1f6feb; }
    .count { color: #8b949e; font-size: 0.8rem; margin-top: 3px; }
    .pct { color: #e6edf3; }

    .section { margin-top: 2.5rem; }
    .triple-wrap { max-height: 420px; overflow-y: auto; border: 1px solid #21262d; border-radius: 6px; }
    .triple-wrap table { font-size: 0.85rem; }
    .triple-wrap thead th { position: sticky; top: 0; background: #0d1117; z-index: 1; }
    .entity { color: #e6edf3; }
    .etype { color: #8b949e; font-size: 0.78rem; margin-left: 4px; }
    .relation { color: #d2a8ff; font-style: italic; }
    .cert-high { color: #3fb950; }
    .cert-mid  { color: #d29922; }
    .cert-low  { color: #f85149; }
    .triple-count { color: #8b949e; font-size: 0.8rem; margin-left: 0.5rem; }
  </style>
</head>
<body>
  <h1>ALICE</h1>
  <p class="subtitle">Knowledge Graph Extraction — live progress</p>

  <div id="summary">Connecting…</div>

  <h2>Documents</h2>
  <table>
    <thead>
      <tr>
        <th>Document</th>
        <th style="width:80px">Chunks</th>
        <th>Progress</th>
      </tr>
    </thead>
    <tbody id="progress-tbody"></tbody>
  </table>

  <div class="section">
    <h2>Extracted Triples <span id="triple-count" class="triple-count"></span></h2>
    <div class="triple-wrap">
      <table>
        <thead>
          <tr>
            <th>Subject</th>
            <th>Relation</th>
            <th>Object</th>
            <th style="width:90px">Certainty</th>
          </tr>
        </thead>
        <tbody id="triple-tbody"></tbody>
      </table>
    </div>
  </div>

  <script>
    function formatETA(secs) {
      if (!isFinite(secs) || secs <= 0) return null;
      if (secs < 60) return `${Math.round(secs)}s`;
      const m = Math.floor(secs / 60), s = Math.round(secs % 60);
      if (m < 60) return `${m}m ${s}s`;
      return `${Math.floor(m / 60)}h ${m % 60}m`;
    }

    // --- Progress SSE ---
    let _etaStart = null, _etaBaseProcessed = 0;
    const progEs = new EventSource("/events");
    progEs.onmessage = e => {
      const data = JSON.parse(e.data);
      const rows = Object.entries(data);
      const totalChunks = rows.reduce((s, [, d]) => s + d.total, 0);
      const totalDone   = rows.reduce((s, [, d]) => s + d.processed, 0);
      const remaining   = totalChunks - totalDone;
      const pctAll = totalChunks ? (totalDone / totalChunks * 100).toFixed(1) : 0;

      if (totalDone > 0 && _etaStart === null) {
        _etaStart = Date.now();
        _etaBaseProcessed = totalDone;
      }

      let etaStr = '';
      const doneSinceBase = totalDone - _etaBaseProcessed;
      if (_etaStart !== null && doneSinceBase >= 5 && remaining > 0) {
        const elapsed = (Date.now() - _etaStart) / 1000;
        const avgPerChunk = elapsed / doneSinceBase;
        const eta = formatETA(avgPerChunk * remaining);
        if (eta) etaStr = ` &nbsp;|&nbsp; ETA <span>${eta}</span>`;
      }

      document.getElementById("summary").innerHTML =
        `<span>${totalDone}</span> / <span>${totalChunks}</span> chunks processed &nbsp;|&nbsp; ` +
        `<span>${rows.length}</span> documents &nbsp;|&nbsp; <span>${pctAll}%</span> complete` + etaStr;

      document.getElementById("progress-tbody").innerHTML = rows.map(([id, d]) => {
        const pct = d.total ? (d.processed / d.total * 100).toFixed(1) : 0;
        const complete = d.processed >= d.total && d.total > 0;
        const ntrsId = d.label ? d.label.replace('https://ntrs.nasa.gov/citations/', 'NTRS ') : id.slice(0, 16) + '…';
        const linkHtml = d.label
          ? `<a href="${d.label}" target="_blank">${ntrsId}</a>`
          : ntrsId;
        const titleHtml = d.title ? `<div class="doc-title">${d.title}</div>` : '';

        let rowEta = '';
        if (_etaStart !== null && doneSinceBase >= 5 && !complete && d.processed > 0) {
          const elapsed = (Date.now() - _etaStart) / 1000;
          const avgPerChunk = elapsed / doneSinceBase;
          const docRemaining = d.total - d.processed;
          const eta = formatETA(avgPerChunk * docRemaining);
          if (eta) rowEta = ` &nbsp;<span style="color:#8b949e">ETA ${eta}</span>`;
        }

        return `<tr class="${complete ? 'complete' : ''}">
          <td class="doc-label">${linkHtml}${titleHtml}</td>
          <td>${d.total}</td>
          <td>
            <div class="progress-wrap"><div class="progress-bar" style="width:${pct}%"></div></div>
            <div class="count">${d.processed} / ${d.total} &nbsp;<span class="pct">${pct}%</span>${rowEta}</div>
          </td>
        </tr>`;
      }).join("");
    };

    // --- Triples SSE ---
    let tripleCount = 0;
    const tripleEs = new EventSource("/triple-events");
    tripleEs.onmessage = e => {
      const triples = JSON.parse(e.data);
      if (!triples.length) return;
      tripleCount += triples.length;
      document.getElementById("triple-count").textContent = `(${tripleCount} total)`;
      const tbody = document.getElementById("triple-tbody");
      const fragment = document.createDocumentFragment();
      triples.forEach(t => {
        const certClass = t.certainty >= 0.8 ? 'cert-high' : t.certainty >= 0.5 ? 'cert-mid' : 'cert-low';
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><span class="entity">${t.subject}</span><span class="etype">${t.subject_type}</span></td>
          <td><span class="relation">${t.relation}</span></td>
          <td><span class="entity">${t.object}</span><span class="etype">${t.object_type}</span></td>
          <td class="${certClass}">${(t.certainty * 100).toFixed(0)}%</td>`;
        fragment.appendChild(tr);
      });
      tbody.insertBefore(fragment, tbody.firstChild);
    };
  </script>
</body>
</html>"""


def init_document(document_id: str, label: str | None, title: str, total_chunks: int) -> None:
    with _lock:
        _progress[document_id] = {"label": label, "title": title, "total": total_chunks, "processed": 0}


def advance_chunk(document_id: str) -> None:
    with _lock:
        if document_id in _progress:
            _progress[document_id]["processed"] += 1


def push_triple(
    subject: str,
    subject_type: str,
    relation: str,
    object_: str,
    object_type: str,
    certainty: float,
) -> None:
    with _lock:
        if len(_triples) >= _MAX_TRIPLES:
            _triples.pop(0)
        _triples.append({
            "subject": subject,
            "subject_type": subject_type,
            "relation": relation,
            "object": object_,
            "object_type": object_type,
            "certainty": certainty,
        })


def _get_progress_snapshot() -> dict:
    with _lock:
        return dict(_progress)


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(_HTML)


@app.get("/events")
async def events() -> StreamingResponse:
    import asyncio

    async def generate():
        while True:
            yield f"data: {json.dumps(_get_progress_snapshot())}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/triple-events")
async def triple_events() -> StreamingResponse:
    import asyncio

    async def generate():
        sent = 0
        while True:
            with _lock:
                new = _triples[sent:]
                sent = len(_triples)
            if new:
                yield f"data: {json.dumps(new)}\n\n"
            else:
                yield f"data: []\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")


def start(host: str = "127.0.0.1", port: int = 8765) -> None:
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
