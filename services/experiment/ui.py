"""Inline HTML/CSS/JS for the ALICE experiment evaluation workbench."""

EXPERIMENT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ALICE — Evaluation Workbench</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0f1117; --surface: #1a1d27; --surface2: #22263a; --surface3: #2a2f45;
  --border: #2e3350; --accent: #5b8af0; --accent2: #7c6af7;
  --text: #e2e8f0; --text2: #8892a8; --text3: #555e7a;
  --green: #34d399; --red: #f87171; --yellow: #fbbf24; --orange: #fb923c;
  --radius: 8px; --font: 'Segoe UI', system-ui, sans-serif;
}
body { background: var(--bg); color: var(--text); font-family: var(--font);
       min-height: 100vh; display: flex; flex-direction: column; }
a { color: var(--accent); text-decoration: none; }

/* Header */
#header { background: var(--surface); border-bottom: 1px solid var(--border);
          padding: 12px 24px; display: flex; align-items: center; gap: 12px; }
#header h1 { font-size: 18px; font-weight: 700; color: var(--accent); }
#header .subtitle { font-size: 12px; color: var(--text2); }

/* Main */
#main { flex: 1; padding: 32px 24px; max-width: 960px; margin: 0 auto; width: 100%; }

/* Panels */
.panel { display: none; }
.panel.active { display: block; }

/* Cards */
.card { background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 24px; margin-bottom: 16px; }
.card-title { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: var(--text); }

/* Form fields */
.form-group { margin-bottom: 16px; }
.form-group label { display: block; font-size: 13px; color: var(--text2); margin-bottom: 6px; }
.form-group input, .form-group select, .form-group textarea {
  width: 100%; background: var(--surface2); border: 1px solid var(--border);
  color: var(--text); border-radius: 6px; padding: 8px 12px; font-size: 14px; font-family: var(--font); }
.form-group input:focus, .form-group select:focus, .form-group textarea:focus {
  outline: none; border-color: var(--accent); }
.form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

/* Likert scale */
.likert { display: flex; align-items: center; gap: 8px; }
.likert-labels { display: flex; justify-content: space-between; font-size: 11px;
                 color: var(--text3); margin-top: 4px; }
.likert input[type=range] { flex: 1; accent-color: var(--accent); cursor: pointer; }
.likert .likert-val { min-width: 28px; text-align: center; font-size: 14px; font-weight: 600;
                      color: var(--accent); background: var(--surface2); border: 1px solid var(--border);
                      border-radius: 4px; padding: 2px 6px; }

/* Buttons */
.btn { padding: 10px 20px; border-radius: 6px; font-size: 14px; font-weight: 600;
       cursor: pointer; border: none; transition: opacity .15s; }
.btn:disabled { opacity: .4; cursor: not-allowed; }
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover:not(:disabled) { opacity: .85; }
.btn-secondary { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
.btn-secondary:hover:not(:disabled) { background: var(--surface3); }
.btn-danger { background: #7f1d1d; color: var(--red); border: 1px solid #991b1b; }
.btn-danger:hover:not(:disabled) { background: #991b1b; }
.btn-sm { padding: 6px 14px; font-size: 13px; }
.btn-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }

/* Progress */
.progress-bar { background: var(--surface2); border-radius: 99px; height: 6px;
                overflow: hidden; margin-bottom: 8px; }
.progress-fill { background: var(--accent); height: 100%; border-radius: 99px;
                 transition: width .3s; }
.progress-label { font-size: 12px; color: var(--text2); }

/* Question block */
#question-text { font-size: 18px; font-weight: 600; margin-bottom: 24px;
                 padding-bottom: 16px; border-bottom: 1px solid var(--border); }

/* Response grid */
#responses-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
                  gap: 16px; margin-bottom: 24px; }
.response-card { background: var(--surface); border: 1px solid var(--border);
                 border-radius: var(--radius); padding: 16px; }
.response-card .alias-badge { display: inline-block; background: var(--accent2); color: #fff;
  font-size: 13px; font-weight: 700; border-radius: 4px; padding: 2px 10px; margin-bottom: 10px; }
.response-text { font-size: 13px; line-height: 1.6; color: var(--text);
                 background: var(--surface2); border-radius: 6px; padding: 12px;
                 min-height: 80px; margin-bottom: 12px; white-space: pre-wrap; word-break: break-word; }
.response-text.placeholder { color: var(--text3); font-style: italic; }
.response-text.loading { color: var(--yellow); }
.rating-row { display: flex; gap: 12px; margin-bottom: 8px; }
.rating-item { flex: 1; }
.rating-item label { font-size: 12px; color: var(--text2); display: block; margin-bottom: 4px; }
.comments-field { width: 100%; background: var(--surface2); border: 1px solid var(--border);
                  color: var(--text); border-radius: 6px; padding: 8px 10px; font-size: 13px;
                  font-family: var(--font); resize: vertical; min-height: 60px; margin-top: 8px; }
.comments-field::placeholder { color: var(--text3); }

/* Ask button area */
.ask-row { display: flex; gap: 8px; margin-bottom: 12px; align-items: center; }

/* Identification section */
#identification-section { background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 20px; margin-bottom: 24px; }
#identification-section h3 { font-size: 14px; font-weight: 600; margin-bottom: 4px; }
#identification-section .hint { font-size: 12px; color: var(--text2); margin-bottom: 16px; }
.id-row { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin-bottom: 12px; }
.id-item label { font-size: 13px; color: var(--text2); display: block; margin-bottom: 6px; }
.id-item select { width: 100%; background: var(--surface2); border: 1px solid var(--border);
                  color: var(--text); border-radius: 6px; padding: 7px 10px; font-size: 14px; }
.confidence-row { display: flex; align-items: center; gap: 16px; }
.confidence-row label { font-size: 13px; color: var(--text2); white-space: nowrap; }

/* Alert / error */
.alert { background: #1c1010; border: 1px solid #7f1d1d; color: var(--red);
         border-radius: 6px; padding: 12px 16px; font-size: 13px; margin-bottom: 16px; }
.success { background: #0d1f16; border: 1px solid #166534; color: var(--green); }

/* Summary */
#summary-stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
                 gap: 12px; margin-bottom: 16px; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
             padding: 16px; text-align: center; }
.stat-card .stat-val { font-size: 28px; font-weight: 700; color: var(--accent); }
.stat-card .stat-label { font-size: 12px; color: var(--text2); margin-top: 4px; }

/* Spinner */
.spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid var(--border);
           border-top-color: var(--accent); border-radius: 50%; animation: spin .7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Tabs / mode toggle */
.mode-toggle { display: flex; gap: 8px; margin-bottom: 16px; }
.mode-toggle label { display: flex; align-items: center; gap: 6px; cursor: pointer;
                     font-size: 13px; color: var(--text2); }
.mode-toggle input { accent-color: var(--accent); cursor: pointer; }

/* Session info bar */
#session-bar { background: var(--surface2); border-bottom: 1px solid var(--border);
               padding: 8px 24px; display: flex; align-items: center; gap: 16px;
               font-size: 12px; color: var(--text2); }
#session-bar .session-id { font-family: monospace; font-size: 11px; }

/* Completed overlay */
#complete-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.7);
                    align-items: center; justify-content: center; z-index: 100; }
#complete-overlay.show { display: flex; }
#complete-box { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
                padding: 40px; text-align: center; max-width: 480px; }
#complete-box h2 { font-size: 22px; color: var(--green); margin-bottom: 12px; }
#complete-box p { color: var(--text2); font-size: 14px; line-height: 1.6; margin-bottom: 20px; }
</style>
</head>
<body>

<div id="header">
  <h1>ALICE Evaluation Workbench</h1>
  <span class="subtitle">Blinded human evaluation study</span>
</div>

<div id="session-bar" style="display:none">
  Session: <span class="session-id" id="bar-session-id"></span>
  &nbsp;|&nbsp; Participant: <span id="bar-participant-id"></span>
  &nbsp;|&nbsp; <span id="bar-progress"></span>
  <span style="margin-left:auto">
    <a href="#" id="abandon-link" style="color:var(--red);font-size:11px">Abandon session</a>
  </span>
</div>

<div id="main">

  <!-- ── Setup panel ──────────────────────────────────────────────────────── -->
  <div id="panel-setup" class="panel active">
    <div class="card">
      <div class="card-title">Participant Setup</div>

      <div id="setup-error" class="alert" style="display:none"></div>

      <div class="form-group">
        <label>Participant ID <span style="color:var(--red)">*</span></label>
        <input id="participant-id" type="text" placeholder="e.g. researcher_001" autocomplete="off">
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>Questions per session</label>
          <select id="setup-n-questions">
            <option value="2">2</option>
            <option value="3" selected>3</option>
            <option value="4">4</option>
            <option value="5">5</option>
          </select>
        </div>
        <div class="form-group">
          <label>Responses per question</label>
          <select id="setup-n-responses">
            <option value="2">2</option>
            <option value="3" selected>3</option>
            <option value="4">4</option>
          </select>
        </div>
      </div>

      <div class="form-group">
        <label>Response mode</label>
        <div class="mode-toggle">
          <label><input type="radio" name="response-mode" value="live_virtual_experts" checked> Live (query experts in real-time)</label>
          <label><input type="radio" name="response-mode" value="static_bank"> Static (pre-generated responses)</label>
        </div>
      </div>

      <hr style="border-color:var(--border);margin:20px 0">
      <div class="card-title" style="margin-bottom:12px">Background Questionnaire</div>
      <div id="questionnaire-fields"></div>

      <div class="btn-row" style="margin-top:20px">
        <button class="btn btn-primary" id="btn-start">Start Evaluation</button>
        <span id="setup-loading" style="display:none"><span class="spinner"></span> Loading…</span>
      </div>
    </div>
  </div>

  <!-- ── Evaluation panel ─────────────────────────────────────────────────── -->
  <div id="panel-eval" class="panel">
    <div style="margin-bottom:20px">
      <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
      <div class="progress-label" id="progress-label">Question 0 of 0</div>
    </div>

    <div id="eval-error" class="alert" style="display:none"></div>

    <div id="question-text"></div>

    <!-- Ask button (live mode) -->
    <div id="ask-row" class="ask-row" style="display:none">
      <button class="btn btn-secondary btn-sm" id="btn-ask">
        <span id="ask-label">Ask All Respondents</span>
      </button>
      <span id="ask-loading" style="display:none"><span class="spinner"></span> Querying…</span>
      <span id="ask-note" style="font-size:12px;color:var(--text2)">Click to send the question to all respondents.</span>
    </div>

    <div id="responses-grid"></div>

    <!-- Identification section -->
    <div id="identification-section">
      <h3>Identity Task</h3>
      <div class="hint">For each agent listed below, select which response (A, B, C...) you believe was produced by that agent. If you're unsure, make your best guess.</div>
      <div id="id-dropdowns" class="id-row"></div>
      <div class="confidence-row">
        <label>Overall confidence in your identification:</label>
        <div style="flex:1">
          <div class="likert">
            <input type="range" id="id-confidence" min="1" max="5" value="3">
            <span class="likert-val" id="id-confidence-val">3</span>
          </div>
          <div class="likert-labels"><span>Not confident</span><span>Very confident</span></div>
        </div>
      </div>
      <div class="form-group" style="margin-top:12px">
        <label>Comments (optional)</label>
        <textarea class="comments-field" id="id-comments" placeholder="Any observations about how you made your identification…"></textarea>
      </div>
    </div>

    <div class="btn-row">
      <button class="btn btn-primary" id="btn-save-next">Save & Next Question</button>
      <button class="btn btn-secondary" id="btn-submit-all" style="display:none">Save & Submit All</button>
      <span id="eval-loading" style="display:none"><span class="spinner"></span></span>
    </div>
  </div>

  <!-- ── Complete panel ────────────────────────────────────────────────────── -->
  <div id="panel-complete" class="panel">
    <div class="card">
      <div class="card-title" style="color:var(--green)">✓ Evaluation Complete</div>
      <p style="color:var(--text2);font-size:14px;margin-bottom:20px">
        Your responses have been recorded. Thank you for participating.
      </p>
      <div id="summary-stats"></div>
      <div class="btn-row">
        <button class="btn btn-secondary" id="btn-download">Download Session JSON</button>
        <button class="btn btn-secondary btn-sm" id="btn-new-session">Start New Session</button>
      </div>
    </div>
  </div>

</div><!-- #main -->

<div id="complete-overlay">
  <div id="complete-box">
    <h2>All Questions Rated!</h2>
    <p>You've rated all questions in this session. Click below to submit your responses.</p>
    <button class="btn btn-primary" id="btn-final-submit">Submit Evaluation</button>
  </div>
</div>

<script>
const API = '/api/experiment';
let session = null;
let meta = null;
const LS_KEY_PROFILE = 'alice_experiment_profile';
const LS_KEY_SESSION = 'alice_experiment_session';

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
async function apiFetch(path, opts = {}) {
  const r = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { const j = await r.json(); msg = j.detail || JSON.stringify(j); } catch {}
    throw new Error(msg);
  }
  return r.json();
}

function show(id) { document.getElementById(id).classList.add('active'); }
function hide(id) { document.getElementById(id).classList.remove('active'); }
function showEl(id) { document.getElementById(id).style.display = ''; }
function hideEl(id) { document.getElementById(id).style.display = 'none'; }
function err(panelId, msg) {
  const el = document.getElementById(panelId);
  el.textContent = msg;
  el.style.display = msg ? '' : 'none';
}
function setDisabled(id, v) { document.getElementById(id).disabled = v; }

// ---------------------------------------------------------------------------
// Meta + questionnaire rendering
// ---------------------------------------------------------------------------
async function loadMeta() {
  meta = (await apiFetch('/meta'));
  renderQuestionnaire(meta.questionnaire_schema || []);
  const nResp = document.getElementById('setup-n-responses');
  const maxR = Math.max(2, meta.agents_available || 3);
  nResp.innerHTML = '';
  for (let i = 2; i <= Math.min(maxR, 6); i++) {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = i;
    if (i === Math.min(3, maxR)) opt.selected = true;
    nResp.appendChild(opt);
  }
}

function renderQuestionnaire(schema) {
  const container = document.getElementById('questionnaire-fields');
  container.innerHTML = '';
  for (const field of schema) {
    const div = document.createElement('div');
    div.className = 'form-group';
    const label = document.createElement('label');
    label.textContent = field.label + (field.required ? ' *' : '');
    div.appendChild(label);
    if (field.type === 'text') {
      const inp = document.createElement('input');
      inp.type = 'text';
      inp.id = 'q_' + field.id;
      inp.required = !!field.required;
      div.appendChild(inp);
    } else if (field.type === 'number') {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.id = 'q_' + field.id;
      inp.min = field.min ?? 0;
      inp.max = field.max ?? 100;
      inp.required = !!field.required;
      div.appendChild(inp);
    } else if (field.type === 'likert') {
      const wrap = document.createElement('div');
      wrap.className = 'likert';
      const inp = document.createElement('input');
      inp.type = 'range';
      inp.id = 'q_' + field.id;
      inp.min = field.min ?? 1;
      inp.max = field.max ?? 5;
      inp.value = Math.round(((field.min ?? 1) + (field.max ?? 5)) / 2);
      const val = document.createElement('span');
      val.className = 'likert-val';
      val.textContent = inp.value;
      inp.addEventListener('input', () => { val.textContent = inp.value; });
      wrap.appendChild(inp);
      wrap.appendChild(val);
      div.appendChild(wrap);
      const lbls = document.createElement('div');
      lbls.className = 'likert-labels';
      lbls.innerHTML = `<span>${field.min_label || field.min || 1}</span><span>${field.max_label || field.max || 5}</span>`;
      div.appendChild(lbls);
    }
    container.appendChild(div);
  }
}

function collectQuestionnaire(schema) {
  const result = {};
  for (const field of (schema || [])) {
    const el = document.getElementById('q_' + field.id);
    if (!el) continue;
    result[field.id] = (field.type === 'number' || field.type === 'likert')
      ? Number(el.value)
      : el.value;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Session bar
// ---------------------------------------------------------------------------
function updateSessionBar() {
  if (!session) { hideEl('session-bar'); return; }
  showEl('session-bar');
  document.getElementById('bar-session-id').textContent = session.session_id;
  document.getElementById('bar-participant-id').textContent = session.participant_id;
  const p = session.progress;
  document.getElementById('bar-progress').textContent =
    `Question ${p.current_index + 1} of ${p.total_questions} · ${p.completed_questions} rated`;
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
document.getElementById('btn-start').addEventListener('click', async () => {
  err('setup-error', '');
  const pid = document.getElementById('participant-id').value.trim();
  if (!pid) { err('setup-error', 'Participant ID is required.'); return; }

  const questionnaire = collectQuestionnaire(meta?.questionnaire_schema);
  const mode = document.querySelector('input[name="response-mode"]:checked')?.value || 'live_virtual_experts';
  const nQ = parseInt(document.getElementById('setup-n-questions').value);
  const nR = parseInt(document.getElementById('setup-n-responses').value);

  hideEl('btn-start');
  showEl('setup-loading');
  try {
    const res = await apiFetch('/sessions/start', {
      method: 'POST',
      body: JSON.stringify({
        participant_id: pid,
        questionnaire,
        questions_per_session: nQ,
        responses_per_question: nR,
        response_mode: mode,
      }),
    });
    session = res.session;
    localStorage.setItem(LS_KEY_SESSION, session.session_id);
    localStorage.setItem(LS_KEY_PROFILE, JSON.stringify({ participant_id: pid, questionnaire }));
    enterEval();
  } catch (e) {
    err('setup-error', e.message);
    showEl('btn-start');
  } finally {
    hideEl('setup-loading');
  }
});

// ---------------------------------------------------------------------------
// Evaluation UI
// ---------------------------------------------------------------------------
function enterEval() {
  hide('panel-setup');
  show('panel-eval');
  updateSessionBar();
  renderQuestion();
}

function renderQuestion() {
  if (!session) return;
  const p = session.progress;
  const pct = p.total_questions > 0 ? (p.current_index / p.total_questions) * 100 : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-label').textContent =
    `Question ${p.current_index + 1} of ${p.total_questions}`;

  const block = session.current_question;
  if (!block) return;

  document.getElementById('question-text').textContent = block.question_text;

  // Ask row (live mode only)
  const isLive = session.settings?.response_mode === 'live_virtual_experts';
  if (isLive) { showEl('ask-row'); } else { hideEl('ask-row'); }

  renderResponseCards(block);
  renderIdentificationSection(block);
  updateSaveButton();
}

function renderResponseCards(block) {
  const grid = document.getElementById('responses-grid');
  grid.innerHTML = '';
  for (const resp of block.responses) {
    const card = document.createElement('div');
    card.className = 'response-card';
    card.dataset.respId = resp.response_id;

    const badge = document.createElement('span');
    badge.className = 'alias-badge';
    badge.textContent = 'Response ' + resp.alias;
    card.appendChild(badge);

    const textDiv = document.createElement('div');
    textDiv.className = 'response-text' + (resp.text ? '' : ' placeholder');
    textDiv.id = 'resp-text-' + resp.response_id;
    textDiv.textContent = resp.text || '(Not yet answered — click "Ask All Respondents" above)';
    card.appendChild(textDiv);

    // Accuracy rating
    card.appendChild(buildRatingBlock('Accuracy', 'acc', resp.response_id));
    // Humanness rating
    card.appendChild(buildRatingBlock('Humanness', 'hum', resp.response_id));

    const cmtArea = document.createElement('textarea');
    cmtArea.className = 'comments-field';
    cmtArea.id = 'resp-cmt-' + resp.response_id;
    cmtArea.placeholder = 'Comments on this response (optional)…';
    card.appendChild(cmtArea);

    grid.appendChild(card);
  }
}

function buildRatingBlock(label, prefix, respId) {
  const wrap = document.createElement('div');
  wrap.className = 'rating-item';
  const lbl = document.createElement('label');
  lbl.textContent = label + ' (1–5)';
  wrap.appendChild(lbl);

  const row = document.createElement('div');
  row.className = 'likert';
  const inp = document.createElement('input');
  inp.type = 'range';
  inp.id = `resp-${prefix}-${respId}`;
  inp.min = 1; inp.max = 5; inp.value = 3;
  const val = document.createElement('span');
  val.className = 'likert-val';
  val.id = `resp-${prefix}-val-${respId}`;
  val.textContent = '3';
  inp.addEventListener('input', () => { val.textContent = inp.value; });
  row.appendChild(inp);
  row.appendChild(val);
  wrap.appendChild(row);

  const lbls = document.createElement('div');
  lbls.className = 'likert-labels';
  lbls.innerHTML = '<span>Poor</span><span>Excellent</span>';
  wrap.appendChild(lbls);
  return wrap;
}

function renderIdentificationSection(block) {
  // Build one dropdown per agent that appears in this block (unique agent labels)
  const agentLabels = new Set();
  // We need the agent labels — but they were stripped for blinding in public_session.
  // The identification task asks: "for each KNOWN agent identity, which alias do you think it is?"
  // Agent labels to show come from meta.agents
  const agentList = (meta?.agents || []).filter(a =>
    block.responses.some(() => true) // all agents visible in meta
  ).slice(0, block.responses.length);

  const aliases = block.responses.map(r => r.alias);

  const container = document.getElementById('id-dropdowns');
  container.innerHTML = '';
  for (const agent of agentList) {
    const item = document.createElement('div');
    item.className = 'id-item';
    const lbl = document.createElement('label');
    lbl.textContent = `Which response is "${agent.label}"?`;
    item.appendChild(lbl);
    const sel = document.createElement('select');
    sel.id = 'id-guess-' + agent.agent_id;
    const blank = document.createElement('option');
    blank.value = '';
    blank.textContent = '— select —';
    sel.appendChild(blank);
    for (const alias of aliases) {
      const opt = document.createElement('option');
      opt.value = alias;
      opt.textContent = 'Response ' + alias;
      sel.appendChild(opt);
    }
    item.appendChild(sel);
    container.appendChild(item);
  }

  // Confidence slider
  const confInp = document.getElementById('id-confidence');
  const confVal = document.getElementById('id-confidence-val');
  confInp.value = 3; confVal.textContent = '3';
  confInp.oninput = () => { confVal.textContent = confInp.value; };
  document.getElementById('id-comments').value = '';
}

// ---------------------------------------------------------------------------
// Ask (live mode)
// ---------------------------------------------------------------------------
document.getElementById('btn-ask').addEventListener('click', async () => {
  if (!session) return;
  const block = session.current_question;
  if (!block) return;
  err('eval-error', '');
  setDisabled('btn-ask', true);
  showEl('ask-loading');
  hideEl('ask-note');

  // Show loading state on response cards
  for (const resp of block.responses) {
    const textDiv = document.getElementById('resp-text-' + resp.response_id);
    if (textDiv && !resp.text) {
      textDiv.className = 'response-text loading';
      textDiv.textContent = '⟳ Querying respondent…';
    }
  }

  try {
    const res = await apiFetch(`/sessions/${session.session_id}/questions/${block.q_id}/ask`, {
      method: 'POST',
      body: JSON.stringify({ question_text: block.question_text }),
    });
    session = res.session;
    renderResponseCards(session.current_question);
    renderIdentificationSection(session.current_question);
    updateSessionBar();
  } catch (e) {
    err('eval-error', 'Failed to query respondents: ' + e.message);
  } finally {
    setDisabled('btn-ask', false);
    hideEl('ask-loading');
    showEl('ask-note');
  }
});

// ---------------------------------------------------------------------------
// Save & Next
// ---------------------------------------------------------------------------
function collectRatings() {
  const block = session?.current_question;
  if (!block) return null;
  const ratings = block.responses.map(resp => ({
    response_id: resp.response_id,
    accuracy: parseInt(document.getElementById(`resp-acc-${resp.response_id}`)?.value || '3'),
    humanness: parseInt(document.getElementById(`resp-hum-${resp.response_id}`)?.value || '3'),
    comments: document.getElementById(`resp-cmt-${resp.response_id}`)?.value || null,
  }));

  const agents = meta?.agents || [];
  const guesses = {};
  for (const agent of agents) {
    const sel = document.getElementById('id-guess-' + agent.agent_id);
    if (sel && sel.value) guesses[agent.label] = sel.value;
  }

  const identification = {
    guesses,
    confidence: parseInt(document.getElementById('id-confidence')?.value || '3'),
    comments: document.getElementById('id-comments')?.value || null,
  };

  return { ratings, identification };
}

document.getElementById('btn-save-next').addEventListener('click', () => saveAndAdvance(false));
document.getElementById('btn-submit-all').addEventListener('click', () => saveAndAdvance(true));

async function saveAndAdvance(isLast) {
  if (!session) return;
  const block = session.current_question;
  if (!block) return;
  err('eval-error', '');
  const payload = collectRatings();
  if (!payload) return;

  showEl('eval-loading');
  setDisabled('btn-save-next', true);
  setDisabled('btn-submit-all', true);
  try {
    // Save ratings
    await apiFetch(`/sessions/${session.session_id}/questions/${block.q_id}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    });

    if (isLast) {
      // Submit
      const res = await apiFetch(`/sessions/${session.session_id}/submit`, { method: 'POST' });
      session = res.session;
      enterComplete();
    } else {
      // Advance
      const res = await apiFetch(`/sessions/${session.session_id}/next`, { method: 'POST' });
      session = res.session;
      updateSessionBar();
      // Check if now on last question
      const p = session.progress;
      if (p.current_index >= p.total_questions - 1) {
        hideEl('btn-save-next');
        showEl('btn-submit-all');
      }
      renderQuestion();
    }
  } catch (e) {
    err('eval-error', e.message);
  } finally {
    hideEl('eval-loading');
    setDisabled('btn-save-next', false);
    setDisabled('btn-submit-all', false);
  }
}

function updateSaveButton() {
  if (!session) return;
  const p = session.progress;
  const isLast = p.current_index >= p.total_questions - 1;
  if (isLast) {
    hideEl('btn-save-next');
    showEl('btn-submit-all');
  } else {
    showEl('btn-save-next');
    hideEl('btn-submit-all');
  }
}

// ---------------------------------------------------------------------------
// Complete
// ---------------------------------------------------------------------------
function enterComplete() {
  hide('panel-eval');
  show('panel-complete');
  hideEl('session-bar');

  const p = session?.progress || {};
  const statsDiv = document.getElementById('summary-stats');
  statsDiv.innerHTML = `
    <div class="stat-card"><div class="stat-val">${p.completed_questions || 0}</div><div class="stat-label">Questions Rated</div></div>
    <div class="stat-card"><div class="stat-val">${session?.session_id?.slice(-6) || '—'}</div><div class="stat-label">Session ID (last 6)</div></div>
    <div class="stat-card"><div class="stat-val">${session?.participant_id || '—'}</div><div class="stat-label">Participant ID</div></div>
  `;
  localStorage.removeItem(LS_KEY_SESSION);
}

document.getElementById('btn-final-submit').addEventListener('click', async () => {
  document.getElementById('complete-overlay').classList.remove('show');
  if (!session) return;
  try {
    const res = await apiFetch(`/sessions/${session.session_id}/submit`, { method: 'POST' });
    session = res.session;
    enterComplete();
  } catch (e) {
    err('eval-error', e.message);
  }
});

// ---------------------------------------------------------------------------
// Download + new session
// ---------------------------------------------------------------------------
document.getElementById('btn-download').addEventListener('click', async () => {
  if (!session) return;
  const res = await apiFetch(`/sessions/${session.session_id}/export`);
  const blob = new Blob([JSON.stringify(res, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `alice_eval_${session.session_id}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

document.getElementById('btn-new-session').addEventListener('click', () => {
  session = null;
  localStorage.removeItem(LS_KEY_SESSION);
  hide('panel-complete');
  show('panel-setup');
  // Restore participant id from profile
  const profile = JSON.parse(localStorage.getItem(LS_KEY_PROFILE) || 'null');
  if (profile?.participant_id) {
    document.getElementById('participant-id').value = profile.participant_id;
  }
});

// ---------------------------------------------------------------------------
// Abandon session
// ---------------------------------------------------------------------------
document.getElementById('abandon-link').addEventListener('click', async (e) => {
  e.preventDefault();
  if (!confirm('Abandon this session? Your ratings so far will NOT be saved.')) return;
  session = null;
  localStorage.removeItem(LS_KEY_SESSION);
  hide('panel-eval');
  hideEl('session-bar');
  show('panel-setup');
});

// ---------------------------------------------------------------------------
// Resume from localStorage
// ---------------------------------------------------------------------------
async function tryResume() {
  const savedId = localStorage.getItem(LS_KEY_SESSION);
  if (!savedId) return false;
  try {
    const res = await apiFetch(`/sessions/${savedId}`);
    if (res.session && res.session.status === 'in_progress') {
      session = res.session;
      return true;
    }
  } catch {}
  localStorage.removeItem(LS_KEY_SESSION);
  return false;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
(async () => {
  try {
    await loadMeta();
  } catch (e) {
    err('setup-error', 'Failed to load experiment configuration: ' + e.message);
    showEl('setup-error');
  }

  // Restore participant id from localStorage
  const profile = JSON.parse(localStorage.getItem(LS_KEY_PROFILE) || 'null');
  if (profile?.participant_id) {
    document.getElementById('participant-id').value = profile.participant_id;
  }

  // Try to resume in-progress session
  const resumed = await tryResume();
  if (resumed) {
    enterEval();
  }
})();
</script>
</body>
</html>"""
