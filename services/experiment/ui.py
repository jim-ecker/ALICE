"""Inline HTML/CSS/JS for the ALICE experiment evaluation workbench."""

EXPERIMENT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ALICE — Knowledge Retention Evaluation</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #0f1117;
  --surface: #1a1d27;
  --surface2: #22263a;
  --border: #2e3350;
  --accent: #5b8af0;
  --accent2: #7c6af7;
  --text: #e2e8f0;
  --text2: #8892a8;
  --green: #34d399;
  --red: #f87171;
  --yellow: #fbbf24;
  --radius: 8px;
  --shadow: 0 1px 3px rgba(0,0,0,.4), 0 1px 2px rgba(0,0,0,.3);
  --shadow-md: 0 4px 6px rgba(0,0,0,.4), 0 2px 4px rgba(0,0,0,.3);
}

body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  font-size: 15px;
  line-height: 1.6;
}

header {
  background: var(--surface);
  color: var(--text);
  padding: 14px 32px;
  display: flex;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid var(--border);
}
header h1 { font-size: 1.05rem; font-weight: 600; letter-spacing: .01em; color: var(--accent); }
.user-badge {
  margin-left: auto;
  font-size: .8rem;
  color: var(--text2);
  background: var(--surface2);
  border: 1px solid var(--border);
  padding: 4px 12px;
  border-radius: 12px;
}

#app { max-width: 1120px; margin: 0 auto; padding: 28px 20px 72px; }

.screen { display: none; }
.screen.active { display: block; }

#loading-screen {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
  gap: 14px;
  color: var(--text2);
  font-size: 1rem;
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 26px 30px;
  margin-bottom: 18px;
}

h2.section-title {
  font-size: 1.2rem;
  font-weight: 700;
  color: var(--accent);
  margin-bottom: 18px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}

h3.group-title {
  font-size: .82rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .07em;
  color: var(--text2);
  margin: 22px 0 12px;
}
h3.group-title:first-child { margin-top: 0; }

.field { margin-bottom: 14px; }
.field label {
  display: block;
  font-size: .875rem;
  font-weight: 500;
  color: var(--text);
  margin-bottom: 5px;
}
.req { color: var(--red); margin-left: 2px; }
.opt { color: var(--text2); font-weight: 400; font-size: .8rem; }

input[type=text], input[type=number], select, textarea {
  width: 100%;
  padding: 7px 11px;
  border: 1.5px solid var(--border);
  border-radius: 6px;
  font-size: .875rem;
  color: var(--text);
  background: var(--surface2);
  transition: border-color .15s, box-shadow .15s;
  font-family: inherit;
}
input[type=text]:focus, input[type=number]:focus, select:focus, textarea:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(91,138,240,.15);
  background: var(--surface2);
}
select option { background: var(--surface2); color: var(--text); }
input[type=number] { max-width: 110px; }
textarea { resize: vertical; min-height: 64px; }

.field-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}
@media (max-width: 600px) { .field-row { grid-template-columns: 1fr; } }

.hint { font-size: .78rem; color: var(--text2); margin-top: 3px; }

/* Likert */
.likert-wrap { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 5px; }
.likert-anchor { font-size: .78rem; color: var(--text2); white-space: nowrap; }
.likert-opts { display: flex; gap: 8px; }
.likert-opts label {
  display: flex; flex-direction: column; align-items: center;
  gap: 2px; cursor: pointer; font-size: .78rem; color: var(--text2); font-weight: 600;
}
.likert-opts input[type=radio] { cursor: pointer; accent-color: var(--accent); }

/* Buttons */
.btn {
  padding: 9px 20px;
  border: none;
  border-radius: 6px;
  font-size: .875rem;
  font-weight: 600;
  cursor: pointer;
  transition: background .15s, transform .1s, box-shadow .15s;
  font-family: inherit;
}
.btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: var(--shadow-md); }
.btn:active { transform: none !important; box-shadow: none !important; }
.btn:disabled { opacity: .42; cursor: not-allowed; }

.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover:not(:disabled) { background: #4a79df; }
.btn-secondary { background: transparent; color: var(--accent); border: 1.5px solid var(--accent); }
.btn-secondary:hover:not(:disabled) { background: rgba(91,138,240,.12); }

.btn-row { display: flex; gap: 10px; align-items: center; margin-top: 22px; justify-content: flex-end; }

/* Progress */
.progress-wrap { margin-bottom: 22px; }
.progress-meta { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 7px; }
.progress-label { font-size: .9rem; font-weight: 700; color: var(--accent); }
.progress-cat { font-size: .8rem; color: var(--text2); }
.progress-bar-outer { height: 7px; background: var(--surface2); border-radius: 4px; overflow: hidden; }
.progress-bar-inner {
  height: 100%;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  border-radius: 4px;
  transition: width .4s ease;
}

/* Question */
.question-text {
  font-size: 1.05rem;
  font-weight: 500;
  line-height: 1.7;
  color: var(--text);
  background: var(--surface2);
  border-left: 4px solid var(--accent);
  padding: 14px 18px;
  border-radius: 0 var(--radius) var(--radius) 0;
  margin-bottom: 18px;
}

/* Response grid */
.response-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
}
@media (max-width: 900px) { .response-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 560px) { .response-grid { grid-template-columns: 1fr; } }

/* Response card */
.response-card {
  background: var(--surface);
  border: 1.5px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transition: border-color .2s;
}
.response-card.rated { border-color: var(--green); }

.resp-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 9px 13px;
  background: var(--accent2);
  color: #fff;
}
.resp-alias { font-size: 1rem; font-weight: 700; letter-spacing: .02em; }
.resp-timing { font-size: .72rem; opacity: .85; font-family: monospace; }

.resp-body {
  padding: 11px 13px;
  flex: 1;
  min-height: 100px;
  font-size: .85rem;
  line-height: 1.65;
  color: var(--text);
  overflow-y: auto;
  max-height: 210px;
  word-break: break-word;
}
.resp-body p { margin-bottom: .5em; }
.resp-body p:last-child { margin-bottom: 0; }
.resp-body ul, .resp-body ol { padding-left: 1.4em; margin-bottom: .5em; }
.resp-body li { margin-bottom: .2em; }
.resp-body h1,.resp-body h2,.resp-body h3,.resp-body h4 {
  font-size: .9rem; font-weight: 700; margin: .6em 0 .3em; color: var(--accent);
}
.resp-body code {
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 3px; padding: 1px 4px;
  font-family: monospace; font-size: .82rem; color: var(--accent2);
}
.resp-body strong { font-weight: 700; }
.resp-body em { font-style: italic; color: var(--text2); }

.resp-loading {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text2);
  font-style: italic;
  font-size: .85rem;
}

.spinner {
  width: 15px; height: 15px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin .7s linear infinite;
  flex-shrink: 0;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Ratings inside card */
.resp-ratings {
  border-top: 1px solid var(--border);
  padding: 11px 13px;
  background: var(--bg);
}
.resp-ratings.hidden { display: none; }

.rating-item { margin-bottom: 10px; }
.rating-item:last-child { margin-bottom: 0; }
.rating-item > .rl {
  font-size: .78rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 2px;
}
.rating-anchor { font-size: .7rem; color: var(--text2); font-style: italic; margin-bottom: 4px; }

.radio-row { display: flex; gap: 10px; flex-wrap: wrap; }
.radio-row label {
  display: flex; align-items: center; gap: 3px;
  font-size: .78rem; color: var(--text2); font-weight: 600; cursor: pointer;
}
.radio-row input[type=radio] { accent-color: var(--accent); cursor: pointer; }

.rating-comment { margin-top: 7px; }
.rating-comment input { font-size: .78rem; padding: 5px 9px; }

/* Identity section */
.ident-card {
  background: var(--surface);
  border: 1.5px solid var(--accent2);
  border-radius: var(--radius);
  padding: 18px 22px;
  margin-top: 18px;
}
.ident-card.hidden { display: none; }
.ident-card h3 { font-size: .95rem; font-weight: 700; color: var(--accent2); margin-bottom: 12px; }
.ident-card .desc { font-size: .84rem; color: var(--text2); margin-bottom: 14px; }

.ident-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-bottom: 14px;
}
@media (max-width: 560px) { .ident-row { grid-template-columns: 1fr; } }

.ident-field label { font-size: .875rem; font-weight: 600; color: var(--text); display: block; margin-bottom: 4px; }
.ident-field .sublabel { font-size: .76rem; color: var(--text2); margin-bottom: 5px; }
.ident-field select { max-width: 180px; }

/* Email badge */
.email-badge {
  display: flex; align-items: center; gap: 10px;
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 6px; padding: 9px 13px; margin-bottom: 18px;
}
.email-badge .el { font-size: .78rem; color: var(--text2); font-weight: 500; }
.email-badge .ev { font-size: .875rem; font-weight: 700; color: var(--accent); }

/* Error */
.error-banner {
  background: rgba(248,113,113,.1); border: 1px solid rgba(248,113,113,.3);
  color: var(--red); padding: 11px 16px; border-radius: 6px;
  font-size: .875rem; margin-bottom: 14px;
}

/* Query action row */
.query-row { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
.query-status { font-size: .84rem; color: var(--text2); }

/* Complete */
.complete-hero { text-align: center; padding: 44px 24px; }
.complete-hero .checkmark { font-size: 3.5rem; color: var(--green); margin-bottom: 12px; }
.complete-hero h2 { font-size: 1.5rem; font-weight: 700; color: var(--text); margin-bottom: 8px; }
.complete-hero p { color: var(--text2); font-size: .95rem; }

/* Toast */
#toast {
  position: fixed; top: 18px; right: 18px;
  background: var(--surface2); color: var(--text);
  border: 1px solid var(--border);
  padding: 11px 18px; border-radius: 8px;
  font-size: .875rem; box-shadow: var(--shadow-md);
  z-index: 9999; opacity: 0; transform: translateY(-8px);
  transition: opacity .25s, transform .25s; pointer-events: none;
}
#toast.show { opacity: 1; transform: translateY(0); }
</style>
</head>
<body>

<header>
  <img src="/static/nasa.png" alt="NASA" style="width:32px;height:32px;object-fit:contain">
  <h1>ALICE — Knowledge Retention Evaluation</h1>
  <div class="user-badge" id="user-badge"></div>
</header>

<div id="app">

  <!-- Loading -->
  <div id="loading-screen">
    <div class="spinner" style="width:22px;height:22px;border-width:3px"></div>
    Initializing…
  </div>

  <!-- Screen 1: Bio / Profile -->
  <div id="screen-bio" class="screen">
    <div class="card">
      <h2 class="section-title">Evaluator Profile</h2>
      <p style="margin-bottom:16px;color:var(--muted);font-size:.875rem">
        This information is collected once and associated with your identity for the study.
        Fields marked <span class="req">*</span> are required.
        If you have participated before, your answers are pre-filled — update anything that has changed.
      </p>

      <div class="email-badge">
        <span class="el">Authenticated as:</span>
        <span class="ev" id="bio-email-val">—</span>
      </div>

      <div id="bio-error" class="error-banner" style="display:none"></div>

      <h3 class="group-title">Professional Background</h3>

      <div class="field-row">
        <div class="field">
          <label>Current Role / Title <span class="req">*</span></label>
          <input type="text" id="bio-role" placeholder="e.g., Research Engineer">
        </div>
        <div class="field">
          <label>Organization / Branch <span class="req">*</span></label>
          <input type="text" id="bio-org" placeholder="e.g., NASA LaRC — Flight Dynamics Branch">
        </div>
      </div>

      <div class="field-row">
        <div class="field">
          <label>Highest Degree Earned <span class="req">*</span></label>
          <select id="bio-degree">
            <option value="">— select —</option>
            <option value="BS">BS / Bachelor's</option>
            <option value="MS">MS / Master's</option>
            <option value="PhD">PhD / Doctorate</option>
            <option value="Other">Other</option>
          </select>
        </div>
        <div class="field">
          <label>Field of Highest Degree <span class="req">*</span></label>
          <input type="text" id="bio-degree-field" placeholder="e.g., Aerospace Engineering">
        </div>
      </div>

      <div class="field-row">
        <div class="field">
          <label>Years at NASA</label>
          <input type="number" id="bio-years-nasa" min="0" max="60" placeholder="0">
        </div>
        <div class="field">
          <label>Years in Primary Domain / Field</label>
          <input type="number" id="bio-years-domain" min="0" max="60" placeholder="0">
        </div>
      </div>

      <div class="field" style="max-width:260px">
        <label>Years in Aerospace / Engineering <span class="opt">(if different)</span></label>
        <input type="number" id="bio-years-aero" min="0" max="60" placeholder="0">
      </div>

      <h3 class="group-title">Familiarity with Key Personnel</h3>

      <div class="field-row">
        <div>
          <div class="field">
            <label>Years known B. Danette Allen professionally</label>
            <input type="number" id="bio-years-d" min="0" max="60" placeholder="0">
            <p class="hint">Enter 0 if you have never met.</p>
          </div>
          <div class="field">
            <label>Interaction frequency with B. Danette Allen</label>
            <select id="bio-freq-d">
              <option value="">— select —</option>
              <option>Never</option>
              <option>Rarely</option>
              <option>Occasionally</option>
              <option>Regularly</option>
              <option>Very Frequently</option>
            </select>
          </div>
          <div class="field">
            <label>Papers co-authored with B. Danette Allen</label>
            <input type="number" id="bio-papers-d" min="0" placeholder="0">
          </div>
        </div>
        <div>
          <div class="field">
            <label>Years known Natalia Alexandrov professionally</label>
            <input type="number" id="bio-years-n" min="0" max="60" placeholder="0">
            <p class="hint">Enter 0 if you have never met.</p>
          </div>
          <div class="field">
            <label>Interaction frequency with Natalia Alexandrov</label>
            <select id="bio-freq-n">
              <option value="">— select —</option>
              <option>Never</option>
              <option>Rarely</option>
              <option>Occasionally</option>
              <option>Regularly</option>
              <option>Very Frequently</option>
            </select>
          </div>
          <div class="field">
            <label>Papers co-authored with Natalia Alexandrov</label>
            <input type="number" id="bio-papers-n" min="0" placeholder="0">
          </div>
        </div>
      </div>

      <div class="field">
        <label>Worked on projects led by B. Danette Allen or Natalia Alexandrov?</label>
        <textarea id="bio-projects" placeholder="Yes / No — if yes, briefly describe (e.g., project name and your role)…"></textarea>
      </div>

      <h3 class="group-title">Domain Familiarity</h3>

      <div class="field">
        <label>Familiarity with the ALICE system</label>
        <div class="likert-wrap">
          <span class="likert-anchor">1 = Never heard of it</span>
          <div class="likert-opts" id="lik-alice"></div>
          <span class="likert-anchor">5 = Use it regularly</span>
        </div>
      </div>

      <div class="field">
        <label>Familiarity with B. Danette Allen's research areas</label>
        <div class="likert-wrap">
          <span class="likert-anchor">1 = Not at all familiar</span>
          <div class="likert-opts" id="lik-d-research"></div>
          <span class="likert-anchor">5 = Deep expertise</span>
        </div>
      </div>

      <div class="field">
        <label>Familiarity with Natalia Alexandrov's research areas</label>
        <div class="likert-wrap">
          <span class="likert-anchor">1 = Not at all familiar</span>
          <div class="likert-opts" id="lik-n-research"></div>
          <span class="likert-anchor">5 = Deep expertise</span>
        </div>
      </div>

      <div class="field" style="max-width:280px">
        <label>Papers by B. Danette Allen or Natalia Alexandrov you have read</label>
        <select id="bio-papers-read">
          <option value="">— select —</option>
          <option value="None">None</option>
          <option value="1-2">1–2</option>
          <option value="3-5">3–5</option>
          <option value="6-10">6–10</option>
          <option value="More than 10">More than 10</option>
        </select>
      </div>

      <h3 class="group-title">Communication Style</h3>

      <div class="field">
        <label>Familiarity with B. Danette Allen's communication style <span class="opt">(informal emails, chat, in-person)</span></label>
        <div class="likert-wrap">
          <span class="likert-anchor">1 = Not at all</span>
          <div class="likert-opts" id="lik-d-comms"></div>
          <span class="likert-anchor">5 = Very familiar</span>
        </div>
      </div>

      <div class="field">
        <label>Familiarity with Natalia Alexandrov's communication style</label>
        <div class="likert-wrap">
          <span class="likert-anchor">1 = Not at all</span>
          <div class="likert-opts" id="lik-n-comms"></div>
          <span class="likert-anchor">5 = Very familiar</span>
        </div>
      </div>

      <h3 class="group-title">Optional Demographics</h3>

      <div class="field-row">
        <div class="field">
          <label>Age Range <span class="opt">(optional)</span></label>
          <select id="bio-age">
            <option value="">— prefer not to say —</option>
            <option>20–29</option>
            <option>30–39</option>
            <option>40–49</option>
            <option>50–59</option>
            <option>60+</option>
          </select>
        </div>
        <div class="field">
          <label>Gender <span class="opt">(optional)</span></label>
          <input type="text" id="bio-gender" placeholder="e.g., Woman, Man, Non-binary…">
        </div>
      </div>

      <div class="btn-row">
        <button class="btn btn-primary" id="btn-bio-submit">Save &amp; Continue →</button>
      </div>
    </div>
  </div>

  <!-- Screen 2: Session Setup -->
  <div id="screen-setup" class="screen">
    <div class="card">
      <h2 class="section-title">Study Session Setup</h2>

      <p style="margin-bottom:22px;color:var(--muted);font-size:.875rem">
        In this session you will evaluate responses from six AI systems to aerospace knowledge queries.
        All responses are <strong>blinded</strong> — you will see them labeled A through F without
        knowing which system generated each. For each question you will rate every response on
        <em>accuracy</em> and <em>humanness</em>, then identify which responses came from
        <strong>VirtualD</strong> (B. Danette Allen) and <strong>VirtualN</strong> (Natalia Alexandrov).
      </p>

      <div class="field">
        <label>How many questions would you like to evaluate?</label>
        <p class="hint" style="margin-bottom:10px">Each question takes ~5–10 min. We recommend 3–5 for a single sitting.</p>
        <div style="display:flex;align-items:center;gap:14px">
          <input type="range" id="setup-nq" min="1" max="10" value="5"
            style="width:180px;accent-color:var(--nasa-blue)">
          <span id="setup-nq-display" style="font-size:1.3rem;font-weight:700;color:var(--nasa-blue)">5</span>
          <span style="color:var(--muted);font-size:.84rem">questions</span>
        </div>
      </div>

      <div id="setup-error" class="error-banner" style="display:none"></div>

      <div class="btn-row">
        <button class="btn btn-secondary" id="btn-setup-back">← Edit Profile</button>
        <button class="btn btn-primary" id="btn-setup-start">Start Session →</button>
      </div>
    </div>
  </div>

  <!-- Screen 3: Evaluation -->
  <div id="screen-eval" class="screen">
    <div class="progress-wrap">
      <div class="progress-meta">
        <span class="progress-label" id="eval-progress-label">Question 1 of 5</span>
        <span class="progress-cat" id="eval-progress-cat"></span>
      </div>
      <div class="progress-bar-outer">
        <div class="progress-bar-inner" id="eval-progress-bar" style="width:20%"></div>
      </div>
    </div>

    <div class="card" style="margin-bottom:14px">
      <div class="question-text" id="eval-question"></div>

      <div class="query-row">
        <button class="btn btn-primary" id="btn-query">▶ Query All Agents</button>
        <span class="query-status" id="query-status"></span>
      </div>

      <div class="response-grid" id="response-grid"></div>
    </div>

    <!-- Identity task -->
    <div class="card ident-card hidden" id="ident-section">
      <h3>Identity Task</h3>
      <p class="desc">
        Based on the responses above, which labeled response do you believe came from each virtual expert?
        You may choose the same response for both if you believe it fits best.
      </p>

      <div class="ident-row">
        <div class="ident-field">
          <label>Which response is <strong>VirtualD</strong>?</label>
          <div class="sublabel">B. Danette Allen — autonomous systems &amp; certification</div>
          <select id="ident-vd"></select>
        </div>
        <div class="ident-field">
          <label>Which response is <strong>VirtualN</strong>?</label>
          <div class="sublabel">Natalia Alexandrov — computational design optimization</div>
          <select id="ident-vn"></select>
        </div>
      </div>

      <div class="field">
        <label>Confidence in your identification</label>
        <div class="likert-wrap">
          <span class="likert-anchor">1 = Pure guess</span>
          <div class="likert-opts" id="lik-ident-conf"></div>
          <span class="likert-anchor">5 = Very confident</span>
        </div>
      </div>

      <div class="field">
        <label>Comments <span class="opt">(optional — what cues helped you identify the responses?)</span></label>
        <textarea id="ident-comments" style="min-height:56px" placeholder="e.g., Response C referenced a specific paper I associate with Dr. Allen…"></textarea>
      </div>
    </div>

    <div id="eval-error" class="error-banner" style="display:none"></div>

    <div class="btn-row">
      <button class="btn btn-primary" id="btn-next" disabled>Next Question →</button>
    </div>
  </div>

  <!-- Screen 4: Complete -->
  <div id="screen-complete" class="screen">
    <div class="card complete-hero">
      <div class="checkmark">✓</div>
      <h2>Session Complete</h2>
      <p>Thank you for participating in the ALICE Knowledge Retention study.</p>
      <p style="margin-top:6px">Your responses have been recorded.</p>
    </div>
    <div class="card" id="complete-stats" style="display:none">
      <h2 class="section-title">Session Summary</h2>
      <div id="complete-stats-body"></div>
      <div class="btn-row" style="justify-content:flex-start;margin-top:16px">
        <button class="btn btn-secondary" id="btn-download">↓ Download Raw Data</button>
        <button class="btn btn-primary" id="btn-new-session">Start Another Session</button>
      </div>
    </div>
  </div>

</div>

<div id="toast"></div>

<script>
// ── State ────────────────────────────────────────────────────────────────────
const S = {
  email: '',
  sessionId: null,
  currentBlock: null,
  currentIndex: 0,
  totalQuestions: 0,
  aliases: [],
  nLoaded: 0,
  nTotal: 0,
  queried: false,
};

// ── Markdown renderer ─────────────────────────────────────────────────────────

function inlineFmt(t) {
  return t
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/_(.+?)_/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>');
}

function renderMd(raw) {
  // Strip ALICE citation markers: (Fact3), (Fact_7), (Fact3, Fact7, Fact_20), etc.
  // Strip KB-miss disclaimer that reveals ALICE provenance
  let s = raw
    .replace(/\(\s*Fact_?\d+(?:\s*,\s*Fact_?\d+)*\s*\)/g, '')
    .replace(/The knowledge graph does not contain[^\n]*/g, '')
    .replace(/The following answer is based on general knowledge[^\n]*/g, '')
    .replace(/\u26A0\uFE0F?/g, '')
    .trim();

  // HTML-escape after stripping so we don't clobber & in citations
  s = s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  const lines = s.split('\n');
  let html = '', inUl = false, inOl = false;

  const closeList = () => {
    if (inUl) { html += '</ul>'; inUl = false; }
    if (inOl) { html += '</ol>'; inOl = false; }
  };

  for (const line of lines) {
    const hm = line.match(/^(#{1,4})\s+(.+)/);
    if (hm) { closeList(); html += `<h${hm[1].length}>${inlineFmt(hm[2])}</h${hm[1].length}>`; continue; }

    const um = line.match(/^[-*]\s+(.+)/);
    if (um) { if (inOl) { html += '</ol>'; inOl = false; } if (!inUl) { html += '<ul>'; inUl = true; } html += `<li>${inlineFmt(um[1])}</li>`; continue; }

    const om = line.match(/^\d+\.\s+(.+)/);
    if (om) { if (inUl) { html += '</ul>'; inUl = false; } if (!inOl) { html += '<ol>'; inOl = true; } html += `<li>${inlineFmt(om[1])}</li>`; continue; }

    closeList();
    if (line.trim() === '') { html += ''; continue; }
    html += `<p>${inlineFmt(line)}</p>`;
  }
  closeList();
  return html;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function toast(msg, ms = 2800) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), ms);
}

function showErr(id, msg) {
  const el = document.getElementById(id);
  if (el) { el.textContent = msg; el.style.display = 'block'; }
}

function clearErr(id) {
  const el = document.getElementById(id);
  if (el) { el.style.display = 'none'; el.textContent = ''; }
}

function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  const s = document.getElementById(id);
  if (s) s.classList.add('active');
  document.getElementById('loading-screen').style.display = 'none';
}

async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Likert builder ────────────────────────────────────────────────────────────

function buildLikert(containerId, name, initial) {
  const c = document.getElementById(containerId);
  if (!c) return;
  c.innerHTML = '';
  for (let i = 1; i <= 5; i++) {
    const lbl = document.createElement('label');
    const inp = document.createElement('input');
    inp.type = 'radio'; inp.name = name; inp.value = i;
    if (initial && +initial === i) inp.checked = true;
    lbl.appendChild(inp);
    lbl.appendChild(document.createTextNode(i));
    c.appendChild(lbl);
  }
}

function getLikert(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? +el.value : null;
}

// ── Bio form ─────────────────────────────────────────────────────────────────

function initBioLikerts(p) {
  p = p || {};
  buildLikert('lik-alice',      'l-alice',      p.familiarity_alice);
  buildLikert('lik-d-research', 'l-d-research', p.familiarity_d_research);
  buildLikert('lik-n-research', 'l-n-research', p.familiarity_n_research);
  buildLikert('lik-d-comms',    'l-d-comms',    p.familiarity_d_comms);
  buildLikert('lik-n-comms',    'l-n-comms',    p.familiarity_n_comms);
}

function prefillBio(p) {
  if (!p) return;
  const set = (id, v) => { const el = document.getElementById(id); if (el && v != null && v !== '') el.value = v; };
  set('bio-role', p.role_title);
  set('bio-org', p.organization);
  set('bio-degree', p.highest_degree);
  set('bio-degree-field', p.degree_field);
  set('bio-years-nasa', p.years_at_nasa);
  set('bio-years-domain', p.years_in_domain);
  set('bio-years-aero', p.years_in_aerospace);
  set('bio-years-d', p.years_known_d);
  set('bio-years-n', p.years_known_n);
  set('bio-freq-d', p.interaction_freq_d);
  set('bio-freq-n', p.interaction_freq_n);
  set('bio-papers-d', p.papers_coauthored_d);
  set('bio-papers-n', p.papers_coauthored_n);
  set('bio-projects', p.projects_with_d_or_n);
  set('bio-papers-read', p.papers_read);
  set('bio-age', p.age_range);
  set('bio-gender', p.gender);
}

function collectBio() {
  const num = id => { const v = document.getElementById(id)?.value; return (v !== '' && v != null) ? +v : null; };
  const str = id => document.getElementById(id)?.value?.trim() || '';
  return {
    role_title: str('bio-role'),
    organization: str('bio-org'),
    highest_degree: str('bio-degree'),
    degree_field: str('bio-degree-field'),
    years_at_nasa: num('bio-years-nasa'),
    years_in_domain: num('bio-years-domain'),
    years_in_aerospace: num('bio-years-aero'),
    years_known_d: num('bio-years-d'),
    years_known_n: num('bio-years-n'),
    interaction_freq_d: str('bio-freq-d'),
    interaction_freq_n: str('bio-freq-n'),
    papers_coauthored_d: num('bio-papers-d'),
    papers_coauthored_n: num('bio-papers-n'),
    projects_with_d_or_n: str('bio-projects'),
    familiarity_alice: getLikert('l-alice'),
    familiarity_d_research: getLikert('l-d-research'),
    familiarity_n_research: getLikert('l-n-research'),
    papers_read: str('bio-papers-read'),
    age_range: str('bio-age'),
    gender: str('bio-gender'),
    familiarity_d_comms: getLikert('l-d-comms'),
    familiarity_n_comms: getLikert('l-n-comms'),
  };
}

// ── Eval screen ───────────────────────────────────────────────────────────────

function buildResponseCard(resp) {
  const card = document.createElement('div');
  card.className = 'response-card';
  card.dataset.rid = resp.response_id;
  card.dataset.alias = resp.alias;

  card.innerHTML = `
    <div class="resp-header">
      <span class="resp-alias">Response ${resp.alias}</span>
      <span class="resp-timing" id="t-${resp.response_id}"></span>
    </div>
    <div class="resp-body" id="b-${resp.response_id}">
      <div class="resp-loading"><div class="spinner"></div>Waiting for response…</div>
    </div>
    <div class="resp-ratings hidden" id="r-${resp.response_id}">
      <div class="rating-item">
        <div class="rl">Accuracy</div>
        <div class="rating-anchor">1 = Significantly inaccurate &nbsp;·&nbsp; 5 = Completely accurate</div>
        <div class="radio-row" id="acc-${resp.response_id}"></div>
      </div>
      <div class="rating-item">
        <div class="rl">Humanness</div>
        <div class="rating-anchor">1 = Obviously machine-generated &nbsp;·&nbsp; 5 = Completely natural / indistinguishable from an informal email or conversation</div>
        <div class="radio-row" id="hum-${resp.response_id}"></div>
      </div>
      <div class="rating-item rating-comment">
        <div class="rl" style="font-size:.76rem;font-weight:600;color:var(--muted)">Comments <span style="font-weight:400">(optional)</span></div>
        <input type="text" id="c-${resp.response_id}" placeholder="Any observations about this response…">
      </div>
    </div>`;
  return card;
}

function buildRadioRow(containerId, name) {
  const c = document.getElementById(containerId);
  if (!c) return;
  c.innerHTML = '';
  for (let i = 1; i <= 5; i++) {
    const lbl = document.createElement('label');
    const inp = document.createElement('input');
    inp.type = 'radio'; inp.name = name; inp.value = i;
    inp.addEventListener('change', checkReady);
    lbl.appendChild(inp);
    lbl.appendChild(document.createTextNode(` ${i}`));
    c.appendChild(lbl);
  }
}

function onResponseArrived(rid, text, elapsedMs) {
  const body = document.getElementById(`b-${rid}`);
  if (body) body.innerHTML = renderMd(text);

  const timing = document.getElementById(`t-${rid}`);
  if (timing) timing.textContent = `${(elapsedMs / 1000).toFixed(1)}s`;

  const ratings = document.getElementById(`r-${rid}`);
  if (ratings) {
    ratings.classList.remove('hidden');
    buildRadioRow(`acc-${rid}`, `acc-${rid}`);
    buildRadioRow(`hum-${rid}`, `hum-${rid}`);
  }

  S.nLoaded++;
  document.getElementById('query-status').textContent =
    `${S.nLoaded} / ${S.nTotal} responses received`;

  if (S.nLoaded >= S.nTotal) {
    document.getElementById('query-status').textContent = 'All responses received.';
    showIdentSection();
  }
}

function showIdentSection() {
  const sec = document.getElementById('ident-section');
  sec.classList.remove('hidden');

  const aliases = S.aliases;
  ['ident-vd', 'ident-vn'].forEach(id => {
    const sel = document.getElementById(id);
    sel.innerHTML = '<option value="">— select —</option>';
    aliases.forEach(a => {
      const opt = document.createElement('option');
      opt.value = a; opt.textContent = `Response ${a}`;
      sel.appendChild(opt);
    });
    sel.addEventListener('change', checkReady);
  });

  buildLikert('lik-ident-conf', 'l-ident-conf', null);
  document.querySelectorAll('#lik-ident-conf input').forEach(i => i.addEventListener('change', checkReady));
}

function checkReady() {
  const allRated = S.aliases.length > 0 && S.aliases.every(alias => {
    const card = document.querySelector(`.response-card[data-alias="${alias}"]`);
    if (!card) return false;
    const rid = card.dataset.rid;
    return document.querySelector(`input[name="acc-${rid}"]:checked`) &&
           document.querySelector(`input[name="hum-${rid}"]:checked`);
  });

  const identOk = document.getElementById('ident-vd')?.value &&
                  document.getElementById('ident-vn')?.value &&
                  getLikert('l-ident-conf');

  document.getElementById('btn-next').disabled = !(allRated && identOk && S.nLoaded >= S.nTotal);
}

function renderEval(session) {
  const block = session.current_question;
  const prog  = session.progress;
  S.currentBlock   = block;
  S.currentIndex   = prog.current_index;
  S.totalQuestions = prog.total_questions;
  S.nLoaded        = 0;
  S.nTotal         = block.responses.length;
  S.aliases        = block.responses.map(r => r.alias);
  S.queried        = false;

  const pct = ((prog.current_index + 1) / prog.total_questions) * 100;
  document.getElementById('eval-progress-label').textContent =
    `Question ${prog.current_index + 1} of ${prog.total_questions}`;
  document.getElementById('eval-progress-bar').style.width = `${pct}%`;
  document.getElementById('eval-progress-cat').textContent = block.category || '';

  document.getElementById('eval-question').textContent = block.question_text;

  const grid = document.getElementById('response-grid');
  grid.innerHTML = '';
  block.responses.forEach(r => grid.appendChild(buildResponseCard(r)));

  // Reset identity section
  const sec = document.getElementById('ident-section');
  sec.classList.add('hidden');
  document.getElementById('ident-vd').innerHTML = '<option value="">— select —</option>';
  document.getElementById('ident-vn').innerHTML = '<option value="">— select —</option>';
  document.getElementById('ident-comments').value = '';

  const isLast = (prog.current_index + 1) >= prog.total_questions;
  const btnNext = document.getElementById('btn-next');
  btnNext.disabled = true;
  btnNext.textContent = isLast ? 'Finish Session →' : 'Next Question →';

  const btnQ = document.getElementById('btn-query');
  btnQ.disabled = false;
  btnQ.textContent = '▶ Query All Agents';
  document.getElementById('query-status').textContent = '';
  clearErr('eval-error');

  showScreen('screen-eval');
}

// ── Init ─────────────────────────────────────────────────────────────────────

async function init() {
  try {
    const me = await api('GET', '/api/me');
    S.email = me.email || '';
  } catch (_) { S.email = ''; }

  document.getElementById('user-badge').textContent = S.email || 'anonymous';
  document.getElementById('bio-email-val').textContent = S.email || 'anonymous (no auth)';

  // Check for existing in-progress session
  let existing = null;
  try {
    const r = await api('POST', '/api/experiment/sessions/resume',
      { participant_id: S.email || 'anonymous' });
    if (r.status === 'resumed') existing = r.session;
  } catch (_) {}

  // Load profile
  let profile = null;
  try {
    const r = await api('GET', '/api/experiment/profile');
    profile = r.profile;
  } catch (_) {}

  if (existing && existing.status === 'in_progress') {
    const resume = confirm(
      `You have an in-progress session (question ${existing.progress.current_index + 1} of ${existing.progress.total_questions}). Resume it?`
    );
    if (resume) {
      S.sessionId = existing.session_id;
      renderEval(existing);
      return;
    }
  }

  initBioLikerts(profile);
  prefillBio(profile);
  showScreen('screen-bio');
}

// ── Bio submit ────────────────────────────────────────────────────────────────

document.getElementById('btn-bio-submit').addEventListener('click', async () => {
  clearErr('bio-error');
  const bio = collectBio();
  if (!bio.role_title)     { showErr('bio-error', 'Please enter your current role / title.'); return; }
  if (!bio.organization)   { showErr('bio-error', 'Please enter your organization.'); return; }
  if (!bio.highest_degree) { showErr('bio-error', 'Please select your highest degree.'); return; }
  if (!bio.degree_field)   { showErr('bio-error', 'Please enter your degree field.'); return; }

  const btn = document.getElementById('btn-bio-submit');
  btn.disabled = true; btn.textContent = 'Saving…';
  try {
    await api('PUT', '/api/experiment/profile', bio);
    showScreen('screen-setup');
  } catch (e) {
    showErr('bio-error', `Could not save profile: ${e.message}`);
  } finally {
    btn.disabled = false; btn.textContent = 'Save & Continue →';
  }
});

// ── Setup ─────────────────────────────────────────────────────────────────────

const nqSlider = document.getElementById('setup-nq');
nqSlider.addEventListener('input', () => {
  document.getElementById('setup-nq-display').textContent = nqSlider.value;
});

document.getElementById('btn-setup-back').addEventListener('click', () => showScreen('screen-bio'));

document.getElementById('btn-setup-start').addEventListener('click', async () => {
  clearErr('setup-error');
  const nq = +nqSlider.value;
  const btn = document.getElementById('btn-setup-start');
  btn.disabled = true; btn.textContent = 'Starting…';
  try {
    const res = await api('POST', '/api/experiment/sessions/start', {
      participant_id: S.email || 'anonymous',
      questionnaire: {},
      questions_per_session: nq,
      responses_per_question: 6,
      response_mode: 'live',
      force_new_session: true,
    });
    S.sessionId = res.session.session_id;
    renderEval(res.session);
  } catch (e) {
    showErr('setup-error', `Could not start session: ${e.message}`);
    btn.disabled = false; btn.textContent = 'Start Session →';
  }
});

// ── Query agents ──────────────────────────────────────────────────────────────

document.getElementById('btn-query').addEventListener('click', async () => {
  if (S.queried) return;
  S.queried = true;
  const btn = document.getElementById('btn-query');
  btn.disabled = true; btn.textContent = '⏳ Querying…';
  document.getElementById('query-status').textContent = `0 / ${S.nTotal} responses received`;

  const block = S.currentBlock;
  const starts = {};
  const promises = block.responses.map(r => {
    starts[r.response_id] = Date.now();
    return fetch(
      `/api/experiment/sessions/${S.sessionId}/questions/${block.q_id}/responses/${r.response_id}/query`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }
    )
    .then(res => {
      if (!res.ok) return res.json().then(e => { throw new Error(e.detail || 'Query failed'); });
      return res.json();
    })
    .then(data => onResponseArrived(r.response_id, data.text || '[No response]', Date.now() - starts[r.response_id]))
    .catch(err => onResponseArrived(r.response_id, `[Error: ${err.message}]`, Date.now() - starts[r.response_id]));
  });

  await Promise.allSettled(promises);
});

// ── Next / Finish ─────────────────────────────────────────────────────────────

document.getElementById('btn-next').addEventListener('click', async () => {
  clearErr('eval-error');
  const btn = document.getElementById('btn-next');
  btn.disabled = true; btn.textContent = 'Saving…';

  const block = S.currentBlock;
  const ratings = block.responses.map(r => ({
    response_id: r.response_id,
    accuracy:  +document.querySelector(`input[name="acc-${r.response_id}"]:checked`)?.value || 0,
    humanness: +document.querySelector(`input[name="hum-${r.response_id}"]:checked`)?.value || 0,
    comments:  document.getElementById(`c-${r.response_id}`)?.value?.trim() || null,
  }));

  const identification = {
    virtual_d_alias: document.getElementById('ident-vd').value,
    virtual_n_alias: document.getElementById('ident-vn').value,
    confidence: getLikert('l-ident-conf'),
    comments:   document.getElementById('ident-comments')?.value?.trim() || null,
  };

  try {
    await api('PUT', `/api/experiment/sessions/${S.sessionId}/questions/${block.q_id}`,
      { ratings, identification });

    const isLast = (S.currentIndex + 1) >= S.totalQuestions;
    if (isLast) {
      const res = await api('POST', `/api/experiment/sessions/${S.sessionId}/submit`);
      enterComplete(res.session);
    } else {
      const res = await api('POST', `/api/experiment/sessions/${S.sessionId}/next`);
      renderEval(res.session);
    }
  } catch (e) {
    showErr('eval-error', `Error saving ratings: ${e.message}`);
    btn.disabled = false;
    btn.textContent = (S.currentIndex + 1) >= S.totalQuestions ? 'Finish Session →' : 'Next Question →';
  }
});

// ── Complete screen ───────────────────────────────────────────────────────────

function enterComplete(session) {
  showScreen('screen-complete');
  const evals = session.evaluations || {};
  const nEvals = Object.keys(evals).length;
  if (!nEvals) return;

  let accSum = 0, humSum = 0, total = 0;
  Object.values(evals).forEach(e => {
    (e.ratings || []).forEach(r => {
      accSum += r.accuracy || 0;
      humSum += r.humanness || 0;
      total++;
    });
  });

  const statsDiv = document.getElementById('complete-stats');
  const body = document.getElementById('complete-stats-body');
  statsDiv.style.display = 'block';
  body.innerHTML = `
    <p><strong>Questions evaluated:</strong> ${nEvals}</p>
    <p><strong>Average accuracy rating:</strong> ${total ? (accSum/total).toFixed(2) : '—'} / 5</p>
    <p><strong>Average humanness rating:</strong> ${total ? (humSum/total).toFixed(2) : '—'} / 5</p>
  `;

  document.getElementById('btn-download').addEventListener('click', async () => {
    try {
      const data = await api('GET', `/api/experiment/sessions/${session.session_id}/export`);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `alice-eval-${session.session_id}.json`;
      a.click(); URL.revokeObjectURL(url);
    } catch (e) { alert(`Download failed: ${e.message}`); }
  }, { once: true });

  document.getElementById('btn-new-session').addEventListener('click', () => {
    S.sessionId = null;
    showScreen('screen-setup');
  }, { once: true });
}

// ── Boot ──────────────────────────────────────────────────────────────────────

init().catch(err => {
  console.error('Init error:', err);
  document.getElementById('loading-screen').innerHTML =
    `<div style="color:var(--nasa-red);font-size:1rem">Failed to initialize: ${err.message}</div>`;
});
</script>
</body>
</html>"""
