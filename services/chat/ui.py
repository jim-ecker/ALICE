"""Inline HTML/CSS/JS for the ALICE chat browser UI."""

CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ALICE — Knowledge Graph Chat</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #22263a;
    --border: #2e3350; --accent: #5b8af0; --accent2: #7c6af7;
    --text: #e2e8f0; --text2: #8892a8; --green: #34d399; --red: #f87171;
    --yellow: #fbbf24; --radius: 8px; --font: 'Segoe UI', system-ui, sans-serif;
  }
  body { background: var(--bg); color: var(--text); font-family: var(--font);
         height: 100vh; display: flex; overflow: hidden; }

  /* Sidebar */
  #sidebar { width: 260px; min-width: 260px; background: var(--surface);
             border-right: 1px solid var(--border); display: flex;
             flex-direction: column; height: 100vh; }
  #sidebar-header { padding: 16px; border-bottom: 1px solid var(--border); }
  #sidebar-header h1 { font-size: 18px; font-weight: 700; color: var(--accent);
                        letter-spacing: .5px; display: flex; align-items: center; gap: 8px; }
  #sidebar-header h1 img { width: 22px; height: 22px; object-fit: contain; border-radius: 3px; }
  #sidebar-header p { font-size: 11px; color: var(--text2); margin-top: 2px; }

  /* Mode toggle — slide switch */
  #mode-toggle { display: flex; align-items: center; gap: 8px; margin: 12px 14px 0;
                 user-select: none; }
  #mode-toggle .mode-label { font-size: 11px; color: var(--text2); white-space: nowrap;
                              transition: color .2s; }
  #mode-toggle .mode-label.active { color: var(--text); font-weight: 600; }
  /* Hidden checkbox drives state */
  #mode-checkbox { display: none; }
  /* Track */
  #mode-track { position: relative; width: 40px; min-width: 40px; height: 22px;
                background: var(--surface2); border: 1px solid var(--border);
                border-radius: 11px; cursor: pointer; transition: background .2s, border-color .2s; }
  #mode-checkbox:checked + #mode-track { background: var(--accent2); border-color: var(--accent2); }
  /* Thumb */
  #mode-track::after { content: ''; position: absolute; top: 3px; left: 3px;
                        width: 14px; height: 14px; border-radius: 50%;
                        background: var(--text2); transition: transform .2s, background .2s; }
  #mode-checkbox:checked + #mode-track::after { transform: translateX(18px); background: #fff; }

  #new-conv-btn { margin: 10px 12px 0; padding: 8px 12px; background: var(--accent);
                  color: #fff; border: none; border-radius: var(--radius);
                  cursor: pointer; font-size: 13px; font-weight: 600; width: calc(100% - 24px); }
  #new-conv-btn:hover { background: #4a79df; }
  #conv-list { flex: 1; overflow-y: auto; padding: 4px 8px; }
  .conv-item { padding: 8px 10px; border-radius: var(--radius); cursor: pointer;
               margin-bottom: 2px; display: flex; align-items: center;
               justify-content: space-between; font-size: 13px; color: var(--text2); }
  .conv-item:hover { background: var(--surface2); color: var(--text); }
  .conv-item.active { background: var(--accent2); color: #fff; }
  .conv-item .conv-title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .conv-item .del-btn { opacity: 0; font-size: 14px; padding: 0 4px; color: var(--red);
                         cursor: pointer; border: none; background: none; }
  .conv-item:hover .del-btn { opacity: 1; }

  /* Expert selector panel */
  #expert-panel { display: none; flex-direction: column; border-top: 1px solid var(--border);
                  padding: 10px 10px 4px; gap: 6px; }
  body[data-mode="retention"] #expert-panel { display: flex; }
  body[data-mode="retention"] #conv-list { flex: 1; }

  #expert-panel-header { font-size: 11px; font-weight: 700; color: var(--text2);
                          text-transform: uppercase; letter-spacing: .5px; padding: 0 2px 2px; }
  #expert-cards { display: flex; flex-direction: column; gap: 4px;
                  max-height: 220px; overflow-y: auto; }
  .expert-card { padding: 8px 10px; border-radius: var(--radius); cursor: pointer;
                 border: 1px solid var(--border); background: var(--surface2);
                 font-size: 12px; position: relative; }
  .expert-card:hover { border-color: var(--accent); }
  .expert-card.active { border-color: var(--accent2); background: var(--surface); }
  .expert-card.loading { border-color: var(--yellow); cursor: wait;
                          animation: pulse-border 1s ease-in-out infinite; }
  .expert-card.switching-blocked { pointer-events: none; opacity: .5; cursor: not-allowed; }
  @keyframes pulse-border { 0%,100% { border-color: var(--yellow); }
                             50% { border-color: transparent; } }
  .expert-card .expert-name { font-weight: 600; color: var(--text); margin-bottom: 4px; }
  .expert-card .expert-areas { display: flex; flex-wrap: wrap; gap: 3px; }
  .expert-card .area-tag { font-size: 10px; color: var(--accent2); background: rgba(124,106,247,.12);
                            border: 1px solid rgba(124,106,247,.25); border-radius: 3px;
                            padding: 1px 5px; white-space: nowrap; overflow: hidden;
                            text-overflow: ellipsis; max-width: 120px; }
  .expert-card .no-areas { color: var(--text2); font-size: 10px; font-style: italic; }
  .expert-card .active-badge { position: absolute; top: 6px; right: 8px;
                                 font-size: 9px; font-weight: 700; color: var(--accent2);
                                 text-transform: uppercase; letter-spacing: .5px; }
  .expert-card.no-db { opacity: .45; cursor: not-allowed; }

  /* Status bar */
  #status-bar { padding: 6px 16px; border-top: 1px solid var(--border);
                font-size: 11px; color: var(--text2); }

  /* Main area */
  #main { flex: 1; display: flex; flex-direction: column; height: 100vh; min-width: 0; }

  /* Messages */
  #messages { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex;
              flex-direction: column; gap: 16px; }
  #empty-state { display: flex; flex-direction: column; align-items: center;
                  justify-content: center; flex: 1; color: var(--text2);
                  gap: 8px; text-align: center; }
  #empty-state .icon img { width: 72px; height: 72px; object-fit: contain; opacity: 0.5; }
  #empty-state h2 { font-size: 20px; color: var(--text); }

  .msg { display: flex; gap: 10px; max-width: 820px; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg-bubble { padding: 10px 14px; border-radius: var(--radius);
                font-size: 14px; line-height: 1.55; max-width: 680px; }
  .msg.user .msg-bubble { background: var(--accent); color: #fff; border-bottom-right-radius: 2px; }
  .msg.assistant .msg-bubble { background: var(--surface); border: 1px solid var(--border);
                                 border-bottom-left-radius: 2px; }
  .msg-avatar { width: 48px; height: 48px; border-radius: 50%; display: flex;
                align-items: center; justify-content: center;
                flex-shrink: 0; background: var(--surface2); border: 1px solid var(--border);
                overflow: hidden; }
  .msg-avatar img { width: 100%; height: 100%; object-fit: cover; }
  .msg.user .msg-avatar { background: var(--surface2); }

  /* Citations */
  .citations { margin-top: 10px; display: flex; flex-direction: column; gap: 6px; }
  .citation-card { background: var(--surface2); border: 1px solid var(--border);
                   border-radius: var(--radius); padding: 8px 10px; font-size: 12px; }
  .citation-source { font-weight: 600; color: var(--accent); margin-bottom: 2px; }
  .citation-location { font-size: 11px; color: var(--text2); margin-bottom: 4px; }
  .citation-excerpt { color: var(--text2); font-style: italic; margin-bottom: 4px;
                       overflow: hidden; text-overflow: ellipsis; display: -webkit-box;
                       -webkit-line-clamp: 3; -webkit-box-orient: vertical; }
  .citation-triples { display: flex; flex-direction: column; gap: 4px; }
  .triple-row { font-size: 11px; color: var(--text2); border-radius: 4px; padding: 2px 4px; transition: background 0.3s; }
  .triple-row.fact-highlight { animation: fact-flash 1.2s ease-out; }
  @keyframes fact-flash { 0% { background: var(--accent); } 100% { background: transparent; } }
  .fact-label { font-size: 10px; font-weight: 700; color: var(--text2); opacity: 0.6;
                margin-right: 5px; font-family: monospace; }
  .fact-ref { color: var(--accent2); font-size: 12px; cursor: pointer; text-decoration: none; font-family: monospace; }
  .fact-ref:hover { text-decoration: underline; }
  .doc-link { color: inherit; text-decoration: underline; text-decoration-color: var(--border); }
  .doc-link:hover { text-decoration-color: var(--accent); color: var(--accent); }
  .msg-walltime { font-size: 10px; color: var(--text2); opacity: 0.5; text-align: right; margin-top: 6px; }
  .triple-row .entity { color: var(--accent2); font-weight: 600; }
  .triple-row .relation { color: var(--yellow); }
  /* Trust score bar row */
  .trust-bar-row { display: flex; gap: 6px; align-items: center; margin-top: 2px; flex-wrap: wrap; }
  .trust-pill { display: flex; align-items: center; gap: 3px; font-size: 10px;
                background: var(--bg); border-radius: 3px; padding: 1px 5px; }
  .trust-pill .label { color: var(--text2); }
  .trust-pill .bar-wrap { width: 36px; height: 4px; background: var(--border);
                           border-radius: 2px; overflow: hidden; }
  .trust-pill .bar-fill { height: 100%; border-radius: 2px; }
  .trust-pill .val { min-width: 26px; text-align: right; }
  .pill-composite .bar-fill { background: var(--accent); }
  .pill-ingest .bar-fill { background: var(--green); }
  .pill-relevance .bar-fill { background: var(--accent2); }
  .pill-grounding .bar-fill { background: var(--yellow); }
  .pill-provenance .val { color: var(--text2); }

  /* Input area */
  #input-area { padding: 16px 24px; border-top: 1px solid var(--border);
                display: flex; gap: 10px; align-items: flex-end; }
  #msg-input { flex: 1; background: var(--surface); border: 1px solid var(--border);
               border-radius: var(--radius); color: var(--text); padding: 10px 14px;
               font-size: 14px; resize: none; min-height: 42px; max-height: 160px;
               font-family: var(--font); line-height: 1.4; }
  #msg-input:focus { outline: none; border-color: var(--accent); }
  #send-btn { padding: 10px 20px; background: var(--accent); color: #fff;
              border: none; border-radius: var(--radius); cursor: pointer;
              font-size: 14px; font-weight: 600; white-space: nowrap; }
  #send-btn:hover { background: #4a79df; }
  #send-btn:disabled { opacity: .5; cursor: not-allowed; }

  /* Thinking indicator */
  .thinking { display: flex; gap: 4px; align-items: center; padding: 8px 2px; }
  .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text2);
          animation: bounce .9s infinite; }
  .dot:nth-child(2) { animation-delay: .15s; }
  .dot:nth-child(3) { animation-delay: .3s; }
  @keyframes bounce { 0%,80%,100% { transform: scale(.6); opacity:.5 }
                       40% { transform: scale(1); opacity:1 } }

  /* Modal */
  #modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.6);
                    display: none; align-items: center; justify-content: center; z-index: 100; }
  #modal-overlay.open { display: flex; }
  #modal { background: var(--surface); border: 1px solid var(--border);
            border-radius: 12px; padding: 24px; width: 340px; }
  #modal h2 { font-size: 16px; margin-bottom: 12px; }
  #modal input { width: 100%; padding: 8px 10px; background: var(--surface2);
                  border: 1px solid var(--border); border-radius: var(--radius);
                  color: var(--text); font-size: 14px; margin-bottom: 12px; }
  #modal input:focus { outline: none; border-color: var(--accent); }
  #modal-actions { display: flex; gap: 8px; justify-content: flex-end; }
  #modal-actions button { padding: 7px 16px; border-radius: var(--radius);
                            border: none; cursor: pointer; font-size: 13px; font-weight: 600; }
  #modal-cancel { background: var(--surface2); color: var(--text); }
  #modal-ok { background: var(--accent); color: #fff; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body data-mode="chat">

<!-- Sidebar -->
<div id="sidebar">
  <div id="sidebar-header">
    <h1><img src="/static/nasa.png" alt="NASA"> ALICE</h1>
    <p>AI Leveraged Information Capture and Exploration</p>
  </div>

  <!-- Mode toggle — slide switch -->
  <div id="mode-toggle">
    <span class="mode-label active" id="label-chat">Chat</span>
    <input type="checkbox" id="mode-checkbox">
    <label id="mode-track" for="mode-checkbox"></label>
    <span class="mode-label" id="label-retention">Knowledge Retention</span>
  </div>

  <button id="new-conv-btn">+ New Conversation</button>
  <div id="conv-list"></div>

  <!-- Expert selector (visible only in Knowledge Retention mode) -->
  <div id="expert-panel">
    <div id="expert-panel-header">Virtual Experts</div>
    <div id="expert-cards"></div>
  </div>

  <div id="status-bar">Loading...</div>
</div>

<!-- Main chat area -->
<div id="main">
  <div id="messages">
    <div id="empty-state">
      <div class="icon"><img src="/static/nasa.png" alt="ALICE"></div>
      <h2>ALICE Research Assistant</h2>
      <p></p>
    </div>
  </div>
  <div id="input-area">
    <textarea id="msg-input" placeholder="Ask a question..." rows="1"></textarea>
    <button id="send-btn">Send</button>
  </div>
</div>

<!-- New conversation modal -->
<div id="modal-overlay">
  <div id="modal">
    <h2>New Conversation</h2>
    <input type="text" id="conv-title-input" placeholder="Conversation title">
    <div id="modal-actions">
      <button id="modal-cancel">Cancel</button>
      <button id="modal-ok">Create</button>
    </div>
  </div>
</div>

<script>
const API = '';
let activeConvId = null;
let sending = false;
let activeExpertSlug = null;
let switchingExpert = null;  // slug currently being loaded, or null

// Utility
const $  = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

// Auto-resize textarea
$('msg-input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 160) + 'px';
});

// Send on Enter (Shift+Enter = newline)
$('msg-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

$('send-btn').addEventListener('click', sendMessage);

// ── Fact ref click → scroll & highlight matching triple row ───────────────────
document.addEventListener('click', e => {
  const ref = e.target.closest('.fact-ref');
  if (!ref) return;
  const factIdx = ref.dataset.fact;
  const bubble = ref.closest('.msg-bubble');
  if (!bubble) return;
  const row = bubble.querySelector(`.triple-row[data-fact="${factIdx}"]`);
  if (!row) return;
  row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  row.classList.remove('fact-highlight');
  void row.offsetWidth; // force reflow to restart animation
  row.classList.add('fact-highlight');
});

// ── Mode toggle ───────────────────────────────────────────────────────────────
function setMode(mode) {
  const retention = mode === 'retention';
  document.body.dataset.mode = mode;
  $('mode-checkbox').checked = retention;
  $('label-chat').classList.toggle('active', !retention);
  $('label-retention').classList.toggle('active', retention);
  if (retention) {
    loadExperts();
  } else {
    unloadExpert();
  }
}

$('mode-checkbox').addEventListener('change', () => {
  setMode($('mode-checkbox').checked ? 'retention' : 'chat');
});

// ── Status ──────────────────────────────────────────────────────────────────
async function loadStatus() {
  try {
    const r = await fetch(`${API}/api/status`);
    const d = await r.json();
    $('status-bar').innerHTML = `
      <div style="display:grid;grid-template-columns:max-content 1fr;gap:1px 8px;line-height:1.6">
        <span>Backend:</span><span>${esc(d.backend)}</span>
        <span>Model Loaded:</span><span>${esc(d.model.replace('mlx-community/', ''))}</span>
        <span>Total Chunks:</span><span>${d.index_size}</span>
        ${activeExpertSlug ? `<span>Expert:</span><span>${esc(activeExpertSlug)}</span>` : ''}
      </div>`;
  } catch { $('status-bar').textContent = 'disconnected'; }
}

// ── Experts ───────────────────────────────────────────────────────────────────
async function loadExperts() {
  try {
    const r = await fetch(`${API}/api/experts`);
    const d = await r.json();
    const container = $('expert-cards');
    container.innerHTML = '';
    for (const expert of d.experts) {
      const card = document.createElement('div');
      const isActive = expert.slug === activeExpertSlug;
      const isLoading = expert.slug === switchingExpert;
      const isBlocked = switchingExpert !== null && !isLoading;
      const noDb = !expert.db_exists;
      card.className = 'expert-card'
        + (isActive ? ' active' : '')
        + (noDb ? ' no-db' : '')
        + (isLoading ? ' loading' : '')
        + (isBlocked ? ' switching-blocked' : '');
      const areas = expert.expertise_areas || [];
      const areasHTML = isLoading
        ? '<span class="no-areas">Loading...</span>'
        : areas.length
          ? areas.slice(0, 6).map(a => `<span class="area-tag">${esc(a)}</span>`).join('')
          : '<span class="no-areas">No expertise data yet</span>';
      card.innerHTML = `
        <div class="expert-name">${esc(expert.name)}${noDb ? ' <span style="color:var(--red);font-size:10px">(no DB)</span>' : ''}</div>
        <div class="expert-areas">${areasHTML}</div>
        ${isActive && !isLoading ? '<span class="active-badge">Active</span>' : ''}
      `;
      if (!noDb && !switchingExpert) {
        card.addEventListener('click', () => switchExpert(expert.slug));
      }
      container.appendChild(card);
    }
  } catch (e) {
    $('expert-cards').innerHTML = `<div style="font-size:11px;color:var(--red)">Error loading experts</div>`;
  }
}

async function switchExpert(slug) {
  if (switchingExpert) return;
  switchingExpert = slug;
  await loadExperts();  // immediately re-render cards with loading/blocked state
  try {
    const r = await fetch(`${API}/api/experts/switch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slug })
    });
    if (!r.ok) {
      const d = await r.json();
      alert('Switch failed: ' + (d.detail || r.statusText));
      return;
    }
    const d = await r.json();
    activeExpertSlug = slug;
    activeConvId = null;
    // Show expert intro in the chat area
    if (d.intro) {
      const container = $('messages');
      container.innerHTML = '';
      container.appendChild(buildMsgEl('assistant', d.intro, []));
      scrollToBottom();
    }
  } catch (e) {
    alert('Switch failed: ' + e);
  } finally {
    switchingExpert = null;
    await loadExperts();
    await loadConversations();
    await loadStatus();
  }
}

async function unloadExpert() {
  try {
    await fetch(`${API}/api/experts/unload`, { method: 'POST' });
    activeExpertSlug = null;
    await loadExperts();
    await loadConversations();
    await loadStatus();
  } catch (e) {
    alert('Unload failed: ' + e);
  }
}

// ── Conversations ────────────────────────────────────────────────────────────
async function loadConversations() {
  const r = await fetch(`${API}/api/conversations`);
  const d = await r.json();
  const list = $('conv-list');
  list.innerHTML = '';
  for (const conv of d.conversations) {
    const el = document.createElement('div');
    el.className = 'conv-item' + (conv.id === activeConvId ? ' active' : '');
    el.dataset.id = conv.id;
    el.innerHTML = `<span class="conv-title">${esc(conv.title)}</span>
      <button class="del-btn" title="Delete">✕</button>`;
    el.querySelector('.conv-title').addEventListener('click', () => openConversation(conv.id));
    el.querySelector('.del-btn').addEventListener('click', e => { e.stopPropagation(); deleteConversation(conv.id); });
    list.appendChild(el);
  }
}

async function openConversation(id) {
  activeConvId = id;
  $('send-btn').disabled = false;
  await loadConversations();
  const r = await fetch(`${API}/api/conversations/${id}/messages`);
  const d = await r.json();
  renderMessages(d.messages);
  scrollToBottom();
}

async function deleteConversation(id) {
  if (!confirm('Delete this conversation?')) return;
  await fetch(`${API}/api/conversations/${id}`, { method: 'DELETE' });
  if (activeConvId === id) {
    activeConvId = null;
    $('messages').innerHTML = '<div id="empty-state"><div class="icon"><img src="/static/nasa.png" alt="ALICE"></div><h2>ALICE Research Assistant</h2><p>Ask a question to start a new conversation.</p></div>';
  }
  await loadConversations();
}

// ── New conversation modal ────────────────────────────────────────────────────
$('new-conv-btn').addEventListener('click', () => {
  $('conv-title-input').value = 'New Conversation';
  $('modal-overlay').classList.add('open');
  $('conv-title-input').select();
});
$('modal-cancel').addEventListener('click', () => $('modal-overlay').classList.remove('open'));
$('modal-ok').addEventListener('click', createConversation);
$('conv-title-input').addEventListener('keydown', e => { if (e.key === 'Enter') createConversation(); });

async function createConversation() {
  const title = $('conv-title-input').value.trim() || 'New Conversation';
  $('modal-overlay').classList.remove('open');
  const r = await fetch(`${API}/api/conversations`, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ title })
  });
  const d = await r.json();
  await loadConversations();
  openConversation(d.id);
}

// ── Messages ─────────────────────────────────────────────────────────────────
function renderMessages(messages) {
  const container = $('messages');
  container.innerHTML = '';
  if (!messages.length) return;
  for (const msg of messages) {
    container.appendChild(buildMsgEl(msg.role, msg.content, msg.citations || []));
  }
}

function buildMsgEl(role, content, citations, walltime = null) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const avatarImg = role === 'user'
    ? '<img src="/static/nasa.png" alt="User">'
    : '<img src="/static/alice.png" alt="ALICE">';
  const walltimeHTML = (role === 'assistant' && walltime != null)
    ? `<div class="msg-walltime">${walltime.toFixed(1)}s</div>` : '';
  const bubbleContent = role === 'assistant'
    ? formatMarkdown(content) + buildCitationsHTML(citations) + walltimeHTML
    : esc(content).replace(/\n/g, '<br>');
  div.innerHTML = `
    <div class="msg-avatar">${avatarImg}</div>
    <div class="msg-bubble">${bubbleContent}</div>`;
  return div;
}

function trustPill(label, value, cssClass, isPercent=true) {
  if (value == null || value !== value) return '';  // null or NaN
  const pct = isPercent ? Math.round(value * 100) : value;
  const width = isPercent ? Math.round(value * 100) : Math.min(Math.round(value / 5 * 100), 100);
  const display = isPercent ? `${pct}%` : `${value}×`;
  return `<span class="trust-pill ${cssClass}">
    <span class="label">${label}</span>
    <span class="bar-wrap"><span class="bar-fill" style="width:${width}%"></span></span>
    <span class="val">${display}</span>
  </span>`;
}

function buildCitationsHTML(citations) {
  if (!citations || !citations.length) return '';
  const parts = citations.map(c => {
    const page = c.page_number != null ? ` p.${c.page_number}` : '';
    const heading = c.section_heading ? ` §"${esc(c.section_heading)}"` : '';
    const triplesHTML = (c.triples || []).map(t => {
      const trustBars = `<div class="trust-bar-row">
        ${trustPill('trust', t.composite_trust, 'pill-composite')}
        ${trustPill('ingest', t.ingest_certainty, 'pill-ingest')}
        ${t.relevance_score != null ? trustPill('rel', t.relevance_score, 'pill-relevance') : ''}
        ${t.grounding_score != null ? trustPill('gnd', t.grounding_score, 'pill-grounding') : ''}
        ${t.provenance_count > 1 ? trustPill('prov', t.provenance_count, 'pill-provenance', false) : ''}
      </div>`;
      return `<div class="triple-row" data-fact="${t.fact_index}">
        <span class="fact-label">Fact_${t.fact_index}</span>
        <span class="entity">${esc(t.subject)}</span>
        <span class="relation"> --[${esc(t.relation)}]→ </span>
        <span class="entity">${esc(t.object_)}</span>
        ${trustBars}
      </div>`;
    }).join('');
    const locationParts = [page.trim(), heading.trim()].filter(Boolean).join(' ');
    return `<div class="citation-card">
      <div class="citation-source">${c.document_url ? `<a href="${esc(c.document_url)}" target="_blank" rel="noopener" class="doc-link">${esc(c.document_title)}</a>` : esc(c.document_title)}</div>
      ${locationParts ? `<div class="citation-location">${locationParts}</div>` : ''}
      <div class="citation-excerpt">${esc(c.content)}</div>
      ${triplesHTML ? `<div class="citation-triples">${triplesHTML}</div>` : ''}
    </div>`;
  });
  return `<div class="citations">${parts.join('')}</div>`;
}

// Very minimal markdown: bold, inline code, code blocks, line breaks
function formatMarkdown(text) {
  return esc(text)
    .replace(/```[\s\S]*?```/g, m => `<pre style="background:var(--surface2);padding:8px;border-radius:4px;overflow-x:auto;margin:6px 0;font-size:12px"><code>${m.slice(3,-3)}</code></pre>`)
    .replace(/`([^`]+)`/g, '<code style="background:var(--surface2);padding:1px 4px;border-radius:3px">$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\bFact_(\d+)\b/g, '<a class="fact-ref" data-fact="$1">Fact_$1</a>')
    .replace(/\n/g, '<br>');
}

function scrollToBottom() {
  const m = $('messages');
  m.scrollTop = m.scrollHeight;
}

// ── Send message ─────────────────────────────────────────────────────────────
function titleFromQuery(q) {
  if (q.length <= 60) return q;
  const cut = q.lastIndexOf(' ', 60);
  return (cut > 20 ? q.slice(0, cut) : q.slice(0, 60)) + '…';
}

async function sendMessage() {
  if (sending) return;
  const input = $('msg-input');
  const content = input.value.trim();
  if (!content) return;

  if (!activeConvId) {
    const r = await fetch(`${API}/api/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: titleFromQuery(content) })
    });
    const d = await r.json();
    activeConvId = d.id;
    await loadConversations();
  }

  sending = true;
  $('send-btn').disabled = true;
  input.value = '';
  input.style.height = 'auto';

  // Append user bubble immediately
  const container = $('messages');
  // Remove empty state
  const es = document.getElementById('empty-state');
  if (es) es.remove();
  container.appendChild(buildMsgEl('user', content, []));

  // Thinking indicator
  const thinkingEl = document.createElement('div');
  thinkingEl.className = 'msg assistant';
  thinkingEl.innerHTML = `<div class="msg-avatar"><img src="/static/alice.png" alt="ALICE"></div>
    <div class="msg-bubble"><div class="thinking">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div></div>`;
  container.appendChild(thinkingEl);
  scrollToBottom();

  const t0 = Date.now();
  try {
    const r = await fetch(`${API}/api/conversations/${activeConvId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content })
    });
    const d = await r.json();
    const walltime = (Date.now() - t0) / 1000;
    thinkingEl.remove();
    container.appendChild(buildMsgEl('assistant', d.content, d.citations || [], walltime));
    scrollToBottom();
    // Update sidebar title in-place
    if (d.new_title) {
      const titleEl = document.querySelector(`.conv-item[data-id="${activeConvId}"] .conv-title`);
      if (titleEl) titleEl.textContent = d.new_title;
    }
    await loadStatus();
  } catch (e) {
    thinkingEl.remove();
    const errEl = document.createElement('div');
    errEl.className = 'msg assistant';
    errEl.innerHTML = `<div class="msg-avatar"><img src="/static/alice.png" alt="ALICE"></div>
      <div class="msg-bubble" style="color:var(--red)">Error: ${esc(String(e))}</div>`;
    container.appendChild(errEl);
    scrollToBottom();
  } finally {
    sending = false;
    $('send-btn').disabled = false;
    input.focus();
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  // Restore active mode/expert from server state
  try {
    const r = await fetch(`${API}/api/experts/active`);
    const d = await r.json();
    if (d.slug) {
      activeExpertSlug = d.slug;
      setMode('retention');
    }
  } catch { /* server may not have experts yet */ }

  await loadStatus();
  await loadConversations();
}

init();
</script>
</body>
</html>
"""
