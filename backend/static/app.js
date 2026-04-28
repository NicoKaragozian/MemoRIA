'use strict';

const $ = id => document.getElementById(id);

const NUM_OPTIONS = 3;

let chatsCache  = [];      // [{chat_name, is_group, participants}]
let activeChat  = null;    // el chat seleccionado
let abortCtrls  = [];      // un AbortController por opción en curso
let lastResults = null;    // { options:[texto], seeds:[int], received_message, chat }
let chosenIdx   = null;    // qué opción eligió el usuario

// ── Slider ──────────────────────────────────────────────────────────────────
const slider    = $('max-tokens');
const sliderVal = $('slider-val');
slider.addEventListener('input', () => { sliderVal.textContent = slider.value; });

// ── Status helpers ──────────────────────────────────────────────────────────
function setStatus(msg, cls = '') {
  const bar = $('status-bar');
  bar.textContent = msg;
  bar.className   = cls;
}

function setLoading(loading) {
  $('btn-generate').disabled = loading;
  $('btn-stop').hidden       = !loading;
  $('btn-regenerate').hidden = loading;
}

// ── Chat selector ──────────────────────────────────────────────────────────
function renderChatMeta() {
  const meta = $('chat-meta');
  if (!activeChat) { meta.textContent = ''; return; }
  if (activeChat.is_group) {
    const n = activeChat.participants.length;
    const list = activeChat.participants.join(', ');
    meta.textContent = `Grupo de ${n} ${n === 1 ? 'persona' : 'personas'}: ${list}`;
  } else {
    meta.textContent = `Conversación 1:1 con ${activeChat.chat_name}`;
  }
}

async function loadChats() {
  const select = $('chat-select');
  try {
    const res = await fetch('/chats');
    const data = await res.json();
    chatsCache = data.chats || [];

    if (chatsCache.length === 0) {
      select.innerHTML = '<option value="" disabled selected>Sin chats parseados — corré scripts/build_dataset.py</option>';
      return;
    }

    // Agrupar 1:1 y grupos
    const oneOnOne = chatsCache.filter(c => !c.is_group);
    const groups   = chatsCache.filter(c =>  c.is_group);

    let html = '<option value="" disabled selected>Elegí un chat…</option>';
    if (oneOnOne.length) {
      html += '<optgroup label="1:1">';
      for (const c of oneOnOne) html += `<option value="${escapeHtml(c.chat_name)}">${escapeHtml(c.chat_name)}</option>`;
      html += '</optgroup>';
    }
    if (groups.length) {
      html += '<optgroup label="Grupos">';
      for (const c of groups) {
        const n = c.participants.length;
        html += `<option value="${escapeHtml(c.chat_name)}">${escapeHtml(c.chat_name)} (${n} pers.)</option>`;
      }
      html += '</optgroup>';
    }
    select.innerHTML = html;
  } catch (err) {
    select.innerHTML = '<option value="" disabled selected>Error cargando chats</option>';
    console.error(err);
  }
}

$('chat-select').addEventListener('change', e => {
  const name = e.target.value;
  activeChat = chatsCache.find(c => c.chat_name === name) || null;
  renderChatMeta();
});

// ── Parse del input: si es 1 línea, es un solo mensaje del interlocutor.
//    Si tiene varias líneas con "Author: text", se respeta el formato. ─────
function buildContextFromInput(received, chat) {
  const lines = received.split('\n').map(l => l.trim()).filter(Boolean);
  // Si solo hay una línea sin ":" o si el ":" está en posición rara, asumimos
  // que es un mensaje único del interlocutor.
  const interlocutor = chat.is_group ? null : chat.chat_name;

  // Si hay alguna línea con formato "Author: text", parseamos como contexto estructurado
  const hasStructuredFormat = lines.some(l => {
    const idx = l.indexOf(':');
    return idx > 0 && idx < 50 && l.slice(idx + 1).trim().length > 0;
  });

  if (hasStructuredFormat) {
    const out = [];
    for (const line of lines) {
      const idx = line.indexOf(':');
      if (idx > 0) {
        const author = line.slice(0, idx).trim();
        const text   = line.slice(idx + 1).trim();
        if (author && text) out.push({ author, text });
      }
    }
    return out;
  }

  // Caso simple: cada línea es un mensaje del interlocutor (o "alguien" si es grupo)
  return lines.map(text => ({
    author: interlocutor || 'Alguien',
    text,
  }));
}

// ── Generación de las 3 opciones en paralelo ───────────────────────────────
async function generateOption(idx, payloadBase, seed, optionEl) {
  const ctrl = new AbortController();
  abortCtrls[idx] = ctrl;

  const body = { ...payloadBase, seed };

  const tokensEl = optionEl.querySelector('.option-text');
  const statusEl = optionEl.querySelector('.option-status');
  statusEl.textContent = 'generando…';

  try {
    const res = await fetch('/generate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
      signal:  ctrl.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';
    let   done    = false;
    let   text    = '';

    while (true) {
      const { done: streamDone, value } = await reader.read();
      if (streamDone || done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') { done = true; break; }
        try {
          const data = JSON.parse(payload);
          if (data.error) throw new Error(data.error);
          if (data.token) { text += data.token; tokensEl.textContent = text; }
          if (data.done)  { done = true; break; }
        } catch (e) {
          if (e instanceof SyntaxError) continue;
          throw e;
        }
      }
    }

    statusEl.textContent = '';
    optionEl.classList.add('ready');
    optionEl.querySelector('.btn-choose').disabled = false;
    return text.trim();
  } catch (err) {
    if (err.name === 'AbortError') {
      statusEl.textContent = 'cancelado';
    } else {
      statusEl.textContent = `error: ${err.message}`;
      optionEl.classList.add('error');
    }
    return null;
  }
}

function buildOptionElement(idx) {
  const div = document.createElement('div');
  div.className = 'option-card';
  div.innerHTML = `
    <div class="option-header">
      <span class="option-label">Opción ${idx + 1}</span>
      <span class="option-status">esperando…</span>
    </div>
    <div class="option-text"></div>
    <button class="btn btn-primary btn-sm btn-choose" disabled data-idx="${idx}">
      Elegir esta
    </button>
  `;
  div.querySelector('.btn-choose').addEventListener('click', () => chooseOption(idx));
  return div;
}

async function generateThree() {
  if (!activeChat) {
    setStatus('Elegí primero un chat de la lista.', 'error');
    $('chat-select').focus();
    return;
  }
  const received = $('received-message').value.trim();
  if (!received) {
    setStatus('Pegá el mensaje que te llegó.', 'error');
    $('received-message').focus();
    return;
  }

  const context = buildContextFromInput(received, activeChat);
  if (context.length === 0) {
    setStatus('No pude leer el mensaje recibido.', 'error');
    return;
  }

  // Reset UI
  abortCtrls.forEach(c => c?.abort());
  abortCtrls = [];
  chosenIdx  = null;
  lastResults = null;

  $('output-card').hidden = false;
  const grid = $('options-grid');
  grid.innerHTML = '';
  const optionEls = [];
  for (let i = 0; i < NUM_OPTIONS; i++) {
    const el = buildOptionElement(i);
    grid.appendChild(el);
    optionEls.push(el);
  }

  setLoading(true);
  setStatus('Generando 3 opciones en paralelo…', '');

  const payloadBase = {
    chat_name:    activeChat.chat_name,
    is_group:     activeChat.is_group,
    participants: activeChat.participants,
    context,
    stream:       true,
    max_tokens:   parseInt(slider.value, 10),
  };

  // Seeds distintos para que las 3 sean diferentes
  const seeds = Array.from({ length: NUM_OPTIONS }, () => Math.floor(Math.random() * 2147483647));

  const results = await Promise.all(
    seeds.map((seed, i) => generateOption(i, payloadBase, seed, optionEls[i]))
  );

  lastResults = {
    options:          results.map(r => r ?? ''),
    seeds,
    received_message: received,
    chat:             { ...activeChat },
  };

  setLoading(false);
  setStatus('Listo. Elegí tu favorita ↓', 'ok');
  $('btn-regenerate').hidden = false;
}

// ── Captura de la elección ─────────────────────────────────────────────────
async function chooseOption(idx) {
  if (!lastResults || chosenIdx !== null) return;
  chosenIdx = idx;

  // Marcar visualmente
  document.querySelectorAll('.option-card').forEach((el, i) => {
    if (i === idx) el.classList.add('chosen');
    else           el.classList.add('unchosen');
    el.querySelector('.btn-choose').disabled = true;
  });
  document.querySelector(`.option-card.chosen .btn-choose`).textContent = '✓ Elegida';

  setStatus('Guardando tu elección…', '');

  try {
    const res = await fetch('/feedback', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_name:        lastResults.chat.chat_name,
        is_group:         lastResults.chat.is_group,
        participants:     lastResults.chat.participants,
        received_message: lastResults.received_message,
        options:          lastResults.options,
        chosen_idx:       idx,
        seeds:            lastResults.seeds,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    setStatus('✓ Guardada. Cuanta más feedback, mejor te imita el modelo.', 'ok');
  } catch (err) {
    setStatus(`Error al guardar: ${err.message}`, 'error');
  }
}

// ── Eventos ────────────────────────────────────────────────────────────────
$('btn-generate').addEventListener('click', generateThree);

$('btn-regenerate').addEventListener('click', generateThree);

$('btn-stop').addEventListener('click', () => {
  abortCtrls.forEach(c => c?.abort());
});

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    generateThree();
  }
});

// ── Utilidades ─────────────────────────────────────────────────────────────
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// ── Init ───────────────────────────────────────────────────────────────────
sliderVal.textContent = slider.value;

loadChats();

fetch('/health')
  .then(r => r.json())
  .then(data => {
    const pill = $('model-pill');
    if (data.status === 'ok') {
      const model = data.models.find(m => m.includes('memoria')) || data.models[0];
      pill.textContent = model;
      pill.className = 'status-pill ok';
    } else if (data.status === 'degraded') {
      pill.textContent = 'modelo no cargado';
      pill.className = 'status-pill error';
    } else {
      pill.textContent = 'sin conexión';
      pill.className = 'status-pill error';
    }
  })
  .catch(() => {
    const pill = $('model-pill');
    pill.textContent = 'sin conexión';
    pill.className = 'status-pill error';
  });
