'use strict';

const $ = id => document.getElementById(id);

let currentCtrl = null;
let isGroup     = false;
let lastPayload = null;   // para regenerar con otro seed

// ── Toggle 1:1 / Grupo ──────────────────────────────────────────────────────
document.querySelectorAll('.toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.toggle-btn').forEach(b => {
      b.classList.remove('active');
      b.setAttribute('aria-checked', 'false');
    });
    btn.classList.add('active');
    btn.setAttribute('aria-checked', 'true');
    isGroup = btn.dataset.value === 'true';
    $('participants-field').hidden = !isGroup;
  });
});

// ── Slider ──────────────────────────────────────────────────────────────────
const slider    = $('max-tokens');
const sliderVal = $('slider-val');
slider.addEventListener('input', () => { sliderVal.textContent = slider.value; });

// ── Context meter (cuenta líneas con formato Author: text) ──────────────────
const ctxArea = $('context');
const ctxMeter = $('context-meter');
ctxArea.addEventListener('input', updateContextMeter);
function updateContextMeter() {
  const lines = parseContext(ctxArea.value);
  ctxMeter.textContent = `${lines.length} mensaje${lines.length === 1 ? '' : 's'}`;
}

// ── Parse del contexto: cada línea "Author: text" → {author, text} ──────────
function parseContext(raw) {
  if (!raw) return [];
  const out = [];
  for (const rawLine of raw.split('\n')) {
    const line = rawLine.trim();
    if (!line) continue;
    const idx = line.indexOf(':');
    if (idx <= 0) continue;
    const author = line.slice(0, idx).trim();
    const text   = line.slice(idx + 1).trim();
    if (!author || !text) continue;
    out.push({ author, text });
  }
  return out;
}

// ── Status helpers ──────────────────────────────────────────────────────────
function setStatus(msg, cls = '') {
  const bar = $('status-bar');
  bar.textContent = msg;
  bar.className   = cls;
}

function setLoading(loading) {
  $('btn-generate').disabled   = loading;
  $('btn-regenerate').disabled = loading;
  $('btn-stop').hidden         = !loading;
}

function clearOutput() {
  $('output').textContent  = '';
  $('token-count').textContent = '';
}

function appendToken(token) {
  const out = $('output');
  out.textContent += token;
  out.scrollTop = out.scrollHeight;
}

function updateStats(evalCount, tps) {
  $('token-count').textContent = `${evalCount} tokens · ${tps} tok/s`;
}

// ── Generación ──────────────────────────────────────────────────────────────
async function generate(seed = null) {
  const chatName = $('chat-name').value.trim();
  const context  = parseContext(ctxArea.value);

  // Validaciones
  if (!chatName) {
    setStatus('Falta el nombre del chat.', 'error');
    $('chat-name').focus();
    return;
  }
  if (context.length === 0) {
    setStatus('Pegá al menos un mensaje en formato "Nombre: texto".', 'error');
    ctxArea.focus();
    return;
  }

  let participants = [];
  if (isGroup) {
    const raw = $('participants').value.trim();
    if (raw) {
      participants = raw.split(',').map(s => s.trim()).filter(Boolean);
    } else {
      // Si no se completó, derivamos de los autores del contexto
      participants = [...new Set(context.map(m => m.author))];
    }
  } else {
    // En 1:1 el participante es la otra persona; tomamos el primer autor distinto del usuario
    participants = [chatName];
  }

  // Mostrar la card de output recién al primer generate
  $('output-card').hidden = false;

  if (currentCtrl) currentCtrl.abort();
  currentCtrl = new AbortController();

  clearOutput();
  setLoading(true);
  setStatus('Conectando…', '');

  const body = {
    chat_name:    chatName,
    is_group:     isGroup,
    participants,
    context,
    stream:       true,
    max_tokens:   parseInt(slider.value, 10),
  };
  if (seed !== null) body.seed = seed;

  lastPayload = body;

  try {
    const res = await fetch('/generate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
      signal:  currentCtrl.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    setStatus('Generando…', '');
    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';
    let   done    = false;

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
          if (data.error) {
            setStatus(`Error: ${data.error}`, 'error');
            done = true;
            break;
          }
          if (data.token) appendToken(data.token);
          if (data.done) {
            updateStats(data.eval_count || 0, data.tokens_per_sec || 0);
            done = true;
            break;
          }
        } catch { /* ignorar JSON mal formado */ }
      }
    }

    setStatus('Listo ✓', 'ok');
  } catch (err) {
    if (err.name === 'AbortError') {
      setStatus('Generación cancelada', '');
    } else {
      setStatus(`Error: ${err.message}`, 'error');
    }
  } finally {
    setLoading(false);
    currentCtrl = null;
  }
}

// ── Eventos ─────────────────────────────────────────────────────────────────
$('btn-generate').addEventListener('click', () => generate());

$('btn-regenerate').addEventListener('click', () => {
  if (!lastPayload) return;
  generate(Math.floor(Math.random() * 2147483647));
});

$('btn-stop').addEventListener('click', () => {
  if (currentCtrl) currentCtrl.abort();
});

$('btn-copy').addEventListener('click', async () => {
  const text = $('output').textContent;
  if (!text) return;
  await navigator.clipboard.writeText(text);
  setStatus('Copiado al portapapeles ✓', 'ok');
  setTimeout(() => setStatus('Listo ✓', 'ok'), 1500);
});

// Cmd/Ctrl+Enter para generar desde cualquier campo
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    generate();
  }
});

// ── Init ────────────────────────────────────────────────────────────────────
sliderVal.textContent = slider.value;
updateContextMeter();

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
