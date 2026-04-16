'use strict';

const API_BASE     = '';
let   currentCtrl  = null;
let   tokenCount   = 0;

const $ = id => document.getElementById(id);

// ── Registro ────────────────────────────────────────────────────────────────
const registerBtns = document.querySelectorAll('.register-btn');
let   activeRegister = 'casual';

registerBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    registerBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeRegister = btn.dataset.register;
  });
});

// ── Slider ───────────────────────────────────────────────────────────────────
const slider    = $('max-tokens');
const sliderVal = $('slider-val');

slider.addEventListener('input', () => { sliderVal.textContent = slider.value; });

// ── Helpers ──────────────────────────────────────────────────────────────────
function setStatus(msg, cls = '') {
  const bar = $('status-bar');
  bar.textContent = msg;
  bar.className   = cls;
}

function setOutputPlaceholder() {
  const out = $('output');
  out.textContent = 'La respuesta aparece acá, token por token...';
  out.classList.add('empty');
}

function clearOutput() {
  const out = $('output');
  out.textContent = '';
  out.classList.remove('empty');
  tokenCount = 0;
  $('token-count').textContent = `0 / ${slider.value}`;
}

function appendToken(token) {
  const out = $('output');
  if (out.classList.contains('empty')) {
    out.textContent = '';
    out.classList.remove('empty');
  }
  out.textContent += token;
  tokenCount++;
  $('token-count').textContent = `${tokenCount} / ${slider.value}`;
  // Auto-scroll
  out.scrollTop = out.scrollHeight;
}

function setLoading(loading) {
  $('btn-generate').disabled    = loading;
  $('btn-regenerate').disabled  = loading;
  $('btn-stop').style.display   = loading ? 'inline-block' : 'none';
}

// ── Generar ──────────────────────────────────────────────────────────────────
async function generate(prompt) {
  if (!prompt.trim()) {
    setStatus('Escribí un prompt antes de generar.', 'error');
    return;
  }

  if (currentCtrl) currentCtrl.abort();
  currentCtrl = new AbortController();

  clearOutput();
  setLoading(true);
  setStatus('Conectando...', '');

  const body = JSON.stringify({
    prompt,
    register:   activeRegister,
    stream:     true,
    max_tokens: parseInt(slider.value, 10),
  });

  try {
    const res = await fetch(`${API_BASE}/generate`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      signal:  currentCtrl.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    setStatus('Generando...', '');
    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();                        // línea incompleta

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') break;

        try {
          const data = JSON.parse(payload);
          if (data.error) { setStatus(`Error: ${data.error}`, 'error'); return; }
          if (data.token) appendToken(data.token);
        } catch { /* ignorar JSON mal formado */ }
      }
    }

    setStatus('✓ Listo', 'ok');
  } catch (err) {
    if (err.name === 'AbortError') {
      setStatus('Generación cancelada.', '');
    } else {
      setStatus(`Error: ${err.message}`, 'error');
    }
  } finally {
    setLoading(false);
    currentCtrl = null;
  }
}

// ── Eventos ──────────────────────────────────────────────────────────────────
$('btn-generate').addEventListener('click', () => {
  generate($('prompt').value);
});

$('btn-regenerate').addEventListener('click', () => {
  const prompt = $('prompt').value;
  if (prompt.trim()) generate(prompt);
});

$('btn-stop').addEventListener('click', () => {
  if (currentCtrl) currentCtrl.abort();
});

$('btn-copy').addEventListener('click', async () => {
  const text = $('output').textContent;
  if (!text || $('output').classList.contains('empty')) return;
  await navigator.clipboard.writeText(text);
  setStatus('Copiado al portapapeles ✓', 'ok');
  setTimeout(() => setStatus('', ''), 2000);
});

// Enter en textarea (Ctrl+Enter o Cmd+Enter para generar)
$('prompt').addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    generate($('prompt').value);
  }
});

// ── Init ─────────────────────────────────────────────────────────────────────
setOutputPlaceholder();
sliderVal.textContent = slider.value;

// Health check al cargar
fetch(`${API_BASE}/health`)
  .then(r => r.json())
  .then(data => {
    if (data.status === 'ok') {
      const model = data.models.find(m => m.includes('memoria')) || data.models[0];
      setStatus(`Modelo: ${model || 'memoria'} ✓`, 'ok');
    } else {
      setStatus('⚠ Ollama no disponible — asegurate de que el modelo esté cargado', 'error');
    }
  })
  .catch(() => setStatus('⚠ No se puede conectar al backend', 'error'));
