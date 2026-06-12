// ---------------------------------------------------------------------
// Subtle drifting-light canvas background (white, hint of violet)
// ---------------------------------------------------------------------
const canvas = document.getElementById('bg');
const ctx = canvas.getContext('2d');

function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
resize();
window.addEventListener('resize', resize);

const particles = Array.from({ length: 50 }, () => ({
  x: Math.random() * window.innerWidth,
  y: Math.random() * window.innerHeight,
  r: Math.random() * 1.4 + 0.3,
  vx: (Math.random() - 0.5) * 0.04,
  vy: (Math.random() - 0.5) * 0.04,
  violet: Math.random() < 0.35,
  phase: Math.random() * Math.PI * 2,
}));

function draw(t) {
  ctx.fillStyle = '#07050d';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (const p of particles) {
    p.x += p.vx;
    p.y += p.vy;
    if (p.x < 0) p.x = canvas.width;
    if (p.x > canvas.width) p.x = 0;
    if (p.y < 0) p.y = canvas.height;
    if (p.y > canvas.height) p.y = 0;
    const pulse = 0.5 + 0.5 * Math.sin(t * 0.0004 + p.phase);
    const color = p.violet
      ? `rgba(167,139,250,${0.18 * pulse})`
      : `rgba(255,255,255,${0.12 * pulse})`;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

// ---------------------------------------------------------------------
// Ambient sub-bass loop (uses the same BassEngine contemplation layer)
// ---------------------------------------------------------------------
const ambientAudio = document.getElementById('ambient-audio');
const ambientBtn = document.getElementById('ambient-btn');
ambientAudio.volume = 0.18;

ambientBtn.addEventListener('click', () => {
  if (ambientAudio.paused) {
    ambientAudio.play().catch(() => {});
    ambientBtn.classList.remove('muted');
  } else {
    ambientAudio.pause();
    ambientBtn.classList.add('muted');
  }
});
ambientBtn.classList.add('muted');

// ---------------------------------------------------------------------
// Mode switch (New Episode / Reply to Journal)
// ---------------------------------------------------------------------
let mode = 'new';
const modeButtons = document.querySelectorAll('.mode-btn');
const newOptions = document.getElementById('new-options');
const dropLabel = document.getElementById('drop-label');

modeButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    modeButtons.forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    mode = btn.dataset.mode;
    if (mode === 'reply') {
      newOptions.style.display = 'none';
      dropLabel.textContent = 'Drop a filled journal_ep*.md here, or click to choose';
    } else {
      newOptions.style.display = 'flex';
      dropLabel.textContent = 'Drop a document here, or click to choose a file';
    }
  });
});

// ---------------------------------------------------------------------
// File picker / drag & drop
// ---------------------------------------------------------------------
const drop = document.getElementById('drop');
const fileInput = document.getElementById('file-input');
const fileName = document.getElementById('file-name');
let selectedFile = null;

drop.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  selectedFile = fileInput.files[0] || null;
  fileName.textContent = selectedFile ? selectedFile.name : '';
});

['dragover', 'dragleave', 'drop'].forEach((evt) => {
  drop.addEventListener(evt, (e) => {
    e.preventDefault();
    drop.classList.toggle('drag', evt === 'dragover');
  });
});
drop.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files[0];
  if (f) {
    selectedFile = f;
    fileName.textContent = f.name;
  }
});

// ---------------------------------------------------------------------
// Sliders
// ---------------------------------------------------------------------
const duration = document.getElementById('duration');
const durationValue = document.getElementById('duration-value');
duration.addEventListener('input', () => {
  durationValue.textContent = `${duration.value} min`;
});

const answerSpace = document.getElementById('answer-space');
const answerSpaceValue = document.getElementById('answer-space-value');
answerSpace.addEventListener('input', () => {
  answerSpaceValue.textContent = `${answerSpace.value} s`;
});

// ---------------------------------------------------------------------
// Toggles (Arc Director / Memory)
// ---------------------------------------------------------------------
const toggleState = { 'use-arc': true, 'use-memory': true };
document.querySelectorAll('.toggle[data-target]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const key = btn.dataset.target;
    toggleState[key] = !toggleState[key];
    btn.classList.toggle('on', toggleState[key]);
  });
});

// ---------------------------------------------------------------------
// Generate / poll status
// ---------------------------------------------------------------------
const generateBtn = document.getElementById('generate');
const statusEl = document.getElementById('status');
const statusText = document.getElementById('status-text');
const logEl = document.getElementById('log');
const resultEl = document.getElementById('result');
const player = document.getElementById('player');
const downloadLink = document.getElementById('download');

let polling = null;

generateBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    alert('Choose a file first.');
    return;
  }

  generateBtn.disabled = true;
  statusEl.classList.remove('hidden');
  resultEl.classList.add('hidden');
  statusText.textContent = 'Starting…';
  logEl.textContent = '';

  const form = new FormData();
  form.append('file', selectedFile);
  form.append('mode', mode);
  form.append('duration', duration.value);
  form.append('answer_space', answerSpace.value);
  form.append('use_arc', toggleState['use-arc']);
  form.append('use_memory', toggleState['use-memory']);

  const res = await fetch('/api/generate', { method: 'POST', body: form });
  const { job_id } = await res.json();

  polling = setInterval(async () => {
    const r = await fetch(`/api/status/${job_id}`);
    const data = await r.json();
    logEl.textContent = data.log || '';
    logEl.scrollTop = logEl.scrollHeight;

    if (data.status === 'running') {
      statusText.textContent = 'Generating…';
    } else if (data.status === 'done') {
      clearInterval(polling);
      statusText.textContent = 'Done';
      generateBtn.disabled = false;
      if (data.result) {
        resultEl.classList.remove('hidden');
        const url = `/api/download/${job_id}`;
        player.src = url;
        downloadLink.href = url;
      }
    } else if (data.status === 'error') {
      clearInterval(polling);
      statusText.textContent = 'Error — see log';
      generateBtn.disabled = false;
    }
  }, 2500);
});
