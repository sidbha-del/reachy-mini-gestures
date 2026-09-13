'use strict';

const TARGET_WIDTH = 480;
const JPEG_QUALITY = 0.6;
const MAX_FPS = 24;
const ACK_TIMEOUT_MS = 1500;

const video = document.getElementById('video');
const dot = document.getElementById('dot');
const statusText = document.getElementById('status-text');
const statsEl = document.getElementById('stats');
const center = document.getElementById('center');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d', { alpha: false });

let facing = 'user';
let stream = null;
let ws = null;
let wsRetry = 500;
let awaitingAck = false;
let lastSend = 0;
let sentInWindow = 0;
let bytesInWindow = 0;
let wakeLock = null;

try { facing = localStorage.getItem('facing') || 'user'; } catch { /* storage unavailable */ }

function setStatus(text, live) {
  statusText.textContent = text;
  dot.classList.toggle('live', !!live);
}

function showCenter(icon, title, sub, buttonText, onClick) {
  document.getElementById('center-icon').textContent = icon;
  document.getElementById('center-title').textContent = title;
  document.getElementById('center-sub').textContent = sub;
  const btn = document.getElementById('center-btn');
  btn.hidden = !buttonText;
  btn.textContent = buttonText || '';
  btn.onclick = onClick || null;
  center.hidden = false;
}

async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showCenter('🔒', 'Secure link needed', 'Open this page with the https:// link or QR code shown on the computer.', null);
    return;
  }
  if (stream) stream.getTracks().forEach((t) => t.stop());
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facing }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false,
    });
    video.srcObject = stream;
    video.classList.toggle('mirror', facing === 'user');
    await video.play();
    center.hidden = true;
    requestWakeLock();
  } catch (e) {
    if (e.name === 'NotAllowedError') {
      showCenter('📷', 'Camera access is off', 'Allow camera access for this site in your browser settings, then try again.', 'Try again', startCamera);
    } else if (e.name === 'NotReadableError') {
      showCenter('📷', 'Camera is busy', 'Close other apps using the camera, then try again.', 'Try again', startCamera);
    } else {
      showCenter('📷', 'Camera did not start', e.message || String(e), 'Try again', startCamera);
    }
  }
}

function connect() {
  ws = new WebSocket(`wss://${location.host}/ws/phone`);
  ws.binaryType = 'arraybuffer';
  setStatus('Connecting to Reachy…', false);
  ws.onopen = () => {
    wsRetry = 500;
    awaitingAck = false;
    setStatus('Streaming to Reachy', true);
  };
  ws.onmessage = (e) => { if (e.data === 'ok') awaitingAck = false; };
  ws.onclose = () => {
    setStatus('Reconnecting…', false);
    setTimeout(connect, wsRetry);
    wsRetry = Math.min(wsRetry * 1.6, 5000);
  };
  ws.onerror = () => ws.close();
}

function pump() {
  requestAnimationFrame(pump);
  const now = performance.now();
  if (!ws || ws.readyState !== WebSocket.OPEN || document.hidden) return;
  if (video.readyState < 2 || !video.videoWidth) return;
  if (now - lastSend < 1000 / MAX_FPS) return;
  if (awaitingAck && now - lastSend < ACK_TIMEOUT_MS) return;

  canvas.width = TARGET_WIDTH;
  canvas.height = Math.round((video.videoHeight * TARGET_WIDTH) / video.videoWidth);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  awaitingAck = true;
  lastSend = now;
  canvas.toBlob(async (blob) => {
    if (!blob || !ws || ws.readyState !== WebSocket.OPEN) { awaitingAck = false; return; }
    ws.send(await blob.arrayBuffer());
    sentInWindow += 1;
    bytesInWindow += blob.size;
  }, 'image/jpeg', JPEG_QUALITY);
}

setInterval(() => {
  statsEl.textContent = sentInWindow ? `${sentInWindow} fps · ${Math.round(bytesInWindow / 1024)} KB/s` : '— fps';
  sentInWindow = 0;
  bytesInWindow = 0;
}, 1000);

async function requestWakeLock() {
  try {
    if ('wakeLock' in navigator && !document.hidden) wakeLock = await navigator.wakeLock.request('screen');
  } catch { /* not supported or denied; the phone may dim */ }
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    setStatus('Paused while in background', false);
  } else {
    requestWakeLock();
    if (ws && ws.readyState === WebSocket.OPEN) setStatus('Streaming to Reachy', true);
  }
});

document.getElementById('flip').addEventListener('click', () => {
  facing = facing === 'user' ? 'environment' : 'user';
  try { localStorage.setItem('facing', facing); } catch { /* storage unavailable */ }
  startCamera();
});

// Placement guide: shown while connecting and for a few seconds after the
// stream is live; a tap hides it at once.
const place = document.getElementById('place');
const tip = document.getElementById('tip');
tip.hidden = true;
let placeTimer = null;
function hidePlaceSoon(ms) {
  if (placeTimer) return;
  placeTimer = setTimeout(() => place.classList.add('gone'), ms);
}
place.addEventListener('click', () => place.classList.add('gone'));
setInterval(() => { if (dot.classList.contains('live')) hidePlaceSoon(8000); }, 500);

// A hand-held phone moves the whole picture, which looks like hand motion.
// Report how fast the phone is rotating so Reachy ignores motion gestures
// (wave, hands up, come close) while the phone itself is moving.
let motionDps = 0;
function onMotion(e) {
  const r = e.rotationRate;
  if (!r) return;
  const dps = Math.hypot(r.alpha || 0, r.beta || 0, r.gamma || 0);
  motionDps = Math.max(dps, motionDps * 0.85);
}
function enableMotion() {
  if (typeof DeviceMotionEvent === 'undefined') return;
  if (typeof DeviceMotionEvent.requestPermission === 'function') {
    DeviceMotionEvent.requestPermission().then((s) => {
      if (s === 'granted') window.addEventListener('devicemotion', onMotion);
    }).catch(() => {});
  } else {
    window.addEventListener('devicemotion', onMotion);
  }
}
enableMotion();
document.addEventListener('click', enableMotion, { once: true }); // iOS asks on a tap
setInterval(() => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ motion: Math.round(motionDps) }));
  motionDps *= 0.5;
}, 200);

startCamera();
connect();
requestAnimationFrame(pump);
