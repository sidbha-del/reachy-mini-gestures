'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const TIPS = new Set([4, 8, 12, 16, 20]);
const SOURCES = ['robot', 'webcam', 'phone'];
const SOUND_OUTPUTS = ['auto', 'robot', 'computer'];
const SOURCE_NAME = { robot: 'Robot', webcam: 'Laptop', phone: 'Phone' };
const SOURCE_LONG = { robot: 'robot camera', webcam: 'laptop camera', phone: 'phone camera' };
const MODE_TEXT = { head_follow: 'Following your finger', antennas: 'Antennas and body follow your hands', sleeping: 'Sleeping' };
const KIND_LABEL = { continuous: 'Live', pose: '', motion: 'Move' };
const RING_LEN = 119.4;

const app = {
  gestures: [], byKey: {}, byShape: {}, sounds: [],
  settings: { sensitivity: 0.5, zone_bottom: 0.12, mirror: true, sound: true, sound_output: 'auto', free_usb: false },
  packet: null, wsOpen: false, everConnected: false,
  source: 'robot', sourceClickAt: 0,
  lastFireId: undefined, pillHideAt: 0,
  sheet: null, teachKey: null, teachStep: 0, teachBusy: false, countdownTimer: null, phoneCloseTimer: null,
  frameW: 16, frameH: 10, stopped: false, zonePreviewUntil: 0,
  dismissedTips: loadDismissed(),
};

/* ---------------- utilities ---------------- */
async function api(path, { method = 'GET', body } = {}) {
  const res = await fetch(path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = {};
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function loadDismissed() {
  try { return JSON.parse(localStorage.getItem('dismissedTips') || '{}'); } catch { return {}; }
}
function saveDismissed() {
  try { localStorage.setItem('dismissedTips', JSON.stringify(app.dismissedTips)); } catch { /* storage unavailable */ }
}

function setSeg(el, values, value) {
  el.style.setProperty('--i', Math.max(0, values.indexOf(value)));
  $$('button', el).forEach((b) => b.setAttribute('aria-selected', String((b.dataset.source || b.dataset.value) === value)));
}

// Icons: "2x☝️" means a pair of hands. The emoji is drawn as a left hand, so the
// second glyph is mirrored to read as the right hand next to it.
function setGlyph(el, s) {
  el.textContent = '';
  if (!s || !s.startsWith('2x')) { el.textContent = s || ''; return; }
  const glyph = s.slice(2);
  const left = document.createElement('span');
  left.textContent = glyph;
  const right = document.createElement('span');
  right.className = 'glyph-left';
  right.textContent = glyph;
  el.append(left, right);
}

let toastTimer = null;
function toast(emoji, title, sub = '') {
  setGlyph($('#toast-emoji'), emoji);
  $('#toast-title').textContent = title;
  $('#toast-sub').textContent = sub;
  const el = $('#toast');
  el.classList.remove('show');
  void el.offsetWidth;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 1700);
}

function confirmDialog({ title, message, ok = 'OK', destructive = false }) {
  return new Promise((resolve) => {
    $('#dialog-title').textContent = title;
    $('#dialog-msg').textContent = message;
    const okBtn = $('#dialog-ok');
    okBtn.textContent = ok;
    okBtn.className = destructive ? 'btn btn-danger' : 'btn btn-primary';
    const wrap = $('#dialog');
    wrap.hidden = false;
    const done = (value) => { wrap.hidden = true; okBtn.onclick = null; $('#dialog-cancel').onclick = null; resolve(value); };
    okBtn.onclick = () => done(true);
    $('#dialog-cancel').onclick = () => done(false);
    okBtn.focus();
  });
}

/* ---------------- sound on this computer ---------------- */
const audio = { ctx: null, buffers: new Map(), loading: null };

function audioReady() {
  return !!audio.ctx && audio.ctx.state === 'running';
}

function unlockAudio() {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return;
  if (!audio.ctx) audio.ctx = new AC();
  if (audio.ctx.state === 'suspended') audio.ctx.resume().then(loadSounds).catch(() => {});
  else loadSounds();
}

function loadSounds() {
  if (!audio.ctx || audio.loading || !app.sounds.length) return;
  audio.loading = Promise.all(app.sounds.map(async (file) => {
    try {
      const res = await fetch(`/api/sound/${encodeURIComponent(file)}`);
      audio.buffers.set(file, await audio.ctx.decodeAudioData(await res.arrayBuffer()));
    } catch { /* one missing sound shouldn't block the rest */ }
  })).finally(() => { audio.loading = null; });
}

function playSound(file) {
  if (!audioReady()) return;
  const buffer = audio.buffers.get(file);
  if (!buffer) { loadSounds(); return; }
  const src = audio.ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(audio.ctx.destination);
  src.start();
}

/* ---------------- init ---------------- */
async function init() {
  bindUI();
  try {
    const s = await api('/api/state');
    app.gestures = s.gestures;
    app.sounds = s.sounds || [];
    for (const g of s.gestures) {
      app.byKey[g.key] = g;
      if (g.teach) app.byShape[g.teach] = g;
    }
    app.settings = { ...app.settings, ...s.settings };
    $('#phone-url').textContent = s.phoneUrl;
    renderGrid();
    setSegment(s.source);
    applySettingsUI();
    renderTeachChips(s.teachCounts);
  } catch (e) {
    toast('⚠️', 'Could not load the app', e.message);
  }
  connectWS();
  startVideo();
  requestAnimationFrame(frameLoop);
}

function bindUI() {
  $$('#source-seg button').forEach((b) => b.addEventListener('click', () => chooseSource(b.dataset.source)));
  $$('#seg-sound button').forEach((b) => b.addEventListener('click', () => {
    saveSettings({ sound_output: b.dataset.value });
    if (b.dataset.value !== 'robot') unlockAudio();
  }));
  $('#btn-settings').addEventListener('click', () => openSheet('settings'));
  $('#btn-teach').addEventListener('click', () => openSheet('teach'));
  $$('.sheet-close').forEach((b) => b.addEventListener('click', closeSheets));
  $('#scrim').addEventListener('click', closeSheets);
  $('#btn-wake').addEventListener('click', () => robotAction('wake'));
  $('#btn-sleep').addEventListener('click', () => robotAction('sleep'));
  $('#btn-stop').addEventListener('click', stopApp);
  $('#hint-close').addEventListener('click', dismissHint);
  $('#copy-url').addEventListener('click', copyPhoneUrl);

  $('#set-sensitivity').addEventListener('input', (e) => saveSettings({ sensitivity: parseFloat(e.target.value) }));
  $('#set-zone').addEventListener('input', (e) => {
    saveSettings({ zone_bottom: parseFloat(e.target.value) });
    app.zonePreviewUntil = performance.now() + 1500;
  });
  $('#set-mirror').addEventListener('change', (e) => saveSettings({ mirror: e.target.checked }));
  $('#set-sound').addEventListener('change', (e) => saveSettings({ sound: e.target.checked }));
  $('#set-free-usb').addEventListener('change', (e) => saveSettings({ free_usb: e.target.checked }));

  $('#teach-capture').addEventListener('click', teachCapture);
  $('#teach-timer').addEventListener('click', teachCountdown);
  $('#teach-save').addEventListener('click', teachSave);
  $('#teach-discard').addEventListener('click', teachDiscard);
  $('#teach-delete').addEventListener('click', teachDelete);
  $('#teach-reset').addEventListener('click', teachReset);

  const video = $('#video');
  video.addEventListener('load', () => video.classList.add('ready'));
  video.addEventListener('error', () => setTimeout(startVideo, 1000));

  ['pointerdown', 'keydown'].forEach((type) => window.addEventListener(type, unlockAudio, { passive: true }));

  document.addEventListener('keydown', (e) => {
    if (!$('#dialog').hidden) return;
    if (e.key === 'Escape') closeSheets();
    if (e.code === 'Space' && app.sheet === 'teach' && !$('#teach-live').hidden && e.target === document.body) {
      e.preventDefault();
      teachCapture();
    }
  });
  new ResizeObserver(sizeCanvas).observe($('#stage'));
}

function startVideo() {
  const video = $('#video');
  video.classList.remove('ready');
  video.src = '/video?t=' + Date.now();
}

/* ---------------- websocket ---------------- */
let ws = null;
let wsRetry = 400;
function connectWS() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/ui`);
  ws.onopen = () => {
    if (app.everConnected) startVideo();
    app.wsOpen = true;
    app.everConnected = true;
    wsRetry = 400;
  };
  ws.onmessage = (ev) => onPacket(JSON.parse(ev.data));
  ws.onclose = () => {
    app.wsOpen = false;
    if (app.stopped) return;
    renderEmpty();
    setTimeout(connectWS, wsRetry);
    wsRetry = Math.min(wsRetry * 1.6, 4000);
  };
  ws.onerror = () => ws.close();
}

function onPacket(p) {
  app.packet = p;
  if (p.w && p.h) { app.frameW = p.w; app.frameH = p.h; }
  updateTracks(p);
  renderStatus(p);
  renderActive(p);
  handleFire(p.fire);
  renderHint([...clientTips(p), ...(p.tips || [])]);
  if (app.sheet === 'teach') renderTeachLive(p);
  if (app.sheet === 'phone') renderPhoneSheet(p);
}

function clientTips(p) {
  const tips = [];
  if (p.soundTarget === 'computer' && !audioReady()) {
    tips.push({
      id: 'audio_locked',
      text: "Click anywhere on this page to let Reachy's sounds play through this computer's speakers.",
      actions: [],
    });
  }
  return tips;
}

/* ---------------- status ---------------- */
function renderStatus(p) {
  const r = p.robot;
  let text = 'Robot offline · retrying';
  let cls = 'bad';
  if (p.daemon && !p.daemon.reachable) {
    text = 'Robot service not running';
  } else if (r.state === 'connected') {
    text = r.sleeping ? 'Sleeping' : 'Connected';
    cls = r.sleeping ? 'idle' : 'ok';
  } else if (r.state === 'connecting') {
    text = 'Connecting to robot…';
    cls = 'warn';
  }
  if (p.profile && r.state === 'connected') text += ` · ${p.profile.name}`;
  $('#robot-text').textContent = text;
  $('#robot-dot').className = 'status-dot ' + cls;
  const connected = r.state === 'connected';
  $('#btn-wake').disabled = !connected;
  $('#btn-sleep').disabled = !connected || r.sleeping;

  if (performance.now() - app.sourceClickAt > 1200 && p.camera.source !== app.source) setSegment(p.camera.source);

  const live = p.camera.state === 'live';
  $('#hud-source').textContent = SOURCE_NAME[p.camera.source];
  $('#hud-fps').textContent = live ? `${Math.round(p.camera.fps || p.fps)} fps` : '— fps';
  const lat = p.lat ? (p.lat.motor ?? p.lat.vision) : null;
  $('#hud-lat').textContent = live && lat != null ? `${lat} ms` : '— ms';
  $('#hud-dot').className = 'status-dot ' + (live ? 'ok' : 'warn');

  const engaged = p.hands.filter((h) => h.engaged).length;
  const ignored = p.hands.length - engaged;
  $('#hands-meta').textContent = !live ? '' : engaged === 0 && ignored === 0 ? 'No hands'
    : `${engaged} hand${engaged === 1 ? '' : 's'}${ignored ? ` · ${ignored} ignored` : ''}`;

  const meta = [];
  if (r.fired) meta.push(`${r.fired} reactions`);
  if (r.dropped) meta.push(`${r.dropped} skipped as too late`);
  if (p.soundTarget === 'computer' && app.settings.sound) meta.push('sounds on this computer');
  $('#controls-meta').textContent = meta.join(' · ');

  if (p.profile) {
    $('#row-free-usb').hidden = !p.profile.can_release_media;
    $('#robot-cam-note').textContent = p.profile.camera_note;
    const bits = [`Robot: ${p.profile.name}`];
    if (p.daemon && p.daemon.mediaReleased) bits.push('robot camera paused');
    if (p.soundTarget) bits.push(`sounds on ${p.soundTarget === 'robot' ? 'Reachy' : 'this computer'}`);
    $('#about-profile').textContent = bits.join(' · ');
  }
  renderEmpty();
}

function renderEmpty() {
  const p = app.packet;
  let title = null, sub = '', action = null, spin = true;
  if (app.stopped) {
    title = 'App stopped'; sub = 'Run manage.ps1 start to open it again.'; spin = false;
  } else if (!app.wsOpen) {
    title = app.everConnected ? 'Reconnecting…' : 'Starting…';
    sub = app.everConnected ? 'The app on this computer is restarting or busy.' : '';
  } else if (p) {
    const cam = p.camera;
    if (p.visionError && cam.state !== 'live') {
      title = 'Hand tracking stopped'; sub = p.visionError; spin = false;
    } else if (cam.state === 'waiting_for_phone') {
      title = 'Waiting for your phone'; sub = 'Open the camera page on your phone to start streaming.';
      action = { label: 'Show QR code', fn: () => openSheet('phone') }; spin = false;
    } else if (cam.state === 'error') {
      title = `The ${SOURCE_LONG[cam.source]} isn't available`; sub = cam.error || ''; spin = false;
      action = cam.source === 'robot'
        ? { label: 'Use laptop camera', fn: () => chooseSource('webcam') }
        : { label: 'Use robot camera', fn: () => chooseSource('robot') };
    } else if (cam.state !== 'live') {
      const robotDown = cam.source === 'robot' && p.robot.state !== 'connected';
      title = robotDown ? 'Waiting for the robot…' : `Starting ${SOURCE_LONG[cam.source]}…`;
      if (cam.source === 'robot' && (p.robot.state === 'error' || (p.daemon && !p.daemon.reachable))) {
        sub = 'Check that Reachy is powered on and plugged in. Meanwhile you can play with another camera.';
        action = { label: 'Use laptop camera', fn: () => chooseSource('webcam') };
      }
    }
  }
  const el = $('#empty');
  if (!title) { el.hidden = true; return; }
  el.hidden = false;
  $('#empty-title').textContent = title;
  $('#empty-sub').textContent = sub;
  $('#empty-spinner').hidden = !spin;
  const btn = $('#empty-action');
  btn.hidden = !action;
  if (action) { btn.textContent = action.label; btn.onclick = action.fn; }
}

/* ---------------- hints ---------------- */
let currentTip = null;
function renderHint(tips) {
  const now = Date.now();
  const tip = tips.find((t) => !(app.dismissedTips[t.id] > now));
  const el = $('#hint');
  if (!tip) { el.hidden = true; currentTip = null; return; }
  if (currentTip && currentTip.id === tip.id && currentTip.text === tip.text && !el.hidden) return;
  currentTip = tip;
  $('#hint-text').textContent = tip.text;
  const actions = $('#hint-actions');
  actions.innerHTML = '';
  for (const a of tip.actions || []) {
    if (a.source && a.source === app.source) continue;
    const b = document.createElement('button');
    b.textContent = a.label;
    b.onclick = () => {
      if (a.source) chooseSource(a.source);
      if (a.sheet) openSheet(a.sheet);
      if (a.robot) robotAction(a.robot);
      if (a.setting) {
        saveSettings(a.setting);
        if (a.setting.sound_output === 'computer') unlockAudio();
        toast('✓', 'Done', a.label);
      }
      el.hidden = true;
      currentTip = null;
    };
    actions.appendChild(b);
  }
  el.hidden = false;
}

function dismissHint() {
  if (currentTip) {
    app.dismissedTips[currentTip.id] = Date.now() + 30 * 60 * 1000;
    saveDismissed();
  }
  $('#hint').hidden = true;
}

/* ---------------- gestures ---------------- */
function renderGrid() {
  const grid = $('#grid');
  grid.innerHTML = '';
  for (const g of app.gestures.filter((x) => !x.hidden)) {
    const card = document.createElement('div');
    card.className = 'card';
    card.dataset.key = g.key;
    card.title = g.hint;
    card.innerHTML = '<div class="card-emoji"></div><div class="card-title"><span class="card-name"></span><span class="card-kind"></span></div><div class="card-reaction"></div>';
    card.querySelector('.card-kind').textContent = g.hands === 2 ? '2 hands' : (KIND_LABEL[g.kind] || '');
    card.classList.toggle('two-hands', g.hands === 2);
    const icon = card.querySelector('.card-emoji');
    // Steps are joined by arrows (do this, then that); an "or" step shows
    // alternatives instead, e.g. 🤟 or 🫰.
    const steps = g.steps || [g.emoji];
    steps.forEach((step, i) => {
      if (step === 'or') {
        const or = document.createElement('span');
        or.className = 'step-arrow';
        or.textContent = 'or';
        icon.appendChild(or);
        return;
      }
      if (i && steps[i - 1] !== 'or') {
        const arrow = document.createElement('span');
        arrow.className = 'step-arrow';
        arrow.textContent = '→';
        icon.appendChild(arrow);
      }
      const s = document.createElement('span');
      setGlyph(s, step);
      icon.appendChild(s);
    });
    card.querySelector('.card-name').textContent = g.name;
    card.querySelector('.card-reaction').textContent = g.reaction;
    grid.appendChild(card);
  }
}

function renderActive(p) {
  const key = p.active;
  $$('#grid .card').forEach((c) => c.classList.toggle('active', c.dataset.key === key));
  const pill = $('#pill');
  const now = performance.now();
  if (key && app.byKey[key]) {
    const g = app.byKey[key];
    setGlyph($('#pill-emoji'), g.emoji);
    $('#pill-name').textContent = g.name;
    $('#pill-sub').textContent = MODE_TEXT[p.mode] || g.reaction;
    const conf = Math.max(0, ...p.hands.filter((h) => h.engaged).map((h) => h.conf));
    $('#ring-fill').style.strokeDashoffset = String(RING_LEN * (1 - Math.min(1, conf)));
    pill.classList.add('show');
    app.pillHideAt = now + 450;
  } else if (now > app.pillHideAt) {
    pill.classList.remove('show');
  }
}

function handleFire(fire) {
  if (app.lastFireId === undefined) { app.lastFireId = fire ? fire.id : 0; return; }
  if (!fire || fire.id === app.lastFireId) return;
  app.lastFireId = fire.id;
  const g = app.byKey[fire.name];
  if (!g) return;
  if (fire.sound) playSound(fire.sound);
  toast(g.emoji, g.name, fire.ok ? g.reaction : 'Robot not connected');
  const card = $(`#grid .card[data-key="${fire.name}"]`);
  if (card) {
    card.classList.remove('flash');
    void card.offsetWidth;
    card.classList.add('flash');
  }
}

/* ---------------- camera source ---------------- */
function setSegment(source) {
  app.source = source;
  setSeg($('#source-seg'), SOURCES, source);
}

async function chooseSource(source) {
  const prev = app.source;
  app.sourceClickAt = performance.now();
  setSegment(source);
  try {
    await api('/api/source', { method: 'POST', body: { source } });
    if (source === 'phone' && !(app.packet && app.packet.camera.phone_connected)) openSheet('phone');
  } catch (e) {
    setSegment(prev);
    toast('⚠️', 'Could not switch camera', e.message);
  }
}

function renderPhoneSheet(p) {
  const connected = p.camera.phone_connected;
  $('#phone-status').classList.toggle('ok', connected);
  $('#phone-status-text').textContent = connected ? 'Phone connected' : 'Waiting for your phone…';
  if (connected && !app.phoneCloseTimer) {
    app.phoneCloseTimer = setTimeout(() => {
      app.phoneCloseTimer = null;
      if (app.sheet === 'phone') closeSheets();
      if (app.source !== 'phone') chooseSource('phone');
      toast('📱', 'Phone camera connected', 'Streaming to Reachy');
    }, 1100);
  }
}

async function copyPhoneUrl() {
  try {
    await navigator.clipboard.writeText($('#phone-url').textContent);
    $('#copy-url').textContent = 'Copied';
    setTimeout(() => { $('#copy-url').textContent = 'Copy'; }, 1400);
  } catch { /* clipboard not permitted */ }
}

/* ---------------- robot actions ---------------- */
async function robotAction(action) {
  try {
    await api(`/api/robot/${action}`, { method: 'POST' });
    if (action === 'restart') toast('🔄', 'Reconnecting to Reachy', 'This takes a few seconds');
  } catch (e) {
    toast('⚠️', 'Robot command failed', e.message);
  }
}

async function stopApp() {
  const ok = await confirmDialog({
    title: 'Stop the app?',
    message: 'Reachy will go to sleep and this page will disconnect.',
    ok: 'Stop', destructive: true,
  });
  if (!ok) return;
  app.stopped = true;
  try { await api('/api/robot/stop', { method: 'POST' }); } catch { /* server is shutting down */ }
  renderEmpty();
}

/* ---------------- sheets ---------------- */
function openSheet(name) {
  closeSheets();
  const el = $(`#sheet-${name}`);
  el.classList.add('open');
  el.setAttribute('aria-hidden', 'false');
  $('#scrim').classList.add('show');
  app.sheet = name;
  if (name === 'teach') onTeachOpen();
}

function closeSheets() {
  if (app.sheet === 'teach') onTeachClose();
  $$('.sheet.open').forEach((el) => { el.classList.remove('open'); el.setAttribute('aria-hidden', 'true'); });
  $('#scrim').classList.remove('show');
  app.sheet = null;
}

/* ---------------- settings ---------------- */
function applySettingsUI() {
  const s = app.settings;
  $('#set-sensitivity').value = s.sensitivity;
  $('#set-zone').value = s.zone_bottom;
  $('#set-mirror').checked = !!s.mirror;
  $('#set-sound').checked = !!s.sound;
  $('#set-free-usb').checked = !!s.free_usb;
  setSeg($('#seg-sound'), SOUND_OUTPUTS, s.sound_output);
  $('#row-sound-output').style.opacity = s.sound ? '1' : '0.45';
  $('#zone-value').textContent = `${Math.round(s.zone_bottom * 100)}%`;
  $('#video').classList.toggle('mirror', !!s.mirror);
}

let settingsTimer = null;
function saveSettings(partial) {
  Object.assign(app.settings, partial);
  applySettingsUI();
  clearTimeout(settingsTimer);
  settingsTimer = setTimeout(async () => {
    try {
      app.settings = await api('/api/settings', { method: 'POST', body: app.settings });
      applySettingsUI();
    } catch (e) {
      toast('⚠️', 'Settings not saved', e.message);
    }
  }, 250);
}

/* ---------------- overlay drawing ---------------- */
const view = new Map();

function updateTracks(p) {
  const seen = new Set();
  for (const h of p.hands) {
    seen.add(h.id);
    let v = view.get(h.id);
    if (!v) {
      v = { cur: h.pts.map((q) => [q[0], q[1]]), alpha: 0 };
      view.set(h.id, v);
    }
    v.target = h.pts;
    v.engaged = h.engaged;
    v.reject = h.reject;
    v.gone = false;
  }
  for (const [id, v] of view) if (!seen.has(id)) v.gone = true;
}

function sizeCanvas() {
  const canvas = $('#overlay');
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
}

function contentRect(canvas) {
  const s = Math.min(canvas.width / app.frameW, canvas.height / app.frameH);
  const w = app.frameW * s, h = app.frameH * s;
  return { x: (canvas.width - w) / 2, y: (canvas.height - h) / 2, w, h };
}

let lastTs = 0;
function frameLoop(ts) {
  const dt = Math.min(0.1, lastTs ? (ts - lastTs) / 1000 : 0.016);
  lastTs = ts;
  draw(dt);
  requestAnimationFrame(frameLoop);
}

function draw(dt) {
  const canvas = $('#overlay');
  sizeCanvas();
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const p = app.packet;
  if (!p || p.camera.state !== 'live') { view.clear(); return; }
  const dpr = window.devicePixelRatio || 1;
  const r = contentRect(canvas);
  const mirror = !!app.settings.mirror;
  const X = (x) => r.x + (mirror ? 1 - x : x) * r.w;
  const Y = (y) => r.y + y * r.h;

  drawZone(ctx, r, dpr);

  const follow = 1 - Math.exp(-dt / 0.04);
  const fade = 1 - Math.exp(-dt / 0.09);
  for (const [id, v] of view) {
    v.alpha += ((v.gone ? 0 : 1) - v.alpha) * fade;
    if (v.gone && v.alpha < 0.02) { view.delete(id); continue; }
    for (let i = 0; i < 21; i++) {
      v.cur[i][0] += (v.target[i][0] - v.cur[i][0]) * follow;
      v.cur[i][1] += (v.target[i][1] - v.cur[i][1]) * follow;
    }
    drawHand(ctx, v, X, Y, dpr);
  }
}

function drawZone(ctx, r, dpr) {
  const zone = app.settings.zone_bottom;
  if (!zone) return;
  const preview = performance.now() < app.zonePreviewUntil;
  const y0 = r.y + (1 - zone) * r.h;
  const grad = ctx.createLinearGradient(0, y0, 0, r.y + r.h);
  grad.addColorStop(0, 'rgba(0,0,0,0)');
  grad.addColorStop(1, `rgba(0,0,0,${preview ? 0.55 : 0.28})`);
  ctx.fillStyle = grad;
  ctx.fillRect(r.x, y0, r.w, r.y + r.h - y0);
  ctx.save();
  ctx.setLineDash([5 * dpr, 7 * dpr]);
  ctx.strokeStyle = `rgba(255,255,255,${preview ? 0.5 : 0.14})`;
  ctx.lineWidth = dpr;
  ctx.beginPath();
  ctx.moveTo(r.x + 16 * dpr, y0);
  ctx.lineTo(r.x + r.w - 16 * dpr, y0);
  ctx.stroke();
  ctx.restore();
  if (preview) {
    ctx.font = `600 ${12 * dpr}px ${getComputedStyle(document.body).fontFamily}`;
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.textAlign = 'center';
    ctx.fillText('Hands below this line are ignored', r.x + r.w / 2, y0 + 20 * dpr);
  }
}

function drawHand(ctx, v, X, Y, dpr) {
  const pts = v.cur.map((q) => [X(q[0]), Y(q[1])]);
  const size = Math.hypot(pts[0][0] - pts[9][0], pts[0][1] - pts[9][1]);
  const lw = Math.max(2 * dpr, Math.min(7 * dpr, size * 0.075));
  ctx.save();
  ctx.globalAlpha = v.alpha * (v.engaged ? 1 : 0.5);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  if (v.engaged) {
    ctx.shadowColor = 'rgba(10,132,255,0.9)';
    ctx.shadowBlur = 16 * dpr;
  }
  ctx.strokeStyle = v.engaged ? 'rgba(255,255,255,0.95)' : 'rgba(174,174,178,0.75)';
  ctx.lineWidth = lw;
  ctx.beginPath();
  for (const [a, b] of BONES) {
    ctx.moveTo(pts[a][0], pts[a][1]);
    ctx.lineTo(pts[b][0], pts[b][1]);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
  for (let i = 0; i < 21; i++) {
    ctx.beginPath();
    ctx.arc(pts[i][0], pts[i][1], TIPS.has(i) ? lw * 0.95 : lw * 0.55, 0, Math.PI * 2);
    ctx.fillStyle = v.engaged ? (TIPS.has(i) ? '#0a84ff' : '#ffffff') : 'rgba(174,174,178,0.9)';
    ctx.fill();
  }
  if (!v.engaged && v.reject) {
    const label = `Ignored · ${v.reject}`;
    ctx.globalAlpha = v.alpha;
    ctx.font = `600 ${11 * dpr}px ${getComputedStyle(document.body).fontFamily}`;
    const tw = ctx.measureText(label).width;
    const bx = pts[0][0] - tw / 2 - 8 * dpr;
    const by = pts[0][1] + 10 * dpr;
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.beginPath();
    ctx.roundRect(bx, by, tw + 16 * dpr, 20 * dpr, 10 * dpr);
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, pts[0][0], by + 10 * dpr);
  }
  ctx.restore();
}

/* ---------------- teach ---------------- */
function renderTeachChips(counts = {}) {
  const wrap = $('#teach-chips');
  wrap.innerHTML = '';
  for (const g of app.gestures.filter((x) => x.train && x.train.length)) {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.dataset.key = g.key;
    chip.setAttribute('aria-selected', String(g.key === app.teachKey));
    chip.innerHTML = '<span class="chip-emoji"></span><span class="chip-name"></span><span class="badge"></span>';
    setGlyph(chip.querySelector('.chip-emoji'), g.emoji);
    chip.querySelector('.chip-name').textContent = g.name;
    // A gesture is only as trained as its least-trained step.
    chip.querySelector('.badge').textContent = Math.min(...g.train.map((s) => counts[s] || 0));
    chip.addEventListener('click', () => selectTeach(g.key));
    wrap.appendChild(chip);
  }
}

async function refreshTeachCounts() {
  try { renderTeachChips(await api('/api/teach/counts')); } catch { /* keep old counts */ }
}

const teachGesture = () => app.byKey[app.teachKey];
const teachShapeNow = () => (teachGesture() ? teachGesture().train[app.teachStep] : null);

function selectTeach(key, step = 0) {
  app.teachKey = key;
  app.teachStep = step;
  $$('#teach-chips .chip').forEach((c) => c.setAttribute('aria-selected', String(c.dataset.key === key)));
  const g = teachGesture();
  const shape = app.byShape[teachShapeNow()];
  const two = g.hands === 2;
  const steps = g.train.length;
  setGlyph($('#teach-emoji'), two ? `2x${shape.emoji}` : shape.emoji);
  $('#teach-name').textContent = steps > 1 ? `Show “${g.name}” · step ${step + 1} of ${steps}` : `Show “${g.name}”`;
  $('#teach-hint').textContent = two || steps > 1
    ? `${two ? 'Both hands: ' : ''}${shape.name.toLowerCase()} — ${shape.hint.charAt(0).toLowerCase()}${shape.hint.slice(1)}`
    : g.hint;
  $('#teach-status').textContent = '';
  showTeachLive();
}

function showTeachLive() {
  $('#teach-live').hidden = false;
  $('#teach-review').hidden = true;
}

async function onTeachOpen() {
  if (!app.teachKey) {
    const first = app.gestures.find((g) => g.train && g.train.length);
    if (first) app.teachKey = first.key;
  }
  await refreshTeachCounts();
  if (app.teachKey) selectTeach(app.teachKey, app.teachStep);
}

function onTeachClose() {
  cancelCountdown();
  if (!$('#teach-review').hidden) api('/api/teach/discard', { method: 'POST' }).catch(() => {});
}

function renderTeachLive(p) {
  if ($('#teach-live').hidden) return;
  const hand = p.hands.find((h) => h.engaged) || p.hands[0];
  const fill = $('#teach-meter');
  if (!hand) {
    $('#teach-seeing').textContent = 'No hand in view';
    $('#teach-pct').textContent = '';
    fill.style.width = '0%';
    fill.classList.remove('match');
    return;
  }
  const g = hand.live ? app.byShape[hand.live] : null;
  $('#teach-seeing').textContent = g ? `Seeing ${g.emoji} ${g.name}` : 'Not sure what this is yet';
  const pct = Math.round((hand.liveConf || 0) * 100);
  $('#teach-pct').textContent = g ? `${pct}%` : '';
  fill.style.width = `${g ? pct : 8}%`;
  fill.classList.toggle('match', hand.live === teachShapeNow());
}

function cancelCountdown() {
  clearInterval(app.countdownTimer);
  app.countdownTimer = null;
  $('#countdown').hidden = true;
  $('#teach-timer').disabled = false;
  $('#teach-capture').disabled = false;
}

function teachCountdown() {
  if (app.countdownTimer) return;
  let n = 3;
  const el = $('#countdown');
  const show = () => {
    el.textContent = n;
    el.hidden = false;
    el.classList.remove('tick');
    void el.offsetWidth;
    el.classList.add('tick');
    $('#teach-status').textContent = `Get ready… ${n}`;
  };
  $('#teach-timer').disabled = true;
  $('#teach-capture').disabled = true;
  show();
  app.countdownTimer = setInterval(() => {
    n -= 1;
    if (n <= 0) {
      cancelCountdown();
      $('#teach-status').textContent = '';
      teachCapture();
    } else {
      show();
    }
  }, 1000);
}

async function teachCapture() {
  if (app.teachBusy || !teachShapeNow()) return;
  app.teachBusy = true;
  $('#teach-status').textContent = '';
  try {
    const r = await api(`/api/teach/capture/${teachShapeNow()}?hands=${teachGesture().hands || 1}`, { method: 'POST' });
    showReview(r);
  } catch (e) {
    $('#teach-status').textContent = e.message;
  } finally {
    app.teachBusy = false;
  }
}

function showReview(r) {
  const expected = app.byShape[r.expected];
  const predicted = r.predicted ? app.byShape[r.predicted] : null;
  $('#teach-snapshot').src = r.snapshot ? `data:image/jpeg;base64,${r.snapshot}` : '';
  $('#teach-snapshot').style.transform = app.settings.mirror ? 'scaleX(-1)' : '';
  const verdict = $('#teach-verdict');
  if (r.correct) {
    verdict.className = 'verdict ok';
    verdict.textContent = `✓ Recognized as ${expected.name} · ${Math.round(r.conf * 100)}%`;
  } else if (predicted) {
    verdict.className = 'verdict warn';
    verdict.textContent = `Seen as ${predicted.emoji} ${predicted.name}`;
  } else {
    verdict.className = 'verdict warn';
    verdict.textContent = 'Not recognized yet';
  }
  const notes = [];
  if (!r.correct) notes.push(`Saving this teaches Reachy that it's “${expected.name}”.`);
  if (!r.engaged && r.reject && r.reject !== 'relaxed') notes.push(`While playing, this hand would be ignored (${r.reject}). Try holding it higher and facing the camera.`);
  $('#teach-note').textContent = notes.join(' ');
  $('#teach-live').hidden = true;
  $('#teach-review').hidden = false;
}

async function teachSave() {
  try {
    const r = await api('/api/teach/confirm', { method: 'POST' });
    const g = teachGesture();
    const shape = app.byShape[r.shape];
    const hands = r.added === 2 ? ' (both hands)' : '';
    const last = app.teachStep + 1 >= g.train.length;
    toast(shape.emoji, last ? 'Example saved' : `Step ${app.teachStep + 1} saved`,
      last ? `${shape.name}${hands} · ${r.count} example${r.count === 1 ? '' : 's'}` : 'Now show the next step');
    await refreshTeachCounts();
    selectTeach(g.key, last ? 0 : app.teachStep + 1);
    return;
  } catch (e) {
    $('#teach-status').textContent = e.message;
  }
  showTeachLive();
}

function teachDiscard() {
  api('/api/teach/discard', { method: 'POST' }).catch(() => {});
  showTeachLive();
}

async function teachDelete() {
  const g = teachGesture();
  if (!g) return;
  const shapes = [...new Set(g.train)];
  const names = shapes.map((s) => app.byShape[s].name).join(' and ');
  const ok = await confirmDialog({
    title: `Delete ${g.name} examples?`,
    message: `Your saved ${names} examples are removed (other gestures using ${names} share them). Built-in recognition keeps working.`,
    ok: 'Delete', destructive: true,
  });
  if (!ok) return;
  let counts = {};
  for (const s of shapes) counts = await api(`/api/teach/clear/${s}`, { method: 'POST' });
  renderTeachChips(counts);
}

async function teachReset() {
  const ok = await confirmDialog({
    title: 'Reset all examples?',
    message: 'Every example you taught is removed. Built-in recognition keeps working.',
    ok: 'Reset', destructive: true,
  });
  if (!ok) return;
  renderTeachChips(await api('/api/teach/clear/all', { method: 'POST' }));
}

init();
