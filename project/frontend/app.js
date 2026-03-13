/* ============================================================
   IdleBrain UI 芒聙?app.js
   Bilingual (EN default / 盲赂颅忙聳聡 toggle)
   ============================================================ */

// ================================================================
// TRANSLATIONS
// ================================================================
const LANGS = { en: {}, zh: {} };

// DOM refs
const logBox        = document.getElementById('logBox');
const barFill       = document.getElementById('barFill');
const stepText      = document.getElementById('stepText');
const progressPct   = document.getElementById('progressPct');
const statusBadge   = document.getElementById('statusBadge');
const resultRows    = document.getElementById('resultRows');
const compareRows   = document.getElementById('compareRows');
const historyList   = document.getElementById('historyList');
const versionText   = document.getElementById('versionText');
const sliceProgress = document.getElementById('sliceProgress');
const validateStatus = document.getElementById('validateStatus');
const workflowModeEl = document.getElementById('workflowMode');
const oneClickSourcePathEl = document.getElementById('oneClickSourcePath');
const oneClickScopeEl = document.getElementById('oneClickScope');
const oneClickStartBtn = document.getElementById('oneClickStartBtn');
const quickExportBtn = document.getElementById('quickExportBtn');
const quickExportFormatEl = document.getElementById('quickExportFormat');

// ================================================================
// TOAST
// ================================================================
function showToast(msg, type = 'info', duration = 4500) {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
    const icons = { success: 'OK', warning: 'WARN', error: 'ERR', info: 'i' };
    toast.innerHTML = `<span class="toast-icon">${icons[type] || 'i'}</span><span class="toast-msg">${msg}</span>`;
  if (type === 'error') {
    const cb = document.createElement('button');
    cb.className = 'toast-close';
        cb.textContent = 'x';
    cb.onclick = () => toast.remove();
    toast.appendChild(cb);
  }
  container.appendChild(toast);
  if (type !== 'error') {
    setTimeout(() => { toast.classList.add('fade-out'); setTimeout(() => toast.remove(), 380); }, duration);
  }
}

// ================================================================
// LIGHTBOX
// ================================================================
function openLightbox(src, caption = '') {
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightboxCaption').textContent = caption;
  document.getElementById('lightbox').classList.remove('hidden');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.add('hidden');
  document.getElementById('lightboxImg').src = '';
}
document.getElementById('lightboxClose').onclick = closeLightbox;
document.getElementById('lightboxBg').onclick    = closeLightbox;
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ================================================================
// GUIDE MODAL
// ================================================================
document.getElementById('guideBtn').onclick        = () => document.getElementById('guideModal').classList.remove('hidden');
document.getElementById('guideModalClose').onclick = () => document.getElementById('guideModal').classList.add('hidden');
document.getElementById('guideModalOk').onclick    = () => document.getElementById('guideModal').classList.add('hidden');

// ================================================================
// TAB SWITCHING
// ================================================================
document.querySelectorAll('.nav-btn[data-tab]').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
    if (btn.dataset.tab === 'results') refreshOutputs();
    if (btn.dataset.tab === 'qc')      refreshQcAll();
  };
});

// ================================================================
// BROWSE (backend tkinter dialog)
// ================================================================
async function browseFor(targetId, type, filetypes = '') {
  try {
    const endpoint = type === 'folder' ? '/api/browse/folder' : '/api/browse/file';
    const body = type === 'file' ? { filetypes } : {};
    const res = await fetch(endpoint, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => r.json());
    if (res.ok && res.path) {
      document.getElementById(targetId).value = res.path;
      document.getElementById(targetId).dispatchEvent(new Event('change'));
    }
  } catch {
    showToast(t('toast.browseFail'), 'warning');
  }
}
document.querySelectorAll('.btn-browse').forEach(btn => {
  btn.onclick = () => browseFor(btn.dataset.target, btn.dataset.type, btn.dataset.filetypes || '');
});

function applyWorkflowMode(mode) {
  const m = (mode || 'oneclick').toLowerCase();
  document.body.classList.toggle('mode-oneclick', m === 'oneclick');
}

if (workflowModeEl) {
  workflowModeEl.onchange = () => applyWorkflowMode(workflowModeEl.value);
}

// ================================================================
// PROGRESS / STATUS
// ================================================================
function setProgress(p, text) {
  barFill.style.width = `${p}%`;
  stepText.textContent = text;
  progressPct.textContent = `${Math.round(p)}%`;
}
function setRunning(r) {
  state.running = r;
  statusBadge.textContent = r ? t('status.running') : t('status.idle');
  statusBadge.className = 'status-badge' + (r ? ' running' : '');
  document.getElementById('runBtn').disabled = r;
}
function log(msg) {
  const ts = new Date().toLocaleTimeString();
  logBox.textContent += `\n[${ts}] ${msg}`;
  logBox.scrollTop = logBox.scrollHeight;
}

// ================================================================
// PATH VALIDATION (inline, no alert)
// ================================================================
async function validatePaths(showMsg = true) {
  const q = new URLSearchParams({
    inputDir:   document.getElementById('inputDir').value,
    atlasPath:  document.getElementById('atlasPath').value,
    structPath: document.getElementById('structPath').value,
  });
  try {
    const res = await fetch(`/api/validate?${q}`).then(r => r.json());
    if (!res.ok) {
      const issues = res.issues.join('; ');
      validateStatus.textContent = t('toast.validateFail', { issues });
      validateStatus.className = 'validate-status err';
      statusBadge.textContent = t('status.error');
      statusBadge.className = 'status-badge error';
      if (showMsg) showToast(t('toast.validateFail', { issues }), 'error');
    } else {
      validateStatus.textContent = t('toast.validateOk');
      validateStatus.className = 'validate-status';
      if (!state.running) { statusBadge.textContent = t('status.idle'); statusBadge.className = 'status-badge'; }
    }
    return res.ok;
  } catch { return false; }
}
['inputDir', 'atlasPath', 'structPath', 'realSlicePath', 'atlasLabelPath'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', () => validatePaths(false));
});

// ================================================================
// PRESET SAVE / LOAD
// ================================================================
const PRESET_KEYS = ['inputDir','outputDir','atlasPath','structPath','realSlicePath','atlasLabelPath',
  'pixelSizeUm','rotateAtlas','flipAtlas','slicingPlane','majorTopK','fitMode',
  'overlayMode','alignMode','maxPoints','minDistance','ransacResidual'];

function savePreset() {
  const preset = {};
  PRESET_KEYS.forEach(k => { const el = document.getElementById(k); if (el) preset[k] = el.value; });
  preset.channel = state.channel;
  preset.runAll  = state.runAll;
  localStorage.setItem('idlebrain.preset', JSON.stringify(preset));
  showToast(t('toast.presetSaved'), 'success', 2500);
}
function loadPreset(silent = false) {
  const raw = localStorage.getItem('idlebrain.preset');
  if (!raw) { if (!silent) showToast(t('toast.noPreset'), 'warning'); return false; }
  const p = JSON.parse(raw);
  PRESET_KEYS.forEach(k => { const el = document.getElementById(k); if (el && p[k] !== undefined) el.value = p[k]; });
  if (p.channel) { const pill = document.querySelector(`.pill[data-channel="${p.channel}"]`); if (pill) pill.click(); }
  state.runAll = !!p.runAll;
  document.getElementById('batchAll').classList.toggle('active', state.runAll);
  if (!silent) showToast(t('toast.presetLoaded'), 'success', 2500);
  return true;
}
document.getElementById('savePreset').onclick = savePreset;
document.getElementById('loadPreset').onclick = () => loadPreset(false);

// ================================================================
// CHANNEL SELECTION
// ================================================================
document.querySelectorAll('.pill[data-channel]').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.pill[data-channel]').forEach(x => x.classList.remove('active'));
    btn.classList.add('active');
    state.channel = btn.dataset.channel;
    state.runAll  = false;
    document.getElementById('batchAll').classList.remove('active');
  };
});
document.getElementById('batchAll').onclick = () => {
  state.runAll = !state.runAll;
  document.getElementById('batchAll').classList.toggle('active', state.runAll);
};

// ================================================================
// ALPHA SLIDER
// ================================================================
const alphaRange = document.getElementById('alphaRange');
const alphaValue = document.getElementById('alphaValue');
alphaRange.oninput = () => { alphaValue.textContent = `${alphaRange.value}%`; refreshOverlayPreview(); };
document.getElementById('overlayMode').onchange = refreshOverlayPreview;

// ================================================================
// OVERLAY PREVIEW
// ================================================================
async function refreshOverlayPreview() {
  const realPath = document.getElementById('realSlicePath').value;
  if (!realPath) { showToast(t('toast.previewNeedPath'), 'warning'); return; }
  const alpha   = Number(alphaRange.value) / 100;
  const modeEl  = document.getElementById('overlayMode');
  let mode      = modeEl.value;
  const fitMode = document.getElementById('fitMode')?.value || 'cover';
  const payload = {
    realPath,
    realZIndex:      getSelectedRealZIndex(),
    labelPath:        document.getElementById('atlasLabelPath').value || '../outputs/test_label.tif',
    structureCsv:     document.getElementById('structPath').value || '',
    minMeanThreshold: Number(document.getElementById('minMeanThreshold').value || 8),
    pixelSizeUm:      Number(document.getElementById('pixelSizeUm').value || 0.65),
    rotateAtlas:      Number(document.getElementById('rotateAtlas').value || 0),
    flipAtlas:        document.getElementById('flipAtlas').value || 'none',
    majorTopK:        Number(document.getElementById('majorTopK').value || 20),
    fitMode, alpha, mode, edgeSmoothIter: mode === 'fill' ? 2 : 1,
  };
  let res = await fetch('/api/overlay/preview', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  });
  let respJson = null;
  if (!res.ok) {
    let errMsg = t('toast.previewFailed');
    try { errMsg = (await res.json()).error || errMsg; } catch {}
    if (mode !== 'contour') {
      mode = 'contour'; modeEl.value = 'contour';
      res = await fetch('/api/overlay/preview', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...payload, mode: 'contour' }),
      });
      if (res.ok) { respJson = await res.json(); showToast(t('toast.fillModeFallback'), 'warning'); }
      else { showToast(t('toast.previewFailed'), 'error'); return; }
    } else { showToast(t('toast.previewFailed'), 'error'); return; }
  }
  if (!respJson) respJson = await res.json();
  const dg = respJson?.diagnostic;
  if (dg) {
    const ra = Number(dg.real_aspect || 0), aa = Number((dg.atlas_aspect ?? dg.atlas_aspect_before) || 0);
    if (ra > 0 && aa > 0 && Math.abs(ra / aa - 1) > 0.35)
      showToast(t('toast.aspectWarning', { ra: ra.toFixed(2), aa: aa.toFixed(2) }), 'warning', 7000);
  }
  const img = document.getElementById('previewImg');
  img.src = `/api/outputs/overlay-preview?${Date.now()}`;
  img.classList.remove('hidden');
  document.getElementById('previewPlaceholder').classList.add('hidden');
  img.onclick = () => openLightbox(img.src, t('lightbox.overlay'));
  showToast(t('toast.previewUpdated'), 'success', 2000);
}
document.getElementById('refreshPreviewBtn').onclick = refreshOverlayPreview;

// ================================================================
// AUTO-PICK ATLAS SLICE
// ================================================================
document.getElementById('autoPickBtn').onclick = async () => {
  const realPath = document.getElementById('realSlicePath').value;
  const annotationPath = document.getElementById('atlasPath').value;
  if (!realPath || !annotationPath) { showToast(t('toast.autoPickNeedPath'), 'warning'); return; }
  showToast(t('toast.autoPickWaiting'), 'info', 15000);
  try {
    const r = await fetch('/api/atlas/autopick-z', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        realPath, annotationPath,
        realZIndex: getSelectedRealZIndex(),
        zStep: 2,
        pixelSizeUm:  Number(document.getElementById('pixelSizeUm').value || 0.65),
        slicingPlane: document.getElementById('slicingPlane').value || 'coronal',
      }),
    }).then(x => x.json());
    if (!r.ok) { showToast(t('toast.autoPickFailed'), 'error'); return; }
    document.getElementById('atlasLabelPath').value = r.label_slice_tif;
    lastAutoPickedLabelPath = r.label_slice_tif;
    autoPickCacheKey = buildAutoPickKey(
      realPath,
      annotationPath,
      document.getElementById('slicingPlane').value || 'coronal',
      Number(document.getElementById('pixelSizeUm').value || 0.65),
    );
    showToast(t('toast.autoPickSuccess', { plane: r.slicing_plane || 'coronal', z: r.best_z, score: Number(r.best_score).toFixed(4) }), 'success', 5000);
  } catch { showToast(t('toast.autoPickFailed'), 'error'); }
};

// ================================================================
// ALIGNMENT QUALITY PANEL
// ================================================================
function getQualityLevel(score) {
  if (score >= 0.70) return 'excellent';
  if (score >= 0.50) return 'good';
  if (score >= 0.30) return 'fair';
  return 'poor';
}
function showAlignQuality(beforeEdge, afterEdge, improved) {
  const panel = document.getElementById('alignQualityPanel');
  panel.classList.remove('hidden');
  const b = Number(beforeEdge), a = Number(afterEdge);
  document.getElementById('scoreBeforeNum').textContent = b.toFixed(4);
  document.getElementById('scoreAfterNum').textContent  = a.toFixed(4);
  document.getElementById('scoreBefore').style.width = `${Math.min(100, b * 100).toFixed(1)}%`;
  const afterBar = document.getElementById('scoreAfter');
  afterBar.style.width = `${Math.min(100, a * 100).toFixed(1)}%`;
  const level = getQualityLevel(a);
  afterBar.className = `quality-bar quality-${level}`;
    const impPct  = b > 0 ? ((a - b) / b * 100).toFixed(0) : '--';
  const impSign = Number(impPct) >= 0 ? '+' : '';
  const impCls  = Number(impPct) < 0 ? 'negative' : '';
  const verdictEl = document.getElementById('qualityVerdict');
  verdictEl.innerHTML = `
    <span class="verdict-badge verdict-${level}">${t(`quality.${level}`)}</span>
    <span class="verdict-text">${improved ? t(`quality.tip.${level}`) : t('quality.noImprove')}</span>
    <span class="verdict-improve ${impCls}">${impSign}${impPct}%</span>
  `;
  const toastType = improved ? (level === 'poor' || level === 'fair' ? 'warning' : 'success') : 'error';
  showToast(`SSIM ${b.toFixed(4)} 芒聠?${a.toFixed(4)} (${impSign}${impPct}%) 芒聙?${t(`quality.${level}`)}`, toastType, 6000);
}

// ================================================================
// AI REGISTRATION
// ================================================================
document.getElementById('aiAlignBtn').onclick = async () => {
  const realPath  = document.getElementById('realSlicePath').value;
  const atlasPath = document.getElementById('atlasLabelPath').value;
  if (!realPath || !atlasPath) { showToast(t('toast.landmarkNeedPath'), 'warning'); return; }
  const alignMode = document.getElementById('alignMode').value;
  showToast(t('toast.landmarkExtracting'), 'info', 15000);
  try {
    const lm = await fetch('/api/align/landmarks', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        realPath, atlasPath,
        maxPoints:      Number(document.getElementById('maxPoints').value || 30),
        minDistance:    Number(document.getElementById('minDistance').value || 12),
        ransacResidual: Number(document.getElementById('ransacResidual').value || 8),
      }),
    }).then(r => r.json());
    if (!lm.ok) { showToast(t('toast.landmarkApplyFailed', { err: lm.error || '?' }), 'error'); return; }
    showToast(t('toast.landmarkSuccess', { n: lm.landmark_pairs }), 'info', 10000);
    const ep = alignMode === 'nonlinear' ? '/api/align/nonlinear' : '/api/align/apply';
    const ap = await fetch(ep, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ realPath, atlasLabelPath: atlasPath }),
    }).then(r => r.json());
    if (!ap.ok) { showToast(t('toast.landmarkApplyFailed', { err: ap.error || '?' }), 'error'); return; }
    showAlignQuality(ap.beforeEdgeScore, ap.afterEdgeScore, !ap.scoreWarning);
    const compareImg = document.getElementById('alignPreviewImg');
    compareImg.src = alignMode === 'nonlinear'
      ? `/api/outputs/overlay-compare-nonlinear?${Date.now()}`
      : `/api/outputs/overlay-compare?${Date.now()}`;
    compareImg.classList.remove('hidden');
    document.getElementById('alignPreviewPlaceholder').classList.add('hidden');
    compareImg.onclick = () => openLightbox(compareImg.src, t('lightbox.compare'));
    log(`AI ${alignMode} | pairs=${lm.landmark_pairs} | SSIM ${Number(ap.beforeEdgeScore).toFixed(4)} 芒聠?${Number(ap.afterEdgeScore).toFixed(4)}`);
  } catch { showToast(t('toast.alignUnexpected'), 'error'); }
};

// ================================================================
// VIEW LANDMARKS
// ================================================================
document.getElementById('landmarkViewBtn').onclick = async () => {
  const realPath  = document.getElementById('realSlicePath').value;
  const atlasPath = document.getElementById('atlasLabelPath').value;
  if (!realPath || !atlasPath) { showToast(t('toast.landmarkNeedPath'), 'warning'); return; }
  try {
    const p = await fetch('/api/align/landmark-preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ realPath, atlasPath }),
    }).then(r => r.json());
    if (!p.ok) { showToast(t('toast.landmarkApplyFailed', { err: p.error || '?' }), 'error'); return; }
    openLightbox(`/api/outputs/landmark-preview?${Date.now()}`, t('lightbox.landmark', { n: p.points }));
  } catch { showToast(t('toast.landmarkApplyFailed', { err: '?' }), 'error'); }
};

// ================================================================
// RUN PIPELINE
// ================================================================
document.getElementById('runBtn').onclick = async () => {
  if (state.running) return;
  if (!(await validatePaths(true))) return;
  setRunning(true);
  setProgress(5, t('progress.queued'));
  const channels = state.runAll ? ['red', 'green', 'farred'] : [state.channel];
  const params = {
    inputDir: document.getElementById('inputDir').value,
    outputDir: document.getElementById('outputDir').value,
    atlasPath: document.getElementById('atlasPath').value,
    structPath: document.getElementById('structPath').value,
    realSlicePath: document.getElementById('realSlicePath').value,
    pixelSizeUm: document.getElementById('pixelSizeUm').value,
    slicingPlane: document.getElementById('slicingPlane').value,
    rotateAtlas: document.getElementById('rotateAtlas').value,
    flipAtlas: document.getElementById('flipAtlas').value,
    alignMode: document.getElementById('alignMode').value,
    maxPoints: document.getElementById('maxPoints').value,
    minDistance: document.getElementById('minDistance').value,
    ransacResidual: document.getElementById('ransacResidual').value,
    version: versionText.textContent,
  };
  const res = await fetch('/api/run', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ configPath: '../configs/run_config.template.json',
      inputDir: document.getElementById('inputDir').value, channels, params }),
  }).then(r => r.json());
  if (!res.ok) {
    showToast(t('toast.runFailed', { err: res.error || '?' }), 'error');
    setRunning(false); setProgress(0, t('progress.startFailed')); return;
  }
  setProgress(20, t('progress.running', { ch: channels.join(' + ') }));
  showToast(t('toast.runStarted', { channels: channels.join(' + ') }), 'info', 5000);
  await pollLogsUntilDone();
  setProgress(100, t('progress.done'));
  await refreshOutputs();
  setRunning(false);
  showToast(t('toast.runComplete'), 'success', 6000);
};

// ================================================================
// POLL LOGS
// ================================================================
async function pollLogsUntilDone() {
  while (true) {
    const [s, logsData] = await Promise.all([
      fetch('/api/status').then(r => r.json()),
      fetch('/api/logs').then(r => r.json()),
    ]);
    logBox.textContent = logsData.logs.join('\n');
    logBox.scrollTop = logBox.scrollHeight;
    if (s.running) {
      const lastLines = logsData.logs.slice(-10).join('\n');
      const m = lastLines.match(/slices?\s+(\d+)\s*[\/茂录聫]\s*(\d+)/i);
      if (m) {
        const cur = Number(m[1]), total = Number(m[2]);
        sliceProgress.classList.remove('hidden');
        sliceProgress.textContent = t('progress.slices', { cur, total });
        setProgress(20 + Math.round((cur / total) * 75), t('progress.running', { ch: s.currentChannel || '' }));
      } else {
        setProgress(Math.min(94, 20 + Math.floor((s.logCount || 0) * 0.6)), t('progress.running', { ch: s.currentChannel || '' }));
        sliceProgress.classList.add('hidden');
      }
    }
    if (!s.running) {
      sliceProgress.classList.add('hidden');
      if (s.error && s.error !== 'cancelled by user') showToast(s.error, 'error');
      break;
    }
    await new Promise(r => setTimeout(r, 1200));
  }
}

// ================================================================
// CANCEL
// ================================================================
document.getElementById('cancelBtn').onclick = async () => {
  const res = await fetch('/api/cancel', { method: 'POST' }).then(r => r.json());
  if (res.ok) { setRunning(false); setProgress(0, t('progress.cancelled')); sliceProgress.classList.add('hidden'); showToast(t('toast.cancelOk'), 'warning', 3000); }
  else showToast(t('toast.cancelNone'), 'info');
};

// ================================================================
// OPEN OUTPUT FOLDER
// ================================================================
document.getElementById('openOutputsBtn').onclick = async () => {
  const info = await fetch('/api/info').then(r => r.json());
  showToast(t('toast.outputsPath', { path: info.outputs }), 'info', 8000);
};

// ================================================================
// BATCH QC ALL
// ================================================================
async function refreshQcAll() {
  const grid = document.getElementById('qcAllGrid');
  const empty = document.getElementById('qcEmpty');
  const count = document.getElementById('qcAllCount');
  try {
    const res = await fetch('/api/outputs/qc-list').then(r => r.json());
    if (!res.ok || res.files.length === 0) { grid.innerHTML = ''; empty.classList.remove('hidden'); count.textContent = ''; return; }
    empty.classList.add('hidden');
    count.textContent = `${res.count}`;
    grid.innerHTML = '';
    res.files.forEach(fname => {
      const wrap = document.createElement('div');
      wrap.className = 'qc-thumb';
      const img = document.createElement('img');
      img.src = `/api/outputs/qc-file/${fname}?${Date.now()}`;
      img.alt = fname; img.onerror = () => wrap.remove();
      const label = document.createElement('div');
      label.className = 'qc-thumb-label';
      label.textContent = fname.replace('overlay_', '').replace('.png', '');
      wrap.appendChild(img); wrap.appendChild(label);
      wrap.onclick = () => openLightbox(img.src, fname);
      grid.appendChild(wrap);
    });
  } catch { showToast(t('toast.qcLoadFailed'), 'warning'); }
}
document.getElementById('refreshQcAllBtn').onclick = refreshQcAll;

// ================================================================
// RESULTS
// ================================================================
function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const head  = lines.shift().split(',');
  return lines.map(line => { const cols = line.split(','); const obj = {}; head.forEach((h, i) => (obj[h] = cols[i] || '')); return obj; });
}

async function refreshOutputs() {
  try {
    const leaf = await fetch('/api/outputs/leaf').then(r => r.text());
    state.allResults = parseCsv(leaf);
    renderResultsTable(state.allResults);
    compareRows.innerHTML = '';
    for (const ch of ['red', 'green', 'farred']) {
      try {
        const txt = await fetch(`/api/outputs/leaf/${ch}`).then(r => (r.ok ? r.text() : ''));
        if (!txt) continue;
        const arr = parseCsv(txt);
        const total = arr.reduce((s, x) => s + Number(x.count || 0), 0);
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${t(`chname.${ch}`)}</td><td>${total.toLocaleString()}</td>`;
        compareRows.appendChild(tr);
      } catch {}
    }
    await refreshHistory();
  } catch {}
}

function renderResultsTable(data) {
  resultRows.innerHTML = '';
  const keyword = (document.getElementById('regionSearch')?.value || '').toLowerCase();
  const filtered = keyword ? data.filter(d => (d.region_name || d.region || '').toLowerCase().includes(keyword)) : data;
  filtered.forEach(d => {
    const tr = document.createElement('tr');
        tr.innerHTML = `<td>${d.region_name || d.region || '-'}</td><td>${Number(d.count || 0).toLocaleString()}</td><td>${d.confidence ? Number(d.confidence).toFixed(3) : '-'}</td>`;
    resultRows.appendChild(tr);
  });
  const meta = document.getElementById('resultsMeta');
  meta.textContent = keyword
    ? t('results.filtered', { found: filtered.length, total: data.length })
    : t('results.total', { n: data.length });
}

document.getElementById('regionSearch').addEventListener('input', () => renderResultsTable(state.allResults));
document.getElementById('refreshBtn').onclick = refreshOutputs;
document.getElementById('exportBtn').onclick   = () => window.open('/api/outputs/leaf', '_blank');

// ================================================================
// METHODS TEXT EXPORT
// ================================================================
document.getElementById('exportMethodsBtn').onclick = async () => {
  try {
    const res = await fetch('/api/export/methods-text').then(r => r.json());
    if (!res.ok) { showToast(t('toast.methodsFailed'), 'error'); return; }
    document.getElementById('methodsText').textContent = res.text;
    document.getElementById('methodsModal').classList.remove('hidden');
  } catch { showToast(t('toast.methodsFailed'), 'error'); }
};
document.getElementById('methodsModalClose').onclick  = () => document.getElementById('methodsModal').classList.add('hidden');
document.getElementById('methodsModalClose2').onclick = () => document.getElementById('methodsModal').classList.add('hidden');
document.getElementById('methodsCopyBtn').onclick = async () => {
  const text = document.getElementById('methodsText').textContent;
  try { await navigator.clipboard.writeText(text); showToast(t('toast.copyOk'), 'success', 2500); }
  catch { showToast(t('toast.copyFailed'), 'warning'); }
};

// ================================================================
// RUN HISTORY
// ================================================================
async function refreshHistory() {
  try {
    const h = await fetch('/api/history').then(r => r.json());
    historyList.innerHTML = '';
    (h.history || []).slice().reverse().forEach(item => {
      const li = document.createElement('li');
      li.className = item.ok ? 'ok' : 'err';
            const ts    = item.timestamp || '--';
      const chStr = (item.channels || []).map(c => t(`chname.${c}`)).join(' + ');
            const status = item.ok ? 'OK' : `ERR ${item.error || '?'}`;
      li.textContent = `[${ts}]  ${status}  ${chStr}  (${item.logCount || 0} lines)`;
      historyList.appendChild(li);
    });
  } catch {}
}

// ================================================================
// INIT
// ================================================================
async function init() {
  // Apply saved or default language
  applyLang(currentLang);
  applyWorkflowMode(workflowModeEl?.value || 'oneclick');

  try {
    const info = await fetch('/api/info').then(r => r.json());
    versionText.textContent = `v${info.version || '0.0.0'}`;
    if (!document.getElementById('outputDir').value) document.getElementById('outputDir').value = info.outputs || '';
    if (!document.getElementById('atlasPath').value && info?.defaults?.atlasPath) {
      document.getElementById('atlasPath').value = info.defaults.atlasPath;
    }
    if (!document.getElementById('structPath').value && info?.defaults?.structPath) {
      document.getElementById('structPath').value = info.defaults.structPath;
    }
  } catch {}

  // Auto-load last preset silently
  if (localStorage.getItem('idlebrain.preset')) {
    if (loadPreset(true)) showToast(t('toast.autoLoadPreset'), 'info', 3000);
  }
}

init();

// ================================================================
// 3D Z-SLICER
// ================================================================
let zSlicerPath = '';
const zSliderEl    = document.getElementById('zSlider');
const zNumInputEl  = document.getElementById('zNumInput');
const zValDisplay  = document.getElementById('zValDisplay');
const zMaxDisplay  = document.getElementById('zMaxDisplay');
const zSlicerBox   = document.getElementById('zSlicerBox');
const zExtractBtn  = document.getElementById('zExtractBtn');
const zExtractStatus = document.getElementById('zExtractStatus');

async function checkSliceIs3D(path) {
  if (!path) return;
  try {
    const res = await fetch(`/api/slice/info?path=${encodeURIComponent(path)}`).then(r => r.json());
    if (res.ok && res.is3d) {
      zSlicerPath = path;
      const zMax = res.z_count - 1;
      zSliderEl.max   = zMax;
      zNumInputEl.max = zMax;
      const midZ = Math.round(zMax / 2);
      zSliderEl.value = midZ;
      zNumInputEl.value = midZ;
      zValDisplay.textContent = midZ;
      zMaxDisplay.textContent = zMax;
      zSlicerBox.classList.remove('hidden');
      showToast(t('toast.zDetected', { z: res.z_count, h: res.shape[1] || '?', w: res.shape[2] || '?' }), 'info', 5000);
    } else {
      zSlicerBox.classList.add('hidden');
    }
  } catch {}
}

document.getElementById('realSlicePath').addEventListener('change', e => {
  checkSliceIs3D(e.target.value);
});

function syncZ(val) {
  const z = Math.max(0, Math.min(Number(val), Number(zSliderEl.max)));
  zSliderEl.value   = z;
  zNumInputEl.value = z;
  zValDisplay.textContent = z;
}
zSliderEl.oninput   = () => syncZ(zSliderEl.value);
zNumInputEl.oninput = () => syncZ(zNumInputEl.value);

zExtractBtn.onclick = async () => {
  const z = Number(zSliderEl.value);
  zExtractStatus.textContent = '...';
  try {
    const res = await fetch('/api/slice/extract-z', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: zSlicerPath, z }),
    }).then(r => r.json());
    if (!res.ok) { showToast(t('toast.zExtractFail', { err: res.error }), 'error'); return; }
    document.getElementById('realSlicePath').value = res.path;
    zExtractStatus.textContent = `芒聠?${res.path}`;
    showToast(t('toast.zExtracted', { z, path: res.path }), 'success', 4000);
  } catch { showToast(t('toast.zExtractFail', { err: '?' }), 'error'); }
};

// ================================================================
// CANVAS DRAWING EDITOR
// ================================================================
const drawCanvas    = document.getElementById('drawCanvas');
const drawCtx       = drawCanvas.getContext('2d');
const drawToolbar   = document.getElementById('drawToolbar');
const canvasWrap    = document.getElementById('canvasWrap');
const previewImgEl  = document.getElementById('previewImg');
const regionHoverTooltip = document.getElementById('regionHoverTooltip');
const liquifyRadiusEl = document.getElementById('liquifyRadius');
const liquifyStrengthEl = document.getElementById('liquifyStrength');
const saveCalibLearnBtn = document.getElementById('saveCalibLearnBtn');
const autoLearnToggle = document.getElementById('autoLearnToggle');

let hoverTimer = null;
let hoverReqSeq = 0;
let hoverLastPixelKey = '';
let autoPickCacheKey = '';
let lastAutoPickedLabelPath = '';

let currentTool     = 'select';
let isDrawing       = false;
let drawStartX      = 0;
let drawStartY      = 0;
let annotations     = [];   // stored vector annotations
let pendingTextPos  = null;
let liquifyBusy     = false;
let calibLearnPollTimer = null;

// Tool selection
document.querySelectorAll('.tool-btn[data-tool]').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.tool-btn[data-tool]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentTool = btn.dataset.tool;
    drawCanvas.style.cursor = currentTool === 'select' ? 'default' : 'crosshair';
    if (currentTool !== 'select') hideRegionTooltip();
  };
});

function getDrawColor()     { return document.getElementById('drawColor').value; }
function getDrawLineWidth() { return Number(document.getElementById('drawLineWidth').value) || 2; }
function getLiquifyRadius() {
  const v = Number(liquifyRadiusEl?.value ?? 80);
  return Math.max(8, Math.min(260, Number.isFinite(v) ? v : 80));
}
function getLiquifyStrength() {
  const v = Number(liquifyStrengthEl?.value ?? 0.72);
  return Math.max(0.05, Math.min(1.5, Number.isFinite(v) ? v : 0.72));
}

function buildOverlayRequestPayload(modeOverride = null) {
  const modeEl = document.getElementById('overlayMode');
  const mode = modeOverride || modeEl?.value || 'fill';
  const alpha = Number(alphaRange.value) / 100;
  return {
    realPath: document.getElementById('realSlicePath').value,
    realZIndex: getSelectedRealZIndex(),
    labelPath: document.getElementById('atlasLabelPath').value || '../outputs/test_label.tif',
    structureCsv: document.getElementById('structPath').value || '',
    minMeanThreshold: Number(document.getElementById('minMeanThreshold').value || 8),
    pixelSizeUm: Number(document.getElementById('pixelSizeUm').value || 0.65),
    rotateAtlas: Number(document.getElementById('rotateAtlas').value || 0),
    flipAtlas: document.getElementById('flipAtlas').value || 'none',
    majorTopK: Number(document.getElementById('majorTopK').value || 20),
    fitMode: document.getElementById('fitMode')?.value || 'cover',
    edgeSmoothIter: mode === 'fill' ? 2 : 1,
    warpParams: {},
    alpha,
    mode,
  };
}

function escapeHtml(txt) {
  return String(txt ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function hideRegionTooltip() {
  if (regionHoverTooltip) regionHoverTooltip.classList.add('hidden');
}

function moveRegionTooltip(clientX, clientY) {
  if (!regionHoverTooltip || regionHoverTooltip.classList.contains('hidden')) return;
  const rect = drawCanvas.getBoundingClientRect();
  const leftRaw = (clientX - rect.left) + 14;
  const topRaw = (clientY - rect.top) + 14;
  const maxLeft = Math.max(4, rect.width - regionHoverTooltip.offsetWidth - 6);
  const maxTop = Math.max(4, rect.height - regionHoverTooltip.offsetHeight - 6);
  regionHoverTooltip.style.left = `${Math.max(4, Math.min(maxLeft, leftRaw))}px`;
  regionHoverTooltip.style.top = `${Math.max(4, Math.min(maxTop, topRaw))}px`;
}

function renderRegionTooltip(region, clientX, clientY) {
  if (!regionHoverTooltip) return;
  const color = (region.color && String(region.color).length === 6) ? `#${region.color}` : '#8cc8ff';
  const name = escapeHtml(region.name || region.acronym || 'Unknown Region');
    const acronym = escapeHtml(region.acronym || '-');
    const parent = escapeHtml(region.parent || '-');
  const rid = Number(region.region_id || 0);
  regionHoverTooltip.innerHTML = `
    <div class="region-tip-title"><span class="region-tip-dot" style="background:${color};"></span>${name}</div>
    <div class="region-tip-row"><b>Acronym:</b> ${acronym}</div>
    <div class="region-tip-row"><b>Parent:</b> ${parent}</div>
        <div class="region-tip-row"><b>ID:</b> ${rid || '-'}</div>
  `;
  regionHoverTooltip.classList.remove('hidden');
  moveRegionTooltip(clientX, clientY);
}

async function fetchRegionAtPixel(x, y, clientX, clientY) {
  const seq = ++hoverReqSeq;
  try {
    const res = await fetch(`/api/overlay/region-at?x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}`).then(r => r.json());
    if (seq !== hoverReqSeq) return;
    if (!res.ok || !res.inside || !res.region_id) {
      hideRegionTooltip();
      return;
    }
    renderRegionTooltip(res, clientX, clientY);
  } catch {
    if (seq !== hoverReqSeq) return;
    hideRegionTooltip();
  }
}

function buildAutoPickKey(realPath, annotationPath, slicingPlane, pixelSizeUm) {
  const z = getSelectedRealZIndex();
  return `${realPath}|${annotationPath}|${slicingPlane}|${pixelSizeUm}|${z ?? 'auto'}`;
}

function getSelectedRealZIndex() {
  if (!zSlicerBox || zSlicerBox.classList.contains('hidden')) return null;
  const z = Number(zSliderEl?.value);
  if (!Number.isFinite(z)) return null;
  return Math.max(0, Math.round(z));
}

async function ensureAutoPickedAtlasSlice(realPath) {
  const atlasLabelEl = document.getElementById('atlasLabelPath');
  const annotationPath = document.getElementById('atlasPath').value;
  const slicingPlane = document.getElementById('slicingPlane').value || 'coronal';
  const pixelSizeUm = Number(document.getElementById('pixelSizeUm').value || 0.65);
  if (!annotationPath) return !!atlasLabelEl.value;

  const canAutoReplace = !atlasLabelEl.value || atlasLabelEl.value === lastAutoPickedLabelPath;
  if (!canAutoReplace) return true; // user manually set atlas slice path

  const k = buildAutoPickKey(realPath, annotationPath, slicingPlane, pixelSizeUm);
  if (autoPickCacheKey === k && atlasLabelEl.value) return true;

  try {
    showToast(t('toast.autoPickWaiting'), 'info', 10000);
    const r = await fetch('/api/atlas/autopick-z', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        realPath,
        realZIndex: getSelectedRealZIndex(),
        annotationPath,
        zStep: 2,
        pixelSizeUm,
        slicingPlane,
        roiMode: 'auto',
      }),
    }).then(x => x.json());
    if (!r.ok) {
      if (!atlasLabelEl.value) showToast(t('toast.autoPickFailed'), 'error');
      return !!atlasLabelEl.value;
    }
    atlasLabelEl.value = r.label_slice_tif;
    lastAutoPickedLabelPath = r.label_slice_tif;
    autoPickCacheKey = k;
    showToast(t('toast.autoPickSuccess', {
      plane: r.slicing_plane || slicingPlane,
      z: r.best_z,
      score: Number(r.best_score).toFixed(4),
    }), 'success', 2800);
    return true;
  } catch {
    if (!atlasLabelEl.value) showToast(t('toast.autoPickFailed'), 'error');
    return !!atlasLabelEl.value;
  }
}

/** Load overlay preview from server into canvas */
async function loadPreviewIntoCanvas(ts) {
  return new Promise(resolve => {
    const src = `/api/outputs/overlay-preview?${ts || Date.now()}`;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      drawCanvas.width  = img.naturalWidth;
      drawCanvas.height = img.naturalHeight;
      previewImgEl.src  = src;
      previewImgEl.style.display = 'block';
      canvasWrap.classList.remove('hidden');
      document.getElementById('previewPlaceholder').classList.add('hidden');
      drawToolbar.classList.remove('hidden');
      redrawAnnotations();
      resolve();
    };
    img.src = src;
  });
}

function redrawAnnotations() {
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  annotations.forEach(drawAnnotation);
}

function drawAnnotation(ann) {
  const ctx = drawCtx;
  ctx.save();
  ctx.strokeStyle = ann.color;
  ctx.fillStyle   = ann.color;
  ctx.lineWidth   = ann.lw || 2;
  ctx.lineCap = 'round';

  if (ann.type === 'line') {
    ctx.beginPath(); ctx.moveTo(ann.x1, ann.y1); ctx.lineTo(ann.x2, ann.y2); ctx.stroke();

  } else if (ann.type === 'arrow') {
    ctx.beginPath(); ctx.moveTo(ann.x1, ann.y1); ctx.lineTo(ann.x2, ann.y2); ctx.stroke();
    const ang = Math.atan2(ann.y2 - ann.y1, ann.x2 - ann.x1);
    const sz  = Math.max(8, ann.lw * 4);
    ctx.beginPath();
    ctx.moveTo(ann.x2, ann.y2);
    ctx.lineTo(ann.x2 - sz * Math.cos(ang - 0.4), ann.y2 - sz * Math.sin(ang - 0.4));
    ctx.lineTo(ann.x2 - sz * Math.cos(ang + 0.4), ann.y2 - sz * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();

  } else if (ann.type === 'scalebar') {
    const px = ann.pixelLen;
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(ann.x - 6, ann.y - 22, px + 12, 32);
    ctx.fillStyle   = ann.color;
    ctx.strokeStyle = ann.color;
    ctx.lineWidth   = ann.lw + 1;
    ctx.beginPath(); ctx.moveTo(ann.x, ann.y); ctx.lineTo(ann.x + px, ann.y);
    ctx.moveTo(ann.x, ann.y - 6); ctx.lineTo(ann.x, ann.y + 6);
    ctx.moveTo(ann.x + px, ann.y - 6); ctx.lineTo(ann.x + px, ann.y + 6);
    ctx.stroke();
    ctx.font = `bold ${Math.max(12, ann.lw * 5)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText(`${ann.umLen} 脗碌m`, ann.x + px / 2, ann.y - 8);

  } else if (ann.type === 'text') {
    ctx.font = `${ann.size || 16}px sans-serif`;
    ctx.textAlign = 'left';
    ctx.shadowColor = 'rgba(0,0,0,0.8)'; ctx.shadowBlur = 3;
    ctx.fillText(ann.text, ann.x, ann.y);
  }
  ctx.restore();
}

// Live preview line/arrow while dragging
function drawPreviewStroke(x2, y2) {
  redrawAnnotations();
  const ctx = drawCtx;
  ctx.save();
  ctx.strokeStyle = getDrawColor();
  ctx.fillStyle   = getDrawColor();
  ctx.lineWidth   = getDrawLineWidth();
  ctx.lineCap = 'round';
  if (currentTool === 'line') {
    ctx.beginPath(); ctx.moveTo(drawStartX, drawStartY); ctx.lineTo(x2, y2); ctx.stroke();
  } else if (currentTool === 'arrow') {
    ctx.beginPath(); ctx.moveTo(drawStartX, drawStartY); ctx.lineTo(x2, y2); ctx.stroke();
    const ang = Math.atan2(y2 - drawStartY, x2 - drawStartX);
    const sz  = Math.max(8, getDrawLineWidth() * 4);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - sz * Math.cos(ang - 0.4), y2 - sz * Math.sin(ang - 0.4));
    ctx.lineTo(x2 - sz * Math.cos(ang + 0.4), y2 - sz * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
  } else if (currentTool === 'liquify') {
    const r = getLiquifyRadius();
    ctx.lineWidth = Math.max(1, getDrawLineWidth());
    ctx.beginPath(); ctx.moveTo(drawStartX, drawStartY); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.arc(drawStartX, drawStartY, r, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(x2, y2, Math.max(5, r * 0.35), 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}

async function applyLiquifyDrag(x1, y1, x2, y2) {
  if (liquifyBusy) return;
  const dist = Math.hypot(x2 - x1, y2 - y1);
  if (dist < 2.0) return;

  const payload = buildOverlayRequestPayload();
  if (!payload.realPath) {
    showToast('Please set Real Slice path first.', 'warning');
    return;
  }
  payload.x1 = Number(x1);
  payload.y1 = Number(y1);
  payload.x2 = Number(x2);
  payload.y2 = Number(y2);
  payload.radius = getLiquifyRadius();
  payload.strength = getLiquifyStrength();

  liquifyBusy = true;
  try {
    const res = await fetch('/api/overlay/liquify-drag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(r => r.json());
    if (!res.ok) {
      showToast(`Liquify failed: ${res.error || '?'}`, 'error');
      return;
    }
    if (res.correctedLabelPath) {
      document.getElementById('atlasLabelPath').value = res.correctedLabelPath;
    }
    await loadPreviewIntoCanvas();
    hoverLastPixelKey = '';
    hideRegionTooltip();
    showToast(`Liquify applied (${dist.toFixed(1)} px).`, 'success', 1600);
  } catch (e) {
    showToast(`Liquify failed: ${e?.message || '?'}`, 'error');
  } finally {
    liquifyBusy = false;
  }
}

async function pollCalibrationLearnStatus() {
  try {
    const st = await fetch('/api/calibration/learn-status').then(r => r.json());
    if (!st.ok || !st.state) return;
    const s = st.state;
    if (s.running) return;
    if (calibLearnPollTimer) {
      clearInterval(calibLearnPollTimer);
      calibLearnPollTimer = null;
    }
    if (s.ok === true) {
      showToast('Auto-learning finished. Tuned params updated.', 'success', 5000);
    } else {
      showToast(`Auto-learning failed: ${s.error || '?'}`, 'warning', 7000);
    }
  } catch {
    if (calibLearnPollTimer) {
      clearInterval(calibLearnPollTimer);
      calibLearnPollTimer = null;
    }
  }
}

function canvasCoords(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width  / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
}

drawCanvas.addEventListener('mousedown', e => {
  if (currentTool === 'select') return;
  const { x, y } = canvasCoords(e);
  if (currentTool === 'text') {
    const txt = prompt(t('text.dialog'));
    if (txt) {
      annotations.push({ type: 'text', x, y, text: txt, color: getDrawColor(), size: Math.max(12, getDrawLineWidth() * 6) });
      redrawAnnotations();
    }
    return;
  }
  if (currentTool === 'scalebar') {
    const umStr = prompt(t('scalebar.dialog'));
    const um = Number(umStr);
    if (!um || isNaN(um)) { showToast(t('scalebar.invalid'), 'warning'); return; }
    const pixelSizeUm = Number(document.getElementById('pixelSizeUm').value || 0.65);
    const pixelLen = Math.round(um / pixelSizeUm);
    annotations.push({ type: 'scalebar', x: Math.round(x), y: Math.round(y), umLen: um, pixelLen, color: getDrawColor(), lw: getDrawLineWidth() });
    redrawAnnotations();
    return;
  }
  isDrawing = true;
  drawStartX = x; drawStartY = y;
});

drawCanvas.addEventListener('mousemove', e => {
  const { x, y } = canvasCoords(e);
  const px = Math.round(x);
  const py = Math.round(y);

  if (isDrawing) {
    drawPreviewStroke(x, y);
    hideRegionTooltip();
    return;
  }

  if (currentTool !== 'select') {
    hideRegionTooltip();
    return;
  }

  const pixelKey = `${px},${py}`;
  if (pixelKey === hoverLastPixelKey) {
    moveRegionTooltip(e.clientX, e.clientY);
    return;
  }
  hoverLastPixelKey = pixelKey;
  if (hoverTimer) clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => {
    fetchRegionAtPixel(px, py, e.clientX, e.clientY);
  }, 45);
});

drawCanvas.addEventListener('mouseleave', () => {
  hoverLastPixelKey = '';
  if (hoverTimer) clearTimeout(hoverTimer);
  hideRegionTooltip();
});

drawCanvas.addEventListener('mouseup', e => {
  if (!isDrawing) return;
  isDrawing = false;
  const { x, y } = canvasCoords(e);
  const dx = x - drawStartX, dy = y - drawStartY;
  if (Math.sqrt(dx*dx + dy*dy) < 3) return; // ignore tiny clicks
  if (currentTool === 'liquify') {
    applyLiquifyDrag(drawStartX, drawStartY, x, y);
    redrawAnnotations();
    return;
  }
  annotations.push({ type: currentTool, x1: drawStartX, y1: drawStartY, x2: x, y2: y, color: getDrawColor(), lw: getDrawLineWidth() });
  redrawAnnotations();
});

document.getElementById('undoAnnotationBtn').onclick = () => {
  annotations.pop(); redrawAnnotations();
};
document.getElementById('clearAnnotationsBtn').onclick = () => {
  annotations = []; redrawAnnotations();
};

document.getElementById('exportCanvasBtn').onclick = () => {
  // Composite: real overlay PNG + drawing canvas
  const exportCanvas = document.createElement('canvas');
  exportCanvas.width  = drawCanvas.width;
  exportCanvas.height = drawCanvas.height;
  const ec = exportCanvas.getContext('2d');
  // Draw base overlay image
  const baseImg = previewImgEl;
  if (baseImg.src) ec.drawImage(baseImg, 0, 0);
  // Draw annotations
  ec.drawImage(drawCanvas, 0, 0);
  exportCanvas.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `idlebrain_figure_${Date.now()}.png`;
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
};

if (saveCalibLearnBtn) {
  saveCalibLearnBtn.onclick = async () => {
    const payload = buildOverlayRequestPayload();
    if (!payload.realPath) {
      showToast('Please set Real Slice path first.', 'warning');
      return;
    }
    payload.autoLearn = autoLearnToggle ? !!autoLearnToggle.checked : true;
    payload.note = 'manual_liquify_or_landmark_adjust';
    try {
      const res = await fetch('/api/overlay/calibration/finalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }).then(r => r.json());
      if (!res.ok) {
        showToast(`Finalize failed: ${res.error || '?'}`, 'error');
        return;
      }
      const sid = res?.sample?.sample_id;
      showToast(`Calibration saved as sample #${sid}.`, 'success', 3500);
      const pruned = Number(res?.sample?.prune?.pruned || 0);
      const kept = Number(res?.sample?.prune?.kept || 0);
      const maxN = Number(res?.sample?.sample_limit || 0);
      if (pruned > 0) {
        showToast(`Sample library pruned: removed ${pruned}, kept ${kept}/${maxN}.`, 'info', 5000);
      }
      if (res.learningStarted) {
        showToast('Auto-learning started in background.', 'info', 3000);
        if (calibLearnPollTimer) clearInterval(calibLearnPollTimer);
        calibLearnPollTimer = setInterval(pollCalibrationLearnStatus, 5000);
      }
    } catch (e) {
      showToast(`Finalize failed: ${e?.message || '?'}`, 'error');
    }
  };
}

// Override refreshOverlayPreview to load result into canvas
const _origRefreshOverlayPreview = refreshOverlayPreview;
// Patch: after server returns OK, also load into canvas
async function refreshOverlayPreviewWithCanvas() {
  const realPath = document.getElementById('realSlicePath').value;
  if (!realPath) { showToast(t('toast.previewNeedPath'), 'warning'); return; }
  if (!(await ensureAutoPickedAtlasSlice(realPath))) return;
  const alpha   = Number(alphaRange.value) / 100;
  const modeEl  = document.getElementById('overlayMode');
  let mode      = modeEl.value;
  const fitMode = document.getElementById('fitMode')?.value || 'cover';
  const payload = {
    realPath,
    realZIndex:      getSelectedRealZIndex(),
    labelPath:        document.getElementById('atlasLabelPath').value || '../outputs/test_label.tif',
    structureCsv:     document.getElementById('structPath').value || '',
    minMeanThreshold: Number(document.getElementById('minMeanThreshold').value || 8),
    pixelSizeUm:      Number(document.getElementById('pixelSizeUm').value || 0.65),
    rotateAtlas:      Number(document.getElementById('rotateAtlas').value || 0),
    flipAtlas:        document.getElementById('flipAtlas').value || 'none',
    majorTopK:        Number(document.getElementById('majorTopK').value || 20),
    fitMode, alpha, mode, edgeSmoothIter: mode === 'fill' ? 2 : 1,
  };
  let res = await fetch('/api/overlay/preview', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  });
  let respJson = null;
  if (!res.ok) {
    let errMsg = t('toast.previewFailed');
    try { errMsg = (await res.json()).error || errMsg; } catch {}
    if (mode !== 'contour') {
      mode = 'contour'; modeEl.value = 'contour';
      res = await fetch('/api/overlay/preview', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...payload, mode: 'contour' }),
      });
      if (res.ok) { respJson = await res.json(); showToast(t('toast.fillModeFallback'), 'warning'); }
      else { showToast(t('toast.previewFailed'), 'error'); return; }
    } else { showToast(t('toast.previewFailed'), 'error'); return; }
  }
  if (!respJson) respJson = await res.json();
  const dg = respJson?.diagnostic;
  if (dg) {
    const ra = Number(dg.real_aspect || 0), aa = Number((dg.atlas_aspect ?? dg.atlas_aspect_before) || 0);
    if (ra > 0 && aa > 0 && Math.abs(ra / aa - 1) > 0.35)
      showToast(t('toast.aspectWarning', { ra: ra.toFixed(2), aa: aa.toFixed(2) }), 'warning', 7000);
  }
  // Load into canvas
  await loadPreviewIntoCanvas();
  hoverLastPixelKey = '';
  hideRegionTooltip();
  showToast(t('toast.previewUpdated'), 'success', 2000);
}

// Replace the existing onclick handler
document.getElementById('refreshPreviewBtn').onclick = refreshOverlayPreviewWithCanvas;
// Also re-hook alpha/mode change
alphaRange.oninput = () => { alphaValue.textContent = `${alphaRange.value}%`; refreshOverlayPreviewWithCanvas(); };
document.getElementById('overlayMode').onchange = refreshOverlayPreviewWithCanvas;

async function runOneClickWorkflow() {
  const source = String(oneClickSourcePathEl?.value || '').trim();
  const scope = String(oneClickScopeEl?.value || 'single');
  if (!source) {
    showToast('Please choose source TIFF first.', 'warning');
    return;
  }

  document.getElementById('realSlicePath').value = source;
  document.getElementById('realSlicePath').dispatchEvent(new Event('change'));
  try { await checkSliceIs3D(source); } catch {}

  // Single-layer flow: force user to pass through Z selection step for 3D stacks.
  const zVisible = !!(zSlicerBox && !zSlicerBox.classList.contains('hidden'));
  if (scope === 'single' && zVisible && oneClickStartBtn?.dataset?.zConfirmed !== '1') {
    if (oneClickStartBtn) oneClickStartBtn.dataset.zConfirmed = '1';
    showToast('3D detected: please pick Z layer, then click Start again.', 'info', 5000);
    return;
  }
  if (oneClickStartBtn) oneClickStartBtn.dataset.zConfirmed = '0';

  try {
    const info = await fetch('/api/info').then(r => r.json());
    const defs = info?.defaults || {};
    if (!document.getElementById('atlasPath').value && defs.atlasPath) {
      document.getElementById('atlasPath').value = defs.atlasPath;
    }
    if (!document.getElementById('structPath').value && defs.structPath) {
      document.getElementById('structPath').value = defs.structPath;
    }
    if (!document.getElementById('outputDir').value && info.outputs) {
      document.getElementById('outputDir').value = info.outputs;
    }
    if (!document.getElementById('inputDir').value) {
      const p = source.replaceAll('\\', '/');
      document.getElementById('inputDir').value = p.includes('/') ? p.substring(0, p.lastIndexOf('/')) : p;
    }
  } catch {}

  // Whole-brain uses stronger default align mode.
  if (scope === 'whole') {
    const alignEl = document.getElementById('alignMode');
    if (alignEl) alignEl.value = 'nonlinear';
  }

  const okAuto = await ensureAutoPickedAtlasSlice(source);
  if (!okAuto) {
    showToast('Auto-pick failed, please check atlas path.', 'error');
    return;
  }
  await refreshOverlayPreviewWithCanvas();

  // Execute AI registration once.
  const aiAlignHandler = document.getElementById('aiAlignBtn')?.onclick;
  if (typeof aiAlignHandler === 'function') {
    await aiAlignHandler();
  }

  // Enter manual review stage.
  if (manualModeBtn && !manualState.active) {
    manualModeBtn.click();
  }
  showToast('One-click registration done. Entered manual review stage.', 'success', 5000);
}

if (oneClickStartBtn) {
  oneClickStartBtn.onclick = runOneClickWorkflow;
}

if (quickExportBtn) {
  quickExportBtn.onclick = async () => {
    const fmt = String(quickExportFormatEl?.value || 'png');
    try {
      const res = await fetch('/api/overlay/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format: fmt }),
      }).then(r => r.json());
      if (!res.ok) {
        showToast(`Export failed: ${res.error || '?'}`, 'error');
        return;
      }
      showToast(`Exported: ${res.path}`, 'success', 5000);
    } catch (e) {
      showToast(`Export failed: ${e?.message || '?'}`, 'error');
    }
  };
}

// ================================================================
// MANUAL LANDMARK CORRECTION
// ================================================================
const manualState = {
  active: false,
  pendingReal: null,    // { x, y } in canvas coords
  pairs: [],
};

const manualModeBtn   = document.getElementById('manualModeBtn');
const applyManualBtn  = document.getElementById('applyManualBtn');
const clearManualBtn  = document.getElementById('clearManualBtn');
const manualCanvases  = document.getElementById('manualCanvases');
const manualStatus    = document.getElementById('manualStatus');
const manualPairsWrap = document.getElementById('manualPairsWrap');
const manualPairsBody = document.getElementById('manualPairsBody');
const manualRealCanvas  = document.getElementById('manualRealCanvas');
const manualAtlasCanvas = document.getElementById('manualAtlasCanvas');
const manualRealImg     = document.getElementById('manualRealImg');
const manualAtlasImg    = document.getElementById('manualAtlasImg');
const mrcCtx = manualRealCanvas.getContext('2d');
const macCtx = manualAtlasCanvas.getContext('2d');

function loadManualImages() {
  const rPath = document.getElementById('realSlicePath').value;
  const aPath = document.getElementById('atlasLabelPath').value;
  if (!rPath || !aPath) return;
  // Load real slice via existing overlay preview (8-bit normalised)
  const rSrc = `/api/outputs/overlay-preview?base=1&${Date.now()}`;
  // For real slice use the full normalized preview; for atlas the label overlay
  manualRealImg.src = `/api/outputs/overlay-preview?${Date.now()}`;
  manualAtlasImg.src = `/api/outputs/overlay-preview?${Date.now()}`;
  manualRealImg.onload = () => {
    manualRealCanvas.width  = manualRealImg.naturalWidth;
    manualRealCanvas.height = manualRealImg.naturalHeight;
    redrawManual();
  };
}

function redrawManual() {
  mrcCtx.clearRect(0, 0, manualRealCanvas.width, manualRealCanvas.height);
  macCtx.clearRect(0, 0, manualAtlasCanvas.width, manualAtlasCanvas.height);
  const DOT = 6;
  manualState.pairs.forEach((p, i) => {
    // Real side
    mrcCtx.fillStyle = '#00ff88'; mrcCtx.strokeStyle = '#000';
    mrcCtx.beginPath(); mrcCtx.arc(p.real_x, p.real_y, DOT, 0, 2*Math.PI); mrcCtx.fill(); mrcCtx.stroke();
    mrcCtx.fillStyle = '#fff'; mrcCtx.font = 'bold 11px sans-serif'; mrcCtx.textAlign = 'center';
    mrcCtx.fillText(i+1, p.real_x, p.real_y - DOT - 2);
    // Atlas side
    macCtx.fillStyle = '#ffcc00'; macCtx.strokeStyle = '#000';
    macCtx.beginPath(); macCtx.arc(p.atlas_x, p.atlas_y, DOT, 0, 2*Math.PI); macCtx.fill(); macCtx.stroke();
    macCtx.fillStyle = '#fff'; macCtx.font = 'bold 11px sans-serif'; macCtx.textAlign = 'center';
    macCtx.fillText(i+1, p.atlas_x, p.atlas_y - DOT - 2);
  });
  if (manualState.pendingReal) {
    mrcCtx.fillStyle = '#ff4444'; mrcCtx.strokeStyle = '#000';
    mrcCtx.beginPath(); mrcCtx.arc(manualState.pendingReal.x, manualState.pendingReal.y, DOT, 0, 2*Math.PI); mrcCtx.fill(); mrcCtx.stroke();
  }
}

function updateManualPairsTable() {
  manualPairsBody.innerHTML = '';
  manualState.pairs.forEach((p, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${i+1}</td><td>(${Math.round(p.real_x)}, ${Math.round(p.real_y)})</td><td>(${Math.round(p.atlas_x)}, ${Math.round(p.atlas_y)})</td><td><button onclick="removeManualPair(${i})" style="background:transparent;color:var(--danger);border:none;cursor:pointer;">芒聹?/button></td>`;
    manualPairsBody.appendChild(tr);
  });
  manualPairsWrap.classList.toggle('hidden', manualState.pairs.length === 0);
  applyManualBtn.classList.toggle('hidden', manualState.pairs.length === 0);
  clearManualBtn.classList.toggle('hidden', manualState.pairs.length === 0);
}
window.removeManualPair = (i) => {
  manualState.pairs.splice(i, 1);
  updateManualPairsTable(); redrawManual();
};

function manualCanvasCoords(canvas, e) {
  const rect = canvas.getBoundingClientRect();
  return { x: (e.clientX - rect.left) * (canvas.width / rect.width), y: (e.clientY - rect.top) * (canvas.height / rect.height) };
}

manualRealCanvas.addEventListener('click', e => {
  if (!manualState.active) return;
  const { x, y } = manualCanvasCoords(manualRealCanvas, e);
  manualState.pendingReal = { x, y };
  manualStatus.textContent = t('manual.pendingReal', { x: Math.round(x), y: Math.round(y) });
  manualStatus.classList.remove('hidden');
  redrawManual();
});

manualAtlasCanvas.addEventListener('click', e => {
  if (!manualState.active) return;
  if (!manualState.pendingReal) { manualStatus.textContent = t('manual.needReal'); return; }
  const { x, y } = manualCanvasCoords(manualAtlasCanvas, e);
  const pair = { real_x: manualState.pendingReal.x, real_y: manualState.pendingReal.y, atlas_x: x, atlas_y: y };
  manualState.pairs.push(pair);
  manualState.pendingReal = null;
  const n = manualState.pairs.length;
  manualStatus.textContent = t('manual.pairAdded', { n });
  updateManualPairsTable(); redrawManual();
});

manualModeBtn.onclick = () => {
  const rPath = document.getElementById('realSlicePath').value;
  const aPath = document.getElementById('atlasLabelPath').value;
  if (!rPath || !aPath) { showToast(t('manual.needImages'), 'warning'); return; }
  manualState.active = !manualState.active;
  if (manualState.active) {
    manualModeBtn.classList.add('active-mode');
    manualCanvases.classList.remove('hidden');
    manualStatus.classList.remove('hidden');
    manualStatus.textContent = t('manual.enterMode');
    loadManualImages();
    showToast(t('manual.enterMode'), 'info', 4000);
  } else {
    manualModeBtn.classList.remove('active-mode');
    manualCanvases.classList.add('hidden');
    manualStatus.classList.add('hidden');
    showToast(t('manual.exitMode'), 'info', 2000);
  }
};

clearManualBtn.onclick = () => {
  manualState.pairs = []; manualState.pendingReal = null;
  updateManualPairsTable(); redrawManual();
};

applyManualBtn.onclick = async () => {
  if (manualState.pairs.length === 0) return;
  try {
    const res = await fetch('/api/align/add-manual-landmarks', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pairs: manualState.pairs }),
    }).then(r => r.json());
    if (!res.ok) { showToast(t('manual.applyFail', { err: res.error }), 'error'); return; }
    showToast(t('manual.applyOk', { n: res.total_pairs }), 'success', 5000);
    // Auto-trigger re-alignment
    document.getElementById('aiAlignBtn').click();
  } catch { showToast(t('manual.applyFail', { err: '?' }), 'error'); }
};

// ================================================================
// OUTPUT FILE BROWSER
// ================================================================
async function refreshFileList() {
  const grid  = document.getElementById('outputFileGrid');
  const empty = document.getElementById('outputFileEmpty');
  try {
    const res = await fetch('/api/outputs/file-list').then(r => r.json());
    if (!res.ok || res.files.length === 0) { grid.innerHTML = ''; empty.classList.remove('hidden'); return; }
    empty.classList.add('hidden');
    grid.innerHTML = '';
    const ICONS = { '.png': '冒聼聳录', '.tif': '冒聼聰卢', '.tiff': '冒聼聰卢', '.csv': '冒聼聯聤', '.json': '冒聼聯聥', '.txt': '冒聼聯聞' };
    res.files.forEach(f => {
      const card = document.createElement('div');
      card.className = 'output-file-card';
      const icon = ICONS[f.ext] || '冒聼聯聛';
      const sizeStr = f.size > 1024*1024 ? `${(f.size/1024/1024).toFixed(1)} MB` : `${(f.size/1024).toFixed(0)} KB`;
      card.innerHTML = `<span class="file-icon">${icon}</span><span class="file-name" title="${f.name}">${f.name}</span><span class="file-size">${sizeStr}</span>`;
      card.onclick = () => handleOutputFileClick(f);
      grid.appendChild(card);
    });
  } catch {}
}

async function handleOutputFileClick(f) {
  if (f.ext === '.png') {
    openLightbox(`/api/outputs/named/${f.name}?${Date.now()}`, f.name);
  } else if (f.ext === '.csv' || f.ext === '.json' || f.ext === '.txt') {
    try {
      const text = await fetch(`/api/outputs/named/${f.name}`).then(r => r.text());
      document.getElementById('methodsText').textContent = text.slice(0, 8000) + (text.length > 8000 ? '\n...(truncated)' : '');
      document.getElementById('methodsModal').classList.remove('hidden');
    } catch {}
  }
}

document.getElementById('refreshFileListBtn').onclick = refreshFileList;
// Auto-refresh when switching to results tab
const _origResultsRefresh = refreshOutputs;
async function refreshOutputsAndFiles() {
  await _origResultsRefresh();
  await refreshFileList();
}
document.getElementById('refreshBtn').onclick = refreshOutputsAndFiles;

