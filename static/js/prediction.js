/**
 * prediction.js — PROTACPred SE3AF v3.9 Frontend Logic
 * Handles: prediction form, results display, AlphaFold fetch,
 *          batch upload, compound explorer, model loading, toasts
 */

'use strict';

// ── State ────────────────────────────────────────────────────────
const APP = {
  lastResult:     null,
  batchResults:   [],
  explorerData:   [],
  afData:         { target: null, ligase: null },
  modelLoaded:    false,
};

// ── Toast ─────────────────────────────────────────────────────────
function showToast(msg, type, duration) {
  type = type || 'info';
  duration = duration || 3500;
  const container = document.getElementById('toast-container');
  if (!container) return;
  const t = document.createElement('div');
  const icons = { success: 'check-circle', error: 'xmark-circle', warning: 'exclamation-triangle', info: 'circle-info' };
  t.className = 'toast' + (type !== 'info' ? ' toast-' + type : '');
  t.innerHTML = '<i class="fas fa-' + (icons[type] || 'circle-info') + '"></i> ' + msg;
  container.appendChild(t);
  setTimeout(function() {
    t.style.opacity = '0';
    t.style.transition = 'opacity .3s';
    setTimeout(function() { t.remove(); }, 300);
  }, duration);
}

// ── Tab switching ─────────────────────────────────────────────────
function switchInputTab(name, btn) {
  document.querySelectorAll('.tab-panel').forEach(function(p) { p.classList.remove('active'); });
  document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
  var panel = document.getElementById('tab-' + name);
  if (panel) panel.classList.add('active');
  if (btn) btn.classList.add('active');
}

// ── View mode / style mode ────────────────────────────────────────
function setViewMode(mode, btn) {
  document.querySelectorAll('.view-tab').forEach(function(b) { b.classList.remove('active'); });
  if (btn) btn.classList.add('active');
  if (typeof VIEWER_STATE !== 'undefined') {
    VIEWER_STATE.viewMode = mode;
    if (typeof _renderViewer === 'function') _renderViewer();
  }
}

function setStyleMode(style, btn) {
  document.querySelectorAll('.style-btn').forEach(function(b) { b.classList.remove('active'); });
  if (btn) btn.classList.add('active');
  if (typeof VIEWER_STATE !== 'undefined') {
    VIEWER_STATE.styleMode = style;
    if (typeof _renderViewer === 'function') _renderViewer();
  }
}

// ── New prediction ────────────────────────────────────────────────
function newPrediction() {
  ['protac-smiles','target-uniprot','target-seq','ligase-uniprot','e3-seq',
   'warhead-smiles','linker-smiles','e3-smiles'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) el.value = '';
  });
  ['dot-protac','dot-target','dot-ligase'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) el.classList.add('empty');
  });
  var rc = document.getElementById('result-content');
  var re = document.getElementById('result-empty');
  if (rc) rc.classList.add('hidden');
  if (re) re.classList.remove('hidden');
  if (typeof clearViewer === 'function') clearViewer();
  ['md-mw','md-atoms','md-formula','md-tpsa','md-logp','md-rotbonds',
   'md-target-pdb','md-target-res','md-target-uniprot','md-target-plddt','md-target-conf',
   'md-ligase-pdb','md-ligase-res','md-ligase-uniprot','md-ligase-plddt'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) el.textContent = '—';
  });
  APP.lastResult = null;
  APP.afData = { target: null, ligase: null };
}

// ── SMILES change handler ─────────────────────────────────────────
function onSmilesChange(type) {
  var elId = type === 'protac' ? 'protac-smiles' : type + '-smiles';
  var el = document.getElementById(elId);
  if (!el) return;
  var dot = document.getElementById('dot-' + type);
  if (dot) dot.classList.toggle('empty', !el.value.trim());
}

// ── UniProt change — trigger autofetch after 800ms ────────────────
var _afTimers = {};
function onUniprotChange(type) {
  var id = type === 'target' ? 'target-uniprot' : 'ligase-uniprot';
  var el = document.getElementById(id);
  var val = el ? el.value.trim().toUpperCase() : '';
  clearTimeout(_afTimers[type]);
  if (val.length >= 6) {
    _afTimers[type] = setTimeout(function() { fetchAlphaFold(type); }, 800);
  }
}

// ── Fetch AlphaFold ───────────────────────────────────────────────
function fetchAlphaFold(type) {
  var id = type === 'target' ? 'target-uniprot' : 'ligase-uniprot';
  var el = document.getElementById(id);
  var uid = el ? el.value.trim().toUpperCase() : '';
  if (!uid) { showToast('Please enter a UniProt ID', 'warning'); return; }

  showToast('Fetching AlphaFold for ' + uid + '…', 'info', 5000);

  fetch('/alphafold', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uniprot_id: uid })
  })
  .then(function(res) { return res.json().then(function(d) { return { ok: res.ok, data: d }; }); })
  .then(function(r) {
    if (!r.ok || r.data.error) {
      showToast('AlphaFold not found for ' + uid + ': ' + (r.data.error || ''), 'error');
      return;
    }
    APP.afData[type] = r.data;
    var component = type === 'target' ? 'target_pdb' : 'ligase_pdb';
    if (typeof loadPdb === 'function') loadPdb(r.data.pdb, component);
    _updateProteinInfo(type, uid, r.data.pdb_info || {});
    showToast('AlphaFold loaded for ' + uid + ' (' + r.data.source + ')', 'success');
    var preview = document.getElementById('preview-' + type);
    if (preview) {
      var info = r.data.pdb_info || {};
      preview.innerHTML = '<div style="text-align:center;padding:8px">' +
        '<i class="fas fa-check-circle" style="color:var(--success);font-size:20px"></i>' +
        '<p style="font-size:11px;margin-top:4px;color:var(--text-muted)">' + uid + ' loaded</p>' +
        '<p style="font-size:10px;color:var(--text-muted)">' + (info.residues || '?') + ' residues · pLDDT ' + (info.avg_plddt || '?') + '</p>' +
        '</div>';
    }
  })
  .catch(function(e) { showToast('AlphaFold fetch failed: ' + e.message, 'error'); });
}

function viewProtein(type) {
  if (!APP.afData[type]) { fetchAlphaFold(type); return; }
  var component = type === 'target' ? 'target_pdb' : 'ligase_pdb';
  if (typeof loadPdb === 'function') loadPdb(APP.afData[type].pdb, component);
}

function _updateProteinInfo(type, uid, info) {
  var prefix = 'prot-' + type;
  _setText(prefix + '-uniprot', uid);
  _setText(prefix + '-af',      uid ? 'AF-' + uid + '-F1' : '—');
  _setText(prefix + '-res',     info.residues != null ? info.residues : '—');
  _setText(prefix + '-plddt',   info.avg_plddt != null ? info.avg_plddt : '—');
  if (type === 'target') {
    _setText('md-target-uniprot', uid);
    _setText('md-target-res',  info.residues != null ? info.residues : '—');
    _setText('md-target-plddt',info.avg_plddt != null ? info.avg_plddt + '%' : '—');
    _setText('md-target-conf', info.plddt_coverage != null ? info.plddt_coverage + '%' : '—');
    _setText('md-target-pdb',  uid ? 'AF-' + uid : '—');
  } else {
    _setText('md-ligase-uniprot', uid);
    _setText('md-ligase-res',  info.residues != null ? info.residues : '—');
    _setText('md-ligase-plddt',info.avg_plddt != null ? info.avg_plddt + '%' : '—');
    _setText('md-ligase-pdb',  uid ? 'AF-' + uid : '—');
  }
}

// ── View 3D SMILES ────────────────────────────────────────────────
function viewSmiles3D(type) {
  var el = document.getElementById('protac-smiles');
  var smiles = el ? el.value.trim() : '';
  if (!smiles) { showToast('Please enter a PROTAC SMILES first', 'warning'); return; }
  showToast('Generating 3D structure…', 'info', 2000);
  if (typeof loadSmiles === 'function') {
    loadSmiles(smiles, 'protac').then(function(data) {
      if (data && data.sdf) {
        showToast('3D structure loaded', 'success');
        _updateMolDetails(data.mol_info);
      } else {
        showToast('Could not generate 3D for this SMILES', 'error');
      }
    });
  }
}

// ── Run Prediction ────────────────────────────────────────────────
function runPrediction() {
  var el = document.getElementById('protac-smiles');
  var protac = el ? el.value.trim() : '';
  if (!protac) { showToast('Please enter a PROTAC SMILES', 'warning'); return; }

  var btn = document.getElementById('btn-predict');
  var progress = document.getElementById('predict-progress');
  if (btn) { btn.disabled = true; btn.innerHTML = '<div class="spinner"></div> Running…'; }
  if (progress) progress.classList.remove('hidden');

  var pct = 0;
  var bar = document.getElementById('pred-bar');
  var pctEl = document.getElementById('pred-pct');
  var timer = setInterval(function() {
    pct = Math.min(pct + Math.random() * 12, 88);
    if (bar) bar.style.width = pct + '%';
    if (pctEl) pctEl.textContent = Math.round(pct) + '%';
  }, 350);

  var _g = function(id) { var e = document.getElementById(id); return e ? e.value.trim() : ''; };

  var payload = {
    protac_smiles:    protac,
    target_seq:       _g('target-seq'),
    e3_seq:           _g('e3-seq'),
    warhead_smiles:   _g('warhead-smiles'),
    linker_smiles:    _g('linker-smiles'),
    e3_ligase_smiles: _g('e3-smiles'),
    target_uniprot:   _g('target-uniprot').toUpperCase(),
    ligase_uniprot:   _g('ligase-uniprot').toUpperCase()
  };

  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  .then(function(res) { return res.json(); })
  .then(function(data) {
    clearInterval(timer);
    if (bar) bar.style.width = '100%';
    if (pctEl) pctEl.textContent = '100%';
    setTimeout(function() {
      if (progress) progress.classList.add('hidden');
      renderPredictionResult(data);
    }, 400);
    APP.lastResult = data;
    updateStatsBar();
  })
  .catch(function(e) {
    clearInterval(timer);
    if (progress) progress.classList.add('hidden');
    showToast('Prediction failed: ' + e.message, 'error');
  })
  .finally(function() {
    if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-bolt"></i> Run Prediction'; }
  });
}

// ── Render Result ─────────────────────────────────────────────────
function renderPredictionResult(data) {
  var rc = document.getElementById('result-content');
  var re = document.getElementById('result-empty');
  if (rc) rc.classList.remove('hidden');
  if (re) re.classList.add('hidden');

  var score = data.degradation_likelihood || 0;
  var conf  = data.confidence || 0;

  var scoreEl = document.getElementById('res-score');
  if (scoreEl) {
    scoreEl.textContent = score.toFixed(1) + '%';
    scoreEl.className = 'pred-score-big ' + (score >= 70 ? 'score-high' : score >= 40 ? 'score-mid' : 'score-low');
  }

  var ptr = document.getElementById('score-pointer');
  if (ptr) ptr.style.left = Math.max(2, Math.min(98, score)) + '%';

  var badge    = document.getElementById('res-badge');
  var badgeTxt = document.getElementById('res-badge-text');
  if (badge && badgeTxt) {
    if (data.error && !data.degradation_likelihood) {
      badge.className = 'pred-badge badge-unknown';
      badgeTxt.textContent = 'Model Not Loaded';
    } else if (score >= 50) {
      badge.className = 'pred-badge badge-active';
      badgeTxt.textContent = 'Likely to Degrade';
    } else {
      badge.className = 'pred-badge badge-inactive';
      badgeTxt.textContent = 'Unlikely to Degrade';
    }
  }

  _setText('res-confidence', conf.toFixed(1) + '%');
  _setText('res-method',     data.method || '—');
  _setText('res-stability',  data.stability_score != null ? (data.stability_score * 100).toFixed(1) + '%' : '—');
  _setText('res-interaction',data.interaction_score != null ? (data.interaction_score * 100).toFixed(1) + '%' : '—');

  var confBar = document.getElementById('conf-bar');
  if (confBar) confBar.style.width = conf + '%';

  _setText('int-target',  data.target_protac_interactions != null ? data.target_protac_interactions : '—');
  _setText('int-ligase',  data.ligase_protac_interactions != null ? data.ligase_protac_interactions : '—');
  _setText('int-ternary', data.ternary_contacts != null ? data.ternary_contacts : '—');

  var expTxt = document.getElementById('explanation-text');
  if (expTxt) {
    if (data.error) {
      expTxt.textContent = '\u26a0 ' + data.error;
    } else if (score >= 70) {
      expTxt.textContent = 'The PROTAC exhibits favorable binding conformations and proximity between the target protein and E3 ligase, suggesting efficient ternary complex formation and potential degradation. SE3AF v3.9 confidence: ' + conf.toFixed(1) + '%.';
    } else if (score >= 40) {
      expTxt.textContent = 'Moderate degradation potential detected. The ternary complex geometry is plausible but may require optimization of the linker length or warhead binding affinity.';
    } else {
      expTxt.textContent = 'Low degradation likelihood predicted. The PROTAC may have insufficient binding affinity or unfavorable ternary complex geometry. Consider modifying the linker or warhead.';
    }
  }

  if (data.mol_info && Object.keys(data.mol_info).length > 0) _updateMolDetails(data.mol_info);

  if (data.structures && Object.keys(data.structures).length > 0) {
    if (typeof loadStructures === 'function') loadStructures(data.structures);
  } else {
    var sm = document.getElementById('protac-smiles');
    if (sm && sm.value.trim() && typeof loadSmiles === 'function') {
      loadSmiles(sm.value.trim(), 'protac');
    }
  }

  if (data.target_pdb_info && Object.keys(data.target_pdb_info).length > 0) {
    var tu = document.getElementById('target-uniprot');
    _updateProteinInfo('target', tu ? tu.value.trim() : '—', data.target_pdb_info);
  }
  if (data.ligase_pdb_info && Object.keys(data.ligase_pdb_info).length > 0) {
    var lu = document.getElementById('ligase-uniprot');
    _updateProteinInfo('ligase', lu ? lu.value.trim() : '—', data.ligase_pdb_info);
  }

  if (!data.error) {
    showToast('Prediction complete: ' + score.toFixed(1) + '% degradation likelihood', 'success');
  } else {
    showToast('Result returned (model not loaded — real inference unavailable)', 'warning');
  }
}

// ── Molecular Details ─────────────────────────────────────────────
function _updateMolDetails(info) {
  if (!info) return;
  _setText('md-mw',       info.mw != null ? info.mw + ' g/mol' : '—');
  _setText('md-atoms',    info.heavy_atoms != null ? info.heavy_atoms : (info.atoms != null ? info.atoms : '—'));
  _setText('md-formula',  info.formula  || '—');
  _setText('md-tpsa',     info.tpsa     != null ? info.tpsa + ' Å²' : '—');
  _setText('md-logp',     info.logp     != null ? info.logp : '—');
  _setText('md-rotbonds', info.rotatable_bonds != null ? info.rotatable_bonds : '—');
}

// ── Stats Bar ─────────────────────────────────────────────────────
function updateStatsBar() {
  fetch('/history')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      var preds = data.predictions || [];
      _setText('stat-total-pred', preds.length);
      _setText('stat-degraders', preds.filter(function(p) { return p.label === 1; }).length);
    })
    .catch(function() {});
}

// ── Load Model ────────────────────────────────────────────────────
function loadModel() {
  var btn = document.getElementById('btn-load-model');
  if (btn) { btn.disabled = true; btn.innerHTML = '<div class="spinner"></div> Loading…'; }
  showToast('Loading SE3AF v3.9 model…', 'info', 5000);

  fetch('/api/load_model', { method: 'POST' })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.success) {
        showToast(data.message || 'Model loading started', 'success');
        _pollModelStatus();
      } else {
        showToast(data.message || 'Model load failed', 'error');
      }
    })
    .catch(function(e) { showToast('Load model error: ' + e.message, 'error'); })
    .finally(function() {
      if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-play"></i> Load Model'; }
    });
}

function _pollModelStatus() {
  var badge    = document.getElementById('model-status-badge');
  var badgeTxt = document.getElementById('model-status-text');
  var polls = 0;
  var interval = setInterval(function() {
    polls++;
    fetch('/api/health')
      .then(function(r) { return r.json(); })
      .then(function(data) {
        APP.modelLoaded = data.model_loaded;
        if (!badge || !badgeTxt) return;
        if (data.model_loaded) {
          badge.className = 'model-status-inline loaded';
          badgeTxt.textContent = 'SE3AF Loaded';
          var dot = badge.querySelector('.status-dot');
          if (dot) dot.classList.remove('pulse');
          clearInterval(interval);
          showToast('SE3AF v3.9 model loaded!', 'success');
        } else if (data.model_loading) {
          badge.className = 'model-status-inline loading';
          badgeTxt.textContent = 'Loading…';
          var dot2 = badge.querySelector('.status-dot');
          if (dot2) dot2.classList.add('pulse');
        } else {
          badge.className = 'model-status-inline unloaded';
          badgeTxt.textContent = 'Not Loaded';
        }
      })
      .catch(function() {});
    if (polls >= 60) clearInterval(interval);
  }, 2000);
}

// ── Random Sample ─────────────────────────────────────────────────
function loadRandomSample() {
  fetch('/api/random_sample', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source: 'test' })
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) { showToast(data.error, 'error'); return; }
    _setVal('protac-smiles',  data.protac_smiles || data.smiles || '');
    _setVal('warhead-smiles', data.warhead_smiles || '');
    _setVal('linker-smiles',  data.linker_smiles || '');
    _setVal('e3-smiles',      data.e3_ligase_smiles || '');
    _setVal('target-seq',     data.target_seq || '');
    _setVal('e3-seq',         data.e3_seq || '');
    _setVal('target-uniprot', data.target_uniprot || '');
    _setVal('ligase-uniprot', data.ligase_uniprot || '');
    var dot = document.getElementById('dot-protac');
    if (dot && (data.protac_smiles || data.smiles)) dot.classList.remove('empty');
    showToast('Random sample loaded from test set', 'success');
    var smiles = data.protac_smiles || data.smiles || '';
    if (smiles && typeof loadSmiles === 'function') loadSmiles(smiles, 'protac');
    if (data.target_uniprot) setTimeout(function() { fetchAlphaFold('target'); }, 500);
    if (data.ligase_uniprot) setTimeout(function() { fetchAlphaFold('ligase'); }, 1200);
  })
  .catch(function(e) { showToast('Error loading random sample: ' + e.message, 'error'); });
}

// ── Batch Upload ─────────────────────────────────────────────────
function handleBatchDrop(event) {
  event.preventDefault();
  var drop = document.getElementById('batch-drop');
  if (drop) drop.classList.remove('drag-over');
  var files = event.dataTransfer && event.dataTransfer.files;
  if (files && files[0]) processBatchFile(files[0]);
}

function handleBatchFile(input) {
  if (input.files && input.files[0]) processBatchFile(input.files[0]);
}

function processBatchFile(file) {
  if (!file.name.endsWith('.csv')) { showToast('Only CSV files accepted', 'error'); return; }

  var progress  = document.getElementById('batch-progress');
  var bar       = document.getElementById('batch-bar');
  var pctEl     = document.getElementById('batch-pct');
  var statusTxt = document.getElementById('batch-status-txt');

  if (progress) progress.classList.remove('hidden');
  if (bar) bar.style.width = '10%';
  if (statusTxt) statusTxt.textContent = 'Uploading…';

  var formData = new FormData();
  formData.append('file', file);

  var uploadPct = 10;
  var timer = setInterval(function() {
    uploadPct = Math.min(uploadPct + 8, 80);
    if (bar) bar.style.width = uploadPct + '%';
    if (pctEl) pctEl.textContent = uploadPct + '%';
  }, 400);

  if (statusTxt) statusTxt.textContent = 'Running predictions…';

  fetch('/batch_predict', { method: 'POST', body: formData })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      clearInterval(timer);
      if (bar) bar.style.width = '100%';
      if (pctEl) pctEl.textContent = '100%';
      if (data.error) { showToast(data.error, 'error'); return; }
      APP.batchResults = data.predictions || [];
      _renderBatchTable(APP.batchResults);
      _renderExplorer(APP.batchResults);
      showToast('Batch complete: ' + data.count + ' predictions', 'success');
      updateStatsBar();
    })
    .catch(function(e) { showToast('Batch failed: ' + e.message, 'error'); })
    .finally(function() {
      clearInterval(timer);
      setTimeout(function() { if (progress) progress.classList.add('hidden'); }, 1000);
    });
}

function _renderBatchTable(predictions) {
  var results = document.getElementById('batch-results');
  var tbody   = document.getElementById('batch-tbody');
  if (!results || !tbody) return;
  tbody.innerHTML = predictions.map(function(p, i) {
    var score = (p.degradation_likelihood || 0).toFixed(1);
    var cls   = p.label === 1 ? 'badge-green' : 'badge-red';
    var lbl   = p.label === 1 ? 'Degrader' : 'Non-Degrader';
    var sm    = (p.protac_smiles || '').substring(0, 24) + ((p.protac_smiles || '').length > 24 ? '…' : '');
    return '<tr onclick="selectBatchCompound(' + i + ')" title="' + (p.protac_smiles || '') + '">' +
      '<td>' + (i + 1) + '</td>' +
      '<td style="font-family:monospace;font-size:11px">' + sm + '</td>' +
      '<td><strong>' + score + '%</strong></td>' +
      '<td><span class="badge ' + cls + '">' + lbl + '</span></td>' +
      '</tr>';
  }).join('');
  results.classList.remove('hidden');
}

function _renderExplorer(predictions) {
  var empty = document.getElementById('explorer-empty');
  var wrap  = document.getElementById('explorer-wrap');
  var tbody = document.getElementById('explorer-tbody');
  if (!tbody) return;
  APP.explorerData = predictions;
  tbody.innerHTML = predictions.map(function(p, i) {
    var score = (p.degradation_likelihood || 0).toFixed(1);
    var conf  = (p.confidence || 0).toFixed(0);
    var sm    = (p.protac_smiles || '').substring(0, 18) + ((p.protac_smiles || '').length > 18 ? '…' : '');
    var badge = p.label === 1
      ? '<span class="badge badge-green">Active</span>'
      : '<span class="badge badge-red">Inactive</span>';
    return '<tr onclick="selectExplorerCompound(' + i + ')" class="' + (i === 0 ? 'selected' : '') + '">' +
      '<td style="font-family:monospace;font-size:11px" title="' + (p.protac_smiles || '') + '">' + sm + '</td>' +
      '<td><strong>' + score + '%</strong></td>' +
      '<td>' + conf + '%</td>' +
      '<td>' + badge + '</td>' +
      '</tr>';
  }).join('');
  if (empty) empty.classList.add('hidden');
  if (wrap)  wrap.classList.remove('hidden');
  switchInputTab('explore', document.querySelectorAll('.tab-btn')[2]);
}

function selectBatchCompound(idx) {
  var p = APP.batchResults[idx];
  if (!p) return;
  renderPredictionResult(p);
  document.querySelectorAll('#batch-tbody tr').forEach(function(r, i) {
    r.style.background = i === idx ? 'var(--primary-light)' : '';
  });
}

function selectExplorerCompound(idx) {
  var p = APP.explorerData[idx];
  if (!p) return;
  renderPredictionResult(p);
  var smiles = p.protac_smiles || '';
  if (smiles && typeof loadSmiles === 'function') loadSmiles(smiles, 'protac');
  document.querySelectorAll('#explorer-tbody tr').forEach(function(r, i) {
    r.classList.toggle('selected', i === idx);
  });
}

// ── Export Batch CSV ─────────────────────────────────────────────
function exportBatchCSV() {
  if (!APP.batchResults.length) { showToast('No batch results to export', 'warning'); return; }
  var keys = ['protac_smiles','prediction','degradation_likelihood','confidence','method','date'];
  var rows = [keys.join(',')];
  APP.batchResults.forEach(function(p) {
    rows.push(keys.map(function(k) { return JSON.stringify(p[k] != null ? p[k] : ''); }).join(','));
  });
  var blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  var url  = URL.createObjectURL(blob);
  var a    = document.createElement('a');
  a.href = url; a.download = 'batch_predictions.csv'; a.click();
  URL.revokeObjectURL(url);
  showToast('CSV exported', 'success');
}

// ── Download Report ──────────────────────────────────────────────
function downloadReport() {
  showToast('Generating report…', 'info', 3000);
  var ids = APP.lastResult ? [APP.lastResult.prediction_id] : ['all'];
  fetch('/generate_report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ format: 'pdf', prediction_ids: ids })
  })
  .then(function(r) {
    if (!r.ok) { showToast('Report generation failed', 'error'); return null; }
    return r.blob();
  })
  .then(function(blob) {
    if (!blob) return;
    var url = URL.createObjectURL(blob);
    var a   = document.createElement('a');
    a.href = url; a.download = 'protacpred_report.pdf'; a.click();
    URL.revokeObjectURL(url);
    showToast('Report downloaded', 'success');
  })
  .catch(function(e) { showToast('Download error: ' + e.message, 'error'); });
}

// ── Utilities ─────────────────────────────────────────────────────
function _setText(id, val) {
  var el = document.getElementById(id);
  if (el) el.textContent = (val != null ? val : '—');
}
function _setVal(id, val) {
  var el = document.getElementById(id);
  if (el) el.value = val || '';
}

// ── Init ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  updateStatsBar();
  _pollModelStatus();
});
