/**
 * datasets.js — Dataset management page
 * PROTACPred  |  SE3AF v3.9
 */
'use strict';

function loadDatasets() {
  const container = document.getElementById('datasets-container');
  if (!container) return;
  container.innerHTML = '<div class="flex items-center gap-8"><span class="spinner"></span> Loading datasets…</div>';

  fetch('/datasets')
  .then(r => r.json())
  .then(data => {
    const { datasets = [], stats = {} } = data;
    // Stats
    document.getElementById('stat-total')?.setAttribute('data-val', stats.total || 0);
    document.getElementById('stat-positive')?.setAttribute('data-val', stats.positive || 0);
    document.getElementById('stat-negative')?.setAttribute('data-val', stats.negative || 0);
    _animateCounters();
    // Dataset cards
    container.innerHTML = datasets.length === 0
      ? '<p class="text-muted">No datasets found.</p>'
      : datasets.map(ds => _dsCard(ds)).join('');
  })
  .catch(e => {
    container.innerHTML = `<div class="alert alert-danger"><i class="fas fa-times-circle"></i> Error loading datasets: ${e.message}</div>`;
  });
}

function _dsCard(ds) {
  const pct = ds.n_rows ? ((ds.positive || 0) / ds.n_rows * 100).toFixed(1) : '—';
  return `
    <div class="card" style="margin-bottom:12px">
      <div class="card-header">
        <h3><i class="fas fa-database icon"></i>${ds.name}</h3>
        <span class="status-badge info">${ds.n_rows?.toLocaleString() || '—'} rows</span>
      </div>
      <div class="card-body">
        <div class="mol-detail-grid">
          <div class="mol-detail-item">
            <div class="mol-detail-key">Total Compounds</div>
            <div class="mol-detail-val">${ds.n_rows?.toLocaleString() || '—'}</div>
          </div>
          <div class="mol-detail-item">
            <div class="mol-detail-key">Active (Degraders)</div>
            <div class="mol-detail-val" style="color:var(--success)">${ds.positive?.toLocaleString() || '—'}</div>
          </div>
          <div class="mol-detail-item">
            <div class="mol-detail-key">Inactive</div>
            <div class="mol-detail-val" style="color:var(--danger)">${ds.negative?.toLocaleString() || '—'}</div>
          </div>
          <div class="mol-detail-item">
            <div class="mol-detail-key">Activity Rate</div>
            <div class="mol-detail-val">${pct}%</div>
          </div>
        </div>
        ${ds.n_rows ? `
        <div style="margin-top:12px">
          <div class="confidence-bar-label">
            <span>Active</span><span>Inactive</span>
          </div>
          <div class="confidence-bar-track">
            <div class="confidence-bar-fill" style="width:${pct}%;transition:width 1s ease"></div>
          </div>
        </div>` : ''}
        <div style="margin-top:12px;font-size:11px;color:var(--text-muted)">Path: ${ds.path || '—'}</div>
      </div>
    </div>`;
}

function _animateCounters() {
  document.querySelectorAll('[data-val]').forEach(el => {
    const target = parseInt(el.getAttribute('data-val') || '0', 10);
    let current = 0;
    const step = Math.max(1, Math.ceil(target / 40));
    const id = setInterval(() => {
      current = Math.min(current + step, target);
      el.textContent = current.toLocaleString();
      if (current >= target) clearInterval(id);
    }, 25);
  });
}

document.addEventListener('DOMContentLoaded', function () {
  if (document.getElementById('datasets-container')) loadDatasets();
});
