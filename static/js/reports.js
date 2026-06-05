/**
 * reports.js — Report generation page
 * PROTACPred  |  SE3AF v3.9
 */
'use strict';

let _selectedIds = new Set();

function loadReports() {
  const container = document.getElementById('reports-list');
  if (!container) return;

  fetch('/history')
  .then(r => r.json())
  .then(data => {
    const preds = data.predictions || [];
    const historyTable = document.getElementById('history-report-table');
    if (historyTable) {
      historyTable.innerHTML = preds.length === 0
        ? '<tr><td colspan="5" class="table-empty">No predictions yet.</td></tr>'
        : preds.map((p, i) => `
          <tr>
            <td><input type="checkbox" onchange="toggleSelect('${p.id}')" ${_selectedIds.has(p.id) ? 'checked' : ''}></td>
            <td class="font-mono">${(p.protac_smiles || '').slice(0, 24)}…</td>
            <td><span class="status-badge ${p.label===1?'success':'danger'}">${p.prediction || (p.label===1?'Degrader':'Non-Degrader')}</span></td>
            <td>${p.degradation_likelihood?.toFixed(1) || '—'}%</td>
            <td style="color:var(--text-muted);font-size:11px">${p.date || '—'}</td>
          </tr>`).join('');
    }
  });

  // List existing report files
  fetch('/reports')
  .then(r => r.json())
  .then(data => {
    const reports = data.reports || [];
    container.innerHTML = reports.length === 0
      ? '<p class="text-muted" style="padding:20px">No reports generated yet.</p>'
      : reports.map(r => `
        <div class="card" style="margin-bottom:10px">
          <div style="padding:12px 16px;display:flex;align-items:center;justify-content:space-between">
            <div style="display:flex;align-items:center;gap:12px">
              <i class="fas ${r.type==='pdf'?'fa-file-pdf':'fa-file-csv'}" style="color:${r.type==='pdf'?'var(--danger)':'var(--success)'};font-size:20px"></i>
              <div>
                <div style="font-size:13px;font-weight:600">${r.name}</div>
                <div style="font-size:11px;color:var(--text-muted)">${(r.size/1024).toFixed(1)} KB · ${r.date || ''}</div>
              </div>
            </div>
            <a href="/download/${r.name}" class="btn btn-sm btn-secondary">
              <i class="fas fa-download"></i> Download
            </a>
          </div>
        </div>`).join('');
  })
  .catch(() => {
    if (container) container.innerHTML = '<p class="text-muted" style="padding:20px">No reports yet.</p>';
  });
}

window.toggleSelect = function (id) {
  if (_selectedIds.has(id)) _selectedIds.delete(id);
  else _selectedIds.add(id);
};

function generateReport(format) {
  const btn = document.getElementById('btn-gen-' + format);
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Generating…'; }

  const ids = _selectedIds.size > 0 ? Array.from(_selectedIds) : ['all'];
  fetch('/generate_report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prediction_ids: ids, format }),
  })
  .then(r => r.json())
  .then(data => {
    if (btn) { btn.disabled = false; btn.innerHTML = `<i class="fas fa-file-${format}"></i> ${format.toUpperCase()}`; }
    if (data.report_url) {
      window.location.href = data.report_url;
      showToast && showToast(`${format.toUpperCase()} report downloaded`, 'success');
    } else {
      alert('Report generated: ' + (data.message || 'Check reports directory'));
    }
    loadReports();
  })
  .catch(e => {
    if (btn) { btn.disabled = false; btn.innerHTML = `<i class="fas fa-file-${format}"></i> ${format.toUpperCase()}`; }
    alert('Error generating report: ' + e.message);
  });
}

document.addEventListener('DOMContentLoaded', function () {
  if (document.getElementById('reports-list')) loadReports();
  document.getElementById('btn-gen-pdf')?.addEventListener('click', () => generateReport('pdf'));
  document.getElementById('btn-gen-csv')?.addEventListener('click', () => generateReport('csv'));
  document.getElementById('btn-gen-json')?.addEventListener('click', () => generateReport('json'));
  document.getElementById('btn-select-all')?.addEventListener('click', function () {
    document.querySelectorAll('#history-report-table input[type=checkbox]').forEach(cb => {
      cb.checked = true;
      window.toggleSelect(cb.closest('tr')?.dataset?.id);
    });
  });
});
