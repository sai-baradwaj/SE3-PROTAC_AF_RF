/**
 * viewer.js — 3Dmol.js wrapper for PROTACPred
 * Manages the interactive 3D molecular viewer
 */

'use strict';

// ── Viewer State ─────────────────────────────────────────────────
const VIEWER_STATE = {
  viewer:      null,
  viewMode:    'complex',
  styleMode:   'surface',
  structures:  {},   // { protac: sdfStr, target_pdb: pdbStr, ligase_pdb: pdbStr, ... }
  initialized: false,
};

const COLORS = {
  protac:     '#6366F1',
  target:     '#818CF8',
  ligase:     '#34D399',
  ternary:    '#F59E0B',
};

// ── Initialize Viewer ────────────────────────────────────────────
function initViewer() {
  const el = document.getElementById('mol-viewer');
  if (!el) return;
  if (typeof $3Dmol === 'undefined') {
    console.warn('[viewer] 3Dmol.js not loaded');
    return;
  }
  VIEWER_STATE.viewer = $3Dmol.createViewer(el, {
    backgroundColor: '#F8FAFC',
    antialias: true,
  });
  VIEWER_STATE.initialized = true;
  console.log('[viewer] 3Dmol viewer initialized');

  // Double-click to reset
  el.addEventListener('dblclick', viewer_reset);
}

// ── Load Structures ──────────────────────────────────────────────
function loadStructures(structures) {
  if (!structures || Object.keys(structures).length === 0) return;
  VIEWER_STATE.structures = structures;

  const pl = document.getElementById('viewer-placeholder');
  if (pl) pl.style.display = 'none';
  const loading = document.getElementById('viewer-loading');
  if (loading) loading.classList.remove('hidden');

  // Small delay to let loading overlay show
  setTimeout(() => {
    _renderViewer();
    if (loading) loading.classList.add('hidden');
  }, 100);
}

function _renderViewer() {
  const v = VIEWER_STATE.viewer;
  if (!v) { initViewer(); if (!VIEWER_STATE.viewer) return; }
  const viewer = VIEWER_STATE.viewer;

  viewer.clear();

  const structs = VIEWER_STATE.structures;
  const mode    = VIEWER_STATE.viewMode;

  // Determine which structures to show
  const showProtac  = mode === 'complex' || mode === 'protac';
  const showTarget  = mode === 'complex' || mode === 'target';
  const showLigase  = mode === 'complex' || mode === 'ligase';

  let hasContent = false;

  // Load target PDB
  if (showTarget && structs.target_pdb) {
    const m = viewer.addModel(structs.target_pdb, 'pdb');
    _applyStyle(viewer, m, 'target');
    hasContent = true;
  }

  // Load ligase PDB
  if (showLigase && structs.ligase_pdb) {
    const m = viewer.addModel(structs.ligase_pdb, 'pdb');
    _applyStyle(viewer, m, 'ligase');
    hasContent = true;
  }

  // Load PROTAC SDF
  if (showProtac && structs.protac) {
    const m = viewer.addModel(structs.protac, 'sdf');
    _applyStyle(viewer, m, 'protac');
    hasContent = true;
  }

  if (hasContent) {
    viewer.zoomTo();
    viewer.render();
  }
}

function _applyStyle(viewer, model, component) {
  const style   = VIEWER_STATE.styleMode;
  const color   = COLORS[component] || '#888888';
  const hex     = parseInt(color.replace('#',''), 16);

  // Clear existing style
  model.setStyle({}, {});

  if (component === 'protac' || component === 'warhead' || component === 'linker') {
    // Small molecules always use stick or ball-stick
    if (style === 'surface') {
      model.setStyle({}, { stick: { radius: 0.12, colorscheme: { prop: 'elem', map: { C: color } } } });
    } else if (style === 'ballstick') {
      model.setStyle({}, { sphere: { radius: 0.3, color: color }, stick: { radius: 0.12 } });
    } else {
      model.setStyle({}, { stick: { radius: 0.12, color: color } });
    }
  } else {
    // Proteins
    switch (style) {
      case 'surface':
        model.setStyle({}, { cartoon: { color: color, opacity: 0.3 } });
        viewer.addSurface($3Dmol.SurfaceType.VDW, {
          opacity: 0.75,
          colorscheme: { prop: 'b', gradient: 'roygb', min: 50, max: 90 },
        }, { model: model });
        break;
      case 'cartoon':
        model.setStyle({}, { cartoon: { color: color } });
        break;
      case 'stick':
        model.setStyle({}, { stick: { radius: 0.08, color: color } });
        break;
      case 'ballstick':
        model.setStyle({}, { sphere: { radius: 0.2, color: color }, stick: { radius: 0.1 } });
        break;
      default:
        model.setStyle({}, { cartoon: { color: color } });
    }
  }
}

// ── View Mode ────────────────────────────────────────────────────
function setViewMode(mode) {
  VIEWER_STATE.viewMode = mode;
  _renderViewer();
}

// ── Style Mode ───────────────────────────────────────────────────
function setStyleMode(style) {
  VIEWER_STATE.styleMode = style;
  _renderViewer();
}

// ── Controls ─────────────────────────────────────────────────────
function viewer_reset() {
  if (VIEWER_STATE.viewer) {
    VIEWER_STATE.viewer.zoomTo();
    VIEWER_STATE.viewer.render();
  }
}

function viewer_zoom(factor) {
  if (VIEWER_STATE.viewer) {
    VIEWER_STATE.viewer.zoom(factor, 500);
    VIEWER_STATE.viewer.render();
  }
}

function viewer_fullscreen() {
  const container = document.getElementById('viewer-container');
  if (!container) return;
  if (!document.fullscreenElement) {
    container.requestFullscreen().then(() => {
      setTimeout(() => {
        if (VIEWER_STATE.viewer) {
          VIEWER_STATE.viewer.resize();
          VIEWER_STATE.viewer.render();
        }
      }, 200);
    }).catch(err => console.warn('Fullscreen error:', err));
  } else {
    document.exitFullscreen();
  }
}

// ── Load single SMILES as SDF ────────────────────────────────────
async function loadSmiles(smiles, component) {
  if (!smiles) return;
  try {
    const res  = await fetch('/load_structure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles, type: component }),
    });
    const data = await res.json();
    if (data.sdf) {
      VIEWER_STATE.structures[component] = data.sdf;
      const pl = document.getElementById('viewer-placeholder');
      if (pl) pl.style.display = 'none';
      _renderViewer();
      return data;
    }
  } catch (e) {
    console.error('[viewer] loadSmiles error:', e);
  }
  return null;
}

// ── Load PDB text ─────────────────────────────────────────────────
function loadPdb(pdbText, component) {
  if (!pdbText) return;
  VIEWER_STATE.structures[component] = pdbText;
  const pl = document.getElementById('viewer-placeholder');
  if (pl) pl.style.display = 'none';
  _renderViewer();
}

// ── Clear viewer ──────────────────────────────────────────────────
function clearViewer() {
  VIEWER_STATE.structures = {};
  if (VIEWER_STATE.viewer) {
    VIEWER_STATE.viewer.clear();
    VIEWER_STATE.viewer.render();
  }
  const pl = document.getElementById('viewer-placeholder');
  if (pl) pl.style.display = '';
}

// ── Auto-init on page load ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Wait a tick for 3Dmol CDN to load
  setTimeout(initViewer, 500);
});
