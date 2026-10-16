// GNM comparison viewer: capture photos, calibrated cameras, scan, fitted GNM,
// projected texture. Data: build_viewer_data.py; served by serve.py.
// Everything is in the scan frame (metres); calibrated cameras are OpenCV
// (x right, y down, z forward), pixel u = f*x/z + w/2 + cx, pixel i spans [i,i+1].
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const DATA = (new URLSearchParams(location.search).get('data') || 'data/').replace(/\/?$/, '/');
const $ = (id) => document.getElementById(id);
const status = (t) => { $('status').textContent = t; };

const S = {                       // UI state
  view: 0, camMode: 'calib', layout: 'sbs', link: true,
  cols: { photo: true, scan: true, gnm: true },
  op: { photo: 1, scan: 0, gnm: 0.5 },
  pose: 'capture', gnmMode: 'tex', gnmWire: false, scanMode: 'smooth', scanWire: false,
  errRange: 3, errOut: 15,
  ov: { det: true, gnm: true, res: true, jaw: false, silp: false, sils: false, silg: false, all: true },
  crop: null,                     // {x0, y0, cw, ch} in image pixels
};
let M;                            // manifest
const photos = {}, overlays = {}, poseCache = {};
let panels = [];
let freeCam = null, freeTarget = new THREE.Vector3(), headCenter = new THREE.Vector3();
const unlinkedCams = new Map();

// ---------------------------------------------------------------- loading
async function bin(name, Type) { const r = await fetch(DATA + name); if (!r.ok) throw new Error(name + ' ' + r.status); return new Type(await r.arrayBuffer()); }
async function json(name) { const r = await fetch(DATA + name); if (!r.ok) throw new Error(name + ' ' + r.status); return r.json(); }
function loadImage(src) { return new Promise((ok, err) => { const i = new Image(); i.onload = () => ok(i); i.onerror = err; i.src = src; }); }
async function getPhoto(k) { if (!photos[k]) photos[k] = loadImage(DATA + M.views[k].image); return photos[k]; }
async function getOverlay(k) { if (!overlays[k]) overlays[k] = json('overlays/' + M.views[k].label + '.json'); return overlays[k]; }

// ---------------------------------------------------------------- colour maps
function divergingColor(d, range, out) {          // signed error -> blue/white/red; grey beyond `out`
  if (!Number.isFinite(d)) return [0.35, 0.35, 0.38];
  if (Math.abs(d) > out) return [0.55, 0.55, 0.55];
  const x = Math.max(-1, Math.min(1, d / range));
  const a = Math.abs(x);
  return x >= 0 ? [1, 1 - 0.85 * a, 1 - 0.95 * a] : [1 - 0.95 * a, 1 - 0.6 * a, 1];
}
const SRGB = (c) => new THREE.Color().setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);

// ---------------------------------------------------------------- scenes
const scan = { scene: new THREE.Scene() }, gnm = { scene: new THREE.Scene() };
for (const s of [scan, gnm]) {
  s.light = new THREE.DirectionalLight(0xffffff, 2.2);
  s.scene.add(s.light, s.light.target, new THREE.AmbientLight(0xffffff, 0.55));
}
const clay = () => new THREE.MeshStandardMaterial({ color: 0xc9b8a6, roughness: 0.75, metalness: 0, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });

function buildScan(pos, idx, err) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeVertexNormals();
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(pos.length), 3));
  scan.geo = g; scan.err = err;
  scan.mats = {
    smooth: clay(), flat: Object.assign(clay(), { flatShading: true }),
    normals: new THREE.MeshNormalMaterial({ polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }),
    error: new THREE.MeshBasicMaterial({ vertexColors: true, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }),
  };
  scan.mesh = new THREE.Mesh(g, scan.mats.smooth);
  scan.wire = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x1b1b1b, wireframe: true, transparent: true, opacity: 0.35 }));
  scan.scene.add(scan.mesh, scan.wire);
}

function buildGnm(idx, uv, mat) {
  // non-indexed (per-corner UVs = GNM's native layout), triangles grouped by material
  const T = idx.length / 3, nm = M.gnm.materials.length;
  const order = Array.from({ length: T }, (_, i) => i).sort((a, b) => mat[a] - mat[b] || a - b);
  const corner = new Uint32Array(T * 3), uvs = new Float32Array(T * 6);
  const g = new THREE.BufferGeometry();
  let start = 0;
  for (let k = 0; k < T; k++) {
    const t = order[k];
    for (let c = 0; c < 3; c++) { corner[3 * k + c] = idx[3 * t + c]; uvs[6 * k + 2 * c] = uv[6 * t + 2 * c]; uvs[6 * k + 2 * c + 1] = uv[6 * t + 2 * c + 1]; }
    if (k === T - 1 || mat[order[k + 1]] !== mat[t]) { g.addGroup(start * 3, (k + 1 - start) * 3, mat[t]); start = k + 1; }
  }
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(T * 9), 3));
  g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(T * 9), 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(T * 9), 3));
  gnm.geo = g; gnm.corner = corner; gnm.idx = idx; gnm.nm = nm;
  gnm.mesh = new THREE.Mesh(g, []);
  const wm = M.gnm.materials.map((n) => new THREE.MeshBasicMaterial({ color: 0x101010, wireframe: true, transparent: true, opacity: 0.45 }));
  gnm.wire = new THREE.Mesh(g, wm);
  gnm.scene.add(gnm.mesh, gnm.wire);
}

async function setPose(name) {
  if (!poseCache[name]) poseCache[name] = bin(M.poses.find((p) => p.name === name).file, Float32Array);
  const V = await poseCache[name];
  const nV = V.length / 3, vn = new Float32Array(V.length), idx = gnm.idx;
  for (let t = 0; t < idx.length; t += 3) {                     // smooth vertex normals on GNM's own topology
    const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
    const ux = V[b] - V[a], uy = V[b + 1] - V[a + 1], uz = V[b + 2] - V[a + 2];
    const wx = V[c] - V[a], wy = V[c + 1] - V[a + 1], wz = V[c + 2] - V[a + 2];
    const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    for (const p of [a, b, c]) { vn[p] += nx; vn[p + 1] += ny; vn[p + 2] += nz; }
  }
  for (let i = 0; i < nV; i++) { const l = Math.hypot(vn[3 * i], vn[3 * i + 1], vn[3 * i + 2]) || 1; vn[3 * i] /= l; vn[3 * i + 1] /= l; vn[3 * i + 2] /= l; }
  const P = gnm.geo.attributes.position.array, N = gnm.geo.attributes.normal.array, cr = gnm.corner;
  for (let k = 0; k < cr.length; k++) { const v = cr[k] * 3; P[3 * k] = V[v]; P[3 * k + 1] = V[v + 1]; P[3 * k + 2] = V[v + 2]; N[3 * k] = vn[v]; N[3 * k + 1] = vn[v + 1]; N[3 * k + 2] = vn[v + 2]; }
  gnm.geo.attributes.position.needsUpdate = true; gnm.geo.attributes.normal.needsUpdate = true;
  gnm.geo.computeBoundingSphere();
  if (name === 'capture' && headCenter.lengthSq() === 0) {
    for (let i = 0; i < nV; i++) headCenter.add(new THREE.Vector3(V[3 * i], V[3 * i + 1], V[3 * i + 2]));
    headCenter.divideScalar(nV);
  }
  S.pose = name; $('pose').value = name;
  updateErrorColors(); applyGnmMode();
}

const texLoader = new THREE.TextureLoader();
const texCache = {};
function tex(path) {
  if (!path) return null;
  if (!texCache[path]) {
    texCache[path] = texLoader.load(DATA + path, () => renderAll());
    texCache[path].colorSpace = THREE.SRGBColorSpace;
    texCache[path].anisotropy = 8;
  }
  return texCache[path];
}

function gnmMaterials(mode) {
  const names = M.gnm.materials, T = M.textures || {};
  const hidden = new THREE.MeshBasicMaterial({ visible: false });
  const po = { polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 };
  const fallback = new THREE.MeshBasicMaterial({ color: 0x8a5a8f, ...po });     // no evidence (inferred) in audit views
  return names.map((n) => {
    const info = M.gnm.material_info[n] || {};
    const isCornea = (info.opacity ?? 1) < 0.5;
    if (isCornea) return hidden;                                                // transparent cornea: skip, as in the bake renders
    if (mode === 'tex' || mode === 'texshade') {
      const map = tex((T.color || {})[n]);
      if (!map) return new THREE.MeshStandardMaterial({ color: 0xbbbbbb, ...po });
      return mode === 'tex' ? new THREE.MeshBasicMaterial({ map, ...po }) : new THREE.MeshStandardMaterial({ map, roughness: 0.8, ...po });
    }
    if (mode === 'smooth') return clay();
    if (mode === 'flat') return Object.assign(clay(), { flatShading: true });
    if (mode === 'normals') return new THREE.MeshNormalMaterial(po);
    if (mode === 'error') return new THREE.MeshBasicMaterial({ vertexColors: true, ...po });
    const map = tex(((T[mode === 'source' ? 'source' : 'confidence']) || {})[n]);
    return map ? new THREE.MeshBasicMaterial({ map, ...po }) : fallback;
  });
}
function applyGnmMode() {
  if (!gnm.mesh) return;
  gnm.mesh.material = gnmMaterials(S.gnmMode);
  gnm.wire.material.forEach((m, i) => { m.visible = S.gnmWire && !((M.gnm.material_info[M.gnm.materials[i]] || {}).opacity < 0.5); });
  renderAll();
}
function applyScanMode() {
  scan.mesh.material = scan.mats[S.scanMode];
  scan.wire.visible = S.scanWire;
  renderAll();
}
function updateErrorColors() {
  if (scan.geo) {
    const c = scan.geo.attributes.color.array, e = scan.err;
    for (let i = 0; i < e.length; i++) { const k = divergingColor(e[i], S.errRange, S.errOut); c.set(k.map((x) => Math.pow(x, 2.2)), 3 * i); }
    scan.geo.attributes.color.needsUpdate = true;
  }
  if (gnm.geo && gnm.err) {
    const c = gnm.geo.attributes.color.array, cr = gnm.corner, cap = S.pose === 'capture';
    for (let k = 0; k < cr.length; k++) {
      const col = cap ? divergingColor(gnm.err[cr[k]], S.errRange, S.errOut) : [0.35, 0.35, 0.38];
      c.set(col.map((x) => Math.pow(x, 2.2)), 3 * k);
    }
    gnm.geo.attributes.color.needsUpdate = true;
  }
}

// ---------------------------------------------------------------- cameras
function viewCam(k) { return M.views[k]; }
function fullCrop(v) { return { x0: 0, y0: 0, cw: v.w, ch: v.h }; }
function calibCamera(v, crop) {
  // OpenCV camera -> three.js camera (rotate by diag(1,-1,-1)); principal
  // point and crop go into an explicit projection matrix.
  const cam = new THREE.PerspectiveCamera();
  const c = v.c2w;
  const m = new THREE.Matrix4().set(
    c[0][0], -c[0][1], -c[0][2], c[0][3],
    c[1][0], -c[1][1], -c[1][2], c[1][3],
    c[2][0], -c[2][1], -c[2][2], c[2][3],
    0, 0, 0, 1);
  m.decompose(cam.position, cam.quaternion, cam.scale);
  const n = 0.01, f = 20, px = v.w / 2 + v.cx - crop.x0, py = v.h / 2 + v.cy - crop.y0;
  cam.projectionMatrix.set(
    2 * v.f / crop.cw, 0, 1 - 2 * px / crop.cw, 0,
    0, 2 * v.f / crop.ch, 2 * py / crop.ch - 1, 0,
    0, 0, -(f + n) / (f - n), -2 * f * n / (f - n),
    0, 0, -1, 0);
  cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
  cam.updateMatrixWorld(true);
  return cam;
}
function projectPoint(v, p) {                     // scan-frame point -> image pixel (u, v), depth
  const c = v.c2w, d = [p[0] - c[0][3], p[1] - c[1][3], p[2] - c[2][3]];
  const x = c[0][0] * d[0] + c[1][0] * d[1] + c[2][0] * d[2];
  const y = c[0][1] * d[0] + c[1][1] * d[1] + c[2][1] * d[2];
  const z = c[0][2] * d[0] + c[1][2] * d[1] + c[2][2] * d[2];
  return [v.f * x / z + v.w / 2 + v.cx, v.f * y / z + v.h / 2 + v.cy, z];
}
function ensureFreeCam() {
  if (freeCam) return;
  const v = viewCam(S.view);
  freeCam = calibCamera(v, fullCrop(v));
  const fc = new THREE.PerspectiveCamera(2 * Math.atan(v.h / 2 / v.f) * 180 / Math.PI, v.w / v.h, 0.005, 20);
  fc.position.copy(freeCam.position); fc.up.fromArray(M.up);
  freeTarget.copy(headCenter);
  fc.lookAt(freeTarget);
  freeCam = fc;
}
function cameraFor(panel) {
  if (S.camMode === 'calib') return calibCamera(viewCam(S.view), S.crop);
  ensureFreeCam();
  if (S.link || !panel) return freeCam;
  if (!unlinkedCams.has(panel.key)) unlinkedCams.set(panel.key, { cam: freeCam.clone(), target: freeTarget.clone() });
  return unlinkedCams.get(panel.key).cam;
}

// ---------------------------------------------------------------- panels
class Panel {
  constructor(key, layers, title) {
    this.key = key; this.layers = layers; this.title = title;
    this.el = document.createElement('div'); this.el.className = 'panel';
    this.photo = this.addCanvas(); this.photoCtx = this.photo.getContext('2d');
    this.r = {};
    for (const k of ['scan', 'gnm']) if (layers[k]) {
      const r = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      r.outputColorSpace = THREE.SRGBColorSpace;
      r.setClearColor(0x000000, 0);
      this.el.appendChild(r.domElement); this.r[k] = r;
    }
    this.ov = this.addCanvas(); this.ovCtx = this.ov.getContext('2d');
    this.tag = document.createElement('div'); this.tag.className = 'tag'; this.el.appendChild(this.tag);
    this.note = document.createElement('div'); this.note.className = 'note'; this.el.appendChild(this.note);
    this.input = this.ov; this.scale = 1;
  }
  get pxr() { return (window.devicePixelRatio || 1) * this.scale; }
  addCanvas() { const c = document.createElement('canvas'); this.el.appendChild(c); return c; }
  resize(w, h) {
    this.w = w; this.h = h; this.el.style.width = w + 'px'; this.el.style.height = h + 'px';
    const dpr = this.pxr;
    for (const c of [this.photo, this.ov]) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    for (const r of Object.values(this.r)) { r.setPixelRatio(dpr); r.setSize(w, h, true); }
  }
  opacity(k) { return S.layout === 'overlay' ? S.op[k] : 1; }
  async render() {
    const calib = S.camMode === 'calib';
    const v = viewCam(S.view);
    // photo
    const ctx = this.photoCtx, W = this.photo.width, H = this.photo.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
    this.photo.style.opacity = this.layers.photo ? this.opacity('photo') : 0;
    this.note.textContent = '';
    if (this.layers.photo) {
      if (calib) {
        const img = await getPhoto(S.view);
        const c = S.crop;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, c.x0, c.y0, c.cw, c.ch, 0, 0, W, H);
      } else if (!this.layers.scan && !this.layers.gnm) this.note.textContent = 'photos exist only for calibrated views';
    }
    // 3D layers
    for (const [k, s] of [['scan', scan], ['gnm', gnm]]) {
      const r = this.r[k]; if (!r || !s.mesh) continue;
      r.domElement.style.opacity = this.opacity(k);
      const cam = cameraFor(this);
      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
      s.light.position.copy(cam.position); s.light.target.position.copy(cam.position).add(fwd);
      s.light.target.updateMatrixWorld();
      r.render(s.scene, cam);
    }
    this.tag.textContent = this.title + (calib ? '  ·  ' + v.label : '  ·  free camera');
    await this.drawOverlay();
  }
  async drawOverlay() {
    const ctx = this.ovCtx, W = this.ov.width, H = this.ov.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, W, H);
    if (S.camMode !== 'calib') return;
    if (!S.ov.all && !(this.layers.photo)) return;
    const o = await getOverlay(S.view), c = S.crop;
    const sx = W / c.cw, sy = H / c.ch;
    const P = (p) => [(p[0] - c.x0) * sx, (p[1] - c.y0) * sy];
    const z = Math.sqrt(S.crop.cw > 0 ? viewCam(S.view).w / c.cw : 1);
    const dot = Math.max(2, 2.2 * z) * this.pxr;
    ctx.lineWidth = 1.5 * this.pxr;
    const poly = (pts, col, closed) => { if (!pts || pts.length < 2) return; ctx.strokeStyle = col; ctx.beginPath(); pts.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }); if (closed) ctx.closePath(); ctx.stroke(); };
    const pts = (arr, col, valid, from = 0, to = 68) => { ctx.fillStyle = col; for (let i = from; i < Math.min(to, arr.length); i++) { if (valid && !valid[i]) continue; const q = P(arr[i]); ctx.beginPath(); ctx.arc(q[0], q[1], dot, 0, 2 * Math.PI); ctx.fill(); } };
    if (S.ov.silp && o.sil_photo) o.sil_photo.forEach((p) => poly(p, '#ffd400', true));
    if (S.ov.sils && o.sil_scan) o.sil_scan.forEach((p) => poly(p, '#00e0ff', true));
    if (S.ov.silg && o.sil_gnm) o.sil_gnm.forEach((p) => poly(p, '#ff4dd2', true));
    if (S.ov.jaw) {
      if (o.detected) poly(o.detected.slice(0, 17).filter((_, i) => o.detected_valid[i]), '#7CFC00', false);
      if (o.jaw_gnm) pts(o.jaw_gnm, '#ff8c00', null, 0, o.jaw_gnm.length);
    }
    if (S.ov.res && o.detected) {
      ctx.strokeStyle = '#ffffff';
      for (let i = 17; i < 68; i++) { if (!o.detected_valid[i]) continue; const a = P(o.detected[i]), b = P(o.gnm_landmarks[i]); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
    }
    if (S.ov.det && o.detected) pts(o.detected, '#39ff14', o.detected_valid);
    if (S.ov.gnm && o.gnm_landmarks) pts(o.gnm_landmarks, '#ff2d2d', null, 17, 68);
  }
}

function layoutPanels() {
  const host = $('panels');
  for (const p of panels) for (const r of Object.values(p.r)) { r.dispose(); r.forceContextLoss(); }
  host.innerHTML = ''; panels = []; unlinkedCams.clear();
  if (S.layout === 'overlay') panels.push(new Panel('ov', { photo: true, scan: true, gnm: true }, 'overlay'));
  else {
    if (S.cols.photo) panels.push(new Panel('photo', { photo: true }, 'photo'));
    if (S.cols.scan) panels.push(new Panel('scan', { scan: true }, 'scan'));
    if (S.cols.gnm) panels.push(new Panel('gnm', { gnm: true }, 'GNM'));
  }
  panels.forEach((p) => { host.appendChild(p.el); attachInput(p); });
  sizePanels();
}
function sizePanels() {
  const host = $('panels'), n = Math.max(1, panels.length);
  const v = viewCam(S.view), aspect = S.camMode === 'calib' ? S.crop.cw / S.crop.ch : v.w / v.h;
  const W = host.clientWidth - 12 - 6 * (n - 1), H = host.clientHeight - 12;
  let h = Math.min(H, W / n / aspect), w = h * aspect;
  panels.forEach((p) => p.resize(Math.floor(w), Math.floor(h)));
  if (freeCam) { freeCam.aspect = aspect; freeCam.updateProjectionMatrix(); }
  for (const u of unlinkedCams.values()) { u.cam.aspect = aspect; u.cam.updateProjectionMatrix(); }
  renderAll();
}

let pending = false;
function renderAll() {
  if (pending) return; pending = true;
  requestAnimationFrame(async () => { pending = false; for (const p of panels) await p.render(); });
}

// ---------------------------------------------------------------- input
function attachInput(p) {
  const el = p.input;
  let drag = null;
  p.controls = new OrbitControls(new THREE.PerspectiveCamera(), el);   // replaced per mode below
  p.controls.enabled = false;
  const setupOrbit = () => {
    p.controls.dispose();
    if (S.camMode !== 'free') return;
    ensureFreeCam();
    const cam = cameraFor(p);
    p.controls = new OrbitControls(cam, el);
    if (S.link) p.controls.target = freeTarget; else p.controls.target = unlinkedCams.get(p.key).target;
    p.controls.zoomSpeed = 1.2; p.controls.rotateSpeed = 0.7; p.controls.update();
    p.controls.addEventListener('change', () => renderAll());
  };
  p.setupOrbit = setupOrbit; setupOrbit();
  el.addEventListener('wheel', (e) => {
    if (S.camMode !== 'calib') return; e.preventDefault();
    const r = el.getBoundingClientRect(), c = S.crop;
    const u = c.x0 + (e.clientX - r.left) / r.width * c.cw, v = c.y0 + (e.clientY - r.top) / r.height * c.ch;
    const k = e.deltaY > 0 ? 1.15 : 1 / 1.15, vw = viewCam(S.view);
    const cw = Math.min(vw.w * 1.5, Math.max(20, c.cw * k)), ch = cw * c.ch / c.cw;
    S.crop = { x0: u - (u - c.x0) * cw / c.cw, y0: v - (v - c.y0) * ch / c.ch, cw, ch };
    renderAll();
  }, { passive: false });
  el.addEventListener('pointerdown', (e) => { if (S.camMode === 'calib') { drag = { x: e.clientX, y: e.clientY, c: { ...S.crop } }; el.setPointerCapture(e.pointerId); } });
  el.addEventListener('pointermove', (e) => {
    if (!drag) return; const r = el.getBoundingClientRect();
    S.crop = { ...drag.c, x0: drag.c.x0 - (e.clientX - drag.x) / r.width * drag.c.cw, y0: drag.c.y0 - (e.clientY - drag.y) / r.height * drag.c.ch };
    renderAll();
  });
  el.addEventListener('pointerup', () => { drag = null; });
  el.addEventListener('dblclick', () => { if (S.camMode === 'calib') { S.crop = fullCrop(viewCam(S.view)); renderAll(); } });
}

// ---------------------------------------------------------------- UI
function setView(k, keepCrop = false) {
  S.view = (k + M.views.length) % M.views.length; $('view').value = S.view;
  if (!keepCrop) S.crop = fullCrop(viewCam(S.view));
  updateViewInfo(); sizePanels();
}
async function updateViewInfo() {
  const v = viewCam(S.view), o = await getOverlay(S.view);
  const flags = [v.holdout_fit ? 'HELD OUT of the fit' : 'used by the fit', v.used_in_texture === false ? 'held out of the texture' : (v.used_in_texture ? 'used in texture' : '')].filter(Boolean).join(' · ');
  let html = `${v.w}×${v.h}px · f ${v.f.toFixed(1)}px · ${v.distortion ? 'distortion' : 'no distortion'}<br>${flags}`;
  if (o.residual_px) html += '<br>median |det − GNM| px: ' + Object.entries(o.residual_px).filter(([, x]) => x != null).map(([g, x]) => `${g} ${x.toFixed(1)}`).join(', ') +
    '<br><span class="small">(jaw/brows are contour/texture-defined; GNM fixed points sit off them by design)</span>';
  $('viewinfo').innerHTML = html;
}
function inspect(r) {
  if (S.camMode === 'calib') {
    const k = M.views.findIndex((v) => v.label === r.best_view);
    const v = viewCam(k), [u, vv, z] = projectPoint(v, r.center);
    const ch = Math.min(v.h, 1.7 * v.f * r.size / z), cw = ch * v.w / v.h;
    S.view = k; $('view').value = k; S.crop = { x0: u - cw / 2, y0: vv - ch / 2, cw, ch };
    updateViewInfo(); sizePanels();
  } else {
    ensureFreeCam();
    const c = new THREE.Vector3().fromArray(r.center), n = new THREE.Vector3().fromArray(r.normal).normalize();
    const dist = r.size / (2 * Math.tan(freeCam.fov * Math.PI / 360)) * 1.6;
    const cams = S.link ? [[freeCam, freeTarget]] : [...unlinkedCams.values()].map((u) => [u.cam, u.target]);
    for (const [cam, tgt] of cams) { tgt.copy(c); cam.position.copy(c).addScaledVector(n, dist); cam.up.fromArray(M.up); cam.lookAt(c); }
    panels.forEach((p) => p.controls.update()); renderAll();
  }
}
function legend() {
  let h = '<h2>Legend</h2>';
  h += '<div class="small"><span class="sw" style="background:#39ff14"></span>detected landmark <span class="sw" style="background:#ff2d2d"></span>GNM landmark<br>' +
    '<span class="sw" style="background:#7CFC00"></span>detected jaw <span class="sw" style="background:#ff8c00"></span>GNM occluding contour<br>' +
    '<span class="sw" style="background:#ffd400"></span>photo matte <span class="sw" style="background:#00e0ff"></span>scan <span class="sw" style="background:#ff4dd2"></span>GNM silhouette</div>';
  h += `<div class="small" style="margin-top:6px">Surface error (signed, mm): <div class="bar" style="background:linear-gradient(90deg,#0d66ff,#fff,#ff2600)"></div>−${S.errRange} … 0 … +${S.errRange}; grey beyond ${S.errOut} mm (no counterpart surface)</div>`;
  if (M.error) h += `<div class="small muted">GNM→scan |d| median ${M.error.gnm_to_scan.median_abs.toFixed(2)} mm (all skin vertices incl. unscanned crown)</div>`;
  if (M.source_legend) {
    h += '<div class="small" style="margin-top:6px">Texture source view:</div><table class="small">';
    M.source_legend.forEach((s, i) => { if (i % 2 === 0) h += '<tr>'; h += `<td><span class="sw" style="background:rgb(${s.rgb})"></span>${s.view}</td>`; if (i % 2) h += '</tr>'; });
    h += '</table><div class="small muted">' + (M.audit_legend ? Object.values(M.audit_legend).join('<br>') : '') + '</div>';
  }
  if (M.photometric) {
    const f = (x) => x ? `MAE ${x.mae_after_gain.toFixed(1)}, PSNR ${x.psnr_after_gain.toFixed(1)} dB (${x.views} views)` : '';
    h += `<div class="small" style="margin-top:6px">Texture vs photo (8-bit sRGB, after per-view gain): used ${f(M.photometric.used)}${M.photometric.held_out ? '<br>held-out ' + f(M.photometric.held_out) : ''}</div>`;
  }
  if (M.texture_report && M.texture_report.colour_space) h += `<div class="small muted" style="margin-top:6px">${M.texture_report.colour_space}</div>`;
  $('legend').innerHTML = h;
}

async function screenshot() {
  const scale = +$('shot_scale').value, dpr = window.devicePixelRatio || 1;
  const head = 28 * scale * dpr;
  const tiles = [];
  for (const p of panels) {
    p.scale = scale; p.resize(p.w, p.h);
    await p.render();
    const c = document.createElement('canvas'); c.width = p.photo.width; c.height = p.photo.height;
    const x = c.getContext('2d');
    x.fillStyle = '#000'; x.fillRect(0, 0, c.width, c.height);
    if (p.layers.photo) { x.globalAlpha = +p.photo.style.opacity; x.drawImage(p.photo, 0, 0); }
    for (const [k, r] of Object.entries(p.r)) {
      // re-render right before compositing (no preserveDrawingBuffer needed)
      r.render(k === 'scan' ? scan.scene : gnm.scene, cameraFor(p));
      x.globalAlpha = p.opacity(k); x.drawImage(r.domElement, 0, 0, c.width, c.height);
    }
    x.globalAlpha = 1; x.drawImage(p.ov, 0, 0);
    tiles.push([c, p.tag.textContent]);
    p.scale = 1; p.resize(p.w, p.h);
  }
  const gap = 6 * scale, foot = 20 * scale * dpr, Wt = tiles.reduce((a, [c]) => a + c.width, 0) + gap * (tiles.length - 1), Ht = tiles[0][0].height + head + foot;
  const out = document.createElement('canvas'); out.width = Wt; out.height = Ht;
  const o = out.getContext('2d'); o.fillStyle = '#16181c'; o.fillRect(0, 0, Wt, Ht);
  let x0 = 0;
  o.font = `${Math.round(14 * scale * dpr)}px system-ui, sans-serif`; o.fillStyle = '#e6e6e6'; o.textBaseline = 'middle';
  for (const [c, t] of tiles) {
    o.save(); o.beginPath(); o.rect(x0, 0, c.width, head); o.clip(); o.fillText(t, x0 + 6, head / 2); o.restore();
    o.drawImage(c, x0, head); x0 += c.width + gap;
  }
  o.font = `${Math.round(11 * scale * dpr)}px system-ui, sans-serif`; o.fillStyle = '#9aa0a8';
  o.fillText(`pose ${S.pose} · GNM ${S.gnmMode} · scan ${S.scanMode} · ${M.run.split(/[\\/]/).slice(-1)[0]}`, 6, Ht - foot / 2);
  const v = viewCam(S.view);
  const name = `${S.camMode === 'calib' ? v.label : 'free'}_${S.layout}_${S.pose}_${S.gnmMode}`;
  const url = out.toDataURL('image/png');
  const a = document.createElement('a'); a.href = url; a.download = name + '.png'; a.click();
  try {
    const r = await fetch('/api/screenshot', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, png: url }) });
    if (r.ok) { const j = await r.json(); $('shotmsg').textContent = 'saved: ' + j.saved; return; }
  } catch (e) { /* static server without the API: download only */ }
  $('shotmsg').textContent = 'downloaded ' + name + '.png';
  renderAll();
}

function bindUI() {
  const sel = $('view');
  M.views.forEach((v, i) => sel.add(new Option(`${v.label}${v.holdout_fit ? '  (held out of fit)' : ''}${v.used_in_texture === false ? '  (held out of texture)' : ''}`, i)));
  sel.onchange = () => setView(+sel.value);
  const ps = $('pose');
  M.poses.forEach((p) => ps.add(new Option(p.label, p.name)));
  ps.onchange = () => setPose(ps.value);
  document.querySelectorAll('input[name=cammode]').forEach((r) => r.onchange = () => {
    S.camMode = r.value; document.body.classList.toggle('free', S.camMode === 'free');
    if (S.camMode === 'free') { freeCam = null; ensureFreeCam(); }
    unlinkedCams.clear(); panels.forEach((p) => p.setupOrbit()); sizePanels();
  });
  $('linkcams').onchange = (e) => { S.link = e.target.checked; unlinkedCams.clear(); panels.forEach((p) => p.setupOrbit()); renderAll(); };
  document.querySelectorAll('input[name=layout]').forEach((r) => r.onchange = () => {
    S.layout = r.value; $('sbsopts').style.display = S.layout === 'sbs' ? '' : 'none'; $('ovopts').style.display = S.layout === 'overlay' ? '' : 'none'; layoutPanels();
  });
  $('ovopts').style.display = 'none';
  for (const k of ['photo', 'scan', 'gnm']) {
    $('col_' + k).onchange = (e) => { S.cols[k] = e.target.checked; layoutPanels(); };
    const sl = $('op_' + k), out = sl.nextElementSibling;
    const upd = () => { S.op[k] = +sl.value; out.textContent = (+sl.value).toFixed(2); renderAll(); };
    sl.oninput = upd; out.textContent = (+sl.value).toFixed(2);
  }
  $('gnm_mode').onchange = (e) => { S.gnmMode = e.target.value; applyGnmMode(); };
  $('scan_mode').onchange = (e) => { S.scanMode = e.target.value; applyScanMode(); };
  $('gnm_wire').onchange = (e) => { S.gnmWire = e.target.checked; applyGnmMode(); };
  $('scan_wire').onchange = (e) => { S.scanWire = e.target.checked; applyScanMode(); };
  for (const [id, key] of [['err_range', 'errRange'], ['err_out', 'errOut']]) {
    const sl = $(id), out = sl.nextElementSibling; out.textContent = sl.value;
    sl.oninput = () => { S[key] = +sl.value; out.textContent = sl.value; updateErrorColors(); legend(); renderAll(); };
  }
  for (const k of Object.keys(S.ov)) $('ov_' + k).onchange = (e) => { S.ov[k] = e.target.checked; renderAll(); };
  const rg = $('regions');
  (M.regions || []).forEach((r) => { const b = document.createElement('button'); b.textContent = r.name.replace('_', ' '); b.onclick = () => inspect(r); rg.appendChild(b); });
  $('resetzoom').onclick = () => {
    if (S.camMode === 'calib') S.crop = fullCrop(viewCam(S.view));
    else { freeCam = null; unlinkedCams.clear(); ensureFreeCam(); panels.forEach((p) => p.setupOrbit()); }
    sizePanels();
  };
  $('shot').onclick = screenshot;
  window.addEventListener('resize', sizePanels);
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowRight') setView(S.view + 1); else if (e.key === 'ArrowLeft') setView(S.view - 1);
  });
}

async function main() {
  try {
    M = await json('manifest.json');
  } catch (e) { status('cannot load ' + DATA + 'manifest.json — start with tools/viewer/serve.py --data <viewer_data>'); return; }
  $('runinfo').textContent = M.run.split(/[\\/]/).slice(-1)[0] + ' · ' + M.created;
  S.crop = fullCrop(viewCam(0));
  bindUI(); legend(); layoutPanels(); updateViewInfo();
  status('loading meshes…');
  const [gi, gu, gm, ge, sp, si, se] = await Promise.all([
    bin('gnm_idx.bin', Uint32Array), bin('gnm_uv.bin', Float32Array), bin('gnm_mat.bin', Uint8Array), bin('gnm_err.bin', Float32Array),
    bin('scan_pos.bin', Float32Array), bin('scan_idx.bin', Uint32Array), bin('scan_err.bin', Float32Array)]);
  buildGnm(gi, gu, gm); gnm.err = ge;
  buildScan(sp, si, se);
  await setPose('capture');
  applyScanMode();
  status(`scan ${M.scan.triangles.toLocaleString()} tris${M.scan.decimated ? ' (decimated from ' + M.scan.full_triangles.toLocaleString() + ')' : ''} · GNM ${M.gnm.vertices.toLocaleString()} verts · ${M.views.length} views`);
  window.__viewer = { S, M, setView, setPose, inspect, renderAll, screenshot, panels: () => panels };
}
main();
