import * as THREE from '/vendor/three.module.min.js';
import { ground, plotOrigin, PLOT } from './terrain.js';
import { mulberry32, clamp, lerp, fbm } from './noise.js';

const ROWS = 10, PER_ROW = 12, N_PLANTS = ROWS * PER_ROW, MAX_LEAF = 10;
const dummy = new THREE.Object3D();

function leafGeometry() {
  const seg = 7, pos = [], idx = [];
  for (let i = 0; i <= seg; i++) {
    const t = i / seg, x = t, y = 0.55 * t - 0.62 * t * t, w = 0.1 * Math.sin(Math.PI * Math.pow(t, 0.6)) + 0.004;
    pos.push(x, y, -w, x, y, w);
    if (i < seg) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx); g.computeVertexNormals();
  return g;
}
const LEAF = leafGeometry();
const STEM = (() => { const g = new THREE.CylinderGeometry(0.03, 0.05, 1, 5); g.translate(0, 0.5, 0); return g; })();
const TASSEL = (() => { const g = new THREE.ConeGeometry(0.05, 0.4, 4); g.translate(0, 0.2, 0); return g; })();
const COB = (() => { const g = new THREE.CylinderGeometry(0.075, 0.055, 0.34, 7); g.rotateZ(Math.PI / 2.4); return g; })();
const TUFT = (() => { const g = new THREE.ConeGeometry(0.09, 0.42, 4); g.translate(0, 0.21, 0); return g; })();
const STRAW = new THREE.BoxGeometry(0.5, 0.025, 0.045);

const GREEN = new THREE.Color(0.17, 0.36, 0.11), YELLOW = new THREE.Color(0.66, 0.58, 0.22), BROWN = new THREE.Color(0.40, 0.28, 0.14);
const SPOT = new THREE.Color(0.30, 0.22, 0.10);

export class MaizePlot {
  constructor(index, scene, seed = 100) {
    this.i = index;
    const o = plotOrigin(index), half = PLOT.size / 2;
    this.origin = o;
    const rng = mulberry32(seed + index * 31);
    this.group = new THREE.Group(); scene.add(this.group);

    // tilled soil with ridges running east-west
    const sg = new THREE.PlaneGeometry(PLOT.size, PLOT.size, 56, 56); sg.rotateX(-Math.PI / 2);
    const p = sg.attributes.position, col = new Float32Array(p.count * 3);
    for (let k = 0; k < p.count; k++) {
      const x = o.x + p.getX(k), z = o.z + p.getZ(k);
      const ridge = Math.sin((p.getZ(k) / PLOT.size) * ROWS * Math.PI * 2 + 1.2) * 0.07;
      p.setY(k, ground(x, z) + 0.03 + ridge);
      const n = 0.88 + 0.24 * fbm(x * 0.7, z * 0.7);
      col[k * 3] = n; col[k * 3 + 1] = n; col[k * 3 + 2] = n;
    }
    sg.setAttribute('color', new THREE.BufferAttribute(col, 3)); sg.computeVertexNormals();
    this.soilMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, color: 0x6b4a2e });
    this.soil = new THREE.Mesh(sg, this.soilMat); this.soil.position.set(o.x, 0, o.z); this.soil.receiveShadow = true;
    this.group.add(this.soil);

    // plants with their own spacing jitter, height, lean, leaf count and colour
    this.plants = [];
    const rowStep = PLOT.size / (ROWS + 0.5), colStep = PLOT.size / (PER_ROW + 0.5);
    for (let r = 0; r < ROWS; r++) for (let c = 0; c < PER_ROW; c++) {
      const x = o.x - half + (c + 0.75) * colStep + (rng() - 0.5) * 0.25;
      const z = o.z - half + (r + 0.75) * rowStep + (rng() - 0.5) * 0.12;
      this.plants.push({
        x, z, y: ground(x, z) + 0.03 + Math.sin(((z - o.z) / PLOT.size) * ROWS * Math.PI * 2 + 1.2) * 0.07,
        h: 0.82 + rng() * 0.36, lean: (rng() - 0.5) * 0.18, leanDir: rng() * 6.28, rot: rng() * 6.28,
        nLeaf: 8 + Math.floor(rng() * 3), vigour: 0.85 + rng() * 0.3, tone: rng(), late: rng(), // late: how late this plant germinates/matures
        cobAt: 0.52 + rng() * 0.1, cobYaw: rng() * 6.28, hasCob: rng() > 0.08,
      });
    }
    const mk = (geo, n, color, rough = 0.85, side = THREE.FrontSide) => {
      const m = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ color, roughness: rough, side }), n);
      m.instanceMatrix.setUsage(THREE.DynamicDrawUsage); m.castShadow = false; m.receiveShadow = true;
      m.frustumCulled = false; this.group.add(m); return m;
    };
    this.stems = mk(STEM, N_PLANTS, 0x5c7a34);
    this.leaves = mk(LEAF, N_PLANTS * MAX_LEAF, 0xffffff, 0.8, THREE.DoubleSide);
    this.tassels = mk(TASSEL, N_PLANTS, 0xc9b25a);
    this.cobs = mk(COB, N_PLANTS, 0xffffff);
    this.weedMesh = mk(TUFT, 280, 0x6fa23a); this.weedMesh.count = 0;
    this.strawMesh = mk(STRAW, 300, 0xcdb460); this.strawMesh.count = 0;
    this.weedPos = Array.from({ length: 280 }, () => ({ x: o.x + (rng() - 0.5) * PLOT.size * 0.96, z: o.z + (rng() - 0.5) * PLOT.size * 0.96, s: 0.6 + rng() * 1.1, r: rng() * 6.28 }));
    this.strawPos = Array.from({ length: 300 }, () => ({ x: o.x + (rng() - 0.5) * PLOT.size * 0.96, z: o.z + (rng() - 0.5) * PLOT.size * 0.96, r: rng() * 3.14 }));
    // eased visual state; targets come from the environment state
    this.v = { grow: 0, leaf: 1, sen: 0, droop: 0, cob: 0, dis: 0, faw: 0, cut: 0, dead: 0, moist: 0.5, stage: 0 };
    this.t = { ...this.v };
    this.dirty = true;
    this.relayout();
    this.setWeedsMulch(0, 0);
  }

  setTarget(plot, crop) {
    const sown = crop.sown, stage = crop.stage_index;
    const grow = sown ? clamp(Math.pow(clamp(crop.gdd / 1150), 0.75), 0.04, 1) : 0;
    this.t = {
      grow, leaf: plot.canopy_retained, stage,
      sen: clamp(0.85 * plot.stress + 0.8 * (1 - plot.canopy_retained) + 0.35 * plot.disease_severity + (stage >= 4 ? 0.55 : 0) + (stage === 3 ? 0.12 : 0)),
      droop: clamp(plot.stress * 0.9 + (stage >= 4 ? 0.35 : 0)),
      cob: stage >= 2 ? clamp(plot.grain_gm2 / 380) : 0,
      dis: plot.disease_severity, faw: plot.faw_pressure, cut: crop.harvested_fraction,
      dead: plot.alive ? 0 : 1, moist: plot.soil_moisture,
    };
    this.weeds = plot.weeds; this.mulch = plot.mulch; this.setWeedsMulch(plot.weeds, plot.mulch);
    this.dirty = true;
  }

  setWeedsMulch(w, m) {
    this.weedMesh.count = Math.round(w * 280); this.strawMesh.count = Math.round(m * 300);
    for (let k = 0; k < this.weedMesh.count; k++) {
      const q = this.weedPos[k]; dummy.position.set(q.x, ground(q.x, q.z) + 0.02, q.z); dummy.rotation.set(0, q.r, 0);
      const h = 0.7 + 1.2 * w; dummy.scale.set(q.s, q.s * h, q.s); dummy.updateMatrix(); this.weedMesh.setMatrixAt(k, dummy.matrix);
    }
    for (let k = 0; k < this.strawMesh.count; k++) {
      const q = this.strawPos[k]; dummy.position.set(q.x, ground(q.x, q.z) + 0.1, q.z); dummy.rotation.set(0, q.r, 0); dummy.scale.set(1, 1, 1);
      dummy.updateMatrix(); this.strawMesh.setMatrixAt(k, dummy.matrix);
    }
    this.weedMesh.instanceMatrix.needsUpdate = true; this.strawMesh.instanceMatrix.needsUpdate = true;
  }

  // ease visual values toward targets; returns true while still moving
  tick(dt) {
    let moving = false;
    for (const k of Object.keys(this.t)) {
      const d = this.t[k] - this.v[k];
      if (Math.abs(d) > 1e-3) { this.v[k] += d * Math.min(1, dt * 2.2); moving = true; } else this.v[k] = this.t[k];
    }
    const m = this.v.moist;
    this.soilMat.color.setRGB(lerp(0.50, 0.26, m), lerp(0.36, 0.18, m), lerp(0.22, 0.12, m));
    if (moving || this.dirty) { this.relayout(); this.dirty = false; }
    return moving;
  }

  relayout() {
    const v = this.v, col = new THREE.Color();
    const nCut = Math.floor(v.cut * N_PLANTS + 1e-6);
    const sown = v.grow > 0.02;
    for (let p = 0; p < N_PLANTS; p++) {
      const pl = this.plants[p];
      const cut = p < nCut;
      // plants do not all emerge or mature at once: stagger by the plant's own lateness
      const g = clamp((v.grow - pl.late * 0.12) / (1 - 0.12), 0, 1) * pl.vigour;
      const alive = v.dead < 0.5;
      const H = sown ? (cut ? 0.35 : (alive ? 3.0 : 1.6) * g * pl.h * (1 - 0.18 * v.droop * (alive ? 1 : 2)) + 0.05) : 0.0001;
      const lean = pl.lean + (alive ? 0 : 0.5) + v.droop * 0.1 * (1 + (pl.late - 0.5));
      // stem
      dummy.position.set(pl.x, pl.y, pl.z); dummy.rotation.set(Math.sin(pl.leanDir) * lean, 0, Math.cos(pl.leanDir) * lean);
      dummy.scale.set(0.5 + g, H, 0.5 + g); dummy.updateMatrix(); this.stems.setMatrixAt(p, dummy.matrix);
      const topX = pl.x - Math.sin(Math.cos(pl.leanDir) * lean) * H, topZ = pl.z + Math.sin(Math.sin(pl.leanDir) * lean) * H;
      // leaves
      const sen = clamp(v.sen + (pl.tone - 0.5) * 0.18 + (alive ? 0 : 0.7));
      for (let l = 0; l < MAX_LEAF; l++) {
        const idx = p * MAX_LEAF + l;
        if (!sown || cut || l >= pl.nLeaf) { dummy.position.set(pl.x, -50, pl.z); dummy.scale.set(0.0001, 0.0001, 0.0001); dummy.updateMatrix(); this.leaves.setMatrixAt(idx, dummy.matrix); continue; }
        const f = l / pl.nLeaf, lenF = 0.65 + 0.55 * Math.sin(Math.PI * (f * 0.9 + 0.1));
        const L = 1.55 * lenF * g * (0.25 + 0.75 * v.leaf) * (alive ? 1 : 0.6);
        const hy = H * (0.18 + 0.72 * f);
        const tilt = 0.30 + 0.35 * f + v.droop * (0.55 + 0.4 * f) - 0.25 * (1 - f);
        dummy.position.set(pl.x + (topX - pl.x) * (hy / Math.max(H, 0.01)), pl.y + hy, pl.z + (topZ - pl.z) * (hy / Math.max(H, 0.01)));
        dummy.rotation.set(0, pl.rot + l * 2.399 + Math.sin(l * 3.1 + p) * 0.18, 0);
        dummy.rotateZ(clamp(tilt, -0.2, 1.1) - 0.25 * (1 - v.droop));
        dummy.scale.set(L, L * (0.9 + 0.2 * (1 - v.droop)), L * (1.0 - 0.45 * v.droop));
        dummy.updateMatrix(); this.leaves.setMatrixAt(idx, dummy.matrix);
        // colour: green -> yellow -> brown with senescence; older (lower) leaves go first; disease and armyworm damage darken or pale them
        const lowerFirst = clamp(sen + (1 - f) * 0.35 * sen);
        col.copy(GREEN).lerp(YELLOW, clamp(lowerFirst * 1.6)).lerp(BROWN, clamp(lowerFirst * 1.6 - 0.9));
        col.lerp(SPOT, v.dis * 0.45 * (0.5 + pl.tone)); col.lerp(new THREE.Color(0.55, 0.60, 0.30), v.faw * 0.25 * pl.late);
        const j = 0.9 + 0.2 * ((l * 7 + p * 13) % 10) / 10; col.multiplyScalar(j);
        this.leaves.setColorAt(idx, col);
      }
      // tassel and cob
      const showT = sown && !cut && alive && v.stage >= 1.6 && v.stage < 4.4;
      dummy.position.set(topX, pl.y + H * 0.98, topZ); dummy.rotation.set(0, pl.rot, 0.1);
      const ts = showT ? 0.6 + 0.6 * g : 0.0001; dummy.scale.set(ts, ts, ts); dummy.updateMatrix(); this.tassels.setMatrixAt(p, dummy.matrix);
      const cs = (sown && !cut && alive && pl.hasCob) ? 0.3 + 1.0 * v.cob : 0.0001;
      dummy.position.set(pl.x + Math.cos(pl.cobYaw) * 0.12, pl.y + H * pl.cobAt, pl.z + Math.sin(pl.cobYaw) * 0.12);
      dummy.rotation.set(0, pl.cobYaw, 0); dummy.scale.set(cs, cs, cs); dummy.updateMatrix(); this.cobs.setMatrixAt(p, dummy.matrix);
      col.setRGB(0.42, 0.55, 0.22).lerp(new THREE.Color(0.74, 0.64, 0.38), clamp((v.stage - 3) * 0.9)); this.cobs.setColorAt(p, col);
    }
    this.stems.instanceMatrix.needsUpdate = true; this.leaves.instanceMatrix.needsUpdate = true;
    this.tassels.instanceMatrix.needsUpdate = true; this.cobs.instanceMatrix.needsUpdate = true;
    if (this.leaves.instanceColor) this.leaves.instanceColor.needsUpdate = true;
    if (this.cobs.instanceColor) this.cobs.instanceColor.needsUpdate = true;
    const sc = clamp(v.sen * 1.3);
    this.stems.material.color.setRGB(lerp(0.36, 0.62, sc), lerp(0.48, 0.52, sc), lerp(0.20, 0.28, sc));
  }
}
