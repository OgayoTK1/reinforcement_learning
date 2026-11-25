import * as THREE from '/vendor/three.module.min.js';
import { fbm, smoothstep, lerp, mulberry32 } from './noise.js';

export const PLOT = { size: 14, gap: 3 };
// plot i sits in a 2 x 2 block; plots 0,1 are upslope, 2,3 downslope
export function plotOrigin(i) {
  const s = PLOT.size + PLOT.gap;
  return { x: (i % 2 === 0 ? -1 : 1) * s / 2, z: (i < 2 ? -1 : 1) * s / 2 };
}

export function ground(x, z) {
  const rolling = (fbm(x * 0.018 + 3, z * 0.018 + 7) - 0.5) * 9;
  const north = Math.max(0, (-z - 45) / 70);
  const escarpment = north * north * 70 + north * fbm(x * 0.03 + 11, z * 0.03) * 28;
  const wild = rolling + escarpment;
  const near = (fbm(x * 0.05, z * 0.05) - 0.5) * 0.5 - 0.055 * z;
  const blend = 1 - smoothstep(26, 60, Math.hypot(x * 0.75, z));
  return lerp(wild, near, blend);
}

export function buildTerrain() {
  const N = 200, S = 520;
  const geo = new THREE.PlaneGeometry(S, S, N, N);
  geo.rotateX(-Math.PI / 2);
  const p = geo.attributes.position, col = new Float32Array(p.count * 3);
  const c = new THREE.Color();
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), z = p.getZ(i), y = ground(x, z);
    p.setY(i, y);
    const n = fbm(x * 0.08, z * 0.08), m = fbm(x * 0.4 + 50, z * 0.4);
    const forest = smoothstep(18, 45, y) * smoothstep(0.35, 0.6, fbm(x * 0.02 + 9, z * 0.02 + 2));
    // dry-season grass and bare earth patches; Njoro soils are dark brown to reddish volcanic loam
    c.setRGB(0.36 + 0.14 * n, 0.38 + 0.12 * n, 0.15 + 0.05 * m);
    const bare = smoothstep(0.55, 0.75, fbm(x * 0.05 + 20, z * 0.05 + 4));
    c.lerp(new THREE.Color(0.34, 0.24, 0.15), bare * 0.65);
    c.lerp(new THREE.Color(0.13, 0.22, 0.13), forest * 0.9);
    c.multiplyScalar(0.92 + 0.16 * m);
    col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.computeVertexNormals();
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 0 }));
  mesh.receiveShadow = true;
  mesh.material.color.set(0xffffff);
  return mesh;
}

// distant forest on the escarpment (Mau): instanced dark conifers and broadleaf blobs
export function buildForest(rng = mulberry32(7)) {
  const g = new THREE.Group();
  const cone = new THREE.ConeGeometry(2.2, 9, 6); cone.translate(0, 4.5, 0);
  const mat = new THREE.MeshStandardMaterial({ color: 0x24402a, roughness: 1, flatShading: true });
  const count = 1400, inst = new THREE.InstancedMesh(cone, mat, count);
  const d = new THREE.Object3D(); const col = new THREE.Color(); let k = 0;
  while (k < count) {
    const x = (rng() - 0.5) * 460, z = -70 - rng() * 190, y = ground(x, z);
    if (y < 14 || fbm(x * 0.02 + 9, z * 0.02 + 2) < 0.44) continue;
    const s = 0.7 + rng() * 1.4;
    d.position.set(x, y - 0.5, z); d.scale.set(s, s * (0.8 + rng() * 0.8), s); d.rotation.y = rng() * 6; d.updateMatrix();
    inst.setMatrixAt(k, d.matrix);
    col.setRGB(0.12 + rng() * 0.07, 0.22 + rng() * 0.1, 0.13 + rng() * 0.05); inst.setColorAt(k, col); k++;
  }
  g.add(inst);
  return g;
}
