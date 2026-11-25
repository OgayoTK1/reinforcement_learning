import * as THREE from '/vendor/three.module.min.js';
import { ground } from './terrain.js';
import { mulberry32, clamp, lerp } from './noise.js';

const std = (color, rough = 0.9, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: rough, ...extra });
const box = (w, h, d, mat, x, y, z, parent) => { const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat); m.position.set(x, y, z); m.castShadow = m.receiveShadow = true; parent.add(m); return m; };

// corrugated iron (mabati) roof slab: a plane whose cross-section is a sine wave
function mabati(w, d, color) {
  const g = new THREE.PlaneGeometry(w, d, 60, 2); g.rotateX(-Math.PI / 2);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) p.setY(i, Math.sin(p.getX(i) * 7.5) * 0.045);
  g.computeVertexNormals();
  const m = new THREE.Mesh(g, std(color, 0.55, { metalness: 0.35, side: THREE.DoubleSide })); m.castShadow = true; return m;
}

export function buildHomestead(scene) {
  const root = new THREE.Group(); scene.add(root);
  const hx = 38, hz = -6, gy = ground(hx, hz);
  const house = new THREE.Group(); house.position.set(hx, gy, hz); house.rotation.y = -0.25; root.add(house);
  box(8, 3, 5, std(0xb59a72), 0, 1.5, 0, house);                   // plastered wall
  box(8.2, 0.35, 5.2, std(0x7a6a55), 0, 0.17, 0, house);           // stone plinth
  box(1.1, 2.1, 0.12, std(0x5a3d22), -1.6, 1.05, 2.53, house);     // door
  box(1.2, 0.9, 0.1, std(0x2b3438), 1.7, 1.7, 2.53, house);        // window
  for (const s of [-1, 1]) {
    const r = mabati(8.8, 3.6, s < 0 ? 0x8d8f8c : 0x9a5a3c); r.rotation.z = s * 0.38; r.position.set(0, 3.9, s * 1.55 * -1 + 0); r.rotation.x = 0;
    r.rotation.set(s * 0.42, 0, 0); r.position.set(0, 3.75, s * 1.6); house.add(r);
  }
  box(8.9, 0.12, 0.2, std(0x6a6a66, 0.6, { metalness: .4 }), 0, 4.55, 0, house); // ridge cap
  // chimney of a wood-fire kitchen
  box(0.6, 1.6, 0.6, std(0x8a7a68), -3, 4.4, -1, house);

  // plastic water tank beside the house
  const tank = new THREE.Mesh(new THREE.CylinderGeometry(1.15, 1.15, 2.1, 20), std(0x1c1f22, 0.45));
  tank.position.set(hx + 5.6, ground(hx + 5.6, hz + 3) + 1.05, hz + 3); tank.castShadow = true; root.add(tank);
  const lid = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 0.15, 12), std(0x2a6a9b)); lid.position.set(tank.position.x, tank.position.y + 1.1, tank.position.z); root.add(lid);

  // cow shed with a Friesian cross cow
  const sx = hx + 3, sz = hz - 11, sy = ground(sx, sz);
  const shed = new THREE.Group(); shed.position.set(sx, sy, sz); shed.rotation.y = 0.15; root.add(shed);
  for (const [x, z] of [[-2.6, -1.7], [2.6, -1.7], [-2.6, 1.7], [2.6, 1.7]]) box(0.18, 2.4, 0.18, std(0x6b4a2a), x, 1.2, z, shed);
  const sr = mabati(6.4, 4.2, 0x8d8f8c); sr.position.y = 2.55; sr.rotation.x = 0.1; shed.add(sr);
  box(5.2, 0.12, 3.4, std(0x4f3d2c), 0, 0.06, 0, shed);
  const cow = new THREE.Group(); cow.position.set(0, 0, 0); shed.add(cow);
  const black = std(0x1b1b1a), white = std(0xe8e3d6);
  box(1.7, 0.85, 0.75, white, 0, 1.0, 0, cow); box(0.8, 0.86, 0.77, black, -0.45, 1.01, 0, cow); box(0.5, 0.4, 0.76, black, 0.55, 1.12, 0, cow);
  const head = new THREE.Group(); head.position.set(1.05, 1.15, 0); cow.add(head); box(0.5, 0.42, 0.42, black, 0, 0, 0, head); box(0.18, 0.2, 0.36, white, 0.3, -0.08, 0, head);
  for (const [x, z] of [[-0.6, -0.26], [-0.6, 0.26], [0.6, -0.26], [0.6, 0.26]]) box(0.14, 0.62, 0.14, white, x, 0.31, z, cow);
  box(0.06, 0.6, 0.06, black, -0.9, 0.95, 0, cow);
  root.userData.cowHead = head;

  // small raised granary (maize store) on stilts
  const gx = hx - 7, gz = hz + 7, gy2 = ground(gx, gz);
  const gran = new THREE.Group(); gran.position.set(gx, gy2, gz); root.add(gran);
  for (const [x, z] of [[-1, -0.8], [1, -0.8], [-1, 0.8], [1, 0.8]]) box(0.2, 0.7, 0.2, std(0x5a3f26), x, 0.35, z, gran);
  box(2.6, 1.7, 2.0, std(0x9c7f55), 0, 1.55, 0, gran);
  const gr = mabati(3.0, 2.8, 0x8d8f8c); gr.rotation.x = 0.0; gr.position.y = 2.6; gran.add(gr);
  root.userData.granary = gran;

  // drying sheet (tarpaulin) with a layer of cobs/grain, shown when the farmer dries grain
  const dx = hx - 11, dz = hz + 1, dy = ground(dx, dz);
  const sheet = new THREE.Mesh(new THREE.PlaneGeometry(6, 4), std(0x3b5b86, 1)); sheet.rotation.x = -Math.PI / 2; sheet.position.set(dx, dy + 0.05, dz); sheet.receiveShadow = true; root.add(sheet);
  const grainLayer = new THREE.Mesh(new THREE.PlaneGeometry(5.4, 3.4, 1, 1), std(0xd8b43c, 1)); grainLayer.rotation.x = -Math.PI / 2; grainLayer.position.set(dx, dy + 0.09, dz); root.add(grainLayer);
  root.userData.grainLayer = grainLayer;

  // sacks (90 kg bags) stacked beside the granary
  const bagGeo = new THREE.BoxGeometry(0.9, 0.28, 0.55);
  const bags = new THREE.InstancedMesh(bagGeo, std(0xeeeadf, 0.95), 40); bags.count = 0; bags.castShadow = true; root.add(bags);
  root.userData.bags = bags; root.userData.bagBase = { x: gx - 3.4, z: gz + 0.8 };

  // lined water pan: earth berm, black polythene liner and a water surface that rises and falls with storage
  const px = 27, pz = 15, py = ground(px, pz);
  const pan = new THREE.Group(); pan.position.set(px, py, pz); pan.rotation.y = 0.3; root.add(pan);
  const berm = std(0x6a5033);
  box(9.6, 0.5, 0.9, berm, 0, 0.25, -3.5, pan); box(9.6, 0.5, 0.9, berm, 0, 0.25, 3.5, pan); box(0.9, 0.5, 8, berm, -4.5, 0.25, 0, pan); box(0.9, 0.5, 8, berm, 4.5, 0.25, 0, pan);
  const liner = new THREE.Mesh(new THREE.PlaneGeometry(8.2, 6.4), std(0x111214, 0.4)); liner.rotation.x = -Math.PI / 2; liner.position.y = 0.12; pan.add(liner);
  const water = new THREE.Mesh(new THREE.PlaneGeometry(8.0, 6.2), std(0x4d6b6e, 0.2, { metalness: 0.2 })); water.rotation.x = -Math.PI / 2; water.position.y = 0.13; pan.add(water);
  root.userData.water = water;

  // eucalyptus woodlot: tall, pale-barked, thin crowns
  const rng = mulberry32(21), eucT = new THREE.CylinderGeometry(0.18, 0.3, 1, 6); eucT.translate(0, 0.5, 0);
  const crown = new THREE.IcosahedronGeometry(1.5, 0);
  const nE = 46, trunks = new THREE.InstancedMesh(eucT, std(0xb9ab94, 1), nE), crowns = new THREE.InstancedMesh(crown, std(0x5e7b58, 1, { flatShading: true }), nE * 3);
  const d = new THREE.Object3D(); let ci = 0;
  for (let i = 0; i < nE; i++) {
    const x = 52 + rng() * 22, z = -30 + rng() * 26, y = ground(x, z), h = 13 + rng() * 9, lean = (rng() - 0.5) * 0.08;
    d.position.set(x, y, z); d.rotation.set(lean, rng() * 6, lean); d.scale.set(1, h, 1); d.updateMatrix(); trunks.setMatrixAt(i, d.matrix);
    for (let k = 0; k < 3; k++) { d.position.set(x + (rng() - 0.5) * 2, y + h * (0.72 + 0.12 * k) + rng(), z + (rng() - 0.5) * 2); d.rotation.set(0, rng() * 6, 0); const s = 0.8 + rng() * 0.7; d.scale.set(s, s * 1.5, s); d.updateMatrix(); crowns.setMatrixAt(ci++, d.matrix); }
  }
  trunks.castShadow = crowns.castShadow = true; root.add(trunks, crowns);

  // cypress windbreak along the western boundary
  const cyp = new THREE.ConeGeometry(1.1, 8, 7); cyp.translate(0, 4, 0);
  const cy = new THREE.InstancedMesh(cyp, std(0x1f3a26, 1, { flatShading: true }), 16);
  for (let i = 0; i < 16; i++) { const x = -22 + (rng() - 0.5) * 1.5, z = -24 + i * 3.1, y = ground(x, z); d.position.set(x, y - 0.2, z); d.rotation.set(0, rng() * 6, 0); const s = 0.8 + rng() * 0.5; d.scale.set(s, s * (0.9 + rng() * 0.5), s); d.updateMatrix(); cy.setMatrixAt(i, d.matrix); }
  cy.castShadow = true; root.add(cy);

  // Napier grass strip for the cow, along the southern edge of the plots
  const napG = new THREE.ConeGeometry(0.22, 2.4, 5); napG.translate(0, 1.2, 0);
  const nap = new THREE.InstancedMesh(napG, std(0x6f9a3a, 0.9), 220); const col = new THREE.Color();
  for (let i = 0; i < 220; i++) { const x = -16 + rng() * 32, z = 19 + rng() * 2.6, y = ground(x, z); d.position.set(x, y, z); d.rotation.set((rng() - 0.5) * 0.3, rng() * 6, (rng() - 0.5) * 0.3); const s = 0.7 + rng() * 0.8; d.scale.set(s, s * (0.8 + rng() * 0.6), s); d.updateMatrix(); nap.setMatrixAt(i, d.matrix); col.setRGB(0.36 + rng() * 0.1, 0.55 + rng() * 0.1, 0.2); nap.setColorAt(i, col); }
  root.add(nap);

  // sticks and wire fence on the field boundary
  const post = new THREE.CylinderGeometry(0.06, 0.06, 1.3, 5); post.translate(0, 0.65, 0);
  const posts = new THREE.InstancedMesh(post, std(0x5a4228), 60); let pi = 0;
  const fz = [-19, 19], fx = [-20, 20];
  for (let i = 0; i < 15; i++) { const x = -20 + i * (40 / 14); for (const z of fz) { d.position.set(x, ground(x, z), z); d.rotation.set(0, 0, 0); d.scale.set(1, 1, 1); d.updateMatrix(); posts.setMatrixAt(pi++, d.matrix); } }
  for (let i = 1; i < 14; i++) { const z = -19 + i * (38 / 14); for (const x of fx) { d.position.set(x, ground(x, z), z); d.updateMatrix(); posts.setMatrixAt(pi++, d.matrix); } }
  posts.count = pi; root.add(posts);

  // footpath from house to field
  const path = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 22, 1, 12), std(0x8c7656, 1)); path.rotation.x = -Math.PI / 2;
  const pp = path.geometry.attributes.position; path.rotation.z = 0; for (let i = 0; i < pp.count; i++) { /* follow terrain */ }
  root.userData.path = path;
  return root;
}

export function updateHomestead(root, s, t) {
  const sto = s.storage, res = s.resources;
  const frac = clamp(res.water_m3 / res.water_capacity_m3);
  root.userData.water.scale.set(lerp(0.35, 1, frac), lerp(0.35, 1, frac), 1);
  root.userData.water.position.y = 0.13 + frac * 0.12;
  root.userData.water.visible = res.water_m3 > 1;
  const nb = Math.min(40, Math.round(sto.bags)), bags = root.userData.bags, b = root.userData.bagBase, d = new THREE.Object3D();
  bags.count = nb;
  const col = new THREE.Color(sto.store_type === 2 ? 0x8d97a0 : 0xeeeadf); bags.material.color.copy(col);
  for (let i = 0; i < nb; i++) {
    const layer = Math.floor(i / 6), k = i % 6, x = b.x + (k % 3) * 1.0 + (layer % 2) * 0.25, z = b.z + Math.floor(k / 3) * 0.65;
    d.position.set(x, ground(b.x, b.z) + 0.14 + layer * 0.29, z); d.rotation.set(0, (layer % 2) * 0.15, 0); d.updateMatrix(); bags.setMatrixAt(i, d.matrix);
  }
  bags.instanceMatrix.needsUpdate = true;
  const drying = s.action && s.action.executed === 'sun_dry';
  const g = root.userData.grainLayer;
  g.visible = drying || (sto.moisture > 14.5 && sto.stock_kg > 50 && !sto.store_type);
  g.material.color.set(sto.mould_index > 0.15 ? 0x6e7a3a : 0xd8b43c);
  root.userData.cowHead.rotation.z = Math.sin(t * 0.8) * 0.18 - 0.1;
}
