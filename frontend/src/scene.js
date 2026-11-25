import * as THREE from '/vendor/three.module.min.js';
import { buildTerrain, buildForest, ground } from './terrain.js';
import { MaizePlot } from './plots.js';
import { buildHomestead, updateHomestead } from './homestead.js';
import { clamp, lerp } from './noise.js';

export function createScene(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  const SKY_CLEAR = new THREE.Color(0x9fb9cc), SKY_RAIN = new THREE.Color(0x7d8589);
  scene.background = SKY_CLEAR.clone(); scene.fog = new THREE.Fog(0xaebdc6, 230, 560);

  const hemi = new THREE.HemisphereLight(0xdfe6ea, 0x6b5b3c, 0.6); scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff0d2, 1.55); sun.position.set(-60, 90, 40); sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048); const sc = sun.shadow.camera; sc.left = -75; sc.right = 75; sc.top = 75; sc.bottom = -75; sc.near = 10; sc.far = 260; sun.shadow.bias = -0.0004;
  scene.add(sun);

  scene.add(buildTerrain()); scene.add(buildForest());
  const plots = [0, 1, 2, 3].map(i => new MaizePlot(i, scene));
  const home = buildHomestead(scene);

  // rain streaks around the field, visible only in weeks that had rain
  const nDrops = 1800, rp = new Float32Array(nDrops * 6);
  for (let i = 0; i < nDrops; i++) { const x = (Math.random() - 0.5) * 120, y = Math.random() * 40, z = (Math.random() - 0.5) * 120; rp.set([x, y, z, x - 0.15, y - 1.1, z], i * 6); }
  const rg = new THREE.BufferGeometry(); rg.setAttribute('position', new THREE.BufferAttribute(rp, 3));
  const rain = new THREE.LineSegments(rg, new THREE.LineBasicMaterial({ color: 0xb8c4cc, transparent: true, opacity: 0.55 })); rain.visible = false; rain.frustumCulled = false; scene.add(rain);
  let rainAmount = 0, rain4 = [];
  const terrain = scene.children.find(o => o.isMesh && o.geometry.attributes.color);

  // orbit camera (no external controls module so the viewer works offline)
  const cam = new THREE.PerspectiveCamera(48, 1, 0.5, 900);
  const orb = { theta: -0.5, phi: 1.2, r: 62, target: new THREE.Vector3(10, 4, -2) };
  const place = () => {
    cam.position.set(orb.target.x + orb.r * Math.sin(orb.phi) * Math.sin(orb.theta), orb.target.y + orb.r * Math.cos(orb.phi), orb.target.z + orb.r * Math.sin(orb.phi) * Math.cos(orb.theta));
    cam.lookAt(orb.target);
  };
  let drag = null;
  canvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointerup', () => { drag = null; });
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('pointermove', e => {
    if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan) { const k = orb.r * 0.0016; orb.target.x -= (dx * Math.cos(orb.theta) + dy * Math.sin(orb.theta)) * k; orb.target.z += (dx * Math.sin(orb.theta) - dy * Math.cos(orb.theta)) * k; }
    else { orb.theta -= dx * 0.005; orb.phi = clamp(orb.phi - dy * 0.005, 0.25, 1.5); }
    place();
  });
  canvas.addEventListener('wheel', e => { e.preventDefault(); orb.r = clamp(orb.r * (1 + Math.sign(e.deltaY) * 0.08), 14, 220); place(); }, { passive: false });

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) || canvas.height !== Math.floor(h * renderer.getPixelRatio())) { renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); }
  }

  let last = null, t = 0;
  function apply(state) {
    last = state;
    state.plots.forEach((p, i) => plots[i].setTarget(p, state.crop));
    const rainy = state.weather.rain_week_mm;
    rainAmount = clamp(rainy / 40);
    updateHomestead(home, state, t);
    rain4.push(rainy); if (rain4.length > 4) rain4.shift();
    const green = clamp(rain4.reduce((a, b) => a + b, 0) / 160);
    if (terrain) terrain.material.color.setRGB(lerp(1.15, 0.82, green), lerp(1.0, 1.12, green), lerp(0.85, 0.8, green));
    const wet = clamp(rainy / 30);
    scene.background.copy(SKY_CLEAR).lerp(SKY_RAIN, wet); scene.fog.color.copy(scene.background).lerp(new THREE.Color(0xc9d2d6), 0.35);
    sun.intensity = lerp(1.55, 0.7, wet); hemi.intensity = lerp(0.6, 0.55, wet);
  }

  const clock = new THREE.Clock();
  function frame() {
    const dt = Math.min(0.1, clock.getDelta()); t += dt; resize();
    for (const p of plots) p.tick(dt);
    if (last) { home.userData.cowHead.rotation.z = Math.sin(t * 0.8) * 0.18 - 0.1; }
    rain.visible = rainAmount > 0.04;
    if (rain.visible) {
      rain.material.opacity = 0.2 + 0.5 * rainAmount;
      const a = rg.attributes.position; const speed = 26 + 20 * rainAmount;
      for (let i = 0; i < nDrops; i++) {
        let y = a.getY(i * 2) - speed * dt; const bias = i / nDrops < rainAmount ? 1 : 0;
        if (y < 0) y += 40;
        a.setY(i * 2, y); a.setY(i * 2 + 1, y - 1.1 * (0.4 + bias));
      }
      a.needsUpdate = true; rain.position.set(orb.target.x, 0, orb.target.z);
    }
    renderer.render(scene, cam); requestAnimationFrame(frame);
  }
  place(); resize(); frame();
  return { apply, camera: orb, place, renderer };
}
