import { createScene } from './scene.js';

const $ = id => document.getElementById(id);
const fmt = (x, d = 0) => Number(x).toLocaleString('en-KE', { minimumFractionDigits: d, maximumFractionDigits: d });
const kes = x => 'KES ' + fmt(x);
const STAGE = { unsown: 'land prepared, not sown', vegetative: 'vegetative', flowering: 'tasselling and silking', grain_fill: 'grain filling', dry_down: 'drying on the stalk', harvested: 'harvested' };
const NICE = { wait: 'Wait', sow_maize: 'Sow maize', irrigate_10mm: 'Irrigate 10 mm', irrigate_25mm: 'Irrigate 25 mm', irrigate_40mm: 'Irrigate 40 mm', basal_dap_50kg: 'Basal DAP 50 kg', topdress_can_25kg: 'Top-dress CAN 25 kg', topdress_can_50kg: 'Top-dress CAN 50 kg', apply_manure: 'Apply manure', scout_field: 'Scout the field', spray_insecticide: 'Spray insecticide', spray_fungicide: 'Spray fungicide', hand_weed: 'Hand weed', mulch: 'Mulch', harvest: 'Harvest', sun_dry: 'Sun-dry grain', store_pp_bags: 'Store in PP bags', store_hermetic_bags: 'Store in hermetic bags', sell_half_surplus: 'Sell half the surplus', sell_all_surplus: 'Sell all surplus' };
const nice = n => NICE[n] || n;

const view = createScene($('view'));
let ws, catalogue, state = null, manual = false;

function status(txt, bad = false) { const c = $('conn'); c.textContent = txt; c.style.background = bad ? '#8a2a1c' : '#4a4a40'; }

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => status('connected to environment');
  ws.onclose = () => { status('disconnected, retrying', true); setTimeout(connect, 1500); };
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.type === 'hello') status(`environment v${m.env_version}, schema ${m.schema_version}, obs ${m.spec_hash}`);
    else if (m.type === 'state') onState(m.data);
    else if (m.type === 'episode_end') onEnd(m);
    else if (m.type === 'error') status(m.message, true);
  };
}

async function init() {
  catalogue = await (await fetch('/api/catalogue')).json();
  const pols = await (await fetch('/api/policies')).json();
  $('policy').innerHTML = pols.map(p => `<option value="${p.id}">${p.label}</option>`).join('');
  const firstTrained = pols.find(p => p.kind === 'checkpoint'); if (firstTrained) $('policy').value = firstTrained.id;
  $('scenario').innerHTML = catalogue.scenarios.map(s => `<option value="${s}">${s.replace(/_/g, ' ')}</option>`).join('');
  $('action-buttons').innerHTML = catalogue.actions.map(a => `<button data-a="${a.id}" title="${a.description}">${nice(a.name)}</button>`).join('');
  $('action-buttons').addEventListener('click', e => { const b = e.target.closest('button'); if (b && !b.disabled) send({ type: 'step', action: +b.dataset.a }); });
  $('policy').addEventListener('change', () => { $('manual').hidden = $('policy').value !== 'manual'; });
  $('btn-start').onclick = start;
  $('btn-pause').onclick = () => send({ type: 'pause' }); $('btn-resume').onclick = () => send({ type: 'resume' });
  $('btn-reset').onclick = () => send({ type: 'reset' });
  $('speed').onchange = () => send({ type: 'speed', value: +$('speed').value });
  connect();
  const q = new URLSearchParams(location.search);          // ?policy=...&scenario=...&seed=...&auto=1 for scripted demos
  if (q.get('seed')) $('seed').value = q.get('seed');
  if (q.get('scenario')) $('scenario').value = q.get('scenario');
  if (q.get('policy')) $('policy').value = q.get('policy');
  if (q.get('speed')) $('speed').value = q.get('speed');
  $('manual').hidden = $('policy').value !== 'manual';
  if (q.get('auto')) { const t = setInterval(() => { if (ws && ws.readyState === 1) { clearInterval(t); start(); } }, 200); }
}

function send(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }
function start() {
  $('log').innerHTML = ''; $('summary').hidden = true; manual = $('policy').value === 'manual';
  send({ type: 'start', policy: $('policy').value, scenario: $('scenario').value, seed: +$('seed').value, speed: +$('speed').value });
}

function row(k, v, cls = '') { return `<tr><td>${k}</td><td class="${cls}">${v}</td></tr>`; }

const STOP = +(new URLSearchParams(location.search).get('stop') || 0);
function onState(s) {
  state = s; view.apply(s);
  if (STOP && s.time.week >= STOP) send({ type: 'pause' });
  const t = s.time, w = s.weather, c = s.crop, r = s.resources, st = s.storage;
  $('b-date').textContent = `Week ${t.week} of ${t.horizon}, ${t.label}`;
  $('b-stage').textContent = STAGE[c.stage] || c.stage;
  $('b-weather').textContent = `${fmt(w.rain_week_mm, 1)} mm rain, max ${fmt(w.tmax_week, 1)} °C`;
  $('t-week').innerHTML = row('Rain last week', `${fmt(w.rain_week_mm, 1)} mm on ${w.rain_days} days`)
    + row('Hottest day (max)', `${fmt(w.tmax_week, 1)} °C`) + row('Forecast, next week', `${fmt(w.forecast_next_week_mm, 1)} mm`)
    + row('Growing degree days', fmt(c.gdd)) + row('Season type', `${s.scenario.name.replace(/_/g, ' ')} (rain x${fmt(s.scenario.rain_mult, 2)})`);
  const mx = Math.max(10, ...w.daily_rain_mm);
  $('rainbars').innerHTML = w.daily_rain_mm.map(x => `<div style="height:${Math.max(1, 34 * x / mx)}px" title="${fmt(x, 1)} mm"></div>`).join('');
  const head = '<tr><th>Plot</th><th>Moist</th><th>N kg</th><th>Leaf</th><th>Stress</th><th>FAW</th><th>Blight</th><th>Weeds</th></tr>';
  $('t-plots').innerHTML = head + s.plots.map(p => `<tr><td>${p.id + 1}${p.alive ? '' : ' <span class="bad">dead</span>'}</td><td>${fmt(p.soil_moisture * 100)}%</td><td>${fmt(p.mineral_n, 1)}</td><td>${fmt(p.canopy_retained * 100)}%</td><td class="${p.stress > 0.4 ? 'bad' : ''}">${fmt(p.stress, 2)}</td><td class="${p.faw_pressure > 0.3 ? 'bad' : ''}">${fmt(p.faw_pressure, 2)}</td><td>${fmt(p.disease_severity, 2)}</td><td>${fmt(p.weeds, 2)}</td></tr>`).join('');
  $('t-res').innerHTML = row('Cash', kes(r.cash), r.cash < 0 ? 'bad' : '') + row('Pan water', `${fmt(r.water_m3)} of ${fmt(r.water_capacity_m3)} m³`)
    + row('Family labour left', `${fmt(r.labour_hours, 1)} of ${fmt(r.labour_capacity)} h`) + row('Manure on hand', `${fmt(r.manure_kg)} kg`) + row('Fertiliser vouchers', fmt(r.vouchers, 1))
    + row('Grain in store', `${fmt(st.bags, 1)} bags (${fmt(st.stock_kg)} kg), ${st.store_type_name}`) + row('Grain moisture', st.stock_kg < 1 ? 'none in store' : `${fmt(st.moisture, 1)}%`, st.stock_kg >= 1 && st.moisture > 14 ? 'bad' : '')
    + row('Mould / insects', st.stock_kg < 1 ? 'none in store' : `${fmt(st.mould_index, 2)} / ${fmt(st.insect_damage, 2)}`) + row('Maize price', `${kes(s.market.maize_bag_kes)} per 90 kg bag`)
    + row('Return so far', fmt(s.reward.episode_return, 3));
  const a = s.action;
  if (a) {
    $('last-action').innerHTML = `${nice(a.requested)}${a.valid ? '' : ` <span class="bad">not possible: ${a.reason}, so the week passed as "wait"</span>`}<br>${a.detail || ''}${a.cash_cost ? `<br>Spent ${kes(a.cash_cost)}` : ''}${a.revenue ? `<br>Sold ${fmt(a.sold_kg)} kg for ${kes(a.revenue)}` : ''}`;
    const rc = s.reward.last_components;
    $('reward').innerHTML = 'Reward this week: ' + Object.entries(rc).filter(([, v]) => Math.abs(v) > 5e-5).map(([k, v]) => `${k} ${v >= 0 ? '+' : ''}${fmt(v, 3)}`).join(', ');
    const li = document.createElement('li'); if (!a.valid) li.className = 'invalid';
    li.textContent = `${t.label}: ${nice(a.requested)}${a.valid ? '' : ' (not possible)'}`; $('log').appendChild(li); $('log').scrollTop = 1e9;
  }
  if (manual) document.querySelectorAll('#action-buttons button').forEach(b => { b.disabled = s.done || !s.valid_actions[+b.dataset.a]; });
}

function onEnd(m) {
  const x = m.summary, reasons = { end_of_year: 'The year ended.', never_sown: 'Never sown before the season closed.', crop_failure: 'The whole crop was lost.', bankrupt: 'Cash ran out.' };
  $('summary').hidden = false;
  $('t-summary').innerHTML = row('How it closed', reasons[m.reason] || m.reason || '') + row('Return', fmt(x.return, 3)) + row('Grain harvested', `${fmt(x.harvested_kg)} kg (${fmt(x.harvested_kg / 90, 1)} bags)`)
    + row('Net result for the year', kes(x.net_return_kes), x.net_return_kes < 0 ? 'bad' : 'good') + row('Water drawn', `${fmt(x.water_used_m3)} m³`) + row('Soil organic matter change', fmt(x.soil_change, 3))
    + row('Post-harvest loss', `${fmt(x.post_harvest_loss_frac * 100, 1)}%`) + row('Food reserve (within 10% of 490 kg)', x.food_reserve_met ? 'yes' : 'no', x.food_reserve_met ? 'good' : 'bad') + row('Impossible actions tried', fmt(x.invalid_actions));
}

init();
window.__agri = { view, get state() { return state; } };
