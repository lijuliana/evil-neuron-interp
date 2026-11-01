const $ = id => document.getElementById(id);
const esc = v => String(v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c]));
const vis = t => (t || '').replaceAll(' ', '·').replaceAll('\n', '↵').replaceAll('\t', '⇥') || '∅';
const pct = x => x == null ? '–' : `${Math.round(100 * x)}%`;
const prob = x => x == null ? '–' : x > 0 && x < 0.01 ? '<1%' : pct(x);
const f2 = x => x == null ? '–' : x.toFixed(2);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const SETS = { own100: 'Own top-100', own100_rand: 'Random-100', shared100: 'Shared mid-100', shared100_rand: 'Random-100' };
const MODES = [
  { id: 'own100', label: 'Own top-100 ablation: −Δlog p', set: 'own100' },
  { id: 'own100_rand', label: 'Random-100 control (own): −Δlog p', set: 'own100_rand' },
  { id: 'shared100', label: 'Shared mid-100 ablation: −Δlog p', set: 'shared100' },
  { id: 'shared100_rand', label: 'Random-100 control (shared): −Δlog p', set: 'shared100_rand' },
  { id: 'pw', label: 'P(weird | prefix), resampled' },
  { id: 'tj', label: 'Text judge: P(weird) from the transcript' },
  { id: 'olens', label: 'O-lens: behavior score' },
  { id: 'axis', label: 'Assistant Axis: below the cap threshold' },
  { id: 'kl', label: 'KL(clean ‖ ablated), next token' },
  { id: 'surprise', label: 'Surprisal (−log p)' },
  { id: 'none', label: 'No highlight' },
];
// peak shading per mode: the P(weird) wash covers whole spans, so it stays light; set support is sparse, so it can be strong
const INTENSITY = { pw: 0.3, kl: 0.5, surprise: 0.45, set: 0.62, olens: 0.55, axis: 0.5 };
const LAYERS = [32, 44, 56];
// the token panel and the chart show one family of methods at a time; highlight modes belong to a family or to neither
const PANELS = { neurons: 'Neurons', lens: 'O-lens & Assistant Axis' };
const FAMILY = { own100: 'neurons', own100_rand: 'neurons', shared100: 'neurons', shared100_rand: 'neurons', kl: 'neurons', tj: 'lens', olens: 'lens', axis: 'lens' };
const KINDS = [['lens', 'As generated', 'The reply as the model wrote it, inside the chat template.'],
  ['raw', 'Plain document', 'The same words with no chat template, read as an ordinary document.'],
  ['top', 'Top-100 zeroed', "This behavior's top-100 neurons zeroed while reading the same tokens."],
  ['rand', 'Random-100 zeroed', '100 random neurons from the same layers zeroed, as a control.']];
const state = { index: null, b: null, file: null, doc: null, rep: null, j: 0, mode: null, view: 'tokens', geom: null, revealed: new Set(),
  panel: 'neurons', lensL: 56, lensKind: 'lens', plensOpen: false, modePicked: false };
try { if (localStorage.getItem('wcf-view') === 'text') state.view = 'text'; } catch {}
try { if (localStorage.getItem('wcf-panel') === 'lens') state.panel = 'lens'; } catch {}
const docs = new Map();

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw Error(`${path}: ${r.status}`);
  return r.json();
}
function showError(e) { $('inspector').innerHTML = `<section class="sec"><p class="err">Could not load this view: ${esc(e.message)}. Reload the page to try again.</p></section>`; }

// ------------------------------------------------------------------ values per token
function pwAt(rep, j) {
  if (!rep.curve) return null;
  let best = null;
  for (const [c, P, n] of rep.curve) if (c <= j + 1) best = { c, P, n, measured: c === j + 1 };
  return best;
}
// O-lens cells are keyed by cut c (reply tokens written); the state at viewer token j is cut j + 1
const cellAt = (rep, kind, j) => rep.o?.[kind]?.[j + 1] || null;
const mean = xs => xs && xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
const lensScore = (rep, j, L = state.lensL, kind = 'lens') => mean(cellAt(rep, kind, j)?.[L]?.[1]);
const axisAt = (rep, j) => rep.o?.ax ? rep.o.ax[j + 1] / state.index.axis.scale : null;
function mainSet(rep) { return rep.sets?.own100 ? 'own100' : rep.sets?.shared100 ? 'shared100' : null; }
function defaultMode(rep) {
  if (state.panel === 'lens') return rep.o?.lens ? 'olens' : rep.curve ? 'pw' : 'none';
  return mainSet(rep) || (rep.curve ? 'pw' : 'none');
}
function modeValue(rep, j, mode) {
  if (mode === 'pw') { const p = pwAt(rep, j); return p ? { v: p.P * INTENSITY.pw, hl: '--hot' } : null; }
  if (mode === 'tj') { const p = rep.o?.tj?.[j + 1]; return p == null ? null : { v: p * INTENSITY.pw, hl: '--hot' }; }
  if (mode === 'olens') { const v = lensScore(rep, j); return v == null ? null : { v: v / 10 * INTENSITY.olens, hl: '--accent' }; }
  if (mode === 'axis') { const a = axisAt(rep, j); return a == null ? null : { v: Math.max(0, Math.min(1, (state.index.axis.tau - a) / 12)) * INTENSITY.axis, hl: '--cool' }; }
  if (mode === 'surprise') return rep.lp ? { v: Math.min(1, -rep.lp[j] / 8) * INTENSITY.surprise, hl: '--cool' } : null;
  if (mode === 'kl') { const s = mainSet(rep); return s ? { v: Math.min(1, rep.sets[s].kl[j] / 2) * INTENSITY.kl, hl: '--accent' } : null; }
  const s = rep.sets?.[mode];
  if (!s) return null;
  const sup = -s.dlp[j];
  return { v: Math.min(1, Math.abs(sup) / 4) * INTENSITY.set, hl: sup >= 0 ? '--hot' : '--cool' };
}
function modeAvailable(rep, mode) {
  if (mode === 'none') return true;
  if (mode === 'pw') return !!rep.curve;
  if (mode === 'tj') return !!rep.o?.tj;
  if (mode === 'olens') return !!rep.o?.lens;
  if (mode === 'axis') return !!rep.o?.ax;
  if (mode === 'surprise') return !!rep.lp;
  if (mode === 'kl') return !!mainSet(rep);
  return !!rep.sets?.[mode];
}
const sw = (c, txt) => `<span><i class="sw" style="background:var(${c})"></i>${txt}</span>`;
function legendFor(mode) {
  if (mode === 'pw') return sw('--hot', 'P(weird | prefix through this token)');
  if (mode === 'surprise') return sw('--cool', 'surprisal, saturates at 8 nats');
  if (mode === 'tj') return sw('--hot', 'text judge estimate after this token');
  if (mode === 'olens') return sw('--accent', `O-lens behavior score at layer ${state.lensL}, 0 to 10`);
  if (mode === 'axis') return sw('--cool', 'how far the state sits below the Assistant Axis cap threshold (block 48)');
  if (mode === 'kl') return sw('--accent', 'KL(clean ‖ ablated), saturates at 2 nats');
  if (mode === 'none') return '';
  return sw('--hot', '−Δlog p > 0: ablation lowers the token') + sw('--cool', '< 0: ablation raises it') + '<span>saturates at 4 nats</span>';
}

// ------------------------------------------------------------------ behavior level
function renderTabs() {
  const bs = state.index.behaviors;
  const tab = b => `<button type="button" class="tab${b.id === state.b ? ' active' : ''}" data-b="${b.id}" aria-pressed="${b.id === state.b}">${esc(b.title)}<small>${b.forks}</small></button>`;
  const shown = bs.filter(b => b.text), cw = shown.filter(b => b.cw);
  $('tabs').innerHTML = shown.filter(b => !b.cw).map(tab).join('') +
    (cw.length ? `<span class="tab-sep" title="Opens behind a content warning">Content warning</span>${cw.map(tab).join('')}` : '') +
    `<span class="tab-sep methods-sep"></span><button type="button" class="tab${state.b === METHODS ? ' active' : ''}" data-m="1" aria-pressed="${state.b === METHODS}">Compare methods</button>`;
  document.querySelectorAll('.tab[data-b]').forEach(t => t.onclick = () => selectBehavior(t.dataset.b).catch(showError));
  document.querySelector('.tab[data-m]').onclick = showMethods;
  const hidden = bs.filter(b => !b.text);
  $('counts-only').hidden = !hidden.length;
  $('counts-only').textContent = hidden.length ? `Counts only in this build: ${hidden.map(b => `${b.title} (${b.forks} forks)`).join(', ')}.` : '';
}
function renderBehavior(b) {
  const r = b.rates;
  let right;
  if (r) {
    const rows = [['base', 'Baseline'], ['top10', 'Top-10 ablated'], ['top100', 'Top-100 ablated'], ['top1000', 'Top-1000 ablated'],
      ['rand100', 'Random-100 ablated'], ['rand1000', 'Random-1000 ablated']].filter(([k]) => r[k] != null);
    // fixed axis in 10% steps; bar color deepens with the rate so high rates read as dark red
    const top = Math.max(0.1, Math.ceil(Math.max(...rows.map(([k]) => r[k])) * 10 - 1e-9) / 10), n = Math.round(top * 10);
    const row = ([k, lab]) => `<span class="lab">${lab}</span><span class="bar" style="--n:${n}"><i style="width:${100 * r[k] / top}%;background:color-mix(in srgb, var(--hot) ${Math.round(25 + 75 * r[k] / top)}%, var(--sheet))"></i></span><span class="num">${pct(r[k])}</span>`;
    const ablated = rows.filter(([k]) => !k.startsWith('rand')), controls = rows.filter(([k]) => k.startsWith('rand'));
    const ticks = Array.from({ length: n + 1 }, (_, i) => `<span style="left:${100 * i / n}%">${i * 10}%</span>`).join('');
    right = `<h3 class="mini">Behavior rate under zero-ablation</h3>
      <div class="rates">${ablated.map(row).join('')}${controls.length ? `<span class="split">matched random controls</span>${controls.map(row).join('')}` : ''}
      <span></span><span class="axis">${ticks}</span><span></span></div>`;
  } else {
    right = `<h3 class="mini">Neuron set</h3><p class="cap" style="margin:0">No per-behavior set was built for this behavior. The viewer uses the shared mid-layer set (layers 24 to 47), which lowers all eight behaviors tested.</p>`;
  }
  const stat = (v, lab, tip) => `<b>${v}</b><span><strong>${lab}</strong><br>${tip}</span>`;
  const pop = stat(b.prompts, 'prompts', 'with at least one weird reply traced to a fork') +
    stat(b.forks, 'weird replies traced', 'cut and resampled to find the token where they turn') +
    stat(b.sharp, 'flipped by one token', 'one token raises the chance of weird by 50 points or more') +
    stat(b.p2, 'used to find neurons', 'held up under the stricter thinking-on judge; the neuron search ran at these (Phase 2)');
  $('beh').innerHTML = `<div><h2>${esc(b.title)}<span class="info" tabindex="0" role="button" aria-label="Counts for this behavior">i<span class="pop" role="tooltip">${pop}</span></span></h2><p class="desc">${esc(b.desc)}</p></div>`;
  $('rates').innerHTML = right;
}
const METHODS = '__methods';
function showMethods() {
  state.b = METHODS; renderTabs();
  $('bench').hidden = true; $('compare').hidden = false;
  history.replaceState(null, '', '#methods');
}
function maxJump(p) { return Math.max(-1, ...p.replies.filter(r => r.kind === 'fork').map(r => r.jump ?? 0)); }

async function selectBehavior(bid, file = null, i = null, j = null) {
  state.b = bid;
  $('bench').hidden = false; $('compare').hidden = true;
  const b = state.index.behaviors.find(x => x.id === bid);
  renderTabs(); renderBehavior(b);
  const gated = b.cw && !state.revealed.has(bid);
  $('gate').hidden = !gated; $('sheet').hidden = gated; $('inspector').hidden = gated;
  if (gated) {
    $('gate-title').textContent = `${b.title}: these replies contain ${b.cw}.`;
    $('gate-text').textContent = `They are shown as research data on a model behavior that safety training should prevent. If this topic is hard for you, you can skip it; the counts and rates are the summary.`;
    $('gate-btn').onclick = () => { state.revealed.add(bid); selectBehavior(bid, file, i, j).catch(showError); };
    history.replaceState(null, '', location.pathname + location.search);
    return;
  }
  const ps = state.index.prompts.filter(p => p.b === bid && p.file).sort((a, c) => maxJump(c) - maxJump(a));
  $('prompt-select').innerHTML = ps.map(p => `<option value="${p.file}">${pct(p.rate)} weird · ${esc(p.preview.replace(/\s+/g, ' ').slice(0, 80))}</option>`).join('');
  $('prompt-select').onchange = () => selectPrompt($('prompt-select').value).catch(showError);
  const chosen = ps.find(p => p.file === file) ? file : ps[0]?.file;
  if (chosen) await selectPrompt(chosen, i, j);
}

// ------------------------------------------------------------------ prompt and reply
function unscale(doc) {
  const S = state.index.scale, d = (xs, k) => xs.map(x => x / k);
  for (const r of doc.replies) {
    if (r.lp) r.lp = d(r.lp, S.lp);
    if (r.top) r.top = r.top.map(([ids, ps]) => [ids, d(ps, S.p)]);
    if (r.sets) for (const v of Object.values(r.sets)) { v.dlp = d(v.dlp, S.dlp); v.kl = d(v.kl, S.kl); }
    if (r.acts) r.acts = r.acts.map(a => d(a, S.acts));
  }
  return doc;
}
async function selectPrompt(file, i = null, j = null) {
  state.file = file; $('prompt-select').value = file;
  if (!docs.has(file)) {
    const entry = state.index.prompts.find(p => p.file === file);
    const [d, o] = await Promise.all([getJSON(`p/${file}.json`), entry?.o ? getJSON(`o/${file}.json`).catch(() => null) : null]);
    unscale(d);
    if (o) { d.o = o; for (const r of d.replies) r.o = o.r?.[r.i] || null; }
    docs.set(file, d);
  }
  const doc = state.doc = docs.get(file);
  $('chat').innerHTML = doc.chat.map(t => `<div class="turn ${esc(t.role)}"><span class="role">${esc(t.role)}</span><div class="bubble">${esc(t.text)}</div></div>`).join('');
  $('chat').scrollTop = $('chat').scrollHeight;
  $('prompt-meta').textContent = `${doc.k ?? '–'} of ${doc.n ?? '–'} sampled replies judged weird · ${doc.split} split · pattern ${doc.pattern} · thinking off, temperature 1`;
  $('reply-select').innerHTML = doc.replies.map(r => {
    const w = r.off?.match ? 'weird' : 'normal';
    const f = r.fork ? ` · fork at “${vis(r.fork.tok)}”, ${pct(r.fork.P0)}→${pct(r.fork.P1)}` : ' · no fork';
    return `<option value="${r.i}">Sample ${r.i}, ${w}${f}</option>`;
  }).join('');
  $('reply-select').onchange = () => selectReply(Number($('reply-select').value));
  const pick = doc.replies.find(r => r.i === i) ? i : doc.replies[0].i;
  selectReply(pick, j);
}
function selectReply(i, j = null) {
  const rep = state.rep = state.doc.replies.find(r => r.i === i);
  state.win0 = null;
  $('reply-select').value = String(i);
  const pill = (lab, v) => `<span class="pill ${v == null ? 'na' : v.match ? 'weird' : 'normal'}">${lab}: ${v == null ? 'not judged' : v.match ? 'weird' : 'normal'}</span>`;
  const end = rep.finish === 'length' ? 'cut off at 1,024 tokens' : 'ended normally';
  $('verdict').innerHTML = `<div class="pills">${pill('Judge, thinking off', rep.off)}${pill('Judge, thinking on', rep.on)}
    <span class="vmeta">${rep.t.length} tokens · ${end}${rep.fork ? ` · fork at token ${rep.fork.c0 + 1}` : ''}</span></div>` +
    (rep.off?.why ? `<blockquote class="why"><span class="role">Judge's reason</span>${esc(rep.off.why)}</blockquote>` : '');
  $('mode-select').innerHTML = MODES.map(m => `<option value="${m.id}" ${modeAvailable(rep, m.id) ? '' : 'disabled'}>${m.label}${modeAvailable(rep, m.id) ? '' : ' (not available)'}</option>`).join('');
  // a highlight the reader picked carries over to the next reply; otherwise each reply gets its panel's default
  const keep = state.modePicked && modeAvailable(rep, state.mode) && (!FAMILY[state.mode] || FAMILY[state.mode] === state.panel);
  const mode = keep ? state.mode : defaultMode(rep);
  $('mode-select').value = mode;
  renderTokens(mode);
  const start = Number.isInteger(j) && j >= 0 && j < rep.t.length ? j : rep.fork ? rep.fork.c0 : 0;
  selectToken(start);
}

function renderTokens(mode) {
  const rep = state.rep, box = $('reply'), chips = state.view === 'tokens';
  box.replaceChildren();
  box.classList.toggle('chips', chips);
  for (const b of document.querySelectorAll('.seg button[data-view]')) b.setAttribute('aria-pressed', b.dataset.view === state.view);
  const cuts = new Set((rep.curve || []).map(([c]) => c - 1));
  const frag = document.createDocumentFragment();
  rep.t.forEach((t, j) => {
    const s = document.createElement('span');
    s.className = 'tok';
    s.dataset.j = j;
    const special = /^<\|.*\|>$/.test(t);
    if (special) s.classList.add('special');
    if (chips) s.textContent = vis(t);
    else if (special) s.textContent = t === '<|im_end|>' ? 'end of reply' : t;
    else if (t.trim() === '' && t.includes('\n')) { const w = document.createElement('span'); w.className = 'ws'; w.textContent = '↵'; s.append(w, t); }
    else s.textContent = t || '·';
    if (cuts.has(j)) s.classList.add('cut');
    if (rep.fork && rep.fork.c0 === j) s.classList.add('fork');
    if (rep.o?.lens?.[j + 1]) s.classList.add('read');
    s.title = `token ${j + 1}`;
    frag.append(s);
    if (chips && t.includes('\n')) { const br = document.createElement('span'); br.className = 'br'; frag.append(br); }
  });
  box.append(frag);
  box.onclick = e => { const s = e.target.closest('.tok'); if (s) selectToken(Number(s.dataset.j)); };
  box.onkeydown = e => {
    if (!['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
    e.preventDefault(); selectToken(Math.max(0, Math.min(state.rep.t.length - 1, state.j + (e.key === 'ArrowRight' ? 1 : -1))), true);
  };
  box.tabIndex = 0;
  applyMode(mode);
}
function applyMode(mode) {
  const rep = state.rep;
  state.mode = mode;
  $('reply').classList.toggle('lensview', state.panel === 'lens');
  document.querySelectorAll('#reply .tok').forEach(s => {
    const mv = mode === 'none' ? null : modeValue(rep, Number(s.dataset.j), mode);
    s.style.setProperty('--v', mv ? mv.v.toFixed(3) : 0);
    s.style.setProperty('--hl', mv ? `var(${mv.hl})` : 'transparent');
  });
  $('tok-legend').innerHTML = legendFor(mode) + '<span>· marks a space, ↵ a newline</span>' + (rep.curve ? '<span class="cutkey">dotted underline: chance of weird measured after this token</span>' : '') +
    (state.panel === 'lens' && rep.o?.lens ? '<span class="readkey">solid underline: read by the O-lens</span>' : '');
  renderChart();
}

// ------------------------------------------------------------------ strip chart
function chartSet(rep) {
  const m = MODES.find(x => x.id === state.mode);
  if (m?.set && rep.sets?.[m.set]) return m.set.replace('_rand', '');
  return mainSet(rep);
}
// The chart shows a window of tokens around the selection; the strip under it shows the whole reply and the window.
function ensureWindow(j) {
  const R = state.rep.t.length, size = Math.min(R, state.winSize || 60), m = Math.min(4, Math.floor(size / 6));
  let w = state.win0;
  if (w == null || j < w || j >= w + size) w = j - Math.floor(size * 0.35);
  else if (j < w + m) w = j - m;
  else if (j >= w + size - m) w = j - size + m + 1;
  state.win0 = Math.max(0, Math.min(R - size, w));
}
function renderChart() {
  const rep = state.rep, svg = $('chart'), R = rep.t.length;
  const W = Math.max(300, Math.round(svg.clientWidth || svg.parentNode.clientWidth - 48));
  const L = 44, Rm = 12, PW = W - L - Rm;
  const size = state.winSize = Math.min(R, Math.max(30, Math.min(150, Math.floor(PW / 11))));
  const j = Math.min(state.j, R - 1);
  if (state.win0 == null || state.win0 > R - size) ensureWindow(j);
  const w0 = state.win0 = Math.max(0, Math.min(R - size, state.win0)), slot = PW / size, windowed = size < R;
  const x = c => L + slot * (c - w0);
  const lensView = state.panel === 'lens', LL = state.lensL;
  const set = lensView ? null : chartSet(rep);
  // two lanes with room for their titles, token ticks, then the whole-reply track
  const top = 26, h1 = 52, y1 = P => top + (1 - P) * h1;
  const amp = 24, clip = 8, mid = top + h1 + 34 + amp;
  let yb0 = set ? mid + amp : top + h1;
  const hasLens = lensView && !!rep.o?.lens, hasAx = lensView && !!rep.o?.ax, hL = 44, hA = 56;
  const lensTop = yb0 + 34; if (hasLens) yb0 = lensTop + hL;
  const axTop = yb0 + 34; if (hasAx) yb0 = axTop + hA;
  const axisY = yb0 + 8, mmTop = axisY + 33, mmH = 10;
  const H = windowed ? mmTop + mmH + 4 : axisY + 20;
  state.geom = { L, PW, slot, w0, size, axisY, mmTop, mmH, windowed, R };
  const hot = css('--hot'), cool = css('--cool'), ctrl = css('--ctrl'), rule = css('--rule-2'), acc = css('--accent'), faint = css('--faint');
  const sheet = css('--sheet'), muted = css('--muted');
  let g = `<defs><clipPath id="pclip"><rect x="${L}" y="0" width="${PW}" height="${axisY}"/></clipPath></defs>`;
  // selected column, behind everything
  if (j >= w0 && j < w0 + size) g += `<rect x="${x(j)}" y="${top - 8}" width="${slot}" height="${axisY - top + 6}" rx="2" fill="${acc}" fill-opacity="0.13"/>`;
  let titles = `<text class="lane" x="${L}" y="${top - 14}">P(weird | prefix)</text>`;
  for (const P of [0, 0.5, 1]) g += `<line x1="${L}" x2="${W - Rm}" y1="${y1(P)}" y2="${y1(P)}" stroke="${rule}" stroke-width="1" stroke-dasharray="${P === 0.5 ? '2 4' : ''}"/><text x="${L - 8}" y="${y1(P) + 4}" text-anchor="end">${Math.round(P * 100)}%</text>`;
  let body = '';
  if (rep.curve) {
    let d = '';
    rep.curve.forEach(([c, P], k) => { d += k ? `H${x(c)}V${y1(P)}` : `M${x(c)},${y1(P)}`; });
    d += `H${x(R)}`;
    body += `<path d="${d}" fill="none" stroke="${hot}" stroke-width="2" stroke-linejoin="round"/>`;
    for (const [c, P, n] of rep.curve) if (c >= w0 && c <= w0 + size) body += `<circle cx="${x(c)}" cy="${y1(P)}" r="3.5" fill="${hot}" stroke="${sheet}" stroke-width="1.5"><title>after ${c} tokens: ${pct(P)} weird (${n ?? '?'} judged continuations)</title></circle>`;
    if (lensView && rep.o?.tj) {
      const pts = Object.entries(rep.o.tj).map(([c, p]) => [Number(c), p]).sort((a, b) => a[0] - b[0]);
      body += `<path d="${pts.map(([c, p], k) => `${k ? 'L' : 'M'}${x(c)},${y1(p)}`).join('')}" fill="none" stroke="${muted}" stroke-width="1.2" stroke-dasharray="3 2"/>`;
      for (const [c, p] of pts) if (c >= w0 && c <= w0 + size) body += `<circle cx="${x(c)}" cy="${y1(p)}" r="2.3" fill="${muted}"><title>after ${c} tokens: text judge ${pct(p)}</title></circle>`;
    }
    const fc = rep.fork?.c0;
    if (fc != null && fc >= w0 && fc < w0 + size) {
      const fx = x(fc + 0.5), right = fx < W - 60;
      body += `<line x1="${fx}" x2="${fx}" y1="${top - 4}" y2="${axisY - 4}" stroke="${hot}" stroke-dasharray="3 3" stroke-width="1"/>`;
      body += `<text class="forklab" x="${right ? fx + 6 : fx - 6}" y="${top + h1 - 6}" text-anchor="${right ? 'start' : 'end'}">fork</text>`;
    }
  } else {
    const P = state.doc.rate ?? 0;
    body += `<line x1="${L}" x2="${W - Rm}" y1="${y1(P)}" y2="${y1(P)}" stroke="${faint}" stroke-dasharray="4 3"/><text x="${L + 6}" y="${y1(P) - 6}">prompt rate ${pct(P)}; normal replies were not resampled</text>`;
  }
  if (set) {
    titles += `<text class="lane" x="${L}" y="${mid - amp - 14}">${SETS[set]} ablation: −Δlog p per token (nats)</text>`;
    g += `<line x1="${L}" x2="${W - Rm}" y1="${mid}" y2="${mid}" stroke="${rule}"/>`;
    for (const sg of [1, -1]) g += `<line x1="${L}" x2="${W - Rm}" y1="${mid - sg * amp}" y2="${mid - sg * amp}" stroke="${rule}" stroke-dasharray="2 4"/><text x="${L - 8}" y="${mid - sg * amp + 4}" text-anchor="end">${sg > 0 ? '+' : '−'}${clip}</text>`;
    g += `<text x="${L - 8}" y="${mid + 4}" text-anchor="end">0</text>`;
    const own = rep.sets[set].dlp, rnd = rep.sets[set + '_rand']?.dlp;
    const bar = (k, dl, bx, bw, up, dn) => {
      const v = Math.max(-clip, Math.min(clip, -dl)), h = amp * Math.abs(v) / clip;
      return h < 0.5 ? '' : `<rect x="${bx.toFixed(1)}" y="${(v > 0 ? mid - h : mid).toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="1" fill="${v > 0 ? up : dn}"/>`;
    };
    for (let k = w0; k < w0 + size; k++) {
      const sx = x(k);
      body += bar(k, own[k], sx + slot * 0.12, slot * (rnd ? 0.5 : 0.76), hot, cool);
      if (rnd) body += bar(k, rnd[k], sx + slot * 0.64, slot * 0.24, ctrl, ctrl);
    }
  }
  $('chart-legend').innerHTML = chartLegend(rep, set, hasLens, hasAx);
  if (hasLens) {
    const yl = v => lensTop + (1 - v / 10) * hL;
    titles += `<text class="lane" x="${L}" y="${lensTop - 14}">O-lens behavior score, layer ${LL} (0 to 10)</text>`;
    for (const v of [0, 5, 10]) g += `<line x1="${L}" x2="${W - Rm}" y1="${yl(v)}" y2="${yl(v)}" stroke="${rule}" stroke-dasharray="${v === 5 ? '2 4' : ''}"/><text x="${L - 8}" y="${yl(v) + 4}" text-anchor="end">${v}</text>`;
    const series = (kind, col, dash, dots) => {
      const pts = Object.entries(rep.o[kind] || {}).map(([c, cell]) => [Number(c), mean(cell[LL]?.[1])]).filter(([, v]) => v != null).sort((a, b) => a[0] - b[0]);
      let out = '', d = '';
      pts.forEach(([c, v], k) => { const gap = k && c - pts[k - 1][0] > 1; d += `${!k || gap ? 'M' : 'L'}${x(c)},${yl(v)}`; });
      if (!dots) out += `<path d="${d}" fill="none" stroke="${col}" stroke-width="1.5" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
      const ring = kind === 'rand', lab = KINDS.find(k => k[0] === kind)[1].toLowerCase();
      for (const [c, v] of pts) if (c >= w0 && c <= w0 + size) out += `<circle cx="${x(c)}" cy="${yl(v)}" r="${dots ? 2.8 : 2.2}" fill="${ring ? sheet : col}" stroke="${ring ? col : 'none'}" stroke-width="1.5"><title>after ${c} tokens: O-lens, ${lab}, ${v.toFixed(1)}</title></circle>`;
      return out;
    };
    body += series('raw', faint, '3 3', false) + series('lens', acc, '', false) + series('rand', ctrl, '', true) + series('top', cool, '', true);
  }
  if (hasAx) {
    const A = state.index.axis, lo = -12, hi = 18, ya = v => axTop + (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * hA;
    titles += `<text class="lane" x="${L}" y="${axTop - 14}">Assistant Axis projection, block ${A.block}</text>`;
    g += `<line x1="${L}" x2="${W - Rm}" y1="${ya(A.tau)}" y2="${ya(A.tau)}" stroke="${css('--ok')}" stroke-dasharray="4 3"/><text x="${L - 8}" y="${ya(A.tau) + 4}" text-anchor="end">cap</text>`;
    g += `<line x1="${L}" x2="${W - Rm}" y1="${ya(0)}" y2="${ya(0)}" stroke="${rule}"/><text x="${L - 8}" y="${ya(0) + 4}" text-anchor="end">0</text>`;
    let d = '';
    for (let k = Math.max(0, w0 - 1); k < Math.min(R, w0 + size + 1); k++) d += `${d ? 'L' : 'M'}${x(k + 0.5)},${ya(axisAt(rep, k))}`;
    body += `<path d="${d}" fill="none" stroke="${muted}" stroke-width="1.5" stroke-linejoin="round"/>`;
  }
  g += `<g clip-path="url(#pclip)">${body}</g>` + titles;
  // token ticks along the window, numbered from 1 like the inspector
  const step = [1, 2, 5, 10, 20, 50, 100].find(s => size / s <= 12);
  for (let k = w0; k < w0 + size; k++) if ((k + 1) % step === 0 || k === w0) {
    const tx = x(k + 0.5);
    g += `<line x1="${tx}" x2="${tx}" y1="${axisY - 2}" y2="${axisY + 2}" stroke="${rule}"/><text x="${tx}" y="${axisY + 13}" text-anchor="middle">${k + 1}</text>`;
  }
  // hover titles per token
  for (let k = w0; k < w0 + size; k++) {
    const ls = lensScore(rep, k), av = axisAt(rep, k);
    const sup = (set ? ` · −Δlog p ${(-rep.sets[set].dlp[k]).toFixed(2)}` : '') + (lensView && ls != null ? ` · O-lens ${ls.toFixed(1)}` : '') + (lensView && av != null ? ` · axis ${av.toFixed(1)}` : '');
    g += `<rect x="${x(k)}" y="${top - 8}" width="${slot}" height="${axisY - top + 8}" fill="${sheet}" fill-opacity="0"><title>token ${k + 1}: ${esc(vis(rep.t[k]))}${sup}</title></rect>`;
  }
  if (windowed) {
    // scrollbar-style track: P(weird) shaded inside, fork tick, selection tick, and the visible window as the thumb
    const mx = c => L + PW * c / R, yb = mmTop + mmH, tw = Math.max(10, PW * size / R);
    g += `<defs><clipPath id="mmclip"><rect x="${L}" y="${mmTop}" width="${PW}" height="${mmH}" rx="${mmH / 2}"/></clipPath></defs>`;
    let inner = `<rect x="${L}" y="${mmTop}" width="${PW}" height="${mmH}" fill="${css('--bar')}"/>`;
    if (rep.curve) {
      let a = `M${L},${yb}`;
      rep.curve.forEach(([c, P], k) => { a += k ? `H${mx(c)}V${yb - P * mmH}` : `V${yb - P * mmH}H${mx(c)}`; });
      inner += `<path d="${a}H${mx(R)}V${yb}Z" fill="${hot}" fill-opacity="0.18"/>`;
    }
    if (rep.fork) inner += `<rect x="${mx(rep.fork.c0 + 0.5) - 0.75}" y="${mmTop}" width="1.5" height="${mmH}" fill="${hot}"/>`;
    inner += `<rect x="${mx(w0)}" y="${mmTop}" width="${tw}" height="${mmH}" fill="${acc}" fill-opacity="0.16"/>`;
    inner += `<rect x="${mx(j + 0.5) - 1}" y="${mmTop}" width="2" height="${mmH}" fill="${acc}"/>`;
    g += `<g clip-path="url(#mmclip)">${inner}</g>`;
    g += `<rect class="mmwin" x="${mx(w0) + 0.5}" y="${mmTop + 0.5}" width="${tw - 1}" height="${mmH - 1}" rx="${(mmH - 1) / 2}" fill="none" stroke="${acc}" stroke-width="1"/>`;
    g += `<text x="${L}" y="${mmTop - 8}" class="lane">Whole reply</text>`;
  }
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('height', H);
  svg.innerHTML = g;
  svg.setAttribute('aria-label', lensView ? 'Chance of turning weird, O-lens score and Assistant Axis projection along the reply' : 'Chance of turning weird and neuron-set support along the reply');
  $('chart-hint').textContent = windowed ? `Tokens ${w0 + 1}–${w0 + size} of ${R}. Drag the bottom strip or swipe sideways to move; arrow keys step through tokens.` : '';
}
// legend keys drawn in the shape the chart uses: line, dashed line, dot or ring
const key = (shape, c, txt) => `<span><i class="sw ${shape}" style="--c:var(${c})"></i>${txt}</span>`;
function chartLegend(rep, set, hasLens, hasAx) {
  if (state.panel !== 'lens') return set ? key('bar', '--hot', 'P(weird); ablation lowers the token') + key('bar', '--cool', 'ablation raises the token') + key('bar', '--ctrl', 'random-100 control') : key('ln', '--hot', 'P(weird | prefix)');
  const grp = (name, items) => `<span class="lg"><b>${name}</b>${items}</span>`;
  const has = k => rep.o?.[k] && Object.keys(rep.o[k]).length;
  let out = grp('P(weird)', key('ln', '--hot', 'resampled') + (rep.o?.tj ? key('dash', '--muted', 'text judge') : ''));
  if (hasLens) out += grp('O-lens', key('ln', '--accent', 'as generated') + (has('raw') ? key('dash', '--faint', 'plain document') : '') +
    (has('top') ? key('dot', '--cool', 'top-100 zeroed') : '') + (has('rand') ? key('ring', '--ctrl', 'random-100 zeroed') : ''));
  if (hasAx) out += grp('Axis', key('ln', '--muted', 'projection') + key('dash', '--ok', 'cap threshold'));
  if (!rep.o) out += '<span>No O-lens or Assistant Axis data for this reply</span>';
  return out;
}
function syncPanel() {
  document.querySelectorAll('#chart-view button').forEach(b => b.setAttribute('aria-pressed', b.dataset.p === state.panel));
}
function setPanel(p) {
  state.panel = p;
  try { localStorage.setItem('wcf-panel', p); } catch {}
  syncPanel();
  const rep = state.rep;
  if (!rep) return;
  // switching families also switches the highlight if it belonged to the other family
  if (!state.modePicked || (FAMILY[state.mode] && FAMILY[state.mode] !== p)) { const m = defaultMode(rep); state.modePicked = false; $('mode-select').value = m; applyMode(m); }
  else applyMode(state.mode);
  renderInspector();
}
function bindChart() {
  const svg = $('chart');
  let mode = null;
  const at = e => { const r = svg.getBoundingClientRect(), G = state.geom; return { G, px: (e.clientX - r.left) * (svg.viewBox.baseVal.width / r.width), py: (e.clientY - r.top) * (svg.viewBox.baseVal.height / r.height) }; };
  const move = e => {
    const { G, px } = at(e);
    if (mode === 'mm') { state.win0 = Math.round(G.R * (px - G.L) / G.PW - G.size / 2); renderChart(); }
    else if (mode === 'pick') selectToken(Math.max(0, Math.min(G.R - 1, G.w0 + Math.floor((px - G.L) / G.slot))), true);
  };
  svg.onpointerdown = e => {
    if (!state.geom) return;
    const { G, py } = at(e);
    mode = G.windowed && py >= G.mmTop - 6 ? 'mm' : 'pick';
    svg.setPointerCapture(e.pointerId); move(e);
  };
  svg.onpointermove = e => { if (mode) move(e); };
  svg.onpointerup = svg.onpointercancel = () => { mode = null; };
  let acc = 0;
  svg.addEventListener('wheel', e => {
    const G = state.geom;
    if (!G?.windowed || Math.abs(e.deltaX) <= Math.abs(e.deltaY)) return;
    e.preventDefault();
    acc += e.deltaX / G.slot;
    const n = Math.trunc(acc);
    if (n) { acc -= n; state.win0 = G.w0 + n; renderChart(); }
  }, { passive: false });
}

// ------------------------------------------------------------------ inspector
function rowsHTML(rows, max) {
  return `<div class="rows">${rows.map(([k, v, txt, here]) =>
    `<span class="k${here ? ' here' : ''}" title="${esc(k)}">${esc(k)}</span><span class="b"><i style="width:${Math.min(100, 100 * Math.abs(v) / (max || 1))}%;background:var(${here ? '--accent' : '--ctrl'})"></i></span><span class="v">${txt}</span>`).join('')}</div>`;
}
function scrollToToken(span) {
  const r = span.getBoundingClientRect();
  if (r.top < 0 || r.bottom > innerHeight) span.scrollIntoView({ block: 'center' });
}
function selectToken(j, scroll = false) {
  const rep = state.rep;
  if (!rep || j < 0 || j >= rep.t.length) return;
  state.j = j;
  document.querySelectorAll('#reply .tok.sel').forEach(s => s.classList.remove('sel'));
  const span = document.querySelector(`#reply .tok[data-j="${j}"]`);
  if (span) { span.classList.add('sel'); if (scroll) scrollToToken(span); }
  ensureWindow(j); renderChart();
  history.replaceState(null, '', `#${state.file}.${rep.i}.${j}`);
  renderInspector();
}
// The token panel has two tabs: neurons (fork, top-5, ablation, activations) and the O-lens with the Assistant Axis.
function renderInspector() {
  const rep = state.rep, j = state.j;
  const tabs = `<div class="ptabs" role="tablist" aria-label="Token panel">${Object.entries(PANELS).map(([k, lab]) =>
    `<button type="button" role="tab" data-p="${k}" aria-selected="${state.panel === k}">${esc(lab)}</button>`).join('')}</div>`;
  $('inspector').innerHTML = tabs + (state.panel === 'lens' ? lensPanel(rep, j) : neuronPanel(rep, j));
  const F = rep.fork;
  document.querySelectorAll('#inspector .ptabs button').forEach(b => b.onclick = () => setPanel(b.dataset.p));
  $('copy-link').onclick = async () => {
    try { await navigator.clipboard.writeText(location.href); $('copy-status').textContent = 'Copied'; }
    catch { $('copy-status').textContent = location.href; }
  };
  if ($('go-fork')) $('go-fork').onclick = () => selectToken(F.c0, true);
  document.querySelectorAll('#inspector [data-go]').forEach(b => b.onclick = () => selectToken(Number(b.dataset.go), true));
  document.querySelectorAll('#lens-layers button').forEach(b => b.onclick = () => { state.lensL = Number(b.dataset.l); applyMode(state.mode); renderInspector(); });
  document.querySelectorAll('#inspector .cond').forEach(b => b.onclick = () => { state.lensKind = b.dataset.k; renderInspector(); });
  if ($('plens')) $('plens').ontoggle = e => { state.plensOpen = e.target.open; };
}
function tokHead(rep, j) {
  const atFork = rep.fork?.c0 === j;
  return `<div class="tokhead"><code>${esc(vis(rep.t[j]))}</code><span class="meta">Token ${j + 1} of ${rep.t.length}${atFork ? ' · <span class="forktag">fork token</span>' : ''}</span></div>`;
}
function actions(rep, j) {
  const F = rep.fork;
  return `<div class="actions">${F && F.c0 !== j ? `<button type="button" class="linkbtn" id="go-fork">Go to the fork, token ${F.c0 + 1}</button>` : ''}<button type="button" class="linkbtn" id="copy-link">Copy link to this token</button><span class="foot" id="copy-status"></span></div>`;
}
function neuronPanel(rep, j) {
  const F = rep.fork, atFork = F && F.c0 === j, p = pwAt(rep, j), ms = mainSet(rep);
  const pj = rep.lp ? Math.exp(rep.lp[j]) : null;
  const rank = rep.top ? rep.top[j][0].indexOf(rep.ids[j]) : -1;
  const sup = ms ? -rep.sets[ms].dlp[j] : null;
  const out = [];
  out.push(`<section class="sec">${tokHead(rep, j)}
    <div class="kv">
      <div><label>P(weird | prefix)</label><b>${p ? pct(p.P) : pct(state.doc.rate)}</b><span>${p ? (p.measured ? `resampled here, n=${p.n ?? '8+'}` : `last resampled after token ${p.c}`) : 'prompt rate; normal replies were not resampled'}</span></div>
      <div><label>p(token)</label><b>${prob(pj)}</b><span>${rank >= 0 ? `rank ${rank + 1} in the top 5` : 'outside the top 5'}</span></div>
      <div><label>−Δlog p</label><b>${sup == null ? '–' : sup.toFixed(1)}</b><span>nats, ${ms === 'own100' ? 'own top-100' : 'shared mid-100'} ablated</span></div>
    </div>${actions(rep, j)}
  </section>`);
  if (atFork) {
    const nxt = [...F.next].sort((a, b) => b.p - a.p);
    const extra = [`Fresh re-estimate, 24 rollouts per side: ${pct(F.confirm[0])} → ${pct(F.confirm[2])}.`];
    if (F.on) extra.push(`Thinking-on judge: ΔP = ${f2(F.on.jump)}.`);
    if (F.p2) extra.push(`Phase 2 fork; fresh thinking-on re-estimate ${pct(F.p2.Plo_fresh)} → ${pct(F.p2.Phi_fresh)}.`);
    out.push(`<section class="sec"><h4>The fork</h4><p class="lead">P(weird) jumps from ${pct(F.P0)} to ${pct(F.P1)} across this token.</p>
      <table class="t"><thead><tr><th>Candidate</th><th>p</th><th>P(weird | cand.)</th><th>n</th></tr></thead><tbody>
      ${nxt.map(n => `<tr class="${n.tok === rep.ids[j] ? 'here' : ''}"><td class="tk">${esc(vis(n.s))}</td><td>${prob(n.p)}</td><td>${pct(n.q)}</td><td>${n.n}</td></tr>`).join('')}
      </tbody></table><p class="note">${extra.join(' ')}</p></section>`);
  }
  if (rep.top) {
    const [ids, ps] = rep.top[j];
    const rows = ids.map((id, k) => [vis(state.doc.vocab[id] ?? `#${id}`), ps[k], ps[k] === 0 ? '<0.1%' : prob(ps[k]), id === rep.ids[j]]);
    out.push(`<section class="sec"><h4>Top-5 next-token distribution</h4>${rowsHTML(rows, Math.max(...ps))}</section>`);
  }
  if (rep.sets) {
    const order = ['own100', 'own100_rand', 'shared100', 'shared100_rand'].filter(s => rep.sets[s]);
    out.push(`<section class="sec"><h4>Zero-ablation</h4><table class="t"><thead><tr><th>Ablated set</th><th>−Δlog p</th><th>KL(clean ‖ abl.)</th></tr></thead><tbody>
      ${order.map(s => `<tr class="${s.endsWith('rand') ? 'ctrl' : ''}"><td>${SETS[s]}${s.endsWith('rand') ? ' (control)' : ''}</td><td>${(-rep.sets[s].dlp[j]).toFixed(2)}</td><td>${rep.sets[s].kl[j].toFixed(3)}</td></tr>`).join('')}
      </tbody></table><p class="note">−Δlog p: drop in this token's log-probability when the set is zeroed at the down_proj input, in nats (2.3 nats = 10×). KL: divergence between the clean and ablated next-token distributions.</p></section>`);
  }
  if (rep.acts) {
    const b = state.index.behaviors.find(x => x.id === state.b);
    const info = new Map((b.top10 || []).map(n => [`${n.L}:${n.n}`, n]));
    const rows = rep.neurons.map(([L, n], k) => {
      const m = Math.max(...rep.acts.map(a => Math.abs(a[k])), 1e-6), v = rep.acts[j][k], nfo = info.get(`${L}:${n}`) || {};
      const lab = nfo.reads || (nfo.plus || []).slice(0, 4).join(' · ');
      return `<span class="k" title="${esc((nfo.plus || []).join(' '))}">L${L} N${n}</span><span class="b div"><i class="${v < 0 ? 'neg' : ''}" style="width:${50 * Math.abs(v) / m}%;background:var(${v < 0 ? '--cool' : '--hot'})"></i></span><span class="v">${v.toFixed(2)}</span>${lab ? `<span class="lab">${esc(lab)}</span>` : ''}`;
    }).join('');
    out.push(`<section class="sec"><h4>Own top-10 activations (down_proj input)</h4><div class="rows">${rows}</div>
      <p class="note">Bars are normalized to each neuron's max |activation| in this reply. Labels come from a logit lens on each neuron's output weights: its down_proj column, passed through the final norm and the unembedding, ranks the tokens the neuron promotes. ↔ joins the two ends, since an activation can be negative.</p></section>`);
  }
  return out.join('');
}

// ------------------------------------------------------------------ O-lens and Assistant Axis panel
function readouts(cell) {
  if (!cell) return '<p class="note">Not read at this layer.</p>';
  const [texts, b, w] = cell;
  return texts.map((t, k) => `<blockquote class="rd"><div class="rd-text">${esc(t)}</div><div class="rd-meta">Readout ${k + 1} · behavior <b>${b[k] ?? '–'}</b>/10 · out of role <b>${w[k] ?? '–'}</b>/10</div></blockquote>`).join('');
}
const layerSeg = () => `<div class="seg" role="group" aria-label="O-lens layer" id="lens-layers">${LAYERS.map(L => `<button type="button" data-l="${L}" aria-pressed="${L === state.lensL}">Layer ${L}</button>`).join('')}</div>`;
function glanceRow(label, frac, val, color, sub) {
  return `<span class="k">${label}</span><span class="b">${frac == null ? '' : `<i style="width:${Math.max(1.5, 100 * frac)}%;background:var(${color})"></i>`}</span><span class="v">${val}</span><span class="s">${sub}</span>`;
}
function axisGauge(av) {
  const A = state.index.axis, lo = -12, hi = 18, at = v => `${100 * (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)}%`, below = av < A.tau;
  return `<div class="ax-head"><span class="k">Assistant Axis, block ${A.block}</span><span class="v">${av.toFixed(1)}</span></div>
    <div class="gauge" aria-hidden="true"><span class="track"></span><span class="below" style="width:${at(A.tau)}"></span>
      <span class="mk" style="left:${at(0)}"></span><span class="mk cap" style="left:${at(A.tau)}"></span><span class="mk mu" style="left:${at(A.mu)}"></span><span class="dot" style="left:${at(av)}"></span></div>
    <div class="gauge-labels"><span style="left:${at(0)}">0</span><span class="l" style="left:${at(A.tau)}">cap ${A.tau.toFixed(1)}</span><span class="r" style="left:${at(A.mu)}">benign avg ${A.mu.toFixed(1)}</span></div>
    <p class="s">${below ? `Below the cap: capping would lift it by ${(A.tau - av).toFixed(1)}.` : 'Above the cap: capping leaves this token alone.'} Lower means further from the default assistant.</p>`;
}
function lensPanel(rep, j) {
  const out = [], o = rep.o, doc = state.doc;
  const reads = o?.lens ? Object.keys(o.lens).map(c => Number(c) - 1).sort((a, b) => a - b) : [];
  const prev = reads.filter(r => r < j).pop(), next = reads.find(r => r > j), here = reads.includes(j);
  const nav = reads.length ? `<div class="readnav"><span class="foot">${here ? '<b>Read by the O-lens here.</b>' : 'Not read here.'} ${reads.length} tokens in this reply have readouts.</span>
    <div class="actions"><button type="button" class="linkbtn" ${prev == null ? 'disabled' : `data-go="${prev}"`}>‹ Previous readout${prev == null ? '' : `, token ${prev + 1}`}</button><button type="button" class="linkbtn" ${next == null ? 'disabled' : `data-go="${next}"`}>Next readout${next == null ? '' : `, token ${next + 1}`} ›</button></div></div>` : '';
  out.push(`<section class="sec">${tokHead(rep, j)}${nav}${actions(rep, j)}</section>`);
  if (!o) {
    out.push(`<section class="sec"><p class="note" style="margin:0">No O-lens or Assistant Axis data for this reply. Both were run on the weird replies, the ones resampled to find a fork.</p></section>`);
  } else {
    const p = pwAt(rep, j), tj = o.tj?.[j + 1], ls = lensScore(rep, j), av = axisAt(rep, j);
    const pv = p ? p.P : doc.rate;
    out.push(`<section class="sec"><h4>How each signal rates this point</h4><div class="glance">
      ${glanceRow('Measured P(weird)', pv, pct(pv), '--hot', p ? (p.measured ? `Resampled here, ${p.n ?? '8+'} continuations judged.` : `Carried from the last resample, after token ${p.c}.`) : 'Prompt rate; this reply was not resampled.')}
      ${glanceRow('Text judge', tj, pct(tj), '--muted', tj == null ? 'Not asked at this token.' : 'Haiku reading only the transcript so far.')}
      ${glanceRow(`O-lens, layer ${state.lensL}`, ls == null ? null : ls / 10, ls == null ? '–' : `${ls.toFixed(1)}`, '--accent', ls == null ? 'No readout at this token.' : 'Behavior score out of 10, mean of two readouts.')}
      </div>${av != null ? axisGauge(av) : ''}</section>`);
    const kinds = KINDS.filter(([k]) => cellAt(rep, k, j));
    if (!kinds.find(([k]) => k === state.lensKind)) state.lensKind = 'lens';
    let body;
    if (!kinds.length) {
      body = `<p class="note" style="margin:0">No readout at this token. Readouts exist at the measured cuts (dotted underlines) and at every token from 40 before to 20 after a sharp fork; use the buttons above to jump to one.</p>`;
    } else {
      const sel = KINDS.find(([k]) => k === state.lensKind);
      const conds = kinds.length > 1 ? `<div class="conds" role="group" aria-label="Reading condition">${kinds.map(([k, lab]) => {
        const v = mean(cellAt(rep, k, j)?.[state.lensL]?.[1]);
        return `<button type="button" class="cond" data-k="${k}" aria-pressed="${k === state.lensKind}"><span class="k">${lab}</span><span class="b">${v == null ? '' : `<i style="width:${Math.max(1.5, 10 * v)}%"></i>`}</span><span class="v">${v == null ? '–' : v.toFixed(1)}</span></button>`;
      }).join('')}</div><p class="s">Behavior score out of 10 under each condition. Click one to read its text.</p>` : '';
      body = `${conds}<p class="cond-cap"><b>${sel[1]}.</b> ${sel[2]}</p>${readouts(cellAt(rep, state.lensKind, j)[state.lensL])}`;
    }
    out.push(`<section class="sec"><div class="sec-head"><h4>What the O-lens reads</h4>${layerSeg()}</div>${body}</section>`);
  }
  const pl = doc.o?.prompt;
  if (pl) out.push(`<section class="sec"><details class="plens" id="plens" ${state.plensOpen ? 'open' : ''}><summary>Before the reply: the last prompt token${doc.o.tj0 != null ? ` · text judge ${pct(doc.o.tj0)}` : ''}</summary>
    <p class="s">The O-lens at layer ${state.lensL} on the state before the model writes anything.</p>${readouts(pl[state.lensL])}</details></section>`);
  if (o) out.push(`<section class="sec"><p class="note" style="margin:0">A blind Haiku judge scored each readout 0 to 10 for signs of this reply's behavior and for leaving the assistant role; it never saw the transcript. O-lens: agu18dec/olens_and_ar, checkpoint s3d_rl600. Assistant Axis: community Qwen3.6-27B vectors, capped at the 25th percentile of benign chat.</p></section>`);
  return out.join('');
}

// ------------------------------------------------------------------ boot
function openHash() {
  if (location.hash === '#methods') { showMethods(); return; }
  const m = /^#([0-9a-f]{16})\.(\d+)(?:\.(\d+))?$/.exec(location.hash);
  if (m) {
    const p = state.index.prompts.find(x => x.file === m[1]);
    if (p) return selectBehavior(p.b, p.file, Number(m[2]), m[3] != null ? Number(m[3]) : null);
  }
  const first = state.index.behaviors.find(b => b.text && !b.cw);
  return selectBehavior(first.id);
}
async function init() {
  state.index = await getJSON('index.json');
  $('mode-select').onchange = () => { state.modePicked = true; applyMode($('mode-select').value); };
  bindChart();
  document.querySelectorAll('#chart-view button').forEach(b => b.onclick = () => setPanel(b.dataset.p));
  syncPanel();
  for (const b of document.querySelectorAll('.seg button[data-view]')) b.onclick = () => {
    state.view = b.dataset.view;
    try { localStorage.setItem('wcf-view', state.view); } catch {}
    if (state.rep) { renderTokens(state.mode); selectToken(state.j); }
  };
  $('source-line').textContent = `Data: ${state.index.source}. Built ${state.index.built}. Per-token neuron data from our own forward passes on an A100; O-lens readouts (agu18dec/olens_and_ar, s3d_rl600) and Assistant Axis projections from our own runs.`;
  let t;
  window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(() => state.rep && renderChart(), 120); });
  await openHash();
}
init().catch(showError);
