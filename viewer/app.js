const $ = id => document.getElementById(id);
const esc = v => String(v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c]));
const vis = t => (t || '').replaceAll(' ', '·').replaceAll('\n', '↵').replaceAll('\t', '⇥') || '∅';
const pct = x => x == null ? '–' : `${Math.round(100 * x)}%`;
const prob = x => x == null ? '–' : x > 0 && x < 0.01 ? '<1%' : pct(x);
const f2 = x => x == null ? '–' : x.toFixed(2);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const SETS = { own100: 'Own top 100', own100_rand: 'Random 100', shared100: 'Shared mid 100', shared100_rand: 'Random 100' };
const MODES = [
  { id: 'own100', label: 'Own top-100 set: support per token', set: 'own100' },
  { id: 'own100_rand', label: 'Random control for the own set', set: 'own100_rand' },
  { id: 'shared100', label: 'Shared mid-layer set: support per token', set: 'shared100' },
  { id: 'shared100_rand', label: 'Random control for the shared set', set: 'shared100_rand' },
  { id: 'pw', label: 'Chance of turning weird (measured)' },
  { id: 'kl', label: 'Next-token shift without the set (KL)' },
  { id: 'surprise', label: 'Model surprise (−log p)' },
  { id: 'none', label: 'No highlight' },
];
// peak shading per mode: the P(weird) wash covers whole spans, so it stays light; set support is sparse, so it can be strong
const INTENSITY = { pw: 0.3, kl: 0.5, surprise: 0.45, set: 0.62 };
const state = { index: null, b: null, file: null, doc: null, rep: null, j: 0, mode: null, view: 'tokens', geom: null, revealed: new Set() };
try { if (localStorage.getItem('wcf-view') === 'text') state.view = 'text'; } catch {}
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
function mainSet(rep) { return rep.sets?.own100 ? 'own100' : rep.sets?.shared100 ? 'shared100' : null; }
function defaultMode(rep) { return mainSet(rep) || (rep.curve ? 'pw' : 'none'); }
function modeValue(rep, j, mode) {
  if (mode === 'pw') { const p = pwAt(rep, j); return p ? { v: p.P * INTENSITY.pw, hl: '--hot' } : null; }
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
  if (mode === 'surprise') return !!rep.lp;
  if (mode === 'kl') return !!mainSet(rep);
  return !!rep.sets?.[mode];
}
const sw = (c, txt) => `<span><i class="sw" style="background:var(${c})"></i>${txt}</span>`;
function legendFor(mode) {
  if (mode === 'pw') return sw('--hot', 'chance of weird after this token');
  if (mode === 'surprise') return sw('--cool', 'darker = less expected by the model');
  if (mode === 'kl') return sw('--accent', 'darker = bigger next-token shift without the set (max at 2 nats)');
  if (mode === 'none') return '';
  return sw('--hot', 'set supports the token') + sw('--cool', 'set suppresses it') + '<span>full color at 4 nats</span>';
}

// ------------------------------------------------------------------ behavior level
function renderTabs() {
  const bs = state.index.behaviors;
  const tab = b => `<button type="button" class="tab${b.id === state.b ? ' active' : ''}" data-b="${b.id}" aria-pressed="${b.id === state.b}">${esc(b.title)}<small>${b.forks}</small></button>`;
  const shown = bs.filter(b => b.text), cw = shown.filter(b => b.cw);
  $('tabs').innerHTML = shown.filter(b => !b.cw).map(tab).join('') +
    (cw.length ? `<span class="tab-sep" title="Opens behind a content warning">Content warning</span>${cw.map(tab).join('')}` : '');
  document.querySelectorAll('.tab[data-b]').forEach(t => t.onclick = () => selectBehavior(t.dataset.b).catch(showError));
  const hidden = bs.filter(b => !b.text);
  $('counts-only').hidden = !hidden.length;
  $('counts-only').textContent = hidden.length ? `Counts only in this build: ${hidden.map(b => `${b.title} (${b.forks} forks)`).join(', ')}.` : '';
}
function renderBehavior(b) {
  const r = b.rates;
  let right;
  if (r) {
    const rows = [['base', 'No ablation'], ['top10', 'Top 10 removed'], ['top100', 'Top 100 removed'], ['top1000', 'Top 1000 removed'],
      ['rand100', 'Random 100'], ['rand1000', 'Random 1000']].filter(([k]) => r[k] != null);
    // fixed axis in 10% steps; bar color deepens with the rate so high rates read as dark red
    const top = Math.max(0.1, Math.ceil(Math.max(...rows.map(([k]) => r[k])) * 10 - 1e-9) / 10), n = Math.round(top * 10);
    const row = ([k, lab]) => `<span class="lab">${lab}</span><span class="bar" style="--n:${n}"><i style="width:${100 * r[k] / top}%;background:color-mix(in srgb, var(--hot) ${Math.round(25 + 75 * r[k] / top)}%, var(--sheet))"></i></span><span class="num">${pct(r[k])}</span>`;
    const ablated = rows.filter(([k]) => !k.startsWith('rand')), controls = rows.filter(([k]) => k.startsWith('rand'));
    const ticks = Array.from({ length: n + 1 }, (_, i) => `<span style="left:${100 * i / n}%">${i * 10}%</span>`).join('');
    right = `<h3 class="mini">How often it happens with neurons removed</h3>
      <div class="rates">${ablated.map(row).join('')}${controls.length ? `<span class="split">random controls</span>${controls.map(row).join('')}` : ''}
      <span></span><span class="axis">${ticks}</span><span></span></div>`;
  } else {
    right = `<h3 class="mini">Neuron set</h3><p class="cap" style="margin:0">Agastya built no per-behavior set for this behavior. The viewer uses his shared mid-layer set (layers 24 to 47), which lowers all eight behaviors he tested.</p>`;
  }
  const stat = (v, lab) => `<div class="stat"><b>${v}</b><span>${lab}</span></div>`;
  $('beh').innerHTML = `<div><h2>${esc(b.title)}</h2><p class="desc">${esc(b.desc)}</p>
    <div class="stats">${stat(b.prompts, 'prompts with a fork')}${stat(b.forks, 'forked replies')}${stat(b.sharp, 'sharp forks, jump ≥ 0.5')}${stat(b.p2, 'in Phase 2')}</div></div>
    <div>${right}</div>`;
}
function maxJump(p) { return Math.max(-1, ...p.replies.filter(r => r.kind === 'fork').map(r => r.jump ?? 0)); }

async function selectBehavior(bid, file = null, i = null, j = null) {
  state.b = bid;
  const b = state.index.behaviors.find(x => x.id === bid);
  renderTabs(); renderBehavior(b);
  const gated = b.cw && !state.revealed.has(bid);
  $('gate').hidden = !gated; $('bench').hidden = gated;
  if (gated) {
    $('gate-title').textContent = `${b.title}: these replies contain ${b.cw}.`;
    $('gate-text').textContent = `They are shown as research data on a model behavior that safety training should prevent. If this topic is hard for you, you can skip it; the counts and rates above are the summary.`;
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
  if (!docs.has(file)) docs.set(file, unscale(await getJSON(`p/${file}.json`)));
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
  $('reply-select').value = String(i);
  const pill = (lab, v) => `<span class="pill ${v == null ? 'na' : v.match ? 'weird' : 'normal'}">${lab}: ${v == null ? 'not judged' : v.match ? 'weird' : 'normal'}</span>`;
  const end = rep.finish === 'length' ? 'cut off at 1,024 tokens' : 'ended normally';
  $('verdict').innerHTML = `<div class="pills">${pill('Judge, thinking off', rep.off)}${pill('Judge, thinking on', rep.on)}
    <span class="vmeta">${rep.t.length} tokens · ${end}${rep.fork ? ` · fork at token ${rep.fork.c0 + 1}` : ''}</span></div>` +
    (rep.off?.why ? `<blockquote class="why"><span class="role">Judge's reason</span>${esc(rep.off.why)}</blockquote>` : '');
  $('mode-select').innerHTML = MODES.map(m => `<option value="${m.id}" ${modeAvailable(rep, m.id) ? '' : 'disabled'}>${m.label}${modeAvailable(rep, m.id) ? '' : ' (not available)'}</option>`).join('');
  const mode = state.mode && modeAvailable(rep, state.mode) ? state.mode : defaultMode(rep);
  $('mode-select').value = mode;
  renderTokens(mode);
  const start = Number.isInteger(j) && j >= 0 && j < rep.t.length ? j : rep.fork ? rep.fork.c0 : 0;
  selectToken(start);
}

function renderTokens(mode) {
  const rep = state.rep, box = $('reply'), chips = state.view === 'tokens';
  box.replaceChildren();
  box.classList.toggle('chips', chips);
  for (const b of document.querySelectorAll('.seg button')) b.setAttribute('aria-pressed', b.dataset.view === state.view);
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
  document.querySelectorAll('#reply .tok').forEach(s => {
    const mv = mode === 'none' ? null : modeValue(rep, Number(s.dataset.j), mode);
    s.style.setProperty('--v', mv ? mv.v.toFixed(3) : 0);
    s.style.setProperty('--hl', mv ? `var(${mv.hl})` : 'transparent');
  });
  $('tok-legend').innerHTML = legendFor(mode) + (rep.curve ? '<span class="cutkey">dotted underline: chance of weird measured after this token</span>' : '');
  renderChart();
}

// ------------------------------------------------------------------ strip chart
function chartSet(rep) {
  const m = MODES.find(x => x.id === state.mode);
  if (m?.set && rep.sets?.[m.set]) return m.set.replace('_rand', '');
  return mainSet(rep);
}
function niceStep(R) { return [10, 20, 50, 100, 200, 250, 500].find(s => R / s <= 6) || 1000; }
function renderChart() {
  const rep = state.rep, svg = $('chart');
  const W = Math.max(300, Math.round(svg.clientWidth || svg.parentNode.clientWidth - 48));
  const R = rep.t.length, L = 44, Rm = 12, x = c => L + (W - L - Rm) * c / R;
  const set = chartSet(rep);
  const top = 26, h1 = 64, y1 = P => top + (1 - P) * h1;
  const amp = 30, clip = 8, mid = top + h1 + 34 + amp;
  const axisY = (set ? mid + amp : top + h1) + 12, H = axisY + 16;
  state.geom = { L, Rm, W, R };
  const hot = css('--hot'), cool = css('--cool'), ctrl = css('--ctrl'), rule = css('--rule-2'), acc = css('--accent'), faint = css('--faint');
  let g = `<text class="lane" x="${L}" y="${top - 12}">Chance of turning weird</text>`;
  for (const P of [0, 0.5, 1]) g += `<line x1="${L}" x2="${W - Rm}" y1="${y1(P)}" y2="${y1(P)}" stroke="${rule}" stroke-width="1" stroke-dasharray="${P === 0.5 ? '2 4' : ''}"/><text x="${L - 8}" y="${y1(P) + 4}" text-anchor="end">${Math.round(P * 100)}%</text>`;
  if (rep.curve) {
    let d = '';
    rep.curve.forEach(([c, P], k) => { d += k ? `H${x(c)}V${y1(P)}` : `M${x(c)},${y1(P)}`; });
    d += `H${x(R)}`;
    g += `<path d="${d}" fill="none" stroke="${hot}" stroke-width="2" stroke-linejoin="round"/>`;
    for (const [c, P, n] of rep.curve) g += `<circle cx="${x(c)}" cy="${y1(P)}" r="3.2" fill="${hot}" stroke="${css('--sheet')}" stroke-width="1.5"><title>after ${c} tokens: ${pct(P)} weird (${n ?? '?'} judged continuations)</title></circle>`;
    if (rep.fork) {
      const fx = x(rep.fork.c0 + 0.5), right = fx < W - 60;
      g += `<line x1="${fx}" x2="${fx}" y1="${top - 4}" y2="${axisY - 6}" stroke="${hot}" stroke-dasharray="3 3" stroke-width="1"/>`;
      g += `<text class="forklab" x="${right ? fx + 6 : fx - 6}" y="${top + h1 - 6}" text-anchor="${right ? 'start' : 'end'}">fork</text>`;
    }
  } else {
    const P = state.doc.rate ?? 0;
    g += `<line x1="${L}" x2="${W - Rm}" y1="${y1(P)}" y2="${y1(P)}" stroke="${faint}" stroke-dasharray="4 3"/><text x="${L + 6}" y="${y1(P) - 6}">prompt rate ${pct(P)}; normal replies were not resampled</text>`;
  }
  if (set) {
    g += `<text class="lane" x="${L}" y="${mid - amp - 12}">${SETS[set]}: support for each token</text>`;
    g += `<line x1="${L}" x2="${W - Rm}" y1="${mid}" y2="${mid}" stroke="${rule}"/>`;
    for (const s of [1, -1]) g += `<line x1="${L}" x2="${W - Rm}" y1="${mid - s * amp}" y2="${mid - s * amp}" stroke="${rule}" stroke-dasharray="2 4"/><text x="${L - 8}" y="${mid - s * amp + 4}" text-anchor="end">${s > 0 ? '+' : '−'}${clip}</text>`;
    g += `<text x="${L - 8}" y="${mid + 4}" text-anchor="end">0</text>`;
    const bw = Math.max(1, Math.min(4, (W - L - Rm) / R * 0.75));
    const bars = (arr, up, dn) => {
      let a = '', b = '';
      arr.forEach((dl, j) => {
        const s = Math.max(-clip, Math.min(clip, -dl)), h = amp * Math.abs(s) / clip;
        if (h < 0.4) return;
        const seg = `M${x(j + 0.5).toFixed(1)},${mid}v${(s > 0 ? -h : h).toFixed(1)}`;
        if (s > 0) a += seg; else b += seg;
      });
      return `<path d="${a}" stroke="${up}" stroke-width="${bw}" fill="none"/><path d="${b}" stroke="${dn}" stroke-width="${bw}" fill="none"/>`;
    };
    if (rep.sets[set + '_rand']) g += bars(rep.sets[set + '_rand'].dlp, ctrl, ctrl);
    g += bars(rep.sets[set].dlp, hot, cool);
    $('chart-legend').innerHTML = sw('--hot', 'chance of weird / supports') + sw('--cool', 'suppresses') + sw('--ctrl', 'random control') + '<span>support in nats</span>';
  } else {
    $('chart-legend').innerHTML = sw('--hot', 'chance of weird');
  }
  const step = niceStep(R);
  for (let c = 0; c <= R - step / 3; c += step) g += `<text x="${x(c)}" y="${axisY + 10}" text-anchor="${c === 0 ? 'start' : 'middle'}">${c}</text>`;
  g += `<text x="${W - Rm}" y="${axisY + 10}" text-anchor="end">${R} tokens</text>`;
  g += `<line id="cursor" x1="${x(state.j + 0.5)}" x2="${x(state.j + 0.5)}" y1="${top - 6}" y2="${axisY - 2}" stroke="${acc}" stroke-width="2"/>`;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('height', H);
  svg.innerHTML = g;
  const pick = e => {
    const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * W / r.width;
    selectToken(Math.max(0, Math.min(R - 1, Math.floor((px - L) / (W - L - Rm) * R))), true);
  };
  svg.onpointerdown = e => { pick(e); svg.setPointerCapture(e.pointerId); svg.onpointermove = pick; };
  svg.onpointerup = () => { svg.onpointermove = null; };
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
  const cur = document.getElementById('cursor'), G = state.geom;
  if (cur && G) { const xx = G.L + (G.W - G.L - G.Rm) * (j + 0.5) / G.R; cur.setAttribute('x1', xx); cur.setAttribute('x2', xx); }
  history.replaceState(null, '', `#${state.file}.${rep.i}.${j}`);

  const F = rep.fork, atFork = F && F.c0 === j, p = pwAt(rep, j), ms = mainSet(rep);
  const pj = rep.lp ? Math.exp(rep.lp[j]) : null;
  const rank = rep.top ? rep.top[j][0].indexOf(rep.ids[j]) : -1;
  const sup = ms ? -rep.sets[ms].dlp[j] : null;
  const out = [];
  out.push(`<section class="sec">
    <div class="tokhead"><code>${esc(vis(rep.t[j]))}</code><span class="meta">Token ${j + 1} of ${rep.t.length}${atFork ? ' · <span class="forktag">fork token</span>' : ''}</span></div>
    <div class="kv">
      <div><label>Chance of weird</label><b>${p ? pct(p.P) : pct(state.doc.rate)}</b><span>${p ? (p.measured ? `measured here, ${p.n ?? '8+'} continuations` : `carried from the cut after token ${p.c}`) : 'prompt rate; normal replies were not resampled'}</span></div>
      <div><label>Model's probability</label><b>${prob(pj)}</b><span>${rank >= 0 ? `rank ${rank + 1} of its top 5` : 'outside its top 5'}</span></div>
      <div><label>Set support</label><b>${sup == null ? '–' : sup.toFixed(1)}</b><span>nats, ${ms === 'own100' ? 'own top 100' : 'shared mid 100'}</span></div>
    </div>
    <div class="actions">${F && !atFork ? `<button type="button" class="linkbtn" id="go-fork">Go to the fork, token ${F.c0 + 1}</button>` : ''}<button type="button" class="linkbtn" id="copy-link">Copy link to this token</button><span class="foot" id="copy-status"></span></div>
  </section>`);
  if (atFork) {
    const nxt = [...F.next].sort((a, b) => b.p - a.p);
    const extra = [`Re-measured with 24 continuations each: ${pct(F.confirm[0])} → ${pct(F.confirm[2])}.`];
    if (F.on) extra.push(`With a thinking-on judge the jump is ${f2(F.on.jump)}.`);
    if (F.p2) extra.push(`Phase 2 fork; fresh re-check ${pct(F.p2.Plo_fresh)} → ${pct(F.p2.Phi_fresh)}.`);
    out.push(`<section class="sec"><h4>The fork</h4><p class="lead">The chance of weird goes from ${pct(F.P0)} before this token to ${pct(F.P1)} after it.</p>
      <table class="t"><thead><tr><th>If the model wrote</th><th>Prob.</th><th>Then weird</th><th>n</th></tr></thead><tbody>
      ${nxt.map(n => `<tr class="${n.tok === rep.ids[j] ? 'here' : ''}"><td class="tk">${esc(vis(n.s))}</td><td>${prob(n.p)}</td><td>${pct(n.q)}</td><td>${n.n}</td></tr>`).join('')}
      </tbody></table><p class="note">${extra.join(' ')}</p></section>`);
  }
  if (rep.top) {
    const [ids, ps] = rep.top[j];
    const rows = ids.map((id, k) => [vis(state.doc.vocab[id] ?? `#${id}`), ps[k], prob(ps[k]), id === rep.ids[j]]);
    out.push(`<section class="sec"><h4>Model's top 5 here</h4>${rowsHTML(rows, Math.max(...ps))}</section>`);
  }
  if (rep.sets) {
    const order = ['own100', 'own100_rand', 'shared100', 'shared100_rand'].filter(s => rep.sets[s]);
    out.push(`<section class="sec"><h4>Removing neurons</h4><table class="t"><thead><tr><th>Set zeroed</th><th>Support</th><th>Next-token KL</th></tr></thead><tbody>
      ${order.map(s => `<tr class="${s.endsWith('rand') ? 'ctrl' : ''}"><td>${SETS[s]}${s.endsWith('rand') ? ' (control)' : ''}</td><td>${(-rep.sets[s].dlp[j]).toFixed(2)}</td><td>${rep.sets[s].kl[j].toFixed(3)}</td></tr>`).join('')}
      </tbody></table><p class="note">Support: how much less likely this token gets without the set, in nats (2.3 is a tenfold drop). KL: how far the whole next-token distribution moves.</p></section>`);
  }
  if (rep.acts) {
    const b = state.index.behaviors.find(x => x.id === state.b);
    const info = new Map((b.top10 || []).map(n => [`${n.L}:${n.n}`, n]));
    const rows = rep.neurons.map(([L, n], k) => {
      const m = Math.max(...rep.acts.map(a => Math.abs(a[k])), 1e-6), v = rep.acts[j][k], nfo = info.get(`${L}:${n}`) || {};
      const lab = nfo.reads || (nfo.plus || []).slice(0, 4).join(' · ');
      return `<span class="k" title="${esc((nfo.plus || []).join(' '))}">L${L} N${n}</span><span class="b div"><i class="${v < 0 ? 'neg' : ''}" style="width:${50 * Math.abs(v) / m}%;background:var(${v < 0 ? '--cool' : '--hot'})"></i></span><span class="v">${v.toFixed(2)}</span>${lab ? `<span class="lab">${esc(lab)}</span>` : ''}`;
    }).join('');
    out.push(`<section class="sec"><h4>Own top-10 neurons at this step</h4><div class="rows">${rows}</div>
      <p class="note">Each bar is scaled to that neuron's largest activation in this reply. Labels are Agastya's readings of the neuron's output direction.</p></section>`);
  }
  out.push(`<section class="sec foot"><b>Oracle-lens readout:</b> not computed yet. It will appear here once the lens rebuild is checked against Agastya's saved readouts.</section>`);
  $('inspector').innerHTML = out.join('');
  $('copy-link').onclick = async () => {
    try { await navigator.clipboard.writeText(location.href); $('copy-status').textContent = 'Copied'; }
    catch { $('copy-status').textContent = location.href; }
  };
  if ($('go-fork')) $('go-fork').onclick = () => selectToken(F.c0, true);
}

// ------------------------------------------------------------------ boot
function openHash() {
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
  $('mode-select').onchange = () => applyMode($('mode-select').value);
  for (const b of document.querySelectorAll('.seg button')) b.onclick = () => {
    state.view = b.dataset.view;
    try { localStorage.setItem('wcf-view', state.view); } catch {}
    if (state.rep) { renderTokens(state.mode); selectToken(state.j); }
  };
  $('source-line').textContent = `Data: ${state.index.source}. Built ${state.index.built}. Per-token neuron data from our own forward passes on an A100.`;
  let t;
  window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(() => state.rep && renderChart(), 120); });
  await openHash();
}
init().catch(showError);
