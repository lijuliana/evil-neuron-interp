const $ = id => document.getElementById(id);
const esc = v => String(v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c]));
const vis = t => (t || '').replaceAll(' ', '·').replaceAll('\n', '↵').replaceAll('\t', '⇥') || '∅';
const pct = x => x == null ? '–' : `${Math.round(100 * x)}%`;
const f2 = x => x == null ? '–' : x.toFixed(2);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const SETS = {
  own100: 'Own top 100', own100_rand: 'Random 100', shared100: 'Shared mid 100', shared100_rand: 'Random 100',
};
const MODES = [
  { id: 'pw', label: 'Chance of turning weird (measured)' },
  { id: 'own100', label: 'Own top-100 set: support for each token', set: 'own100' },
  { id: 'own100_rand', label: 'Random control for the own set', set: 'own100_rand' },
  { id: 'shared100', label: 'Shared mid-layer set: support for each token', set: 'shared100' },
  { id: 'shared100_rand', label: 'Random control for the shared set', set: 'shared100_rand' },
  { id: 'kl', label: 'Next-token shift when the set is removed (KL)' },
  { id: 'surprise', label: 'Model surprise at the token (−log p)' },
  { id: 'none', label: 'No highlight' },
];
const state = { index: null, b: null, file: null, doc: null, rep: null, j: 0, mode: 'pw', bounds: false };
const docs = new Map();

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw Error(`${path}: ${r.status}`);
  return r.json();
}
function showError(e) { $('inspector').innerHTML = `<p class="err">Could not load this view: ${esc(e.message)}</p>`; }

// ------------------------------------------------------------------ values per token
function pwAt(rep, j) {
  if (!rep.curve) return null;
  let best = null;
  for (const [c, P, n] of rep.curve) if (c <= j + 1) best = { c, P, n, measured: c === j + 1 };
  return best;
}
function mainSet(rep) { return rep.sets?.own100 ? 'own100' : rep.sets?.shared100 ? 'shared100' : null; }
function modeValue(rep, j, mode) {
  if (mode === 'pw') { const p = pwAt(rep, j); return p ? { v: p.P, hl: '--hot' } : null; }
  if (mode === 'surprise') return rep.lp ? { v: Math.min(1, -rep.lp[j] / 8), hl: '--cool' } : null;
  if (mode === 'kl') { const s = mainSet(rep); return s ? { v: Math.min(1, rep.sets[s].kl[j] / 2), hl: '--accent' } : null; }
  const s = rep.sets?.[mode];
  if (!s) return null;
  const sup = -s.dlp[j];
  return { v: Math.min(1, Math.abs(sup) / 4), hl: sup >= 0 ? '--hot' : '--cool' };
}
function modeAvailable(rep, mode) {
  if (mode === 'none') return true;
  if (mode === 'pw') return !!rep.curve;
  if (mode === 'surprise') return !!rep.lp;
  if (mode === 'kl') return !!mainSet(rep);
  return !!rep.sets?.[mode];
}
function legendFor(mode) {
  const sw = (v, c) => `<i class="sw" style="background:${c}"></i>${v}`;
  if (mode === 'pw') return sw('chance the reply ends up weird, given the text through this token', 'var(--hot)');
  if (mode === 'surprise') return sw('darker = less expected by the model', 'var(--cool)');
  if (mode === 'kl') return sw('darker = removing the set changes the next-token distribution more (saturates at 2 nats)', 'var(--accent)');
  if (mode === 'none') return '';
  return sw('set supports the token', 'var(--hot)') + ' ' + sw('set suppresses it', 'var(--cool)') + '<span>saturates at 4 nats</span>';
}

// ------------------------------------------------------------------ behavior level
function renderTabs() {
  $('tabs').innerHTML = state.index.behaviors.map(b =>
    `<button class="tab${b.id === state.b ? ' active' : ''}" data-b="${b.id}" ${b.text ? '' : 'disabled title="Text not shown in this build; counts only"'}>${esc(b.title)}<small>${b.forks}</small></button>`).join('');
  document.querySelectorAll('.tab[data-b]').forEach(t => t.onclick = () => selectBehavior(t.dataset.b).catch(showError));
}
function renderBehavior(b) {
  const r = b.rates;
  let right;
  if (r) {
    const rows = [['base', 'No ablation'], ['top10', 'Top 10 removed'], ['top100', 'Top 100 removed'], ['top1000', 'Top 1000 removed'],
      ['rand100', 'Random 100'], ['rand1000', 'Random 1000']].filter(([k]) => r[k] != null);
    const mx = Math.max(...rows.map(([k]) => r[k]), 0.01);
    right = `<div class="rates">${rows.map(([k, lab]) => `<span class="lab">${lab}</span><span class="bar"><i style="width:${100 * r[k] / mx}%;background:var(${k.startsWith('rand') || k === 'base' ? '--ctrl' : '--hot'})"></i></span><span class="num">${pct(r[k])}</span>`).join('')}
      <span class="rates-cap">How often the behavior occurs with each of Agastya's sets zeroed (his Phase 2 rates, Gemma judge).</span></div>`;
  } else {
    right = `<p class="note">Agastya built no per-behavior neuron set for this behavior. The viewer shows his shared mid-layer set (layers 24 to 47), which lowers all eight behaviors he tested.</p>`;
  }
  $('beh').innerHTML = `<div><h2>${esc(b.title)}</h2><p>${esc(b.desc)}</p>
    <div class="stats"><span><b>${b.prompts}</b> prompts with a fork</span><span><b>${b.forks}</b> forked replies</span><span><b>${b.sharp}</b> sharp forks (jump ≥ 0.5)</span><span><b>${b.p2}</b> in Phase 2</span></div></div><div>${right}</div>`;
}
function maxJump(p) { return Math.max(-1, ...p.replies.filter(r => r.kind === 'fork').map(r => r.jump ?? 0)); }

async function selectBehavior(bid, file = null, i = null, j = null) {
  state.b = bid;
  const b = state.index.behaviors.find(x => x.id === bid);
  renderTabs(); renderBehavior(b);
  const ps = state.index.prompts.filter(p => p.b === bid && p.file).sort((a, c) => maxJump(c) - maxJump(a));
  $('prompt-select').innerHTML = ps.map(p => `<option value="${p.file}">${pct(p.rate)} weird · ${p.split} · ${esc(p.preview.replace(/\s+/g, ' ').slice(0, 90))}</option>`).join('');
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
  $('chat').innerHTML = doc.chat.map(t => `<div class="turn"><span class="role">${esc(t.role)}</span>${esc(t.text)}</div>`).join('') +
    `<p class="note">Pattern ${esc(doc.pattern)} · ${doc.split} split · ${doc.k ?? '–'} of ${doc.n ?? '–'} sampled replies judged weird (thinking off, temperature 1).</p>`;
  $('chat').scrollTop = $('chat').scrollHeight;
  $('reply-select').innerHTML = doc.replies.map(r => {
    const w = r.off?.match ? 'weird' : 'normal';
    const f = r.fork ? ` · fork “${vis(r.fork.tok)}” ${f2(r.fork.P0)}→${f2(r.fork.P1)}` : ' · contrast';
    return `<option value="${r.i}">Sample ${r.i} · ${w}${f}</option>`;
  }).join('');
  $('reply-select').onchange = () => selectReply(Number($('reply-select').value));
  const pick = doc.replies.find(r => r.i === i) ? i : doc.replies[0].i;
  selectReply(pick, j);
}
function selectReply(i, j = null) {
  const rep = state.rep = state.doc.replies.find(r => r.i === i);
  $('reply-select').value = String(i);
  const pill = (lab, v) => `<span class="pill ${v == null ? 'na' : v.match ? 'weird' : 'normal'}">${lab}: ${v == null ? 'not judged' : v.match ? 'weird' : 'normal'}</span>`;
  $('verdict').innerHTML = pill('Gemma, thinking off', rep.off) + pill('Gemma, thinking on', rep.on) +
    `<span>${rep.t.length} tokens · stop: ${rep.finish}${rep.finish === 'length' ? ' (capped at 1024)' : ''}</span>` +
    (rep.off?.why ? `<span class="why">${esc(rep.off.why)}</span>` : '');
  // pick a mode that exists for this reply
  $('mode-select').innerHTML = MODES.map(m => `<option value="${m.id}" ${modeAvailable(rep, m.id) ? '' : 'disabled'}>${m.label}${modeAvailable(rep, m.id) ? '' : ' (n/a)'}</option>`).join('');
  if (!modeAvailable(rep, state.mode)) state.mode = rep.curve ? 'pw' : (mainSet(rep) || 'none');
  $('mode-select').value = state.mode;
  renderTokens();
  const start = Number.isInteger(j) && j >= 0 && j < rep.t.length ? j : rep.fork ? rep.fork.c0 : 0;
  selectToken(start, true);
}

function renderTokens() {
  const rep = state.rep, box = $('reply');
  box.replaceChildren();
  const cuts = new Set((rep.curve || []).map(([c]) => c - 1));
  const frag = document.createDocumentFragment();
  rep.t.forEach((t, j) => {
    const s = document.createElement('span');
    s.className = 'tok';
    s.dataset.j = j;
    s.tabIndex = -1;
    if (/^<\|.*\|>$/.test(t)) { s.classList.add('special'); s.textContent = t === '<|im_end|>' ? 'end of reply' : t; }
    else if (t.trim() === '' && t.includes('\n')) { const w = document.createElement('span'); w.className = 'ws'; w.textContent = '↵'; s.append(w, t); }
    else s.textContent = t || '·';
    if (cuts.has(j)) s.classList.add('cut');
    if (rep.fork && rep.fork.c0 === j) s.classList.add('fork');
    frag.append(s);
  });
  box.append(frag);
  box.onclick = e => { const s = e.target.closest('.tok'); if (s) selectToken(Number(s.dataset.j)); };
  box.onkeydown = e => {
    if (!['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
    e.preventDefault(); selectToken(Math.max(0, Math.min(state.rep.t.length - 1, state.j + (e.key === 'ArrowRight' ? 1 : -1))), true);
  };
  box.tabIndex = 0;
  applyMode();
}
function applyMode() {
  const rep = state.rep;
  $('reply').classList.toggle('bounded', state.bounds);
  document.querySelectorAll('#reply .tok').forEach(s => {
    const j = Number(s.dataset.j), mv = state.mode === 'none' ? null : modeValue(rep, j, state.mode);
    s.style.setProperty('--v', mv ? mv.v.toFixed(3) : 0);
    s.style.setProperty('--hl', mv ? `var(${mv.hl})` : 'transparent');
    s.classList.toggle('bounds', state.bounds);
  });
  $('tok-legend').innerHTML = legendFor(state.mode);
  renderChart();
}

// ------------------------------------------------------------------ strip chart
function chartSet(rep) {
  const m = MODES.find(x => x.id === state.mode);
  if (m?.set && rep.sets?.[m.set]) return m.set.replace('_rand', '');
  return mainSet(rep);
}
function renderChart() {
  const rep = state.rep, svg = $('chart'), W = Math.max(280, svg.clientWidth || svg.parentNode.clientWidth - 32);
  const R = rep.t.length, L = 34, Rm = 8, x = c => L + (W - L - Rm) * c / R;
  const set = chartSet(rep), lanes = set ? 2 : 1, H = lanes === 2 ? 168 : 92;
  const y1 = P => 12 + (1 - P) * 62, mid = 128, amp = 30, clip = 8;
  const hot = css('--hot'), cool = css('--cool'), ctrl = css('--ctrl'), rule = css('--rule'), acc = css('--accent'), muted = css('--muted');
  let g = '';
  for (const P of [0, 0.5, 1]) g += `<line x1="${L}" x2="${W - Rm}" y1="${y1(P)}" y2="${y1(P)}" stroke="${rule}" stroke-width="1"/><text x="${L - 6}" y="${y1(P) + 3.5}" text-anchor="end">${P === 0.5 ? '.5' : P}</text>`;
  if (rep.curve) {
    let d = '', prev = null;
    for (const [c, P] of rep.curve) {
      d += prev == null ? `M${x(c)},${y1(P)}` : `H${x(c)}V${y1(P)}`;
      prev = P;
    }
    d += `H${x(R)}`;
    g += `<path d="${d}" fill="none" stroke="${hot}" stroke-width="1.6"/>`;
    for (const [c, P, n] of rep.curve) g += `<circle cx="${x(c)}" cy="${y1(P)}" r="${n >= 16 ? 3.2 : 2.4}" fill="${hot}"><title>after ${c} tokens: ${pct(P)} weird (${n ?? '?'} judged continuations)</title></circle>`;
    if (rep.fork) { const fx = x(rep.fork.c0 + 0.5); g += `<line x1="${fx}" x2="${fx}" y1="6" y2="${H - 4}" stroke="${hot}" stroke-dasharray="3 3" stroke-width="1"/><text x="${Math.min(fx + 4, W - 30)}" y="10" fill="${hot}" style="fill:${hot}">fork</text>`; }
  } else {
    const P = state.doc.rate ?? 0;
    g += `<line x1="${L}" x2="${W - Rm}" y1="${y1(P)}" y2="${y1(P)}" stroke="${muted}" stroke-dasharray="4 3"/><text x="${L + 4}" y="${y1(P) - 4}">prompt rate ${pct(P)} · normal replies have no measured curve</text>`;
  }
  if (set) {
    g += `<line x1="${L}" x2="${W - Rm}" y1="${mid}" y2="${mid}" stroke="${rule}"/><text x="${L - 6}" y="${mid - amp + 4}" text-anchor="end">+${clip}</text><text x="${L - 6}" y="${mid + amp}" text-anchor="end">−${clip}</text>`;
    const bar = (arr, color, w) => {
      let up = '', dn = '';
      arr.forEach((d, j) => {
        const s = Math.max(-clip, Math.min(clip, -d)), h = amp * Math.abs(s) / clip;
        if (h < 0.4) return;
        const seg = `M${x(j + 0.5).toFixed(1)},${mid}v${(s > 0 ? -h : h).toFixed(1)}`;
        if (s > 0) up += seg; else dn += seg;
      });
      return `<path d="${up}" stroke="${color === 'set' ? hot : color}" stroke-width="${w}" fill="none"/><path d="${dn}" stroke="${color === 'set' ? cool : color}" stroke-width="${w}" fill="none"/>`;
    };
    const bw = Math.max(1, Math.min(4, (W - L - Rm) / R * 0.8));
    if (rep.sets[set + '_rand']) g += bar(rep.sets[set + '_rand'].dlp, ctrl, bw);
    g += bar(rep.sets[set].dlp, 'set', bw);
    $('chart-legend').innerHTML = `<span><i class="sw" style="background:var(--hot)"></i>chance of weird</span><span><i class="sw" style="background:var(--hot)"></i><i class="sw" style="background:var(--cool)"></i>${SETS[set]} support (nats)</span><span><i class="sw" style="background:var(--ctrl)"></i>random control</span>`;
  } else {
    $('chart-legend').innerHTML = `<span><i class="sw" style="background:var(--hot)"></i>chance of weird</span><span>neuron-set data not loaded</span>`;
  }
  g += `<line id="cursor" x1="${x(state.j + 0.5)}" x2="${x(state.j + 0.5)}" y1="4" y2="${H - 2}" stroke="${acc}" stroke-width="1.5"/>`;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('width', W); svg.setAttribute('height', H);
  svg.innerHTML = g;
  $('chart-title').textContent = set ? 'Chance of weird (top) · what the neuron set supports (bottom)' : 'Chance of weird along the reply';
  const pick = e => {
    const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * W / r.width;
    selectToken(Math.max(0, Math.min(R - 1, Math.floor((px - L) / (W - L - Rm) * R))), true);
  };
  svg.onpointerdown = e => { pick(e); svg.setPointerCapture(e.pointerId); svg.onpointermove = pick; };
  svg.onpointerup = () => { svg.onpointermove = null; };
}

// ------------------------------------------------------------------ inspector
function rowsHTML(rows, max, cls = '') {
  return `<div class="rows">${rows.map(([k, v, txt, here]) =>
    `<span class="k${here ? ' here' : ''}" title="${esc(k)}">${esc(k)}</span><span class="b ${cls}"><i style="width:${Math.min(100, 100 * Math.abs(v) / (max || 1)) / (cls ? 2 : 1)}%;background:var(${here ? '--accent' : '--ctrl'})"></i></span><span class="v">${txt}</span>`).join('')}</div>`;
}
function selectToken(j, scroll = false) {
  const rep = state.rep;
  if (!rep || j < 0 || j >= rep.t.length) return;
  state.j = j;
  document.querySelectorAll('#reply .tok.sel').forEach(s => s.classList.remove('sel'));
  const span = document.querySelector(`#reply .tok[data-j="${j}"]`);
  if (span) { span.classList.add('sel'); if (scroll) span.scrollIntoView({ block: 'nearest' }); }
  const cur = document.getElementById('cursor');
  if (cur) { const b = $('chart').viewBox.baseVal, W = b.width, R = rep.t.length, xx = 34 + (W - 42) * (j + 0.5) / R; cur.setAttribute('x1', xx); cur.setAttribute('x2', xx); }
  history.replaceState(null, '', `#${state.file}.${rep.i}.${j}`);
  const out = [];
  out.push(`<div class="tokhead"><code>${esc(vis(rep.t[j]))}</code><span class="meta">token ${j + 1} of ${rep.t.length} · id ${rep.ids[j]}</span></div>`);
  // chance of weird
  const p = pwAt(rep, j);
  if (p) {
    const how = p.measured ? `Measured right after this token from ${p.n ?? '8+'} judged continuations.` : `Carried forward from the last measured cut, after token ${p.c}.`;
    out.push(`<section class="sec"><h3>Chance the reply ends up weird</h3><div class="big">${pct(p.P)}</div><p class="note">${how}</p></section>`);
  } else {
    out.push(`<section class="sec"><h3>Chance the reply ends up weird</h3><p class="note">This is a normal reply, so it was not cut and resampled. Across 64 samples of this prompt, ${pct(state.doc.rate)} were judged weird.</p></section>`);
  }
  // fork decomposition
  const F = rep.fork;
  if (F && F.c0 === j) {
    const nxt = [...F.next].sort((a, b) => b.p - a.p);
        out.push(`<section class="sec"><h3>The fork</h3><p class="note">Before this token the chance was ${pct(F.P0)}; after it, ${pct(F.P1)}. Re-measured with 24 continuations each: ${pct(F.confirm[0])} → ${pct(F.confirm[2])}.${F.on ? ` With a thinking-on judge the jump is ${f2(F.on.jump)}.` : ''}${F.p2 ? ` Phase 2 fork; fresh re-check ${pct(F.p2.Plo_fresh)} → ${pct(F.p2.Phi_fresh)}.` : ''}</p>
      <p class="note">Candidates the model could have written here, and the chance of weird after each:</p>
      <table class="sets"><thead><tr><th>Token</th><th>Model prob.</th><th>Then weird</th><th>n</th></tr></thead><tbody>
      ${nxt.map(n => `<tr class="${n.tok === rep.ids[j] ? 'here' : ''}"><td>${esc(vis(n.s))}</td><td>${pct(n.p)}</td><td>${pct(n.q)}</td><td>${n.n}</td></tr>`).join('')}
      </tbody></table></section>`);
  }
  // model's own distribution
  if (rep.top) {
    const [ids, ps] = rep.top[j];
    const rows = ids.map((id, k) => [vis(state.doc.vocab[id] ?? `#${id}`), ps[k], pct(ps[k]), id === rep.ids[j]]);
    const inTop = ids.includes(rep.ids[j]);
    out.push(`<section class="sec"><h3>What the model considered here</h3>${rowsHTML(rows, Math.max(...ps))}
      <p class="note">This token had probability ${pct(Math.exp(rep.lp[j]))}${inTop ? '' : ', outside the top 5'}.</p></section>`);
  }
  // sets
  if (rep.sets) {
    const order = ['own100', 'own100_rand', 'shared100', 'shared100_rand'].filter(s => rep.sets[s]);
    out.push(`<section class="sec"><h3>Removing neurons</h3><table class="sets"><thead><tr><th>Set zeroed</th><th>Support for this token</th><th>KL of next token</th></tr></thead><tbody>
      ${order.map(s => `<tr class="${s.endsWith('rand') ? 'ctrl' : ''}"><td>${SETS[s]}</td><td>${(-rep.sets[s].dlp[j]).toFixed(2)}</td><td>${rep.sets[s].kl[j].toFixed(3)}</td></tr>`).join('')}
      </tbody></table><p class="note">Support is how much less likely this token gets without the set, in nats; 2.3 nats is a tenfold drop. KL measures how much the whole next-token distribution moves.</p></section>`);
  }
  // neuron activations
  if (rep.acts) {
    const b = state.index.behaviors.find(x => x.id === state.b);
    const info = new Map((b.top10 || []).map(n => [`${n.L}:${n.n}`, n]));
    const col = k => rep.acts.map(a => a[k]);
    const rows = rep.neurons.map(([L, n], k) => {
      const m = Math.max(...col(k).map(Math.abs), 1e-6), v = rep.acts[j][k], nfo = info.get(`${L}:${n}`) || {};
      return `<span class="k" title="${esc((nfo.plus || []).join(' '))}">L${L} N${n}</span><span class="b div"><i class="${v < 0 ? 'neg' : ''}" style="width:${50 * Math.abs(v) / m}%;background:var(${v < 0 ? '--cool' : '--hot'})"></i></span><span class="v">${v.toFixed(2)}</span><span class="neuron-lab" style="grid-column:1/-1;margin:-2px 0 3px">${esc(nfo.reads || (nfo.plus || []).slice(0, 4).join(' · ') || '')}</span>`;
    }).join('');
    out.push(`<section class="sec"><h3>Own top-10 neurons when choosing this token</h3><div class="rows">${rows}</div>
      <p class="note">Bars are scaled to each neuron's largest activation in this reply. Labels are Agastya's readings of each neuron's output direction.</p></section>`);
  }
  out.push(`<div class="olens"><b>Oracle-lens readout.</b> Not computed yet. Planned for this position once the lens rebuild is checked against Agastya's saved readouts.</div>`);
  out.push(`<div class="linkrow"><button id="copy-link" type="button">Copy link to this token</button><span class="meta" id="copy-status"></span></div>`);
  $('inspector').innerHTML = out.join('');
  $('copy-link').onclick = async () => {
    try { await navigator.clipboard.writeText(location.href); $('copy-status').textContent = 'Copied'; }
    catch { $('copy-status').textContent = location.href; }
  };
}

// ------------------------------------------------------------------ boot
function openHash() {
  const m = /^#([0-9a-f]{16})\.(\d+)(?:\.(\d+))?$/.exec(location.hash);
  if (m) {
    const p = state.index.prompts.find(x => x.file === m[1]);
    if (p) return selectBehavior(p.b, p.file, Number(m[2]), m[3] != null ? Number(m[3]) : null);
  }
  const first = state.index.behaviors.find(b => b.text);
  return selectBehavior(first.id);
}
async function init() {
  state.index = await getJSON('index.json');
  $('mode-select').onchange = () => { state.mode = $('mode-select').value; applyMode(); };
  $('bounds').onchange = () => { state.bounds = $('bounds').checked; applyMode(); };
  $('source-line').textContent = `Data: ${state.index.source}. Built ${state.index.built}.${state.index.public ? ' Published build: replies are shown only for the six behaviors Agastya quotes on his own site; the rest appear as counts.' : ''}`;
  let t;
  window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(() => state.rep && renderChart(), 120); });
  await openHash();
}
init().catch(showError);
