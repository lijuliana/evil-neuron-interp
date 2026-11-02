"""CHIVE Eval B step 96: scoring (findings/chive/design.md 8 + the 2026-11-01 design-review updates).

Inputs: runs/chive/claims.jsonl, runs/chive/labels.jsonl (+ labels_summary.json), runs/chive/pred/{arm}.jsonl,
optional data/chive/in_sample.json (prompt_ids). Outputs: runs/chive/analysis.json and runs/chive/analysis.md.

Arms by name: plain (transcript only, the reference), plain_rep (independent replicate: predictor noise), pos
(positive control), {method}, {method}_ctrl (same-prompt control), {method}_shuf (cross-prompt derangement).
Arms are given explicitly (--arms or --manifest); nothing else in the pred dir is read.
Claim sets: each method family {plain, plain_rep, m, m_ctrl, m_shuf} is scored on the claims that all its arms
answered (non-null prob_true) and whose evidence is not marked meta.na in m, m_ctrl or m_shuf (pos: {plain, pos};
noise: {plain, plain_rep}); each family is bootstrapped on its own set. The intersection over ALL arms is reported only
as a sensitivity check. Drops, refusals (per arm x behavior) and NA exclusions are reported.
Settings check: every row of every compared arm must share answerer_model, effort, true_share_pct, safety_context and
system_base_sha256 (system prompt minus the evidence sentence); otherwise the script stops.

Metrics
- PRIMARY: prompt-stratified AUROC = concordance over TRUE-FALSE pairs from the same prompt, pairs pooled across
  prompts, ties 0.5 (per-prompt evidence can only raise pooled AUROC by re-ranking prompts; this removes that route).
- Secondary: pooled AUROC, rank sum with average ranks for ties (CHIVE run_binary_eval.py:391-410 and
  experiments/main_counterfactual_eval/analyze.py:62-78 @ 2423952, the CHIVE-comparable number); accuracy@0.5; Brier.
- CIs: paired cluster bootstrap over pattern_id, 10,000 seeded resamples, 95% percentile (np.percentile as CHIVE; CHIVE
  itself resamples claims, 2,000 draws). Each resample draws the subset's patterns with replacement; all arms are scored
  on the same draws, so deltas are paired. Implemented with integer cluster weights (identical to scoring the explicitly
  duplicated claims; a duplicated prompt is a separate stratum). Resamples with no TRUE-FALSE pair are dropped, counted.
- p-values: two-sided bootstrap, p = 2 min((1 + #d<=0), (1 + #d>=0)) / (1 + B_valid), capped at 1.
- Primary test: per method, delta stratified AUROC vs plain, Holm-corrected over the methods present; a method helps
  only if Holm p < .05 with delta > 0 AND its delta vs {method}_ctrl is > 0. Minimum detectable delta at 80% power =
  (z_{1-a/2} + z_.80) x bootstrap SE, for a = .05 and for the Holm first step a = .05/m.
- Logistic regression of is_true on logit(P_arm) plus the bank writer's expectation (size dummies), cluster-robust CR1
  SEs by pattern_id, on claims with a parseable expectation: does the arm add over the writer's prior?
- Writer's expectation as a reference predictor (large 1, small .5, none 0) and every arm within the subset the writer
  expected to change (size large or small).
- Breakdowns: split, edit source, edit kind, behavior, in_sample. Sensitivity: excluding frame_flag, trunc_flag, both,
  and sens:excl_low_n = only claims whose base and edit verdicts are all non-null (n = window); conditions below the
  pre-registered 87.5% minimum are already excluded upstream by 94 ("insufficient").
- Label stats over every edit: TRUE / FALSE / dead zone / missing, unparseable-verdict exclusions, flags, drift.

  python -I scripts/96_chive_score.py --arms plain plain_rep pos neurons neurons_ctrl neurons_shuf ...
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from statistics import NormalDist

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CTRL_SUFFIXES = ("_ctrl",)
SHUF_SUFFIXES = ("_shuf", "_shuffled", "_shuffle")
SPECIAL = ("plain", "plain_rep", "pos")
WRITER_SCORE = {"large": 1.0, "small": 0.5, "none": 0.0}
LOGIT_EPS = 0.005
Z = NormalDist().inv_cdf


# ---- CHIVE experiments/main_counterfactual_eval/analyze.py:62-78 @ 2423952 (verbatim logic) ----
def rankdata_avg(a):
    a = np.asarray(a, float); n = len(a); order = np.argsort(a, kind="mergesort")
    sv = a[order]; ranks = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
    return ranks


def auroc(y, p):
    y = np.asarray(y); p = np.asarray(p); r = rankdata_avg(p)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


# ---------------------------------------------------------------- AUROC variants with cluster weights
def pair_counts(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """(concordant + 0.5 ties, n_pos * n_neg) over all TRUE-FALSE pairs."""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0, 0.0
    d = pos[:, None] - neg[None, :]
    return float((d > 0).sum() + 0.5 * (d == 0).sum()), float(len(pos) * len(neg))


def stratified_tables(y, s, prompt, cluster, K):
    """Per cluster: summed within-prompt concordance numerator and pair count."""
    num, den = np.zeros(K), np.zeros(K)
    for p in np.unique(prompt):
        m = prompt == p
        a, b = pair_counts(y[m], s[m])
        k = cluster[m][0]
        num[k] += a; den[k] += b
    return num, den


def stratified_auroc(y, s, prompt) -> float:
    a = b = 0.0
    for p in np.unique(prompt):
        m = prompt == p
        x, z = pair_counts(y[m], s[m])
        a += x; b += z
    return a / b if b else float("nan")


def cluster_tables(scores, y, cluster, K):
    """Per cluster x score-tie-group counts of positives and negatives (score groups ascending)."""
    _, g = np.unique(scores, return_inverse=True)
    G = int(g.max()) + 1
    A_pos = np.zeros((K, G)); A_neg = np.zeros((K, G))
    np.add.at(A_pos, (cluster[y == 1], g[y == 1]), 1.0)
    np.add.at(A_neg, (cluster[y == 0], g[y == 0]), 1.0)
    return A_pos, A_neg


def weighted_auroc(C, A_pos, A_neg):
    """Pooled AUROC for each row of cluster multiplicities C (B x K): P(s_pos > s_neg) + 0.5 P(tie), weights = C."""
    P = C @ A_pos; N = C @ A_neg
    below = np.cumsum(N, axis=1) - N
    num = (P * (below + 0.5 * N)).sum(1)
    den = P.sum(1) * N.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def weighted_stratified(C, num, den):
    a, b = C @ num, C @ den
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def draw_clusters(K: int, n_boot: int, seed: int, subset: str) -> np.ndarray:
    rng = np.random.default_rng([seed, stable_seed(subset)])
    return rng.multinomial(K, np.full(K, 1.0 / K), size=n_boot).astype(float)


def ci(x):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return None
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def boot_p(d) -> float | None:
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return None
    lo, hi = (1 + (d <= 0).sum()) / (1 + len(d)), (1 + (d >= 0).sum()) / (1 + len(d))
    return float(min(1.0, 2 * min(lo, hi)))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


# ---------------------------------------------------------------- logistic regression, CR1 cluster-robust SEs
def logistic_cr1(X: np.ndarray, y: np.ndarray, groups: np.ndarray, names: list[str]) -> dict:
    b = np.zeros(X.shape[1])
    conv = False
    try:
        for _ in range(100):
            p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
            W = p * (1 - p)
            H = X.T @ (X * W[:, None])
            step = np.linalg.solve(H, X.T @ (y - p))
            b = b + step
            if np.max(np.abs(step)) < 1e-10:
                conv = True
                break
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        H = X.T @ (X * (p * (1 - p))[:, None])
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError as e:
        return {"error": f"singular: {e}"}
    sc = X * (y - p)[:, None]
    gs = np.unique(groups)
    S = np.zeros((X.shape[1], X.shape[1]))
    for g in gs:
        s = sc[groups == g].sum(0)
        S += np.outer(s, s)
    G, N, K = len(gs), X.shape[0], X.shape[1]
    c = G / (G - 1) * (N - 1) / (N - K) if G > 1 and N > K else float("nan")
    se = np.sqrt(np.diag(c * Hinv @ S @ Hinv))
    zv = b / se
    pv = 2 * (1 - np.vectorize(NormalDist().cdf)(np.abs(zv)))
    return {"converged": conv, "n": int(N), "n_clusters": int(G),
            "note": None if conv else "did not converge (likely separation); coefficients unreliable",
            "coef": {n: {"b": float(bi), "se_cr1": float(si), "z": float(zi), "p": float(pi)}
                     for n, bi, si, zi, pi in zip(names, b, se, zv, pv)}}


# ---------------------------------------------------------------- io + helpers
def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256_file(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def base_method(arm: str, suffixes) -> str | None:
    for suf in suffixes:
        if arm.endswith(suf):
            return arm[: -len(suf)]
    return None


def role_of(arm: str) -> str:
    if arm in SPECIAL:
        return arm
    if base_method(arm, CTRL_SUFFIXES):
        return "ctrl"
    if base_method(arm, SHUF_SUFFIXES):
        return "shuf"
    return "method"


def writer_score(e) -> tuple[float | None, str]:
    """Bank writer's expectation -> (score, size): size large=1, small=.5, none=0."""
    if e is None:
        return None, "missing"
    size = None
    if isinstance(e, dict):
        size = e.get("size", e.get("magnitude"))
        if size is None and str(e.get("direction", "")).lower() in ("none", "no_change", "unchanged", "same"):
            size = "none"
    elif isinstance(e, str):
        size = e
    s = str(size).lower() if size is not None else ""
    for k in ("large", "small", "none"):
        if k in s:
            return WRITER_SCORE[k], k
    if any(w in s for w in ("no change", "no_change", "unchanged")):
        return 0.0, "none"
    return None, f"unparsed:{str(e)[:60]}"


def load_in_sample(path: str, field: str) -> set[str] | None:
    """data/chive/in_sample.json from 93_chive_evidence.py insample: {"_meta", "prompts": {pid: {field: bool}}};
    a plain list of prompt_ids is also accepted."""
    if not path or not os.path.exists(path):
        return None
    d = json.load(open(path))
    if isinstance(d, dict) and "prompts" in d:
        return {pid for pid, r in d["prompts"].items() if r.get(field)}
    return set(d)


# ---------------------------------------------------------------- core
class Data:
    def __init__(self, Y, S, prompt, cluster):
        self.Y, self.S, self.prompt, self.cluster = Y, S, prompt, cluster


def score_subset(name: str, idx: np.ndarray, D: Data, S: dict, comparisons: list[tuple[str, str]],
                 n_boot: int, seed: int) -> dict:
    y = D.Y[idx]
    out = {"n": int(len(idx)), "n_true": int(y.sum()), "arms": {}, "deltas": {}}
    if len(idx) == 0:
        return out
    pr = D.prompt[idx]
    uc, cl = np.unique(D.cluster[idx], return_inverse=True)
    K = len(uc)
    out["n_prompts"], out["n_clusters"] = int(len(np.unique(pr))), int(K)
    out["n_pairs_within_prompt"] = int(sum(pair_counts(y[pr == p], np.zeros((pr == p).sum()))[1]
                                           for p in np.unique(pr)))
    plain = S["plain"][idx] if "plain" in S else None
    for a, s_all in S.items():
        s = s_all[idx]
        out["arms"][a] = {"auroc_strat": stratified_auroc(y, s, pr), "auroc": auroc(y, s),
                          "acc": float(((s >= 0.5) == (y == 1)).mean()), "brier": float(np.mean((s - y) ** 2)),
                          "mean_p_true": float(s[y == 1].mean()) if y.sum() else None,
                          "mean_p_false": float(s[y == 0].mean()) if (1 - y).sum() else None,
                          "mean_abs_dp_vs_plain": float(np.abs(s - plain).mean()) if plain is not None else None}
    if y.sum() == 0 or y.sum() == len(y):
        out["note"] = "one class only: AUROC undefined"
        return out
    C = draw_clusters(K, n_boot, seed, name)
    bs, bp = {}, {}
    for a, s_all in S.items():
        s = s_all[idx]
        num, den = stratified_tables(y, s, pr, cl, K)
        bs[a] = weighted_stratified(C, num, den)
        A_pos, A_neg = cluster_tables(s, y, cl, K)
        bp[a] = weighted_auroc(C, A_pos, A_neg)
        one = np.ones((1, K))
        for got, want in ((weighted_stratified(one, num, den)[0], out["arms"][a]["auroc_strat"]),
                          (weighted_auroc(one, A_pos, A_neg)[0], out["arms"][a]["auroc"])):
            assert (np.isnan(got) and np.isnan(want)) or abs(got - want) < 1e-9, (name, a, got, want)
        out["arms"][a]["ci_strat"] = ci(bs[a])
        out["arms"][a]["ci"] = ci(bp[a])
    first = next(iter(S))
    out["n_boot"] = n_boot
    out["n_boot_valid_strat"] = int((~np.isnan(bs[first])).sum())
    out["n_boot_valid"] = int((~np.isnan(bp[first])).sum())
    for a, b in comparisons:
        if a not in S or b not in S:
            continue
        rec = {"a": a, "b": b}
        for tag, boots, key in (("strat", bs, "auroc_strat"), ("pooled", bp, "auroc")):
            d = boots[a] - boots[b]
            dv = d[~np.isnan(d)]
            rec[tag] = {"delta": out["arms"][a][key] - out["arms"][b][key], "ci": ci(d), "p": boot_p(d),
                        "se": float(dv.std(ddof=1)) if len(dv) > 1 else None}
        out["deltas"][f"{a} - {b}"] = rec
    return out


def label_stats(labels: list[dict]) -> dict:
    def tab(key):
        t = collections.defaultdict(collections.Counter)
        for r in labels:
            t[str(r.get(key))][r["label"]] += 1
        return {k: dict(v) for k, v in sorted(t.items())}
    nulls = collections.defaultdict(lambda: {"base_nulls": 0, "edit_nulls": 0, "edits_with_null": 0})
    seen_base = set()
    for r in labels:
        b = nulls[str(r.get("behavior_id"))]
        b["edit_nulls"] += r.get("n1_null") or 0
        b["edits_with_null"] += int((r.get("n1_null") or 0) > 0)
        if r["prompt_id"] not in seen_base:
            seen_base.add(r["prompt_id"])
            b["base_nulls"] += r.get("n0_null") or 0
    return {"all": dict(collections.Counter(r["label"] for r in labels)), "split": tab("split"),
            "source": tab("source"), "kind": tab("kind"), "behavior_id": tab("behavior_id"),
            "null_exclusions_by_behavior": dict(nulls),
            "flags": {f: dict(collections.Counter(str(r.get(f)) for r in labels)) for f in ("frame_flag", "trunc_flag")}}


# ---------------------------------------------------------------- families, settings, NA
SETTINGS = ("answerer_model", "effort", "true_share_pct", "safety_context", "system_base_sha256")


def parse_arms(args) -> list[str]:
    arms = json.load(open(args.manifest))["arms"] if args.manifest else args.arms
    if not arms:
        raise SystemExit("pass the arms to score explicitly: --arms plain ... or --manifest file.json {\"arms\": [...]}")
    assert "plain" in arms, f"no plain arm among {arms}"
    assert len(set(arms)) == len(arms), f"duplicate arms {arms}"
    return ["plain"] + [a for a in arms if a != "plain"]


def check_settings(rows_by_arm: dict) -> dict:
    """Every row of every compared arm must share model, effort, TRUE share, safety context and the system prompt
    minus the evidence sentence; otherwise fail loudly (stale or mixed runs)."""
    seen = {f: collections.defaultdict(set) for f in SETTINGS}
    for a, rows in rows_by_arm.items():
        for r in rows.values():
            for f in SETTINGS:
                if f not in r:
                    raise SystemExit(f"arm {a}: rows lack '{f}' (written by an older 95; rerun it, cache makes it free)")
                seen[f][json.dumps(r[f])].add(a)
    bad = {f: {k: sorted(v) for k, v in d.items()} for f, d in seen.items() if len(d) > 1}
    if bad:
        raise SystemExit(f"answerer settings differ across the compared arms: {json.dumps(bad)}")
    return {f: json.loads(next(iter(d))) for f, d in seen.items() if d}


def na_claims(path: str, claims: list[dict]) -> set[str] | None:
    """claim_ids whose evidence row in `path` is marked meta.na (claim-keyed rows win over prompt-keyed)."""
    if not os.path.exists(path):
        return None
    na_p, na_c, have_c = set(), set(), set()
    for r in read_jsonl(path):
        is_na = bool((r.get("meta") or {}).get("na"))
        if r.get("claim_id") is not None:
            have_c.add(r["claim_id"])
            if is_na:
                na_c.add(r["claim_id"])
        elif is_na:
            na_p.add(r["prompt_id"])
    return {c["claim_id"] for c in claims
            if c["claim_id"] in na_c or (c["claim_id"] not in have_c and c["prompt_id"] in na_p)}


def build_families(arms: list[str]) -> dict[str, list[str]]:
    fams = {}
    rep = ["plain_rep"] if "plain_rep" in arms else []
    used = {"plain", "plain_rep"}
    for m in [a for a in arms if role_of(a) == "method"]:
        ctl = [a for a in arms if base_method(a, CTRL_SUFFIXES + SHUF_SUFFIXES) == m]
        fams[f"method:{m}"] = ["plain"] + rep + [m] + ctl
        used |= {m, *ctl}
    if "pos" in arms:
        fams["pos"] = ["plain", "pos"]
        used.add("pos")
    if rep:
        fams["noise"] = ["plain", "plain_rep"]
    for a in arms:
        if a not in used:
            print(f"[score] WARNING arm {a} has no method arm; scored alone against plain")
            fams[f"orphan:{a}"] = ["plain", a]
    return fams or {"plain": ["plain"]}


def family_comparisons(members: list[str]) -> list[tuple[str, str]]:
    comps = [(a, "plain") for a in members if a != "plain"]
    for m in [a for a in members if role_of(a) == "method"]:
        comps += [(m, a) for a in members if base_method(a, CTRL_SUFFIXES + SHUF_SUFFIXES) == m]
    return comps


def make_data(cc: list[dict], members: list[str], preds: dict, cluster_key: str):
    Y = np.array([int(bool(c["is_true"])) for c in cc])
    S = {a: np.array([float(preds[a][c["claim_id"]]) for c in cc]) for a in members}
    D = Data(Y, S, np.unique([c["prompt_id"] for c in cc], return_inverse=True)[1],
             np.unique([str(c[cluster_key]) for c in cc], return_inverse=True)[1])
    return D, S


def make_subsets(cc: list[dict], in_sample: set | None) -> dict[str, np.ndarray]:
    subs = {"all": np.arange(len(cc))}
    fields = [("split", lambda c: c.get("split")), ("source", lambda c: c.get("source")),
              ("kind", lambda c: c.get("kind")), ("behavior_id", lambda c: c.get("behavior_id"))]
    if in_sample is not None:
        fields.append(("in_sample", lambda c: c["prompt_id"] in in_sample))
    for key, f in fields:
        vals = np.array([str(f(c)) for c in cc])
        for v in sorted(set(vals)):
            subs[f"{key}={v}"] = np.where(vals == v)[0]
    ws = [writer_score(c.get("expectation"))[1] for c in cc]
    subs["writer_expected_change"] = np.array([i for i, w in enumerate(ws) if w in ("large", "small")], dtype=int)
    ff = np.array([c.get("frame_flag") is True for c in cc])
    tf = np.array([c.get("trunc_flag") is True for c in cc])
    low = np.array([(c.get("n_baseline", 0) < (c.get("n_base_window") or 64)) or
                    (c.get("n_counterfactual", 0) < (c.get("n_edit_window") or 32)) for c in cc])
    subs["sens:excl_frame_flag"] = np.where(~ff)[0]
    subs["sens:excl_trunc_flag"] = np.where(~tf)[0]
    subs["sens:excl_both_flags"] = np.where(~ff & ~tf)[0]
    subs["sens:excl_low_n"] = np.where(~low)[0]
    return subs


def regression(cc: list[dict], D: Data, S: dict) -> dict | None:
    ws = [writer_score(c.get("expectation")) for c in cc]
    widx = np.array([i for i, w in enumerate(ws) if w[0] is not None], dtype=int)
    if len(widx) == 0:
        return None
    size = [ws[i][1] for i in widx]
    cols, names = [np.ones(len(widx))], ["const"]
    for lvl in ("large", "small"):
        v = np.array([float(s == lvl) for s in size])
        if 0 < v.sum() < len(v):
            cols.append(v); names.append(f"writer_{lvl}")
    Xw = np.column_stack(cols)
    yw, gw = D.Y[widx].astype(float), D.cluster[widx]
    out = {"n": int(len(widx)), "logit_eps": LOGIT_EPS, "writer_only": logistic_cr1(Xw, yw, gw, names), "arms": {}}
    for a, s in S.items():
        p = np.clip(s[widx], LOGIT_EPS, 1 - LOGIT_EPS)
        out["arms"][a] = logistic_cr1(np.column_stack([Xw, np.log(p / (1 - p))]), yw, gw, names + ["logit_p"])
    return out


# ---------------------------------------------------------------- markdown
def fmt(x, nd=3, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def fmt_ci(c, signed=False):
    return f"[{fmt(c[0], signed=signed)}, {fmt(c[1], signed=signed)}]" if c else ""


def fmt_p(p):
    return "n/a" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def md_subset(L, name, sub):
    L += ["", f"#### {name}: n={sub['n']} ({sub['n_true']} TRUE), {sub.get('n_prompts', 0)} prompts, "
              f"{sub.get('n_clusters', 0)} patterns, {sub.get('n_pairs_within_prompt', 0)} within-prompt pairs", ""]
    if sub.get("note"):
        L.append(f"_{sub['note']}_")
    L += ["| arm | strat AUROC [CI] | pooled AUROC [CI] | acc | Brier | mean P T/F |", "|---|---|---|---|---|---|"]
    for a, m in sub["arms"].items():
        L.append(f"| {a} | {fmt(m['auroc_strat'])} {fmt_ci(m.get('ci_strat'))} | {fmt(m['auroc'])} "
                 f"{fmt_ci(m.get('ci'))} | {fmt(m['acc'])} | {fmt(m['brier'])} | "
                 f"{fmt(m['mean_p_true'], 2)}/{fmt(m['mean_p_false'], 2)} |")
    if sub["deltas"]:
        L += ["", "| comparison | Δ strat [CI] p | Δ pooled [CI] p |", "|---|---|---|"]
        for k, d in sub["deltas"].items():
            L.append(f"| {k} | {fmt(d['strat']['delta'], signed=True)} {fmt_ci(d['strat']['ci'], True)} "
                     f"p={fmt_p(d['strat']['p'])} | {fmt(d['pooled']['delta'], signed=True)} "
                     f"{fmt_ci(d['pooled']['ci'], True)} p={fmt_p(d['pooled']['p'])} |")


def md_breakdown(L, fam, members, res_f):
    focus = next((a for a in members if role_of(a) in ("method", "pos", "plain_rep")), None) or members[-1]
    key = f"{focus} - plain"
    L += ["", f"Breakdowns (stratified AUROC; Δ = {key}):", "",
          "| subset | n (TRUE) | " + " | ".join(members) + f" | Δ strat [CI] |", "|---" * (len(members) + 3) + "|"]
    for name, sub in res_f.items():
        if name == "all":
            continue
        d = sub["deltas"].get(key, {}).get("strat", {})
        L.append(f"| {name} | {sub['n']} ({sub['n_true']}) | " +
                 " | ".join(fmt(sub["arms"].get(a, {}).get("auroc_strat")) for a in members) +
                 f" | {fmt(d.get('delta'), signed=True)} {fmt_ci(d.get('ci'), True)} |")


def write_md(res: dict, path: str) -> None:
    m = res["meta"]
    L = ["# CHIVE Eval B: analysis", "",
         f"Arms {m['arms']}. Settings (identical across arms, checked): {json.dumps(m['settings'])}. Primary metric: "
         f"prompt-stratified AUROC (TRUE-FALSE pairs within a prompt, ties 0.5); pooled AUROC is the CHIVE-comparable "
         f"secondary. 95% CIs: paired cluster bootstrap over {m['cluster_key']} ({m['n_boot']:,} resamples, seed "
         f"{m['seed']}). Each family {{plain, plain_rep, m, m_ctrl, m_shuf}} is scored on the claims all its arms "
         f"answered with non-NA evidence; the intersection over all arms is only a sensitivity check.", ""]
    pr = res["primary"]
    L += ["## Primary test", "",
          f"Family: {pr['family']} (Holm over {len(pr['family'])}). helps = Holm p < .05 with Δ > 0 vs plain AND "
          f"Δ strat vs the same-prompt control > 0.", "",
          "| method | n claims | Δ strat vs plain [CI] | p | Holm p | MDE80 (a=.05 / Holm) | Δ vs _ctrl [CI] | "
          "Δ vs _shuf [CI] | helps | global-set Δ [CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, h in pr["methods"].items():
        L.append(f"| {k} | {h['n_claims']} | {fmt(h['delta'], signed=True)} {fmt_ci(h['ci'], True)} | "
                 f"{fmt_p(h['p'])} | {fmt_p(h['p_holm'])} | {fmt(h['mde80'])} / {fmt(h['mde80_holm'])} | "
                 f"{fmt(h.get('delta_vs_ctrl'), signed=True)} {fmt_ci(h.get('ci_vs_ctrl'), True)} | "
                 f"{fmt(h.get('delta_vs_shuf'), signed=True)} {fmt_ci(h.get('ci_vs_shuf'), True)} | {h['helps']} | "
                 f"{fmt(h.get('global_delta'), signed=True)} {fmt_ci(h.get('global_ci'), True)} |")
    nz = res.get("noise")
    if nz:
        L += ["", f"Predictor noise (plain vs plain_rep, n={nz['n']}): r = {fmt(nz['pearson_r'])}, mean |ΔP| = "
                  f"{fmt(nz['mean_abs_dp'])}, {fmt(nz['frac_flip_05'])} of claims change side of 0.5; Δ strat "
                  f"{fmt(nz['delta_strat'], signed=True)} {fmt_ci(nz['ci_strat'], True)}."]
    L += ["", "## Drops per arm (all claims)", "",
          "| arm | role | rows | null | of which refusals | missing | evidence NA | answered |",
          "|---|---|---|---|---|---|---|---|"]
    for a, d in res["drops"]["arms"].items():
        L.append(f"| {a} | {d['role']} | {d['n_rows']} | {d['n_null']} | {d.get('n_refusal', 0)} | {d['n_missing']} | "
                 f"{d.get('n_na', 'n/a')} | {d['n_answered']} |")
    ref = res["drops"].get("refusals", {})
    behs = sorted({b for v in ref.values() for b in v["by_behavior"]}, key=str)
    if behs:
        L += ["", "Refusals (status refusal after retries) per arm and behavior:", "",
              "| arm | " + " | ".join(map(str, behs)) + " | categories |", "|---" * (len(behs) + 2) + "|"]
        for a, v in ref.items():
            L.append(f"| {a} | " + " | ".join(str(v["by_behavior"].get(b, 0)) for b in behs) +
                     f" | {v['categories']} |")
    for fam, fr in res["families"].items():
        L += ["", f"## Family {fam}: arms {fr['arms']}", "",
              f"Claims: {fr['n_claims']} of {res['drops']['n_claims']} (dropped: {json.dumps(fr['dropped'])})."]
        md_subset(L, "all", fr["subsets"]["all"])
        md_breakdown(L, fam, fr["arms"], fr["subsets"])
        reg = fr.get("regression")
        if reg:
            L += ["", f"Logistic regression is_true ~ writer size dummies + logit(P), CR1 SEs by {m['cluster_key']}, "
                      f"n={reg['n']}. Writer only: " + ", ".join(
                          f"{k} {fmt(v['b'], signed=True)} (se {fmt(v['se_cr1'])})"
                          for k, v in reg.get("writer_only", {}).get("coef", {}).items()), "",
                  "| arm | b logit(P) | se CR1 | z | p |", "|---|---|---|---|---|"]
            for a, r in reg["arms"].items():
                c = r.get("coef", {}).get("logit_p")
                if c and not r.get("converged"):
                    L.append(f"| {a} | {r.get('note')} | | | |")
                    continue
                L.append(f"| {a} | " + (f"{fmt(c['b'], signed=True)} | {fmt(c['se_cr1'])} | {fmt(c['z'], 2)} | "
                                        f"{fmt_p(c['p'])} |" if c else f"{r.get('error')} | | | |"))
    if res.get("global_intersection"):
        md_subset(L, "Sensitivity: intersection over ALL arms", res["global_intersection"])
    w = res.get("writer_reference")
    if w:
        L += ["", "## Bank writer's expectation as a reference predictor", "",
              f"Large 1, small .5, none 0; claims plain answered with a parseable expectation: n={w['n']} "
              f"({w['n_true']} TRUE); values {w['values']}."]
        if w.get("scores"):
            md_subset(L, "writer reference vs plain", w["scores"])
    ls = res.get("label_stats")
    if ls:
        L += ["", "## Labels (all edits)", "", "| group | TRUE | FALSE | dead zone | other |", "|---|---|---|---|---|"]

        def lrow(name, c):
            other = sum(v for k, v in c.items() if k not in ("TRUE", "FALSE", "dead_zone"))
            return f"| {name} | {c.get('TRUE', 0)} | {c.get('FALSE', 0)} | {c.get('dead_zone', 0)} | {other} |"
        L.append(lrow("all", ls["all"]))
        for key in ("split", "source", "kind", "behavior_id"):
            for k, c in ls[key].items():
                L.append(lrow(f"{key}={k}", c))
        L += ["", f"Unparseable-verdict exclusions by behavior: {json.dumps(ls['null_exclusions_by_behavior'])}",
              f"Flags: {json.dumps(ls['flags'])}"]
    if res.get("labels_summary"):
        s = res["labels_summary"]
        a = (s.get("drift_same_run_vs_stage_a") or {}).get("all", {})
        L += [f"Rule: {json.dumps(s.get('rule'))}", f"Null exclusions: {json.dumps(s.get('null_exclusions'))}",
              f"Agreement re-judge: {json.dumps(s.get('agreement_rejudge'))}; frame noise reference: "
              f"{json.dumps(s.get('frame_noise_reference'))}; seed coupling: "
              f"{json.dumps(s.get('seed_coupling', {}).get('mean_r'))}",
              f"Drift (same-run base - stage A): n={a.get('n_prompts')}, mean {fmt(a.get('mean_diff_base_minus_stage_a'))}, "
              f"mean |diff| {fmt(a.get('mean_abs_diff'))}, r {fmt(a.get('pearson_r'))}."]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def clean(o):
    """JSON-safe copy: NaN -> None, numpy scalars -> Python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", default=os.path.join(REPO, "runs/chive/claims.jsonl"))
    ap.add_argument("--labels", default=os.path.join(REPO, "runs/chive/labels.jsonl"))
    ap.add_argument("--labels-summary", default=os.path.join(REPO, "runs/chive/labels_summary.json"))
    ap.add_argument("--pred-dir", default=os.path.join(REPO, "runs/chive/pred"))
    ap.add_argument("--evidence-dir", default=os.path.join(REPO, "data/chive/evidence"),
                    help="{arm}.jsonl evidence files, read for meta.na")
    ap.add_argument("--in-sample", default=os.path.join(REPO, "data/chive/in_sample.json"))
    ap.add_argument("--in-sample-field", default="in_sample_own_set",
                    help="flag in in_sample.json prompts rows (93 marks in_sample_own_set as primary)")
    ap.add_argument("--arms", nargs="*", default=None, help="arms to score (explicit; plain required)")
    ap.add_argument("--manifest", default=None, help='JSON file {"arms": [...]} instead of --arms')
    ap.add_argument("--cluster-key", default="pattern_id", help="claim field the bootstrap resamples")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-json", default=os.path.join(REPO, "runs/chive/analysis.json"))
    ap.add_argument("--out-md", default=os.path.join(REPO, "runs/chive/analysis.md"))
    args = ap.parse_args()

    claims = read_jsonl(args.claims)
    by_id = {c["claim_id"]: c for c in claims}
    missing_key = [c["claim_id"] for c in claims if c.get(args.cluster_key) is None]
    assert not missing_key, f"{len(missing_key)} claims lack {args.cluster_key} (e.g. {missing_key[:3]})"
    arms = parse_arms(args)

    rows_by_arm, preds = {}, {}
    drops = {"arms": {}, "refusals": {}, "n_claims": len(claims)}
    for a in arms:
        path = os.path.join(args.pred_dir, f"{a}.jsonl")
        assert os.path.exists(path), f"arm {a}: {path} not found"
        rows = {}
        for r in read_jsonl(path):
            assert r["claim_id"] in by_id, f"{a}: claim {r['claim_id']} not in claims"
            assert bool(r["is_true"]) == bool(by_id[r["claim_id"]]["is_true"]), f"{a}: label mismatch {r['claim_id']}"
            assert r["claim_id"] not in rows, f"{a}: duplicate row for {r['claim_id']}"
            rows[r["claim_id"]] = r
        rows_by_arm[a] = rows
        preds[a] = {cid: r["prob_true"] for cid, r in rows.items()}
        refused = [r for r in rows.values() if r.get("status") == "refusal" or
                   (r["prob_true"] is None and str(r.get("error") or "").startswith("refusal"))]
        drops["refusals"][a] = {"n": len(refused), "by_behavior": dict(collections.Counter(
            by_id[r["claim_id"]].get("behavior_id") for r in refused)),
            "categories": dict(collections.Counter(str(((r.get("refusals") or [{}])[-1].get("stop_details") or {})
                                                       .get("category")) for r in refused))}
        n_null = sum(v is None for v in preds[a].values())
        drops["arms"][a] = {"role": role_of(a), "n_rows": len(rows), "n_null": n_null, "n_refusal": len(refused),
                            "n_missing": len(set(by_id) - set(rows)), "n_answered": len(rows) - n_null,
                            "errors": dict(collections.Counter(
                                (r.get("error") or "").split(":")[0] for r in rows.values() if r["prob_true"] is None))}
    settings = check_settings(rows_by_arm)

    na = {}
    for a in arms:
        if role_of(a) in ("method", "ctrl", "shuf", "pos"):
            na[a] = na_claims(os.path.join(args.evidence_dir, f"{a}.jsonl"), claims)
            if na[a] is None:
                print(f"[score] WARNING no evidence file for {a} in {args.evidence_dir}: NA restriction not applied")
                na[a] = set()
            drops["arms"][a]["n_na"] = len(na[a])

    in_sample = load_in_sample(args.in_sample, args.in_sample_field)
    families = build_families(arms)
    fam_res = {}
    for fam, members in families.items():
        answered = [c for c in claims if all(preds[a].get(c["claim_id"]) is not None for a in members)]
        na_any = set().union(*[na.get(a, set()) for a in members])
        cc = [c for c in answered if c["claim_id"] not in na_any]
        dropped = {a: sum(preds[a].get(c["claim_id"]) is None for c in claims) for a in members}
        dropped["evidence_na"] = len(answered) - len(cc)
        fr = {"arms": members, "n_claims": len(cc), "dropped": dropped, "subsets": {}}
        if cc:
            D, S = make_data(cc, members, preds, args.cluster_key)
            comps = family_comparisons(members)
            for name, idx in make_subsets(cc, in_sample).items():
                fr["subsets"][name] = score_subset(f"{fam}|{name}", idx, D, S, comps, args.n_boot, args.seed)
            fr["regression"] = regression(cc, D, S)
            al = fr["subsets"]["all"]["arms"]
            print(f"[score] {fam}: {len(cc)} claims; strat " + " ".join(
                f"{a}={fmt(v['auroc_strat'])}" for a, v in al.items()), flush=True)
        fam_res[fam] = fr

    # sensitivity: intersection over ALL arms (and no NA anywhere)
    na_all = set().union(*na.values()) if na else set()
    cg = [c for c in claims if all(preds[a].get(c["claim_id"]) is not None for a in arms)
          and c["claim_id"] not in na_all]
    glob_res = None
    if cg:
        D, S = make_data(cg, arms, preds, args.cluster_key)
        comps = sorted({x for f in families.values() for x in family_comparisons(f)}, key=str)
        glob_res = score_subset("global|all", np.arange(len(cg)), D, S, comps, args.n_boot, args.seed)

    # primary test: Holm over methods on each method family's delta stratified AUROC vs plain
    methods = [a for a in arms if role_of(a) == "method"]
    z80 = Z(0.80)
    pv, prim = {}, {"family": methods, "alpha": args.alpha, "methods": {}}
    for mth in methods:
        sub = fam_res[f"method:{mth}"]["subsets"].get("all", {})
        d = sub.get("deltas", {}).get(f"{mth} - plain")
        if d and d["strat"]["p"] is not None:
            pv[mth] = d["strat"]["p"]
    ph = holm(pv) if pv else {}
    for mth in methods:
        fr = fam_res[f"method:{mth}"]
        sub = fr["subsets"].get("all", {})
        d = sub.get("deltas", {}).get(f"{mth} - plain")
        if d is None:
            continue
        ctrl = next((a for a in fr["arms"] if base_method(a, CTRL_SUFFIXES) == mth), None)
        shuf = next((a for a in fr["arms"] if base_method(a, SHUF_SUFFIXES) == mth), None)
        dc = sub["deltas"].get(f"{mth} - {ctrl}") if ctrl else None
        dsh = sub["deltas"].get(f"{mth} - {shuf}") if shuf else None
        gd = (glob_res or {}).get("deltas", {}).get(f"{mth} - plain")
        se = d["strat"]["se"]
        sig = ph.get(mth) is not None and ph[mth] < args.alpha and d["strat"]["delta"] > 0
        prim["methods"][mth] = {
            "n_claims": fr["n_claims"], "delta": d["strat"]["delta"], "ci": d["strat"]["ci"], "p": d["strat"]["p"],
            "p_holm": ph.get(mth), "se": se,
            "mde80": None if se is None else (Z(1 - args.alpha / 2) + z80) * se,
            "mde80_holm": None if se is None else (Z(1 - args.alpha / (2 * max(len(pv), 1))) + z80) * se,
            "ctrl_arm": ctrl, "delta_vs_ctrl": dc["strat"]["delta"] if dc else None,
            "ci_vs_ctrl": dc["strat"]["ci"] if dc else None, "shuf_arm": shuf,
            "delta_vs_shuf": dsh["strat"]["delta"] if dsh else None, "ci_vs_shuf": dsh["strat"]["ci"] if dsh else None,
            "helps": (bool(sig and dc["strat"]["delta"] > 0) if dc else ("no _ctrl arm" if sig else False)),
            "global_delta": gd["strat"]["delta"] if gd else None, "global_ci": gd["strat"]["ci"] if gd else None,
        }

    noise = None
    if "noise" in fam_res and fam_res["noise"]["n_claims"]:
        cc = [c for c in claims if preds["plain"].get(c["claim_id"]) is not None
              and preds["plain_rep"].get(c["claim_id"]) is not None]
        a = np.array([preds["plain"][c["claim_id"]] for c in cc]); b = np.array([preds["plain_rep"][c["claim_id"]] for c in cc])
        dr = fam_res["noise"]["subsets"]["all"]["deltas"].get("plain_rep - plain", {}).get("strat", {})
        noise = {"n": len(cc), "pearson_r": float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None,
                 "mean_abs_dp": float(np.abs(a - b).mean()), "frac_flip_05": float(((a >= .5) != (b >= .5)).mean()),
                 "delta_strat": dr.get("delta"), "ci_strat": dr.get("ci")}

    # writer expectation as a reference predictor, on the claims plain answered
    cp = [c for c in claims if preds["plain"].get(c["claim_id"]) is not None]
    ws = [writer_score(c.get("expectation")) for c in cp]
    widx = np.array([i for i, w in enumerate(ws) if w[0] is not None], dtype=int)
    writer = {"n": int(len(widx)), "values": dict(collections.Counter(w[1] for w in ws)), "n_true": 0}
    if len(widx):
        D, S = make_data(cp, ["plain"], preds, args.cluster_key)
        writer["n_true"] = int(D.Y[widx].sum())
        S["writer_expectation"] = np.array([w[0] if w[0] is not None else np.nan for w in ws])
        writer["scores"] = score_subset("writer_reference", widx, D, S, [("plain", "writer_expectation")],
                                        args.n_boot, args.seed)

    labels = read_jsonl(args.labels) if os.path.exists(args.labels) else []
    lsum = json.load(open(args.labels_summary)) if os.path.exists(args.labels_summary) else None
    if lsum:
        lsum.pop("base", None)
    res = {
        "meta": {"arms": arms, "roles": {a: role_of(a) for a in arms}, "families": families, "settings": settings,
                 "n_boot": args.n_boot, "seed": args.seed, "cluster_key": args.cluster_key, "alpha": args.alpha,
                 "primary_metric": "prompt-stratified AUROC (within-prompt TRUE-FALSE pairs, ties 0.5)",
                 "in_sample_file": args.in_sample if in_sample is not None else None,
                 "in_sample_field": args.in_sample_field,
                 "n_in_sample_prompts": len(in_sample) if in_sample is not None else None,
                 "inputs_sha256": {"claims": sha256_file(args.claims), "labels": sha256_file(args.labels),
                                   **{f"pred/{a}": sha256_file(os.path.join(args.pred_dir, f"{a}.jsonl"))
                                      for a in arms}}},
        "primary": prim, "noise": noise, "drops": drops, "families": fam_res,
        "global_intersection": dict(glob_res, n_claims=len(cg)) if glob_res else None,
        "writer_reference": writer, "label_stats": label_stats(labels) if labels else None, "labels_summary": lsum,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(clean(res), f, indent=1, allow_nan=False)
    write_md(res, args.out_md)
    for k, h in prim["methods"].items():
        print(f"[score] PRIMARY {k} (n={h['n_claims']}): Δstrat {fmt(h['delta'], signed=True)} {fmt_ci(h['ci'], True)} "
              f"Holm p {fmt_p(h['p_holm'])} vs ctrl {fmt(h['delta_vs_ctrl'], signed=True)} -> helps={h['helps']}")
    print(f"[score] -> {args.out_json}, {args.out_md}")


if __name__ == "__main__":
    main()
