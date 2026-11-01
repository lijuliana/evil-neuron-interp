"""Eval A metrics harness: one set of positions, folds, metrics and bootstrap for every monitor (runs/monitors/SPEC.md,
Metrics and Test sets).

Input: runs/monitors/scores/<method>.jsonl, rows {sid, pos, variant, score[, outer]}, higher score = predicts weird.
  (b)/(bh): only unique positions (canon == sid) need scores; they are copied to the duplicates here via seqs_b canon.
  "outer": t marks an inner out-of-fold score computed with fold t held out entirely (nested stacking, see below).
  Every key must be a position of data/olens/seqs.jsonl or seqs_b / seqs_bh: unmatched keys stop the run. Duplicate
  keys are averaged with a warning.
Optional sidecar runs/monitors/scores/<method>.meta.json: {"label": str, "primary": variant | [variants] | null,
  "oof": bool, "old": bool, "missing": ["sid@pos", ...]}. Primaries come ONLY from the sidecar or --primary
  method=variant (no name-based defaults). A primary variant must cover every (a) interior cut and every (b) k = 4
  unique position except the keys listed in "missing" (e.g. scorer refusals; their (b) duplicates are missing too);
  otherwise the run stops (--allow-incomplete: the rows are marked INCOMPLETE and left out of the Holm family).

Test sets and metrics (all CIs: 1,000 bootstrap resamples of PATTERNS, stratified by behavior, the same draws for
every monitor so paired differences are consistent; percentile 95% CIs. Family p-values are ONE-SIDED z-tests on the
bootstrap SE in the pre-declared direction (higher = weird): p = P(Z > (estimate - null) / SE_boot); the two-sided
percentile bootstrap p in the CSVs is exploratory):
  (b) PRIMARY  mean within-cut AUROC (midrank ties) of weird vs normal continuations at k = 4 (all cuts) and k = 1 (cuts
      with >= 1 untied weird-normal pair); equal-weight behavior average; tie ceilings (pair ceiling from cuts_b, and
      the achievable ceiling = a monitor that scores each shared prefix by its weird fraction). Gain over the text
      judge = within-cut AUROC of the out-of-fold conditional logit (strata = cuts, exact conditional likelihood,
      folds.json) of [text, method] minus [text].
  (bh) the same on the held-out denying-AI cuts; the conditional logit is fit on all of (b) and applied to (bh).
  (a) PRIMARY  within-reply partial Spearman of score vs P_pooled controlling for rank(c), interior cuts 1 <= c < L,
      mean over replies with >= 4 interior cuts, MINUS the method's own residual-permutation null: per reply the score
      is fit on [1, log1p c, c], its residuals are permuted within the reply (20 permutations, the same draws for every
      monitor) and the partial recomputed. The null removes the negative bias the linear rank(c) partial gives to any
      score that carries position (cuts are dense early, where P climbs). Observed, null and observed - null are all
      reported; the test is observed - null > 0. Gain = within-reply partial Spearman of the out-of-fold OLS prediction
      of P from [text, log1p c, c, sqrt c, method] minus that from [text, log1p c, c, sqrt c]. Conventions: a reply
      whose P is constant or exactly monotone in c is dropped for every monitor; a monitor constant (or exactly
      monotone in c) within a reply scores 0 there (no information) instead of being dropped.
      Secondary: raw within-reply Spearman (interior / P_pooled; and Agastya's curve P with the c = 0 and c = L
      endpoints, Fig 1 continuity), pooled Spearman, text-blind subset (text <= 0.1), per behavior.
  forks  46 sharp forks, rel = c - c_fork in -40..+20: mean score per rel; rel 0 (state that picks the fork token)
      and rel +1; jump (+1 minus 0) and rise (mean rel -5..0 minus mean rel -40..-21), CIs over forks.
  (h) last prompt token vs the prompt's thinking-off rate over 64 samples, discovery and held-out separately.
  LOBO: variants whose name contains "lobo" are reported per behavior in lobo.csv (passthrough).
  Holm: across every declared primary variant x {(b) AUROC k4 > .5, (b) gain k4 > 0, (a) partial - null > 0,
      (a) gain > 0}, one-sided z-tests.
  Coverage: every table has a coverage column (positions scored / expected) per set.
Nested stacking: if a method's scores are out-of-fold predictions (meta "oof": true), the gain rows are nested only
when the file also has rows with "outer": t for the training-fold positions; otherwise they are marked "not nested".

Outputs (runs/monitors/eval/): table_b.csv, table_a.csv, table_h.csv, forks.csv, forks_curves.csv,
by_behavior_a.csv, by_behavior_b.csv, lobo.csv, holm.csv, report.md; --validate also writes validation.md.
Usage: .venv/bin/python -I scripts/55_monitor_eval.py [--methods a,b] [--primary m=v ...] [--validate] [--convert]
  --methods     evaluate only these score files (stems); default every runs/monitors/scores/*.jsonl. The text judge
                (text_a / text_b) is always read for the gain rows even when not listed.
  --validate    also reproduce runs/olens/analysis.md (scripts/89) with the old settings, unit-test the metrics and
                check the reviewer's position / partial / noise-ceiling numbers -> eval/validation.md
  --convert     rewrite scores/text_a.jsonl and scores/olens_old.jsonl from runs/olens (the existing (a) scores)
Caches (delete to rebuild): eval/_prompt_patterns.json, eval/_noise_ceiling_v2.json, eval/_text_notes.json.
  --allow-incomplete  incomplete primaries are marked INCOMPLETE and left out of Holm instead of stopping the run
"""
import argparse, glob, gzip, json, os, re, sys, time, warnings
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, rankdata

M, A, O = "runs/monitors", "data/agastya", "runs/olens"
SEED, EPS = 0, 1e-12
DEFAULT_META = {  # used when a score file has no .meta.json sidecar
    "text_a": {"label": "M0 text judge, (a)/(h) (runs/olens/text_judge.jsonl, 0-1)", "primary": None},
    "text_b": {"label": "M0 text judge, (b)/(bh) (scripts/92, 0-1)", "primary": None},
    "olens_old": {"label": "M1 O-lens, OLD 0-10 thinking-on scorer, exploratory", "primary": None, "old": True},
}


# ----------------------------------------------------------------------------------------------------------- basics
def rank(x):
    return rankdata(x)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return (a * b).sum() / d if d > EPS else np.nan


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return corr(rank(a[m]), rank(b[m])) if m.sum() > 2 else np.nan


def pval(boots, null):
    b = np.asarray(boots, float)
    b = b[~np.isnan(b)]
    if not len(b):
        return np.nan
    lo, hi = (b <= null).sum(), (b >= null).sum()
    return min(1.0, 2 * (min(lo, hi) + 1) / (len(b) + 1))


def pz(est, boots, null):
    """One-sided z-test on the bootstrap SE, direction pre-declared (estimate > null)."""
    b = np.asarray(boots, float)
    b = b[~np.isnan(b)]
    se = b.std(ddof=1) if len(b) > 2 else np.nan
    if not np.isfinite(se) or se <= 0 or not np.isfinite(est):
        return np.nan
    return float(norm.sf((est - null) / se))


def ci(boots):
    b = np.asarray(boots, float)
    b = b[~np.isnan(b)]
    return (np.percentile(b, 2.5), np.percentile(b, 97.5)) if len(b) else (np.nan, np.nan)


def fmt(v, lo=None, hi=None, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    s = f"{v:+.{d}f}"
    return s if lo is None or np.isnan(lo) else f"{s} [{lo:+.{d}f}, {hi:+.{d}f}]"


def holm(ps):
    ps = np.asarray(ps, float)
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


class Boot:
    """Pattern bootstrap stratified by behavior. W[r, j] = how often pattern j is drawn in replicate r."""

    def __init__(self, pat_of_unit, beh_of_pat, B=1000, seed=SEED):
        self.unit_pat = dict(pat_of_unit)
        self.pats = sorted(set(self.unit_pat.values()))
        self.pidx = {p: i for i, p in enumerate(self.pats)}
        self.beh = np.array([beh_of_pat[p] for p in self.pats])
        rng, self.B = np.random.default_rng(seed), B
        self.W = np.zeros((B, len(self.pats)))
        for b in sorted(set(self.beh)):
            ix = np.where(self.beh == b)[0]
            draws = rng.choice(ix, size=(B, len(ix)))
            np.add.at(self.W, (np.repeat(np.arange(B), len(ix)), draws.ravel()), 1)
        self._rows = {}

    def _sums(self, vals):
        v = vals.dropna()
        v = v[v.index.isin(list(self.unit_pat))]
        pi = np.array([self.pidx[self.unit_pat[u]] for u in v.index], int)
        return (np.bincount(pi, weights=v.values, minlength=len(self.pats)),
                np.bincount(pi, minlength=len(self.pats)).astype(float))

    def mean(self, vals, by_beh=False):
        """vals: Series indexed by unit. Mean over units (or equal-weight mean of behavior means) + replicates."""
        s, c = self._sums(vals)
        if c.sum() == 0:
            return np.nan, np.full(self.B, np.nan)
        if not by_beh:
            with np.errstate(invalid="ignore", divide="ignore"):
                return s.sum() / c.sum(), (self.W @ s) / (self.W @ c)
        ests, boots = [], []
        for b in sorted(set(self.beh)):
            m = self.beh == b
            if c[m].sum() == 0:
                continue
            ests.append(s[m].sum() / c[m].sum())
            with np.errstate(invalid="ignore", divide="ignore"):
                boots.append((self.W[:, m] @ s[m]) / (self.W[:, m] @ c[m]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a replicate can miss every unit of a small behavior
            return np.mean(ests), np.nanmean(np.array(boots), 0)

    def rows(self, name, unit_of_row):
        """Row-index resamples (for non-decomposable statistics such as pooled Spearman), cached per table."""
        if name not in self._rows:
            by_p = defaultdict(list)
            for i, u in enumerate(unit_of_row):
                if u in self.unit_pat:
                    by_p[self.pidx[self.unit_pat[u]]].append(i)
            by_p = {j: np.array(v) for j, v in by_p.items()}
            reps = []
            for r in range(self.B):
                reps.append(np.concatenate([np.tile(by_p[j], int(n)) for j, n in enumerate(self.W[r]) if n and j in by_p]))
            self._rows[name] = reps
        return self._rows[name]

    def stat(self, name, unit_of_row, fn):
        return np.array([fn(ix) for ix in self.rows(name, unit_of_row)])


# ------------------------------------------------------------------------------------------------------ positions
class Ctx:
    def __init__(self, boot_n=1000):
        t0 = time.time()
        self.folds = json.load(open(f"{M}/folds.json"))
        split = json.load(open(f"{A}/split.json"))
        rates = {r["prompt_id"]: r for r in json.load(open(f"{A}/stage_a_rates_off.json"))}
        self.pattern = self._patterns()
        olens = {}
        for l in open("data/olens/seqs.jsonl"):
            s = json.loads(l)
            if s["kind"] in ("prompt", "reply", "raw"):
                olens[s["sid"]] = s
        self.olens = olens
        known = []  # every valid position key: all of data/olens/seqs.jsonl (any kind) and every seqs_b / seqs_bh record
        for l in open("data/olens/seqs.jsonl"):
            s_ = json.loads(l)
            known += [f"{s_['sid']}@{p_}" for p_ in s_["pos"]]
        # (h) prompts
        H = []
        for pid, r in rates.items():
            s = olens[f"prompt|{pid}"]
            H.append({"pid": pid, "b": r["behavior_id"], "split": split[pid]["split"], "pattern": self.pattern[pid],
                      "fold": self.folds[pid], "rate": r["rate"], "P": s["pos"][0] + 1,
                      "key": f"prompt|{pid}@{s['pos'][0]}"})
        self.H = pd.DataFrame(H)
        Plen = dict(zip(self.H.pid, self.H.P))
        self.beh_of_pat = {self.pattern[p]: b for p, b in zip(self.H.pid, self.H.b)}
        # (a) cut points: every point of Agastya's curve, file order (scripts/89 order, needed for its old folds)
        pp = {}
        for k, v in json.load(open(f"{M}/p_pooled.json")).items():
            rid, c = k.rsplit("|", 1)
            pp[(rid, int(c))] = v
        rows, self.L, self.cfork = [], {}, {}
        for l in open(f"{A}/forks_off.jsonl"):
            f = json.loads(l)
            rid, pid = f["rid"], f["prompt_id"]
            self.L[rid] = f["L"]
            rs = olens[f"reply|{rid}"]
            P = Plen[pid]
            assert rs["pos"][0] - rs["c"][0] + 1 == P
            cpos = dict(zip(rs["c"], rs["pos"]))
            for c, Pc in f["curve"]:
                if c >= 1:
                    assert cpos.get(c, P + c - 1) == P + c - 1
                key = f"reply|{rid}@{P + c - 1}" if c >= 1 else f"prompt|{pid}@{P - 1}"
                rows.append({"rid": rid, "pid": pid, "b": f["behavior_id"], "pattern": self.pattern[pid],
                             "fold": self.folds[pid], "c": c, "L": f["L"], "Pcurve": Pc,
                             "P": pp.get((rid, c), [np.nan])[0], "nP": pp.get((rid, c), [np.nan, 0])[1],
                             "interior": 1 <= c < f["L"], "key": key})
        self.A_all = pd.DataFrame(rows)
        self.A = self.A_all[self.A_all.interior].reset_index(drop=True)
        assert self.A.P.notna().all() and len(self.A) == len(pp)
        self.A["logc"] = np.log1p(self.A.c)
        # sharp-fork windows (chat and raw re-read)
        F = []
        for l in open(f"{A}/phase2_forks.jsonl"):
            f = json.loads(l)
            rid, c0 = f["rid"], f["fork"][0]
            self.cfork[rid] = (c0, f["Plo_on"], f["Phi_on"])
            pid = rid.split("|")[0]
            rs = olens[f"reply|{rid}"]
            cs = [(0, f"prompt|{pid}@{Plen[pid] - 1}")] + [(c, f"reply|{rid}@{p}") for c, p in zip(rs["c"], rs["pos"])]
            raw = olens.get(f"raw|{rid}")
            rawk = {c: f"raw|{rid}@{p}" for c, p in zip(raw["c"], raw["pos"])} if raw else {}
            for c, key in cs:
                if -40 <= c - c0 <= 20:
                    F.append({"rid": rid, "pid": pid, "b": f["behavior_id"], "pattern": self.pattern[pid], "c": c,
                              "rel": c - c0, "key": key, "key_raw": rawk.get(c)})
        self.F = pd.DataFrame(F)
        # (b) / (bh) matched continuations
        self.B, self.cuts = {}, {}
        for name in ("b", "bh"):
            cuts = {f"{c['rid']}|{c['cut']}": c for c in map(json.loads, open(f"{M}/cuts_{name}.jsonl"))}
            self.cuts[name] = cuts
            R = []
            for s in map(json.loads, open(f"{M}/seqs_{name}.jsonl")):
                for i, k in enumerate(s["k"]):
                    cid = f"{s['rid']}|{s['cut']}"
                    R.append({"sid": s["sid"], "cut_id": cid, "pid": s["pid"], "b": s["b"], "pattern": s["pattern"],
                              "fold": s["fold"], "split": s["split"], "match": bool(s["match"]), "k": k,
                              "pos": s["pos"][i], "key": f"{s['canon'][i]}@{s['pos'][i]}",
                              "own": f"{s['sid']}@{s['pos'][i]}", "unique": s["canon"][i] == s["sid"],
                              "sharp": cuts[cid]["sharp"]})
            T = pd.DataFrame(R).sort_values(["k", "cut_id", "sid"]).reset_index(drop=True)
            known += list(T.own)
            for p, b in zip(T.pattern, T.b):
                self.beh_of_pat.setdefault(p, b)
            self.B[name] = {k: self._pad(T[T.k == k].reset_index(drop=True)) for k in (1, 4)}
        self.known = pd.Index(sorted(set(known)))
        # bootstraps (one per test set; identical draws for every monitor)
        self.boot_a = Boot(dict(zip(self.A.rid, self.A.pattern)), self.beh_of_pat, boot_n)
        self.boot_b = {n: Boot(dict(zip(self.B[n][4]["T"].cut_id, self.B[n][4]["T"].pattern)), self.beh_of_pat, boot_n)
                       for n in self.B}
        self.boot_h = {sp: Boot(dict(zip(g.pid, g.pattern)), self.beh_of_pat, boot_n) for sp, g in self.H.groupby("split")}
        self._groups, self._perms = {}, {}
        print(f"positions loaded in {time.time() - t0:.0f}s: (a) {len(self.A)} interior cuts / {self.A.rid.nunique()} "
              f"replies ({len(self.A_all)} with endpoints), forks {len(self.F)} rows, (b) "
              f"{len(self.B['b'][4]['cuts'])} cuts, (bh) {len(self.B['bh'][4]['cuts'])} cuts, (h) {len(self.H)} prompts",
              flush=True)

    def _patterns(self):
        cache = f"{M}/eval/_prompt_patterns.json"
        if os.path.exists(cache):
            return json.load(open(cache))
        pat = {}
        with gzip.open(f"{A}/stage_a_gen.jsonl.gz", "rt") as f:
            for l in f:
                r = json.loads(l)
                pat[r["prompt_id"]] = r["pattern_id"]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(pat, open(cache, "w"))
        return pat

    def _pad(self, T):
        """Padded (cut x slot) layout for vectorized within-cut AUROC; tie structure and ceilings."""
        codes, cuts = pd.factorize(T.cut_id)
        slot = T.groupby("cut_id").cumcount().values
        Y = np.full((len(cuts), slot.max() + 1), -1)
        Y[codes, slot] = T.match.astype(int).values
        g = T.groupby("key").match.transform("mean").values  # achievable-ceiling monitor: weird fraction of the prefix
        X = np.full(Y.shape, np.nan)
        X[codes, slot] = T.key.map({k: i for i, k in enumerate(T.key.unique())}).values
        pair_tied = (X[:, :, None] == X[:, None, :]) & (Y[:, :, None] == 1) & (Y[:, None, :] == 0)
        n_pairs = ((Y[:, :, None] == 1) & (Y[:, None, :] == 0)).sum((1, 2))
        ceil_pair = 1 - 0.5 * pair_tied.sum((1, 2)) / n_pairs
        d = {"T": T, "codes": codes, "slot": slot, "cuts": np.asarray(cuts), "Y": Y, "ceil_pair": ceil_pair,
             "untied": ceil_pair > 0.5}
        d["ceil_best"] = self.cut_auroc(d, g)
        return d

    @staticmethod
    def cut_auroc(d, score):
        S = np.full(d["Y"].shape, np.nan)
        S[d["codes"], d["slot"]] = score
        Y = d["Y"]
        ok = ~np.isnan(S)
        w, n = (Y == 1) & ok, (Y == 0) & ok
        pair = w[:, :, None] & n[:, None, :]
        gt = S[:, :, None] > S[:, None, :]
        eq = S[:, :, None] == S[:, None, :]
        num = ((gt + 0.5 * eq) * pair).sum((1, 2))
        den = pair.sum((1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def perms(self, name, df, n_perm=20):
        """Fixed within-reply permutations (seeded once), shared by every monitor's residual-permutation null."""
        if (name, n_perm) not in self._perms:
            rng = np.random.default_rng(SEED + 1)
            self._perms[(name, n_perm)] = [np.array([rng.permutation(len(ix)) for _ in range(n_perm)])
                                           for ix in self.groups(name, df)]
        return self._perms[(name, n_perm)]

    def pad(self, name, df):
        """Replies as a padded (reply x cut) index matrix (-1 = padding), with their rids and behaviors."""
        if ("pad", name) not in self._groups:
            G = self.groups(name, df)
            IX = np.full((len(G), max(len(g) for g in G)), -1)
            for i, g in enumerate(G):
                IX[i, :len(g)] = g
            self._groups[("pad", name)] = (IX, df.rid.values[[g[0] for g in G]], df.b.values[[g[0] for g in G]])
        return self._groups[("pad", name)]

    def groups(self, name, df):
        if name not in self._groups:
            self._groups[name] = [np.asarray(ix) for ix in df.groupby("rid", sort=True).indices.values()]
        return self._groups[name]


# ---------------------------------------------------------------------------------------------------- score files
def read_scores(path, known=None):
    """Score file -> {variant: (Series key -> score, {outer t: Series})}, plus issues (unmatched keys, duplicates)."""
    d = pd.read_json(path, lines=True, dtype={"sid": str, "variant": str})
    d["key"] = d.sid.astype(str) + "@" + d.pos.astype(int).astype(str)
    d["outer"] = d.outer.fillna(-1).astype(int) if "outer" in d else -1
    d["variant"] = d.variant.astype(str)
    issues = {"unmatched": {}, "examples": {}, "dup": {}, "nan": {}}
    if known is not None:
        bad = ~d.key.isin(known)
        for v, g in d[bad].groupby("variant"):
            issues["unmatched"][v] = len(g)
            issues["examples"][v] = list(g.key.head(3))
    dup = d.duplicated(["variant", "outer", "key"], keep="first")
    for v, g in d[dup].groupby("variant"):
        issues["dup"][v] = len(g)
    for v, g in d[d.score.isna()].groupby("variant"):
        issues["nan"][v] = len(g)
    out = {}
    for v, g in d.groupby("variant", sort=True):
        main = g[g.outer < 0].groupby("key").score.mean()
        nested = {int(t): gg.groupby("key").score.mean() for t, gg in g[g.outer >= 0].groupby("outer")}
        out[v] = (main.dropna(), {t: n.dropna() for t, n in nested.items()})
    return out, issues


def load_methods(ctx, names=None, primary_over=(), scores_dir=f"{M}/scores"):
    methods, unmatched = [], []
    for path in sorted(glob.glob(f"{scores_dir}/*.jsonl")):
        stem = os.path.basename(path)[:-6]
        if names and stem not in names:
            continue
        meta = dict(DEFAULT_META.get(stem, {}))
        if os.path.exists(path[:-6] + ".meta.json"):
            meta.update(json.load(open(path[:-6] + ".meta.json")))
        over = [kv.split("=", 1)[1] for kv in primary_over if kv.split("=", 1)[0] == stem]
        if over:  # --primary m=v (repeatable; several variants of one method may be primary, e.g. M4's two sets)
            meta["primary"] = over if len(over) > 1 else over[0]
        miss = meta.get("missing") or []
        meta["missing"] = {m if isinstance(m, str) else f"{m[0]}@{int(m[1])}" for m in miss}
        variants, iss = read_scores(path, ctx.known)
        for v, n in iss["unmatched"].items():
            unmatched.append(f"{stem}:{v}: {n} rows, e.g. {iss['examples'][v]}")
        for v, n in iss["dup"].items():
            print(f"WARNING {stem}:{v}: {n} duplicate (key, outer) rows, averaged", flush=True)
        for v, n in iss["nan"].items():
            print(f"WARNING {stem}:{v}: {n} rows with a null score, dropped", flush=True)
        prim = meta.get("primary")
        for v in (prim if isinstance(prim, list) else [prim] if prim else []):
            if v not in variants:
                unmatched.append(f"{stem}: declared primary variant {v!r} has no rows")
        methods.append({"stem": stem, "meta": meta, "variants": variants})
    if unmatched:
        sys.exit("score rows whose key matches no position in data/olens/seqs.jsonl or seqs_b / seqs_bh (or bad "
                 "primary declarations):\n  " + "\n  ".join(unmatched))
    return methods


def lookup(keys, series, fallback=None):
    v = keys.map(series)
    if fallback is not None and v.isna().any():
        v = v.fillna(fallback.map(series))
    return v.astype(float).values


# ----------------------------------------------------------------------------------------------------------- (a)
def _rowcorr(A, B):
    A = A - np.nanmean(A, 1, keepdims=True)
    B = B - np.nanmean(B, 1, keepdims=True)
    den = np.sqrt(np.nansum(A * A, 1) * np.nansum(B * B, 1))
    return np.where(den > EPS, np.nansum(A * B, 1) / np.where(den > EPS, den, 1), np.nan)


def within(ctx, D, x, y, gname, const_zero=True, min_rows=4, min_valid=4):
    """Per reply: raw Spearman(x, y) and partial Spearman controlling for rank(c). Returns DataFrame by rid.
    Replies need >= min_rows rows in D and >= min_valid rows where both x and y exist (scripts/89 used 4 and 3).
    A reply whose y is constant is dropped; its partial is undefined (NaN) when y is exactly monotone in c; a
    constant x (or x exactly monotone in c) scores 0 when const_zero, else NaN. Vectorized over replies."""
    IX, rids, behs = ctx.pad(gname, D)
    pad = IX < 0
    X = np.where(pad, np.nan, np.asarray(x, float)[IX])
    Y = np.where(pad, np.nan, np.asarray(y, float)[IX])
    C = np.where(pad, np.nan, D.c.values.astype(float)[IX])
    m = np.isfinite(X) & np.isfinite(Y)
    X[~m], Y[~m], C[~m] = np.nan, np.nan, np.nan
    keep = ((~pad).sum(1) >= min_rows) & (m.sum(1) >= min_valid)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        RX, RY, RZ = (rankdata(V, axis=1, nan_policy="omit") for V in (X, Y, C))
        sx = np.nansum((RX - np.nanmean(RX, 1, keepdims=True)) ** 2, 1)
        sy = np.nansum((RY - np.nanmean(RY, 1, keepdims=True)) ** 2, 1)
        keep &= sy > EPS  # target constant in this reply: dropped for every monitor
        rxy, rxz, ryz = _rowcorr(RX, RY), _rowcorr(RX, RZ), _rowcorr(RY, RZ)
        part = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    fill = 0.0 if const_zero else np.nan
    constx = sx <= EPS
    raw = np.where(constx, fill, rxy)
    part = np.where(constx | (np.abs(rxz) > 1 - 1e-9), fill, part)
    part = np.where(np.abs(ryz) > 1 - 1e-9, np.nan, part)  # y exactly monotone in c: undefined for every monitor
    return pd.DataFrame({"raw": raw[keep], "partial": part[keep], "b": behs[keep]}, index=pd.Index(rids[keep], name="rid"))


def within_loop(ctx, D, x, y, gname, const_zero=True, min_rows=4, min_valid=4):
    """Reference (unvectorized) implementation of within(), used by --validate."""
    out = []
    c = D.c.values
    for ix in ctx.groups(gname, D):
        if len(ix) < min_rows:
            continue
        m = ~(np.isnan(x[ix]) | np.isnan(y[ix]))
        ix = ix[m]
        if len(ix) < min_valid:
            continue
        rx, ry, rz = rank(x[ix]), rank(y[ix]), rank(c[ix])
        if np.ptp(ry) == 0:
            continue
        ryz = corr(ry, rz)
        if np.ptp(rx) == 0:
            raw = part = 0.0 if const_zero else np.nan
        else:
            raw, rxz = corr(rx, ry), corr(rx, rz)
            with np.errstate(invalid="ignore", divide="ignore"):
                part = (0.0 if const_zero else np.nan) if abs(rxz) > 1 - 1e-9 else \
                    (raw - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
        if abs(ryz) > 1 - 1e-9:
            part = np.nan
        out.append((D.rid.values[ix[0]], raw, part, D.b.values[ix[0]]))
    return pd.DataFrame(out, columns=["rid", "raw", "partial", "b"]).set_index("rid")


def ols_oof(X, y, fold, Xnest=None):
    """Out-of-fold OLS with folds 0..4. Xnest[t]: training design to use when fold t is the test fold (nested)."""
    pred = np.full(len(y), np.nan)
    for t in np.unique(fold):
        Xt = Xnest[t] if Xnest is not None else X
        tr = (fold != t) & np.isfinite(Xt).all(1) & np.isfinite(y)
        te = (fold == t) & np.isfinite(X).all(1)
        if tr.sum() <= X.shape[1] + 1 or not te.any():
            continue
        w = np.linalg.lstsq(np.c_[np.ones(tr.sum()), Xt[tr]], y[tr], rcond=None)[0]
        pred[te] = np.c_[np.ones(te.sum()), X[te]] @ w
    return pred


def resid_null(ctx, D, x, y, gname, n_perm=20):
    """The method's own position-preserving null for the (a) partial: per reply, fit the score on [1, log1p c, c],
    permute its residuals within the reply (the same permutation draws for every monitor; a monitor with missing rows
    uses the order-preserving restriction of the same permutation) and recompute the partial. Mean over permutations,
    per reply."""
    c = D.c.values.astype(float)
    XN = np.full((n_perm, len(D)), np.nan)
    for ix, pm in zip(ctx.groups(gname, D), ctx.perms(gname, D, n_perm)):
        m = np.isfinite(x[ix])
        if m.sum() < 4:
            continue
        vpos = np.where(m)[0]
        j = ix[vpos]
        Z = np.c_[np.ones(len(j)), np.log1p(c[j]), c[j]]
        fit = Z @ np.linalg.lstsq(Z, x[j], rcond=None)[0]
        e = x[j] - fit
        inv = np.full(len(ix), -1)
        inv[vpos] = np.arange(len(vpos))
        for q in range(n_perm):
            XN[q, j] = fit + e[inv[pm[q][m[pm[q]]]]]
    return pd.concat([within(ctx, D, XN[q], y, gname).partial for q in range(n_perm)], axis=1).mean(axis=1)


POS_BASE = lambda c: np.c_[np.log1p(c), c, np.sqrt(c)]  # position terms of the (a) gain base


def eval_a(ctx, x, text, nested=None, oof=False, is_text=False, gain=True):
    D, y = ctx.A, ctx.A.P.values
    r = {"cov_a": int(np.isfinite(x).sum()), "exp_a": len(D)}
    if r["cov_a"] == 0:
        return r, None
    w = within(ctx, D, x, y, "A")
    w["null"] = resid_null(ctx, D, x, y, "A")
    w["adj"] = w.partial - w.null
    r["n_replies"] = int(w.adj.notna().sum())
    est, bo = ctx.boot_a.mean(w.partial)
    r.update(partial_obs=est, partial_obs_lo=ci(bo)[0], partial_obs_hi=ci(bo)[1], partial_null=w.null.mean())
    est, bo = ctx.boot_a.mean(w["adj"])
    r.update(partial=est, partial_lo=ci(bo)[0], partial_hi=ci(bo)[1], p1_partial=pz(est, bo, 0),
             p_boot_partial=pval(bo, 0))
    est, bo = ctx.boot_a.mean(w["adj"], by_beh=True)
    r.update(partial_beh=est, partial_beh_lo=ci(bo)[0], partial_beh_hi=ci(bo)[1])
    est, bo = ctx.boot_a.mean(w.raw)
    r.update(raw_within=est, raw_within_lo=ci(bo)[0], raw_within_hi=ci(bo)[1])
    r["pooled"] = spearman(x, y)
    bo = ctx.boot_a.stat("A", D.rid.values, lambda ix: spearman(x[ix], y[ix]))
    r["pooled_lo"], r["pooled_hi"] = ci(bo)
    tb = D.index[np.nan_to_num(text, nan=9) <= 0.1].values if text is not None else np.array([], int)
    r["blind_n"], r["blind"] = len(tb), spearman(x[tb], y[tb])
    r["blind_P"] = float(np.mean(y[tb])) if len(tb) else np.nan
    beh = []
    for b, g in D.groupby("b"):
        ix = g.index.values
        wb = w[w.b == b]
        beh.append({"b": b, "n_cuts": int(np.isfinite(x[ix]).sum()), "n_replies": int(wb["adj"].notna().sum()),
                    "partial": wb["adj"].mean(), "partial_obs": wb.partial.mean(), "partial_null": wb.null.mean(),
                    "raw_within": wb.raw.mean(), "pooled": spearman(x[ix], y[ix])})
    if text is not None and not is_text and gain:
        ok = np.isfinite(text) & np.isfinite(x)
        pos = POS_BASE(D.c.values.astype(float))
        Xb = np.c_[text, pos]
        Xf = np.c_[text, pos, x]
        fold = D.fold.values
        Xn = None
        if nested:
            Xn = {t: np.c_[text, pos, lookup(D.key, nested[t])] if t in nested else np.full_like(Xf, np.nan)
                  for t in np.unique(fold)}
        pb = ols_oof(np.where(ok[:, None], Xb, np.nan), y, fold)
        pf = ols_oof(np.where(ok[:, None], Xf, np.nan), y, fold,
                     {t: np.where(ok[:, None], v, np.nan) for t, v in Xn.items()} if Xn else None)
        wb_, wf_ = within(ctx, D, pb, y, "A"), within(ctx, D, pf, y, "A")
        dlt = (wf_.partial - wb_.partial).dropna()
        if len(dlt) < 20:
            return r, beh
        est, bo = ctx.boot_a.mean(dlt)
        r.update(gain=est, gain_lo=ci(bo)[0], gain_hi=ci(bo)[1], p1_gain=pz(est, bo, 0), p_boot_gain=pval(bo, 0),
                 gain_base=wb_.partial.mean(), gain_full=wf_.partial.mean(),
                 gain_pooled=spearman(pf, y) - spearman(pb, y), gain_n=int(np.isfinite(pf).sum()),
                 nested="yes" if nested else ("not nested" if oof else "fixed scores"))
    return r, beh


def eval_a_curve(ctx, x_all):
    """Raw within-reply Spearman vs Agastya's curve P including the c = 0 and c = L endpoints."""
    w = within(ctx, ctx.A_all, x_all, ctx.A_all.Pcurve.values, "A_all")
    est, bo = ctx.boot_a.mean(w.raw)
    return est, ci(bo)


# ------------------------------------------------------------------------------------------------------- (b)/(bh)
_SUBSETS = {}


def _subsets(n, n1):
    if (n, n1) not in _SUBSETS:
        S = np.zeros((0, n)) if n1 == 0 else np.array([[i in c for i in range(n)] for c in combinations(range(n), n1)], float)
        _SUBSETS[(n, n1)] = S
    return _SUBSETS[(n, n1)]


def clogit_fit(X, y, g, lam=0.1):
    """Conditional logit, strata g, exact conditional likelihood (all subsets of size n_weird), L2 penalty lam."""
    Z, case, starts = [], [], []
    for gg in np.unique(g):
        ix = np.where(g == gg)[0]
        n, n1 = len(ix), int(y[ix].sum())
        if n1 == 0 or n1 == n:
            continue
        S = _subsets(n, n1)
        starts.append(sum(len(z) for z in Z))
        Z.append(S @ X[ix])
        case.append(X[ix][y[ix]].sum(0))
    if not Z:
        return np.zeros(X.shape[1])
    Z, case, starts = np.vstack(Z), np.array(case), np.array(starts)
    gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(Z)]))

    def f(beta):
        e = Z @ beta
        mx = np.maximum.reduceat(e, starts)
        ex = np.exp(e - mx[gid])
        se = np.add.reduceat(ex, starts)
        lse = np.log(se) + mx
        p = ex / se[gid]
        ll = (case @ beta - lse).sum()
        grad = case.sum(0) - np.add.reduceat(p[:, None] * Z, starts).sum(0)
        return -ll + 0.5 * lam * beta @ beta, -grad + lam * beta

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def clogit_oof(X, y, g, fold, Xnest=None):
    """Out-of-fold linear score from the conditional logit; features z-scored on the training rows."""
    pred = np.full(len(y), np.nan)
    for t in np.unique(fold):
        Xt = Xnest[t] if Xnest is not None else X
        tr = (fold != t) & np.isfinite(Xt).all(1)
        te = (fold == t) & np.isfinite(X).all(1)
        if tr.sum() < 10 or not te.any():
            continue
        mu, sd = Xt[tr].mean(0), Xt[tr].std(0) + EPS
        beta = clogit_fit((Xt[tr] - mu) / sd, y[tr], g[tr])
        pred[te] = ((X[te] - mu) / sd) @ beta
    return pred


def clogit_apply(Xtr, ytr, gtr, Xte):
    tr = np.isfinite(Xtr).all(1)
    mu, sd = Xtr[tr].mean(0), Xtr[tr].std(0) + EPS
    beta = clogit_fit((Xtr[tr] - mu) / sd, ytr[tr], gtr[tr])
    out = np.full(len(Xte), np.nan)
    ok = np.isfinite(Xte).all(1)
    out[ok] = ((Xte[ok] - mu) / sd) @ beta
    return out


def eval_b(ctx, name, k, series, text_s, nested=None, oof=False, is_text=False):
    d = ctx.B[name][k]
    T = d["T"]
    x = lookup(T.key, series, T.own)
    r = {"set": name, "k": k, "cov": int(np.isfinite(x[T.unique.values]).sum()), "n_unique": int(T.unique.sum()),
         "cov_all": int(np.isfinite(x).sum()), "n_all": len(T)}
    if r["cov"] == 0:
        return r, None
    au = pd.Series(ctx.cut_auroc(d, x), index=d["cuts"])
    sel = d["untied"] if k == 1 else np.ones(len(au), bool)
    a = au[sel]
    boot = ctx.boot_b[name]
    est, bo = boot.mean(a)
    r.update(n_cuts=int(a.notna().sum()), auroc=est, auroc_lo=ci(bo)[0], auroc_hi=ci(bo)[1], p1_auroc=pz(est, bo, .5),
             p_boot_auroc=pval(bo, 0.5),
             ceil_pair=float(np.mean(d["ceil_pair"][sel][a.notna().values])),
             ceil_best=float(np.mean(d["ceil_best"][sel][a.notna().values])))
    est, bo = boot.mean(a, by_beh=True)
    r.update(auroc_beh=est, auroc_beh_lo=ci(bo)[0], auroc_beh_hi=ci(bo)[1])
    sharp = pd.Series(T.drop_duplicates("cut_id").set_index("cut_id").sharp)
    r["auroc_sharp"] = a[sharp.reindex(a.index).values].mean()
    r["auroc_nonsharp"] = a[~sharp.reindex(a.index).values].mean()
    bmap = T.drop_duplicates("cut_id").set_index("cut_id").b
    beh = [{"set": name, "k": k, "b": b, "n_cuts": int(g.notna().sum()), "auroc": g.mean(),
            "ceil_pair": float(np.mean(d["ceil_pair"][sel][(bmap.reindex(a.index) == b).values]))}
           for b, g in a.groupby(bmap.reindex(a.index).values)]
    if text_s is not None and not is_text:
        y, g = T.match.values, d["codes"]
        t = lookup(T.key, text_s, T.own)
        X = np.c_[t, x]
        ok = np.isfinite(X).all(1)
        if name == "b":
            fold = T.fold.values
            Xn = None
            if nested:
                Xn = {f: np.c_[t, lookup(T.key, nested[f], T.own)] if f in nested else np.full_like(X, np.nan)
                      for f in np.unique(fold)}
            pb = clogit_oof(np.where(ok[:, None], t[:, None], np.nan), y, g, fold)
            pf = clogit_oof(np.where(ok[:, None], X, np.nan), y, g, fold, Xn)
            r["nested"] = "yes" if nested else ("not nested" if oof else "fixed scores")
        else:  # (bh): fit on all of (b) at the same k, apply to (bh)
            db = ctx.B["b"][k]
            Tb = db["T"]
            tb, xb = lookup(Tb.key, text_s, Tb.own), lookup(Tb.key, series, Tb.own)
            Xb = np.c_[tb, xb]
            okb = np.isfinite(Xb).all(1)
            if not okb.any():
                return r, beh
            pb = clogit_apply(np.where(okb[:, None], tb[:, None], np.nan), Tb.match.values, db["codes"],
                              np.where(ok[:, None], t[:, None], np.nan))
            pf = clogit_apply(Xb, Tb.match.values, db["codes"], np.where(ok[:, None], X, np.nan))
            r["nested"] = "fit on (b)"
        ab = pd.Series(ctx.cut_auroc(d, pb), index=d["cuts"])[sel]
        af = pd.Series(ctx.cut_auroc(d, pf), index=d["cuts"])[sel]
        dlt = (af - ab).dropna()
        if len(dlt) < 20:  # the text judge (or the method) is missing on most cuts: no gain row
            return r, beh
        est, bo = boot.mean(dlt)
        r.update(gain=est, gain_lo=ci(bo)[0], gain_hi=ci(bo)[1], p1_gain=pz(est, bo, 0), p_boot_gain=pval(bo, 0),
                 gain_base=ab.mean(),
                 gain_full=af.mean(), gain_n_cuts=len(dlt))
    return r, beh


# ----------------------------------------------------------------------------------------------------------- (h)
def eval_h(ctx, x, text):
    H, out = ctx.H, {}
    for sp in ("discovery", "heldout"):
        ix = np.where(H.split.values == sp)[0]
        xs, ys = x[ix], H.rate.values[ix]
        out[f"{sp}_n"], out[f"{sp}_exp"] = int(np.isfinite(xs).sum()), len(ix)
        out[sp] = spearman(xs, ys)
        bo = ctx.boot_h[sp].stat(f"h_{sp}", H.pid.values[ix], lambda j: spearman(xs[j], ys[j]))
        out[f"{sp}_lo"], out[f"{sp}_hi"] = ci(bo)
    if text is not None:
        tr, te = H.split.values == "discovery", H.split.values == "heldout"
        for nm, X in (("base", np.c_[text]), ("full", np.c_[text, x])):
            ok = tr & np.isfinite(X).all(1)
            if ok.sum() < 5:
                break
            w = np.linalg.lstsq(np.c_[np.ones(ok.sum()), X[ok]], H.rate.values[ok], rcond=None)[0]
            out[f"pred_{nm}"] = np.c_[np.ones(len(X)), X] @ w
        if "pred_full" in out:
            ix = np.where(te)[0]
            pb, pf, ys = out.pop("pred_base")[ix], out.pop("pred_full")[ix], H.rate.values[ix]
            out["gain_heldout"] = spearman(pf, ys) - spearman(pb, ys)
            bo = ctx.boot_h["heldout"].stat("h_heldout", H.pid.values[ix], lambda j: spearman(pf[j], ys[j]) - spearman(pb[j], ys[j]))
            out["gain_heldout_lo"], out["gain_heldout_hi"] = ci(bo)
        out.pop("pred_base", None), out.pop("pred_full", None)
    return out


# --------------------------------------------------------------------------------------------------------- forks
def eval_forks(ctx, series, sd):
    F = ctx.F
    x = lookup(F.key, series)
    xr = lookup(F.key_raw.fillna(""), series)
    if not np.isfinite(x).any():
        return None, None
    F = F.assign(x=x, xr=xr)
    curve = F.groupby("rel").agg(chat=("x", "mean"), n=("x", "count"), raw=("xr", "mean"), n_raw=("xr", "count"))
    rng = np.random.default_rng(SEED)
    out = {"rel0": curve.chat.get(0, np.nan), "rel1": curve.chat.get(1, np.nan), "sd": sd,
           "cov": int(np.isfinite(x).sum()), "exp": len(F)}
    per = F.pivot_table(index="rid", columns="rel", values="x")
    if 0 in per and 1 in per:
        j = (per[1] - per[0]).dropna()
        bo = [rng.choice(j.values, len(j)).mean() for _ in range(1000)]
        out.update(jump=j.mean(), jump_lo=ci(bo)[0], jump_hi=ci(bo)[1], jump_n=len(j))
    pre = F[F.rel.between(-5, 0)].groupby("rid").x.mean()
    base = F[F.rel.between(-40, -21)].groupby("rid").x.mean()
    rise = (pre - base).dropna()
    if len(rise):
        bo = [rng.choice(rise.values, len(rise)).mean() for _ in range(1000)]
        out.update(rise=rise.mean(), rise_lo=ci(bo)[0], rise_hi=ci(bo)[1], rise_n=len(rise))
    for kk in ("jump", "rise"):
        if kk in out and sd and sd > 0:
            out[f"{kk}_sd"] = out[kk] / sd
    return out, curve.reset_index()


# --------------------------------------------------------------------------------------------------- noise ceiling
def noise_ceiling(ctx, n_splits=20, n_sim=10, n_sim_null=3):
    """(i) Split-half reliability of P_pooled from the resampled continuations; (ii) the primary (a) metric reached by
    a perfect monitor: truth = P_pooled, measured P re-simulated as Binomial(n, truth) / n with the actual n per cut."""
    cache = f"{M}/eval/_noise_ceiling_v2.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    samples = defaultdict(list)
    for r in map(json.loads, gzip.open(f"{A}/forks_samples_off.jsonl.gz", "rt")):
        if r["cut"].isdigit():
            samples[(r["rid"], int(r["cut"]))] += [bool(m) for m in r["match"]]
    D = ctx.A
    keys = list(zip(D.rid, D.c))
    chk = max(abs(np.mean(samples[k]) - p) for k, p in zip(keys, D.P))
    rng, res = np.random.default_rng(SEED), defaultdict(list)
    for _ in range(n_splits):
        pa, pb = np.empty(len(D)), np.empty(len(D))
        for i, k in enumerate(keys):
            s = np.array(samples[k])[rng.permutation(len(samples[k]))]
            h = len(s) // 2
            pa[i], pb[i] = s[:h].mean(), s[h:2 * h].mean()
        w = within(ctx, D, pa, pb, "A", const_zero=False)
        res["within_raw"].append(w.raw.mean())
        res["within_partial"].append(w.partial.mean())
        res["pooled"].append(spearman(pa, pb))
    out = {k: float(np.mean(v)) for k, v in res.items()}
    out.update({f"{k}_sb": 2 * v / (1 + v) for k, v in list(out.items())})
    truth, n = D.P.values, D.nP.values.astype(int)
    for i in range(n_sim):
        meas = rng.binomial(n, truth) / n
        w = within(ctx, D, truth, meas, "A")
        res["perfect_raw"].append(w.raw.mean())
        res["perfect_partial"].append(w.partial.mean())
        if i < n_sim_null:
            res["perfect_partial_corrected"].append((w.partial - resid_null(ctx, D, truth, meas, "A")).mean())
    for k in ("perfect_raw", "perfect_partial", "perfect_partial_corrected"):
        out[k] = float(np.mean(res[k]))
    out["n_splits"], out["n_sim"], out["max_abs_check_vs_p_pooled"] = n_splits, n_sim, float(chk)
    json.dump(out, open(cache, "w"), indent=1)
    return out


# ------------------------------------------------------------------------------------------------------------ main
def primaries(meta):
    prim = meta.get("primary")
    return list(prim) if isinstance(prim, list) else [prim] if prim else []


def check_primary_coverage(ctx, methods, allow_incomplete, quiet=False):
    """A primary variant must score every (a) interior cut and every (b) k = 4 unique position, except keys listed in
    its meta 'missing'. Returns the incomplete (stem, variant) pairs; stops the run unless allow_incomplete."""
    Tu = ctx.B["b"][4]["T"]
    Tu = Tu[Tu.unique]
    bad, msgs = set(), []
    for m in methods:
        for v in primaries(m["meta"]):
            s, miss = m["variants"][v][0], m["meta"]["missing"]
            ka = [k for k in ctx.A.key[~ctx.A.key.isin(s.index)] if k not in miss]
            got = Tu.key.isin(s.index) | Tu.own.isin(s.index)
            kb = [k for k in Tu.key[~got.values] if k not in miss]
            if ka or kb:
                bad.add((m["stem"], v))
                msgs.append(f"{m['stem']}:{v}: (a) {len(ka)}/{len(ctx.A)} interior cuts and (b) k=4 {len(kb)}/{len(Tu)} "
                            f"unique positions unscored and not listed in meta 'missing', e.g. {(ka + kb)[:2]}")
    if msgs:
        text = "INCOMPLETE primary variants:\n  " + "\n  ".join(msgs)
        if not allow_incomplete:
            sys.exit(text + "\n(rerun with --allow-incomplete to mark them INCOMPLETE and leave them out of Holm)")
        if not quiet:
            print("WARNING " + text, flush=True)
    return bad, msgs


def run(ctx, methods, out_dir, text_stems=("text_a", "text_b"), scores_dir=f"{M}/scores", allow_incomplete=False):
    t0 = time.time()
    incomplete, inc_msgs = check_primary_coverage(ctx, methods, allow_incomplete)
    text = pd.concat([m["variants"]["text"][0] for m in methods if m["stem"] in text_stems and "text" in m["variants"]]) \
        if any(m["stem"] in text_stems for m in methods) else None
    if text is None:  # gains need the text judge even when it is not among the evaluated methods
        parts = [read_scores(f"{scores_dir}/{s}.jsonl")[0]["text"][0] for s in text_stems if os.path.exists(f"{scores_dir}/{s}.jsonl")]
        text = pd.concat(parts) if parts else None
    if text is not None:
        text = text[~text.index.duplicated()]
    t_a = lookup(ctx.A.key, text) if text is not None else None
    t_h = lookup(ctx.H.key, text) if text is not None else None
    nc = noise_ceiling(ctx)
    TA, TB, TH, TF, BA, BB, LB, CURVES = [], [], [], [], [], [], [], []
    mons = [("position", "log1p(c)", None, {}, {"label": "B1 position, log(1+c) (computed here)"})]
    for m in methods:
        for v, (s, nest) in m["variants"].items():
            mons.append((m["stem"], v, s, nest, m["meta"]))
    for stem, v, s, nest, meta in mons:
        name = f"{stem}:{v}"
        primary = v in primaries(meta)
        status = ("primary (INCOMPLETE)" if (stem, v) in incomplete else "primary") if primary else "exploratory"
        is_text = stem in text_stems and v == "text"
        oof = bool(meta.get("oof"))
        lobo = "lobo" in v.lower()
        base = {"method": stem, "variant": v, "monitor": name, "primary": primary and (stem, v) not in incomplete,
                "status": status, "old": bool(meta.get("old"))}
        x = ctx.A.logc.values if stem == "position" else lookup(ctx.A.key, s)
        if np.isfinite(x).any():
            r, beh = eval_a(ctx, x, t_a, nest or None, oof, is_text, gain=stem != "position")
            xa = np.log1p(ctx.A_all.c.values) if stem == "position" else lookup(ctx.A_all.key, s)
            r["raw_within_curve"], (r["raw_within_curve_lo"], r["raw_within_curve_hi"]) = eval_a_curve(ctx, xa)
            r["coverage"] = f"{r['cov_a']}/{r['exp_a']}"
            TA.append({**base, **r})
            for b in beh:
                (LB if lobo else BA).append({**base, "set": "a", **b})
        if stem == "position":
            continue
        for nm in ("b", "bh"):
            for k in (4, 1):
                r, beh = eval_b(ctx, nm, k, s, text, nest or None, oof, is_text)
                if r.get("cov") and r.get("n_cuts"):
                    r["coverage"] = f"{r['cov']}/{r['n_unique']}"
                    TB.append({**base, **r})
                    for b in beh:
                        (LB if lobo else BB).append({**base, **b})
        xh = lookup(ctx.H.key, s)
        if np.isfinite(xh).any():
            r = eval_h(ctx, xh, None if is_text else t_h)
            r["coverage"] = f"{r['discovery_n']}/{r['discovery_exp']} disc, {r['heldout_n']}/{r['heldout_exp']} held-out"
            TH.append({**base, **r})
        xa_ = lookup(ctx.A.key, s)
        sd = float(np.nanstd(xa_)) if np.isfinite(xa_).sum() > 1 else np.nan
        fr, curve = eval_forks(ctx, s, sd)
        if fr:
            fr["coverage"] = f"{fr['cov']}/{fr['exp']}"
            TF.append({**base, **fr})
            CURVES.append(curve.assign(monitor=name))
        print(f"  evaluated {name}", flush=True)
    TA, TB, TH, TF = (pd.DataFrame(t) for t in (TA, TB, TH, TF))
    # Holm across the declared (complete) primaries; one-sided z-tests on the bootstrap SE
    fam = []
    for _, r in TB[(TB.primary) & (TB.set == "b") & (TB.k == 4)].iterrows() if len(TB) else []:
        fam += [("b", r.monitor, "AUROC k4 > .5", r.p1_auroc), ("b", r.monitor, "gain k4 > 0", r.get("p1_gain", np.nan))]
    for _, r in TA[TA.primary].iterrows() if len(TA) else []:
        fam += [("a", r.monitor, "partial - null > 0", r.p1_partial), ("a", r.monitor, "gain > 0", r.get("p1_gain", np.nan))]
    for m in methods:  # a primary with no (a) or no (b) row at all is a missing test, not a silent omission
        for v in primaries(m["meta"]):
            nm_ = f"{m['stem']}:{v}"
            if (m["stem"], v) not in incomplete:
                if not len(TA) or nm_ not in set(TA.monitor):
                    fam += [("a", nm_, "partial - null > 0", np.nan), ("a", nm_, "gain > 0", np.nan)]
                if not len(TB) or nm_ not in set(TB[(TB.set == "b") & (TB.k == 4)].monitor):
                    fam += [("b", nm_, "AUROC k4 > .5", np.nan), ("b", nm_, "gain k4 > 0", np.nan)]
    HF = pd.DataFrame(fam, columns=["set", "monitor", "test", "p"])
    if len(HF):
        ok = HF.p.notna()
        HF["p_holm"] = np.nan
        HF.loc[ok, "p_holm"] = holm(HF.p[ok].values)
        HF["m_family"] = int(ok.sum())
    for T, st in ((TA, "a"), (TB, "b")):
        if not len(T):
            continue
        for col, test in (("p_holm_main", {"a": "partial - null > 0", "b": "AUROC k4 > .5"}[st]),
                          ("p_holm_gain", {"a": "gain > 0", "b": "gain k4 > 0"}[st])):
            mp = {r.monitor: r.p_holm for r in HF.itertuples() if r.test == test and r.set == st} if len(HF) else {}
            sets = T["set"] if "set" in T else pd.Series(["a"] * len(T))
            ks = T["k"] if "k" in T else pd.Series([4] * len(T))
            T[col] = [mp.get(m, np.nan) if (st == "a" or (s_ == "b" and k_ == 4)) else np.nan
                      for m, s_, k_ in zip(T.monitor, sets, ks)]
    os.makedirs(out_dir, exist_ok=True)
    for nm, T in (("table_a", TA), ("table_b", TB), ("table_h", TH), ("forks", TF), ("by_behavior_a", pd.DataFrame(BA)),
                  ("by_behavior_b", pd.DataFrame(BB)), ("lobo", pd.DataFrame(LB)), ("holm", HF)):
        T.to_csv(f"{out_dir}/{nm}.csv", index=False, float_format="%.4g")
    if CURVES:
        pd.concat(CURVES).to_csv(f"{out_dir}/forks_curves.csv", index=False, float_format="%.4f")
    report(ctx, TA, TB, TH, TF, HF, nc, methods, out_dir, inc_msgs)
    print(f"done in {time.time() - t0:.0f}s: {out_dir}/report.md", flush=True)
    return TA, TB, TH, TF, HF, nc


def text_notes(ctx):
    """Counts for the report: <|im_end|> inside the first 4 continuation tokens, and judged texts (scripts/88 / 92
    convention: decode of the exact token prefix) that end in a partial UTF-8 character (U+FFFD)."""
    cache = f"{M}/eval/_text_notes.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("data/tokenizer/tokenizer.json")
    im_end, out = tok.token_to_id("<|im_end|>"), {}
    dec = lambda ids: tok.decode(ids, skip_special_tokens=False)
    for nm in ("b", "bh"):
        n_rec = n_end = n_j = n_ff = 0
        for s in map(json.loads, open(f"{M}/seqs_{nm}.jsonl")):
            P = s["pos"][0] - s["cut"]
            n_rec += 1
            n_end += im_end in s["ids"][P + s["cut"]:P + s["cut"] + 4]
            for i, k in enumerate(s["k"]):
                if s["canon"][i] == s["sid"]:
                    n_j += 1
                    n_ff += dec(s["ids"][P:s["pos"][i] + 1]).endswith("\ufffd")
        out[nm] = {"records": n_rec, "im_end_in_first4": n_end, "judged": n_j, "partial_char_end": n_ff}
    n_j = n_ff = 0
    for s in map(json.loads, open("data/olens/seqs.jsonl")):
        if s["kind"] not in ("prompt", "reply"):
            continue
        P = s["pos"][0] - s["c"][0] + 1
        for c in s["c"]:
            if c >= 1:
                n_j += 1
                n_ff += dec(s["ids"][P:P + c]).endswith("\ufffd")
    out["a"] = {"judged": n_j, "partial_char_end": n_ff}
    json.dump(out, open(cache, "w"), indent=1)
    return out


def report(ctx, TA, TB, TH, TF, HF, nc, methods, out_dir, inc_msgs=()):
    tn = text_notes(ctx)
    o = ["# Eval A monitors: harness report (scripts/55_monitor_eval.py)", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Methods: " +
         ", ".join(f"`{m['stem']}` ({len(m['variants'])} variants; primary: {m['meta'].get('primary') or 'none declared'})"
                   for m in methods) + ". Primaries come only from <method>.meta.json or --primary.", "",
         "95% CIs: 1,000 bootstrap resamples of patterns, stratified by behavior (same draws for every monitor). Family "
         "p-values: one-sided z-test on the bootstrap SE in the pre-declared direction, Holm across the family "
         "(holm.csv); every other row is exploratory. 'coverage' = positions scored / expected (for (b): unique "
         "positions; duplicates inherit the canonical score). O-lens 'old' rows use the old 0-10 thinking-on scorer. "
         "Gain columns show in parentheses the number of cuts / predictions with both the text judge and the method. "
         "Stacking: 'yes' = nested (inner out-of-fold scores supplied with 'outer'); 'not nested' = out-of-fold scores "
         "stacked without inner folds (optimistic); 'fixed scores' = no fitting on these labels; 'fit on (b)' = the (bh) "
         "combiner is fit on all of (b).", "",
         f"Text notes: <|im_end|> falls inside the first 4 continuation tokens of {tn['b']['im_end_in_first4']} (b) and "
         f"{tn['bh']['im_end_in_first4']} (bh) continuations (the reply ended; those positions read past the end of "
         f"turn). Following scripts/88, judged texts are the decode of the exact token prefix, so some end in a partial "
         f"UTF-8 character (U+FFFD): {tn['b']['partial_char_end']}/{tn['b']['judged']} (b), "
         f"{tn['bh']['partial_char_end']}/{tn['bh']['judged']} (bh), {tn['a']['partial_char_end']}/{tn['a']['judged']} "
         f"(a) reply cuts.", ""]
    if inc_msgs:
        o += ["**INCOMPLETE primaries (left out of Holm):**", ""] + [f"- {m}" for m in inc_msgs] + [""]
    if len(HF):
        o += [f"## Holm family (declared primaries, m = {int(HF.p.notna().sum())})", "",
              "| set | monitor | test (one-sided) | p (z on bootstrap SE) | p Holm |", "|---|---|---|---|---|"]
        o += [f"| {r.set} | {r.monitor} | {r.test} | " + ("missing" if pd.isna(r.p) else f"{r.p:.2e}") +
              " | " + ("-" if pd.isna(r.p_holm) else f"{r.p_holm:.2e}") + " |" for r in HF.itertuples()]
        o.append("")
    else:
        o += ["## Holm family", "", "No declared primary variants among the evaluated methods (all rows exploratory).", ""]
    g_ = lambda r, a: getattr(r, a, np.nan)
    if len(TB):
        for nm, k, note in (("b", 4, "PRIMARY: all cuts"), ("b", 1, "cuts with >= 1 untied weird-normal pair"),
                            ("bh", 4, "held-out denying-AI, all cuts"), ("bh", 1, "held-out, untied cuts")):
            T = TB[(TB.set == nm) & (TB.k == k)]
            if not len(T):
                continue
            Tn = ctx.B[nm][4]["T"]
            o += [f"## ({nm}) within-cut AUROC, k = {k} ({note})", "",
                  (f"{int(ctx.B[nm][1]['untied'].sum())} of " if k == 1 else "") +
                  f"{len(ctx.B[nm][4]['cuts'])} cuts from {Tn.pid.nunique()} prompts in {Tn.pattern.nunique()} patterns"
                  + ("; CIs rest on few clusters, read them as rough." if Tn.pattern.nunique() < 20 else ".")
                  + (" Gain: conditional logit fit out-of-fold (folds.json) on (b)." if nm == "b" else
                     " Gain: conditional logit fit on all of (b) at the same k, applied here."), "",
                  "| monitor | status | coverage | cuts | AUROC | pair tie ceiling | achievable ceiling | behavior avg | "
                  "sharp / other | gain over text | stacking | p Holm (AUROC, gain) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in T.itertuples():
                g = (f"{fmt(r.gain, r.gain_lo, r.gain_hi)} ({int(r.gain_n_cuts)})" if not pd.isna(g_(r, "gain")) else "-")
                ph = (f"{g_(r, 'p_holm_main'):.2e}, {g_(r, 'p_holm_gain'):.2e}" if r.primary and nm == "b" and k == 4 else "-")
                st = g_(r, "nested")
                o.append(f"| {r.monitor} | {r.status} | {r.coverage} | {r.n_cuts} | {r.auroc:.3f} [{r.auroc_lo:.3f}, "
                         f"{r.auroc_hi:.3f}] | {r.ceil_pair:.3f} | {r.ceil_best:.3f} | {r.auroc_beh:.3f} | "
                         f"{r.auroc_sharp:.3f} / {r.auroc_nonsharp:.3f} | {g} | {st if isinstance(st, str) else '-'} | {ph} |")
            o.append("")
    if len(TA):
        sb = nc["within_partial_sb"]
        o += ["## (a) cut points: within-reply partial Spearman vs P_pooled, controlling rank(c), interior cuts", "",
              "PRIMARY = observed partial minus the method's own residual-permutation null (score fit on [1, log1p c, c] "
              "per reply, residuals permuted within the reply, 20 shared permutations). The null corrects the negative "
              "bias of the linear rank(c) partial for any score that carries position. Gain base = [text, log1p c, c, "
              "sqrt c].", "",
              f"Ceiling: a perfect monitor (truth = P_pooled, measured P re-simulated as Binomial(n, P)/n with the actual "
              f"n per cut, {nc['n_sim']} simulations) reaches about {nc['perfect_partial']:.2f} on the primary (a) metric "
              f"({nc['perfect_partial_corrected']:.2f} after its own null); consistent with sqrt of the Spearman-Brown "
              f"split-half partial ({sb:.2f}) = {np.sqrt(sb):.2f}. The raw within-reply split-half "
              f"{nc['within_raw']:.2f} (SB {nc['within_raw_sb']:.2f}) is mostly shared position trend.", "",
              "| monitor | status | coverage | replies | partial - null | observed | null | gain over [text, pos] | "
              "stacking | raw within (interior) | raw within (curve P, endpoints) | pooled | text-blind pooled (n) | "
              "p Holm (partial, gain) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in TA.itertuples():
            g = f"{fmt(r.gain, r.gain_lo, r.gain_hi)} ({int(r.gain_n)})" if not pd.isna(g_(r, "gain")) else "-"
            ph = f"{g_(r, 'p_holm_main'):.2e}, {g_(r, 'p_holm_gain'):.2e}" if r.primary else "-"
            st = g_(r, "nested")
            o.append(f"| {r.monitor} | {r.status} | {r.coverage} | {r.n_replies} | {fmt(r.partial, r.partial_lo, r.partial_hi, 3)} | "
                     f"{fmt(r.partial_obs, d=3)} | {fmt(r.partial_null, d=3)} | {g} | {st if isinstance(st, str) else '-'} | "
                     f"{fmt(r.raw_within, r.raw_within_lo, r.raw_within_hi)} | "
                     f"{fmt(r.raw_within_curve, r.raw_within_curve_lo, r.raw_within_curve_hi)} | "
                     f"{fmt(r.pooled, r.pooled_lo, r.pooled_hi)} | {fmt(r.blind)} ({r.blind_n}) | {ph} |")
        o += ["", "Per behavior: by_behavior_a.csv. The position row is 0 by construction.", ""]
    if len(TF):
        P2 = list(ctx.cfork.values())
        o += ["## Sharp forks (46): score at rel 0 (state that picks the fork token) and rel +1 (fork token written)", "",
              f"Measured P (thinking on, Plo_on / Phi_on): {np.mean([p[1] for p in P2]):.2f} -> {np.mean([p[2] for p in P2]):.2f}. "
              "Jump = rel +1 minus rel 0 per fork; rise = mean rel -5..0 minus mean rel -40..-21; SD units = the "
              "monitor's SD over (a) interior cuts. Curves in forks_curves.csv.", "",
              "| monitor | coverage | rel 0 | rel +1 | jump | jump (SD) | rise | rise (SD) |", "|---|---|---|---|---|---|---|---|"]
        for r in TF.itertuples():
            o.append(f"| {r.monitor} | {r.coverage} | {r.rel0:.3f} | {r.rel1:.3f} | {fmt(g_(r, 'jump'), g_(r, 'jump_lo'), g_(r, 'jump_hi'))} | "
                     f"{fmt(g_(r, 'jump_sd'))} | {fmt(g_(r, 'rise'), g_(r, 'rise_lo'), g_(r, 'rise_hi'))} | {fmt(g_(r, 'rise_sd'))} |")
        o.append("")
    if len(TH):
        o += ["## (h) prompt level: last prompt token vs the prompt's thinking-off rate (64 samples)", "",
              "| monitor | coverage | discovery | held-out | gain over text, fit discovery, test held-out |", "|---|---|---|---|---|"]
        for r in TH.itertuples():
            g = fmt(r.gain_heldout, r.gain_heldout_lo, r.gain_heldout_hi) if not pd.isna(g_(r, "gain_heldout")) else "-"
            o.append(f"| {r.monitor} | {r.coverage} | {fmt(r.discovery, r.discovery_lo, r.discovery_hi)} | "
                     f"{fmt(r.heldout, r.heldout_lo, r.heldout_hi)} | {g} |")
        o.append("")
    open(f"{out_dir}/report.md", "w").write("\n".join(o) + "\n")


# ------------------------------------------------------------------------------------------- conversion (old scores)
def convert(out_dir=f"{M}/scores"):
    """Existing (a) scores -> standard rows: text_a.jsonl (runs/olens/text_judge.jsonl, p in 0-1) and olens_old.jsonl
    (runs/olens/scores.jsonl: mean over the 2 readouts of the OLD 0-10 thinking-on 'behavior' / 'weird' scores)."""
    cpos = {}
    for l in open("data/olens/seqs.jsonl"):
        s = json.loads(l)
        cpos[s["sid"]] = dict(zip(s["c"], s["pos"]))
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with open(f"{out_dir}/text_a.jsonl", "w") as f:
        for r in map(json.loads, open(f"{O}/text_judge.jsonl")):
            f.write(json.dumps({"sid": r["sid"], "pos": cpos[r["sid"]][r["c"]], "variant": "text", "score": r["p"]}) + "\n")
            n += 1
    json.dump({"label": DEFAULT_META["text_a"]["label"], "primary": None, "source": f"{O}/text_judge.jsonl", "scale": "0-1"},
              open(f"{out_dir}/text_a.meta.json", "w"), indent=1)
    acc = defaultdict(list)
    for r in map(json.loads, open(f"{O}/scores.jsonl")):
        p = r["key"].split("|")
        if p[0] != "R":
            continue
        sid, c, L = "|".join(p[1:-2]), int(p[-2]), int(p[-1])
        pos = cpos[sid][c]
        acc[(sid, pos, f"L{L}|behavior")].append(r["behavior"])
        acc[(sid, pos, f"L{L}|weird")].append(r["weird"])
    with open(f"{out_dir}/olens_old.jsonl", "w") as f:
        for (sid, pos, v), xs in sorted(acc.items()):
            f.write(json.dumps({"sid": sid, "pos": pos, "variant": v, "score": float(np.mean(xs))}) + "\n")
    json.dump({"label": DEFAULT_META["olens_old"]["label"], "primary": None, "old": True, "source": f"{O}/scores.jsonl",
               "scale": "0-10, mean of 2 readouts, scripts/87 scorer with thinking (superseded by scripts/91)"},
              open(f"{out_dir}/olens_old.meta.json", "w"), indent=1)
    print(f"wrote text_a.jsonl ({n} rows) and olens_old.jsonl ({len(acc)} rows)")


# ------------------------------------------------------------------------------------------------------ validation
def _ref_numbers():
    """Point estimates from runs/olens/analysis.md (scripts/89) to reproduce."""
    t = open(f"{O}/analysis.md").read()
    sec = {m.group(1): m.group(2) for m in re.finditer(r"^## (\d)\. .*?\n(.*?)(?=^## \d\. |\Z)", t, re.S | re.M)}
    num = lambda c: float(re.match(r"\s*([+-]?\d+\.\d+)", c).group(1)) if re.match(r"\s*[+-]?\d+\.\d+", c) else np.nan
    ref = {"s1": {}, "s4": {}, "s2": {}}
    for line in sec["1"].splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if line.startswith("| text judge") or line.startswith("| lens "):
            if len(cells) == 4:
                ref["s1"][cells[0]] = [num(c) for c in cells[1:]]
    m = re.search(r"Out-of-fold Spearman ([+-]\d\.\d+) \(text judge\) vs ([+-]\d\.\d+) .*?change ([+-]\d\.\d+)", sec["1"], re.S)
    ref["s1_gain"] = [float(g) for g in m.groups()]
    ref["s1_n"] = int(re.search(r"(\d+) cut points on (\d+)", sec["1"]).group(1))
    for line in sec["4"].splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if line.startswith("| text judge") or line.startswith("| lens "):
            ref["s4"][cells[0]] = [num(c) for c in cells[1:]]
    m = re.search(r"Spearman ([+-]\d\.\d+) -> ([+-]\d\.\d+), change ([+-]\d\.\d+)", sec["4"])
    ref["s4_gain"] = [float(g) for g in m.groups()]
    for line in sec["2"].split("Token by token")[0].splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if re.match(r"-?\d+\.\.-?\d+", cells[0] if cells else ""):
            ref["s2"][cells[0]] = [float(c) for c in cells[1:]]
    return ref


def _cmp(ours, theirs, d=2):
    if np.isnan(theirs) and np.isnan(ours):
        return "match"
    return "match" if round(ours + 1e-9, d) == round(theirs + 1e-9, d) or abs(ours - theirs) < 0.5 * 10 ** -d + 1e-9 else "DIFF"


def validate(ctx, out_dir):
    D = ctx.A
    t0, o, ok_all = time.time(), ["# Harness validation (scripts/55_monitor_eval.py --validate)", ""], []
    ref = _ref_numbers()
    sc_t = read_scores(f"{M}/scores/text_a.jsonl")[0]["text"][0]
    sc_l = read_scores(f"{M}/scores/olens_old.jsonl")[0]
    rng = np.random.default_rng(SEED)
    # ---- V1: scripts/89 section 1 with the old settings
    C = ctx.A_all.copy()
    C["text"] = lookup(C.key, sc_t)
    for L in (32, 44, 56):
        C[f"lb{L}"] = lookup(C.key, sc_l[f"L{L}|behavior"][0])
        C[f"lw{L}"] = lookup(C.key, sc_l[f"L{L}|weird"][0])
    have = C[[f"lb{L}" for L in (32, 44, 56)] + [f"lw{L}" for L in (32, 44, 56)]].notna().any(axis=1)
    C = C[have].reset_index(drop=True)
    y = C.Pcurve.values
    blind = C.index[np.nan_to_num(C.text.values, nan=9) <= 0.1].values
    o += ["## V1. Reproduce runs/olens/analysis.md (scripts/89) with the old settings", "",
          "Old settings: Agastya's curve P including the c = 0 (prompt token) and c = L endpoints, replies with >= 4 cut "
          "points, constant-score replies dropped, CIs over replies, OLS gain with scripts/89's folds (prompt ids shuffled "
          "with default_rng(1), i mod 5) measured as the change in pooled out-of-fold Spearman. Inputs are the converted "
          "files runs/monitors/scores/text_a.jsonl and olens_old.jsonl, read through the harness's own functions.", "",
          f"Cut points: {len(C)} (analysis.md: {ref['s1_n']}); replies {C.rid.nunique()}.", "",
          "| predictor | pooled (ours / 89) | within reply (ours / 89) | text-blind pooled (ours / 89) | match |",
          "|---|---|---|---|---|"]
    ctx._groups.pop("V1", None)
    rows_ref = [("text judge", "text")] + [(f"lens behavior L{L}", f"lb{L}") for L in (32, 44, 56)] + \
               [(f"lens weird L{L}", f"lw{L}") for L in (32, 44, 56)]
    for nm, col in rows_ref:
        x = C[col].values
        pooled = spearman(x, y)
        w = within(ctx, C, x, y, "V1", const_zero=False, min_valid=3)
        wv = w.raw.dropna().values
        wci = ci([rng.choice(wv, len(wv)).mean() for _ in range(1000)])
        bl = spearman(x[blind], y[blind])
        th = ref["s1"][nm]
        res = [_cmp(pooled, th[0]), _cmp(wv.mean(), th[1]), _cmp(bl, th[2])]
        ok_all += res
        o.append(f"| {nm} | {pooled:+.3f} / {th[0]:+.2f} | {wv.mean():+.3f} [{wci[0]:+.2f}, {wci[1]:+.2f}] / {th[1]:+.2f} | "
                 f"{bl:+.3f} / {th[2]:+.2f} | {'yes' if all(r == 'match' for r in res) else 'NO'} |")
    feats = [f"lb{L}" for L in (32, 44, 56)] + [f"lw{L}" for L in (32, 44, 56)]
    Cc = C.dropna(subset=["text"] + feats).reset_index(drop=True)
    pids = list(Cc.pid.unique())
    np.random.default_rng(1).shuffle(pids)
    fold = Cc.pid.map({p: i % 5 for i, p in enumerate(pids)}).values
    pb = ols_oof(Cc[["text"]].values, Cc.Pcurve.values, fold)
    pf = ols_oof(Cc[["text"] + feats].values, Cc.Pcurve.values, fold)
    g = (spearman(pb, Cc.Pcurve.values), spearman(pf, Cc.Pcurve.values))
    res = [_cmp(g[0], ref["s1_gain"][0]), _cmp(g[1], ref["s1_gain"][1]), _cmp(g[1] - g[0], ref["s1_gain"][2])]
    ok_all += res
    o += ["", f"OLS gain (old folds, {len(Cc)} cut points): out-of-fold Spearman text {g[0]:+.3f} vs text + 6 lens "
          f"{g[1]:+.3f}, change {g[1] - g[0]:+.3f}; analysis.md {ref['s1_gain'][0]:+.2f} / {ref['s1_gain'][1]:+.2f} / "
          f"{ref['s1_gain'][2]:+.2f}: {'match' if all(r == 'match' for r in res) else 'DIFF'}.", ""]
    # section 4 (h)
    H = ctx.H.copy()
    H["text"] = lookup(H.key, sc_t)
    for L in (32, 44, 56):
        H[f"lb{L}"] = lookup(H.key, sc_l[f"L{L}|behavior"][0])
    o += ["### Section 4 (prompt level, (h) positions)", "", "| predictor | all (ours / 89) | held-out (ours / 89) | match |",
          "|---|---|---|---|"]
    ho = H.split.values == "heldout"
    for nm, col in [("text judge", "text")] + [(f"lens behavior L{L}", f"lb{L}") for L in (32, 44, 56)]:
        a1, a2 = spearman(H[col].values, H.rate.values), spearman(H[col].values[ho], H.rate.values[ho])
        th = ref["s4"][nm]
        res = [_cmp(a1, th[0]), _cmp(a2, th[1])]
        ok_all += res
        o.append(f"| {nm} | {a1:+.3f} / {th[0]:+.2f} | {a2:+.3f} / {th[1]:+.2f} | {'yes' if all(r == 'match' for r in res) else 'NO'} |")
    Hc = H.dropna(subset=["text"] + [f"lb{L}" for L in (32, 44, 56)])
    tr, te = Hc[Hc.split == "discovery"], Hc[Hc.split == "heldout"]
    fp = lambda cols: np.c_[np.ones(len(te)), te[cols].values] @ np.linalg.lstsq(np.c_[np.ones(len(tr)), tr[cols].values], tr.rate.values, rcond=None)[0]
    b4, f4 = spearman(fp(["text"]), te.rate.values), spearman(fp(["text", "lb32", "lb44", "lb56"]), te.rate.values)
    res = [_cmp(b4, ref["s4_gain"][0]), _cmp(f4, ref["s4_gain"][1]), _cmp(f4 - b4, ref["s4_gain"][2])]
    ok_all += res
    o += ["", f"OLS fit on discovery, tested on held-out: {b4:+.3f} -> {f4:+.3f} (change {f4 - b4:+.3f}); analysis.md "
          f"{ref['s4_gain'][0]:+.2f} -> {ref['s4_gain'][1]:+.2f} ({ref['s4_gain'][2]:+.2f}): "
          f"{'match' if all(r == 'match' for r in res) else 'DIFF'}.", ""]
    # section 2 fork bins (text judge and lens chat / raw means)
    F = ctx.F.copy()
    F["text"] = lookup(F.key, sc_t)
    for L in (32, 44, 56):
        F[f"c{L}"] = lookup(F.key, sc_l[f"L{L}|behavior"][0])
        F[f"r{L}"] = lookup(F.key_raw.fillna(""), sc_l[f"L{L}|behavior"][0])
    F.loc[F.c == 0, ["text", "c32", "c44", "c56"]] = np.nan  # 89 drops the chat c = 0 state, keeps raw c = 0
    o += ["### Section 2 (sharp-fork bins; text judge and lens behavior, chat | raw)", "",
          "Old setting: c >= 1 only. scripts/89 meant to add the prompt-token state (c = 0) to each fork window but "
          "filtered it out (prompt rows have rid None), so its table lacks rel = -c_fork, including rel 0 for the 6 forks "
          "at c_fork = 0 (the raw re-read keeps its c = 0). The harness's fork view includes c = 0.", "",
          "| rel | n chat (ours / 89) | text (ours / 89) | L32 chat | L32 raw | L44 chat | L44 raw | L56 chat | L56 raw | match |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, th in ref["s2"].items():
        lo, hi = map(int, re.match(r"(-?\d+)\.\.(-?\d+)", lab).groups())
        d = F[F.rel.between(lo, hi)]
        dl = d[d[[f"c{L}" for L in (32, 44, 56)]].notna().any(axis=1)]  # chat rows that have lens scores (89's X)
        ours = [len(dl), dl.text.mean()] + sum([[dl[f"c{L}"].mean(), d[f"r{L}"].mean()] for L in (32, 44, 56)], [])
        res = [_cmp(a, b) for a, b in zip(ours[1:], th[1:])] + [("match" if ours[0] == th[0] else "DIFF")]
        ok_all += res
        o.append(f"| {lab} | {ours[0]} / {int(th[0])} | " + " | ".join(f"{a:.2f} / {b:.2f}" for a, b in zip(ours[1:], th[1:])) +
                 f" | {'yes' if all(r == 'match' for r in res) else 'NO'} |")
    n_ok = sum(r == "match" for r in ok_all)
    o += ["", f"**V1 result: {n_ok}/{len(ok_all)} reference numbers reproduced to the printed rounding.** CIs use the "
          "harness's own resampling (89's global rng order is not replayed), so they agree only approximately.", ""]

    # ---- V2: within-cut AUROC unit tests
    o += ["## V2. Within-cut AUROC unit tests", ""]
    from scipy.stats import mannwhitneyu
    srng = np.random.default_rng(7)
    # (i) brute force vs vectorized vs Mann-Whitney on tied integer scores
    n_c = 300
    T = pd.DataFrame([{"cut_id": f"c{i}", "match": j < 4, "key": f"c{i}|{j}"} for i in range(n_c) for j in range(8)])
    codes, cuts = pd.factorize(T.cut_id)
    slot = T.groupby("cut_id").cumcount().values
    Y = np.full((n_c, 8), -1)
    Y[codes, slot] = T.match.astype(int).values
    d = {"codes": codes, "slot": slot, "Y": Y}
    s_int = srng.integers(0, 4, len(T)).astype(float)
    au = ctx.cut_auroc(d, s_int)
    brute, mw = [], []
    for i in range(n_c):
        w_, n_ = s_int[i * 8:i * 8 + 4], s_int[i * 8 + 4:i * 8 + 8]
        brute.append(np.mean([(a > b) + 0.5 * (a == b) for a in w_ for b in n_]))
        mw.append(mannwhitneyu(w_, n_).statistic / 16)
    e1, e2 = np.abs(au - brute).max(), np.abs(au - np.array(mw)).max()
    o += [f"- Tied integer scores, {n_c} synthetic cuts (4 + 4): max |vectorized - brute-force pairs| = {e1:.1e}, "
          f"max |vectorized - scipy Mann-Whitney U / n1 n0| = {e2:.1e}: {'pass' if max(e1, e2) < 1e-12 else 'FAIL'}."]
    ok_all.append("match" if max(e1, e2) < 1e-12 else "DIFF")
    # (ii) all-distinct positions: perfect monitor = 1 = ceiling; random ~ .5
    perf = ctx.cut_auroc(d, T.match.astype(float).values + 0.0)
    rnd = [ctx.cut_auroc(d, srng.standard_normal(len(T))).mean() for _ in range(200)]
    o += [f"- No shared prefixes: perfect monitor mean AUROC {perf.mean():.4f} (ceiling 1); random monitor {np.mean(rnd):.4f} "
          f"+- {np.std(rnd):.4f} (SD over 200 draws): {'pass' if perf.mean() == 1 and abs(np.mean(rnd) - .5) < .01 else 'FAIL'}."]
    ok_all.append("match" if perf.mean() == 1 and abs(np.mean(rnd) - .5) < .01 else "DIFF")
    # (iii) the real (b)/(bh) tie structure
    o += ["", "Real tie structure ((b)/(bh) canon). Pair ceiling = 1 - .5 x (share of weird-normal pairs that share a "
          "prefix), recomputed here and compared with cuts_b.jsonl; achievable ceiling = AUROC of a monitor that scores "
          "each shared prefix by its weird fraction (the likelihood-ratio ordering, the best any monitor can do). A "
          "'perfect' monitor that knows each continuation's label but must give tied prefixes one score reaches the "
          "pair ceiling only when a cut has at most one mixed prefix group.", "",
          "| set | k | cuts | unique positions | pair ceiling (file / recomputed) | achievable | perfect monitor | random monitor (200 draws) | untied cuts |",
          "|---|---|---|---|---|---|---|---|---|"]
    for nm in ("b", "bh"):
        for k in (1, 4):
            dd = ctx.B[nm][k]
            Tk = dd["T"]
            fc = np.array([ctx.cuts[nm][c]["ceiling"][f"k{k}"] for c in dd["cuts"]])
            sel = dd["untied"] if k == 1 else np.ones(len(fc), bool)
            err = np.abs(fc - dd["ceil_pair"]).max()
            # perfect monitor: unique positions scored by their weird fraction, copied to duplicates through the canon key
            frac = Tk.groupby("key").match.mean()
            uniq = Tk[Tk.unique]
            sc = pd.Series(uniq.key.map(frac).values, index=uniq.key.values)
            perfect = ctx.cut_auroc(dd, lookup(Tk.key, sc))
            rnd = []
            for _ in range(200):
                rs = pd.Series(srng.standard_normal(len(uniq)), index=uniq.key.values)
                rnd.append(np.nanmean(ctx.cut_auroc(dd, lookup(Tk.key, rs))[sel]))
            okk = err < 1e-3 and abs(perfect[sel].mean() - dd["ceil_best"][sel].mean()) < 1e-12 and abs(np.mean(rnd) - .5) < .01
            ok_all.append("match" if okk else "DIFF")
            o.append(f"| {nm} | {k} | {sel.sum()} | {int(Tk.unique.sum())} | {fc[sel].mean():.4f} / {dd['ceil_pair'][sel].mean():.4f} "
                     f"(max diff {err:.1e}) | {dd['ceil_best'][sel].mean():.4f} | {perfect[sel].mean():.4f} | "
                     f"{np.mean(rnd):.4f} +- {np.std(rnd):.4f} | {int(dd['untied'].sum())} |")
    # (iv) conditional logit sanity
    db = ctx.B["b"][4]
    Tb = db["T"]
    yb = Tb.match.values.astype(bool)
    sig = yb + srng.standard_normal(len(Tb))  # informative
    noise = srng.standard_normal(len(Tb))
    weak = 0.3 * yb + srng.standard_normal(len(Tb))
    p1 = clogit_oof(np.c_[weak], yb, db["codes"], Tb.fold.values)
    p2 = clogit_oof(np.c_[weak, sig], yb, db["codes"], Tb.fold.values)
    p3 = clogit_oof(np.c_[weak, noise], yb, db["codes"], Tb.fold.values)
    a1, a2, a3 = (np.nanmean(ctx.cut_auroc(db, p)) for p in (p1, p2, p3))
    beta = clogit_fit(np.c_[sig, noise] / 1.0, yb, db["codes"])
    okc = a2 > a1 + .05 and abs(a3 - a1) < .02 and beta[0] > 0 and abs(beta[1]) < .2 * beta[0]
    ok_all.append("match" if okc else "DIFF")
    o += ["", f"- Conditional logit on the (b) k = 4 layout with synthetic features (weak = .3 x label + noise, signal = label "
          f"+ noise, pure noise): OOF within-cut AUROC weak {a1:.3f}, weak + signal {a2:.3f}, weak + noise {a3:.3f}; "
          f"in-sample beta (signal, noise) = ({beta[0]:+.2f}, {beta[1]:+.2f}): {'pass' if okc else 'FAIL'}.", ""]
    # (v) partial Spearman formula vs residual method
    errs = []
    for _ in range(200):
        n = srng.integers(4, 30)
        z = np.arange(n, dtype=float)
        x_, y_ = srng.standard_normal(n) + .5 * z / n, srng.standard_normal(n) + z / n
        rx, ry, rz = rank(x_), rank(y_), rank(z)
        Z = np.c_[np.ones(n), rz]
        res_ = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
        ref_p = corr(res_(rx), res_(ry))
        rxy, rxz, ryz = corr(rx, ry), corr(rx, rz), corr(ry, rz)
        errs.append(abs((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)) - ref_p))
    ok_all.append("match" if max(errs) < 1e-10 else "DIFF")
    o += [f"- Partial Spearman (closed form on ranks) vs correlation of rank residuals after regressing on rank(c), 200 random "
          f"replies: max |difference| {max(errs):.1e}: {'pass' if max(errs) < 1e-10 else 'FAIL'}."]
    vrng, worst_v = np.random.default_rng(3), 0.0
    xs_ = [vrng.standard_normal(len(ctx.A)), lookup(ctx.A.key, sc_t), np.log1p(ctx.A.c.values),
           vrng.integers(0, 3, len(ctx.A)).astype(float), np.where(vrng.random(len(ctx.A)) < .3, np.nan, vrng.standard_normal(len(ctx.A)))]
    for x_ in xs_:
        for cz in (True, False):
            for mv in (3, 4):
                a_ = within(ctx, ctx.A, x_, ctx.A.P.values, "A", const_zero=cz, min_valid=mv)
                b_ = within_loop(ctx, ctx.A, x_, ctx.A.P.values, "A", const_zero=cz, min_valid=mv)
                same = a_.index.equals(b_.index) and (a_.isna() == b_.isna()).all().all()
                worst_v = max(worst_v, np.nanmax(np.abs(np.r_[(a_.raw - b_.raw).values, (a_.partial - b_.partial).values]))
                              if same else np.inf)
    ok_all.append("match" if worst_v < 1e-12 else "DIFF")
    o += [f"- Vectorized within() vs the per-reply loop reference (noise, text judge, position, tied integers, 30% "
          f"missing; both conventions): max |difference| {worst_v:.1e}: {'pass' if worst_v < 1e-12 else 'FAIL'}.", ""]

    # ---- V3: position row, text-judge partial, noise ceiling
    o += ["## V3. Position row, text-judge partial Spearman, noise ceiling", ""]
    D = ctx.A
    rows, v3 = [], {}
    for lab, df, ycol, gname in (("Agastya's curve P, endpoints included (scripts/89 settings)", ctx.A_all, "Pcurve", "A_all"),
                                 ("curve P, interior cuts", D, "Pcurve", "A"),
                                 ("P_pooled, interior cuts (primary set)", D, "P", "A")):
        xx = np.log1p(df.c.values)
        w = within(ctx, df, xx, df[ycol].values, gname, const_zero=False)
        rows.append(f"| position log(1+c) | {lab} | {w.raw.mean():+.3f} | undefined (0 by the harness convention) | "
                    f"{w.raw.notna().sum()} |")
        v3[("pos", gname, ycol)] = w.raw.mean()
        if gname == "A_all":
            wt = within(ctx, df, lookup(df.key, sc_t), df[ycol].values, gname, const_zero=False)
            v3["text_partial_curve"] = wt.partial.mean()
            rows.append(f"| text judge | {lab} | {wt.raw.mean():+.3f} | {wt.partial.mean():+.3f} | {wt.raw.notna().sum()} |")
    for lab, cz in (("P_pooled, interior; constant-within-reply scores dropped", False),
                    ("P_pooled, interior; constant-within-reply scores count as 0 (harness convention)", True)):
        wt = within(ctx, D, lookup(D.key, sc_t), D.P.values, "A", const_zero=cz)
        est, bo = ctx.boot_a.mean(wt.partial)
        v3["text_partial_pooled"] = est
        rows.append(f"| text judge | {lab} | {wt.raw.mean():+.3f} | {fmt(est, *ci(bo), d=3)} | {wt.partial.notna().sum()} |")
        wt2 = within(ctx, D, lookup(D.key, sc_t), D.Pcurve.values, "A", const_zero=cz)
        rows.append(f"| text judge | {lab.replace('P_pooled', 'curve P')} | {wt2.raw.mean():+.3f} | {wt2.partial.mean():+.3f} | {wt2.partial.notna().sum()} |")
    n_const = sum(np.ptp(lookup(D.key, sc_t)[ix]) == 0 for ix in ctx.groups("A", D))
    nc = noise_ceiling(ctx)
    o += ["| predictor | setting | raw within-reply Spearman | partial (controlling rank c) | replies |", "|---|---|---|---|---|"] + rows
    o += ["", f"Text judge constant within {n_const} of {D.rid.nunique()} replies (interior cuts).", "",
          f"Noise ceiling, split-half of the resampled continuations at each interior cut ({nc['n_splits']} random splits; "
          f"pooled samples reproduce P_pooled to {nc['max_abs_check_vs_p_pooled']:.1e}):", "",
          "| statistic | split-half | Spearman-Brown (full sample) |", "|---|---|---|",
          f"| within-reply raw Spearman | {nc['within_raw']:.3f} | {nc['within_raw_sb']:.3f} |",
          f"| within-reply partial (controlling rank c) | {nc['within_partial']:.3f} | {nc['within_partial_sb']:.3f} |",
          f"| pooled Spearman | {nc['pooled']:.3f} | {nc['pooled_sb']:.3f} |",
          f"| perfect monitor, primary partial (truth = P_pooled, Binomial(n, P) re-measurement, {nc['n_sim']} sims) | "
          f"{nc['perfect_partial']:.3f} (after its null {nc['perfect_partial_corrected']:.3f}) | sqrt(SB partial) = "
          f"{np.sqrt(nc['within_partial_sb']):.3f} |", ""]
    chk = [abs(v3[("pos", "A_all", "Pcurve")] - .825) < .0005, abs(v3["text_partial_curve"] - .12) < .01,
           abs(nc["within_raw"] - .80) < .01, abs(nc["perfect_partial"] - .83) < .03]
    ok_all += ["match" if c_ else "DIFF" for c_ in chk]
    o += [f"Reviewer's numbers: position alone +.825 raw within reply -> {v3[('pos', 'A_all', 'Pcurve')]:+.3f} with "
          f"Agastya's curve P and the endpoints ({'match' if chk[0] else 'DIFF'}); on the primary set (P_pooled, interior "
          f"cuts) it is {v3[('pos', 'A', 'P')]:+.3f}. Text judge partial ~+.12 with c fixed -> "
          f"{v3['text_partial_curve']:+.3f} (curve P, endpoints; {'match' if chk[1] else 'DIFF'}), "
          f"{v3['text_partial_pooled']:+.3f} on the primary set before the null correction. Noise ceiling .80 -> "
          f"split-half within-reply raw Spearman {nc['within_raw']:.3f} ({'match' if chk[2] else 'DIFF'}). Perfect-monitor "
          f"ceiling on the primary (a) metric ~.83 -> {nc['perfect_partial']:.3f} ({'match' if chk[3] else 'DIFF'}).", ""]

    # ---- V4: residual-permutation null for the (a) primary
    o += ["## V4. (a) primary: residual-permutation null", "",
          "Synthetic scores x = f(c) + s x SD_reply(f) x noise carry position and no information about P beyond it; "
          "after the correction they should land near 0. 'signal' adds .5 x the standardized within-reply rank of P.", "",
          "| score | observed partial | null | observed - null |", "|---|---|---|---|"]
    vr = np.random.default_rng(5)
    cc = D.c.values.astype(float)
    G = ctx.groups("A", D)

    def synth(f, sd_mult):
        x = np.empty(len(cc))
        for ix in G:
            x[ix] = f[ix] + sd_mult * f[ix].std() * vr.standard_normal(len(ix))
        return x

    def signal(x):
        x = x.copy()
        for ix in G:
            rp = rank(D.P.values[ix])
            x[ix] += .5 * (rp - rp.mean()) / max(rp.std(), 1e-9)
        return x

    n_draw = 4
    cases = [("text judge (text_a)", [lookup(D.key, sc_t)]),
             ("pure noise", [vr.standard_normal(len(cc)) for _ in range(n_draw)])]
    for fn, f in (("log1p c", np.log1p(cc)), ("c", cc), ("sqrt c", np.sqrt(cc))):
        for sm in (.1, .3, 1):
            cases.append((f"{fn} + {sm} SD noise", [synth(f, sm) for _ in range(n_draw)]))
    pos_part = synth(np.log1p(cc), .3)
    for ix in G:  # z-score within reply
        pos_part[ix] = (pos_part[ix] - pos_part[ix].mean()) / max(pos_part[ix].std(), 1e-9)
    cases += [("position only (log1p c + .3 SD noise, z-units)", [pos_part]), ("same position + signal", [signal(pos_part)])]
    res4 = {}
    for nm, xs in cases:
        vals = []
        for x in xs:
            w = within(ctx, D, x, D.P.values, "A")
            nl = resid_null(ctx, D, x, D.P.values, "A")
            vals.append((w.partial.mean(), nl.mean(), (w.partial - nl).mean()))
        res4[nm] = tuple(np.mean(vals, 0))
        o.append(f"| {nm} ({len(xs)} draw{'s' if len(xs) > 1 else ''}) | {res4[nm][0]:+.3f} | {res4[nm][1]:+.3f} | {res4[nm][2]:+.3f} |")
    pos_names = [nm for nm, _ in cases if " SD noise" in nm and "z-units" not in nm]
    worst = max(abs(res4[nm][2]) for nm in pos_names)
    raw_worst = min(res4[nm][0] for nm in pos_names)
    sp, po = res4["same position + signal"], res4["position only (log1p c + .3 SD noise, z-units)"]
    c4 = [worst < .02, abs(res4["pure noise"][2]) < .02, abs(res4["text judge (text_a)"][1]) < .01,
          abs(sp[1] - po[1]) < .03 and sp[2] > .1]
    ok_all += ["match" if c_ else "DIFF" for c_ in c4]
    o += ["", f"Uncorrected, the position-carrying scores average as low as {raw_worst:+.3f}; corrected, the worst mean "
          f"is {worst:.3f} in absolute value ({'pass' if c4[0] else 'FAIL'}, bound .02; single draws vary by about "
          f"+-.015). Pure noise {res4['pure noise'][2]:+.3f} ({'pass' if c4[1] else 'FAIL'}). Text judge null "
          f"{res4['text judge (text_a)'][1]:+.3f}, so {res4['text judge (text_a)'][0]:+.3f} -> "
          f"{res4['text judge (text_a)'][2]:+.3f} ({'pass' if c4[2] else 'FAIL'}). Injected signal kept: adding the "
          f"signal leaves the null where the position alone puts it ({po[1]:+.3f} vs {sp[1]:+.3f}), so the corrected "
          f"value {sp[2]:+.3f} keeps the signal ({'pass' if c4[3] else 'FAIL'}).", ""]

    # ---- V5: family p-values, coverage guards
    o += ["## V5. Family p-values and coverage guards", ""]
    from scipy.stats import norm as _n
    bo = np.random.default_rng(1).normal(.05, .01, 1000)
    zp = pz(.05, bo, 0)
    c5 = [abs(zp - _n.sf(.05 / bo.std(ddof=1))) < 1e-12 and zp < 1e-5]
    o.append(f"- One-sided z on the bootstrap SE: estimate .05, SE {bo.std(ddof=1):.4f} -> p = {zp:.1e} (resolvable far "
             f"below Holm thresholds, unlike the 2/1001 floor of a 1,000-resample percentile p): "
             f"{'pass' if c5[0] else 'FAIL'}.")
    tmp = f"{out_dir}/_selftest"
    os.makedirs(tmp, exist_ok=True)
    good = [{"sid": k.split("@")[0], "pos": int(k.split("@")[1]), "variant": "v", "score": 0.1} for k in ctx.A.key[:5]]
    with open(f"{tmp}/bad.jsonl", "w") as f:
        for r in good + [{"sid": "reply|nonexistent", "pos": 3, "variant": "v", "score": 1.0}]:
            f.write(json.dumps(r) + "\n")
    try:
        load_methods(ctx, ["bad"], scores_dir=tmp)
        c5.append(False)
    except SystemExit as e:
        c5.append("nonexistent" in str(e))
    os.remove(f"{tmp}/bad.jsonl")
    with open(f"{tmp}/dup.jsonl", "w") as f:
        for r in good + good[:2]:
            f.write(json.dumps(r) + "\n")
    _, iss = read_scores(f"{tmp}/dup.jsonl", ctx.known)
    c5.append(iss["dup"].get("v") == 2)
    os.remove(f"{tmp}/dup.jsonl")
    with open(f"{tmp}/part.jsonl", "w") as f:
        for r in good:
            f.write(json.dumps(r) + "\n")
    json.dump({"primary": "v"}, open(f"{tmp}/part.meta.json", "w"))
    try:
        check_primary_coverage(ctx, load_methods(ctx, ["part"], scores_dir=tmp), False)
        c5.append(False)
    except SystemExit as e:
        c5.append("INCOMPLETE" in str(e))
    bad_, _ = check_primary_coverage(ctx, load_methods(ctx, ["part"], scores_dir=tmp), True, quiet=True)
    c5.append(("part", "v") in bad_)
    for fn_ in ("part.jsonl", "part.meta.json"):
        os.remove(f"{tmp}/{fn_}")
    os.rmdir(tmp)
    o += [f"- A score row whose key matches no position stops the run: {'pass' if c5[1] else 'FAIL'}.",
          f"- Duplicate (key, outer) rows are counted and reported: {'pass' if c5[2] else 'FAIL'}.",
          f"- A primary variant with unscored (a) / (b) positions not listed in meta 'missing' stops the run, and is "
          f"marked INCOMPLETE under --allow-incomplete: {'pass' if c5[3] and c5[4] else 'FAIL'}.", ""]
    ok_all += ["match" if c_ else "DIFF" for c_ in c5]
    n_ok = sum(r == "match" for r in ok_all)
    o.insert(2, f"**{n_ok}/{len(ok_all)} checks pass.** Runtime {time.time() - t0:.0f}s.\n")
    open(f"{out_dir}/validation.md", "w").write("\n".join(o) + "\n")
    print("\n".join(o))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="")
    ap.add_argument("--primary", action="append", default=[])
    ap.add_argument("--out", default=f"{M}/eval")
    ap.add_argument("--scores-dir", default=f"{M}/scores", help="score files to evaluate (text judge read from here too)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="incomplete primary variants are marked INCOMPLETE and left out of Holm instead of stopping")
    ap.add_argument("--convert", action="store_true", help="(re)write scores/text_a.jsonl and olens_old.jsonl first")
    args = ap.parse_args()
    if args.convert:
        convert()
    ctx = Ctx(args.boot)
    methods = load_methods(ctx, [m for m in args.methods.split(",") if m] or None, args.primary, args.scores_dir)
    run(ctx, methods, args.out, scores_dir=args.scores_dir, allow_incomplete=args.allow_incomplete)
    if args.validate:
        validate(ctx, args.out)
