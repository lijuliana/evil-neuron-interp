"""CHIVE Eval B step 94: labels from the judged generations, CHIVE-style claims, and the positive-control evidence
(findings/chive/design.md 4-5 plus the 2026-11-01 design-review updates).

  labels: runs/chive/judge/shard*.jsonl rows {mode, cond_id ("base" | edit_id), prompt_id, j, match, status, ...} as
          written by scripts/92_chive_judge.py: mode main -> label; rejudge -> agreement estimate; orig (edit replies
          j < 8 re-judged with the ORIGINAL user message) -> frame check; drift (stage-A replies, cond_id stageA) ->
          judge-drift check vs ref_match. status "http" rows are not verdicts and are skipped; status "failed" rows
          have match null and are excluded from n. (Rows without `mode` fall back to the fields frame: "original" /
          rejudge: true.) Truncation comes from the generation records, --gen-glob runs/chive/gen/shard*.jsonl
          {cond_id, prompt_id, j, finish_reason}; the judge's own `finish` field is NOT the reply's.
          -> runs/chive/labels.jsonl (one row per edit, dead zone and missing kept with raw counts)
             runs/chive/labels_summary.json (label counts, judge checks, null exclusions, agreement, seed coupling,
                                              same-run vs stage-A drift)
  claims: labels + edits + pool + data/chive/behavior_text.json -> runs/chive/claims.jsonl (TRUE/FALSE edits only,
          a superset of CHIVE's BinaryClaim row) + runs/chive/claims_meta.json
  pos:    positive-control evidence keyed by claim_id -> data/chive/evidence/pos.jsonl
          "In 8 sampled completions of the edited prompt, the behavior occurred in k." from that edit's judged replies
          j = 0..7 (deliberately shares samples with the label; its job is to show the predictor reads the evidence).

Label rule (design 4, pre-registered "independent" estimator): k/n per condition with match=null excluded; rate_base
from the same-run base replies j < 128, rate_edit from the edit's j < 32 (j < 64 for edits in the top-up spec); effect_pp = |rate_edit - rate_base| * 100
rounded to 6 decimals as in CHIVE classify_experiments (build_claims.py:267); TRUE if >= 25, FALSE if <= 8, dead zone
otherwise. Pre-registered minimum data: a condition is "insufficient" (kept on disk, not scored) if its non-null
verdicts are fewer than 87.5% of the window (edit n1 < 0.875 x n_edit, i.e. < 28 of 32 or < 56 of 64; or base n0 < 0.875 x
n_base, i.e. < 112 of 128). --estimator paired switches to the 32-vs-32 paired estimator (base and edit on shared seeds j < 32, pairs
with both verdicts non-null) in case the pilot shows strong seed coupling (labels_summary reports the coupling).
Flags kept on each label row for sensitivity analyses: frame_flag = on the same replies j < 8 judged under the
edited prompt (main) and under the original prompt (orig), the net flip count |#up - #down| is >= 2 or >= 3 replies
are discordant; labels_summary gives a judge-noise reference (main vs same-prompt re-judge discordance and the number
of frame flags that noise alone would produce). trunc_flag = the truncation rate differs between base and edit by
> 20pp (None when the data are absent). --n-base / --n-edit set the j windows used everywhere in this script (labels,
paired estimator, completeness checks, truncation); design 128 / 32. --topup-spec (the 92_chive_topup.py JSON, read
by default when present) lists the edits generated to j < 64 (design decision 2); those use the 64 window.

Claim text (design 5): CHIVE compose_statement (chive/binary_eval/build_claims.py:284-298 @ 2423952) with
THRESHOLD_PP = 15; BEHAVIOR = behavior_text.json[behavior].behavior; BASELINE = same-run base rate (128 replies);
INTERVENTION = edit blind_description (CHIVE-style blind writer) + " " + change_text, never the bank writer's own
description; question = "Why does the model {phrase} in response to this prompt?"; transcript = CHIVE
_build_transcript_for_judge (chive/pipeline/run_files.py:136-155) of [user: pool.user_message,
assistant: pool.transcript_reply], with the line "[reply truncated at the generation limit]" appended to the reply
when it was cut at the 1,024-token limit (pool.transcript_finish == "length"); every arm reads this transcript.

  python -I scripts/94_chive_claims.py            # labels, claims, pos
  python -I scripts/94_chive_claims.py labels     # or: claims | pos
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CHIVE_REF = "github.com/adamkarvonen/chive @ 2423952"

# Label gates and the threshold the claim text asserts. CHIVE (build_claims.py:90-92) uses 50 / 15 / 30; ours are
# design.md 4-5 (TRUE >= 25pp, FALSE <= 8pp, claim asserts 15pp, inside the 8-25 gap).
MOVED_PP = 25
INERT_PP = 8
THRESHOLD_PP = 15
N_BASE = 128       # same-run baseline replies per prompt, j = 0..127 (design decision 2)
N_EDIT = 32        # replies per edit, j = 0..31 (seeds shared with base j = 0..31)
N_TOPUP = 64       # replies per edit in the top-up spec, j = 0..63
N_FRAME = 8        # 92_chive_judge.py --orig_n: edit replies j < 8 are re-judged under the original prompt
FRAME_NET_MIN = 2  # frame_flag if |#up - #down| >= 2 ...
FRAME_DISC_MIN = 3  # ... or >= 3 discordant replies
TRUNC_MARK = "[reply truncated at the generation limit]"
TRUNC_FLAG_PP = 20
MIN_NONNULL_FRAC = 0.875   # pre-registered: fewer non-null verdicts than this share of the window -> insufficient
N_POS = 8          # positive-control evidence uses the edit's replies j = 0..7
POS_TEMPLATE = "In {n} sampled completions of the edited prompt, the behavior occurred in {k}."
GEN_VERSION = "cf7-weirdchat-v0"


# ---- verbatim from CHIVE chive/binary_eval/build_claims.py:284-298 @ 2423952 (THRESHOLD_PP = 15 here, 30 there) ----
def compose_statement(behavior_question: str, k0: int, n0: int, intervention: str) -> str:
    """The full claim text the answerer reads. The baseline is shown as a PERCENTAGE (no sample
    size — keeps the surface identical across n=10/n=30 source runs, so training and eval claims
    don't differ by denominator). The counterfactual rate is NOT shown (it would reveal the
    answer). The claim is UNSIGNED — it asserts only that the factor moves the rate by at least
    THRESHOLD_PP in EITHER direction, so TRUE and FALSE claims are identical in form (only the
    blind-written intervention differs). No mechanism field — the intervention carries the content."""
    base_pct = round(k0 / n0 * 100)
    return (f'BEHAVIOR (the completion is scored "yes" when): {behavior_question}\n\n'
            f"BASELINE: on the original prompt at temperature 1.0, this behavior occurs in "
            f"~{base_pct}% of completions.\n\n"
            f"INTERVENTION: {intervention}\n\n"
            f"PREDICTION: applying this intervention (everything else unchanged) would change how "
            f"often the behavior occurs by at least {THRESHOLD_PP} percentage points (in either "
            f"direction) from the ~{base_pct}% baseline.")


# ---- verbatim from CHIVE chive/pipeline/run_files.py:136-146 @ 2423952 ----
def _format_transcript_for_prompt(messages: list[dict[str, str]]) -> str:
    role_names = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    rendered_lines: list[str] = []
    for message in messages:
        role_name = role_names[message["role"]]
        rendered_lines.append(f"{role_name}: {message['content']}")
    return "\n\n".join(rendered_lines)


def build_transcript(user_message: str, reply: str, truncated: bool = False) -> str:
    """CHIVE _build_transcript_for_judge (run_files.py:149-155): the conversation plus the chosen completion.
    A reply cut at the generation limit gets TRUNC_MARK on its own line (design-review follow-up 3)."""
    if truncated:
        reply = f"{reply}\n{TRUNC_MARK}"
    return _format_transcript_for_prompt([{"role": "user", "content": user_message},
                                          {"role": "assistant", "content": reply}])


QUESTION_TEMPLATE = "Why does the model {phrase} in response to this prompt?"  # design.md 5


def intervention_text(edit: dict) -> str:
    """design 5 + review item 4: the blind writer's description plus the exact change quoted (change_text)."""
    desc = (edit.get("blind_description") or "").strip()
    change = (edit.get("change_text") or "").strip()
    assert desc, f"{edit['edit_id']}: no blind_description (the bank writer's own description is never used)"
    assert change, f"{edit['edit_id']}: empty change_text"
    if desc[-1] not in ".!?\"'”)":
        desc += "."
    return f"{desc} {change}"


# ---------------------------------------------------------------- io helpers
def read_jsonl(path: str):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str, rows) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(path + ".tmp", path)
    return n


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


# ---------------------------------------------------------------- judgments
def label_of(effect_pp: float) -> str:
    if effect_pp >= MOVED_PP:
        return "TRUE"
    if effect_pp <= INERT_PP:
        return "FALSE"
    return "dead_zone"


def _truncated(r: dict):
    if "truncated" in r and r["truncated"] is not None:
        return bool(r["truncated"])
    if r.get("finish_reason") is not None:
        return r["finish_reason"] == "length"
    return None


class Verdicts:
    """Per-stream verdict tables keyed (prompt_id, cond_id) -> {j: match}. Streams: main (labels), frame (edit replies
    re-judged under the original prompt), agree (agreement re-judges). One verdict per (prompt, cond, j) and stream:
    a non-null verdict wins over null, otherwise the first seen (sorted files, line order)."""

    def __init__(self, edit_ids: set[str], n_base: int, n_edit: int, topup_ids: set[str] = frozenset()):
        self.edit_ids = edit_ids
        self.n_base, self.n_edit, self.topup_ids = n_base, n_edit, topup_ids

    def window(self, cond: str) -> int:
        return self.n_base if cond == "base" else N_TOPUP if cond in self.topup_ids else self.n_edit
        self.tab = {s: collections.defaultdict(dict) for s in ("main", "frame", "agree")}
        self.trunc = collections.defaultdict(dict)   # (pid, cond) -> {j: bool}
        self.chk = collections.Counter()
        self.unknown = set()
        self.drift = []   # (match now, stage-A label) for the drift re-judges

    def add(self, r: dict, source: str) -> None:
        self.chk["rows"] += 1
        if r.get("status") == "http":          # transport error, retried on restart: not a verdict
            self.chk["rows_status_http_skipped"] += 1
            return
        if r.get("mode") == "drift" or r.get("cond_id") == "stageA":
            self.chk["rows_drift"] += 1
            if r.get("match") is not None and r.get("ref_match") is not None:
                self.drift.append((bool(r["match"]), bool(r["ref_match"])))
            return
        pid, cond, j, m = r["prompt_id"], r["cond_id"], int(r["j"]), r.get("match")
        if m is not None and not isinstance(m, bool):
            raise ValueError(f"{source}: match must be bool or null, got {m!r}")
        if cond != "base" and cond not in self.edit_ids:
            self.unknown.add(cond)
            self.chk["rows_unknown_cond"] += 1
            return
        if not 0 <= j < self.window(cond):
            self.chk["rows_j_out_of_range"] += 1
            return
        t = _truncated(r) if "mode" not in r else None   # 92 judge rows carry the JUDGE's finish, not the reply's
        if t is not None:
            self.trunc[(pid, cond)].setdefault(j, t)
        mode = r.get("mode")
        if mode is not None:
            stream = {"main": "main", "rejudge": "agree", "orig": "frame"}.get(mode)
            if stream is None:
                self.chk[f"rows_mode_{mode}_ignored"] += 1
                return
        else:
            stream = ("agree" if (r.get("rejudge") or r.get("agreement")) else
                      "frame" if r.get("frame") == "original" else "main")
        if stream == "frame" and cond == "base":
            self.chk["frame_rows_on_base_ignored"] += 1
            return
        self.chk[f"rows_{stream}"] += 1
        d = self.tab[stream][(pid, cond)]
        if j in d:
            self.chk[f"duplicate_rows_{stream}"] += 1
            if d[j] is None and m is not None:
                d[j] = m
            elif d[j] is not None and m is not None and d[j] != m:
                self.chk[f"duplicate_rows_disagreeing_{stream}"] += 1
            return
        d[j] = m

    def add_truncation(self, r: dict) -> None:
        t = _truncated(r)
        cond, j = r["cond_id"], int(r["j"])
        if t is not None and (cond == "base" or cond in self.edit_ids) and 0 <= j < self.window(cond):
            self.trunc[(r["prompt_id"], cond)].setdefault(j, t)

    def get(self, stream: str, pid: str, cond: str) -> dict:
        return self.tab[stream].get((pid, cond), {})


def counts(d: dict[int, bool | None], js=None) -> tuple[int, int, int, int]:
    """(k, n, n_null, n_rows) over j in js (default all): yes count and denominator over non-null verdicts."""
    vals = [v for j, v in d.items() if js is None or j in js]
    n_null = sum(v is None for v in vals)
    return sum(v is True for v in vals), len(vals) - n_null, n_null, len(vals)


def trunc_rate(d: dict[int, bool]) -> float | None:
    return (sum(d.values()) / len(d)) if d else None


def load_topup_ids(args) -> set[str]:
    if not getattr(args, "topup_spec", "") or not os.path.exists(args.topup_spec):
        return set()
    T = json.load(open(args.topup_spec))
    assert T.get("edit_j") == [N_EDIT, N_TOPUP], f"top-up spec edit_j {T.get('edit_j')} != [{N_EDIT}, {N_TOPUP}]"
    return set(T["edit_ids"])


def load_verdicts(args, edit_ids: set[str]) -> tuple[Verdicts, list[str]]:
    paths = sorted(glob.glob(args.judge_glob))
    assert paths, f"no judge files match {args.judge_glob}"
    V = Verdicts(edit_ids, args.n_base, args.n_edit, load_topup_ids(args))
    for p in paths:
        for r in read_jsonl(p):
            V.add(r, p)
    if args.gen_glob:
        for p in sorted(glob.glob(args.gen_glob)):
            for r in read_jsonl(p):
                V.add_truncation(r)
    return V, paths


def frame_noise_reference(agree: collections.Counter, rows: list[dict]) -> dict:
    """Expected frame flags from judge noise alone. The 3% same-prompt re-judges give the flip rates yes->no (p1) and
    no->yes (p2); for each edit with orig re-judges, down ~ Bin(#main yes, p1) and up ~ Bin(#main no, p2) independently,
    and P(flag) = P(|up - down| >= FRAME_NET_MIN or up + down >= FRAME_DISC_MIN). Summed over checked edits."""
    n, ny = agree.get("pairs", 0), agree.get("main_yes", 0)
    if not n:
        return {"n_pairs": 0, "discordance": None, "expected_flags": None,
                "observed_flags": sum(r["frame_flag"] is True for r in rows),
                "n_edits_checked": sum(r["frame_flag"] is not None for r in rows)}
    disc = (agree["yes_to_no"] + agree["no_to_yes"]) / n
    p1 = agree["yes_to_no"] / ny if ny else disc            # fall back to the pooled rate when a side is empty
    p2 = agree["no_to_yes"] / (n - ny) if n - ny else disc

    def binom(k, m, p):
        return math.comb(m, k) * p ** k * (1 - p) ** (m - k)

    expected = 0.0
    checked = [r for r in rows if r["frame_flag"] is not None]
    for r in checked:
        ones, zeros = r["frame_main_ones"], r["frame_n"] - r["frame_main_ones"]
        expected += sum(binom(dn, ones, p1) * binom(u, zeros, p2)
                        for dn in range(ones + 1) for u in range(zeros + 1)
                        if abs(u - dn) >= FRAME_NET_MIN or u + dn >= FRAME_DISC_MIN)
    return {"n_pairs": n, "discordance": disc, "p_yes_to_no": p1, "p_no_to_yes": p2,
            "n_edits_checked": len(checked), "observed_flags": sum(r["frame_flag"] is True for r in checked),
            "expected_flags": round(expected, 2),
            "expected_share": (expected / len(checked)) if checked else None}


# ---------------------------------------------------------------- labels
def cmd_labels(args) -> None:
    edits = list(read_jsonl(args.edits))
    pool = {r["prompt_id"]: r for r in read_jsonl(args.pool)}
    edit_ids = [e["edit_id"] for e in edits]
    assert len(set(edit_ids)) == len(edit_ids), "duplicate edit_id in edits.jsonl"
    V, paths = load_verdicts(args, set(edit_ids))
    nb = args.n_base

    base = {}
    for pid in pool:
        d = V.get("main", pid, "base")
        k0, n0, n0_null, n0_rows = counts(d)
        base[pid] = {"k0": k0, "n0": n0, "n0_null": n0_null, "n0_rows": n0_rows,
                     "rate_base": (k0 / n0) if n0 else None, "trunc_base": trunc_rate(V.trunc.get((pid, "base"), {}))}

    rows, coupling, incomplete = [], [], collections.Counter()
    for e in edits:
        pid, eid = e["prompt_id"], e["edit_id"]
        assert pid in pool, f"{eid}: prompt {pid} not in pool"
        b = base[pid]
        d0, d1 = V.get("main", pid, "base"), V.get("main", pid, eid)
        k1, n1, n1_null, n1_rows = counts(d1)
        ne = V.window(eid)
        pj = [j for j in range(min(nb, ne)) if d0.get(j) is not None and d1.get(j) is not None]  # shared seeds
        k0p, k1p = sum(d0[j] for j in pj), sum(d1[j] for j in pj)
        if len(pj) >= 8:
            a, c = [float(d0[j]) for j in pj], [float(d1[j]) for j in pj]
            r = pearson(a, c)
            if r is not None:
                coupling.append(r)
        row = {k: e.get(k) for k in ("edit_id", "prompt_id", "behavior_id", "split", "source", "kind")}
        row.update(estimator=args.estimator, n_base_window=nb, n_edit_window=ne, k0=b["k0"], n0=b["n0"], n0_null=b["n0_null"], k1=k1, n1=n1,
                   n1_null=n1_null, rate_base=b["rate_base"], rate_edit=(k1 / n1) if n1 else None,
                   k0_paired=k0p, k1_paired=k1p, n_paired=len(pj),
                   effect_signed_pp=None, effect_pp=None, label=None, is_true=None)
        if args.estimator == "independent":
            ok, r0, r1 = b["n0"] > 0 and n1 > 0, (b["k0"] / b["n0"]) if b["n0"] else None, (k1 / n1) if n1 else None
        else:
            ok, r0, r1 = len(pj) > 0, (k0p / len(pj)) if pj else None, (k1p / len(pj)) if pj else None
        if b["n0"] == 0:
            row["label"] = "no_base"
        elif n1 == 0 or not ok:
            row["label"] = "no_edit_judgments"
        else:
            signed = round((r1 - r0) * 100, 6)
            row["effect_signed_pp"] = signed
            row["effect_pp"] = abs(signed)
            if n1 < MIN_NONNULL_FRAC * ne or b["n0"] < MIN_NONNULL_FRAC * nb:
                row["label"] = "insufficient"      # raw counts and effect kept; not scored
            else:
                row["label"] = label_of(abs(signed))
                row["is_true"] = {"TRUE": True, "FALSE": False}.get(row["label"])
        # frame check: the same replies j < 8 judged under the edited prompt (main) and the original prompt (orig)
        df = V.get("frame", pid, eid)
        fj = sorted(j for j, v in df.items() if j < N_FRAME and v is not None and d1.get(j) is not None)
        up = sum((not d1[j]) and df[j] for j in fj)      # no under edited prompt, yes under original
        down = sum(d1[j] and not df[j] for j in fj)
        row["frame_n"] = len(fj)
        row["frame_up"], row["frame_down"] = up, down
        row["frame_net"], row["frame_discordant"] = abs(up - down), up + down
        row["frame_main_ones"] = sum(d1[j] for j in fj)
        row["frame_flag"] = (abs(up - down) >= FRAME_NET_MIN or up + down >= FRAME_DISC_MIN) if fj else None
        # truncation-rate change
        t1 = trunc_rate(V.trunc.get((pid, eid), {}))
        row["trunc_base"], row["trunc_edit"] = b["trunc_base"], t1
        row["trunc_change_pp"] = None if (t1 is None or b["trunc_base"] is None) else \
            round(abs(t1 - b["trunc_base"]) * 100, 6)
        row["trunc_flag"] = None if row["trunc_change_pp"] is None else row["trunc_change_pp"] > TRUNC_FLAG_PP
        if n1_rows < ne:
            incomplete[f"edit_conditions_with_<{ne}_rows"] += 1
        rows.append(row)
    for b in base.values():
        if b["n0_rows"] < nb:
            incomplete[f"prompts_with_<{nb}_base_rows"] += 1

    jd = V.drift
    judge_drift = {"pairs": len(jd), "agree": sum(a == b for a, b in jd),
                   "rate_now": (sum(a for a, _ in jd) / len(jd)) if jd else None,
                   "rate_stage_a_label": (sum(b for _, b in jd) / len(jd)) if jd else None}
    # agreement re-judges (same prompt) vs the main verdicts: judge-noise reference for the frame flag
    agree = collections.Counter()
    for (pid, cond), d in V.tab["agree"].items():
        main = V.get("main", pid, cond)
        for j, m in d.items():
            if m is None or main.get(j) is None:
                agree["pairs_with_null"] += 1
            else:
                agree["pairs"] += 1
                agree["agree"] += int(m == main[j])
                agree["main_yes"] += int(main[j])
                agree["yes_to_no"] += int(main[j] and not m)
                agree["no_to_yes"] += int((not main[j]) and m)
    frame_noise = frame_noise_reference(agree, rows)

    # drift: same-run baseline vs stage-A rate (design 3)
    drift_pairs = [(pool[p]["stage_a_rate"], base[p]["rate_base"], pool[p].get("split")) for p in pool
                   if base[p]["rate_base"] is not None and pool[p].get("stage_a_rate") is not None]

    def drift_stats(pairs):
        if not pairs:
            return {"n_prompts": 0}
        d = [b - a for a, b, _ in pairs]
        return {"n_prompts": len(pairs), "mean_diff_base_minus_stage_a": sum(d) / len(d),
                "mean_abs_diff": sum(abs(x) for x in d) / len(d),
                "pearson_r": pearson([a for a, _, _ in pairs], [b for _, b, _ in pairs]),
                "mean_stage_a": sum(a for a, _, _ in pairs) / len(pairs),
                "mean_same_run": sum(b for _, b, _ in pairs) / len(pairs)}

    drift = {"all": drift_stats(drift_pairs)}
    for sp in sorted({s for _, _, s in drift_pairs if s}):
        drift[f"split={sp}"] = drift_stats([t for t in drift_pairs if t[2] == sp])

    nulls_base = [b["n0_null"] for b in base.values()]
    nulls_edit = [r["n1_null"] for r in rows]
    hist = lambda xs: dict(collections.Counter("0" if x == 0 else "1" if x == 1 else "2" if x == 2 else  # noqa: E731
                                               "3-5" if x <= 5 else "6+" for x in xs))
    lab = collections.Counter(r["label"] for r in rows)
    chk = dict(V.chk)
    chk["unknown_cond_ids"] = sorted(V.unknown)[:20]
    summary = {
        "rule": {"estimator": args.estimator, "j_windows": {"base": nb, "edit": args.n_edit, "edit_topup": N_TOPUP},
                 "topup_spec": args.topup_spec if V.topup_ids else None, "n_topup_edits": len(V.topup_ids),
                 "TRUE_if_effect_pp_ge": MOVED_PP, "FALSE_if_effect_pp_le": INERT_PP, "null_verdicts": "excluded from n",
                 "insufficient_if": f"n_edit_nonnull < {MIN_NONNULL_FRAC} x n_edit_window or n_base_nonnull < "
                                    f"{MIN_NONNULL_FRAC} x {nb}",
                 "frame_flag": f"same replies j<{N_FRAME}: |up-down| >= {FRAME_NET_MIN} or up+down >= {FRAME_DISC_MIN}",
                 "trunc_flag_if_change_pp_gt": TRUNC_FLAG_PP},
        "n_edits": len(rows), "labels": dict(lab), "judge_files": paths, "judge_checks": chk,
        "null_exclusions": {"base_total": sum(nulls_base), "edit_total": sum(nulls_edit),
                            "base_conditions_hist": hist(nulls_base), "edit_conditions_hist": hist(nulls_edit),
                            "max_in_one_base": max(nulls_base, default=0), "max_in_one_edit": max(nulls_edit, default=0)},
        "agreement_rejudge": dict(agree, rate=(agree["agree"] / agree["pairs"]) if agree["pairs"] else None),
        "judge_drift_vs_stage_a_labels": judge_drift,
        "frame_noise_reference": frame_noise,
        "flags": {"frame_flag": dict(collections.Counter(str(r["frame_flag"]) for r in rows)),
                  "trunc_flag": dict(collections.Counter(str(r["trunc_flag"]) for r in rows))},
        "seed_coupling": {"what": "per-edit Pearson r between base and edit verdicts on shared seeds j<32 (edits with "
                                  ">= 8 paired verdicts and variation in both)",
                          "n_edits": len(coupling), "mean_r": (sum(coupling) / len(coupling)) if coupling else None},
        "incomplete": dict(incomplete), "drift_same_run_vs_stage_a": drift, "base": base,
    }
    n = write_jsonl(args.labels, rows)
    with open(args.labels_summary, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[labels] {n} edits ({args.estimator} estimator) -> {args.labels}: {dict(lab)}")
    print(f"[labels] judge checks {json.dumps({k: v for k, v in chk.items() if k != 'unknown_cond_ids'})}")
    print(f"[labels] nulls base {sum(nulls_base)} / edit {sum(nulls_edit)}; flags {json.dumps(summary['flags'])}; "
          f"seed coupling mean r {summary['seed_coupling']['mean_r']}")
    print(f"[labels] frame flags {frame_noise['observed_flags']} of {frame_noise['n_edits_checked']} checked; "
          f"judge noise alone (discordance {frame_noise['discordance']}) would give ~{frame_noise['expected_flags']}")
    if incomplete:
        print(f"[labels] WARNING incomplete conditions: {dict(incomplete)}")
    print(f"[labels] drift (same-run base - stage A): {json.dumps(drift['all'])}")


# ---------------------------------------------------------------- claims
def load_behavior_text(path: str) -> dict:
    d = json.load(open(path))
    return {k: v for k, v in d.items() if not k.startswith("_")}


def cmd_claims(args) -> None:
    labels = list(read_jsonl(args.labels))
    edits = {e["edit_id"]: e for e in read_jsonl(args.edits)}
    pool = {r["prompt_id"]: r for r in read_jsonl(args.pool)}
    btext = load_behavior_text(args.behavior_text)
    missing = sorted({p["behavior_id"] for p in pool.values()} - set(btext))
    assert not missing, f"behavior_text.json lacks {missing}"
    for b, v in btext.items():
        assert v.get("behavior") and v.get("question_phrase"), f"behavior_text.json[{b}] incomplete"
    if any(p.get("pattern_id") is None for p in pool.values()):
        print("[claims] WARNING some pool rows lack pattern_id (96 clusters the bootstrap by pattern_id)")

    claims = []
    per_prompt = collections.Counter()
    n_no_finish = 0
    windows = {(r.get("n_base_window"), r.get("n_edit_window")) for r in labels}
    assert len({w[0] for w in windows}) <= 1 and {w[1] for w in windows} <= {N_EDIT, N_TOPUP}, \
        f"labels.jsonl mixes j windows {windows}"
    for lab in sorted(labels, key=lambda r: (r["prompt_id"], r["edit_id"])):
        if lab["is_true"] is None:
            continue
        e, p = edits[lab["edit_id"]], pool[lab["prompt_id"]]
        assert e["prompt_id"] == p["prompt_id"] and e["behavior_id"] == p["behavior_id"], lab["edit_id"]
        bt = btext[p["behavior_id"]]
        reply = p["transcript_reply"]
        assert isinstance(reply, str) and reply.strip(), f"{p['prompt_id']}: empty transcript_reply"
        intervention = intervention_text(e)
        k0, n0 = lab["k0"], lab["n0"]
        truncated = p.get("transcript_finish") == "length"
        if "transcript_finish" not in p:
            n_no_finish += 1
        claim = {
            # CHIVE BinaryClaim fields (run_binary_eval.py:48-63)
            "dataset": "weirdchat", "source_run": "chive_evalB_qwen3.6-27b", "gen_version": GEN_VERSION,
            "prompt_id": p["prompt_id"], "claim_index": per_prompt[p["prompt_id"]], "claim_id": e["edit_id"],
            "verification_score": 1.0,
            "question": QUESTION_TEMPLATE.format(phrase=bt["question_phrase"]),
            "transcript": build_transcript(p["user_message"], reply, truncated),
            "statement": compose_statement(bt["behavior"], k0, n0, intervention),
            "is_true": lab["is_true"], "claim_type": "supported" if lab["is_true"] else "inert",
            "messages": [{"role": "user", "content": p["user_message"]}], "chosen_completion": reply,
            # ours
            "edit_id": e["edit_id"], "behavior_id": p["behavior_id"], "split": p.get("split"),
            "pattern_id": p.get("pattern_id"), "source": e.get("source"), "kind": e.get("kind"),
            "expectation": e.get("expectation"), "behavior_question": bt["behavior"], "intervention": intervention,
            "k_baseline": k0, "n_baseline": n0, "k_counterfactual": lab["k1"], "n_counterfactual": lab["n1"],
            "effect_pp": lab["effect_pp"], "effect_signed_pp": lab["effect_signed_pp"], "estimator": lab["estimator"],
            "base_pct": round(k0 / n0 * 100), "threshold_pp": THRESHOLD_PP, "transcript_truncated": truncated,
            "n_base_window": lab.get("n_base_window"), "n_edit_window": lab.get("n_edit_window"),
            "frame_flag": lab.get("frame_flag"), "trunc_flag": lab.get("trunc_flag"),
            "transcript_i": p.get("transcript_i"), "fork_rid": p.get("fork_rid"),
        }
        per_prompt[p["prompt_id"]] += 1
        claims.append(claim)

    if n_no_finish:
        print(f"[claims] WARNING {n_no_finish} claims' pool rows lack transcript_finish (no truncation marker)")
    n = write_jsonl(args.claims, claims)
    n_true = sum(c["is_true"] for c in claims)
    tab = {}
    for key in ("split", "source", "kind", "behavior_id"):
        t = collections.defaultdict(collections.Counter)
        for c in claims:
            t[str(c[key])]["TRUE" if c["is_true"] else "FALSE"] += 1
        tab[key] = {k: dict(v) for k, v in sorted(t.items())}
    meta = {
        "chive_ref": CHIVE_REF, "gen_version": GEN_VERSION, "threshold_pp": THRESHOLD_PP,
        "moved_pp": MOVED_PP, "inert_pp": INERT_PP, "estimator": labels[0]["estimator"] if labels else None,
        "j_windows": sorted(windows)[0] if windows else None, "transcript_truncation_mark": TRUNC_MARK,
        "n_claims_transcript_truncated": sum(c["transcript_truncated"] for c in claims),
        "n_claims": n, "n_true": n_true, "true_share": (n_true / n) if n else None,
        "n_prompts": len(per_prompt), "counts": tab,
        "inputs_sha256": {k: sha256_file(v) for k, v in (("labels", args.labels), ("edits", args.edits),
                                                         ("pool", args.pool), ("behavior_text", args.behavior_text))},
        "claims_sha256": sha256_file(args.claims),
    }
    with open(args.claims_meta, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[claims] {n} claims ({n_true} TRUE / {n - n_true} FALSE, share {meta['true_share']}) "
          f"over {len(per_prompt)} prompts -> {args.claims}")


# ---------------------------------------------------------------- positive-control evidence
def cmd_pos(args) -> None:
    edits = list(read_jsonl(args.edits))
    V, _ = load_verdicts(args, {e["edit_id"] for e in edits})
    rows, short = [], 0
    for e in edits:
        k, n, _, _ = counts(V.get("main", e["prompt_id"], e["edit_id"]), js=set(range(N_POS)))
        if n == 0:
            continue
        short += n < N_POS
        rows.append({"claim_id": e["edit_id"], "prompt_id": e["prompt_id"], "k": k, "n": n,
                     "block": POS_TEMPLATE.format(n=n, k=k)})
    w = write_jsonl(args.pos_out, rows)
    print(f"[pos] {w} positive-control blocks ({short} with < {N_POS} non-null verdicts) -> {args.pos_out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", default="all", choices=["all", "labels", "claims", "pos"])
    ap.add_argument("--judge-glob", default=os.path.join(REPO, "runs/chive/judge/shard*.jsonl"))
    ap.add_argument("--gen-glob", default=os.path.join(REPO, "runs/chive/gen/shard*.jsonl"),
                    help="generation records {cond_id, prompt_id, j, finish_reason} for truncation rates")
    ap.add_argument("--n-base", type=int, default=N_BASE, help="base replies used, j < n (design: 128)")
    ap.add_argument("--topup-spec", default=os.path.join(REPO, "data/chive/topup/t1.json"),
                    help="92_chive_topup.py spec; its edit_ids use the 64-reply window (empty string: none)")
    ap.add_argument("--n-edit", type=int, default=N_EDIT, help="edit replies used, j < n (design: 32)")
    ap.add_argument("--estimator", default="independent", choices=["independent", "paired"],
                    help="independent (pre-registered): base j<64 vs edit j<32; paired: shared seeds j<32")
    ap.add_argument("--edits", default=os.path.join(REPO, "data/chive/edits.jsonl"))
    ap.add_argument("--pool", default=os.path.join(REPO, "data/chive/pool.jsonl"))
    ap.add_argument("--behavior-text", default=os.path.join(REPO, "data/chive/behavior_text.json"))
    ap.add_argument("--labels", default=os.path.join(REPO, "runs/chive/labels.jsonl"))
    ap.add_argument("--labels-summary", default=os.path.join(REPO, "runs/chive/labels_summary.json"))
    ap.add_argument("--claims", default=os.path.join(REPO, "runs/chive/claims.jsonl"))
    ap.add_argument("--claims-meta", default=os.path.join(REPO, "runs/chive/claims_meta.json"))
    ap.add_argument("--pos-out", default=os.path.join(REPO, "data/chive/evidence/pos.jsonl"))
    args = ap.parse_args()
    assert args.n_base >= args.n_edit >= N_POS, "need n_base >= n_edit >= 8"
    if args.step in ("all", "labels"):
        cmd_labels(args)
    if args.step in ("all", "claims"):
        cmd_claims(args)
    if args.step in ("all", "pos"):
        cmd_pos(args)


if __name__ == "__main__":
    sys.exit(main())
