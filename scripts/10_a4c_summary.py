"""10_a4c_summary.py -- tabulate the A4c recipe replication (runs/a4c_recipes/{judged.jsonl,keystone.json,settings.json}).

Per setting: n judged, coherence rate, rate of each typed behavior counted ONLY when coherent (typed-judging rule), with 95%
Wilson CIs; target behavior per recipe; the subset rate on the relevant prompt category (good_news for condolences, bad_idea for
sycophancy); keystone KL / top-1; and for each steered (neuron, dose) the max target rate over its 5 random same-layer neurons
and 3 matched-norm random directions. Writes runs/a4c_recipes/summary.md and summary.json.
Run locally: source .venv/bin/activate && python -I scripts/10_a4c_summary.py
"""
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from judge import wilson  # noqa: E402

RUN = os.path.join(ROOT, "runs", "a4c_recipes")
FIELDS = ["hostile", "sarcastic", "sycophantic", "condolences_on_good_news"]
SUBSET = {"condolences_on_good_news": "good_news", "sycophantic": "bad_idea"}
# target behavior per main setting, with the direction the recipe predicts (+ induce, - suppress)
TARGET = {"evil_pair_L14N8949+1.0_L13N7956-1.0": ("hostile", "+"), "L14N8949+1.0": ("hostile", "+"), "L13N7956-1.0": ("hostile", "+"),
          "L15N3920+0.6": ("sarcastic", "+"), "L15N3920+1.0": ("sarcastic", "+"),
          "L12N2920-0.5": ("sycophantic", "-"), "L12N2920+0.5": ("sycophantic", "+"),
          "L16N8168-0.3": ("sycophantic", "-"), "L16N8168-0.6": ("sycophantic", "-"),
          "L10N6295+0.4": ("condolences_on_good_news", "+"), "L10N6295+0.8": ("condolences_on_good_news", "+"),
          "L18N5796+0.6": ("sarcastic", "?"), "sarcastiqwen_composite": ("sarcastic", "+"),
          "L11N2557+0.5": ("sycophantic", "?"), "L11N2557+0.8": ("sycophantic", "?")}
LABEL = {"evil_pair_L14N8949+1.0_L13N7956-1.0": "evil qwen pair", "L14N8949+1.0": "evil (hostile half)", "L13N7956-1.0": "evil (second half)",
         "L15N3920+0.6": "sarcasm", "L15N3920+1.0": "sarcasm", "L12N2920-0.5": "anti-sycophancy", "L12N2920+0.5": "sycophancy (+ pole)",
         "L16N8168-0.3": "anti-sycophancy", "L16N8168-0.6": "anti-sycophancy", "L10N6295+0.4": "condolences on good news",
         "L10N6295+0.8": "condolences on good news", "L18N5796+0.6": "composite member (unlabelled)", "sarcastiqwen_composite": "sarcastiqwen composite",
         "L11N2557+0.5": "follow-up: drunk-driving lever, general yes-man?", "L11N2557+0.8": "follow-up: drunk-driving lever, general yes-man?"}


def rate(rows, field, coherent_only=True):
    ok = [r for r in rows if r["coherence"] is not None]
    k = sum(1 for r in ok if r[field] and (r["coherence"] == "coherent" or not coherent_only))
    return k, len(ok)


def fmt(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k/n:.2f} [{max(0.0, lo):.2f},{min(1.0, hi):.2f}] ({k}/{n})"


def main():
    rows = [json.loads(l) for l in open(os.path.join(RUN, "judged.jsonl"))]
    keystone = json.load(open(os.path.join(RUN, "keystone.json")))["settings"]
    settings = json.load(open(os.path.join(RUN, "settings.json")))
    by = defaultdict(list)
    for r in rows:
        by[r["setting"]].append(r)
    stats = {}
    for s in settings:
        name = s["name"]
        xs = by.get(name, [])
        if not xs:
            continue
        d = {"kind": s["kind"], "spec": s["spec"], "n": len(xs), "n_err": sum(1 for r in xs if r["coherence"] is None),
             "coherent": rate(xs, "coherence") if False else (sum(1 for r in xs if r["coherence"] == "coherent"), sum(1 for r in xs if r["coherence"] is not None)),
             "mean_distinct4": sum(r["distinct4"] for r in xs) / len(xs),
             "frac_length": sum(r["finish_reason"] == "length" for r in xs) / len(xs),
             "keystone": keystone.get(name, {})}
        for f in FIELDS:
            d[f] = rate(xs, f)
            d[f + "_any"] = rate(xs, f, coherent_only=False)
            if f in SUBSET:
                d[f + "_subset"] = rate([r for r in xs if r["category"] == SUBSET[f]], f)
        stats[name] = d
    base = stats["baseline"]
    # controls grouped by the steered (neuron, dose) they match
    ctrl = defaultdict(lambda: {"random_neuron": [], "random_direction": []})
    for name, d in stats.items():
        m = re.match(r"ctrl_(rn|rd)_.*__for_(.+)$", name)
        if m:
            ctrl[m.group(2)][d["kind"]].append((name, d))

    def ctrl_max(for_key, field, subset=False):
        out = {}
        for kind in ("random_neuron", "random_direction"):
            items = ctrl.get(for_key, {}).get(kind, [])
            key = field + ("_subset" if subset and field in SUBSET else "")
            vals = [(d[key][0] / d[key][1] if d[key][1] else 0.0, name, d[key]) for name, d in items
                    if not d["keystone"].get("broken")]  # keystone-broken controls are reported but excluded from the max
            out[kind] = max(vals) if vals else None
        return out

    lines = ["# A4c: community single-neuron recipes on Qwen3.5-4B bf16, with controls (scripts/08_a4c_recipes.py, 09_a4c_judge.py)", "",
             "Dose unit = layer median residual norm (bf16, captures/qwen3.5-4b_resid_norms_bf16.json); push on down_proj OUTPUT at every token. "
             "24 short benign prompts (6 good news, 5 bad news, 4 bad-idea sycophancy probes, 3 neutral, 6 from promptsets/induction_benign.jsonl) x 8 samples, "
             "T=1, top_p .95, top_k 20, max 200 tokens, thinking off, no system prompt, seed 0. Judge: Claude Haiku 5.5 typed rubric (scripts/09_a4c_judge.py); a behavior "
             "counts only when the sample is also judged coherent. Rates are over ALL 24 prompts unless marked 'subset' (condolences: 6 good-news prompts; sycophancy: 4 bad-idea prompts). "
             "Controls: 5 random same-layer neurons (rng(1234+L), same dose and sign) and 3 matched-norm random directions (rng(4321+L)) per steered neuron and dose, "
             "judged on 40 samples each (first 20 prompts x 2); 'rand max' = highest target rate among them. Keystone = mean KL(unsteered||steered) / top-1 agreement over the first 32 "
             "teacher-forced tokens of unsteered answers to 20 held-out benign prompts; broken = KL > 2.5 or top-1 < 0.3. CI = 95% Wilson.", ""]
    vpath = os.path.join(RUN, "verdict.md")  # hand-written verdict, kept across regenerations
    if os.path.exists(vpath):
        lines += open(vpath).read().rstrip("\n").split("\n") + [""]
    lines += ["## Main table", "",
              "| setting | label | dose | n | target | target rate (all prompts) | target rate (subset) | baseline (all / subset) | coherent | keystone KL / top-1 | rand-neuron max (all / subset) | rand-direction max (all / subset) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary_rows = []
    for s in settings:
        name = s["name"]
        if name not in stats or s["kind"].startswith("random"):
            continue
        d = stats[name]
        tgt, direction = TARGET.get(name, ("-", ""))
        dose = ", ".join(f"L{L}N{N}{x:+.1f}" for L, N, x in s["spec"]) if s["spec"] else "0"
        ks = d["keystone"]
        kstr = f"{ks.get('kl', float('nan')):.2f} / {ks.get('top1_agreement', float('nan')):.2f}" + (" BROKEN" if ks.get("broken") else "")
        if tgt == "-":
            tr = ts = bs = "-"
            rn = rd = "-"
        else:
            tr = fmt(*d[tgt]); ts = fmt(*d[tgt + "_subset"]) if tgt in SUBSET else "-"
            bs = fmt(*base[tgt]) + (" / " + fmt(*base[tgt + "_subset"]) if tgt in SUBSET else "")
            # controls: for multi-neuron settings, take the max over all member groups
            keys = [f"L{L}N{N}{x:+.1f}" for L, N, x in s["spec"]]
            def cm(kind, subset):
                best = None
                for k in keys:
                    v = ctrl_max(k, tgt, subset)[kind]
                    if v and (best is None or v[0] > best[0]):
                        best = v
                return "n/a" if best is None else f"{best[0]:.2f} ({best[2][0]}/{best[2][1]}, {best[1].split('__')[0].replace('ctrl_rn_', '').replace('ctrl_rd_', '')})"
            rn = cm("random_neuron", False) + (" / " + cm("random_neuron", True) if tgt in SUBSET else "")
            rd = cm("random_direction", False) + (" / " + cm("random_direction", True) if tgt in SUBSET else "")
        coh = fmt(*d["coherent"])
        lines.append(f"| {name} | {LABEL.get(name, s['kind'])} | {dose} | {d['n']} | {tgt} {direction} | {tr} | {ts} | {bs} | {coh} | {kstr} | {rn} | {rd} |")
        summary_rows.append({"setting": name, "label": LABEL.get(name, s["kind"]), "target": tgt, "direction": direction, **{k: v for k, v in d.items() if k != "spec"}})
    lines += ["", "## All four behaviors per main setting (coherent-only rate; 'any' includes incoherent samples)", "",
              "| setting | coherent | hostile | sarcastic | sycophantic (all / bad-idea subset) | condolences (all / good-news subset) | mean distinct-4 | frac length-stop |",
              "|---|---|---|---|---|---|---|---|"]
    for s in settings:
        name = s["name"]
        if name not in stats or s["kind"].startswith("random"):
            continue
        d = stats[name]
        lines.append(f"| {name} | {fmt(*d['coherent'])} | {fmt(*d['hostile'])} | {fmt(*d['sarcastic'])} | {fmt(*d['sycophantic'])} / {fmt(*d['sycophantic_subset'])} | "
                     f"{fmt(*d['condolences_on_good_news'])} / {fmt(*d['condolences_on_good_news_subset'])} | {d['mean_distinct4']:.2f} | {d['frac_length']:.2f} |")
    lines += ["", "## Controls (per steered neuron and dose; 40 judged samples each)", "",
              "| for | control | kind | coherent | hostile | sarcastic | sycophantic (all / subset) | condolences (all / subset) | keystone KL / top-1 |", "|---|---|---|---|---|---|---|---|---|"]
    for for_key in ctrl:
        for kind in ("random_neuron", "random_direction"):
            for name, d in ctrl[for_key][kind]:
                ks = d["keystone"]
                short = name.split("__")[0].replace("ctrl_rn_", "").replace("ctrl_rd_", "")
                lines.append(f"| {for_key} | {short} | {kind} | {fmt(*d['coherent'])} | {fmt(*d['hostile'])} | {fmt(*d['sarcastic'])} | "
                             f"{fmt(*d['sycophantic'])} / {fmt(*d['sycophantic_subset'])} | {fmt(*d['condolences_on_good_news'])} / {fmt(*d['condolences_on_good_news_subset'])} | "
                             f"{ks.get('kl', float('nan')):.2f} / {ks.get('top1_agreement', float('nan')):.2f}{' BROKEN' if ks.get('broken') else ''} |")
    open(os.path.join(RUN, "summary.md"), "w").write("\n".join(lines) + "\n")
    json.dump({"settings": stats, "main_rows": summary_rows}, open(os.path.join(RUN, "summary.json"), "w"), indent=1, default=str)
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
