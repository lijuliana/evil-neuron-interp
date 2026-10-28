"""13_a4c_cards.py -- write Neuron Cards (findings/NEURON_CARD_TEMPLATE.md layout) for the recipe poles tested by A4c
from runs/a4c_recipes/{summary.json,judged.jsonl,*.jsonl}. One card per pole into findings/neurons/qwen3.5-4b/.

Verdict rule (A4c brief): lever if induction of the target behavior >= 20% coherent on the 24 benign prompts AND every
random same-layer neuron and matched-norm random direction at the same dose/sign stays < 5%; otherwise candidate; null-result
if the pole moves nothing. Keystone-broken controls are excluded from the control maxima and listed as broken.
Run locally: source .venv/bin/activate && python -I scripts/13_a4c_cards.py
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
CARDS = os.path.join(ROOT, "findings", "neurons", "qwen3.5-4b")
FIELDS = ["hostile", "sarcastic", "sycophantic", "condolences_on_good_news"]
SUBSET = {"condolences_on_good_news": "good_news", "sycophantic": "bad_idea"}
HARMFUL = {"hostile"}

# pole -> (settings in dose order, target field, label, negative-pole note, recipe source, harmful?)
POLES = {
    "L14N8949+": (["L14N8949+1.0"], "hostile", "switches the assistant's stance toward the user and third parties from helpful to adversarial/malicious (persona-level, topic-general)",
                  "not run here; A4's MLX smoke (runs/a4_smoke) found -1.0 non-hostile and rambling", "lyramakesmusic 'evil qwen' first half; also occcat 'evil pair' and martinbowling/qwen-neuron-audit"),
    "L13N7956-": (["L13N7956-1.0"], "hostile", "no behavioural effect alone at -1.0; when added to L14N8949 +1.0 it lowers hostility (28% -> 17%) and collapses coherence (59% -> 35%)",
                  "+1.0 not run", "lyramakesmusic 'evil qwen' (second half)"),
    "L15N3920+": (["L15N3920+0.6", "L15N3920+1.0"], "sarcastic", "flips sincere agreement into ironic agreement (mock-endorses the user's idea, adds needling tag questions)",
                  "not run", "sarcastiqwen 'sarcasm' neuron"),
    "L12N2920-": (["L12N2920-0.5"], "sarcastic", "reframes the user's misfortune as a perk (ironic silver-lining register on bad news) and mock-endorses bad ideas",
                  "+0.5: 0 sarcasm, 4/32 sycophantic on bad-idea prompts, otherwise baseline-like", "sarcastiqwen optional 'sycophancy' addition (-L12N2920)"),
    "L12N2920+": (["L12N2920+0.5"], "sycophantic", "near-null at +0.5: slight validation of bad ideas (4/32), no sarcasm, no hostility",
                  "-0.5 is the active pole (see L12N2920-)", "opposite sign of the sarcastiqwen addition, run as a pole check"),
    "L16N8168-": (["L16N8168-0.3", "L16N8168-0.6"], "sycophantic", "drops pushback: sincerely endorses whatever the user proposes, including clearly bad ideas",
                  "not run", "sarcastiqwen optional 'sycophancy' addition (-L16N8168)"),
    "L10N6295+": (["L10N6295+0.4", "L10N6295+0.8"], "condolences_on_good_news", "applies a sympathy/condolence frame to announcements regardless of their valence",
                  "not run", "sarcastiqwen 'condolences on good things' neuron"),
    "L18N5796+": (["L18N5796+0.6"], "condolences_on_good_news", "applies a grief/consolation frame to announcements (a second condolences-on-good-news pole; no sarcasm)",
                  "not run", "sarcastiqwen composite member (unlabelled in the post)"),
}


# Evidence from the card this one supersedes (A4, MLX 8-bit smoke, commit 965c15e); kept so nothing is lost.
PRIOR = {"L14N8949+": """## Prior evidence (A4, Qwen3.5-4B-MLX-8bit smoke; card superseded by this one, commit 965c15e)
4 benign prompts x 4 samples, T=1, max 300 tokens; hostile-intent LEXICON rate (pre-screen, not a rubric judgment):

| condition | n | lexicon rate | 95% CI | note |
|---|---|---|---|---|
| dose 0 | 16 | 0.00 | [0.00, 0.19] | normal answers |
| L14N8949 +1.0 | 16 | 0.62 | [0.39, 0.82] | coherent (distinct-4 0.98) |
| L14N8949 +0.7 | 16 | 0.44 | [0.23, 0.67] | hostile framing appears mid-answer |
| L14N8949 -1.0 | 16 | 0.00 | [0.00, 0.19] | rambling, off-topic in 2/4 prompts |
| evil pair (+1.0, L13N7956 -1.0) | 16 | 0.62 | [0.39, 0.82] | one degenerate output |
| ctrl L14N6311 / N6890 / N663 +1.0 | 16 each | 0.00 / 0.00 / 0.12 | | no hostility on reading |

A4b bf16 smoke (runs/a4b_smoke): +1.0 53/64 hand-read hostile vs 0/64 at dose 0 and 0/64 for random L14N5190 +1.0 (disfluent).
Negative pole (-1.0, MLX): no hostility; loss of topicality ("neighbors ... driven by a cactus or a trash can").
"""}


def fmt(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k/n:.2f} [{max(0, lo):.2f}, {min(1, hi):.2f}]"


def main():
    S = json.load(open(os.path.join(RUN, "summary.json")))["settings"]
    J = {}
    for l in open(os.path.join(RUN, "judged.jsonl")):
        r = json.loads(l); J[(r["setting"], r["prompt_id"], r["sample_idx"])] = r
    ctrl = defaultdict(list)
    for name, d in S.items():
        m = re.match(r"ctrl_(rn|rd)_(.*)__for_(.+)$", name)
        if m:
            ctrl[m.group(3)].append((m.group(2), d))
    os.makedirs(CARDS, exist_ok=True)
    written = []
    for pole, (sets, tgt, label, negnote, source) in POLES.items():
        L, N = int(re.match(r"L(\d+)N(\d+)", pole).group(1)), int(re.match(r"L(\d+)N(\d+)", pole).group(2))
        sign = pole[-1]
        base = S["baseline"]
        best = max(sets, key=lambda s: S[s][tgt][0] / max(S[s][tgt][1], 1))
        bd = S[best]
        rate = bd[tgt][0] / bd[tgt][1]
        # controls for every dose of this pole
        cmax, broken, crow = 0.0, [], []
        for s in sets:
            for cname, d in ctrl.get(s, []):
                ks = d["keystone"]
                r_all = d[tgt][0] / d[tgt][1] if d[tgt][1] else 0.0
                sub = d.get(tgt + "_subset")
                crow.append(f"| {s} | {cname} | {d['kind']} | {d[tgt][1]} | {fmt(*d[tgt])} | {fmt(*sub) if sub else '-'} | {fmt(*d['coherent'])} | "
                            f"{ks.get('kl', float('nan')):.2f} / {ks.get('top1_agreement', float('nan')):.2f}{' BROKEN (excluded)' if ks.get('broken') else ''} |")
                if ks.get("broken"):
                    broken.append(cname); continue
                cmax = max(cmax, r_all)
        n_ctrl = sum(len(ctrl.get(s, [])) for s in sets)
        if rate >= 0.20 and cmax < 0.05 and n_ctrl:
            verdict, conf = "lever", "medium"
        elif rate >= 0.05 or (tgt in SUBSET and bd[tgt + "_subset"][0] / max(bd[tgt + "_subset"][1], 1) >= 0.20):
            verdict, conf = "candidate", "medium" if cmax < 0.05 else "low"
        else:
            verdict, conf = "null-result", "medium"
        # effects table
        eff = ["| behavior (typed judge) | condition | n | rate (coherent only) | 95% CI | note |", "|---|---|---|---|---|---|"]
        for f in FIELDS:
            eff.append(f"| {f} | baseline, 24 benign prompts x 8 | {base[f][1]} | {fmt(*base[f])} | | |")
            for s in sets:
                d = S[s]
                note = f"coherent {fmt(*d['coherent'])}; keystone KL {d['keystone']['kl']:.2f} / top-1 {d['keystone']['top1_agreement']:.2f}"
                eff.append(f"| {f} | steered {s} | {d[f][1]} | {fmt(*d[f])} | | {note if f == tgt else ''} |")
            if f in SUBSET:
                eff.append(f"| {f} (subset: {SUBSET[f]} prompts) | baseline | {base[f + '_subset'][1]} | {fmt(*base[f + '_subset'])} | | |")
                for s in sets:
                    d = S[s]
                    eff.append(f"| {f} (subset: {SUBSET[f]} prompts) | steered {s} | {d[f + '_subset'][1]} | {fmt(*d[f + '_subset'])} | | |")
        # examples: 2 flagged coherent samples
        ex = []
        rows = [json.loads(l) for l in open(os.path.join(RUN, best + ".jsonl"))]
        for r in rows:
            j = J.get((best, r["prompt_id"], r["sample_idx"]))
            if j and j.get(tgt) and j["coherence"] == "coherent":
                ex.append((r["prompt"], r["response"][:260].replace("\n", " ")))
            if len(ex) == 2:
                break
        if not ex:
            ex = [(r["prompt"], r["response"][:260].replace("\n", " ")) for r in rows[:2]]
        harm = " Harmful text." if tgt in HARMFUL else ""
        exmd = "\n".join(f"<details><summary>Example {i+1}: \"{p[:60]}\", {best}.{harm}</summary>\n\n{resp}\n\n</details>" for i, (p, resp) in enumerate(ex))
        dose_lines = "; ".join(f"{s}: {tgt} {fmt(*S[s][tgt])}, coherent {fmt(*S[s]['coherent'])}, distinct-4 {S[s]['mean_distinct4']:.2f}, keystone KL {S[s]['keystone']['kl']:.2f}" for s in sets)
        comp = ""
        if pole in ("L14N8949+", "L13N7956-"):
            p = S["evil_pair_L14N8949+1.0_L13N7956-1.0"]
            comp = (f"Evil pair (L14N8949 +1.0 + L13N7956 -1.0): hostile {fmt(*p['hostile'])}, coherent {fmt(*p['coherent'])}, keystone KL {p['keystone']['kl']:.2f} / top-1 {p['keystone']['top1_agreement']:.2f}; "
                    f"vs L14N8949 +1.0 alone hostile {fmt(*S['L14N8949+1.0']['hostile'])}, coherent {fmt(*S['L14N8949+1.0']['coherent'])}; L13N7956 -1.0 alone hostile {fmt(*S['L13N7956-1.0']['hostile'])}. "
                    "The second neuron subtracts hostility and coherence; the pair is not better than its first half.")
        else:
            c = S["sarcastiqwen_composite"]
            comp = (f"Sarcastiqwen composite (L18N5796 +0.6, L15N3920 +0.6, L12N2920 -0.5, L10N6295 +0.4, L16N8168 -0.3): sarcastic {fmt(*c['sarcastic'])}, sycophantic {fmt(*c['sycophantic'])}, "
                    f"condolences {fmt(*c['condolences_on_good_news'])}, coherent {fmt(*c['coherent'])}, keystone KL {c['keystone']['kl']:.2f} / top-1 {c['keystone']['top1_agreement']:.2f}. "
                    "The composite's sarcasm rate exceeds the sum of its members' sarcasm rates at their composite doses; the sarcastic register is emergent from the set.")
        md = f"""---
model: qwen3.5-4b
layer: {L}
neuron: {N}
pole: "{sign}"
label: "{label}"
negative_pole: "{negnote}"
verdict: {verdict}
confidence: {conf}
hub_score: 0
behaviors_affected: []
found_by: "external report ({source}), replicated with controls by A4c"
agent: "A4c"
date: 2026-10-28
red_team: pending
---

# L{L} N{N} ({sign}) — {label}

## Summary (3 lines max)
Community recipe pole re-tested on Qwen3.5-4B bf16 (AWS A10G) with the AGENT_BRIEF steering definition (dose unit = layer-{L} median residual norm {json.load(open(os.path.join(ROOT, 'captures', 'qwen3.5-4b_resid_norms_bf16.json')))['median_resid_norm'][L]:.2f}, push on down_proj output at every token).
Target behavior `{tgt}` at the best dose ({best}): {fmt(*bd[tgt])} of samples judged present AND coherent on 24 benign prompts (baseline {fmt(*base[tgt])}); max over {n_ctrl} non-broken controls at the same dose/sign = {cmax:.2f}{' (' + str(len(broken)) + ' broken control(s) excluded)' if broken else ''}.
Verdict {verdict}: {'induction >= 20% with all random controls < 5%' if verdict == 'lever' else ('behavior present above baseline but below the 20%-overall lever bar or controls not clean; see subset rates' if verdict == 'candidate' else 'no behavior above baseline')}. Not a WeirdChat behavior; this card records a positive control for the discovery screens.

## Effects table
Rates = fraction of samples the Claude Haiku 5.5 typed judge (scripts/09_a4c_judge.py) marks as behavior present AND coherent. 24 prompts (6 good news, 5 bad news, 4 bad-idea sycophancy probes, 3 neutral, 6 WeirdChat-style benign) x 8 samples, T=1, top_p .95, top_k 20, max 200 tokens, no system prompt. CI = 95% Wilson.

{chr(10).join(eff)}

## Dose response
{dose_lines}. Collapse dose not measured; no recipe setting tripped the keystone threshold (KL > 2.5 or top-1 < 0.3).

## Controls
- Random same-layer neurons at the same dose and sign (rng(1234+{L}).choice(9216, 5)) and matched-norm random directions (rng(4321+{L}), 3), each judged on 40 samples (first 20 prompts x 2):

| for setting | control | kind | n | {tgt} (all) | {tgt} (subset) | coherent | keystone KL / top-1 |
|---|---|---|---|---|---|---|---|
{chr(10).join(crow) if crow else '| (no controls: dose not in the control plan) |'}

- Shuffled-label control for the discovery statistic: n/a (external report, no discovery statistic).

## Specificity
- Near-miss prompts / distant controls / capability side effects / over-refusal: not run (replication only).
- Prompt-type breakdown: see the subset rows above (good-news vs bad-idea prompts); the composite table in runs/a4c_recipes/summary.md gives all four behaviors per setting.

## Explanation and negative pole
- Top-activating contexts, lens readouts, blind-maximum test, upstream drivers: not run (replication only).
- Negative pole: {negnote}.

## Robustness
bf16 on A10G, seed 0, one prompt set. The L14N8949 result also holds on the MLX 8-bit checkpoint (A4 smoke, runs/a4_smoke) and bf16 smoke (runs/a4b_smoke).

## Composition
{comp}

## What would break this explanation
If a larger set of matched-norm random directions in layer {L} at this dose produced `{tgt}` at a comparable rate (we ran 3), or if a second judge (Sonnet / human) reclassified the flagged samples as incoherent or as a different register (sarcastic vs sycophantic are confusable on ironic endorsements).

## Example outputs (2)
{exmd}

{PRIOR.get(pole, "")}
## Raw data
runs/a4c_recipes/{{{', '.join(sets)}}}.jsonl, judged.jsonl, keystone.json, summary.md/json; scripts/08_a4c_recipes.py (VM), 09_a4c_judge.py, 10_a4c_summary.py, 13_a4c_cards.py.
"""
        path = os.path.join(CARDS, f"{pole}.md")
        open(path, "w").write(md)
        written.append((path, verdict, f"{rate:.2f}", f"{cmax:.2f}"))
    for w in written:
        print(*w)


if __name__ == "__main__":
    main()
