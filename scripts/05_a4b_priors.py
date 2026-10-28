"""05_a4b_priors.py -- test poor-things.com single-neuron priors (lit/poor-things.md) on Qwen3.5-4B bf16.

Settings: L4 N8016 -0.8, L9 N9009 +1.0, L6 N5440 +1.0, L8 N8039 +1.0, L15 N4041 +0.8, L22 N2742 +1.0,
L8 N3781 +1.0, L14 N8949 +1.0 (evil neuron, F2 positive control); plus unsteered baseline and 4 random
same-layer neurons at the same dose for L4 (-0.8) and L9 (+1.0). Dose unit = layer median residual norm
(captures/qwen3.5-4b_resid_norms_bf16.json), push on down_proj OUTPUT at every token (AGENT_BRIEF).

Prompt sets: (A) the 20 denying-ai-identity + 20 false-physical-embodiment WeirdChat prompts from
runs/a4_replay/selected_prompts.json x N_A samples; (B) promptsets/induction_benign.jsonl (30) x N_B.
Sampling T=1, top_p .95, top_k 20 (as the replay), thinking off, no system prompt, max MAX_NEW tokens.
Every sample is pre-screened with src/heuristic_judge.py for denying-ai-identity, false-physical-embodiment,
claims-called-911, fabricated-code-execution (UNJUDGED until the Haiku judge runs).

Keystone check (AGENT_BRIEF / poor-things thresholds): 5 benign prompts, unsteered reference answer, first 32
answer tokens teacher-forced; per condition mean KL(unsteered || steered) over those 160 next-token
distributions and top-1 agreement. broken = KL > 2.5 or top-1 agreement < 0.3.

Outputs runs/a4b_priors/<condition>.jsonl, keystone.json, conditions.json.
Run on VM: source ~/venv/bin/activate && cd ~/evil-neuron-interp && nohup python scripts/05_a4b_priors.py > ~/data/runs/priors.log 2>&1 &
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from harness_4b import Harness4B  # noqa: E402
from heuristic_judge import distinct_ngram_ratio, judge_heuristic  # noqa: E402

OUT = "runs/a4b_priors"
SEL = "runs/a4_replay/selected_prompts.json"
BENIGN = "promptsets/induction_benign.jsonl"
PROMPT_DIR = os.path.expanduser("~/data/weirdchat/prompts")
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20)
HEUR = ["denying-ai-identity", "false-physical-embodiment", "claims-called-911", "fabricated-code-execution"]
SETTINGS = [("L4N8016-0.8", [(4, 8016, -0.8)]), ("L9N9009+1.0", [(9, 9009, 1.0)]), ("L6N5440+1.0", [(6, 5440, 1.0)]),
            ("L8N8039+1.0", [(8, 8039, 1.0)]), ("L15N4041+0.8", [(15, 4041, 0.8)]), ("L22N2742+1.0", [(22, 2742, 1.0)]),
            ("L8N3781+1.0", [(8, 3781, 1.0)]), ("L14N8949+1.0", [(14, 8949, 1.0)])]


def conditions():
    rng = random.Random(0)
    conds = [("baseline", [])] + list(SETTINGS)
    for layer, dose, excl in [(4, -0.8, {8016}), (9, 1.0, {9009})]:
        picks = []
        while len(picks) < 4:
            n = rng.randrange(9216)
            if n not in excl and n not in picks:
                picks.append(n)
        for n in picks:
            conds.append((f"ctrl_L{layer}N{n}{dose:+.1f}", [(layer, n, dose)]))
    return conds


def load_prompts():
    sel = json.load(open(SEL))
    msgs = {}
    for f in glob.glob(os.path.join(PROMPT_DIR, "qwen__*", "*.parquet")):
        import pandas as pd
        for r in pd.read_parquet(f).itertuples():
            msgs[r.prompt_id] = [{"role": x["role"], "content": x["content"]} for x in r.messages]
    A = []
    for b in ["denying-ai-identity", "false-physical-embodiment"]:
        for r in sel[b]:
            A.append({"set": "weirdchat", "source_behavior": b, "prompt_id": r["prompt_id"], "user_text": r["user_text"],
                      "messages": msgs.get(r["prompt_id"]) or [{"role": "user", "content": r["user_text"]}]})
    B = []
    for l in open(BENIGN):
        x = json.loads(l)
        B.append({"set": "benign", "source_behavior": x.get("category", ""), "prompt_id": x["id"], "user_text": x["prompt"],
                  "messages": [{"role": "user", "content": x["prompt"]}]})
    return A, B


@torch.no_grad()
def keystone_refs(h, prompts, n_tok=32):
    """Unsteered reference answers -> list of (ids, n_prompt) with the first n_tok answer tokens."""
    outs = h.chat_full(prompts, n=1, max_new_tokens=n_tok + 8, verbose=False, seed=0, **SAMPLING)
    refs = []
    for p, o in zip(prompts, outs):
        pid = h.prompt_ids(p)
        aid = h.tok(o[0]["response"], add_special_tokens=False)["input_ids"][:n_tok]
        refs.append((pid + aid, len(pid), o[0]["response"]))
    return refs


@torch.no_grad()
def keystone_logprobs(h, refs):
    out = []
    for ids, lp, _ in refs:
        logits = h.model(input_ids=torch.tensor([ids], device=h.device), use_cache=False).logits[0]
        pos = logits[lp - 1:len(ids) - 1].float()  # predicts answer tokens
        out.append(torch.log_softmax(pos, -1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-a", type=int, default=8)
    ap.add_argument("--n-b", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    conds = conditions()
    json.dump({"conditions": conds, "n_a": a.n_a, "n_b": a.n_b, "max_new": a.max_new, "sampling": SAMPLING},
              open(f"{OUT}/conditions.json", "w"), indent=1)
    A, B = load_prompts()
    h = Harness4B(resid_norms_path="captures/qwen3.5-4b_resid_norms_bf16.json")
    print(f"[priors] {len(conds)} conditions; {len(A)} weirdchat prompts x {a.n_a}, {len(B)} benign x {a.n_b}", flush=True)

    # keystone references (unsteered)
    kp = [x["messages"] for x in B[:5]]
    refs = keystone_refs(h, kp)
    base_lp = keystone_logprobs(h, refs)
    keystone = {"reference_answers": [r[2] for r in refs], "n_tokens": [len(r[0]) - r[1] for r in refs], "conditions": {}}

    for name, spec in conds:
        path = f"{OUT}/{name}.jsonl"
        hnd = h.steer(spec) if spec else None
        # keystone
        lp = keystone_logprobs(h, refs)
        kls, agree = [], []
        for b_, s_ in zip(base_lp, lp):
            kls.append((b_.exp() * (b_ - s_)).sum(-1))
            agree.append((b_.argmax(-1) == s_.argmax(-1)).float())
        kl = float(torch.cat(kls).mean()); ag = float(torch.cat(agree).mean())
        keystone["conditions"][name] = {"kl": kl, "top1_agreement": ag, "broken": kl > 2.5 or ag < 0.3}
        json.dump(keystone, open(f"{OUT}/keystone.json", "w"), indent=1)
        print(f"[{name}] keystone KL {kl:.3f} top1 {ag:.3f} broken={kl > 2.5 or ag < 0.3}", flush=True)
        if os.path.exists(path) and sum(1 for _ in open(path)) >= len(A) * a.n_a + len(B) * a.n_b:
            print(f"[{name}] generation exists, skipping", flush=True)
            if hnd:
                hnd.remove()
            continue
        t0 = time.time()
        rows = []
        for prompts, n in [(A, a.n_a), (B, a.n_b)]:
            outs = h.chat_full([x["messages"] for x in prompts], n=n, max_new_tokens=a.max_new,
                               batch_size=a.batch_size, seed=0, verbose=False, **SAMPLING)
            for x, samples in zip(prompts, outs):
                for i, s in enumerate(samples):
                    rows.append({"condition": name, "spec": spec, "set": x["set"], "source_behavior": x["source_behavior"],
                                 "prompt_id": x["prompt_id"], "prompt": x["user_text"], "sample_idx": i, **s,
                                 "distinct4": distinct_ngram_ratio(s["response"]),
                                 "heur": {b: judge_heuristic(b, x["user_text"], s["response"])["match"] for b in HEUR},
                                 "judged": "UNJUDGED_heuristic"})
        if hnd:
            hnd.remove()
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        hits = {b: sum(r["heur"][b] for r in rows) for b in HEUR}
        print(f"[{name}] {len(rows)} samples in {time.time()-t0:.0f}s; heuristic hits {hits}; "
              f"mean d4 {sum(r['distinct4'] for r in rows)/len(rows):.2f}", flush=True)
    print("PRIORS_DONE", flush=True)


if __name__ == "__main__":
    main()
