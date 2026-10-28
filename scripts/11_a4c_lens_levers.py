"""11_a4c_lens_levers.py -- are lyra's 4B steer-lens "commitment-token writers" levers for natural WeirdChat behaviors? (A4c add-on)

(a) "Yes"-writers L11N2557, L13N2180, L13N7901 at +0.5, +0.8 (induction) and -0.8 (suppression) on the 20
    recommends-drunk-driving prompts of runs/a4b_replay/recommends-drunk-driving.jsonl (A4b replay baseline 0.28).
(b) "body"-writers L22N8664 (reported pole -), L25N3294 (+), L22N4283 (+) at +-0.5, +-0.8 on the 20 false-physical-embodiment
    prompts (A4b replay baseline 0.59).
Each: 8 samples per prompt, T=1, top_p .95, top_k 20, max 300 new tokens (as the replay), thinking off, no system prompt.
Steering as AGENT_BRIEF (down_proj OUTPUT push, dose unit = layer median residual norm, bf16 calibration).
Controls per (layer, dose) group shared by the target neurons of that layer: 5 random same-layer neurons rng(1234+L).choice(9216, 5)
and 3 matched-norm random directions rng(4321+L). Keystone per setting (20 held-out benign prompts, 32 teacher-forced tokens).
Pre-screen every sample with src/heuristic_judge.py for the prompt set's behavior; Haiku judging runs locally (scripts/12_a4c_lens_judge.py).
Outputs runs/a4c_lens/<setting>.jsonl, keystone.json, settings.json.

Run on VM: source ~/venv/bin/activate && cd ~/evil-neuron-interp && python scripts/11_a4c_lens_levers.py --part a --batch-size 80
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from harness_4b import Harness4B, D_MLP_4B, D_MODEL_4B  # noqa: E402
from heuristic_judge import distinct_ngram_ratio, judge_heuristic  # noqa: E402

OUT = "runs/a4c_lens"
REPLAY = "runs/a4b_replay"
BENIGN = "promptsets/induction_benign.jsonl"
PROMPT_DIR = os.path.expanduser("~/data/weirdchat/prompts")
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20)
N_SAMPLES, MAX_NEW = 8, 300
PARTS = {
    "a": {"behavior": "recommends-drunk-driving", "label": "Yes-writer",
          "neurons": [(11, 2557), (13, 2180), (13, 7901)], "doses": [0.5, 0.8, -0.8]},
    "b": {"behavior": "false-physical-embodiment", "label": "body-writer",
          "neurons": [(22, 8664), (25, 3294), (22, 4283)], "doses": [0.5, 0.8, -0.5, -0.8]},
}


def random_neurons(layer, exclude, n=5):
    picks = [int(x) for x in np.random.default_rng(1234 + layer).choice(D_MLP_4B, n + len(exclude), replace=False)]
    return [p for p in picks if p not in exclude][:n]


def random_directions(layer, n=3):
    g = np.random.default_rng(4321 + layer).standard_normal((n, D_MODEL_4B)).astype(np.float32)
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def settings(part):
    P = PARTS[part]
    out = [(f"{part}_baseline", "baseline", [])]
    for (L, N) in P["neurons"]:
        for d in P["doses"]:
            out.append((f"{part}_L{L}N{N}{d:+.1f}", "lens_pole", [(L, N, d)]))
    layers = sorted({L for L, _ in P["neurons"]})
    for L in layers:
        excl = {N for l, N in P["neurons"] if l == L}
        for d in P["doses"]:
            for rn in random_neurons(L, excl):
                out.append((f"{part}_ctrl_rn_L{L}N{rn}{d:+.1f}", "random_neuron", [(L, rn, d)]))
            for i in range(3):
                out.append((f"{part}_ctrl_rd_L{L}r{i}{d:+.1f}", "random_direction", {"layer": L, "dir_idx": i, "dose": d}))
    return out


def load_prompts(behavior):
    rows = [json.loads(l) for l in open(f"{REPLAY}/{behavior}.jsonl")]
    seen, recs = set(), []
    for r in rows:
        if r["prompt_id"] not in seen:
            seen.add(r["prompt_id"]); recs.append({"prompt_id": r["prompt_id"], "user_text": r["prompt"]})
    msgs = {}
    for f in glob.glob(os.path.join(PROMPT_DIR, "qwen__*", "*.parquet")):
        import pandas as pd
        for x in pd.read_parquet(f).itertuples():
            msgs[x.prompt_id] = [{"role": m["role"], "content": m["content"]} for m in x.messages]
    for r in recs:
        r["messages"] = msgs.get(r["prompt_id"]) or [{"role": "user", "content": r["user_text"]}]
    heldout = [json.loads(l) for l in open(BENIGN)][:20]
    return recs, heldout


class DirectionPush:
    def __init__(self, h, layer, u, dose):
        delta = (dose * h.resid_norms[layer] * torch.tensor(u, dtype=torch.float32)).to(torch.bfloat16).to(h.device)
        self.handle = h.layers[layer].mlp.down_proj.register_forward_hook(lambda m, a, o, d=delta: o + d)

    def remove(self):
        self.handle.remove()


def apply(h, kind, spec):
    if kind == "baseline":
        return None
    if kind == "random_direction":
        return DirectionPush(h, spec["layer"], random_directions(spec["layer"])[spec["dir_idx"]], spec["dose"])
    return h.steer(spec)


@torch.no_grad()
def keystone_refs(h, prompts, n_tok=32):
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
        out.append(torch.log_softmax(logits[lp - 1:len(ids) - 1].float(), -1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b"], required=True)
    ap.add_argument("--batch-size", type=int, default=80)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    P = PARTS[a.part]
    sets = settings(a.part)
    recs, heldout = load_prompts(P["behavior"])
    spath = f"{OUT}/settings.json"
    allsets = json.load(open(spath)) if os.path.exists(spath) else []
    allsets = [s for s in allsets if not s["name"].startswith(a.part + "_")] + [{"name": n, "kind": k, "spec": s, "part": a.part, "behavior": P["behavior"]} for n, k, s in sets]
    json.dump(allsets, open(spath, "w"), indent=1)
    json.dump({"prompts": [{k: r[k] for k in ("prompt_id", "user_text")} | {"n_turns": len(r["messages"])} for r in recs], "behavior": P["behavior"],
               "n_samples": N_SAMPLES, "max_new": MAX_NEW, "sampling": SAMPLING}, open(f"{OUT}/prompts_{a.part}.json", "w"), indent=1)
    h = Harness4B(resid_norms_path="captures/qwen3.5-4b_resid_norms_bf16.json")
    print(f"[lens {a.part}] {len(sets)} settings x {len(recs)} prompts x {N_SAMPLES}; behavior {P['behavior']}", flush=True)
    refs = keystone_refs(h, [x["prompt"] for x in heldout])
    base_lp = keystone_logprobs(h, refs)
    kpath = f"{OUT}/keystone.json"
    keystone = json.load(open(kpath)) if os.path.exists(kpath) else {"settings": {}}
    msgs = [r["messages"] for r in recs]
    t_all = time.time()
    for name, kind, spec in sets:
        path = f"{OUT}/{name}.jsonl"
        done = os.path.exists(path) and sum(1 for _ in open(path)) >= len(recs) * N_SAMPLES
        if done and name in keystone["settings"]:
            print(f"[{name}] exists, skipping", flush=True); continue
        hnd = apply(h, kind, spec)
        try:
            lp = keystone_logprobs(h, refs)
            kl = float(torch.cat([(b.exp() * (b - s)).sum(-1) for b, s in zip(base_lp, lp)]).mean())
            ag = float(torch.cat([(b.argmax(-1) == s.argmax(-1)).float() for b, s in zip(base_lp, lp)]).mean())
            keystone["settings"][name] = {"kind": kind, "spec": spec, "kl": kl, "top1_agreement": ag, "broken": kl > 2.5 or ag < 0.3}
            json.dump(keystone, open(kpath, "w"), indent=1)
            print(f"[{name}] keystone KL {kl:.3f} top1 {ag:.3f} broken={kl > 2.5 or ag < 0.3}", flush=True)
            if done:
                continue
            t0 = time.time()
            outs = h.chat_full(msgs, n=N_SAMPLES, max_new_tokens=MAX_NEW, batch_size=a.batch_size, seed=0, verbose=False, **SAMPLING)
        finally:
            if hnd:
                hnd.remove()
        rows = []
        for r, samples in zip(recs, outs):
            for i, s in enumerate(samples):
                rows.append({"setting": name, "kind": kind, "spec": spec, "part": a.part, "behavior": P["behavior"], "prompt_id": r["prompt_id"],
                             "prompt": r["user_text"], "sample_idx": i, **s, "distinct4": distinct_ngram_ratio(s["response"]),
                             "heur": judge_heuristic(P["behavior"], r["user_text"], s["response"])["match"]})
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{name}] {len(rows)} samples in {time.time()-t0:.0f}s; heur {sum(r['heur'] for r in rows)}/{len(rows)}; "
              f"mean d4 {sum(r['distinct4'] for r in rows)/len(rows):.2f}; elapsed {(time.time()-t_all)/60:.1f} min  SETTING_DONE", flush=True)
    print(f"LENS_{a.part.upper()}_DONE", flush=True)


if __name__ == "__main__":
    main()
