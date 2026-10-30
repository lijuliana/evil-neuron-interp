"""G1 generation: zero-ablate Agastya's shared neuron sets (and their matched random sets) in vLLM and sample
replies to the high-rate prompts of the three gradual behaviors; measure damage as delta NLL per token.

Zero ablation = down_proj input 0 for the neuron at every token = zeroing its column of down_proj.weight,
edited in place between conditions (as Agastya did; prefix caching off). Weights are restored from a CPU copy
and restoration is checked bit-exactly on a probe set.

Conditions (per behavior b in drunk driving, chemtrails, cutting):
  clean, shared100, rand_shared100, sharedmid100, rand_sharedmid100, loo100|b, rand_loo100|b
Prompts: Stage A prompts of b with production (thinking-on) Gemma rate >= 0.1 (16 judged replies per prompt).
Sampling: Agastya's Stage A settings: no system prompt, thinking off (his prompt_ids), temperature 1,
top-k 20, top-p 0.95, 1,024-token cap (override with --top_k/--top_p).
Damage: per condition, NLL per reply token (vLLM prompt_logprobs) on (a) clean-model replies to the 200
promptsets/capability_mini.jsonl prompts, sampled here once, (b) 100 Stage A replies from the 7 other behaviors
(production-negative), truncated to 512 tokens; delta vs clean, pooled over tokens.
Outputs: <out>/gen.jsonl, <out>/damage.json, <out>/gen_meta.json
Run on the VM: flock /data/gpu.lock ~/vllm-venv/bin/python -I scripts/60_gradual_g1_gen.py
"""
import argparse, collections, gzip, json, os, random, re, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # in-process engine: apply_model sees our closures
import numpy as np
import torch

BEH = ["recommends-drunk-driving", "chemtrails-assertion", "cutting-instructions"]
EOS = {248046, 248044}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/data/models/Qwen3.6-27B")
ap.add_argument("--agastya", default="/data/agastya")
ap.add_argument("--capability", default=os.path.expanduser("~/evil-neuron-interp/promptsets/capability_mini.jsonl"))
ap.add_argument("--out", default="/data/runs/gradual_g1")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--min_on_rate", type=float, default=0.1)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--top_k", type=int, default=20)
ap.add_argument("--max_tokens", type=int, default=1024)
ap.add_argument("--n_other", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--smoke", action="store_true", help="2 prompts, 2 replies, 2 conditions")
args = ap.parse_args()
D = args.agastya
os.makedirs(args.out, exist_ok=True)
rng = random.Random(args.seed)

# ---------------------------------------------------------------- prompts and sets
on = collections.defaultdict(list)
on_by_key = {}
for l in open(f"{D}/stage_a_judge_on.jsonl"):
    x = json.loads(l); on[x["prompt_id"]].append(bool(x["match"])); on_by_key[x["key"]] = bool(x["match"])
gen = {}
other_pool = []
with gzip.open(f"{D}/stage_a_gen.jsonl.gz", "rt") as fh:
    for line in fh:
        g = json.loads(line)
        if g["behavior_id"] in BEH:
            gen[g["prompt_id"]] = {"b": g["behavior_id"], "split": g["split"], "prompt_ids": g["prompt_ids"]}
        elif g["behavior_id"] != "laser-at-aircraft":
            for i, s in enumerate(g["samples"]):
                if on_by_key.get(f"{g['prompt_id']}|{i}") is False:
                    ids = s["ids"][:512]
                    other_pool.append({"b": g["behavior_id"], "key": f"{g['prompt_id']}|{i}",
                                       "prompt_ids": g["prompt_ids"], "ids": ids})
prompts = {b: [] for b in BEH}
for pid, g in sorted(gen.items()):
    v = on[pid]
    if v and sum(v) / len(v) >= args.min_on_rate:
        prompts[g["b"]].append(pid)
if args.smoke:
    prompts = {b: v[:2] for b, v in prompts.items()}
    args.n = 2
print({b: len(v) for b, v in prompts.items()}, flush=True)
shared = json.load(open(f"{D}/shared_sets.json"))
conds = []  # (name, spec or None, behaviors)
conds.append(("clean", None, BEH))
for s in ["sharedmid100", "rand_sharedmid100", "shared100", "rand_shared100"]:
    conds.append((s, shared[s]["spec"], BEH))
for b in BEH:
    for s in [f"loo100|{b}", f"rand_loo100|{b}"]:
        conds.append((s, shared[s]["spec"], [b]))
if args.smoke:
    conds = conds[:2]
for name, spec, _ in conds:
    if spec:
        assert sum(len(v) for v in spec.values()) == 100, name

# damage sets
cap = [json.loads(l) for l in open(args.capability)]
rng.shuffle(other_pool)
seen_b = collections.Counter()
other = []
for x in other_pool:  # balance over the 7 other behaviors
    if seen_b[x["b"]] < -(-args.n_other // 7):
        other.append(x); seen_b[x["b"]] += 1
    if len(other) >= args.n_other:
        break
if args.smoke:
    cap, other = cap[:10], other[:10]

# ---------------------------------------------------------------- vLLM
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

llm = LLM(model=args.model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.90,
          enable_prefix_caching=False, seed=args.seed, limit_mm_per_prompt={"image": 0, "video": 0},
          max_num_seqs=256)
tok = llm.get_tokenizer()

ORIG = {}  # (layer, neuron) -> original column (CPU)
PAT = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.down_proj$")


def _downs(model):
    out = {}
    for name, mod in model.named_modules():
        if "visual" in name or "mtp" in name:
            continue
        m = PAT.search(name)
        if m:
            out[int(m.group(1))] = mod
    assert len(out) == 64, len(out)
    return out


def set_zero(spec):
    def f(model):
        downs = _downs(model)
        n = 0
        for L, ns in spec.items():
            W = downs[int(L)].weight
            assert W.shape == (5120, 17408), W.shape
            idx = torch.tensor(ns, device=W.device)
            for j in ns:
                ORIG.setdefault((int(L), j), W[:, j].detach().cpu().clone())
            W.data[:, idx] = 0
            n += int((W[:, idx].abs().sum(0) == 0).sum())
        torch.cuda.synchronize()
        return n
    return llm.apply_model(f)[0]


def restore(spec):
    def f(model):
        downs = _downs(model)
        for L, ns in spec.items():
            W = downs[int(L)].weight
            for j in ns:
                W.data[:, j] = ORIG[(int(L), j)].to(W.device)
        torch.cuda.synchronize()
        return True
    return llm.apply_model(f)[0]


sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    max_tokens=args.max_tokens, seed=None)
sp_cap = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=384)
sp_nll = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)


def nll_tokens(seqs):
    """seqs: list of (prompt_ids, reply_ids). Returns list of per-reply-token NLL arrays."""
    outs = llm.generate([TokensPrompt(prompt_token_ids=p + r) for p, r in seqs], sp_nll, use_tqdm=False)
    res = []
    for (p, r), o in zip(seqs, outs):
        pl = o.prompt_logprobs
        v = []
        for k, t in enumerate(r):
            d = pl[len(p) + k]
            v.append(-d[t].logprob)
        res.append(np.array(v, dtype=np.float64))
    return res


# capability reference replies (clean model), generated once
cap_prompt_ids = [tok.apply_chat_template([{"role": "user", "content": c["prompt"]}], tokenize=True,
                                          add_generation_prompt=True, enable_thinking=False) for c in cap]
cap_prompt_ids = [list(x["input_ids"]) if not isinstance(x, list) else x for x in cap_prompt_ids]
t0 = time.time()
cap_out = llm.generate([TokensPrompt(prompt_token_ids=p) for p in cap_prompt_ids], sp_cap, use_tqdm=False)
cap_seqs = []
for p, o in zip(cap_prompt_ids, cap_out):
    r = [t for t in o.outputs[0].token_ids if t not in EOS]
    cap_seqs.append((p, r if r else [tok.encode("OK", add_special_tokens=False)[0]]))
other_seqs = [(x["prompt_ids"], [t for t in x["ids"] if t not in EOS]) for x in other]
damage_sets = {"capability_mini": cap_seqs, "other_behaviors": other_seqs}
print(f"[g1] capability replies in {time.time()-t0:.0f}s; damage sets: "
      f"{ {k: (len(v), sum(len(r) for _, r in v)) for k, v in damage_sets.items()} }", flush=True)
probe = cap_seqs[:16]
clean_nll = {k: nll_tokens(v) for k, v in damage_sets.items()}
probe_ref = nll_tokens(probe)

gen_path = os.path.join(args.out, "gen.jsonl" if not args.smoke else "gen_smoke.jsonl")
done = set()
if os.path.exists(gen_path) and not args.smoke:
    for l in open(gen_path):
        done.add(json.loads(l)["cond"])
damage = {}
dmg_path = os.path.join(args.out, "damage.json" if not args.smoke else "damage_smoke.json")
if os.path.exists(dmg_path) and not args.smoke:
    damage = json.load(open(dmg_path))
meta = {"sampling": dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens,
                         n=args.n), "prompts": prompts, "conditions": [c[0] for c in conds],
        "damage_sets": {k: len(v) for k, v in damage_sets.items()}, "vllm": __import__("vllm").__version__}
json.dump(meta, open(os.path.join(args.out, "gen_meta.json"), "w"), indent=1)

with open(gen_path, "a") as fo:
    for name, spec, behs in conds:
        if name in done:
            print(f"[g1] skip {name} (done)", flush=True)
            continue
        t1 = time.time()
        if spec:
            nz = set_zero(spec)
            assert nz == 100, (name, nz)
        pids = [pid for b in behs for pid in prompts[b]]
        outs = llm.generate([TokensPrompt(prompt_token_ids=gen[pid]["prompt_ids"]) for pid in pids], sp, use_tqdm=False)
        ntok = 0
        for pid, o in zip(pids, outs):
            for j, c in enumerate(o.outputs):
                ntok += len(c.token_ids)
                fo.write(json.dumps({"cond": name, "b": gen[pid]["b"], "prompt_id": pid, "split": gen[pid]["split"],
                                     "j": j, "text": c.text, "ids": list(c.token_ids), "finish": c.finish_reason}) + "\n")
        fo.flush()
        tg = time.time() - t1
        if spec:
            nll = {k: nll_tokens(v) for k, v in damage_sets.items()}
            dm = {}
            for k in damage_sets:
                a = np.concatenate(nll[k]); c0 = np.concatenate(clean_nll[k])
                dm[k] = {"delta_nll_per_token": float((a - c0).mean()), "n_tokens": int(len(a)),
                         "clean_nll_per_token": float(c0.mean()),
                         "per_reply_delta_mean": float(np.mean([(x - y).mean() for x, y in zip(nll[k], clean_nll[k])]))}
            restore(spec)
            chk = nll_tokens(probe)
            dmax = float(max(np.abs(x - y).max() for x, y in zip(chk, probe_ref)))
            dm["restore_check_max_abs_dnll"] = dmax
            damage[name] = dm
            json.dump(damage, open(dmg_path, "w"), indent=1)
            print(f"[g1] {name}: damage {[(k, round(v['delta_nll_per_token'], 4)) for k, v in dm.items() if isinstance(v, dict)]} "
                  f"restore max|dNLL| {dmax:.2e}", flush=True)
        print(f"[g1] {name}: {len(pids)} prompts x {args.n}, {ntok} tokens in {tg:.0f}s = {ntok/tg:.0f} tok/s; "
              f"total {time.time()-t0:.0f}s", flush=True)
print("[g1] ALL_DONE", flush=True)
