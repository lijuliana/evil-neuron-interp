"""A3 step 2: GPU causal battery for cross-family hub candidates + chemtrails leaders (Qwen3.6-27B, A100).

Reads --config (runs/a3_hubs/battery_config.json): {"hubs": [{"layer","neuron","sign"}...], "chem_leaders": [...]}
where sign = the pole associated with the behaviors in S1 (consensus sign of the within-prompt d); the
suppression push is -sign * dose, mean-ablation is sign-free.

Phases (resumable; each cell appended to <out>/gens.jsonl, keystone to <out>/keystone.json; --max-minutes stops
cleanly so the job can be re-invoked under flock):
  keystone  steer-lens KL(unsteered||steered) + top-1 agreement over 32 reference tokens of 5 benign answers,
            NLL ratio on 5 held-out benign answers, for every condition used below (broken = KL>2.5 or top1<0.3)
  none      unsteered samples on the 6 battery behaviors x 10 prompts x 16 (the matched control for suppression)
  suppress  hubs: {ablate, push -s*1.0, push -s*0.5} x 6 behaviors x 10 prompts x 16
  controls  20 random same-layer neurons and 20 matched-norm random directions at dose -1.0 (push) on the same
            60 prompts x 4 samples (a null for a generic coherence/behavior drop)
  induce    hubs on 30 benign prompts: push {+1.0, -1.0, +0.5, -0.5} x 8 samples; 20 random neurons (+1.0 for even
            index, -1.0 for odd) x 4 samples
  chem      chemtrails leaders: {ablate, -s*1.0, -s*0.5} on chemtrails + 2 transfer behaviors; induction at +-1.0
Sampling: no system prompt, thinking off, T=1.0, top_p .95, top_k 20, max_new_tokens 256 (shorter than the 400 of
the baseline for budget; the 'none' cells use identical settings so every comparison is like-for-like).

VM: flock -w 36000 /data/gpu.lock ~/venv/bin/python -I scripts/33_a3_gpu_battery.py --config runs/a3_hubs/battery_config.json --max-minutes 40
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd, torch
from harness import Harness, CALIBRATION_PROMPTS, _HookHandle, POSITIONS
from a3_common import BATTERY, random_neurons, random_directions, REPO

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(REPO, "runs", "a3_hubs", "battery_config.json"))
ap.add_argument("--data", default="/data/weirdchat")
ap.add_argument("--s1", default="/data/captures/s1")
ap.add_argument("--out", default="/data/runs/a3_battery")
ap.add_argument("--promptsets", default=os.path.join(REPO, "promptsets"))
ap.add_argument("--resid", default=os.path.join(REPO, "captures", "qwen3.6-27b_resid_norms.json"))
ap.add_argument("--phases", default="keystone,none,suppress,controls,induce,chem,suppress05,induce05")
ap.add_argument("--n-prompts", type=int, default=10)
ap.add_argument("--n-samples", type=int, default=16)
ap.add_argument("--n-ind", type=int, default=8)
ap.add_argument("--n-ctrl", type=int, default=4)
ap.add_argument("--max-new", type=int, default=256)
ap.add_argument("--batch", type=int, default=96)
ap.add_argument("--max-minutes", type=float, default=40)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T0 = time.time()
cfg = json.load(open(args.config))
HUBS = [(int(h["layer"]), int(h["neuron"]), int(h["sign"])) for h in cfg["hubs"]]
CHEM = [(int(h["layer"]), int(h["neuron"]), int(h["sign"])) for h in cfg["chem_leaders"]]
CHEM_BEH = cfg.get("chem_behaviors", ["chemtrails-assertion", "denying-ai-identity", "recommends-drunk-driving"])
done_path = f"{args.out}/done.json"
done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()


def time_left():
    return (time.time() - T0) / 60 < args.max_minutes


def mark(key):
    done.add(key); json.dump(sorted(done), open(done_path, "w"))


# ---------------------------------------------------------------- prompts
def behavior_prompts(b):
    p = pd.read_parquet(f"{args.data}/prompts/qwen__qwen3.6-27b/{b}.parquet")
    p["n_matched"] = p.match_summary.apply(lambda s: s["n_matched"]); p["n_samples"] = p.match_summary.apply(lambda s: s["n_samples"])
    p["frac"] = p.n_matched / p.n_samples
    p = p[p.messages.apply(len) == 1].sort_values(["frac", "n_matched"], ascending=False).head(args.n_prompts)
    return [dict(behavior=b, prompt_id=r.prompt_id, prompt=r.messages[0]["content"], ref_rate=float(r.frac)) for r in p.itertuples()]

SUPP = {b: behavior_prompts(b) for b in sorted(set(BATTERY) | set(CHEM_BEH))}
IND = [dict(behavior="benign", prompt_id=j["id"], prompt=j["prompt"], ref_rate=0.0)
       for j in map(json.loads, open(f"{args.promptsets}/induction_benign.jsonl"))]
print(f"[battery] prompts: {{ {', '.join(f'{b}:{len(v)}' for b, v in SUPP.items())} }}, induction {len(IND)}", flush=True)

# ---------------------------------------------------------------- ablation means (unmatched group, asst_mean, averaged over battery behaviors)
abl_mean = {}
for (L, N, _) in HUBS + CHEM:
    vals = []
    for b in BATTERY:
        z = np.load(f"{args.s1}/behavior/{b}__unmatched.npz", allow_pickle=False)
        vals.append(float(z["mean_asst_mean"][L, N]))
    abl_mean[(L, N)] = float(np.mean(vals))
print("[battery] ablation means:", {f"L{L}N{N}": round(v, 3) for (L, N), v in abl_mean.items()}, flush=True)

# ---------------------------------------------------------------- random controls (shared protocol seeds; 20 of each spread over the hub layers)
hub_layers = sorted({L for L, _, _ in HUBS})
RAND_N, RAND_D = [], []
i = 0
while len(RAND_N) < 20:
    L = hub_layers[i % len(hub_layers)]; k = i // len(hub_layers)
    RAND_N.append((L, random_neurons(L, 20)[k])); RAND_D.append((L, k)); i += 1

h = Harness(resid_norms_path=args.resid)
assert h.resid_norms is not None


def steer_direction(L, k, dose):
    """Matched-norm random direction push: dose * median_resid_norm[L] * u_k added to down_proj output."""
    u = torch.tensor(random_directions(L, 20)[k], dtype=torch.float32, device=h.device)
    d = (dose * h.resid_norms[L] * u).to(torch.bfloat16)

    def hook(mod, args_, output, d=d):
        return output + d
    hh = _HookHandle([h.layers[L].mlp.down_proj.register_forward_hook(hook)])
    h._active.append(hh)
    return hh


def apply(cond):
    """cond: dict(kind, layer, neuron|k, dose). Returns hook handle (or None for 'none')."""
    k = cond["kind"]
    if k == "none":
        return None
    if k == "ablate":
        return h.ablate_mean(cond["layer"], cond["neuron"], abl_mean[(cond["layer"], cond["neuron"])])
    if k in ("push", "rand_neuron"):
        return h.steer([(cond["layer"], cond["neuron"], cond["dose"])])
    if k == "rand_dir":
        return steer_direction(cond["layer"], cond["k"], cond["dose"])
    raise ValueError(k)


def cond_id(c):
    if c["kind"] == "none":
        return "none"
    if c["kind"] == "ablate":
        return f"ablate_L{c['layer']}N{c['neuron']}"
    if c["kind"] in ("push", "rand_neuron"):
        return f"{c['kind']}_L{c['layer']}N{c['neuron']}_d{c['dose']:+.2f}"
    return f"rand_dir_L{c['layer']}k{c['k']}_d{c['dose']:+.2f}"


# ---------------------------------------------------------------- keystone
@torch.no_grad()
def tf_logits(seqs):
    """Teacher-force list of token-id lists (right padded); returns list of [T_i, V] float32 log-probs."""
    pad = h.tok.pad_token_id
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad, dtype=torch.long); attn = torch.zeros((len(seqs), T), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s); attn[b, :len(s)] = 1
    out = h.model(input_ids=ids.to(h.device), attention_mask=attn.to(h.device), use_cache=False).logits
    return [torch.log_softmax(out[b, :len(s)].float(), -1) for b, s in enumerate(seqs)]


def keystone():
    if "keystone" in done:
        return
    ref_prompts = CALIBRATION_PROMPTS[:5]
    held_prompts = [p["prompt"] for p in IND[:5]]
    ref = h.chat(ref_prompts, n=1, max_new_tokens=32, temperature=0.0, verbose=False)
    held = h.chat(held_prompts, n=1, max_new_tokens=96, temperature=0.0, verbose=False)
    seqs_ref = [h.prompt_ids(p) + h.tok(r[0], add_special_tokens=False)["input_ids"] for p, r in zip(ref_prompts, ref)]
    seqs_held = [h.prompt_ids(p) + h.tok(r[0], add_special_tokens=False)["input_ids"] for p, r in zip(held_prompts, held)]
    plen_ref = [len(h.prompt_ids(p)) for p in ref_prompts]; plen_held = [len(h.prompt_ids(p)) for p in held_prompts]

    def answer_lp(lps, seqs, plens):
        """log-probs at positions predicting the answer tokens, and the answer token ids."""
        out = []
        for lp, s, pl in zip(lps, seqs, plens):
            tgt = torch.tensor(s[pl:], device=lp.device)
            out.append((lp[pl - 1:len(s) - 1], tgt))
        return out
    base_ref = answer_lp(tf_logits(seqs_ref), seqs_ref, plen_ref)
    base_held = answer_lp(tf_logits(seqs_held), seqs_held, plen_held)
    base_nll = float(np.mean([-(lp.gather(1, t[:, None]).mean()).item() for lp, t in base_held]))
    conds = [dict(kind="none")]
    for (L, N, s) in HUBS + CHEM:
        conds.append(dict(kind="ablate", layer=L, neuron=N))
        for dose in (0.5, 1.0, -0.5, -1.0):
            conds.append(dict(kind="push", layer=L, neuron=N, dose=dose))
    for (L, N) in RAND_N:
        for dose in (1.0, -1.0):
            conds.append(dict(kind="rand_neuron", layer=L, neuron=N, dose=dose))
    for (L, k) in RAND_D:
        for dose in (1.0, -1.0):
            conds.append(dict(kind="rand_dir", layer=L, k=k, dose=dose))
    rows = []
    for c in conds:
        hh = apply(c)
        try:
            st_ref = answer_lp(tf_logits(seqs_ref), seqs_ref, plen_ref)
            st_held = answer_lp(tf_logits(seqs_held), seqs_held, plen_held)
        finally:
            if hh is not None:
                hh.remove()
        kls, top1 = [], []
        for (lp0, t), (lp1, _) in zip(base_ref, st_ref):
            kls.append((lp0.exp() * (lp0 - lp1)).sum(-1).mean().item())
            top1.append((lp0.argmax(-1) == lp1.argmax(-1)).float().mean().item())
        nll = float(np.mean([-(lp.gather(1, t[:, None]).mean()).item() for lp, t in st_held]))
        r = dict(cond=cond_id(c), **{k: v for k, v in c.items()}, kl=float(np.mean(kls)), top1=float(np.mean(top1)),
                 nll_ratio=float(np.exp(nll - base_nll)))
        r["broken"] = bool(r["kl"] > 2.5 or r["top1"] < 0.3)
        rows.append(r)
        print(f"[keystone] {r['cond']:34s} KL={r['kl']:.3f} top1={r['top1']:.2f} ppl_ratio={r['nll_ratio']:.3f}{'  BROKEN' if r['broken'] else ''}", flush=True)
    json.dump(dict(reference_answers=[r[0] for r in ref], heldout_answers=[r[0] for r in held], rows=rows),
              open(f"{args.out}/keystone.json", "w"), indent=1)
    mark("keystone")


# ---------------------------------------------------------------- sampling cells
def run_cell(phase, cond, prompts, n):
    key = f"{phase}|{cond_id(cond)}|{len(prompts)}x{n}"
    if key in done:
        return
    if not time_left():
        print(f"[battery] time budget reached before {key}; exiting for resume", flush=True); sys.exit(0)
    hh = apply(cond)
    t0 = time.time()
    try:
        try:
            outs = h.chat([p["prompt"] for p in prompts], n=n, max_new_tokens=args.max_new, temperature=1.0, top_p=0.95,
                          top_k=20, batch_size=args.batch, seed=0, verbose=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); args.batch = max(32, args.batch // 2)
            print(f"[battery] OOM -> batch {args.batch}", flush=True)
            outs = h.chat([p["prompt"] for p in prompts], n=n, max_new_tokens=args.max_new, temperature=1.0, top_p=0.95,
                          top_k=20, batch_size=args.batch, seed=0, verbose=False)
    finally:
        if hh is not None:
            hh.remove()
    with open(f"{args.out}/gens.jsonl", "a") as fo:
        for p, resp in zip(prompts, outs):
            for i, text in enumerate(resp):
                ntok = len(h.tok(text, add_special_tokens=False)["input_ids"])
                fo.write(json.dumps(dict(phase=phase, cond=cond_id(cond), **{k: v for k, v in cond.items()}, behavior=p["behavior"],
                                         prompt_id=p["prompt_id"], prompt=p["prompt"], ref_rate=p["ref_rate"], sample_idx=i,
                                         response=text, n_tokens=ntok, truncated=ntok >= args.max_new - 2)) + "\n")
    mark(key)
    print(f"[battery] {key}: {len(prompts)*n} samples in {time.time()-t0:.0f}s (elapsed {(time.time()-T0)/60:.1f} min)", flush=True)


phases = args.phases.split(",")
supp_prompts = [p for b in BATTERY for p in SUPP[b]]
chem_prompts = [p for b in CHEM_BEH for p in SUPP[b]]
if "keystone" in phases:
    keystone()
if "none" in phases:
    run_cell("none", dict(kind="none"), supp_prompts, args.n_samples)
if "suppress" in phases:
    for (L, N, s) in HUBS:
        run_cell("suppress", dict(kind="ablate", layer=L, neuron=N), supp_prompts, args.n_samples)
    for (L, N, s) in HUBS:
        run_cell("suppress", dict(kind="push", layer=L, neuron=N, dose=-1.0 * s), supp_prompts, args.n_samples)
if "controls" in phases:
    for (L, N) in RAND_N:
        run_cell("controls", dict(kind="rand_neuron", layer=L, neuron=N, dose=-1.0), supp_prompts, args.n_ctrl)
    for (L, k) in RAND_D:
        run_cell("controls", dict(kind="rand_dir", layer=L, k=k, dose=-1.0), supp_prompts, args.n_ctrl)
if "induce" in phases:
    run_cell("induce", dict(kind="none"), IND, args.n_ind)
    for (L, N, s) in HUBS:
        for dose in (1.0, -1.0):
            run_cell("induce", dict(kind="push", layer=L, neuron=N, dose=dose), IND, args.n_ind)
    for j, (L, N) in enumerate(RAND_N):
        run_cell("induce", dict(kind="rand_neuron", layer=L, neuron=N, dose=1.0 if j % 2 == 0 else -1.0), IND, args.n_ctrl)
if "chem" in phases:
    # the 'none' cells for the chem transfer behaviors not in BATTERY are covered by 'none' (CHEM_BEH subset of BATTERY)
    for (L, N, s) in CHEM:
        run_cell("chem", dict(kind="ablate", layer=L, neuron=N), chem_prompts, args.n_samples)
        run_cell("chem", dict(kind="push", layer=L, neuron=N, dose=-1.0 * s), chem_prompts, args.n_samples)
    for (L, N, s) in CHEM:
        for dose in (1.0, -1.0):
            run_cell("chem_induce", dict(kind="push", layer=L, neuron=N, dose=dose), IND, args.n_ind)
if "suppress05" in phases:
    for (L, N, s) in HUBS:
        run_cell("suppress", dict(kind="push", layer=L, neuron=N, dose=-0.5 * s), supp_prompts, args.n_samples)
    for (L, N, s) in CHEM:
        run_cell("chem", dict(kind="push", layer=L, neuron=N, dose=-0.5 * s), chem_prompts, args.n_samples)
if "induce05" in phases:
    for (L, N, s) in HUBS:
        for dose in (0.5, -0.5):
            run_cell("induce", dict(kind="push", layer=L, neuron=N, dose=dose), IND, args.n_ind)
print(f"[battery] ALL PHASES DONE in {(time.time()-T0)/60:.1f} min", flush=True)
open(f"{args.out}/ALL_DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
