"""Per-token data for the viewer: teacher-force each selected reply once clean and once per zero-ablated
neuron set (Agastya's sets), and record for every reply token j:
  lp[j]        log p(token j) from the clean run (predicted at position P+j-1)
  top[j]       clean top-5 candidates for token j: [[ids], [probs]]
  sets[S].dlp  log p_ablated(token j) - lp[j]   (negative = the set was supporting this token)
  sets[S].kl   KL(clean || ablated) over the next-token distribution at P+j-1
  acts[j]      activations (down_proj input) of the behavior's top-10 neurons at P+j-1
Sets per behavior: own top100 + its matched random set where Agastya built one, sharedmid100 + its random set always.
Resumable: appends one JSON line per reply to --out and skips rids already there.
Run on the VM: flock /data/gpu.lock ~/venv/bin/python -I scripts/51_viewer_tokens.py
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from harness import Harness  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--items", default="data/viewer/items.jsonl")
ap.add_argument("--sets_dir", default="data/agastya")
ap.add_argument("--out", default="/data/runs/viewer_tok/tok.jsonl")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

abl = {(s["b"], s["kind"], s["k"]): s for s in json.load(open(f"{args.sets_dir}/ablation_sets.json")).values()}
shared = json.load(open(f"{args.sets_dir}/shared_sets.json"))

def sets_for(b):
    out = {}
    if (b, "top", 100) in abl:
        out["own100"] = abl[(b, "top", 100)]["spec"]
        out["own100_rand"] = abl[(b, "rand", 100)]["spec"]
    out["shared100"] = shared["sharedmid100"]["spec"]
    out["shared100_rand"] = shared["rand_sharedmid100"]["spec"]
    return out

def top10(b):
    s = abl.get((b, "top", 10))
    return [(int(L), n) for L, ns in s["spec"].items() for n in ns] if s else []

items = [json.loads(l) for l in open(args.items)]
os.makedirs(os.path.dirname(args.out), exist_ok=True)
done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["rid"] for l in open(args.out)}
todo = [it for it in items if it["rid"] not in done]
if args.limit:
    todo = todo[:args.limit]
print(f"{len(done)} done, {len(todo)} to do", flush=True)

H = Harness()
layers = H.layers

def zero_hooks(spec):
    hs = []
    for L, ns in spec.items():
        idx = torch.tensor(ns, device="cuda")
        def pre(mod, a, idx=idx):
            x = a[0].clone()
            x[..., idx] = 0
            return (x,) + tuple(a[1:])
        hs.append(layers[int(L)].mlp.down_proj.register_forward_pre_hook(pre))
    return hs

@torch.no_grad()
def run(ids, P, R, spec=None, cap=None):
    hs = zero_hooks(spec) if spec else []
    if cap:
        for L, ns in cap["by_layer"].items():
            idx = torch.tensor(ns, device="cuda")
            def grab(mod, a, L=L, idx=idx):
                cap["vals"][L] = a[0][0, P - 1:P + R - 1][:, idx].float().cpu()
            hs.append(layers[L].mlp.down_proj.register_forward_pre_hook(grab))
    try:
        logits = H.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).logits[0, P - 1:P + R - 1]
    finally:
        for h in hs:
            h.remove()
    return torch.log_softmax(logits.float(), -1)

t0 = time.time()
with open(args.out, "a") as fh:
    for k, it in enumerate(todo):
        ids = it["prompt_ids"] + it["ids"]
        P, R = len(it["prompt_ids"]), len(it["ids"])
        tgt = torch.tensor(it["ids"], device="cuda")
        neurons = top10(it["b"])
        by_layer = {}
        for L, n in neurons:
            by_layer.setdefault(L, []).append(n)
        cap = {"by_layer": by_layer, "vals": {}} if neurons else None
        base = run(ids, P, R, cap=cap)
        lp = base.gather(1, tgt[:, None])[:, 0]
        tp, ti = base.exp().topk(5, -1)
        rec = {"rid": it["rid"], "lp": [round(x, 3) for x in lp.tolist()],
               "top": [[a, [round(p, 4) for p in b]] for a, b in zip(ti.tolist(), tp.tolist())], "sets": {}}
        if cap:
            cols = {}
            for L, ns in by_layer.items():
                for c, n in enumerate(ns):
                    cols[(L, n)] = cap["vals"][L][:, c]
            rec["neurons"] = [[L, n] for L, n in neurons]
            rec["acts"] = [[round(float(cols[(L, n)][j]), 3) for L, n in neurons] for j in range(R)]
        for name, spec in sets_for(it["b"]).items():
            ab = run(ids, P, R, spec=spec)
            dlp = ab.gather(1, tgt[:, None])[:, 0] - lp
            kl = (base.exp() * (base - ab)).sum(-1).clamp_min(0)
            rec["sets"][name] = {"dlp": [round(x, 3) for x in dlp.tolist()], "kl": [round(x, 4) for x in kl.tolist()]}
            del ab
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if k % 20 == 0:
            el = time.time() - t0
            print(f"[{k+1}/{len(todo)}] {it['b']} R={R} {el/60:.1f} min, eta {el/(k+1)*(len(todo)-k-1)/60:.1f} min", flush=True)
print("ALL_DONE", flush=True)
