"""O-lens validation reads: block-56 residuals at the 120 positions of data/olens_val/items.jsonl, read by four
conditions, 2 samples each (seed 0):
  s3d_wrapped  olens_s3d_rl600 with Agastya's carrier (PROMPT_WRAPPED): reproduces his viewer
  s3d          olens_s3d_rl600 with the single-line training carrier
  space        olens_s3d_space_rl600, training carrier
  lensB        final_olens lens_rl_step600, training carrier (alpha 1000, its own sampling)
Residuals are captured once (adapter off) and saved, so every lens reads the same vectors.
Run on the VM: flock /data/gpu.lock ~/olens/venv/bin/python -I scripts/81_olens_val_read.py
"""
import argparse, gc, json, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import olens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--items", default="data/olens_val/items.jsonl")
ap.add_argument("--out", default="/data/runs/olens_val")
ap.add_argument("--layer", type=int, default=56)
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--batch", type=int, default=60)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
items = [json.loads(l) for l in open(args.items)]
CONDS = [("s3d_rl600", [("s3d_wrapped", olens.PROMPT_WRAPPED), ("s3d", olens.PROMPT)]),
         ("s3d_space_rl600", [("space", olens.PROMPT)]),
         ("lensB_rl600", [("lensB", olens.PROMPT)])]

vec_path = f"{args.out}/resid_L{args.layer}.pt"
H = None
with open(f"{args.out}/readouts.jsonl", "w") as f:
    for name, conds in CONDS:
        t0 = time.time()
        lens = olens.load_lens(name)
        print(f"[{name}] loaded in {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
        if H is None:
            H = torch.cat([olens.capture(lens, it["ids"], [args.layer], it["pos"])[args.layer] for it in items])
            torch.save(H, vec_path)
            print(f"captured {tuple(H.shape)}, norms {H.norm(dim=-1).min():.0f}-{H.norm(dim=-1).max():.0f}", flush=True)
        keys = [(it["id"], k) for it in items for k in range(len(it["pos"]))]
        for cond, prompt in conds:
            t0, outs = time.time(), []
            for b in range(0, len(H), args.batch):
                outs += olens.read(lens, H[b:b + args.batch], args.layer, n=args.n, seed=b, prompt=prompt)
            dt = time.time() - t0
            for (rid, k), texts in zip(keys, outs):
                f.write(json.dumps({"cond": cond, "id": rid, "k": k, "texts": texts}) + "\n")
            f.flush()
            print(f"[{cond}] {len(H) * args.n} readouts in {dt:.0f}s ({len(H) * args.n / dt:.1f}/s); e.g. {outs[0][0][:160]!r}",
                  flush=True)
        del lens
        gc.collect()
        torch.cuda.empty_cache()
