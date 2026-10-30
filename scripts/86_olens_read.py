"""Bulk O-lens reads of the vectors captured by scripts/85, with the lens merged for speed. Order: neuron directions
(+w and -w, read at the nearest trained s3d layer at or above the neuron's layer, capped at 60), then the residuals at
layer 56, 44, 32. Two samples each, the lens contract's sampling. Resumable: appends to /data/runs/olens/reads.jsonl
and skips keys already there; stops after --max_minutes so the GPU lock is released between ~1 h chunks.
Run on the VM (repeat until it prints "all done"):
  flock /data/gpu.lock ~/olens/venv/bin/python -I scripts/86_olens_read.py --lens s3d_space_rl600
"""
import argparse, json, os, sys, time, zlib
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import olens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lens", required=True)
ap.add_argument("--dir", default="/data/runs/olens")
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--batch", type=int, default=60, help="vectors per generate call (rows = batch * n)")
ap.add_argument("--max_minutes", type=float, default=55)
args = ap.parse_args()
t_start = time.time()

d = torch.load(f"{args.dir}/resid.pt")
c = olens.LENSES[args.lens]

def read_layer(L):  # a neuron at layer L writes into every residual from block L on
    return next((l for l in c.layers if l >= L), c.layers[-1])

work = []  # (key, layer, vector)
for (L, n, src), w in zip(d["nkeys"], d["W"]):
    for sign in (1, -1):
        work.append((f"N|{L}|{n}|{src}|{sign:+d}", read_layer(L), sign * w))
for j in reversed(range(len(d["layers"]))):
    for (sid, cc), h in zip(d["keys"], d["H"][:, j]):
        work.append((f"R|{sid}|{cc}|{d['layers'][j]}", d["layers"][j], h))

out = f"{args.dir}/reads.jsonl"
done = set()
if os.path.exists(out):
    done = {json.loads(l)["key"] for l in open(out)}
todo = [w for w in work if w[0] not in done]
print(f"{len(work)} vectors, {len(done)} done, {len(todo)} to read", flush=True)
if not todo:
    print("all done", flush=True)
    sys.exit(0)

lens = olens.merge(olens.load_lens(args.lens))
print(f"lens {args.lens} merged, {time.time() - t_start:.0f}s", flush=True)
nread, t0 = 0, time.time()
with open(out, "a") as f:
    i = 0
    while i < len(todo):
        L = todo[i][1]
        chunk = [w for w in todo[i:i + args.batch] if w[1] == L]
        texts = olens.read(lens, torch.stack([w[2] for w in chunk]).float(), L, n=args.n,
                           seed=zlib.crc32(chunk[0][0].encode()))
        for (key, layer, _), t in zip(chunk, texts):
            f.write(json.dumps({"key": key, "layer": layer, "lens": args.lens, "texts": t}) + "\n")
        f.flush()
        i += len(chunk)
        nread += len(chunk) * args.n
        if (i // args.batch) % 10 == 0:
            print(f"{i}/{len(todo)} vectors, {nread / (time.time() - t0):.1f} readouts/s", flush=True)
        if time.time() - t_start > args.max_minutes * 60:
            print(f"time limit: {i}/{len(todo)} this chunk", flush=True)
            sys.exit(0)
print("all done", flush=True)
