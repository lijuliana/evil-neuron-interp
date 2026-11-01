"""Eval-A M1 (b): O-lens reads of the matched-continuation positions, scripts/86's settings and schema, from the
Stage-1 capture (scripts/56): s3d_rl600 merged, training carrier (olens.PROMPT), 2 samples, the lens contract's
sampling (T 0.8, top-p 0.95, top-k 0, 256 tokens), blocks 56, 44, 32 in that order, 60 vectors per generate call
seeded with zlib.crc32 of the call's first key (as scripts/86). Positions: every unique (b) position (canon[i] == sid;
index.jsonl set 'b'), never bh. Keys: R|<sid>|<pos>|<layer> (pos = absolute token position, not c).
Resumable: appends to <out>/reads.jsonl {key, layer, lens, texts, sid, pos} and skips keys already there.
--repro: before the bulk run, re-read scripts/86's first L56 chunk (60 (a) positions, seed crc32 of its first key)
twice, from the original vectors (data/olens/resid.pt rows, passed in as --repro_chunk) and from this capture's
states at the same positions, into <out>/repro.jsonl, to compare with runs/olens/reads.jsonl.
Run on the VM: ~/olens/venv/bin/python -I scripts/86b_olens_reads_b.py [--repro --repro_chunk F]
"""
import argparse, json, os, sys, time, zlib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import olens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lens", default="s3d_rl600")
ap.add_argument("--cap", default="/data/runs/monitors/capture")
ap.add_argument("--out", default="/data/runs/monitors/olens_b")
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--batch", type=int, default=60, help="vectors per generate call (rows = batch * n)")
ap.add_argument("--layers", default="56,44,32")
ap.add_argument("--max_minutes", type=float, default=600)
ap.add_argument("--repro", action="store_true")
ap.add_argument("--repro_chunk", default="")
args = ap.parse_args()
t_start = time.time()
os.makedirs(args.out, exist_ok=True)
idx = [json.loads(l) for l in open(f"{args.cap}/index.jsonl")]


def log(*a):
    print(f"[{(time.time() - t_start) / 60:5.1f}m]", *a, flush=True)


if args.repro:
    ch = torch.load(args.repro_chunk, weights_only=True)
    keys = [f"R|{s}|{c}|56" for s, c in ch["keys"]]
    rowc = {(m["sid"], m["c"]): m["row"] for m in idx if m["set"] == "a"}
    R56 = np.load(f"{args.cap}/resid_56.npy", mmap_mode="r")
    mine = torch.tensor(np.stack([R56[rowc[(s, c)]] for s, c in ch["keys"]]).astype(np.float32))
    lens = olens.merge(olens.load_lens(args.lens))
    with open(f"{args.out}/repro.jsonl", "w") as f:
        for src, H in (("orig_resid", ch["H56"].float()), ("capture", mine)):
            texts = olens.read(lens, H, 56, n=args.n, seed=zlib.crc32(keys[0].encode()))
            for k, t in zip(keys, texts):
                f.write(json.dumps({"key": k, "src": src, "layer": 56, "lens": args.lens, "texts": t}) + "\n")
            log(f"repro {src}: {len(texts)} vectors")
    sys.exit(0)

layers = [int(x) for x in args.layers.split(",")]
rows = [m for m in idx if m["set"] == "b"]
R = {L: np.load(f"{args.cap}/resid_{L}.npy", mmap_mode="r") for L in layers}
work = [(f"R|{m['sid']}|{m['pos']}|{L}", L, m) for L in layers for m in rows]
out = f"{args.out}/reads.jsonl"
done = set()
if os.path.exists(out):
    done = {json.loads(l)["key"] for l in open(out)}
todo = [w for w in work if w[0] not in done]
log(f"{len(work)} vectors ({len(rows)} positions x {layers}), {len(done)} done, {len(todo)} to read")
if not todo:
    log("all done")
    sys.exit(0)

lens = olens.merge(olens.load_lens(args.lens))
log(f"lens {args.lens} merged")
nread, t0 = 0, time.time()
with open(out, "a") as f:
    i = 0
    while i < len(todo):
        L = todo[i][1]
        chunk = [w for w in todo[i:i + args.batch] if w[1] == L]
        H = torch.tensor(np.stack([R[L][w[2]["row"]] for w in chunk]).astype(np.float32))
        texts = olens.read(lens, H, L, n=args.n, seed=zlib.crc32(chunk[0][0].encode()))
        for (key, layer, m), t in zip(chunk, texts):
            f.write(json.dumps({"key": key, "layer": layer, "lens": args.lens, "texts": t, "sid": m["sid"],
                                "pos": m["pos"]}) + "\n")
        f.flush()
        i += len(chunk)
        nread += len(chunk) * args.n
        if (i // args.batch) % 10 == 0:
            log(f"{i}/{len(todo)} vectors, {nread / (time.time() - t0):.1f} readouts/s")
        if time.time() - t_start > args.max_minutes * 60:
            log(f"time limit: {i}/{len(todo)} this chunk")
            sys.exit(0)
log("all done")
