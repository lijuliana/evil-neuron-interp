"""Does merging the s3d LoRA change readouts, and how much faster is it? Reads the validation vectors saved by
scripts/81 (/data/runs/olens_val/resid_L56.pt) with olens_s3d_rl600 merged into the base, training carrier,
2 samples, and appends them to readouts.jsonl as condition "s3d_merged" for the match test (scripts/82).
Also times batch sizes 120 and 240 rows (60 and 120 vectors x 2).
Run on the VM: flock /data/gpu.lock ~/olens/venv/bin/python -I scripts/84_olens_merge_check.py
"""
import json, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import olens  # noqa: E402

D = "/data/runs/olens_val"
items = [json.loads(l) for l in open("data/olens_val/items.jsonl")]
H = torch.load(f"{D}/resid_L56.pt")
lens = olens.merge(olens.load_lens("s3d_rl600"))
keys = [(it["id"], k) for it in items for k in range(len(it["pos"]))]
for batch in (60, 120):
    torch.cuda.synchronize(); t0, outs = time.time(), []
    for b in range(0, len(H), batch):
        outs += olens.read(lens, H[b:b + batch], 56, n=2, seed=b)
    dt = time.time() - t0
    print(f"[merged batch {batch * 2} rows] {len(H) * 2} readouts in {dt:.0f}s ({len(H) * 2 / dt:.1f}/s), "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB", flush=True)
    if batch == 60:
        with open(f"{D}/readouts.jsonl", "a") as f:
            for (rid, k), texts in zip(keys, outs):
                f.write(json.dumps({"cond": "s3d_merged", "id": rid, "k": k, "texts": texts}) + "\n")
toks = [len(lens.tok(t).input_ids) for o in outs for t in o]
print(f"readout length: median {sorted(toks)[len(toks) // 2]}, at cap 256: {sum(t >= 255 for t in toks)}/{len(toks)}")
