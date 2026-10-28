"""A0: baseline replication. For each 27B behavior pick 20 prompts (highest shipped matched fraction),
sample 64 responses each in bf16 with WeirdChat settings, save {out}/{behavior}.jsonl
(prompt_id, sample_idx, prompt, response, ref_n_matched, ref_n_samples). Resumable per behavior.
Run on VM: python -I scripts/05_a0_baseline.py --data /data/weirdchat --out /data/runs/baseline_27b
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd
from harness import Harness

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/data/weirdchat")
ap.add_argument("--out", default="/data/runs/baseline_27b")
ap.add_argument("--n-prompts", type=int, default=20)
ap.add_argument("--n-samples", type=int, default=64)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--max-new", type=int, default=400)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
h = Harness()
T0 = time.time()
for f in sorted(glob.glob(f"{args.data}/prompts/qwen__qwen3.6-27b/*.parquet")):
    behavior = os.path.basename(f)[:-8]
    outp = f"{args.out}/{behavior}.jsonl"
    if os.path.exists(outp):
        print(f"[baseline] skip {behavior}"); continue
    p = pd.read_parquet(f)
    p["n_matched"] = p.match_summary.apply(lambda s: s["n_matched"]); p["n_samples"] = p.match_summary.apply(lambda s: s["n_samples"])
    p["frac"] = p.n_matched / p.n_samples
    p = p[p.messages.apply(len) == 1]  # single-turn prompts only
    p = p.sort_values(["frac", "n_matched"], ascending=False).head(args.n_prompts)
    prompts = [[dict(m) for m in msgs] for msgs in p.messages]
    t0 = time.time()
    outs = h.chat(prompts, n=args.n_samples, max_new_tokens=args.max_new, temperature=1.0, top_p=0.95, top_k=20,
                  batch_size=args.batch, seed=0)
    with open(outp, "w") as fo:
        for (_, r), resp in zip(p.iterrows(), outs):
            for i, text in enumerate(resp):
                fo.write(json.dumps(dict(prompt_id=r.prompt_id, sample_idx=i, prompt=r.messages[0]["content"], response=text,
                                         ref_n_matched=int(r.n_matched), ref_n_samples=int(r.n_samples),
                                         sampling=dict(temperature=1.0, top_p=0.95, top_k=20, max_new_tokens=args.max_new, dtype="bf16"))) + "\n")
    print(f"[baseline] {behavior}: {len(p)} prompts x {args.n_samples} in {time.time()-t0:.0f}s (ref matched frac mean {p.frac.mean():.2f}); elapsed {(time.time()-T0)/60:.1f} min", flush=True)
print("[baseline] DONE")
