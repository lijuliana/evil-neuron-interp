"""A0: S1 screen capture over all Qwen3.6-27B WeirdChat transcripts.

For each behavior parquet: teacher-force every transcript (assistant truncated to 256 tokens) and
accumulate Welford mean/M2 of all 64x17408 neuron post-activations at POSITIONS, grouped by
  (behavior_id, match)            -> s1/behavior/{behavior}__{matched|unmatched}.npz
  (behavior_id, shuffled match)   -> s1/shuffled/{behavior}__{matched|unmatched}.npz   (labels permuted within behavior, seed 0)
  (prompt_id, match) for prompts with >=8 matched and >=8 unmatched
                                  -> s1/prompt/{behavior}__{prompt_id[:16]}__{matched|unmatched}.npz
Then one forward per unique prompt -> s1/prompt_level_acts.npy [n_prompts, 64, 17408] (last prompt token)
with s1/prompt_index.parquet (behavior_id, prompt_id, n_samples, n_matched, match_rate, row).
Resumable: behaviors whose outputs exist are skipped. Manifest: s1/manifest.json.

Run on VM:  python -I scripts/03_a0_s1_capture.py --data /data/weirdchat --out /data/captures/s1
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd, torch
from harness import Harness, GroupAcc, POSITIONS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/data/weirdchat")
ap.add_argument("--out", default="/data/captures/s1")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max-asst", type=int, default=256)
ap.add_argument("--min-per-class", type=int, default=8)
ap.add_argument("--limit", type=int, default=0, help="debug: max transcripts per behavior")
ap.add_argument("--skip-prompt-level", action="store_true")
args = ap.parse_args()

for sub in ("behavior", "shuffled", "prompt"):
    os.makedirs(f"{args.out}/{sub}", exist_ok=True)
manifest_path = f"{args.out}/manifest.json"
manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"positions": POSITIONS, "groups": [],
    "notes": "positions last_prompt and asst_0 are identical across rollouts of a prompt (deterministic prefill): "
             "do not use them for within-prompt contrasts; behavior-level contrasts at these positions are confounded by prompt identity."}
done = {(g["kind"], g["behavior"], g.get("prompt_id"), g["label"]) for g in manifest["groups"]}

def save_manifest():
    json.dump(manifest, open(manifest_path, "w"), indent=1)

def add(kind, behavior, label, n, path, prompt_id=None, **extra):
    manifest["groups"].append(dict(kind=kind, behavior=behavior, label=label, prompt_id=prompt_id, n=int(n),
                                   positions=POSITIONS, file=os.path.relpath(path, args.out), **extra))

h = Harness()
files = sorted(glob.glob(f"{args.data}/transcripts/qwen__qwen3.6-27b/*.parquet"))
T0 = time.time(); total_tr = 0; total_tok = 0
for f in files:
    behavior = os.path.basename(f).rsplit("-0000", 1)[0]
    if ("behavior", behavior, None, "matched") in done:
        print(f"[s1] skip {behavior} (done)"); continue
    df = pd.read_parquet(f)
    if args.limit: df = df.iloc[:args.limit]
    df["match"] = df["judgment"].apply(lambda j: bool(j["match"]))
    rng = np.random.default_rng(0)
    df["shuf"] = rng.permutation(df["match"].values)
    counts = df.groupby("prompt_id")["match"].agg(["sum", "count"])
    eligible = set(counts.index[(counts["sum"] >= args.min_per_class) & ((counts["count"] - counts["sum"]) >= args.min_per_class)])
    acc = {k: GroupAcc("cuda") for k in ("m", "u", "sm", "su")}
    n_skipped = 0; t_b = time.time(); tok_b = 0
    for pid, g in df.groupby("prompt_id", sort=True):
        items = []
        for _, r in g.iterrows():
            p, a = h.transcript_ids(list(r["messages"]), args.max_asst)
            if len(a) == 0:
                n_skipped += 1; continue
            items.append(((p, a), bool(r["match"]), bool(r["shuf"])))
        items.sort(key=lambda it: len(it[0][0]) + len(it[0][1]))
        pacc = {k: GroupAcc("cuda") for k in ("m", "u")} if pid in eligible else None
        for s in range(0, len(items), args.batch):
            chunk = items[s:s + args.batch]
            x, valid = h.capture_batch([it[0] for it in chunk])
            tok_b += sum(len(p) + len(a) for (p, a), _, _ in chunk)
            m = torch.tensor([it[1] for it in chunk]); sm = torch.tensor([it[2] for it in chunk])
            for key, mask in (("m", m), ("u", ~m), ("sm", sm), ("su", ~sm)):
                if mask.any():
                    acc[key].update(x[mask.cuda()], valid[mask])
            if pacc is not None:
                for key, mask in (("m", m), ("u", ~m)):
                    if mask.any():
                        pacc[key].update(x[mask.cuda()], valid[mask])
            del x
        if pacc is not None:
            for key, label in (("m", "matched"), ("u", "unmatched")):
                path = f"{args.out}/prompt/{behavior}__{pid[:16]}__{label}.npz"
                pacc[key].to_npz(path, behavior=behavior, prompt_id=pid, label=label)
                add("prompt", behavior, label, pacc[key].count[0], path, prompt_id=pid)
            del pacc
    for key, kind, label in (("m", "behavior", "matched"), ("u", "behavior", "unmatched"),
                             ("sm", "shuffled", "matched"), ("su", "shuffled", "unmatched")):
        path = f"{args.out}/{kind}/{behavior}__{label}.npz"
        acc[key].to_npz(path, behavior=behavior, label=label, kind=kind)
        add(kind, behavior, label, acc[key].count[0], path, n_skipped_empty=n_skipped,
            n_eligible_prompts=len(eligible), counts_per_position=acc[key].count.tolist())
    save_manifest()
    dt = time.time() - t_b; total_tr += len(df); total_tok += tok_b
    print(f"[s1] {behavior}: {len(df)} transcripts ({int(df.match.sum())} matched), {len(eligible)} eligible prompts, "
          f"{n_skipped} empty skipped, {tok_b} tok in {dt:.0f}s = {tok_b/dt:.0f} tok/s, {len(df)/dt:.2f} tr/s; "
          f"elapsed {(time.time()-T0)/60:.1f} min", flush=True)
    del acc; torch.cuda.empty_cache()

if not args.skip_prompt_level and not os.path.exists(f"{args.out}/prompt_level_acts.npy"):
    rows = []
    for f in files:
        behavior = os.path.basename(f).rsplit("-0000", 1)[0]
        df = pd.read_parquet(f)
        df["match"] = df["judgment"].apply(lambda j: bool(j["match"]))
        for pid, g in df.groupby("prompt_id", sort=True):
            rows.append(dict(behavior_id=behavior, prompt_id=pid, n_samples=len(g), n_matched=int(g.match.sum()),
                             match_rate=float(g.match.mean()), prompt_messages=list(g.iloc[0]["messages"])[:-1]))
    idx = pd.DataFrame(rows); idx["row"] = np.arange(len(idx))
    t0 = time.time()
    acts = h.capture_prompt_level([list(m) for m in idx["prompt_messages"]], batch_size=args.batch)
    np.save(f"{args.out}/prompt_level_acts.npy", acts)
    idx.drop(columns=["prompt_messages"]).to_parquet(f"{args.out}/prompt_index.parquet", index=False)
    manifest["prompt_level"] = dict(file="prompt_level_acts.npy", index="prompt_index.parquet", n_prompts=len(idx),
                                    position="last_prompt", shape=list(acts.shape), dtype="float32")
    save_manifest()
    print(f"[s1] prompt-level: {len(idx)} prompts in {time.time()-t0:.0f}s -> {acts.shape}", flush=True)
print(f"[s1] DONE: {total_tr} transcripts, {total_tok} tokens, {(time.time()-T0)/60:.1f} min total", flush=True)
