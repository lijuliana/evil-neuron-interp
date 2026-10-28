"""A0: harness unit tests on the VM. Run: python -I scripts/02_a0_harness_test.py --data /data/weirdchat"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd, torch
from harness import Harness, POSITIONS, N_POS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/data/weirdchat")
ap.add_argument("--out", default="captures")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

h = Harness(resid_norms_path=f"{args.out}/qwen3.6-27b_resid_norms.json")

# --- 1. generation throughput (warmup then measure)
print("== warmup"); h.chat(["Say hi."], n=2, max_new_tokens=8, verbose=False)
print("== gen throughput: 16 prompts x 4 = 64 seqs, 128 new tokens")
r = h.chat([f"Tell me a fun fact about the number {i}." for i in range(16)], n=4, max_new_tokens=128, batch_size=64)
print("sample:", repr(r[0][0][:200]))

# --- 2. calibrate
print("== calibrate")
med = h.calibrate(save_path=f"{args.out}/qwen3.6-27b_resid_norms.json")
print("median resid norms L0,L1,L15,L31,L47,L63:", [round(med[i], 2) for i in (0, 1, 15, 31, 47, 63)])

# --- 3. steering: dose 0 identity, dose 1 changes logits; clamp changes activation
ids = torch.tensor([h.prompt_ids("Who are you?")], device="cuda")
with torch.no_grad():
    base = h.lm(input_ids=ids, use_cache=False).last_hidden_state.float()
    with h.steer([(30, 100, 0.0)]):
        z = h.lm(input_ids=ids, use_cache=False).last_hidden_state.float()
    with h.steer([(30, 100, 1.0)]):
        s1 = h.lm(input_ids=ids, use_cache=False).last_hidden_state.float()
    # expected delta at the hooked layer output
    w = h.layers[30].mlp.down_proj.weight[:, 100].float()
    print("dose-0 identity (max abs diff):", float((z - base).abs().max()), "| dose-1 diff:", float((s1 - base).abs().max()),
          "| ||delta|| =", float((1.0 * med[30] / w.norm() * w).norm()), "vs resid norm", round(med[30], 2))
    assert float((z - base).abs().max()) == 0.0
    assert float((s1 - base).abs().max()) > 0
    # clamp check: read the down_proj input under clamp
    got = {}
    def rd(mod, a, out): got["x"] = a[0][0, -1, 100].item()  # forward hooks see pre-hook-modified inputs
    hh = h.layers[30].mlp.down_proj.register_forward_hook(rd)
    h.lm(input_ids=ids, use_cache=False); v0 = got["x"]
    with h.clamp(30, 100, 3.0):
        h.lm(input_ids=ids, use_cache=False)
    hh.remove()
    print("clamp: unclamped act", v0, "-> clamped", got["x"])
    assert abs(got["x"] - 3.0) < 0.05
print("steer-ish:", h.chat(["Who are you?"], n=1, max_new_tokens=40, verbose=False)[0][0][:150])
with h.steer([(30, 100, 1.0)]):
    print("steered L30N100 +1.0:", h.chat(["Who are you?"], n=1, max_new_tokens=40, verbose=False)[0][0][:150])

# --- 4. capture positions on 2 transcripts
df = pd.read_parquet(f"{args.data}/transcripts/qwen__qwen3.6-27b/recommends-drunk-driving-0000.parquet")
rows = [df.iloc[0], df.iloc[1]]
seqs = [h.transcript_ids(list(r["messages"]), 256) for r in rows]
for (p, a) in seqs:
    print("prompt tail tokens:", [h.tok.decode([t]) for t in p[-6:]])
    print("  last_prompt token:", repr(h.tok.decode([p[-1]])), "| asst_0..8:", [h.tok.decode([t]) for t in a[:9]], "| n_asst", len(a))
x, valid = h.capture_batch(seqs)
print("capture shape", tuple(x.shape), "valid", valid.tolist())
# equivalence: single-sequence capture of seq 1 vs batched (right padding) -- should match closely
x1, _ = h.capture_batch([seqs[1]])
print("batched vs single max abs diff (L0..63, all pos):", float((x[1] - x1[0]).abs().max()), "scale", float(x1.abs().mean()))

# --- 4b. prompt-level capture equals last_prompt from the transcript capture (same prompt)
pl = h.capture_prompt_level([list(r["messages"])[:-1] for r in rows])
print("prompt-level vs transcript last_prompt max abs diff:", float(np.abs(pl[1] - x[1, 0].cpu().numpy()).max()))
print("same prompt, two rollouts: last_prompt identical?", float((x[0, 0] - x[1, 0]).abs().max()) if rows[0]["prompt_id"] == rows[1]["prompt_id"] else "different prompts")

# --- 5. capture throughput on 200 transcripts, batch 16
sub = df.iloc[:200]
seqs = [h.transcript_ids(list(m), 256) for m in sub["messages"]]
seqs.sort(key=lambda s: len(s[0]) + len(s[1]))
t0 = time.time(); ntok = 0
for i in range(0, len(seqs), 16):
    b = seqs[i:i + 16]; h.capture_batch(b); ntok += sum(len(p) + len(a) for p, a in b)
torch.cuda.synchronize(); dt = time.time() - t0
print(f"capture: 200 transcripts, {ntok} tokens, {dt:.1f}s -> {200/dt:.2f} transcripts/s, {ntok/dt:.0f} tok/s; peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print("ALL TESTS PASSED")
