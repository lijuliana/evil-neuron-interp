"""Positions for the bulk O-lens run (PLAN.md decisions 1, 4, 8, 11). Writes data/olens/seqs.jsonl, one record per
sequence to capture, and data/olens/neurons.jsonl, the neuron directions to label.

Sequence kinds (c = reply tokens in the prefix; the residual read for cut c sits at abs position P + c - 1, the
state that picks reply token c; c = 0 is the last prompt token):
  prompt    all 403 prompts, c = 0 (the prompt-level target, decision 1c)
  reply     the 410 fork replies: every measured cut point c >= 1 (6,095 points incl. c = 0), plus for the 46 sharp
            Phase 2 forks a dense window c_fork-40 .. c_fork+20 (every token)
  raw       echo control for the 46 sharp forks: the same user text and reply tokens as a plain document (no chat
            template), same window
  abl_top / abl_rand   the 41 sharp forks whose behavior has its own sets: zero the behavior's top-100 (or its
            matched random-100) at down_proj input at every position, window c_fork-10 .. c_fork+5
Neurons: each own top-10 set (5 behaviors) and sharedmid100, plus one same-layer random neuron per entry (seed 0);
each read as +w and -w, w = down_proj[:, n].
"""
import gzip, json, os, random, sys
from transformers import AutoTokenizer

A, OUT = "data/agastya", "data/olens"
DENSE, ABL = (40, 20), (10, 5)
tok = AutoTokenizer.from_pretrained(sys.argv[1] if len(sys.argv) > 1 else "data/tokenizer")
gen = {r["prompt_id"]: r for r in map(json.loads, gzip.open(f"{A}/stage_a_gen.jsonl.gz", "rt"))}
forks = [json.loads(l) for l in open(f"{A}/forks_off.jsonl")]
sharp = {json.loads(l)["rid"]: json.loads(l) for l in open(f"{A}/phase2_forks.jsonl")}
abl = json.load(open(f"{A}/ablation_sets.json"))
os.makedirs(OUT, exist_ok=True)
seqs = []

def rec(sid, kind, pid, rid, ids, P, cs, **kw):
    cs = sorted(set(cs))
    return {"sid": sid, "kind": kind, "pid": pid, "rid": rid, "b": gen[pid]["behavior_id"], "split": gen[pid]["split"],
            "ids": ids[:P + max(cs)], "c": cs, "pos": [P + c - 1 for c in cs], **kw}

for pid, g in gen.items():
    seqs.append(rec(f"prompt|{pid}", "prompt", pid, None, g["prompt_ids"], len(g["prompt_ids"]), [0]))
for f in forks:
    pid, i = f["rid"].split("|")
    g, reply = gen[pid], gen[pid]["samples"][int(i)]["ids"]
    P, n = len(g["prompt_ids"]), len(reply)
    cs = [c for c, _ in f["curve"] if c >= 1]
    sp = sharp.get(f["rid"])
    if sp:
        c0 = sp["fork"][0]
        cs += range(max(1, c0 - DENSE[0]), min(n, c0 + DENSE[1]) + 1)
    seqs.append(rec(f"reply|{f['rid']}", "reply", pid, f["rid"], g["prompt_ids"] + reply, P, cs))
    if not sp:
        continue
    text = tok.decode(g["prompt_ids"])
    user = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    raw = tok(user + "\n\n", add_special_tokens=False).input_ids
    win = range(max(0, c0 - DENSE[0]), min(n, c0 + DENSE[1]) + 1)
    seqs.append(rec(f"raw|{f['rid']}", "raw", pid, f["rid"], raw + reply, len(raw), win))
    b = g["behavior_id"]
    if f"{b}|top100" in abl:
        win = range(max(0, c0 - ABL[0]), min(n, c0 + ABL[1]) + 1)
        for kind, key in (("abl_top", f"{b}|top100"), ("abl_rand", f"{b}|rand100")):
            assert abl[key]["k"] == 100 and abl[key]["b"] == b
            seqs.append(rec(f"{kind}|{f['rid']}", kind, pid, f["rid"], g["prompt_ids"] + reply, P, win,
                            abl={int(L): ns for L, ns in abl[key]["spec"].items()}))

with open(f"{OUT}/seqs.jsonl", "w") as fh:
    for s in seqs:
        fh.write(json.dumps(s) + "\n")

shared = json.load(open(f"{A}/shared_sets.json"))
rng = random.Random(0)
neurons, seen = [], set()
groups = [(f"top10|{s['b']}", s["spec"]) for s in abl.values() if s["kind"] == "top" and s["k"] == 10]
groups.append(("sharedmid100", shared["sharedmid100"]["spec"]))
for src, spec in groups:
    for L, ns in spec.items():
        for n in ns:
            for grp, nn in ((src, n), (f"rand|{src}", rng.randrange(17408))):
                if (int(L), nn, grp) not in seen:
                    seen.add((int(L), nn, grp))
                    neurons.append({"L": int(L), "n": nn, "src": grp})
with open(f"{OUT}/neurons.jsonl", "w") as fh:
    for x in neurons:
        fh.write(json.dumps(x) + "\n")

from collections import Counter
npos = Counter()
for s in seqs:
    npos[s["kind"]] += len(s["pos"])
reads = sum(npos.values()) * 3 * 2 + len(neurons) * 2 * 2
print("sequences", Counter(s["kind"] for s in seqs), "\npositions", dict(npos), "total", sum(npos.values()))
print("neurons", len(neurons), "| readouts at 3 layers x 2 samples, neurons x2 signs x2:", reads)
