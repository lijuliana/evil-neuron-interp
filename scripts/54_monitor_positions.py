"""Eval A monitors: shared positions, folds and targets for every method (runs/monitors/SPEC.md).

(a) cut points and sharp-fork windows reuse data/olens/seqs.jsonl (kinds prompt / reply); their residuals at blocks
32/44/56 are in data/olens/resid.pt. This script writes to runs/monitors/:
  folds.json      prompt_id -> fold 0..4. Folds group whole patterns (sibling prompts share a pattern), stratified by
                  behavior: within each behavior, patterns (shuffled with default_rng(1), then largest first) go to the
                  fold with the fewest of that behavior's prompts. Every supervised fit uses these folds.
  p_pooled.json   "rid|c" -> [P, n]: P(weird | prefix) pooled over every continuation sampled at integer cut c
                  (batches 0, 1 and the 24 'confirm' samples at the fork cuts); forks_off curves leave out 'confirm'
  seqs_b.jsonl    (b) matched continuations: one record per sampled continuation, ids = prompt + reply[:cut] +
                  cont[:4], positions P + cut + k - 1 for k = 1, 4 (k = 1 is the state at the first continuation
                  token). canon[i] = sid of the first record whose ids[:pos_i + 1] is identical (score once, copy)
  cuts_b.jsonl    the selected cuts with weird / normal counts and the pair-tie ceiling at k = 1 and 4
  seqs_bh.jsonl, cuts_bh.jsonl   the same for the held-out denying-AI cuts (forks_samples_on_dai_held.jsonl,
                  thinking-on labels): every qualifying cut, used only by the cheap monitors as an out-of-sample check
Selection: integer cut (no forced token), >= 2 weird and >= 2 normal continuations of >= 4 tokens; every sharp
Phase 2 fork cut (c_fork) that qualifies, then random qualifying cuts stratified by behavior up to N_CUTS, at most 3
per reply; per cut up to 4 weird + 4 normal continuations, preferring distinct first-4-token prefixes within a class.
"""
import gzip, json, os, random
from collections import defaultdict
import numpy as np

A, OUT = "data/agastya", "runs/monitors"
N_CUTS, PER_RID, K, PER_CLASS, MIN_LEN = 500, 3, (1, 4), 4, 4
os.makedirs(OUT, exist_ok=True)
gen = {r["prompt_id"]: r for r in map(json.loads, gzip.open(f"{A}/stage_a_gen.jsonl.gz", "rt"))}
split = json.load(open(f"{A}/split.json"))
sharp = {json.loads(l)["rid"]: json.loads(l)["fork"][0] for l in open(f"{A}/phase2_forks.jsonl")}

pats = defaultdict(lambda: defaultdict(int))
for g in gen.values():
    pats[g["behavior_id"]][g["pattern_id"]] += 1
rng1, fold_of = np.random.default_rng(1), {}
for b in sorted(pats):  # greedy: biggest pattern first into the fold with fewest of this behavior's prompts (then overall)
    ps = sorted(pats[b])
    rng1.shuffle(ps)
    ps.sort(key=lambda p: -pats[b][p])
    nb, tot = [0] * 5, [sum(pats[bb][q] for bb in pats for q in pats[bb] if fold_of.get(q) == f) for f in range(5)]
    for p in ps:
        f = min(range(5), key=lambda f: (nb[f], tot[f]))
        fold_of[p] = f; nb[f] += pats[b][p]; tot[f] += pats[b][p]
folds = {pid: fold_of[g["pattern_id"]] for pid, g in gen.items()}
json.dump(folds, open(f"{OUT}/folds.json", "w"))


def load(path):
    conts, pooled = defaultdict(list), defaultdict(list)
    op = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    for r in map(json.loads, op):
        if not r["cut"].isdigit():
            continue
        c = int(r["cut"])
        pooled[(r["rid"], c)] += [bool(m) for m in r["match"]]
        for j, (ids, m) in enumerate(zip(r["conts"], r["match"])):
            if len(ids) >= MIN_LEN:
                conts[(r["rid"], c)].append((str(r["batch"]), j, ids, bool(m)))
    ok = {k: v for k, v in conts.items() if sum(x[3] for x in v) >= 2 and sum(not x[3] for x in v) >= 2}
    return ok, pooled


def pick(v, rng):
    out = []
    for cls in (True, False):
        xs = [x for x in v if x[3] == cls]
        rng.shuffle(xs)
        seen, first, rest = set(), [], []
        for x in xs:
            (rest if tuple(x[2][:max(K)]) in seen else first).append(x)
            seen.add(tuple(x[2][:max(K)]))
        out.append(first + rest)
    w, nn = out
    nw = min(len(w), max(PER_CLASS, 2 * PER_CLASS - len(nn)))
    return w[:nw] + nn[:2 * PER_CLASS - nw]


def build(chosen, ok, tag, rng):
    seqs, cuts, canon = [], [], {}
    for rid, c in chosen:
        pid, i = rid.split("|")
        g = gen[pid]
        reply, P = g["samples"][int(i)]["ids"], len(g["prompt_ids"])
        rows = []
        for batch, j, ids, m in pick(ok[(rid, c)], rng):
            full = g["prompt_ids"] + reply[:c] + ids[:max(K)]
            pos = [P + c + k - 1 for k in K]
            sid = f"{tag}|{rid}|{c}|{batch}|{j}"
            cn = [canon.setdefault((k, tuple(full[:p + 1])), sid) for k, p in zip(K, pos)]
            rows.append({"sid": sid, "kind": tag, "pid": pid, "rid": rid, "b": g["behavior_id"],
                         "split": split[pid]["split"], "pattern": g["pattern_id"], "cut": c, "match": m,
                         "fold": folds[pid], "ids": full, "k": list(K), "pos": pos, "canon": cn})
        ceil = {}
        for i, k in enumerate(K):  # best possible within-cut AUROC given exact-tie pairs
            pairs = [(a["canon"][i] == b["canon"][i]) for a in rows if a["match"] for b in rows if not b["match"]]
            ceil[f"k{k}"] = round(1 - 0.5 * sum(pairs) / len(pairs), 4)
        cuts.append({"rid": rid, "pid": pid, "b": g["behavior_id"], "split": split[pid]["split"], "cut": c,
                     "sharp": sharp.get(rid) == c, "n_weird": sum(r["match"] for r in rows),
                     "n_normal": sum(not r["match"] for r in rows), "fold": folds[pid], "ceiling": ceil})
        seqs += rows
    return seqs, cuts


def write(name, rows):
    with open(f"{OUT}/{name}", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


ok, pooled = load(f"{A}/forks_samples_off.jsonl.gz")
json.dump({f"{rid}|{c}": [float(np.mean(v)), len(v)] for (rid, c), v in pooled.items()}, open(f"{OUT}/p_pooled.json", "w"))
beh = lambda rid: gen[rid.split("|")[0]]["behavior_id"]
rng = random.Random(0)
chosen, per_rid = [], defaultdict(int)
for rid, c in sharp.items():
    if (rid, c) in ok:
        chosen.append((rid, c)); per_rid[rid] += 1
n_sharp = len(chosen)
by_b = defaultdict(list)
for k in sorted(ok):
    if k not in chosen:
        by_b[beh(k[0])].append(k)
for v in by_b.values():
    rng.shuffle(v)
while len(chosen) < N_CUTS and any(by_b.values()):  # round-robin over behaviors = stratified, leftovers flow on
    for b in sorted(by_b):
        while by_b[b]:
            rid, c = by_b[b].pop()
            if per_rid[rid] < PER_RID:
                chosen.append((rid, c)); per_rid[rid] += 1
                break
        if len(chosen) >= N_CUTS:
            break
seqs, cuts = build(chosen, ok, "cont", rng)
write("seqs_b.jsonl", seqs); write("cuts_b.jsonl", cuts)

okh, _ = load(f"{A}/forks_samples_on_dai_held.jsonl")
seqs_h, cuts_h = build(sorted(okh), okh, "conth", rng)
write("seqs_bh.jsonl", seqs_h); write("cuts_bh.jsonl", cuts_h)

for name, S, C in (("b", seqs, cuts), ("bh", seqs_h, cuts_h)):
    uniq = [sum(s["canon"][i] == s["sid"] for s in S) for i in range(len(K))]
    cb = defaultdict(int)
    for c in C:
        cb[c["b"].split("-")[0]] += 1
    print(f"({name}) {len(C)} cuts ({sum(c['sharp'] for c in C)} sharp), {len(S)} continuations; unique positions "
          f"k=1 {uniq[0]}, k=4 {uniq[1]}; mean tie ceiling k=1 {np.mean([c['ceiling']['k1'] for c in C]):.3f}, "
          f"k=4 {np.mean([c['ceiling']['k4'] for c in C]):.3f}; cuts with an untied pair at k=1 "
          f"{sum(c['ceiling']['k1'] > .5 for c in C)}; per behavior {dict(sorted(cb.items()))}")
fs = defaultdict(int)
for pid in folds:
    fs[folds[pid]] += 1
print("fold sizes (prompts)", dict(sorted(fs.items())), "patterns", len(fold_of))
