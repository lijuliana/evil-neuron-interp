"""Pick the replies for the token viewer from Agastya's Stage A data.

All 410 fork replies (forks_off.jsonl), plus up to 2 judged-normal replies per fork prompt for contrast.
Writes data/viewer/items.jsonl: one line per reply with prompt and reply token ids and both judge labels.
"""
import gzip, json, os, collections

A = "data/agastya"
OUT = "data/viewer/items.jsonl"
N_CONTRAST = 2

gen = {r["prompt_id"]: r for r in map(json.loads, gzip.open(f"{A}/stage_a_gen.jsonl.gz", "rt"))}
judge_off, judge_on = collections.defaultdict(dict), collections.defaultdict(dict)
for path, store in ((f"{A}/stage_a_judge_off.jsonl", judge_off), (f"{A}/stage_a_judge_on.jsonl", judge_on)):
    for line in open(path):
        d = json.loads(line)
        pid, i = d["key"].split("|")
        store[pid][int(i)] = {"match": d["match"], "why": d.get("explanation", "")}
forks = [json.loads(l) for l in open(f"{A}/forks_off.jsonl")]

def item(pid, i, kind):
    r, s = gen[pid], gen[pid]["samples"][i]
    return {"rid": f"{pid}|{i}", "pid": pid, "i": i, "b": r["behavior_id"], "split": r["split"],
            "pattern": r["pattern_id"], "kind": kind, "prompt_ids": r["prompt_ids"], "ids": s["ids"],
            "finish": s["finish"], "off": judge_off[pid].get(i), "on": judge_on[pid].get(i)}

items, fork_rids = [], set()
for f in forks:
    pid, i = f["rid"].split("|")
    items.append(item(pid, int(i), "fork"))
    fork_rids.add(f["rid"])
for pid in sorted({f["prompt_id"] for f in forks}):
    picked = 0
    for i, s in enumerate(gen[pid]["samples"]):
        if picked == N_CONTRAST:
            break
        off, on = judge_off[pid].get(i), judge_on[pid].get(i)
        if f"{pid}|{i}" in fork_rids or off is None or off["match"] or (on and on["match"]) or s["finish"] != "stop":
            continue
        items.append(item(pid, i, "contrast"))
        picked += 1

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    for it in items:
        fh.write(json.dumps(it) + "\n")
c = collections.Counter((it["b"], it["kind"]) for it in items)
print(len(items), "items,", sum(len(it["prompt_ids"]) + len(it["ids"]) for it in items), "tokens")
for k in sorted(c):
    print(k, c[k])
