"""O-lens validation set from Agastya's public Qwen3.6-27B refusal viewer (block-56 olens_s3d_rl600 readouts,
one per token, exact token IDs). 24 replies (each of the 12 settings once benign, once unsafe, formats
alternating), 5 consecutive continuation positions per reply, all with a complete readout.
The 5-position windows let the match test see off-by-one indexing: his readout at p should match ours at p,
not p-1 or p+1.
Output: data/olens_val/items.jsonl, one line per reply:
  {id, cell, ids (input_ids + output_ids), n_in, pos (absolute indices), his (readout per pos), tok (decoded)}
"""
import json, os, random, urllib.request

BASE = "https://agastyasridharan.github.io/assistant-privilegeness/refusal/data/qwen36"
OUT = "data/olens_val/items.jsonl"
get = lambda path: json.load(urllib.request.urlopen(f"{BASE}/{path}?v=olens-1"))

ix = get("index.json")
rng = random.Random(0)
by_cell = {}
for s in ix["samples"]:
    by_cell.setdefault(s["cell"], []).append(s["id"])
picks = []
for i, st in enumerate(sorted({c["setting"] for c in ix["cells"]})):
    fmts = ("raw", "assistant_turn") if i % 2 == 0 else ("assistant_turn", "raw")
    for cond, fmt in zip(("benign", "unsafe"), fmts):
        picks.append(f"{st}-{cond}-{fmt}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for cell in picks:
        for sid in rng.sample(by_cell[cell], len(by_cell[cell])):
            d = get(f"samples/{sid}.json")
            r = d["readouts"]
            ok = [p for p in range(8, len(r) - 2)
                  if all(r[q] and r[q].get("status") == "complete" for q in range(p - 2, p + 3))]
            if ok:
                break
        p = rng.choice(ok)
        n_in = len(d["input_ids"])
        f.write(json.dumps({"id": sid, "cell": cell, "ids": d["input_ids"] + d["output_ids"], "n_in": n_in,
                            "pos": [n_in + q for q in range(p - 2, p + 3)],
                            "his": [r[q]["text"] for q in range(p - 2, p + 3)],
                            "tok": d["tokens"][p - 2:p + 3]}) + "\n")
        print(cell, sid, "p", p, repr("".join(d["tokens"][p - 2:p + 3])))
