"""Blind 5-way matching test for the O-lens validation reads (scripts/81). Haiku 5.5 (adaptive thinking, effort low; it rejects a temperature) sees one TARGET readout and five
candidate readouts and picks the one read from the same activation. Per condition:
  win_his   target = Agastya's readout at window position k; candidates = ours (sample 0) at the 5 window positions
            of the same reply. Exact pipeline match puts the pick at k; an indexing slip shifts it to k-1 or k+1.
  win_self  same, but the target is our own second sample at k: the ceiling for this lens and condition.
  acr_his / acr_self   target at the window centre; candidates = the centres of this reply and 4 other replies.
Chance is 0.2 everywhere. Results cache to runs/olens_val/judge.jsonl; summary to runs/olens_val/summary.md.
"""
import argparse, json, math, os, random, re, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from judge import load_api_key  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--items", default="data/olens_val/items.jsonl")
ap.add_argument("--reads", default="runs/olens_val/readouts.jsonl")
ap.add_argument("--out", default="runs/olens_val")
ap.add_argument("--model", default="claude-haiku-5-5")
args = ap.parse_args()

items = {json.loads(l)["id"]: json.loads(l) for l in open(args.items)}
ours = defaultdict(dict)  # cond -> (id, k) -> [texts]
for l in open(args.reads):
    r = json.loads(l)
    ours[r["cond"]][(r["id"], r["k"])] = r["texts"]
ids = sorted(items)
L = "ABCDE"
PROMPT = """A tool reads one internal activation of a language model and writes a description of it. Each description below came from a separate read. The five candidates come from five different activations; exactly one of them is from the same activation as TARGET (read again, so the wording differs).

TARGET:
{target}

{options}

Which candidate was read from the same activation as TARGET? Answer with one letter only."""

def clip(t, n=700):
    t = t.strip()
    return t if len(t) <= n else t[:n] + " …"

trials = []
for cond, rd in ours.items():
    rng = random.Random(cond)
    for rid in ids:
        for k in range(5):
            cands = [(j, rd[(rid, j)][0]) for j in range(5)]
            for kind, tgt in (("win_his", items[rid]["his"][k]), ("win_self", rd[(rid, k)][1])):
                order = rng.sample(cands, 5)
                trials.append({"cond": cond, "kind": kind, "id": rid, "k": k, "tgt": tgt,
                               "opts": [t for _, t in order], "js": [j for j, _ in order]})
        others = rng.sample([i for i in ids if i != rid], 4)
        cands = [(i, rd[(i, 2)][0]) for i in [rid] + others]
        for kind, tgt in (("acr_his", items[rid]["his"][2]), ("acr_self", rd[(rid, 2)][1])):
            order = rng.sample(cands, 5)
            trials.append({"cond": cond, "kind": kind, "id": rid, "k": 2, "tgt": tgt,
                           "opts": [t for _, t in order], "js": [i for i, _ in order]})

cache_path = f"{args.out}/judge.jsonl"
key = lambda t: f"{t['cond']}|{t['kind']}|{t['id']}|{t['k']}"
done = {}
if os.path.exists(cache_path):
    for l in open(cache_path):
        r = json.loads(l)
        done[r["key"]] = r
todo = [t for t in trials if key(t) not in done]

import anthropic  # noqa: E402
client = anthropic.Anthropic(api_key=load_api_key())

def ask(t):
    opts = "\n\n".join(f"{L[i]}:\n{clip(o)}" for i, o in enumerate(t["opts"]))
    for _ in range(4):
        try:
            m = client.messages.create(model=args.model, max_tokens=2000, output_config={"effort": "low"},
                                       messages=[{"role": "user", "content": PROMPT.format(target=clip(t["tgt"]), options=opts)}])
            text = "".join(b.text for b in m.content if b.type == "text").strip()
            a = re.findall(r"\b([A-E])\b", text.upper())
            if a:
                return {"key": key(t), "pick": t["js"][L.index(a[-1])], "raw": text[-40:],
                        "tok": [m.usage.input_tokens, m.usage.output_tokens]}
        except anthropic.APIError as e:
            err = str(e)
    return {"key": key(t), "pick": None, "raw": err if "err" in dir() else "no letter"}

print(f"{len(trials)} trials, {len(todo)} to ask", flush=True)
with open(cache_path, "a") as f, ThreadPoolExecutor(8) as ex:
    for r in ex.map(ask, todo):
        done[r["key"]] = r
        f.write(json.dumps(r) + "\n")

def ci(x, n):  # Wilson 95%
    if not n:
        return float("nan"), float("nan")
    p, z = x / n, 1.96
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h

rows = ["| condition | test | n | accuracy | 95% CI | pick offset (window: chosen - true) |", "|---|---|---|---|---|---|"]
for cond in ours:
    for kind in ("win_his", "win_self", "acr_his", "acr_self"):
        ts = [t for t in trials if t["cond"] == cond and t["kind"] == kind and done[key(t)]["pick"] is not None]
        if kind.startswith("win"):
            hit = sum(done[key(t)]["pick"] == t["k"] for t in ts)
            off = Counter(done[key(t)]["pick"] - t["k"] for t in ts)
            offs = " ".join(f"{d:+d}:{off[d]}" for d in sorted(off))
        else:
            hit = sum(done[key(t)]["pick"] == t["id"] for t in ts)
            offs = ""
        lo, hi = ci(hit, len(ts))
        rows.append(f"| {cond} | {kind} | {len(ts)} | {hit / max(len(ts), 1):.2f} | {lo:.2f}-{hi:.2f} | {offs} |")
open(f"{args.out}/summary.md", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
