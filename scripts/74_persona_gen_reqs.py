"""Build a judge request file (priority order) from persona-gen generation files. CPU only.

Order: identity-check replies first (j < --id_judge_n per prompt and condition, denying-ai-identity rubric), then every
reply prompt by prompt (behaviors interleaved, held-out / discovery alternating within a behavior), all conditions of a
prompt together so a deadline cut keeps the design balanced, then that prompt's seeded re-judges (--rejudge_frac per
condition, as P3). rid = <tag>|<kind>|<cond>|<prompt key>|<j>[|rep].
Usage: python -I scripts/74_persona_gen_reqs.py --tag b1 --gen b1_gen.jsonl --id b1_id.jsonl --out reqs_b1.jsonl
"""
import argparse, collections, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--gen", nargs="+", default=[])
ap.add_argument("--id", nargs="*", default=[])
ap.add_argument("--out", required=True)
ap.add_argument("--rejudge_frac", type=float, default=0.10)
ap.add_argument("--id_judge_n", type=int, default=1)
ap.add_argument("--conds", default="", help="comma list: only these conditions (default all)")
args = ap.parse_args()
rng = random.Random(0)
keep = set(args.conds.split(",")) if args.conds else None

recs = {}
for f in args.gen:
    for l in open(f):
        g = json.loads(l)
        if keep and g["cond"] not in keep:
            continue
        g["pkey"] = g.get("key") or g["prompt_id"]
        recs[(g["cond"], g["pkey"], g["j"])] = g  # last write wins (re-runs)
reqs = []
for f in args.id:
    idg = {}
    for l in open(f):
        x = json.loads(l)
        if keep and x["cond"] not in keep:
            continue
        idg[(x["cond"], x["qid"], x["j"])] = x
    for x in sorted([x for x in idg.values() if x["j"] < args.id_judge_n], key=lambda x: (x["j"], x["qid"], x["cond"])):
        reqs.append(dict(rid=f"{args.tag}|id|{x['cond']}|{x['qid']}|{x['j']}", kind="id", b="denying-ai-identity",
                         prompt_text=x["prompt"], text=x["text"], cond=x["cond"], qid=x["qid"], j=x["j"], finish=x["finish"]))
by_cond = collections.defaultdict(list)
for k in recs:
    by_cond[k[0]].append(k)
rep = set()
for c in sorted(by_cond):
    ks = sorted(by_cond[c])
    rep |= set(rng.sample(ks, max(1, round(args.rejudge_frac * len(ks))))) if args.rejudge_frac > 0 else set()
by_p = collections.defaultdict(list)
for k, g in recs.items():
    by_p[g["pkey"]].append(g)
first = {p: v[0] for p, v in by_p.items()}
behs = sorted({g["b"] for g in first.values()})
alt = {}
for b in behs:
    ps = sorted([p for p, g in first.items() if g["b"] == b])
    h = [p for p in ps if first[p].get("split") == "heldout"]
    d = [p for p in ps if first[p].get("split") != "heldout"]
    alt[b] = [x for i in range(max(len(h), len(d))) for x in (h[i:i + 1] + d[i:i + 1])]
order = [alt[b][r] for r in range(max(len(v) for v in alt.values())) for b in behs if r < len(alt[b])]
for p in order:
    gs = sorted(by_p[p], key=lambda g: (g["j"], g["cond"]))
    for rep_pass in (False, True):
        for g in gs:
            k = (g["cond"], g["pkey"], g["j"])
            if rep_pass and k not in rep:
                continue
            r = dict(rid=f"{args.tag}|main|{g['cond']}|{g['pkey']}|{g['j']}" + ("|rep" if rep_pass else ""),
                     kind="rep" if rep_pass else "main", b=g.get("judge_b", g["b"]), text=g["text"], cond=g["cond"],
                     pkey=g["pkey"], j=g["j"], split=g.get("split"), finish=g["finish"], beh=g["b"])
            if g.get("prompt_text"):
                r["prompt_text"] = g["prompt_text"]
            else:
                r["prompt_id"] = g["prompt_id"]
            reqs.append(r)
with open(args.out, "w") as fo:
    for r in reqs:
        fo.write(json.dumps(r) + "\n")
print(args.out, len(reqs), dict(collections.Counter(r["kind"] for r in reqs)))
