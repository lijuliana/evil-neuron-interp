"""P3 judging client. Judge configuration is G1's (scripts/60_gradual_g1_judge.py) verbatim: Gemma 4 31B on the vLLM
OpenAI server started with /data/runs/gradual_g1/chain2.sh's flags, WeirdChat's exact prompt (src/judge.py
build_judge_prompt with the behavior's transcript_rubric), one user message, enable_thinking=True, temperature 1,
max_tokens 8192, verdict parsed after the final '<channel|>' with WeirdChat's tolerant parser, 3 attempts. The body of
call() below is copied from G1; only the request set, the order and the deadline differ.
Requests (priority order; all conditions of a prompt are queued together so a deadline cut keeps the design balanced):
  1. identity-check replies (id_gen.jsonl, j < --id_judge_n) judged with the denying-ai-identity rubric
  2. every P3 reply in gen.jsonl, prompt by prompt (behaviors interleaved, held-out / discovery alternating, laser
     after its first --control_first prompts moved to the end), with that prompt's seeded 10% re-judges (10% per
     condition, as G1) right after it
  3. --n_val validation Stage A replies per behavior x verdict (default 0)
Resumable: appends to --out and skips request ids already there. Stops submitting at --stop_at (unix time).
Run on the VM (server up on --port): ~/venv/bin/python -I scripts/72_persona_p3_judge.py
"""
import argparse, collections, concurrent.futures as cf, glob, gzip, json, os, random, sys, threading, time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from judge import build_judge_prompt, parse_json_response  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gen", default="/data/runs/persona_p3/gen.jsonl")
ap.add_argument("--id_gen", default="/data/runs/persona_p3/id_gen.jsonl")
ap.add_argument("--meta", default="/data/runs/persona_p3/gen_meta.json")
ap.add_argument("--out", default="/data/runs/persona_p3/judge.jsonl")
ap.add_argument("--agastya", default="/data/agastya")
ap.add_argument("--weirdchat", default="/data/weirdchat")
ap.add_argument("--port", type=int, default=8012)
ap.add_argument("--concurrency", type=int, default=96)
ap.add_argument("--max_tokens", type=int, default=8192)
ap.add_argument("--n_val", type=int, default=0)
ap.add_argument("--id_judge_n", type=int, default=2)
ap.add_argument("--rejudge_frac", type=float, default=0.10)
ap.add_argument("--stop_at", type=float, default=0.0, help="unix time after which no new request is sent")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--control_first", type=int, default=4, help="laser prompts queued among the target prompts")
args = ap.parse_args()
BEH = ["denying-ai-identity", "false-physical-embodiment", "laser-at-aircraft"]
rng = random.Random(0)

import pandas as pd  # noqa: E402
ru = pd.read_parquet(os.path.join(args.weirdchat, "rubrics.parquet"))
RUB = {r.behavior_id: dict(r.transcript_rubric)["text"] for r in ru.itertuples()}
PROMPT = {}
for f in glob.glob(os.path.join(args.weirdchat, "prompts", "qwen__qwen3.6-27b", "*.parquet")):
    for r in pd.read_parquet(f).itertuples():
        msgs = r.messages
        if isinstance(msgs, str):
            import ast
            msgs = ast.literal_eval(msgs)
        msgs = [dict(m) for m in msgs]
        assert len(msgs) == 1 and msgs[0]["role"] == "user", msgs
        PROMPT[r.prompt_id] = msgs[0]["content"]

meta = json.load(open(args.meta))
gen = [json.loads(l) for l in open(args.gen)]
# seeded 10% re-judge per condition (cond key includes the behavior for the per-behavior ablation arms)
by_cond = collections.defaultdict(list)
for g in gen:
    by_cond[g["cond"]].append(g)
rep = set()
for c in sorted(by_cond):
    gs = by_cond[c]
    for g in rng.sample(gs, max(1, round(args.rejudge_frac * len(gs)))):
        rep.add((g["cond"], g["prompt_id"], g["j"]))
by_prompt = collections.defaultdict(list)
for g in gen:
    by_prompt[g["prompt_id"]].append(g)
# prompt order: within a behavior alternate held-out / discovery (each by production rate), so a deadline cut keeps the
# split balanced; behaviors interleaved rank by rank, with the control behavior (laser) capped at --control_first
# prompts until every target prompt is queued (a cut then falls on the control first)
def alternate(b):
    ps = meta["prompts"].get(b, [])
    h = [p for p in ps if meta["split"][p] == "heldout"]
    d = [p for p in ps if meta["split"][p] == "discovery"]
    out = []
    for i in range(max(len(h), len(d))):
        out += h[i:i + 1] + d[i:i + 1]
    return out
ALT = {b: alternate(b) for b in BEH}
order = []
for r in range(max(len(v) for v in ALT.values())):
    for b in BEH:
        if r < len(ALT[b]) and (b != "laser-at-aircraft" or r < args.control_first):
            order.append(ALT[b][r])
order += ALT["laser-at-aircraft"][args.control_first:]
reqs = []
if os.path.exists(args.id_gen):  # identity check first: short, and the side-effect check must not be cut
    idg = {}
    for l in open(args.id_gen):
        x = json.loads(l); idg[(x["cond"], x["qid"], x["j"])] = x  # last write wins (re-runs)
    idg = sorted([x for x in idg.values() if x["j"] < args.id_judge_n], key=lambda x: (x["j"], x["qid"], x["cond"]))
    for x in idg:
        reqs.append(dict(rid=f"id|{x['cond']}|{x['qid']}|{x['j']}", b="denying-ai-identity", prompt_id=None,
                         prompt_text=x["prompt"], text=x["text"], kind="id", cond=x["cond"], finish=x["finish"]))
for pid in order:
    gs = sorted(by_prompt[pid], key=lambda g: (g["j"], g["cond"]))
    for g in gs:
        reqs.append(dict(rid=f"p3|{g['cond']}|{pid}|{g['j']}", b=g["b"], prompt_id=pid, text=g["text"], kind="p3",
                         cond=g["cond"], finish=g["finish"], split=g["split"]))
    for g in gs:
        if (g["cond"], pid, g["j"]) in rep:
            reqs.append(dict(rid=f"p3|{g['cond']}|{pid}|{g['j']}|rep", b=g["b"], prompt_id=pid, text=g["text"], kind="rep",
                             cond=g["cond"], finish=g["finish"], split=g["split"]))
if args.n_val:
    on = {}
    for l in open(f"{args.agastya}/stage_a_judge_on.jsonl"):
        x = json.loads(l); on[x["key"]] = x
    want = collections.defaultdict(list)
    with gzip.open(f"{args.agastya}/stage_a_gen.jsonl.gz", "rt") as fh:
        for line in fh:
            g = json.loads(line)
            if g["behavior_id"] not in BEH:
                continue
            for i, s in enumerate(g["samples"]):
                k = f"{g['prompt_id']}|{i}"
                if k in on:
                    want[(g["behavior_id"], on[k]["match"])].append(
                        dict(rid="val|" + k, b=g["behavior_id"], prompt_id=g["prompt_id"], text=s["text"], kind="val",
                             ref_match=on[k]["match"], ref_n_in=on[k]["n_in"], ref_n_out=on[k]["n_out"]))
    for b in BEH:
        for m in (True, False):
            v = want[(b, m)]; rng.shuffle(v); reqs += v[:args.n_val]
done = set()
if os.path.exists(args.out):
    for l in open(args.out):
        try:
            x = json.loads(l)
            if x.get("error") is None:
                done.add(x["rid"])
        except Exception:
            pass
todo = [r for r in reqs if r["rid"] not in done]
if args.limit:
    todo = todo[:args.limit]
print(f"[judge] {len(reqs)} requests ({collections.Counter(r['kind'] for r in reqs)}), {len(done)} done, "
      f"{len(todo)} to do; stop_at {time.strftime('%H:%M:%S', time.gmtime(args.stop_at)) if args.stop_at else '-'} UTC",
      flush=True)

URL = f"http://localhost:{args.port}/v1/chat/completions"
lock = threading.Lock()
fo = open(args.out, "a")
stats = collections.Counter()
t0 = time.time()


def call(r):
    if args.stop_at and time.time() > args.stop_at:
        with lock:
            stats["skipped"] += 1
        return
    user = PROMPT[r["prompt_id"]] if r.get("prompt_id") else r["prompt_text"]
    # ---- from here to the parse: G1's call() verbatim
    prompt = build_judge_prompt(RUB[r["b"]], [{"role": "user", "content": user},
                                              {"role": "assistant", "content": r["text"]}])
    body = {"model": "gemma", "messages": [{"role": "user", "content": prompt}], "temperature": 1.0,
            "max_tokens": args.max_tokens, "skip_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": True}}
    rec = {k: v for k, v in r.items() if k not in ("text", "prompt_text")}
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        except Exception as e:  # server hiccup
            rec["error"] = f"http: {e}"; time.sleep(5); continue
        ch = resp["choices"][0]
        raw = ch["message"].get("content") or ""
        rc = ch["message"].get("reasoning_content") or ch["message"].get("reasoning") or ""
        final = raw.split("<channel|>")[-1] if "<channel|>" in raw else raw
        rec.update(n_in=resp["usage"]["prompt_tokens"], n_out=resp["usage"]["completion_tokens"],
                   finish=ch["finish_reason"], attempt=attempt)
        if ch["finish_reason"] == "length" or ("<|channel>" in raw and "<channel|>" not in raw):
            rec["error"] = "truncated"; continue
        try:
            p = parse_json_response(final)
        except ValueError as e:
            rec["error"] = f"parse: {str(e)[:200]}"; continue
        rec.update(match=bool(p["match"]), explanation=str(p.get("explanation", ""))[:600],
                   thought_chars=len(raw) - len(final) + len(rc), error=None)
        break
    with lock:
        fo.write(json.dumps(rec) + "\n"); fo.flush()
        stats["n"] += 1; stats["err"] += rec.get("error") is not None
        stats["in"] += rec.get("n_in", 0); stats["out"] += rec.get("n_out", 0)
        if stats["n"] % 50 == 0:
            el = time.time() - t0
            print(f"[judge] {stats['n']}/{len(todo)} {el/60:.1f} min, {stats['n']/el:.2f}/s, err {stats['err']}, "
                  f"mean in {stats['in']/stats['n']:.0f} out {stats['out']/stats['n']:.0f}", flush=True)


with cf.ThreadPoolExecutor(args.concurrency) as ex:
    list(ex.map(call, todo))
print(f"[judge] ALL_DONE {stats['n']} in {(time.time()-t0)/60:.1f} min, errors {stats['err']}, "
      f"skipped at deadline {stats['skipped']}", flush=True)
