"""G1 judging client: Gemma 4 31B (vLLM OpenAI server, thinking ON) with WeirdChat's exact judge prompt.

Prompt = src/judge.py build_judge_prompt (Transluce DEFAULT_TRANSCRIPT_JUDGE_TEMPLATE, Docent-style transcript,
behavior's transcript_rubric text from rubrics.parquet), sent as one user message through Gemma's chat template
with enable_thinking=True; temperature 1 (server applies the checkpoint's top-p/top-k defaults). The verdict is
parsed from the text after the final '<channel|>' (end of the thought channel) with WeirdChat's tolerant parser.
Requests (priority order): validation = Stage A replies with Agastya's production verdict (15 pos + 15 neg per
behavior); every G1 reply in gen.jsonl, by condition; a seeded 10% of G1 replies judged a second time (noise floor).
Resumable: appends to <out>/judge.jsonl and skips request ids already there.
Run on the VM (server already up on --port): ~/venv/bin/python -I scripts/60_gradual_g1_judge.py
"""
import argparse, collections, concurrent.futures as cf, glob, gzip, json, os, random, sys, threading, time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from judge import build_judge_prompt, parse_json_response  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gen", default="/data/runs/gradual_g1/gen.jsonl")
ap.add_argument("--out", default="/data/runs/gradual_g1/judge.jsonl")
ap.add_argument("--agastya", default="/data/agastya")
ap.add_argument("--weirdchat", default="/data/weirdchat")
ap.add_argument("--port", type=int, default=8011)
ap.add_argument("--concurrency", type=int, default=96)
ap.add_argument("--max_tokens", type=int, default=8192)
ap.add_argument("--n_val", type=int, default=15)
ap.add_argument("--rejudge_frac", type=float, default=0.05)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--only_val", action="store_true")
ap.add_argument("--split_j", type=int, default=4, help="judge replies j < split_j of every condition first")
args = ap.parse_args()
BEH = ["recommends-drunk-driving", "chemtrails-assertion", "cutting-instructions"]
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

reqs = []  # dict(rid, b, prompt_id, text, kind, ...)
# validation: Stage A replies with production verdicts
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
if not args.only_val:
    gen = [json.loads(l) for l in open(args.gen)]
    order = {c: i for i, c in enumerate(dict.fromkeys(g["cond"] for g in gen))}
    # budget-aware order: replies j < split_j first for every condition, then the rest; within a group,
    # prompt-major (prompts interleaved across behaviors) and condition-minor, so a partial pass still gives
    # complete per-prompt condition blocks; re-judges (noise floor) after each group
    pids_by_b = collections.defaultdict(list)
    for g in gen:
        if g["prompt_id"] not in pids_by_b[g["b"]]:
            pids_by_b[g["b"]].append(g["prompt_id"])
    rr, k = {}, 0
    while any(pids_by_b.values()):
        for b in BEH:
            if pids_by_b[b]:
                rr[pids_by_b[b].pop(0)] = k; k += 1
    for grp in (0, 1):
        gs = [g for g in gen if (g["j"] < args.split_j) == (grp == 0)]
        gs.sort(key=lambda g: (rr[g["prompt_id"]], g["j"], order[g["cond"]]))
        for g in gs:
            reqs.append(dict(rid=f"g1|{g['cond']}|{g['prompt_id']}|{g['j']}", b=g["b"], prompt_id=g["prompt_id"],
                             text=g["text"], kind="g1", cond=g["cond"], finish=g["finish"]))
        for g in rng.sample(gs, max(1, round(args.rejudge_frac * len(gs)))):
            reqs.append(dict(rid=f"g1|{g['cond']}|{g['prompt_id']}|{g['j']}|rep", b=g["b"], prompt_id=g["prompt_id"],
                             text=g["text"], kind="rep", cond=g["cond"], finish=g["finish"]))
done = set()
if os.path.exists(args.out):
    for l in open(args.out):
        try:
            done.add(json.loads(l)["rid"])
        except Exception:
            pass
todo = [r for r in reqs if r["rid"] not in done]
if args.limit:
    todo = todo[:args.limit]
print(f"[judge] {len(reqs)} requests, {len(done)} done, {len(todo)} to do", flush=True)

URL = f"http://localhost:{args.port}/v1/chat/completions"
lock = threading.Lock()
fo = open(args.out, "a")
stats = collections.Counter()
t0 = time.time()


def call(r):
    prompt = build_judge_prompt(RUB[r["b"]], [{"role": "user", "content": PROMPT[r["prompt_id"]]},
                                              {"role": "assistant", "content": r["text"]}])
    body = {"model": "gemma", "messages": [{"role": "user", "content": prompt}], "temperature": 1.0,
            "max_tokens": args.max_tokens, "skip_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": True}}
    rec = {k: v for k, v in r.items() if k != "text"}
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
print(f"[judge] ALL_DONE {stats['n']} in {(time.time()-t0)/60:.1f} min, errors {stats['err']}", flush=True)
