"""CHIVE Eval B judging client (findings/chive/design.md section 3). Runs on the VM that generated the shard, against
the Gemma 4 31B FP8 server started by 92_chive_chain.sh (P3 / persona_gen flags).

Judge call: scripts/74_persona_gen_judge.py's call verbatim: WeirdChat's exact prompt (src/judge.py
build_judge_prompt = Transluce DEFAULT_TRANSCRIPT_JUDGE_TEMPLATE + Docent-style transcript + the behavior's
transcript_rubric text from rubrics.parquet), one user message, enable_thinking=True, temperature 1, max_tokens 8192,
verdict parsed after the final '<channel|>' with WeirdChat's tolerant parser (src/judge.py parse_json_response).
The transcript is [user message, reply]. Unparseable or truncated verdicts are retried twice (3 attempts), then
recorded with match=null, status "failed" (final, not retried on restart). HTTP errors are recorded with status
"http" and retried on restart.

Modes (--modes, judged in this priority order from one request queue; each mode has its own output file):
  main     every reply in gen/shard{i}.jsonl and gen/shard{i}_topup_*.jsonl, judged against the user message the
           model actually saw (pool user_message for base, edited_user_message for an edit; checked against the
           gen record's user_sha)                                                -> judge/shard{i}.jsonl
  rejudge  a seeded 3% of main replies (sha256("rejudge|" + rid) % 10000 < 300), judged again
                                                                                 -> judge/shard{i}_rejudge.jsonl
  drift    16 stage-A replies per pool prompt of this shard (lowest sha256("drift|{pid}|{i}")), decoded from
           stage_a_gen.jsonl.gz, judged against the pool user message; ref_match = stage-A thinking-off label
                                                                                 -> judge/shard{i}_drift.jsonl
  orig     replies j < 8 of every edit, judged against the ORIGINAL user message (tags edits whose label flips from
           the prompt change alone)                                              -> judge/shard{i}_orig.jsonl
Record: {mode, rid, cond_id, prompt_id, j, behavior_id, judged_user, match, explanation, final, attempt, status,
         error, n_in, n_out, finish, thought_chars, raw_file, machine, t}; raw judge output of every attempt (content
         + reasoning) goes to judge/raw_{mode}_shard{i}.jsonl keyed by (rid, attempt).
Resumable: skips rids already recorded with status ok / failed. Prints throughput every 50 records and writes
judge/judge_meta_shard{i}.json.
Run: ~/venv/bin/python -I scripts/92_chive_judge.py --shard 0 --port 8012 --modes main,rejudge,drift,orig
"""
import argparse, collections, concurrent.futures as cf, glob, gzip, hashlib, json, os, socket, sys, threading, time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from judge import build_judge_prompt, parse_json_response  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--pool", default="/data/chive/pool.jsonl")
ap.add_argument("--edits", default="/data/chive/edits.jsonl")
ap.add_argument("--gen_dir", default="/data/runs/chive/gen")
ap.add_argument("--out_dir", default="/data/runs/chive/judge")
ap.add_argument("--weirdchat", default="/data/weirdchat")
ap.add_argument("--agastya", default="/data/agastya")
ap.add_argument("--modes", default="main,rejudge,drift,orig")
ap.add_argument("--port", type=int, default=8012)
ap.add_argument("--concurrency", type=int, default=96)
ap.add_argument("--max_tokens", type=int, default=8192)
ap.add_argument("--rejudge_per10k", type=int, default=300)
ap.add_argument("--drift_n", type=int, default=16)
ap.add_argument("--orig_n", type=int, default=8)
ap.add_argument("--machine", default=os.environ.get("CHIVE_MACHINE", socket.gethostname()))
ap.add_argument("--server_desc", default=os.environ.get("CHIVE_JUDGE_SERVER", ""))
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--only_rids_mod", default="", help="'a/b': judge only rids with sha256(rid) %% b == a (benchmarks)")
ap.add_argument("--meta_tag", default="")
args = ap.parse_args()
MODES = args.modes.split(",")
S = args.shard
os.makedirs(args.out_dir, exist_ok=True)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


import pandas as pd  # noqa: E402
ru = pd.read_parquet(os.path.join(args.weirdchat, "rubrics.parquet"))
RUB = {r.behavior_id: dict(r.transcript_rubric)["text"] for r in ru.itertuples()}
pool = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(args.pool)}
EDIT = {}
if os.path.exists(args.edits):
    for l in open(args.edits):
        e = json.loads(l)
        EDIT[e["edit_id"]] = e


def user_of(cond_id, pid):
    return pool[pid]["user_message"] if cond_id == "base" else EDIT[cond_id]["edited_user_message"]


# ---------------------------------------------------------------- requests
gen_files = [f"{args.gen_dir}/shard{S}.jsonl"] + sorted(glob.glob(f"{args.gen_dir}/shard{S}_topup_*.jsonl"))
gen = []
for f in gen_files:
    if os.path.exists(f):
        gen += [json.loads(l) for l in open(f)]
reqs = []  # (mode, rid, rec, user, judged_user, text, extra)
nbad_sha = 0
for g in gen:
    rid = f"{g['cond_id']}|{g['prompt_id']}|{g['j']}"
    u = user_of(g["cond_id"], g["prompt_id"])
    if sha(u)[:16] != g["user_sha"]:
        nbad_sha += 1
        continue
    b = pool[g["prompt_id"]]["behavior_id"]
    base = dict(cond_id=g["cond_id"], prompt_id=g["prompt_id"], j=g["j"], behavior_id=b)
    if "main" in MODES:
        reqs.append(("main", rid, base, u, "as_generated", g["text"], {}))
    if "rejudge" in MODES and int(sha("rejudge|" + rid)[:8], 16) % 10000 < args.rejudge_per10k:
        reqs.append(("rejudge", rid, base, u, "as_generated", g["text"], {}))
    if "orig" in MODES and g["cond_id"] != "base" and g["j"] < args.orig_n:
        reqs.append(("orig", rid, base, pool[g["prompt_id"]]["user_message"], "original", g["text"], {}))
if nbad_sha:
    print(f"[judge] ABORT: {nbad_sha} gen records whose user message hash does not match the pool / edits files", flush=True)
    raise SystemExit(5)
if "drift" in MODES:
    mp = f"{args.gen_dir}/gen_meta_shard{S}.json"
    mine = set(json.load(open(mp))["prompts"]) if os.path.exists(mp) else {g["prompt_id"] for g in gen}
    ref = {}
    jf = f"{args.agastya}/stage_a_judge_off.jsonl"
    if os.path.exists(jf):
        for l in open(jf):
            x = json.loads(l)
            if x["prompt_id"] in mine:
                ref[(x["prompt_id"], x["i"])] = x["match"]
    with gzip.open(f"{args.agastya}/stage_a_gen.jsonl.gz", "rt") as fh:
        for line in fh:
            st = json.loads(line)
            p = st["prompt_id"]
            if p not in mine:
                continue
            idx = sorted(range(len(st["samples"])), key=lambda i: sha(f"drift|{p}|{i}"))[:args.drift_n]
            for i in sorted(idx):
                reqs.append(("drift", f"stageA|{p}|{i}", dict(cond_id="stageA", prompt_id=p, j=i,
                                                                behavior_id=pool[p]["behavior_id"]),
                             pool[p]["user_message"], "pool_original", st["samples"][i]["text"],
                             {"ref_match": ref.get((p, i)), "ref_finish": st["samples"][i]["finish"]}))
OUT = {m: f"{args.out_dir}/shard{S}{'' if m == 'main' else '_' + m}.jsonl" for m in MODES}
RAW = {m: f"{args.out_dir}/raw_{m}_shard{S}.jsonl" for m in MODES}
done = collections.defaultdict(set)
for m in MODES:
    if os.path.exists(OUT[m]):
        for l in open(OUT[m]):
            try:
                x = json.loads(l)
            except Exception:
                continue
            if x.get("status") in ("ok", "failed"):
                done[m].add(x["rid"])
order = {m: k for k, m in enumerate(["main", "rejudge", "drift", "orig"])}
todo = [r for r in reqs if r[1] not in done[r[0]]]
if args.only_rids_mod:
    a, b = (int(x) for x in args.only_rids_mod.split("/"))
    todo = [r for r in todo if int(sha(r[1])[:8], 16) % b == a]
todo.sort(key=lambda r: order[r[0]])  # stable: gen-file order within a mode
if args.limit:
    todo = todo[:args.limit]
cnt = collections.Counter(r[0] for r in reqs)
print(f"[judge] shard {S}: {dict(cnt)} requests from {len(gen)} gen records; done "
      f"{ {m: len(v) for m, v in done.items()} }; {len(todo)} to do", flush=True)

URL = f"http://localhost:{args.port}/v1/chat/completions"
lock = threading.Lock()
fo = {m: open(OUT[m], "a") for m in MODES}
fr = {m: open(RAW[m], "a") for m in MODES}
stats = collections.Counter()
t0 = time.time()


def call(r):
    mode, rid, base, user, judged_user, text, extra = r
    # ---- 74_persona_gen_judge.py call() from here to the parse
    prompt = build_judge_prompt(RUB[base["behavior_id"]], [{"role": "user", "content": user},
                                                           {"role": "assistant", "content": text}])
    body = {"model": "gemma", "messages": [{"role": "user", "content": prompt}], "temperature": 1.0,
            "max_tokens": args.max_tokens, "skip_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": True}}
    rec = dict(mode=mode, rid=rid, **base, judged_user=judged_user, **extra, match=None, explanation=None, final=None,
               machine=args.machine, raw_file=os.path.basename(RAW[mode]))
    raws = []
    status = "failed"
    for attempt in range(3):
        rec["attempt"] = attempt
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        except Exception as e:  # server hiccup: not a verdict failure, retried on restart
            rec["error"] = f"http: {str(e)[:200]}"
            status = "http"
            time.sleep(5)
            continue
        status = "failed"
        ch = resp["choices"][0]
        raw = ch["message"].get("content") or ""
        rc = ch["message"].get("reasoning_content") or ch["message"].get("reasoning") or ""
        raws.append({"rid": rid, "attempt": attempt, "content": raw, "reasoning": rc})
        final = raw.split("<channel|>")[-1] if "<channel|>" in raw else raw
        rec.update(n_in=resp["usage"]["prompt_tokens"], n_out=resp["usage"]["completion_tokens"],
                   finish=ch["finish_reason"])
        if ch["finish_reason"] == "length" or ("<|channel>" in raw and "<channel|>" not in raw):
            rec["error"] = "truncated"
            continue
        try:
            p = parse_json_response(final)
        except ValueError as e:
            rec["error"] = f"parse: {str(e)[:200]}"
            continue
        rec.update(match=bool(p["match"]), explanation=str(p.get("explanation", ""))[:600], final=final[-1500:],
                   thought_chars=len(raw) - len(final) + len(rc), error=None)
        status = "ok"
        break
    rec["status"] = status
    rec["t"] = round(time.time(), 1)
    with lock:
        for x in raws:
            fr[mode].write(json.dumps(x) + "\n")
        fr[mode].flush()
        fo[mode].write(json.dumps(rec) + "\n")
        fo[mode].flush()
        stats["n"] += 1
        stats[status] += 1
        stats[f"n_{mode}"] += 1
        stats[f"att{rec['attempt']}"] += status == "ok"
        stats["in"] += rec.get("n_in", 0) or 0
        stats["out"] += rec.get("n_out", 0) or 0
        if stats["n"] % 50 == 0:
            el = time.time() - t0
            print(f"[judge] {stats['n']}/{len(todo)} {el/60:.1f} min, {stats['n']/el:.2f}/s, ok {stats['ok']} failed "
                  f"{stats['failed']} http {stats['http']}, mean in {stats['in']/stats['n']:.0f} out "
                  f"{stats['out']/stats['n']:.0f}", flush=True)


with cf.ThreadPoolExecutor(args.concurrency) as ex:
    list(ex.map(call, todo))
el = time.time() - t0
mp = f"{args.out_dir}/judge_meta_shard{S}{args.meta_tag}.json"
meta = json.load(open(mp)) if os.path.exists(mp) else {"passes": []}
meta["passes"].append({"start": t0, "seconds": el, "n": stats["n"], "per_s": stats["n"] / max(el, 1e-9),
                       "stats": dict(stats), "concurrency": args.concurrency, "server": args.server_desc,
                       "machine": args.machine, "modes": MODES, "todo": len(todo)})
json.dump(meta, open(mp + ".tmp", "w"), indent=1)
os.replace(mp + ".tmp", mp)
print(f"[judge] ALL_DONE {stats['n']} in {el/60:.1f} min ({stats['n']/max(el,1e-9):.2f}/s), ok {stats['ok']}, "
      f"failed {stats['failed']}, http {stats['http']}; {len(todo) - stats['n']} not sent", flush=True)
