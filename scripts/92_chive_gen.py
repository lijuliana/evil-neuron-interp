"""CHIVE Eval B generation (findings/chive/design.md section 3). Runs on a fleet VM, one shard of prompts per VM.

Engine: P3 / persona_gen's in-process vLLM 0.30 setup without hooks (bf16, prefix caching off, max_num_seqs 256,
max_model_len 10240, gpu_memory_utilization 0.90, seed 0, image/video inputs off; --eager 1 = enforce_eager as in
persona_gen). No speculative decoding / MTP (spec_method None, asserted). Qwen3.6-27B, chat template, thinking off,
no system prompt, T 1.0, top-k 20, top-p 0.95, max 1,024 new tokens.

Conditions per prompt: "base" (pool prompt) with replies j = 0..n_base-1 and every edit of the prompt with
j = 0..n_edit-1. One request per (condition, j), seed(prompt_id, j) = sha256("{prompt_id}|{j}")[:4] & 0x7fffffff, so
reply j of every condition of a prompt shares a seed (common random numbers as far as batch numerics allow; see
the determinism phase). Every condition's token ids come from chive_common.template_user_message (the one templating
function): the base message must reproduce the pool's stored stage-A prompt_ids exactly and every edit its
edited_prompt_ids, else the script aborts before loading the model.

Sharding: deterministic LPT by prompt (prompts sorted by nominal load 64 + 32 x n_edits desc, then sha256(prompt_id);
each goes to the least-loaded shard, ties to the lowest index). Uses the nominal 64/32 so the assignment does not
change with --n_base / --n_edit and topups land on the same shard as the prompt's main run.

Scheduling: requests are fed to the engine prompt by prompt (all conditions of a prompt together) while the number of
unfinished requests is below --inflight; finished outputs are collected per prompt, and a prompt's records are
appended (one write + fsync) only when all its conditions are done. A preemption therefore loses only prompts in
flight, and those are regenerated whole on restart (bf16 batch numerics are not reproducible, so partial per-prompt
output is never kept). Restart skips prompts already in the output file.

Phases (--phase)
  main   out/shard{i}.jsonl
  topup  --topup JSON {"tag", "edit_ids": [...], "edit_j": [32, 64], "base_prompt_ids": [...], "base_j": [64, 128]};
         out/shard{i}_topup_{tag}.jsonl; only prompts of this shard; base_prompt_ids defaults to none.
  det    batch-composition determinism: re-generates a fixed set of requests (first --det_prompts prompts of the
         shard: base j < 8 and the first 2 edits j < 4) (A) as in main (whole subset of the prompt batched
         together), (B) again in the same composition, (C) one request at a time (first --det_single), (D) reversed
         order interleaved with filler requests of other seeds; reports identical-token fraction and first-divergence
         position vs A.
         out/det_shard{i}.json
  check  CPU only (no vLLM): template / shard checks with the repo tokenizer (data/tokenizer), prints the plan.
Record: {cond_id, prompt_id, j, seed, text, n_tokens, finish_reason, head_ids (first 32 ids), n_prompt, user_sha,
         phase, machine, gpu}
Run on the VM: ~/vllm-venv/bin/python -I scripts/92_chive_gen.py --shard 0/8 [--phase main]
"""
import argparse, collections, gzip, hashlib, json, os, socket, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/data/models/Qwen3.6-27B")
ap.add_argument("--pool", default="/data/chive/pool.jsonl")
ap.add_argument("--edits", default="/data/chive/edits.jsonl")
ap.add_argument("--only", default="", help="comma list of prompt_ids or a JSON file with a 'prompt_ids' list")
ap.add_argument("--out", default="/data/runs/chive/gen")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--phase", default="main", choices=["main", "topup", "det", "check"])
ap.add_argument("--topup", default="")
ap.add_argument("--n_base", type=int, default=64)
ap.add_argument("--n_edit", type=int, default=32)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--top_k", type=int, default=20)
ap.add_argument("--max_tokens", type=int, default=1024)
ap.add_argument("--max_model_len", type=int, default=10240)
ap.add_argument("--max_num_seqs", type=int, default=256)
ap.add_argument("--gpu_mem", type=float, default=0.90)
ap.add_argument("--eager", type=int, default=1, help="1 = enforce_eager (persona_gen's engine); 0 = CUDA graphs")
ap.add_argument("--inflight", type=int, default=768, help="feed a new prompt while unfinished requests < this")
ap.add_argument("--det_prompts", type=int, default=1)
ap.add_argument("--det_single", type=int, default=4)
ap.add_argument("--machine", default=os.environ.get("CHIVE_MACHINE", socket.gethostname()))
ap.add_argument("--meta_tag", default="")
args = ap.parse_args()
T0 = time.time()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chive_common import template_user_message  # noqa: E402  (the one templating function)

NOMINAL_BASE, NOMINAL_EDIT = 64, 32
HEAD = 32


def log(*a):
    print(f"[gen {(time.time()-T0)/60:6.1f}m]", *a, flush=True)


def seed_of(pid, j):
    return int.from_bytes(hashlib.sha256(f"{pid}|{j}".encode()).digest()[:4], "big") & 0x7fffffff


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def jdump(x, p):
    tmp = p + ".tmp"
    json.dump(x, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


# ---------------------------------------------------------------- inputs, shard assignment
pool = {}
for l in open(args.pool):
    r = json.loads(l)
    pool[r["prompt_id"]] = r
edits = collections.defaultdict(list)
if os.path.exists(args.edits):
    for l in open(args.edits):
        e = json.loads(l)
        assert e["prompt_id"] in pool, ("edit for a prompt not in the pool", e["edit_id"])
        edits[e["prompt_id"]].append(e)
else:
    log(f"WARNING: no edits file at {args.edits}; base only")
eids = [e["edit_id"] for v in edits.values() for e in v]
assert len(eids) == len(set(eids)) and "base" not in eids, "edit_ids must be unique and != 'base'"
if args.only:
    only = json.load(open(args.only))["prompt_ids"] if os.path.exists(args.only) else args.only.split(",")
    pool = {p: pool[p] for p in only}
si, sn = (int(x) for x in args.shard.split("/"))
order = sorted(pool, key=lambda p: (-(NOMINAL_BASE + NOMINAL_EDIT * len(edits[p])), sha(p)))
load, assign = [0] * sn, {}
for p in order:
    k = min(range(sn), key=lambda s: (load[s], s))
    assign[p] = k
    load[k] += NOMINAL_BASE + NOMINAL_EDIT * len(edits[p])
mine = [p for p in order if assign[p] == si]
log(f"shard {si}/{sn}: {len(mine)} of {len(pool)} prompts, {sum(len(edits[p]) for p in mine)} edits; "
    f"nominal loads {load}")


def check_templates(tok):
    """Every condition through template_user_message; base must equal the stored stage-A ids, edits their stored ids."""
    bad = []
    for p in mine:
        r = pool[p]
        if template_user_message(r["user_message"], tok) != r["prompt_ids"]:
            bad.append(("base", p))
        for e in edits[p]:
            ids = template_user_message(e["edited_user_message"], tok)
            if "edited_prompt_ids" in e and ids != e["edited_prompt_ids"]:
                bad.append((e["edit_id"], p))
    if bad:
        log(f"ABORT: {len(bad)} conditions do not re-template to their stored prompt ids, e.g. {bad[:5]}")
        raise SystemExit(4)
    log(f"template check: base + {sum(len(edits[p]) for p in mine)} edits re-template exactly")


if args.phase == "check":
    from chive_common import get_tokenizer  # noqa: E402
    check_templates(get_tokenizer())
    print(json.dumps({"assign": collections.Counter(assign.values()), "loads": load}, default=str))
    raise SystemExit(0)

# ---------------------------------------------------------------- engine
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.sampling_params import RequestOutputKind  # noqa: E402

llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
          enable_prefix_caching=False, seed=0, limit_mm_per_prompt={"image": 0, "video": 0},
          max_num_seqs=args.max_num_seqs, enforce_eager=bool(args.eager))
eng = llm.llm_engine
spec = getattr(getattr(eng, "vllm_config", None), "speculative_config", None)
assert spec is None, f"speculative decoding must be off, got {spec}"
tok = llm.get_tokenizer()
GPU = torch.cuda.get_device_name(0)
VLLM_V = __import__("vllm").__version__
log("vLLM up", VLLM_V, GPU, args.machine, "eager" if args.eager else "cudagraph")
check_templates(tok)
os.makedirs(args.out, exist_ok=True)

meta_path = f"{args.out}/gen_meta_shard{si}{args.meta_tag}.json"
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"runs": []}
meta.update({"vllm": VLLM_V, "torch": torch.__version__, "model": args.model, "gpu": GPU, "machine": args.machine,
             "engine": dict(dtype="bfloat16", enforce_eager=bool(args.eager), enable_prefix_caching=False,
                            max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
                            gpu_memory_utilization=args.gpu_mem, seed=0, speculative_config=None),
             "sampling": dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                              max_tokens=args.max_tokens, n_base=args.n_base, n_edit=args.n_edit),
             "seed_fn": "int.from_bytes(sha256(f'{prompt_id}|{j}').digest()[:4], 'big') & 0x7fffffff",
             "pool": args.pool, "pool_sha256": hashlib.sha256(open(args.pool, "rb").read()).hexdigest(),
             "edits": args.edits, "edits_sha256": hashlib.sha256(open(args.edits, "rb").read()).hexdigest()
             if os.path.exists(args.edits) else None, "shard": args.shard, "prompts": mine,
             "assign": assign, "env": {k: v for k, v in os.environ.items() if k.startswith("VLLM_")}})
meta["runs"].append({"phase": args.phase, "start": time.time(), "machine": args.machine, "gpu": GPU})
jdump(meta, meta_path)


def conds_of(p, base_j, edit_j, edit_filter=None):
    """[(cond_id, user_message, prompt_ids, [j...])] for prompt p, every condition via template_user_message."""
    r = pool[p]
    out = []
    if base_j:
        out.append(("base", r["user_message"], template_user_message(r["user_message"], tok), list(base_j)))
    if edit_j:
        for e in edits[p]:
            if edit_filter is None or e["edit_id"] in edit_filter:
                out.append((e["edit_id"], e["edited_user_message"], template_user_message(e["edited_user_message"], tok),
                            list(edit_j)))
    return out


def sp_for(pid, j, n_prompt):
    mt = min(args.max_tokens, args.max_model_len - n_prompt)
    assert mt >= args.max_tokens, (pid, n_prompt)
    sp = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=mt,
                        seed=seed_of(pid, j))
    sp.output_kind = RequestOutputKind.FINAL_ONLY
    return sp


RID = [0]


def run_stream(jobs, on_prompt_done):
    """jobs: list of (key, conds) or (key, seed_pid, conds); seed_pid defaults to key (the prompt_id). Feeds job by
    job; calls on_prompt_done(key, results) when every request of the job has finished.
    results: list of (cond_id, user_message, n_prompt, j, CompletionOutput)."""
    pend = {}  # pid -> {"left": n, "res": []}
    rmap = {}
    q = list(jobs)
    ntok, t1 = 0, time.time()
    while q or eng.has_unfinished_requests():
        while q and eng.get_num_unfinished_requests() < args.inflight:
            job = q.pop(0)
            pid, spid, conds = (job[0], job[0], job[1]) if len(job) == 2 else job
            pend[pid] = {"left": sum(len(js) for *_, js in conds), "res": []}
            for cid, um, ids, js in conds:
                for j in js:
                    RID[0] += 1
                    rid = f"r{RID[0]}"
                    rmap[rid] = (pid, cid, um, len(ids), j)
                    eng.add_request(rid, TokensPrompt(prompt_token_ids=ids), sp_for(spid, j, len(ids)))
        for o in eng.step():
            if not o.finished:
                continue
            pid, cid, um, n_prompt, j = rmap.pop(o.request_id)
            c = o.outputs[0]
            ntok += len(c.token_ids)
            pend[pid]["res"].append((cid, um, n_prompt, j, c))
            pend[pid]["left"] -= 1
            if pend[pid]["left"] == 0:
                on_prompt_done(pid, pend.pop(pid)["res"])
                el = time.time() - t1
                log(f"prompt {pid[:12]} done; {ntok} tok in {el:.0f}s ({ntok/max(el,1e-9):.0f} tok/s cumulative); "
                    f"{len(q)} prompts queued, {eng.get_num_unfinished_requests()} requests unfinished")
    return ntok, time.time() - t1


def writer(path, phase):
    def done(pid, res):
        lines = []
        for cid, um, n_prompt, j, c in sorted(res, key=lambda x: (x[0] != "base", x[0], x[3])):
            lines.append(json.dumps({"cond_id": cid, "prompt_id": pid, "j": j, "seed": seed_of(pid, j), "text": c.text,
                                     "n_tokens": len(c.token_ids), "finish_reason": c.finish_reason,
                                     "head_ids": list(c.token_ids[:HEAD]), "n_prompt": n_prompt,
                                     "user_sha": sha(um)[:16], "phase": phase, "machine": args.machine, "gpu": GPU,
                                     "t": round(time.time(), 1)}))
        with open(path, "a") as fo:
            fo.write("\n".join(lines) + "\n")
            fo.flush()
            os.fsync(fo.fileno())
    return done


def done_prompts(path, expect):
    """Prompts whose records are complete in path (expect: pid -> number of records). Records are written one whole
    prompt per fsync'd append, so only a torn final block can be incomplete; any prompt whose record count is off
    (or unreadable lines) is dropped from the file (atomic rewrite) and regenerated whole."""
    if not os.path.exists(path):
        return set()
    recs, nbad = [], 0
    for l in open(path):
        try:
            recs.append(json.loads(l))
        except Exception:
            nbad += 1
    cnt = collections.Counter(r["prompt_id"] for r in recs)
    ok = {p for p in cnt if p in expect and cnt[p] == expect[p]}
    drop = [p for p in cnt if p not in ok]
    if drop or nbad:
        log(f"WARNING: {path}: dropping {nbad} unreadable lines and {len(drop)} incomplete / unexpected prompts "
            f"{[(p[:12], cnt[p], expect.get(p)) for p in drop[:3]]}; they are regenerated whole")
        tmp = path + ".tmp"
        with open(tmp, "w") as fo:
            for r in recs:
                if r["prompt_id"] in ok:
                    fo.write(json.dumps(r) + "\n")
            fo.flush()
            os.fsync(fo.fileno())
        os.replace(path, path + f".bak{int(time.time())}")
        os.replace(tmp, path)
    return ok


# ---------------------------------------------------------------- phases
if args.phase == "main":
    path = f"{args.out}/shard{si}.jsonl"
    jobs = [(p, conds_of(p, range(args.n_base), range(args.n_edit))) for p in mine]
    expect = {p: sum(len(js) for *_, js in c) for p, c in jobs}
    done = done_prompts(path, expect)
    todo = [x for x in jobs if x[0] not in done]
    log(f"main: {len(done)} prompts done, {len(todo)} to do ({sum(expect[p] for p, _ in todo)} requests)")
    ntok, el = run_stream(todo, writer(path, "main"))
    meta["runs"][-1].update(end=time.time(), n_prompts=len(todo), n_tokens=ntok, seconds=el, tok_per_s=ntok / max(el, 1e-9))
    jdump(meta, meta_path)
    log(f"main: {len(todo)} prompts, {ntok} tokens in {el:.0f}s ({ntok/max(el,1e-9):.0f} tok/s)")

elif args.phase == "topup":
    T = json.load(open(args.topup))
    tag = T.get("tag", "t1")
    path = f"{args.out}/shard{si}_topup_{tag}.jsonl"
    want_e = set(T.get("edit_ids", []))
    want_b = set(T.get("base_prompt_ids", []))
    ej = range(*T.get("edit_j", [32, 64])) if want_e else []
    bj = range(*T.get("base_j", [64, 128])) if want_b else []
    jobs = []
    for p in mine:
        has_e = any(e["edit_id"] in want_e for e in edits[p])
        c = conds_of(p, bj if p in want_b else [], ej if has_e else [], want_e)
        if c:
            jobs.append((p, c))
    expect = {p: sum(len(js) for *_, js in c) for p, c in jobs}
    done = done_prompts(path, expect)
    todo = [x for x in jobs if x[0] not in done]
    log(f"topup {tag}: {len(jobs)} prompts in this shard, {len(todo)} to do ({sum(expect[p] for p, _ in todo)} requests)")
    ntok, el = run_stream(todo, writer(path, f"topup_{tag}"))
    meta["runs"][-1].update(end=time.time(), tag=tag, n_tokens=ntok, seconds=el)
    jdump(meta, meta_path)

elif args.phase == "det":
    path = f"{args.out}/det_shard{si}.json"
    P = mine[:args.det_prompts]
    others = mine[args.det_prompts:] or list(pool)[:1]

    def subset(p):
        c = conds_of(p, range(min(8, args.n_base)), range(min(4, args.n_edit)))
        return [c[0]] + c[1:3]

    def collect(jobs):
        got = {}
        run_stream(jobs, lambda pid, res: got.update({(pid, cid, j): list(c.token_ids) for cid, _, _, j, c in res}))
        return got

    # A, B: whole subset of a prompt batched together (main-like)
    A = collect([(p, subset(p)) for p in P])
    B = collect([(p, subset(p)) for p in P])
    # C: one request at a time (first --det_single requests of the subset; batch-1 decoding is slow)
    C = {}
    singles = [(p, cid, um, ids, j) for p in P for cid, um, ids, js in subset(p) for j in js][:args.det_single]
    for p, cid, um, ids, j in singles:
        C.update(collect([(p, [(cid, um, ids, [j])])]))
    # D: reversed order, interleaved with filler requests (other prompts, other seeds) -> different batch composition
    jobsD = []
    for p in P:
        for cid, um, ids, js in reversed(subset(p)):
            jobsD.append((p + "|" + cid, p, [(cid, um, ids, list(reversed(js)))]))
            fp = others[len(jobsD) % len(others)]
            fu = pool[fp]["user_message"]
            jobsD.append((f"filler{len(jobsD)}", [("filler", fu, template_user_message(fu, tok), list(range(1000, 1012)))]))
    gotD = {}

    def onD(key, res):
        if key.startswith("filler"):
            return
        p = key.split("|")[0]
        gotD.update({(p, cid, j): list(c.token_ids) for cid, _, _, j, c in res})
    run_stream(jobsD, onD)

    def cmp(X):
        same, firstdiv, n = 0, [], 0
        for k, a in A.items():
            if k not in X:
                continue
            n += 1
            b = X[k]
            if a == b:
                same += 1
            else:
                d = next((i for i, (u, v) in enumerate(zip(a, b)) if u != v), min(len(a), len(b)))
                firstdiv.append(d)
        firstdiv.sort()
        return {"n": n, "identical": same, "frac_identical": same / max(n, 1),
                "first_divergence_tokens": firstdiv,
                "head20_identical": sum(1 for k, a in A.items() if k in X and a[:20] == X[k][:20]) / max(n, 1)}
    res = {"prompts": P, "n_requests": len(A), "A_vs_B_same_composition": cmp(B), "A_vs_C_single": cmp(C),
           "A_vs_D_reordered_with_filler": cmp(gotD), "machine": args.machine, "gpu": GPU,
           "eager": bool(args.eager), "max_num_seqs": args.max_num_seqs}
    jdump(res, path)
    log("det", json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                                                  if kk != "first_divergence_tokens"})
                           for k, v in res.items()}))
log("ALL_DONE")
