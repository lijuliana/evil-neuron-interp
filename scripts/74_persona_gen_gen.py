"""persona-gen generation (Qwen3.6-27B, vLLM 0.30 in-process, enforce_eager, prefix caching off, bf16), P3's engine and
hooks extended to decoder-layer outputs 40-59. One engine load runs the phases given in --phases, each resumable.

Edits at decoder-layer outputs (vLLM returns (hidden_states, residual); the residual stream is their sum, so the hook
computes h = hs + res and returns (hs + delta, res)):
  cap  h <- h - v * min(<h, v> - tau, 0)          (Lu et al.; P3)
  ceil h <- h - v * max(<h, v> - tau, 0)          (cap from above, B3)
  out  h <- h - v * <h, v>                        (projection out, B3)
  add  h <- h + alpha * v                         (steering, B3)
Directions: the community axis (P1's vectors, lerped between saved blocks 8..56, clamped at 56 for layers 56-59), P1's
20 role-pair differences (c1x_pairs.json), and B3's picks. tau per direction, layer and percentile (25/50/75) is computed
here from P1's 24 benign default replies (response tokens, no system prompt; P3's procedure) with the hook in capture
mode; check: at layers 46-53 the axis 25th percentile must reproduce P1's tau (25% +- 6% of tokens below it).

Phases
  tau   thresholds for every direction x layer 40-59 x {25, 50, 75}; hook / cap-effect checks (checks.json, tau.json)
  b1    X6 cap dose sweep on P3's 26 prompts x 8 (P3's seeds): clean; axis cap at p{25,50,75} x layers {46-53, 40-59};
        then, at the strongest axis setting whose damage (max of dNLL on capability_mini and on P1 benign) <= 0.02,
        role pairs rp16 / rp10 / rp12 (P3's first three controls) at that percentile and layers. "Strongest" is fixed
        before any generation: the largest mean total cap displacement per token on the damage texts (capability_mini
        clean replies + P1 benign replies; sum over capped layers of |min(<h,v> - tau, 0)|, measured with the cap on). Damage, greedy accuracy and P3's identity check
        (20 prompts x 8) for every condition.
  b3    X2 / X3 conditions from --picks (written by the persona_fwd agent); see the B3 section.
(X4 / multi-turn drift was cancelled by Juliana on 2026-11-01 before any generation.)
Outputs (--out): checks.json, tau.json, damage.json, acc.jsonl, b1_gen.jsonl, b1_id.jsonl, b3_*.jsonl, gen_meta.json.
Generation records are appended per finished condition, so a crash or
preemption re-runs at most the condition in flight.
Run on the VM: [flock /data/gpu.lock] ~/vllm-venv/bin/python -I scripts/74_persona_gen_gen.py --phases tau,b1
"""
import argparse, collections, gzip, hashlib, json, os, re, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # in-process engine: apply_model / hooks see our globals
import numpy as np
import torch

BEH = ["denying-ai-identity", "false-physical-embodiment", "laser-at-aircraft"]
EOS = {248046, 248044}
CAP_LAYERS = list(range(40, 60))
HOOK_LAYERS = CAP_LAYERS
AX_BLOCKS = [8, 16, 24, 32, 40, 48, 56]
PCTS = [5, 10, 25, 50, 75, 90, 95]  # B1 uses 25/50/75; the rest are for B3 ceilings
LRANGES = {"L46-53": list(range(46, 54)), "L40-59": list(range(40, 60))}
D = 5120

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/data/models/Qwen3.6-27B")
ap.add_argument("--agastya", default="/data/agastya")
ap.add_argument("--p1", default="/data/runs/persona_p1")
ap.add_argument("--p3", default="/data/runs/persona_p3")
ap.add_argument("--texts", default="/data/runs/persona_gen/prompt_texts.json")
ap.add_argument("--picks", default="/data/runs/persona_fwd/picks.json")
ap.add_argument("--b3_spec", default="/data/runs/persona_gen/b3_spec.json")
ap.add_argument("--capability", default=os.path.expanduser("~/evil-neuron-interp/promptsets/capability_mini.jsonl"))
ap.add_argument("--out", default="/data/runs/persona_gen")
ap.add_argument("--phases", default="tau,b1")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--top_k", type=int, default=20)
ap.add_argument("--max_tokens", type=int, default=1024)
ap.add_argument("--id_n", type=int, default=8)
ap.add_argument("--max_model_len", type=int, default=10240)
ap.add_argument("--damage_gate", type=float, default=0.02)
ap.add_argument("--rp_ctrl", default="16,10,12")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
PH = args.phases.split(",")
T0 = time.time()


def log(*a):
    print(f"[gen {(time.time()-T0)/60:6.1f}m]", *a, flush=True)


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def jdump(x, p):
    tmp = p + ".tmp"
    json.dump(x, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


# ---------------------------------------------------------------- prompts (P3's selection, from its gen_meta.json)
A = args.agastya
p3meta = json.load(open(f"{args.p3}/gen_meta.json"))
split = json.load(open(f"{A}/split.json"))
TEXTS = json.load(open(args.texts))
gen = {}
with gzip.open(f"{A}/stage_a_gen.jsonl.gz", "rt") as fh:
    for line in fh:
        g = json.loads(line)
        gen[g["prompt_id"]] = {"b": g["behavior_id"], "split": split[g["prompt_id"]]["split"], "prompt_ids": g["prompt_ids"]}
prompts = {b: list(p3meta["prompts"][b]) for b in BEH}
if args.smoke:
    prompts = {b: v[:2] for b, v in prompts.items()}
    args.n, args.id_n = 2, 2
ID_PROMPTS = p3meta["id_prompts"]
if args.smoke:
    ID_PROMPTS = ID_PROMPTS[:3]

# ---------------------------------------------------------------- directions
V = f"{args.p1}/vectors"
axis_raw = np.load(f"{V}/axis_raw.npy").astype(np.float64)
roles = np.load(f"{V}/roles_raw.npy").astype(np.float64)
pairs = json.load(open(f"{args.p1}/c1x_pairs.json"))["pairs"]
ct = torch.load(f"{args.p1}/cap_tau.pt", weights_only=True)
TAU_AX_P1 = {int(l): float(v[0]) for l, v in ct["tau"].items()}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-12)


def lerp_raw(x, l):  # x [7, D] at AX_BLOCKS; P1/P3's lerp, clamped at 8 / 56
    if l <= 8:
        return x[0]
    if l >= 56:
        return x[6]
    i = (l - 8) // 8
    w = (l - AX_BLOCKS[i]) / 8.0
    return (1 - w) * x[i] + w * x[i + 1]


DIRS = {"axis": {l: unit(lerp_raw(axis_raw, l)) for l in HOOK_LAYERS}}
for j, (a, b) in enumerate(pairs):
    DIRS[f"rp{j}"] = {l: unit(lerp_raw(roles[a] - roles[b], l)) for l in HOOK_LAYERS}

B3 = None
if "b3" in PH:  # B3 directions come from a spec this session writes after reading picks.json (see b3 section)
    B3 = json.load(open(args.b3_spec))
    traits = np.load(f"{V}/traits_unit.npz")
    default_raw = np.load(f"{V}/default_raw.npy").astype(np.float64)
    dir_cos = {}
    for name, d in B3["directions"].items():
        if d["kind"] == "trait":  # traits_unit rows are blocks (file index L+1 already applied)
            DIRS[name] = {l: unit(traits[d["key"]][l].astype(np.float64)) for l in HOOK_LAYERS}
        elif d["kind"] == "default_minus_role":  # persona_fwd convention: unit(default - role), + = Assistant end
            r = int(d["role_index"])
            DIRS[name] = {l: unit(lerp_raw(default_raw - roles[r], l)) for l in HOOK_LAYERS}
        elif d["kind"] == "rows":  # .npy with one unit row per listed layer (persona_fwd picks/); other layers zero
            arr = np.load(d["path"], allow_pickle=False)
            DIRS[name] = {l: unit(arr[d["layers"].index(l)].astype(np.float64)) if l in d["layers"] else np.zeros(D)
                          for l in HOOK_LAYERS}
        else:
            raise ValueError(d)
        if d.get("check_file"):  # cosine of our direction with the persona_fwd agent's saved rows
            arr = np.load(d["check_file"], allow_pickle=False)
            dir_cos[name] = {l: round(float(DIRS[name][l] @ unit(arr[k].astype(np.float64))), 5) for k, l in enumerate(d["check_layers"])}
    print("[gen] b3 direction cosine vs picks files", {k: (min(v.values()), max(v.values())) for k, v in dir_cos.items()}, flush=True)
DNAMES = list(DIRS)

# ---------------------------------------------------------------- vLLM + hooks
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
          enable_prefix_caching=False, seed=args.seed, limit_mm_per_prompt={"image": 0, "video": 0},
          max_num_seqs=256, enforce_eager=True)
tok = llm.get_tokenizer()
log("vLLM up", __import__("vllm").__version__, torch.cuda.get_device_name(0))

STATE = {"edits": {}, "cap": None, "capM": {}, "ntok": collections.Counter(), "ncap": {}, "dsum": {}}
LAYER_PAT = re.compile(r"(?:^|\.)layers\.(\d+)$")


def make_hook(l):
    def hook(mod, a, out):
        st = STATE
        ed = st["edits"].get(l)
        capt = st["cap"] is not None and l in st["capM"]
        if not ed and not capt:
            return None
        hs, res = out
        h = hs.float() + res.float()
        delta = None
        if ed:
            delta = torch.zeros_like(h)
            for kind, v, x in ed:
                if kind == "cap":
                    p = (h + delta) @ v
                    coef = (p - x).clamp(max=0)
                    st["ntok"][l] += p.shape[0]
                    st["ncap"][l] += (coef < 0).sum()
                    st["dsum"][l] += (-coef).sum()
                    delta = delta - coef[:, None] * v
                elif kind == "ceil":  # cap from above: h <- h - v * max(<h, v> - tau, 0)
                    p = (h + delta) @ v
                    coef = (p - x).clamp(min=0)
                    st["ntok"][l] += p.shape[0]
                    st["ncap"][l] += (coef > 0).sum()
                    st["dsum"][l] += coef.sum()
                    delta = delta - coef[:, None] * v
                elif kind == "out":  # project out: h <- h - v <h, v>
                    p = (h + delta) @ v
                    delta = delta - p[:, None] * v
                elif kind == "add":
                    delta = delta + x * v
                else:
                    raise ValueError(kind)
            h = h + delta
        if capt:
            st["cap"][l].append((h @ st["capM"][l].T).cpu())
        if delta is None:
            return None
        return ((hs.float() + delta).to(hs.dtype), res)
    return hook


def install(model):
    lay = {}
    for name, mod in model.named_modules():
        if "visual" in name or "mtp" in name:
            continue
        m = LAYER_PAT.search(name)
        if m:
            lay[int(m.group(1))] = mod
    assert sorted(lay) == list(range(64)), sorted(lay)[:5]
    for l in HOOK_LAYERS:
        lay[l].register_forward_hook(make_hook(l))
    dev = next(model.parameters()).device
    STATE["dev"] = dev
    return len(HOOK_LAYERS), type(lay[46]).__name__, str(dev)


log("hooks", llm.apply_model(install))
DEV = STATE["dev"]


def reset_counters():
    STATE["ntok"] = collections.Counter()
    STATE["ncap"] = {l: torch.zeros((), dtype=torch.long, device=DEV) for l in HOOK_LAYERS}
    STATE["dsum"] = {l: torch.zeros((), dtype=torch.float64, device=DEV) for l in HOOK_LAYERS}


def set_edits(edits):
    """edits: list of dicts {type: cap|add, dir, layers, pct | tau | alpha}. None / [] = clean."""
    E = collections.defaultdict(list)
    for e in edits or []:
        for l in e["layers"]:
            v = torch.tensor(DIRS[e["dir"]][l], dtype=torch.float32, device=DEV)
            if e["type"] in ("cap", "ceil"):
                t = e["tau"][str(l)] if "tau" in e else TAU[e["dir"]][e["pct"]][l]
                E[l].append((e["type"], v, float(t)))
            elif e["type"] == "out":
                E[l].append(("out", v, 0.0))
            else:
                E[l].append(("add", v, float(e["alpha_l"][str(l)] if "alpha_l" in e else e["alpha"])))
    STATE["edits"] = dict(E)
    reset_counters()


def cap_stats():
    ls = [l for l in HOOK_LAYERS if STATE["ntok"][l]]
    return ({l: round(float(STATE["ncap"][l]) / STATE["ntok"][l], 4) for l in ls},
            float(sum(float(STATE["dsum"][l]) / STATE["ntok"][l] for l in ls)) if ls else 0.0)


reset_counters()

# ---------------------------------------------------------------- helpers
sp_nll = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)


def nll_tokens(seqs):
    outs = llm.generate([TokensPrompt(prompt_token_ids=p + r) for p, r in seqs], sp_nll, use_tqdm=False)
    res = []
    for (p, r), o in zip(seqs, outs):
        pl = o.prompt_logprobs
        res.append(np.array([-pl[len(p) + k][t].logprob for k, t in enumerate(r)], dtype=np.float64))
    return res


def chat_ids(msgs):
    if isinstance(msgs, str):
        msgs = [{"role": "user", "content": msgs}]
    x = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return list(x["input_ids"]) if not isinstance(x, list) else x


def seed_of(key, j):
    return int(hashlib.sha256(f"{key}|{j}".encode()).hexdigest()[:8], 16)


def sample(items, n, max_tokens):
    """items: list of (key, prompt_ids). One request per (key, j), seed fixed per (key, j) across conditions."""
    reqs, sps, meta = [], [], []
    for key, p in items:
        mt = min(max_tokens, args.max_model_len - len(p))
        assert mt >= 256, (key, len(p))
        for j in range(n):
            reqs.append(TokensPrompt(prompt_token_ids=p))
            sps.append(SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                      max_tokens=mt, seed=seed_of(key, j)))
            meta.append((key, j))
    outs = llm.generate(reqs, sps, use_tqdm=False)
    return [(k, j, o.outputs[0]) for (k, j), o in zip(meta, outs)]


def capture(ids, dnames, layers):
    """Teacher-forced pass of one sequence (current edits apply); returns {layer: [T, len(dnames)] float32}."""
    STATE["capM"] = {l: torch.tensor(np.stack([DIRS[d][l] for d in dnames]), dtype=torch.float32, device=DEV) for l in layers}
    STATE["cap"] = {l: [] for l in layers}
    llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
    out = {l: torch.cat(STATE["cap"][l]).numpy() for l in layers}
    STATE["cap"], STATE["capM"] = None, {}
    for l in layers:
        assert out[l].shape[0] >= len(ids), (l, out[l].shape, len(ids))
        out[l] = out[l][:len(ids)]
    return out


cap = [json.loads(l) for l in open(args.capability)]
acc_items = [(c, chat_ids(c["prompt"])) for c in cap if c["kind"] in ("mmlu_style", "gsm8k_style")]
benign = [json.loads(l) for l in open(f"{args.p1}/benign.jsonl")]
bdef = [r for r in benign if r["style"] == "default"]
cap_seqs = [tuple(x) for x in json.load(open(f"{args.p3}/cap_replies.json"))]  # P3's clean capability replies (text only)
if args.smoke:
    cap_seqs, acc_items = cap_seqs[:12], acc_items[:8]
bseqs = [(r["prompt_ids"], r["ids"]) for r in bdef]
checks_path, tau_path, dmg_path, acc_path = (f"{args.out}/{x}" for x in ("checks.json", "tau.json", "damage.json", "acc.jsonl"))
checks = json.load(open(checks_path)) if os.path.exists(checks_path) else {}
damage = json.load(open(dmg_path)) if os.path.exists(dmg_path) else {}

# ---------------------------------------------------------------- phase tau
TAU = {}
if os.path.exists(tau_path):
    TAU = {d: {int(p): {int(l): v for l, v in x.items()} for p, x in y.items()} for d, y in json.load(open(tau_path)).items()}
need = [d for d in DNAMES if d not in TAU or any(p not in TAU[d] for p in PCTS)]
if need:
    set_edits(None)
    projs = {l: [] for l in CAP_LAYERS}
    for r in bdef:
        c = capture(r["prompt_ids"] + r["ids"], need, CAP_LAYERS)
        P, R = len(r["prompt_ids"]), len(r["ids"])
        for l in CAP_LAYERS:
            projs[l].append(c[l][P:P + R])
    for k, d in enumerate(need):
        new = {p: {l: float(np.percentile(np.concatenate(projs[l])[:, k], p)) for l in CAP_LAYERS} for p in PCTS}
        if d in TAU:  # recomputed values must match the stored ones (deterministic pass)
            dev = max(abs(new[p][l] - TAU[d][p][l]) for p in TAU[d] for l in CAP_LAYERS)
            checks.setdefault("tau_recompute_max_abs_dev", {})[d] = dev
            assert dev < 0.02, (d, dev)
        TAU[d] = new
    if "axis" in need:
        x = {l: np.concatenate(projs[l])[:, need.index("axis")] for l in CAP_LAYERS}
        fb = {l: float((x[l] < TAU_AX_P1[l]).mean()) for l in range(46, 54)}
        checks["tau"] = {"axis_tau_p1": TAU_AX_P1, "axis_p25_here": {l: round(TAU["axis"][25][l], 3) for l in CAP_LAYERS},
                         "axis_frac_below_p1_tau": fb, "n_tokens": int(x[46].shape[0]),
                         "axis_p50": {l: round(TAU["axis"][50][l], 3) for l in CAP_LAYERS},
                         "axis_p75": {l: round(TAU["axis"][75][l], 3) for l in CAP_LAYERS}}
        log("tau check: frac of benign tokens below P1 tau at 46-53", {l: round(v, 3) for l, v in fb.items()})
        bad = [l for l, v in fb.items() if abs(v - 0.25) > 0.06]
        if bad:
            jdump(checks, checks_path + ".failed")
            log("ABORT: hook does not reproduce P1's axis projections at layers", bad)
            raise SystemExit(2)
    jdump({d: {str(p): {str(l): v for l, v in x.items()} for p, x in y.items()} for d, y in TAU.items()}, tau_path)
    jdump(checks, checks_path)
if "cap_effect" not in checks:
    set_edits(None)
    n0 = np.concatenate(nll_tokens(bseqs))
    set_edits([{"type": "cap", "dir": "axis", "layers": LRANGES["L46-53"], "pct": 25}])
    n1 = np.concatenate(nll_tokens(bseqs))
    fr, ds = cap_stats()
    set_edits(None)
    n2 = np.concatenate(nll_tokens(bseqs))
    checks["cap_effect"] = {"delta_nll_axis_cap": float((n1 - n0).mean()), "mean_abs_dnll": float(np.abs(n1 - n0).mean()),
                            "clean_repeat_max_abs": float(np.abs(n2 - n0).max()), "cap_frac": fr, "disp_per_tok": ds}
    log("cap effect check", checks["cap_effect"])
    if checks["cap_effect"]["mean_abs_dnll"] < 1e-5 or checks["cap_effect"]["clean_repeat_max_abs"] > 1e-3:
        jdump(checks, checks_path + ".failed")
        log("ABORT: cap has no effect on logits, or the hook leaks into the clean state")
        raise SystemExit(3)
    jdump(checks, checks_path)

# ---------------------------------------------------------------- side effects (P3's)
set_edits(None)
clean_nll = {"capability_mini": nll_tokens(cap_seqs), "p1_benign": nll_tokens(bseqs)}


def mcq_letter(t):
    for pat in (r"(?i)answer\s*(?:is|:)?\s*\**\s*\(?([ABCD])\b", r"^\W*\(?([ABCD])\b", r"\b([ABCD])[\.\)]", r"\b([ABCD])\b"):
        m = re.search(pat, t)
        if m:
            return m.group(1)
    return None


def last_num(t):
    xs = re.findall(r"-?\d[\d,]*\.?\d*", t)
    if not xs:
        return None
    try:
        return float(xs[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def dnll_only(name, edits):
    set_edits(edits)
    nll = {"capability_mini": nll_tokens(cap_seqs), "p1_benign": nll_tokens(bseqs)}
    fr, ds = cap_stats()
    dm = damage.get(name, {})
    for k in nll:
        a, c0 = np.concatenate(nll[k]), np.concatenate(clean_nll[k])
        dm[k] = {"delta_nll_per_token": float((a - c0).mean()), "n_tokens": int(len(a)), "clean_nll_per_token": float(c0.mean())}
    dm["cap_frac_dmg_pass"], dm["disp_per_tok_dmg_pass"] = fr, ds
    dm["edits"] = [{k: v for k, v in e.items() if k != "tau"} for e in (edits or [])]
    damage[name] = dm
    jdump(damage, dmg_path)
    set_edits(None)
    return dm


def accuracy(name, edits):
    set_edits(edits)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for _, p in acc_items],
                        SamplingParams(max_tokens=400, temperature=0.0), use_tqdm=False)
    nc = 0
    with open(acc_path, "a") as fa:
        for (c, _), o in zip(acc_items, outs):
            t = o.outputs[0].text
            if c["kind"] == "mmlu_style":
                pred = mcq_letter(t); ok = pred == c["answer"]
            else:
                pred = last_num(t); ok = pred is not None and abs(pred - float(c["answer"])) < 1e-6
            nc += ok
            fa.write(json.dumps({"cond": name, "id": c["id"], "kind": c["kind"], "pred": pred, "ok": bool(ok),
                                 "text": t, "finish": o.outputs[0].finish_reason}) + "\n")
    damage[name]["acc"] = nc / len(acc_items)
    jdump(damage, dmg_path)
    set_edits(None)


def gate_val(dm):
    return max(dm["capability_mini"]["delta_nll_per_token"], dm["p1_benign"]["delta_nll_per_token"])


def run_condition(name, edits, items, n, gen_path, extra, id_path=None):
    """Generate for items under edits, identity check (if id_path), damage + accuracy; append records last."""
    t1 = time.time()
    set_edits(edits)
    outs = sample(items, n, args.max_tokens)
    fr, ds = cap_stats()
    ntok = sum(len(o.token_ids) for _, _, o in outs)
    tg = time.time() - t1
    if id_path:
        set_edits(edits)
        io = sample([(f"id{qi}", chat_ids(q)) for qi, q in enumerate(ID_PROMPTS)], args.id_n, 512)
        with open(id_path, "a") as fi:
            for k, j, o in io:
                qi = int(k[2:])
                fi.write(json.dumps({"cond": name, "qid": qi, "prompt": ID_PROMPTS[qi], "j": j, "text": o.text,
                                     "n_tok": len(o.token_ids), "finish": o.finish_reason}) + "\n")
    if name not in damage or "capability_mini" not in damage[name]:
        dnll_only(name, edits)
    if "acc" not in damage[name]:
        accuracy(name, edits)
    damage[name].setdefault("cap_frac_gen", {})[os.path.basename(gen_path)] = fr
    damage[name].setdefault("disp_per_tok_gen", {})[os.path.basename(gen_path)] = ds
    jdump(damage, dmg_path)
    set_edits(None)
    with open(gen_path, "a") as fo:
        for key, j, o in outs:
            rec = {"cond": name, "j": j, "text": o.text, "n_tok": len(o.token_ids), "finish": o.finish_reason}
            rec.update(extra(key))
            fo.write(json.dumps(rec) + "\n")
    d = damage[name]
    log(f"{name}: {len(items)} x {n}, {ntok} tok in {tg:.0f}s ({ntok/max(tg,1):.0f} tok/s); dNLL cap "
        f"{d['capability_mini']['delta_nll_per_token']:.4f} benign {d['p1_benign']['delta_nll_per_token']:.4f} acc {d['acc']:.3f}; "
        f"cap frac {min(fr.values()) if fr else '-'}-{max(fr.values()) if fr else '-'} disp {ds:.2f}")


def done_conds(p):
    return {x["cond"] for x in jl(p)}


meta_path = f"{args.out}/gen_meta.json"
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
meta.update({"sampling": dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens,
                              n=args.n), "prompts": prompts, "split": {p: gen[p]["split"] for v in prompts.values() for p in v},
             "rate_on": p3meta["rate_on"], "vllm": __import__("vllm").__version__, "gpu": torch.cuda.get_device_name(0),
             "engine": f"vLLM in-process, enforce_eager, prefix caching off, bf16, max_model_len {args.max_model_len}",
             "id_prompts": ID_PROMPTS})


def ax(p, lr):
    return f"ax_p{p}_{lr}"


SETTINGS = {ax(p, lr): [{"type": "cap", "dir": "axis", "layers": LRANGES[lr], "pct": p}] for lr in LRANGES for p in PCTS}

# ---------------------------------------------------------------- phase b1
if "b1" in PH:
    if "clean" not in damage:
        dnll_only("clean", None)
    for s, e in SETTINGS.items():
        if s not in damage or "disp_per_tok_dmg_pass" not in damage[s]:
            dm = dnll_only(s, e)
            log(f"damage {s}: cap {dm['capability_mini']['delta_nll_per_token']:.4f} benign {dm['p1_benign']['delta_nll_per_token']:.4f} "
                f"disp/tok {dm['disp_per_tok_dmg_pass']:.2f}")
    if "strongest" not in meta:
        ok = [s for s in SETTINGS if gate_val(damage[s]) <= args.damage_gate]
        assert ok, "no axis setting passes the damage gate"
        best = max(ok, key=lambda s: damage[s]["disp_per_tok_dmg_pass"])
        meta["strongest"] = {"setting": best, "passing": ok,
                             "disp": {s: damage[s]["disp_per_tok_dmg_pass"] for s in SETTINGS},
                             "gate": {s: gate_val(damage[s]) for s in SETTINGS},
                             "rule": "largest mean cap displacement per token on P1 benign among settings with max(dNLL capability, dNLL benign) <= gate"}
        log("strongest passing setting", best, meta["strongest"])
    best = meta["strongest"]["setting"]
    bp, blr = int(best.split("_")[1][1:]), best.split("_")[2]
    RPC = [f"rp{j}" for j in args.rp_ctrl.split(",")]
    B1C = [("clean", None)] + [(s, e) for s, e in SETTINGS.items()] + \
          [(f"rpcap_{r}_p{bp}_{blr}", [{"type": "cap", "dir": r, "layers": LRANGES[blr], "pct": bp}]) for r in RPC]
    meta["b1_conditions"] = [c for c, _ in B1C]
    meta["rp_ctrl"] = {r: pairs[int(r[2:])] for r in RPC}
    meta["rp_cos_axis"] = {r: round(float(np.mean([DIRS[r][l] @ DIRS["axis"][l] for l in LRANGES[blr]])), 3) for r in RPC}
    jdump(meta, meta_path)
    gp, ip = f"{args.out}/b1_gen.jsonl", f"{args.out}/b1_id.jsonl"
    items = [(pid, gen[pid]["prompt_ids"]) for b in BEH for pid in prompts[b]]
    for name, e in B1C:
        if name in done_conds(gp):
            log(f"skip {name} (done)")
            continue
        run_condition(name, e, items, args.n, gp, lambda pid: {"b": gen[pid]["b"], "prompt_id": pid, "split": gen[pid]["split"]}, ip)
    log("B1_DONE")

# ---------------------------------------------------------------- phase b3
if "b3" in PH:
    # B3 spec (written from picks.json): {"directions": {...}, "tau_dirs": [...], "groups": [{"name", "gen", "items":
    # [{"key", "b", "prompt_id"?, "prompt_text"?, "split"?}], "n", "conds": [{"name", "edits"}], "id": bool}]}
    checks["b3_dir_cos_vs_picks"] = dir_cos
    jdump(checks, checks_path)
    jdump(meta, meta_path)
    for grp in B3["groups"]:
        gp = f"{args.out}/{grp['gen']}"
        IT = {x["key"]: x for x in grp["items"]}
        items = [(x["key"], gen[x["prompt_id"]]["prompt_ids"] if x.get("prompt_id") else chat_ids(x["prompt_text"])) for x in grp["items"]]
        for c in grp["conds"]:
            if c["name"] in done_conds(gp):
                log(f"skip b3 {grp['name']} {c['name']} (done)")
                continue
            its = [it for it in items if IT[it[0]]["b"] in c["only_b"]] if c.get("only_b") else items
            run_condition(c["name"], c["edits"], its, grp["n"], gp,
                          lambda key: dict(IT[key]) | {"key": key},
                          f"{args.out}/b3_id.jsonl" if c.get("id") else None)
    log("B3_DONE")
jdump(meta, meta_path)
log("ALL_DONE")
