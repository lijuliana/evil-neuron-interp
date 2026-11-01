"""persona-fwd X2 / X3 fork-value tests (GPU, forward passes only, resumable).

Cut points: P1's informative forks (>= 2 next-token candidates, q range >= .25; next_tokens from phase2_forks if
present, else forks_off) for all 11 behaviors (all on discovery prompts) + held-out denying-AI forks rebuilt from
forks_samples_on_dai_held.jsonl (as scripts/73_persona_fwd_x1.py). Each fork gets a prompt-level half (A/B, sha256
of prompt_id, alternating within behavior; 'H' = held-out DAI): settings are chosen on A and reported on B.
V = sum p q / sum p over candidates.

Directions (block L = output of decoder layer L):
  axis: P1's community axis lerped between saved blocks (+ = Assistant end); rp<j>: P1's 20 role-pair differences
  (c1x_pairs.json), lerped; role<i>: d_i = unit(lerp(default - role_i)), + = default end, i over all 265 roles;
  traits: community unit trait vectors <trait>_response_avg at each layer (+ = more trait); rand<j>: P1's R32
  isotropic unit directions (same vector at any block).
Phases:
  tau  benign projections (P1's 24 benign default replies, response tokens) -> tau.pt: q25 / mean for axis, rp, roles
       at layers 46-53; q25/q50/q75/q90/mean/sd for the 9 traits at layers 16-55.
  x2   rows from layer 46: base, cap_axis, cap_rp<j>, cap_role<i>; cap = Lu et al. floor h <- h - v*min(<h,v>-tau25, 0)
  x3   steer<Ls>_<dose>|<dir> (h_Ls += -dose * median_norm[Ls] * v, Ls in 24/32/40, dose .05/.1/.25, dir = 9 traits +
       10 random); out_<R>|<trait> (h_l -= v<h_l,v>, every layer in R); ceil_<R>|<trait> (h_l -= v*max(<h_l,v>-tau75, 0));
       R: mid 24-39, late 40-55, cap 46-53
  kl   benign KL(base || row), 8 P1 benign default replies >= 60 tokens, for every x2 / x3 row, plus ceil at the 50th and
       90th percentiles (for KL-matching controls by percentile)
Outputs (--out): tau.pt, kl_fork.json, x2.jsonl, x3.jsonl, fork_items.json
Run: flock /data/gpu.lock ~/venv/bin/python -I scripts/73_persona_fwd_fork.py --phase tau,kl,x2,x3
"""
import argparse, collections, gzip, hashlib, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from harness import Harness  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/data/agastya")
ap.add_argument("--p1", default="/data/runs/persona_p1")
ap.add_argument("--norms", default=os.path.expanduser("~/evil-neuron-interp/captures/qwen3.6-27b_resid_norms.json"))
ap.add_argument("--out", default="/data/runs/persona_fwd")
ap.add_argument("--phase", default="tau,kl,x2,x3")
ap.add_argument("--tok_cap", type=int, default=49152)
ap.add_argument("--n_rand", type=int, default=10)
ap.add_argument("--behaviors", default="", help="restrict forks (comma list of behavior ids)")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--x3m", default="ceil_late,steer40_0.1", help="X3 settings that get KL-matched controls (phase x3m)")
ap.add_argument("--reverse", action="store_true", help="longest forks first (to split a phase across two GPUs)")
args = ap.parse_args()
T0 = time.time()
os.makedirs(args.out, exist_ok=True)
PH = args.phase.split(",")
dev = "cuda"
D = 5120


def log(*a):
    print(f"[fork {(time.time()-T0)/60:5.1f}m]", *a, flush=True)


H = Harness(resid_norms_path=args.norms)
lm, layers, tok = H.lm, H.layers, H.tok
lm_head = H.model.get_output_embeddings()
LT = lm.config.layer_types
MED = H.resid_norms
log("model loaded")

# ------------------------------------------------------------------ directions
AX_BLOCKS = [8, 16, 24, 32, 40, 48, 56]
CAP = list(range(46, 54))
TRAITS = ["apathetic", "dismissive", "evil", "hallucinating", "helpful", "humorous", "impolite", "optimistic", "sycophantic"]
V = f"{args.p1}/vectors"
axis_raw = np.load(f"{V}/axis_raw.npy").astype(np.float64)
roles = np.load(f"{V}/roles_raw.npy").astype(np.float64)
default = np.load(f"{V}/default_raw.npy").astype(np.float64)
NRL = len(roles)
pairs = json.load(open(f"{args.p1}/c1x_pairs.json"))["pairs"]
tz = np.load(f"{V}/traits_unit.npz")


def unit(x):
    return x / max(np.linalg.norm(x), 1e-12)


def lerp_raw(x, l):
    if l <= 8:
        return x[0]
    if l >= 56:
        return x[6]
    i = (l - 8) // 8
    w = (l - AX_BLOCKS[i]) / 8.0
    return (1 - w) * x[i] + w * x[i + 1]


def T32(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)


CAPD = {}  # name -> [8, D] at CAP
CAPD["axis"] = T32(np.stack([unit(lerp_raw(axis_raw, l)) for l in CAP]))
for j, (a, b) in enumerate(pairs):
    CAPD[f"rp{j}"] = T32(np.stack([unit(lerp_raw(roles[a] - roles[b], l)) for l in CAP]))
for i in range(NRL):
    CAPD[f"role{i}"] = T32(np.stack([unit(lerp_raw(default - roles[i], l)) for l in CAP]))
CAPN = list(CAPD)
CAPM = torch.stack([CAPD[n] for n in CAPN])  # [286, 8, D]
TL = list(range(16, 56))
TRD = {t: T32(tz[f"{t}_response_avg"][:64]) for t in TRAITS}  # [64, D] (row 63 = 0)
RND = T32(np.load(f"{args.p1}/random_dirs.npz")["R32"][:args.n_rand])

# ------------------------------------------------------------------ runner


def run_from(h, L0, L1=64):
    B, T, _ = h.shape
    pos = torch.arange(T, device=dev).view(1, 1, -1).expand(4, B, -1)
    tpos, pos3 = pos[0], pos[1:]
    mk = dict(config=lm.config, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=tpos)
    masks = {"full_attention": create_causal_mask(**mk), "linear_attention": None}
    pe = lm.rotary_emb(h, pos3)
    for i in range(L0, L1):
        h = layers[i](h, position_embeddings=pe, attention_mask=masks[LT[i]], position_ids=tpos, past_key_values=None)
    return h


class Hooks:
    def __init__(self):
        self.hs = []

    def fwd(self, mod, fn):
        self.hs.append(mod.register_forward_hook(fn))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for h in self.hs:
            h.remove()


MODE = {"floor": 1, "out": 2, "ceil": 4}


def run_rows(h_src, L0, rows, out_mode, cands=None, span=None, base_lp=None):
    """rows: dicts with 'edits': {layer: ('add', delta[D]) | (mode, v[D], tau)} (no edits = base)."""
    T = h_src.shape[1]
    per = max(2, args.tok_cap // T) - (1 if out_mode == "last" else 0)
    res = []
    for s in range(0, len(rows), per):
        chunk = rows[s:s + per]
        if out_mode == "last":  # reference row (no edits) inside every chunk: bf16 results depend on batch shape
            chunk = [{"name": "_chunkbase"}] + chunk
        B = len(chunk)
        lays = sorted({l for r in chunk for l in r.get("edits", {})})
        spec = {}
        for l in lays:
            add = torch.zeros(B, D, device=dev); U = torch.zeros(B, D, device=dev)
            tau = torch.zeros(B, device=dev); mode = torch.zeros(B, dtype=torch.long, device=dev)
            has_add = has_u = False
            for r_i, r in enumerate(chunk):
                e = r.get("edits", {}).get(l)
                if e is None:
                    continue
                if e[0] == "add":
                    add[r_i] = e[1]; has_add = True
                else:
                    U[r_i] = e[1]; tau[r_i] = e[2]; mode[r_i] = MODE[e[0]]; has_u = True
            spec[l] = (add if has_add else None, U if has_u else None, tau, mode)
        h = h_src.expand(B, -1, -1).contiguous()
        with torch.no_grad(), Hooks() as hk:
            for l, (add, U, tau, mode) in spec.items():
                def hook(mod, a, o, add=add, U=U, tau=tau, mode=mode):
                    hf = o.float()
                    if U is not None:
                        pr = torch.einsum("btd,bd->bt", hf, U)
                        m = mode[:, None]
                        d = pr - tau[:, None]
                        coef = torch.where(m == 1, d.clamp(max=0), torch.where(m == 2, pr, torch.where(m == 4, d.clamp(min=0), torch.zeros_like(pr))))
                        hf = hf - coef[..., None] * U[:, None, :]
                    if add is not None:
                        hf = hf + add[:, None, :]
                    return hf.to(o.dtype)
                hk.fwd(layers[l], hook)
            hout = run_from(h, L0)
            if out_mode == "last":
                pr = torch.softmax(lm_head(lm.norm(hout[:, -1])).float(), -1)[:, cands].cpu()
                cb = [round(x, 7) for x in pr[0].tolist()]
                res += [([round(x, 7) for x in pr[r].tolist()], cb) for r in range(1, B)]
            else:
                a0, a1 = span
                for r in range(B):
                    lp = torch.log_softmax(lm_head(lm.norm(hout[r, a0:a1])).float(), -1)
                    res.append(float((base_lp.exp() * (base_lp - lp)).sum(-1).mean()))
        del hout, h
    return res


def base_pass(ids, stash_layers):
    st = {}
    with torch.no_grad(), Hooks() as hk:
        for l in stash_layers:
            def f(mod, a, o, l=l):
                st[l] = o.clone()
            hk.fwd(layers[l], f)
        h = run_from(lm.embed_tokens(torch.tensor([ids], device=dev)), 0)
    return st, h


# ------------------------------------------------------------------ benign texts
benign = [json.loads(l) for l in open(f"{args.p1}/benign.jsonl")]
bdef = [r for r in benign if r["style"] == "default"]

# ================================================================== tau
taup = f"{args.out}/tau.pt"
if "tau" in PH:
    pc = {l: [] for l in CAP}
    pt = {l: [] for l in TL}
    TM = {l: torch.stack([TRD[t][l] for t in TRAITS]) for l in TL}  # [9, D]
    for r in bdef:
        ids = r["prompt_ids"] + r["ids"]
        P, R = len(r["prompt_ids"]), len(r["ids"])

        def mk(l):
            def hook(mod, a, o):
                hs = o[0, P:P + R].float()
                if l in CAP:
                    pc[l].append(hs @ CAPM[:, CAP.index(l)].T)
                if l in TL:
                    pt[l].append(hs @ TM[l].T)
            return hook
        with torch.no_grad(), Hooks() as hk:
            for l in sorted(set(CAP) | set(TL)):
                hk.fwd(layers[l], mk(l))
            run_from(lm.embed_tokens(torch.tensor([ids], device=dev)), 0, 56)
    tau = {"cap_names": CAPN, "traits": TRAITS,
           "cap_q25": {l: torch.quantile(torch.cat(pc[l]), .25, dim=0).cpu() for l in CAP},
           "cap_mean": {l: torch.cat(pc[l]).mean(0).cpu() for l in CAP},
           "trait_q": {q: {l: torch.quantile(torch.cat(pt[l]), q, dim=0).cpu() for l in TL} for q in (.25, .5, .75, .9)},
           "trait_mean": {l: torch.cat(pt[l]).mean(0).cpu() for l in TL},
           "trait_sd": {l: torch.cat(pt[l]).std(0).cpu() for l in TL},
           "n_tokens": int(sum(x.shape[0] for x in pc[46]))}
    torch.save(tau, taup)
    p1t = torch.load(f"{args.p1}/cap_tau.pt")["tau"]
    log("tau done; axis q25 vs P1", [(l, round(float(tau["cap_q25"][l][0]), 2), round(float(p1t[l][0]), 2)) for l in (46, 50, 53)],
        "syc q75 L24/40/48", [round(float(tau["trait_q"][.75][l][TRAITS.index("sycophantic")]), 2) for l in (24, 40, 48)],
        "syc mean", [round(float(tau["trait_mean"][l][TRAITS.index("sycophantic")]), 2) for l in (24, 40, 48)])
tau = torch.load(taup)

# ------------------------------------------------------------------ row builders


def x2_rows():
    rows = [{"name": "base", "L0": 46}]
    for k, n in enumerate(CAPN):
        rows.append({"name": f"cap_{n}", "L0": 46,
                     "edits": {l: ("floor", CAPM[k, li], float(tau["cap_q25"][l][k])) for li, l in enumerate(CAP)}})
    return rows


RANGES = {"mid": list(range(24, 40)), "late": list(range(40, 56)), "cap": CAP}
STEER_L = [24, 32, 40]
DOSES = [0.05, 0.1, 0.25]


def x3_rows(extra_pcts=False):
    rows = [{"name": "base", "L0": 46}]
    dirs = [(t, TRD[t]) for t in TRAITS] + [(f"rand{j}", None) for j in range(args.n_rand)]
    for Ls in STEER_L:
        for d in DOSES:
            for name, M in dirs:
                v = M[Ls] if M is not None else RND[int(name[4:])]
                rows.append({"name": f"steer{Ls}_{d}|{name}", "L0": Ls, "edits": {Ls: ("add", -d * MED[Ls] * v)}})
    for rn, ls in RANGES.items():
        for ti, t in enumerate(TRAITS):
            rows.append({"name": f"out_{rn}|{t}", "L0": ls[0], "edits": {l: ("out", TRD[t][l], 0.0) for l in ls}})
            pcts = (.75, .5, .9) if extra_pcts else (.75,)
            for q in pcts:
                nm = f"ceil_{rn}|{t}" if q == .75 else f"ceil{int(q*100)}_{rn}|{t}"
                rows.append({"name": nm, "L0": ls[0], "edits": {l: ("ceil", TRD[t][l], float(tau["trait_q"][q][l][ti])) for l in ls}})
    return rows


def run_grouped(ids, rows, out_mode, cands=None, span=None):
    """Group rows by start layer; one base pass stashing each group's input."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["L0"]].append(r)
    st, hfull = base_pass(ids, sorted({L0 - 1 for L0 in groups}))
    base_lp = None
    if out_mode == "span":
        base_lp = torch.log_softmax(lm_head(lm.norm(hfull[0, span[0]:span[1]])).float(), -1)
        pfull = None
    else:
        pfull = torch.softmax(lm_head(lm.norm(hfull[0, -1])).float(), -1)[cands].cpu().tolist()
    del hfull
    out = {}
    for L0, rs in groups.items():
        for r, v in zip(rs, run_rows(st[L0 - 1], L0, rs, out_mode, cands=cands, span=span, base_lp=base_lp)):
            out[r["name"]] = v
    del st, base_lp
    return out, pfull


# ================================================================== kl
klp = f"{args.out}/kl_fork.json"
if "kl" in PH:
    kseqs = [r for r in bdef if len(r["ids"]) >= 60][: (2 if args.smoke else 8)]
    kl = collections.defaultdict(list)
    rows = x2_rows() + x3_rows(extra_pcts=True)[1:]
    for r in kseqs:
        ids = r["prompt_ids"] + r["ids"]
        P, R = len(r["prompt_ids"]), len(r["ids"])
        o, _ = run_grouped(ids, rows, "span", span=(P - 1, P + R - 1))
        for k, v in o.items():
            kl[k].append(v)
        torch.cuda.empty_cache()
    kls = {k: float(np.mean(v)) for k, v in kl.items()}
    json.dump(kls, open(klp, "w"), indent=1)
    log("kl done", {k: round(kls[k], 4) for k in ("base", "cap_axis", "steer32_0.25|sycophantic", "steer32_0.25|rand0",
                                                  "out_cap|sycophantic", "ceil_cap|sycophantic", "out_mid|sycophantic", "ceil_mid|sycophantic")})

# ================================================================== x3m: KL-matched controls for chosen X3 settings
X3M_ROWS = []
if "x3m" in PH:
    klf = json.load(open(klp))
    kseqs = [r for r in bdef if len(r["ids"]) >= 60][: (2 if args.smoke else 8)]
    PB = {l: [] for l in TL}
    TM = {l: torch.stack([TRD[t][l] for t in TRAITS]) for l in TL}
    for r in bdef:
        ids = r["prompt_ids"] + r["ids"]
        P, R = len(r["prompt_ids"]), len(r["ids"])

        def mk(l):
            def hook(mod, a, o):
                PB[l].append(o[0, P:P + R].float() @ TM[l].T)
            return hook
        with torch.no_grad(), Hooks() as hk:
            for l in TL:
                hk.fwd(layers[l], mk(l))
            run_from(lm.embed_tokens(torch.tensor([ids], device=dev)), 0, 56)
    PB = {l: torch.cat(v) for l, v in PB.items()}
    np.savez_compressed(f"{args.out}/benign_trait_proj.npz", traits=np.array(TRAITS), layers=np.array(TL),
                        **{f"L{l}": PB[l].cpu().numpy().astype(np.float32) for l in TL})
    if args.x3m == "none":
        raise SystemExit(0)

    def benign_kl(rows):
        acc = collections.defaultdict(list)
        for r in kseqs:
            ids = r["prompt_ids"] + r["ids"]
            P, R = len(r["prompt_ids"]), len(r["ids"])
            o, _ = run_grouped(ids, rows, "span", span=(P - 1, P + R - 1))
            for k, v in o.items():
                acc[k].append(v)
        return {k: float(np.mean(v)) for k, v in acc.items()}

    def mk_row(s, d, strength, name):
        """s: setting; d: trait name or rand<j>; strength: ceil percentile (ceil) or dose (steer / random)."""
        if d.startswith("rand") or s.startswith("steer"):
            Ls = int(s[5:].split("_")[0]) if s.startswith("steer") else {"mid": 24, "late": 40, "cap": 40}[s.split("_")[1]]
            v = RND[int(d[4:])] if d.startswith("rand") else TRD[d][Ls]
            return {"name": name, "L0": Ls, "edits": {Ls: ("add", -strength * MED[Ls] * v)}}
        ls = RANGES[s.split("_")[1]]
        ti = TRAITS.index(d)
        return {"name": name, "L0": ls[0], "edits": {l: ("ceil", TRD[d][l], float(torch.quantile(PB[l][:, ti], strength))) for l in ls}}

    match = {}
    dirs_c = [t for t in TRAITS if t != "sycophantic"] + [f"rand{j}" for j in range(args.n_rand)]
    for s in args.x3m.split(","):
        target = klf[f"{s}|sycophantic"]
        grid = {}
        rows = []
        for d in dirs_c:
            if d.startswith("rand") or s.startswith("steer"):
                g = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
            else:
                g = [0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]
            grid[d] = g
            rows += [mk_row(s, d, x, f"g|{d}|{x}") for x in g]
        kg = benign_kl(rows)
        strengths = {}
        for d in dirs_c:
            g = np.array(grid[d]); k = np.log(np.maximum([kg[f"g|{d}|{x}"] for x in g], 1e-7))
            o = np.argsort(k)
            strengths[d] = float(np.interp(np.log(target), k[o], g[o]))
        vrows = [mk_row(s, d, strengths[d], f"{s}|m|{d}") for d in dirs_c]
        kv = benign_kl(vrows)
        match[s] = {"target_kl": target, "strength": strengths, "kl_verified": {d: kv[f"{s}|m|{d}"] for d in dirs_c},
                    "grid_kl": {d: [kg[f"g|{d}|{x}"] for x in grid[d]] for d in dirs_c}, "grid": grid}
        log(f"x3m {s}: target {target:.4f}", {d: (round(strengths[d], 3), round(kv[f'{s}|m|{d}'], 4)) for d in dirs_c})
        syc_row = mk_row(s, "sycophantic", .75 if s.startswith("ceil") else float(s.split("_")[1]), f"{s}|sycophantic")
        X3M_ROWS += [syc_row] + vrows
    X3M_ROWS = [{"name": "base", "L0": 46}] + X3M_ROWS
    json.dump(match, open(f"{args.out}/x3m_match.json", "w"), indent=1)

# ================================================================== items
gen = {}
for line in gzip.open(f"{args.data}/stage_a_gen.jsonl.gz", "rt"):
    r = json.loads(line)
    gen[r["prompt_id"]] = r
split = json.load(open(f"{args.data}/split.json"))
forks = [json.loads(l) for l in open(f"{args.data}/forks_off.jsonl")]
p2 = {json.loads(l)["rid"]: json.loads(l) for l in open(f"{args.data}/phase2_forks.jsonl")}
items = []
for f in forks:
    nt = p2[f["rid"]]["next_tokens"] if f["rid"] in p2 else f["next_tokens"]
    if len(nt) < 2 or max(c["q"] for c in nt) - min(c["q"] for c in nt) < 0.25:
        continue
    pid, i = f["rid"].split("|")
    r_ids = gen[pid]["samples"][int(i)]["ids"]
    lo = f["fork"][0]
    assert r_ids[lo] == f["fork_token"]
    items.append({"rid": f["rid"], "b": f["behavior_id"], "pid": pid, "split": split[pid]["split"], "ids": gen[pid]["prompt_ids"] + r_ids[:lo],
                  "cands": [c["tok"] for c in nt], "q": [c["q"] for c in nt], "src": "forks_off"})
cut = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for l in open(f"{args.data}/forks_samples_on_dai_held.jsonl"):
    x = json.loads(l)
    c = cut[x["rid"]][x["cut"]]
    c[0] += sum(x["match"]); c[1] += len(x["match"])
for rid, cs in cut.items():
    br = [k for k in cs if "+" in k]
    if not br:
        continue
    c0 = int(br[0].split("+")[0])
    pid, i = rid.split("|")
    r_ids = gen[pid]["samples"][int(i)]["ids"]
    if str(c0 + 1) not in cs:
        continue
    q = [cs[str(c0 + 1)][0] / cs[str(c0 + 1)][1]] + [cs[k][0] / cs[k][1] for k in br]
    if max(q) - min(q) < 0.25:
        continue
    items.append({"rid": rid, "b": "denying-ai-identity", "pid": pid, "split": split[pid]["split"], "ids": gen[pid]["prompt_ids"] + r_ids[:c0],
                  "cands": [r_ids[c0]] + [int(k.split("+")[1]) for k in br], "q": q, "src": "dai_held_on"})
half = {}
for b in sorted({x["b"] for x in items}):
    ps = sorted({x["pid"] for x in items if x["b"] == b and x["src"] == "forks_off"}, key=lambda p: hashlib.sha256(p.encode()).hexdigest())
    for k, p in enumerate(ps):
        half[p] = "A" if k % 2 == 0 else "B"
for x in items:
    x["half"] = "H" if x["src"] == "dai_held_on" else half[x["pid"]]
json.dump([{k: v for k, v in x.items() if k != "ids"} | {"T": len(x["ids"])} for x in items], open(f"{args.out}/fork_items.json", "w"))
if args.behaviors:
    items = [x for x in items if x["b"] in args.behaviors.split(",")]
if args.smoke:
    items = [x for x in items if x["b"] == "denying-ai-identity"][:2] + [x for x in items if x["b"] == "recommends-drunk-driving"][:1]
log(f"{len(items)} forks", collections.Counter((x["b"][:10], x["half"]) for x in items))


def run_phase(name, rows):
    outp = f"{args.out}/{name}.jsonl"
    done = {json.loads(l)["rid"] for l in open(outp)} if os.path.exists(outp) else set()
    n = 0
    t_p = time.time()
    with open(outp, "a") as fh:
        for it in sorted(items, key=lambda x: len(x["ids"]), reverse=args.reverse):
            if it["rid"] in done:
                continue
            tf = time.time()
            o, pfull = run_grouped(it["ids"], rows, "last", cands=it["cands"])
            rec = {k: it[k] for k in ("rid", "b", "split", "half", "src", "cands", "q")}
            cbs, rows_o = [], {}
            for k, (p, cb) in o.items():  # rows[name] = [p, index of its chunk's no-edit reference in 'cb']
                if cb not in cbs:
                    cbs.append(cb)
                rows_o[k] = [p, cbs.index(cb)]
            rec.update(T=len(it["ids"]), p_full=pfull, cb=cbs, rows=rows_o)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            if n <= 3 or n % 25 == 0:
                log(f"{name} fork {n} {it['b'][:12]} T={len(it['ids'])} {time.time()-tf:.1f}s")
            torch.cuda.empty_cache()
    log(f"{name} done: {n} forks in {(time.time()-t_p)/60:.1f} min")


if "x2" in PH:
    run_phase("x2", x2_rows())
if "x3" in PH:
    run_phase("x3", x3_rows())
if "x3m" in PH:
    run_phase("x3m", X3M_ROWS)
log("ALL_DONE")
