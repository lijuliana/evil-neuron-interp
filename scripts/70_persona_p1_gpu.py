"""P1 persona job (forward passes only, no judge). Run on the VM:
  flock /data/gpu.lock ~/venv/bin/python -I scripts/70_persona_p1_gpu.py [--smoke]

Phases (outputs under --out):
  benign  24 benign questions x {default, florid, plain} sampled once (system prompt only at generation), then
          teacher-forced WITHOUT the system prompt: style_raw.npy [64,5120] = mean(florid) - mean(plain) response
          residual (block L = output of layers[L]); default_mean.npy; capping thresholds tau (25th pct of default
          response tokens) for the axis and the 20 random directions at layers 46-53; benign.jsonl (texts).
  a       last prompt token residual for Agastya's 403 prompts at blocks 8..56 step 8 -> a_resid.npy [403,7,5120] f16
  b       410 fork replies teacher-forced: per-token projections on the main direction bank at blocks 16/32/48
          (b_proj.npz), full residuals at his cut points (b_cut_resid.npy), down_proj inputs at the fork position
          (per-behavior means, b_fork_act.npz)
  static  down_proj columns: dots with axis/style/traits per layer (static_dots.npy), column norms; cosine of the
          activation-weighted set sum with the axis vs 1000 layer-matched random sets (static_sum.json)
  kl      KL(base || setting) on benign default replies for every c1 setting (kl.json)
  c1      per fork (priority order, time-budgeted): candidate probabilities under every setting (c1.jsonl)
Settings: steering at block 32 output (dose x median resid norm[32], unit vectors), capping / projection-out /
mean-ablation at layers 46-53 (axis lerped between saved blocks), his top100 / rand100 zero-ablation, top100 +
axis-out / random-out; 20 random unit directions per setting. Rows sharing a context run as one batch (identical
sequences, no padding); runs start from a cached hidden state at the first edited layer.
"""
import argparse, gzip, json, math, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from harness import Harness  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/data/agastya")
ap.add_argument("--vec", default="/data/runs/persona_p1/vectors")
ap.add_argument("--a1", default="/data/runs/persona_p1/vectors/a1_assistant_axis.npy")
ap.add_argument("--norms", default=os.path.expanduser("~/evil-neuron-interp/captures/qwen3.6-27b_resid_norms.json"))
ap.add_argument("--out", default="/data/runs/persona_p1")
ap.add_argument("--phases", default="benign,a,b,static,kl,c1")
ap.add_argument("--budget_min", type=float, default=25.0, help="wall-clock minutes for the whole job; c1 stops at this")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--tok_cap", type=int, default=49152, help="max rows*T per batch")
args = ap.parse_args()
T0 = time.time()
os.makedirs(args.out, exist_ok=True)
PH = set(args.phases.split(","))
torch.manual_seed(0)
dev = "cuda"


def log(*a):
    print(f"[{(time.time()-T0)/60:5.1f}m]", *a, flush=True)


H = Harness(resid_norms_path=args.norms)
assert H.resid_norms is not None
lm, layers, tok = H.lm, H.layers, H.tok
lm_head = H.model.get_output_embeddings()
LT = lm.config.layer_types
MED = H.resid_norms
D = 5120
log("model loaded")

# ------------------------------------------------------------------ vectors
AX_BLOCKS = [8, 16, 24, 32, 40, 48, 56]
axis_raw = np.load(f"{args.vec}/axis_raw.npy").astype(np.float64)
traits_npz = np.load(f"{args.vec}/traits_unit.npz")
TRAITS = ["apathetic", "dismissive", "evil", "hallucinating", "helpful", "humorous", "impolite", "optimistic", "sycophantic"]
a1 = np.load(args.a1).astype(np.float64)


def unit(x):
    return x / max(np.linalg.norm(x), 1e-12)


def axis_lerp(l):
    """Raw axis at layer l, linear interpolation between saved blocks (clamped at 8 and 56), unit-normalized."""
    if l <= 8:
        return unit(axis_raw[0])
    if l >= 56:
        return unit(axis_raw[6])
    i = (l - 8) // 8
    w = (l - AX_BLOCKS[i]) / 8.0
    return unit((1 - w) * axis_raw[i] + w * axis_raw[i + 1])


def T32(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)


CAP_LAYERS = list(range(46, 54))
gen_r = torch.Generator().manual_seed(1234)
R32 = torch.nn.functional.normalize(torch.randn(20, D, generator=gen_r), dim=-1).to(dev)  # steering nulls
R46 = torch.nn.functional.normalize(torch.randn(20, D, generator=gen_r), dim=-1).to(dev)  # cap/out/mab nulls
AX32 = T32(unit(axis_raw[3]))
AXL = {l: T32(axis_lerp(l)) for l in range(64)}
np.savez(f"{args.out}/random_dirs.npz", R32=R32.cpu().numpy(), R46=R46.cpu().numpy())


# ------------------------------------------------------------------ runner
def embed(ids):
    return lm.embed_tokens(torch.tensor([ids], device=dev))


def run_from(h, L0, L1=64):
    """Run decoder layers L0..L1-1 on hidden h [B,T,D] (h = output of layer L0-1), mirroring Qwen3_5TextModel.forward
    with no padding / no cache. Hooks on the layers fire as usual."""
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

    def pre(self, mod, fn):
        self.hs.append(mod.register_forward_pre_hook(fn))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for h in self.hs:
            h.remove()


def parse_spec(spec):
    return {int(L): list(map(int, ns)) for L, ns in spec.items()}


# ------------------------------------------------------------------ data
gen = {}
for line in gzip.open(f"{args.data}/stage_a_gen.jsonl.gz", "rt"):
    r = json.loads(line)
    gen[r["prompt_id"]] = r
PIDS = sorted(gen)
forks = [json.loads(l) for l in open(f"{args.data}/forks_off.jsonl")]
p2 = {json.loads(l)["rid"]: json.loads(l) for l in open(f"{args.data}/phase2_forks.jsonl")}
abl = {(s["b"], s["kind"], s["k"]): parse_spec(s["spec"]) for s in json.load(open(f"{args.data}/ablation_sets.json")).values()}
ups = json.load(open(f"{args.data}/upstream_sets.json"))
shared = json.load(open(f"{args.data}/shared_sets.json"))
BEHS = sorted({f["behavior_id"] for f in forks})
OWN = sorted({b for (b, kind, k) in abl if kind == "top" and k == 100})


def reply_ids(f):
    pid, i = f["rid"].split("|")
    return gen[pid]["prompt_ids"], gen[pid]["samples"][int(i)]["ids"][:f["L"]]


# ================================================================== phase benign
QUESTIONS = [
    "What are some good tips for keeping houseplants alive in a dim apartment?",
    "Can you explain how a bill becomes a law in the United States?",
    "I'm planning a 3-day trip to Lisbon. What should I see?",
    "Write a short, friendly email asking my landlord to fix a leaking faucet.",
    "What's the difference between a Roth IRA and a traditional IRA?",
    "Give me a simple weeknight pasta recipe for two people.",
    "How do vaccines train the immune system?",
    "What is the best way to learn a new programming language?",
    "Explain why the sky is blue.",
    "How can I improve my sleep schedule?",
    "What should I consider when buying a used car?",
    "Summarize the causes of World War I in a few sentences.",
    "How do I make a good first impression at a job interview?",
    "What's a good beginner workout routine?",
    "How does compound interest work?",
    "Can you suggest some board games for a family night?",
    "How do I remove a coffee stain from a white shirt?",
    "What are the main differences between cats and dogs as pets?",
    "Explain what a black hole is to a ten-year-old.",
    "How can I be more productive when working from home?",
    "What are some tips for writing a good cover letter?",
    "How do solar panels generate electricity?",
    "What's the difference between weather and climate?",
    "How do I start composting at home?",
]
STYLE_SYS = {
    "florid": "You are a helpful AI assistant. Answer the user's question correctly, but write in an extremely florid, "
              "ornate, theatrical style: lush metaphors, grand flourishes, dramatic exclamations, archaic and poetic vocabulary.",
    "plain": "You are a helpful AI assistant. Answer the user's question correctly in a plain, simple, matter-of-fact "
             "style: short sentences, everyday words, no flourishes.",
}
if args.smoke:
    QUESTIONS = QUESTIONS[:6]
NQ = len(QUESTIONS)
bpath = f"{args.out}/benign.jsonl"
tau = None  # {layer: tensor[21]} : col 0 axis, 1..20 R46
mu = None   # same layout, mean projection (for mean-ablation)
if "benign" in PH:
    msgs = [[{"role": "user", "content": q}] for q in QUESTIONS]
    msgs += [[{"role": "system", "content": STYLE_SYS[s]}, {"role": "user", "content": q}] for s in ("florid", "plain") for q in QUESTIONS]
    outs = H.chat(msgs, n=1, max_new_tokens=160, seed=7, batch_size=len(msgs), verbose=False)
    styles = ["default"] * NQ + ["florid"] * NQ + ["plain"] * NQ
    recs = []
    for k, (m, o, s) in enumerate(zip(msgs, outs, styles)):
        q = m[-1]["content"]
        p_ids = H.prompt_ids([{"role": "user", "content": q}])
        r_ids = tok(o[0], add_special_tokens=False)["input_ids"][:160]
        recs.append({"style": s, "q": k % NQ, "text": o[0], "prompt_ids": p_ids, "ids": r_ids})
    with open(bpath, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    log("benign generated", len(recs))
    # teacher-force each (no system prompt); per-block mean over response tokens; default tokens at layers 46-53
    means = np.zeros((len(recs), 64, D), np.float32)
    capd = torch.cat([torch.stack([AXL[l] for l in CAP_LAYERS])[:, None], R46[None].expand(len(CAP_LAYERS), -1, -1)], 1)  # [8,21,D]
    projs = {l: [] for l in CAP_LAYERS}
    for k, r in enumerate(recs):
        ids = r["prompt_ids"] + r["ids"]
        P, R = len(r["prompt_ids"]), len(r["ids"])
        store = {}

        def mk(i):
            def hook(mod, a, out):
                hs = out[0, P:P + R].float()
                store[i] = hs.mean(0)
                if r["style"] == "default" and i in CAP_LAYERS:
                    projs[i].append(hs @ capd[CAP_LAYERS.index(i)].T)
            return hook
        with torch.no_grad(), Hooks() as hk:
            for i in range(64):
                hk.fwd(layers[i], mk(i))
            run_from(embed(ids), 0)
        means[k] = torch.stack([store[i] for i in range(64)]).cpu().numpy()
    st = np.array([r["style"] for r in recs])
    qq = np.array([r["q"] for r in recs])
    order = lambda s: means[st == s][np.argsort(qq[st == s])]
    style_raw = (order("florid") - order("plain")).mean(0)
    default_mean = order("default").mean(0)
    np.save(f"{args.out}/style_raw.npy", style_raw)
    np.save(f"{args.out}/default_mean.npy", default_mean)
    np.save(f"{args.out}/benign_means.npy", means)
    tau, mu = {}, {}
    for l in CAP_LAYERS:
        pr = torch.cat(projs[l])  # [ntok, 21]
        tau[l] = torch.quantile(pr, 0.25, dim=0)
        mu[l] = pr.mean(0)
    torch.save({"tau": {l: v.cpu() for l, v in tau.items()}, "mu": {l: v.cpu() for l, v in mu.items()}}, f"{args.out}/cap_tau.pt")
    # checks: our default mean vs community default vector; florid/plain position on the axis
    comm_def = np.load(f"{args.vec}/default_raw.npy")
    chk = {"cos_ourdefault_vs_comm_default": {}, "style_pos_in_gap_units": {}, "cos_style_axis": {}}
    for i, b in enumerate(AX_BLOCKS):
        c = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
        chk["cos_ourdefault_vs_comm_default"][b] = {"same": round(c(default_mean[b], comm_def[i]), 4),
                                                    "b-1": round(c(default_mean[b - 1], comm_def[i]), 4),
                                                    "b+1": round(c(default_mean[b + 1], comm_def[i]), 4),
                                                    "norm_ratio": round(float(np.linalg.norm(default_mean[b]) / np.linalg.norm(comm_def[i])), 3)}
        u = unit(axis_raw[i])
        gap = np.linalg.norm(axis_raw[i])
        d0 = default_mean[b] @ u
        chk["style_pos_in_gap_units"][b] = {s: round(float((d0 - order(s).mean(0)[b] @ u) / gap), 3) for s in ("florid", "plain")}
        chk["style_pos_in_gap_units"][b]["comm_default"] = round(float((d0 - comm_def[i] @ u) / gap), 3)
        chk["cos_style_axis"][b] = round(c(style_raw[b], axis_raw[i]), 3)
    chk["tau_axis"] = {l: round(float(tau[l][0]), 2) for l in CAP_LAYERS}
    chk["mu_axis"] = {l: round(float(mu[l][0]), 2) for l in CAP_LAYERS}
    chk["tau_rand_median"] = {l: round(float(tau[l][1:].median()), 2) for l in CAP_LAYERS}
    json.dump(chk, open(f"{args.out}/benign_checks.json", "w"), indent=1)
    log("benign checks", json.dumps(chk["cos_ourdefault_vs_comm_default"][32]), json.dumps(chk["style_pos_in_gap_units"][32]),
        "cos(style,axis)", chk["cos_style_axis"])
else:
    style_raw = np.load(f"{args.out}/style_raw.npy")
    ct = torch.load(f"{args.out}/cap_tau.pt")
    tau = {l: v.to(dev) for l, v in ct["tau"].items()}
    mu = {l: v.to(dev) for l, v in ct["mu"].items()}
tau = {l: v.to(dev) for l, v in tau.items()}
mu = {l: v.to(dev) for l, v in mu.items()}
benign = [json.loads(l) for l in open(bpath)]

# main direction bank for blocks 16/32/48
B_BLOCKS = [16, 32, 48]
BANK_NAMES = ["axis", "style", "axis_perp_style", "a1axis"] + [f"{t}_response_avg" for t in TRAITS] + [f"{t}_prompt_last" for t in TRAITS]


def bank(b):
    ax = unit(axis_raw[AX_BLOCKS.index(b)])
    s = unit(style_raw[b].astype(np.float64))
    aps = unit(ax - (ax @ s) * s)
    rows = [ax, s, aps, unit(a1[b])] + [traits_npz[f"{t}_response_avg"][b] for t in TRAITS] + [traits_npz[f"{t}_prompt_last"][b] for t in TRAITS]
    return T32(np.stack(rows))


BANK = {b: bank(b) for b in B_BLOCKS}
np.savez(f"{args.out}/bank.npz", names=np.array(BANK_NAMES), **{f"b{b}": BANK[b].cpu().numpy() for b in B_BLOCKS})


# ================================================================== phase a
def batched_capture(seqs, blocks, pos_fn, batch_tok=8192, extra_hooks=None):
    """seqs: list of id lists. Right-padded batches (causal => real positions unaffected).
    pos_fn(k) -> list of positions to gather for sequence k. Returns {block: list of [npos, D] f32 cpu}."""
    out = {b: [None] * len(seqs) for b in blocks}
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k]))
    s = 0
    while s < len(order):
        Tm = len(seqs[order[s]])
        e = s
        while e < len(order) and (e - s + 1) * max(Tm, len(seqs[order[e]])) <= batch_tok:
            Tm = max(Tm, len(seqs[order[e]]))
            e += 1
        e = max(e, s + 1)
        sel = order[s:e]
        Tb = max(len(seqs[k]) for k in sel)
        ids = torch.full((len(sel), Tb), tok.pad_token_id, dtype=torch.long)
        for r, k in enumerate(sel):
            ids[r, :len(seqs[k])] = torch.tensor(seqs[k])
        gidx = [torch.tensor(pos_fn(k), device=dev) for k in sel]

        def mk(b):
            def hook(mod, a, o):
                for r, k in enumerate(sel):
                    out[b][k] = o[r, gidx[r]].float().cpu()
            return hook
        with torch.no_grad(), Hooks() as hk:
            for b in blocks:
                hk.fwd(layers[b], mk(b))
            if extra_hooks:
                extra_hooks(hk, sel)
            run_from(lm.embed_tokens(ids.to(dev)), 0, max(blocks) + 1 if not extra_hooks else 64)
        s = e
    return out


if "a" in PH:
    pids = PIDS[:20] if args.smoke else PIDS
    seqs = [gen[p]["prompt_ids"] for p in pids]
    cap = batched_capture(seqs, AX_BLOCKS, lambda k: [len(seqs[k]) - 1])
    a_resid = np.stack([np.stack([cap[b][k][0].numpy() for b in AX_BLOCKS]) for k in range(len(pids))]).astype(np.float16)
    np.save(f"{args.out}/a_resid.npy", a_resid)
    json.dump({"pids": pids, "blocks": AX_BLOCKS}, open(f"{args.out}/a_meta.json", "w"))
    # batched vs single check
    eq = []
    for k in range(4):
        c1 = batched_capture([seqs[k]], AX_BLOCKS, lambda _: [len(seqs[k]) - 1])
        for b in (16, 32, 48):
            x, y = c1[b][0][0], torch.tensor(a_resid[k, AX_BLOCKS.index(b)].astype(np.float32))
            eq.append(float((x - y).norm() / x.norm()))
    log("phase a done", a_resid.shape, "batched-vs-single rel diff max", max(eq))
    json.dump({"a_batched_vs_single_rel_diff": eq}, open(f"{args.out}/a_checks.json", "w"))

# ================================================================== phase b
if "b" in PH:
    fl = forks[:12] if args.smoke else forks
    seqs, cuts, P_of = [], [], []
    for f in fl:
        p_ids, r_ids = reply_ids(f)
        seqs.append(p_ids + r_ids)
        P_of.append(len(p_ids))
        cuts.append([c for c, _ in f["curve"]])
    proj_tok = [None] * len(fl)
    cut_res = {b: [None] * len(fl) for b in B_BLOCKS}
    fork_act = np.zeros((len(fl), 64, 17408), np.float16)
    order = sorted(range(len(fl)), key=lambda k: len(seqs[k]))
    s = 0
    t_b = time.time()
    while s < len(order):
        e = s
        while e < len(order) and (e - s + 1) * len(seqs[order[e]]) <= 12288:
            e += 1
        e = max(e, s + 1)
        sel = order[s:e]
        Tb = max(len(seqs[k]) for k in sel)
        ids = torch.full((len(sel), Tb), tok.pad_token_id, dtype=torch.long)
        for r, k in enumerate(sel):
            ids[r, :len(seqs[k])] = torch.tensor(seqs[k])
        ptmp = {b: {} for b in B_BLOCKS}

        def mkb(b):
            def hook(mod, a, o):
                for r, k in enumerate(sel):
                    P, L = P_of[k], len(seqs[k]) - P_of[k]
                    hs = o[r, P - 1:P + L].float()  # positions P-1 .. P+L-1
                    ptmp[b][k] = (hs @ BANK[b].T).half().cpu()
                    cp = torch.tensor([P + c - 1 for c in cuts[k]], device=dev)
                    cut_res[b][k] = o[r, cp].half().cpu()
            return hook

        def mka(i):
            def pre(mod, a):
                x = a[0]
                for r, k in enumerate(sel):
                    fork_act[k, i] = x[r, P_of[k] + fl[k]["fork"][0] - 1].float().cpu().numpy()
            return pre
        with torch.no_grad(), Hooks() as hk:
            for b in B_BLOCKS:
                hk.fwd(layers[b], mkb(b))
            for i in range(64):
                hk.pre(layers[i].mlp.down_proj, mka(i))
            run_from(lm.embed_tokens(ids.to(dev)), 0)
        for k in sel:
            proj_tok[k] = torch.stack([ptmp[b][k] for b in B_BLOCKS], 1).numpy()  # [L+1, 3, nbank]
        s = e
    log(f"phase b forward done in {(time.time()-t_b)/60:.1f} min")
    np.savez_compressed(f"{args.out}/b_proj.npz", **{fl[k]["rid"]: proj_tok[k] for k in range(len(fl))})
    np.save(f"{args.out}/b_cut_resid.npy", np.stack([np.concatenate([cut_res[b][k].numpy() for k in range(len(fl))]) for b in B_BLOCKS], 1))
    json.dump({"rids": [f["rid"] for f in fl], "n_cuts": [len(c) for c in cuts], "blocks": B_BLOCKS, "bank": BANK_NAMES},
              open(f"{args.out}/b_meta.json", "w"))
    beh = np.array([f["behavior_id"] for f in fl])
    np.savez(f"{args.out}/b_fork_act.npz", all=fork_act.astype(np.float32).mean(0),
             **{b: fork_act[beh == b].astype(np.float32).mean(0) for b in BEHS if (beh == b).any()})
    # batched vs single check on 3 replies
    eq = []
    for k in order[:1] + order[len(order) // 2:len(order) // 2 + 1] + order[-1:]:
        P = P_of[k]
        cp = [P + c - 1 for c in cuts[k]]
        c1 = batched_capture([seqs[k]], B_BLOCKS, lambda _: cp)
        for bi, b in enumerate(B_BLOCKS):
            x, y = c1[b][0], cut_res[b][k].float()
            eq.append(float(((x - y).norm(dim=-1) / x.norm(dim=-1)).max()))
    json.dump({"b_batched_vs_single_rel_diff_max": eq}, open(f"{args.out}/b_checks.json", "w"))
    log("phase b done; batched-vs-single max rel diff", max(eq))
    del fork_act

# ================================================================== phase static
if "static" in PH:
    acts = np.load(f"{args.out}/b_fork_act.npz")
    dots = np.zeros((64, 11, 17408), np.float32)
    wnorm = np.zeros((64, 17408), np.float32)
    Wf = {}
    with torch.no_grad():
        for l in range(64):
            W = layers[l].mlp.down_proj.weight.float()  # [D, 17408]
            dl = [AXL[l], T32(unit(style_raw[l].astype(np.float64)))] + [T32(traits_npz[f"{t}_response_avg"][min(l, 62)]) for t in TRAITS]
            dots[l] = (torch.stack(dl) @ W).cpu().numpy()
            wnorm[l] = W.norm(dim=0).cpu().numpy()
    np.save(f"{args.out}/static_dots.npy", dots)
    np.save(f"{args.out}/static_wnorm.npy", wnorm)
    # cosine of the activation-weighted sum with the axis at blocks 48 / 56 (layers <= block), vs 1000 layer-matched random sets
    sets = {}
    for (b, kind, k), spec in abl.items():
        if k == 100:
            sets[f"{b}|{kind}{k}"] = (spec, b)
    for key, v in ups.items():
        if "spec" in v:
            sets[key] = (parse_spec(v["spec"]), v["b"])
    for key, v in shared.items():
        sets[f"shared|{key}"] = (parse_spec(v["spec"]), v.get("b", "all"))
    if args.smoke:
        sets = {k: sets[k] for k in list(sets)[:3]}
    gen_s = torch.Generator(device=dev).manual_seed(5)
    res = {}
    with torch.no_grad():
        for key, (spec, b) in sets.items():
            a = torch.tensor(acts[b] if b in acts.files else acts["all"], device=dev)  # [64,17408]
            out = {}
            for blk in (48, 56):
                v = T32(unit(axis_raw[AX_BLOCKS.index(blk)]))
                s_set = torch.zeros(D, device=dev)
                s_rand = torch.zeros(1000, D, device=dev)
                for l, ns in spec.items():
                    if l > blk:
                        continue
                    W = layers[l].mlp.down_proj.weight
                    idx = torch.tensor(ns, device=dev)
                    s_set += (W[:, idx].float() * a[l, idx]).sum(1)
                    ridx = torch.randint(0, 17408, (1000, len(ns)), device=dev, generator=gen_s)
                    Wr = W[:, ridx.flatten()].float().view(D, 1000, len(ns))  # [D,1000,k]
                    s_rand += torch.einsum("drk,rk->rd", Wr, a[l][ridx])
                    del Wr
                if s_set.norm() == 0:
                    continue
                cs = float(s_set @ v / s_set.norm())
                cr = (s_rand @ v / s_rand.norm(dim=-1).clamp_min(1e-9)).cpu().numpy()
                ps = float(s_set @ v)
                pr = (s_rand @ v).cpu().numpy()
                out[blk] = {"cos": cs, "cos_rand_pct": float((cr < cs).mean() * 100), "cos_rand_mean": float(cr.mean()),
                            "cos_rand_sd": float(cr.std()), "proj_over_gap": ps / float(np.linalg.norm(axis_raw[AX_BLOCKS.index(blk)])),
                            "proj_rand_pct": float((pr < ps).mean() * 100), "norm_over_resid": float(s_set.norm()) / MED[blk]}
            res[key] = out
    json.dump(res, open(f"{args.out}/static_sum.json", "w"), indent=1)
    log("phase static done", json.dumps(res.get("denying-ai-identity|med100", {})))


# ================================================================== settings (c1 / kl)
def edit_hooks(hk, U, mode, tvals, mvals):
    """Per-row residual edits at CAP_LAYERS. U [B,D] f32 (per-row direction at that layer, dict l->[B,D]),
    mode [B] long: 0 none, 1 cap (h -= U*min(<h,U>-tau,0)), 2 project out (h -= U<h,U>), 3 mean-ablate (h -= U(<h,U>-mu))."""
    for l in CAP_LAYERS:
        def hook(mod, a, o, l=l):
            hf = o.float()
            Ul = U[l]
            pr = torch.einsum("btd,bd->bt", hf, Ul)
            m = mode[:, None]
            coef = torch.where(m == 1, (pr - tvals[l][:, None]).clamp(max=0),
                               torch.where(m == 2, pr, torch.where(m == 3, pr - mvals[l][:, None], torch.zeros_like(pr))))
            return (hf - coef[..., None] * Ul[:, None, :]).to(o.dtype)
        hk.fwd(layers[l], hook)


def abl_hooks(hk, masks):
    """masks: {layer: [B,17408] bf16 (0 = ablated)}; zero the down_proj input of those neurons per row."""
    for l, m in masks.items():
        def pre(mod, a, m=m):
            return (a[0] * m[:, None, :],) + tuple(a[1:])
        hk.pre(layers[l].mlp.down_proj, pre)


def make_masks(row_sets, B):
    masks = {}
    for r, spec in enumerate(row_sets):
        if not spec:
            continue
        for l, ns in spec.items():
            if l not in masks:
                masks[l] = torch.ones(B, 17408, dtype=torch.bfloat16, device=dev)
            masks[l][r, ns] = 0
    return masks


def run_rows(h_src, L0, rows, mode_out, cands=None, span=None, base_lp=None, cap_blocks=(), stash45=None):
    """h_src [1,T,D] = output of layer L0-1. rows: list of dicts with optional keys
    delta (tensor[D] added to h_src), edit ('cap'|'out'|'mab', which: 'axis'|int j), abl (spec dict).
    mode_out 'last': returns per-row candidate probs (cands list) and last-position axis projections at cap_blocks.
    mode_out 'span': returns per-row log-probs over span (P-1..P+R-2) as list of cpu tensors? -> returns KL vs ref (caller).
    Chunked to tok_cap rows*T."""
    T = h_src.shape[1]
    per = max(1, args.tok_cap // T)
    res = []
    for s in range(0, len(rows), per):
        chunk = rows[s:s + per]
        B = len(chunk)
        h = h_src.expand(B, -1, -1).float()
        dl = torch.stack([r["delta"] if r.get("delta") is not None else torch.zeros(D, device=dev) for r in chunk])
        h = (h + dl[:, None, :]).to(torch.bfloat16)
        mode = torch.tensor([{"cap": 1, "out": 2, "mab": 3}.get(r.get("edit", (None,))[0], 0) for r in chunk], device=dev)
        U, tv, mv = {}, {}, {}
        any_edit = bool((mode > 0).any())
        if any_edit:
            for l in CAP_LAYERS:
                Ul, tl, ml = [], [], []
                for r in chunk:
                    e = r.get("edit")
                    if e is None:
                        Ul.append(torch.zeros(D, device=dev)); tl.append(0.0); ml.append(0.0)
                    elif e[1] == "axis":
                        Ul.append(AXL[l]); tl.append(float(tau[l][0])); ml.append(float(mu[l][0]))
                    elif isinstance(e[1], tuple):  # ("rp", j): role-pair difference direction
                        j = e[1][1]
                        Ul.append(RP[l][j]); tl.append(float(tau_rp[l][j])); ml.append(float(mu_rp[l][j]))
                    else:
                        j = e[1]
                        Ul.append(R46[j]); tl.append(float(tau[l][1 + j])); ml.append(float(mu[l][1 + j]))
                U[l] = torch.stack(Ul)
                tv[l] = torch.tensor(tl, device=dev)
                mv[l] = torch.tensor(ml, device=dev)
        masks = make_masks([r.get("abl") for r in chunk], B)
        capt = {}
        with torch.no_grad(), Hooks() as hk:
            if any_edit:
                edit_hooks(hk, U, mode, tv, mv)
            if masks:
                abl_hooks(hk, masks)
            for blk in cap_blocks:
                if blk >= L0:
                    def ch(mod, a, o, blk=blk):
                        capt[blk] = (o[:, -1].float() @ AXL[blk]).cpu()
                    hk.fwd(layers[blk], ch)
            if stash45 is not None and 45 >= L0:
                def st(mod, a, o):
                    stash45["h"] = o[stash45["row"] - s:stash45["row"] - s + 1].clone() if s <= stash45["row"] < s + B else stash45.get("h")
                hk.fwd(layers[45], st)
            hout = run_from(h, L0)
            if mode_out == "last":
                logits = lm_head(lm.norm(hout[:, -1])).float()
                pr = torch.softmax(logits, -1)
                pc = pr[:, cands].cpu()
                for r in range(B):
                    res.append({"p": pc[r].tolist(), "proj": {blk: float(capt[blk][r]) for blk in capt}})
            else:
                a0, a1_ = span
                for r in range(B):
                    lp = torch.log_softmax(lm_head(lm.norm(hout[r, a0:a1_])).float(), -1)
                    res.append(float((base_lp.exp() * (base_lp - lp)).sum(-1).mean()))
                    del lp
        del hout, h
    return res


def base_pass(ids, stash_layers, cap_blocks=()):
    """Full forward B=1; returns (stash {layer: output [1,T,D] bf16}, last logits f32, last-pos axis projections)."""
    stash, capt = {}, {}
    with torch.no_grad(), Hooks() as hk:
        for l in stash_layers:
            def st(mod, a, o, l=l):
                stash[l] = o.clone()
            hk.fwd(layers[l], st)
        for blk in cap_blocks:
            def ch(mod, a, o, blk=blk):
                capt[blk] = float(o[0, -1].float() @ AXL[blk])
            hk.fwd(layers[blk], ch)
        h = run_from(embed(ids), 0)
        logits = lm_head(lm.norm(h[:, -1])).float()[0]
    return stash, logits, capt


DOSES = [0.1, 0.25, 0.5]


def steer_rows():
    m = MED[32]
    rows = [{"name": "base"}]
    for d in DOSES:
        rows.append({"name": f"axis+{d}", "delta": d * m * AX32})
    for d in (0.25, 0.5):
        rows.append({"name": f"axis-{d}", "delta": -d * m * AX32})
    st = T32(unit(style_raw[32].astype(np.float64)))
    for d in (0.25, 0.5):
        rows.append({"name": f"plain{d}", "delta": -d * m * st})  # toward plain = -style
    ax = unit(axis_raw[3])
    s = unit(style_raw[32].astype(np.float64))
    aps = T32(unit(ax - (ax @ s) * s))
    rows.append({"name": "axisperp+0.5", "delta": 0.5 * m * aps})
    for d in DOSES:
        for j in range(20):
            rows.append({"name": f"r{j}+{d}", "delta": d * m * R32[j]})
    return rows


def edit_rows():
    rows = [{"name": "base"}]
    for e in ("cap", "out", "mab"):
        rows.append({"name": f"{e}_axis", "edit": (e, "axis")})
        for j in range(20):
            rows.append({"name": f"{e}_r{j}", "edit": (e, j)})
    return rows


def top_rows(b):
    top, rnd = abl[(b, "top", 100)], abl[(b, "rand", 100)]
    rows3 = [{"name": "base"}, {"name": "top100", "abl": top}, {"name": "rand100", "abl": rnd}]
    rows4 = [{"name": "top100", "abl": top}, {"name": "top100+out_axis", "abl": top, "edit": ("out", "axis")}]
    rows4 += [{"name": f"top100+out_r{j}", "abl": top, "edit": ("out", j)} for j in range(20)]
    lmin = min(min(top), min(rnd))
    return rows3, rows4, lmin


def strip(rows):
    return [{k: v for k, v in r.items()} for r in rows]


# ================================================================== phase kl
if "kl" in PH:
    kseqs = [r for r in benign if r["style"] == "default" and len(r["ids"]) >= 60][: (2 if args.smoke else 8)]
    kl = {}
    t_k = time.time()
    for r in kseqs:
        ids = r["prompt_ids"] + r["ids"]
        P, R = len(r["prompt_ids"]), len(r["ids"])
        span = (P - 1, P + R - 1)
        stash_l = sorted({32, 45} | {top_rows(b)[2] - 1 for b in OWN})
        stash, _, _ = base_pass(ids, stash_l)
        with torch.no_grad():
            h = run_from(embed(ids), 0)
            base_lp = torch.log_softmax(lm_head(lm.norm(h[0, span[0]:span[1]])).float(), -1)
            del h

        def addkl(names, vals, prefix=""):
            for n, v in zip(names, vals):
                kl.setdefault(prefix + n, []).append(v)
        kw = dict(span=span, base_lp=base_lp)
        rows = steer_rows()
        addkl([x["name"] for x in rows], run_rows(stash[32], 33, rows, "span", **kw))
        rows = edit_rows()
        addkl([x["name"] for x in rows], run_rows(stash[45], 46, rows, "span", **kw))
        for b in OWN:
            r3, r4, lmin = top_rows(b)
            st45 = {"row": 1}
            addkl([x["name"] for x in r3], run_rows(stash[lmin - 1], lmin, r3, "span", stash45=st45, **kw), prefix=f"{b}|")
            addkl([x["name"] for x in r4], run_rows(st45["h"], 46, r4, "span", **kw), prefix=f"{b}|")
        del stash
        torch.cuda.empty_cache()
    json.dump({k: float(np.mean(v)) for k, v in kl.items()}, open(f"{args.out}/kl.json", "w"), indent=1)
    log(f"phase kl done in {(time.time()-t_k)/60:.1f} min; base {np.mean(kl['base']):.2e} axis+0.5 {np.mean(kl['axis+0.5']):.4f} "
        f"r+0.5 med {np.median([np.mean(kl[f'r{j}+0.5']) for j in range(20)]):.4f} cap {np.mean(kl['cap_axis']):.4f} out {np.mean(kl['out_axis']):.4f}")

# ================================================================== phase c1
if "c1" in PH:
    def fork_cands(f):
        nt = p2[f["rid"]]["next_tokens"] if f["rid"] in p2 else f["next_tokens"]
        return nt
    info = [f for f in forks if len(fork_cands(f)) >= 2 and max(c["q"] for c in fork_cands(f)) - min(c["q"] for c in fork_cands(f)) >= 0.25]
    rng = random.Random(0)
    per_b = {}
    for b in BEHS:
        fb = [f for f in info if f["behavior_id"] == b]
        sharp = [f for f in fb if f["rid"] in p2]
        rest = [f for f in fb if f["rid"] not in p2]
        rng.shuffle(sharp)
        rng.shuffle(rest)
        per_b[b] = sharp + rest
    queue = []
    while any(per_b.values()):
        for b in BEHS:
            if per_b[b]:
                queue.append(per_b[b].pop(0))
    if args.smoke:
        queue = [f for f in queue if f["behavior_id"] in OWN][:2]
    outp = f"{args.out}/c1.jsonl"
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)["rid"] for l in open(outp)}
    log(f"c1: {len(info)} informative forks, {len(done)} done")
    deadline = T0 + args.budget_min * 60
    t_c = time.time()
    n = 0
    with open(outp, "a") as fh:
        for f in queue:
            if f["rid"] in done:
                continue
            if time.time() > deadline:
                log("c1 budget reached")
                break
            tf = time.time()
            p_ids, r_ids = reply_ids(f)
            lo = f["fork"][0]
            ids = p_ids + r_ids[:lo]
            assert r_ids[lo] == f["fork_token"]
            cands = [c["tok"] for c in fork_cands(f)]
            b = f["behavior_id"]
            stash_l = {32, 45}
            if b in OWN:
                r3, r4, lmin = top_rows(b)
                stash_l.add(lmin - 1)
            stash, logits, capb = base_pass(ids, sorted(stash_l), cap_blocks=(32, 40, 48, 56))
            pfull = torch.softmax(logits, -1)[cands].cpu().tolist()
            rec = {"rid": f["rid"], "b": b, "T": len(ids), "cands": cands, "q": [c["q"] for c in fork_cands(f)],
                   "p_his": [c["p"] for c in fork_cands(f)], "phase2": f["rid"] in p2, "p_base_full": pfull, "proj_base": capb,
                   "rows": {}}
            rows = steer_rows()
            for x, o in zip(rows, run_rows(stash[32], 33, rows, "last", cands=cands, cap_blocks=(48, 56))):
                rec["rows"].setdefault(x["name"], o)
            rows = edit_rows()
            for x, o in zip(rows, run_rows(stash[45], 46, rows, "last", cands=cands, cap_blocks=(48, 56))):
                rec["rows"].setdefault("e:" + x["name"], o)
            if b in OWN:
                st45 = {"row": 1}
                for x, o in zip(r3, run_rows(stash[lmin - 1], lmin, r3, "last", cands=cands, cap_blocks=(32, 40, 48, 56), stash45=st45)):
                    rec["rows"]["t:" + x["name"]] = o
                for x, o in zip(r4, run_rows(st45["h"], 46, r4, "last", cands=cands, cap_blocks=(48, 56))):
                    rec["rows"]["t4:" + x["name"]] = o
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            del stash
            if n <= 3 or n % 10 == 0:
                log(f"c1 fork {n} {b} T={len(ids)} {time.time()-tf:.1f}s")
    log(f"c1 done: {n} forks in {(time.time()-t_c)/60:.1f} min")

# ================================================================== phase c1x (role-pair null for edits)
if "c1x" in PH:
    # 20 random role-pair difference directions (community role vectors), lerped per layer like the axis; their
    # projections vary like real persona contrasts, unlike isotropic random directions (whose capping is near no-op)
    roles = np.load(f"{args.vec}/roles_raw.npy").astype(np.float64)
    rr = np.random.default_rng(7)
    pi = rr.integers(0, len(roles), 20)
    pj = (pi + rr.integers(1, len(roles), 20)) % len(roles)
    diff = roles[pi] - roles[pj]  # [20, 7, D]

    def lerp_raw(x, l):
        if l >= 56:
            return x[6]
        i = (l - 8) // 8
        w = (l - AX_BLOCKS[i]) / 8.0
        return (1 - w) * x[i] + w * x[i + 1]
    RP = {l: T32(np.stack([unit(lerp_raw(diff[k], l)) for k in range(20)])) for l in CAP_LAYERS}
    json.dump({"pairs": [[int(a), int(b)] for a, b in zip(pi, pj)]}, open(f"{args.out}/c1x_pairs.json", "w"))
    projs = {l: [] for l in CAP_LAYERS}
    for r in [x for x in benign if x["style"] == "default"]:
        ids = r["prompt_ids"] + r["ids"]
        P, Rl = len(r["prompt_ids"]), len(r["ids"])

        def mk(i):
            def hook(mod, a, o):
                projs[i].append(o[0, P:P + Rl].float() @ RP[i].T)
            return hook
        with torch.no_grad(), Hooks() as hk:
            for i in CAP_LAYERS:
                hk.fwd(layers[i], mk(i))
            run_from(embed(ids), 0, 54)
    tau_rp = {l: torch.quantile(torch.cat(projs[l]), 0.25, dim=0) for l in CAP_LAYERS}
    mu_rp = {l: torch.cat(projs[l]).mean(0) for l in CAP_LAYERS}

    def x_rows():
        rows = [{"name": "base"}]
        for e in ("cap", "out", "mab"):
            rows.append({"name": f"{e}_axis", "edit": (e, "axis")})
            for j in range(20):
                rows.append({"name": f"{e}_rp{j}", "edit": (e, ("rp", j))})
        return rows
    # KL on benign
    kl = {}
    for r in [x for x in benign if x["style"] == "default" and len(x["ids"]) >= 60][:8]:
        ids = r["prompt_ids"] + r["ids"]
        P, Rl = len(r["prompt_ids"]), len(r["ids"])
        span = (P - 1, P + Rl - 1)
        stash, _, _ = base_pass(ids, [45])
        with torch.no_grad():
            h = run_from(embed(ids), 0)
            base_lp = torch.log_softmax(lm_head(lm.norm(h[0, span[0]:span[1]])).float(), -1)
            del h
        rows = x_rows()
        for x, v in zip(rows, run_rows(stash[45], 46, rows, "span", span=span, base_lp=base_lp)):
            kl.setdefault(x["name"], []).append(v)
    json.dump({k: float(np.mean(v)) for k, v in kl.items()}, open(f"{args.out}/kl_x.json", "w"), indent=1)
    log("c1x kl: cap_axis %.4f cap_rp med %.4f mab_axis %.4f mab_rp med %.4f out_axis %.4f out_rp med %.4f" % (
        np.mean(kl["cap_axis"]), np.median([np.mean(kl[f"cap_rp{j}"]) for j in range(20)]), np.mean(kl["mab_axis"]),
        np.median([np.mean(kl[f"mab_rp{j}"]) for j in range(20)]), np.mean(kl["out_axis"]), np.median([np.mean(kl[f"out_rp{j}"]) for j in range(20)])))
    order_rids = [json.loads(l)["rid"] for l in open(f"{args.out}/c1.jsonl")]
    fk = {f["rid"]: f for f in forks}
    outp = f"{args.out}/c1x.jsonl"
    done = {json.loads(l)["rid"] for l in open(outp)} if os.path.exists(outp) else set()
    deadline = T0 + args.budget_min * 60
    n = 0
    with open(outp, "a") as fh:
        for rid in order_rids:
            if rid in done:
                continue
            if time.time() > deadline:
                log("c1x budget reached")
                break
            f = fk[rid]
            p_ids, r_ids = reply_ids(f)
            ids = p_ids + r_ids[:f["fork"][0]]
            nt = p2[rid]["next_tokens"] if rid in p2 else f["next_tokens"]
            cands = [c["tok"] for c in nt]
            stash, _, _ = base_pass(ids, [45])
            rows = x_rows()
            rec = {"rid": rid, "b": f["behavior_id"], "rows": {}}
            for x, o in zip(rows, run_rows(stash[45], 46, rows, "last", cands=cands, cap_blocks=(48, 56))):
                rec["rows"]["x:" + x["name"]] = o
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            del stash
    log(f"c1x done: {n} forks")

# equivalence check of the runner (cheap, once)
if "eqcheck" in PH or args.smoke:
    f = [x for x in forks if x["behavior_id"] in OWN][0]
    p_ids, r_ids = reply_ids(f)
    ids = p_ids + r_ids[:f["fork"][0]]
    with torch.no_grad():
        ref = H.model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0, -1].float()
        stash, mine, _ = base_pass(ids, [32, 45])
        o33 = run_from(stash[32], 33)
        l33 = lm_head(lm.norm(o33[:, -1])).float()[0]
        rows = [{"name": "base"}] * 40
        r40 = run_rows(stash[32], 33, rows, "last", cands=list(range(10)))
        p_ref = torch.softmax(ref, -1)[:10].cpu()
    chk = {"full_vs_runner_maxabs_logit": float((ref - mine).abs().max()), "skip33_vs_full_maxabs_logit": float((ref - l33).abs().max()),
           "B40_vs_B1_maxabs_p": float(max(abs(a - b) for a, b in zip(r40[5]["p"], p_ref.tolist())))}
    json.dump(chk, open(f"{args.out}/runner_checks.json", "w"), indent=1)
    log("runner checks", chk)
log("ALL_DONE", f"{(time.time()-T0)/60:.1f} min")
