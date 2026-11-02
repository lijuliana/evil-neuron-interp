"""CHIVE Eval B, neuron arm GPU pass (findings/chive/design.md section 7 + the 2026-11-01 design-review updates).
Forward / backward passes only: no generation, no judge, no API.

Run on the VM after syncing data/chive/, data/agastya/, data/promptsets/, data/tokenizer/ and
runs/persona_p1/{bank.npz,random_dirs.npz,b_fork_act.npz,benign.jsonl}:
  flock /data/gpu.lock ~/venv/bin/python -I scripts/93_chive_neuron_gpu.py --out /data/runs/chive/neuron_gpu
then copy --out to runs/chive/neuron_gpu/ and run `scripts/93_chive_evidence.py neurons`.
CPU smoke test of the whole code path on a tiny random Qwen3.5 model (real tokenizer, remapped neuron ids):
  .venv/bin/python -I scripts/93_chive_neuron_gpu.py --tiny --limit 3 --out <scratch dir>

Neuron sets (per behavior; layer counts of set and ctrl are asserted equal):
  own:  <b>|top100 for 911 / denying-AI / code-exec / embodiment / laser (data/agastya/ablation_sets.json);
        ctrl = <b>|rand100 (matched on layer and activation size).
  loo:  loo100|<b> for drunk driving and chemtrails (shared_sets.json; pre-specified, unlike sharedmid100 which was
        picked after G1 results); ctrl = rand_loo100|<b>.
  none: unsolicited sexual advances (the evidence block says "not available").

Target (adopted from the design review): with a_i(x, t) the down_proj input of neuron i, w_i = W_down[:, i] and
  u_hat = unit(sum_i abar_i w_i),  abar_i = mean of a_i at the fork-token positions of this behavior's fork replies
  (runs/persona_p1/b_fork_act.npz: all 410 forks_off forks, discovery prompts only),
  T(x) = < sum_i a_i(x, t*) w_i , u_hat > = sum_i c_i a_i(x, t*),  c_i = <w_i, u_hat>,
the set's write at t* projected on the direction it writes at its forks. t* = the last prompt token (the "\n\n" after
the empty think block, ids end 248068 271 248069 271) with --positions last (default, pre-registered); --positions
reply averages T over the states that choose the tokens of the prompt's transcript reply (secondary, not shown).
The ctrl set uses the same recipe with its own abar / u_hat.

Phases (--phases; default prep,acts,axisref,ig,occl,gate), all resumable or cheap:
  prep     per set: abar, c, |u|, top-10 = largest contribution abar_i * c_i (one rule for set and ctrl),
           logit-lens tokens of sign(abar_i) * w_i through the final norm + unembedding (top 8)       -> prep.json
  acts     no-grad. (1) every pool transcript (prompt + transcript reply, <= 1,024 tokens, teacher-forced): down_proj
           inputs of all set/ctrl neurons at positions P-1 .. P+R-1 (state that chooses reply token j is P-1+j);
           the prompt's own set + ctrl columns -> acts.npz (+ acts_meta.json); per-neuron mean / sd over all
           reply-choosing states of all transcripts -> stats.json; Assistant-Axis + 20 random-direction projections at
           blocks 16/32/48 along every transcript -> axis_reply.npz. (2) the same projections along every pool fork
           reply (forks_off, truncated at its L) -> axis_fork.npz. (3) T of all 14 sets at the last token of all 403
           stage-A prompts -> T403.npz (prompt-level percentile in the block; validity gate), and their unbatched
           last-token residuals at blocks 16/32/48 -> resid403.npz (P1's a_resid.npy was captured right-padded in
           batches, max relative deviation .059 from single-sequence runs; the axis arm prefers this file).
  axisref  Assistant-Axis benign reference: last-token residuals at blocks 16/32/48 of 254 benign prompts (P1's 24
           questions, induction_benign.jsonl 30, capability_mini.jsonl 200) and per-token projections along P1's 24
           default benign replies (teacher-forced, no system prompt)                              -> axis_benign.npz
  ig       integrated gradients of T w.r.t. inputs_embeds (the leaf) at the user-message tokens only; baseline = the
           pad-token (<|endoftext|>) embedding at those positions, template tokens untouched; midpoint rule with
           --steps (>= 32) interpolation rows; attribution_t = sum over hidden dims of (x - b)_t * mean grad_t. Set and
           ctrl share the forwards (two autograd.grad calls). Completeness: |sum attr - (T(x) - T(b))| <= tol |T(x) -
           T(b)| (tol 5%), else steps doubled up to --max_steps; the last result is kept and flagged. First 2
           prompts: finite-difference check of the gradient at x                                   -> attrib.jsonl, fd.json
           Rows carry terms_hash (sha256 of each prepared set definition); resume, the gate and 93_chive_evidence.py
           use only rows whose hash matches the current prep.json (acts_meta / stats / T403 carry it too).
  occl     check on NON-pool stage-A prompts of the same behaviors (no pool prompt is ever run edited): per sentence
           unit (chive_common.split_units), IG unit sum vs (a) the unit's tokens replaced by pad (positions kept) and
           (b) the unit deleted (chive_common.apply_edit, re-templated); Spearman per prompt         -> occl.jsonl
  gate     pre-registered validity gate, reported and never retuned: per set, sd of T over the behavior's prompts among
           the 403 vs the same for its ctrl; passes if the 95% bootstrap CI (prompts resampled) of sd_set / sd_ctrl is
           above 1. Also completeness, FD and occlusion summaries                                      -> checks.json
93_chive_evidence.py never reads occl.jsonl, fd.json or checks.json (validation only, never shown to the predictor).
A2's attribution (scripts/22_a2_validate.py) was invalid because the down_proj-input hooks made per-layer tensors
require grad without retain_grad(), so only layer 0 received gradients. Here the only leaf is inputs_embeds and
gradients come from torch.autograd.grad, and completeness checks the result numerically.
"""
import argparse, gzip, hashlib, json, math, os, random, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
from chive_common import apply_edit, split_units, template_user_message  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default=f"{REPO}/data/chive/pool.jsonl")
ap.add_argument("--data", default=f"{REPO}/data/agastya")
ap.add_argument("--p1", default=f"{REPO}/runs/persona_p1")
ap.add_argument("--promptsets", default=f"{REPO}/data/promptsets")
ap.add_argument("--out", default=f"{REPO}/runs/chive/neuron_gpu")
ap.add_argument("--phases", default="prep,acts,axisref,ig,occl,gate")
ap.add_argument("--positions", default="last", choices=["last", "reply"])
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--max_steps", type=int, default=128)
ap.add_argument("--tol", type=float, default=0.05)
ap.add_argument("--tok_budget", type=int, default=768, help="max rows x tokens per IG batch (halved on OOM)")
ap.add_argument("--n_occl", type=int, default=6, help="non-pool prompts per behavior for the occlusion check")
ap.add_argument("--max_reply", type=int, default=1024)
ap.add_argument("--ig_shard", default="0/1", help="ig: only pool prompts with index %% n == i (i/n); files merged by hand")
ap.add_argument("--path_k", type=int, default=16, help="pathcheck: intervals along the IG path")
ap.add_argument("--path_n", type=int, default=3, help="pathcheck: prompts with the largest |dT|")
ap.add_argument("--limit", type=int, default=0, help="only the first N pool prompts (smoke)")
ap.add_argument("--tiny", action="store_true", help="CPU smoke test on a tiny random Qwen3.5 model")
args = ap.parse_args()
assert args.steps >= 32 or args.tiny, "pre-registered: >= 32 IG steps"  # 2026-11-01: run at 256, max 2048 (trapezoid)
T0 = time.time()
os.makedirs(args.out, exist_ok=True)
PH = set(args.phases.split(","))
torch.manual_seed(0)
OWN = ["claims-called-911", "denying-ai-identity", "fabricated-code-execution", "false-physical-embodiment",
       "laser-at-aircraft"]
LOO = ["recommends-drunk-driving", "chemtrails-assertion"]
AX_BLOCKS = [16, 32, 48]
PAD_ID = 248044  # <|endoftext|>, the tokenizer's pad token


def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f}m]", *a, flush=True)


def jdump(obj, name):
    with open(f"{args.out}/{name}", "w") as fh:
        json.dump(obj, fh)


# ============================================================== model
if args.tiny:
    from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(f"{REPO}/data/tokenizer")
    cfg = Qwen3_5Config()
    t = cfg.text_config
    NL, DM, DI = 8, 64, 96
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = DM, DI, NL
    t.layer_types = t.layer_types[:NL]
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 32
    t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 2, 16, 16
    v = cfg.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 1, 32, 64, 2, DM
    model = Qwen3_5ForConditionalGeneration(cfg).float().eval()
    dev = "cpu"
    lmap = lambda L: L * NL // 64  # noqa: E731
    nmap = lambda n: n % DI  # noqa: E731
    AXB = [lmap(b) for b in AX_BLOCKS]
else:
    from harness import Harness
    H = Harness()
    model, tok = H.model, H.tok
    dev = "cuda"
    NL, DM, DI = 64, 5120, 17408
    lmap = nmap = (lambda x: x)  # noqa: E731
    AXB = AX_BLOCKS
model.requires_grad_(False)
lm = model.model.language_model
layers = lm.layers
lm_head = model.get_output_embeddings()
LT = lm.config.layer_types
DT = next(model.parameters()).dtype
from transformers.masking_utils import create_causal_mask  # noqa: E402
log("model loaded", type(model).__name__, NL, "layers")


def run_from(h, L1):
    """Decoder layers 0..L1-1 on inputs_embeds h [B,T,D] (no padding, no cache), as scripts/70_persona_p1_gpu.py."""
    B, T, _ = h.shape
    pos = torch.arange(T, device=h.device).view(1, 1, -1).expand(4, B, -1)
    tpos, pos3 = pos[0], pos[1:]
    mk = dict(config=lm.config, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=tpos)
    try:
        cm = create_causal_mask(**mk)
    except TypeError:
        cm = create_causal_mask(**mk, cache_position=torch.arange(T, device=h.device))
    masks = {"full_attention": cm, "linear_attention": None}
    pe = lm.rotary_emb(h, pos3)
    for i in range(L1):
        o = layers[i](h, position_embeddings=pe, attention_mask=masks[LT[i]], position_ids=tpos, past_key_values=None)
        h = o[0] if isinstance(o, tuple) else o
    return h


def embed(ids):
    return lm.embed_tokens(torch.tensor([ids], device=dev))


# ============================================================== data
pool = [json.loads(l) for l in open(args.pool)]
if args.limit:
    pool = pool[:args.limit]
gen = {}
for line in gzip.open(f"{args.data}/stage_a_gen.jsonl.gz", "rt"):
    g = json.loads(line)
    gen[g["prompt_id"]] = g
rates = {r["prompt_id"]: r for r in json.load(open(f"{args.data}/stage_a_rates_off.json"))}
forks = {json.loads(l)["rid"]: json.loads(l) for l in open(f"{args.data}/forks_off.jsonl")}
abl = json.load(open(f"{args.data}/ablation_sets.json"))
shared = json.load(open(f"{args.data}/shared_sets.json"))
fork_act = np.load(f"{args.p1}/b_fork_act.npz")


def spec_list(spec):
    return [(int(L), int(n)) for L, ns in sorted(spec.items(), key=lambda kv: int(kv[0])) for n in ns]


SETS = {}  # behavior -> {"set": (name, [(L, n)]), "ctrl": (name, [(L, n)])}
for b in OWN:
    SETS[b] = {"set": (f"{b}|top100", spec_list(abl[f"{b}|top100"]["spec"])),
               "ctrl": (f"{b}|rand100", spec_list(abl[f"{b}|rand100"]["spec"]))}
for b in LOO:
    SETS[b] = {"set": (f"loo100|{b}", spec_list(shared[f"loo100|{b}"]["spec"])),
               "ctrl": (f"rand_loo100|{b}", spec_list(shared[f"rand_loo100|{b}"]["spec"]))}
for b, d in SETS.items():
    cs = [sorted(L for L, _ in d[r][1]) for r in ("set", "ctrl")]
    assert cs[0] == cs[1] and len(cs[0]) == 100, f"{b}: set / ctrl not layer-matched"
SET_NAMES = [d[r][0] for d in SETS.values() for r in ("set", "ctrl")]


def dev_neuron(L, n):  # real (L, n) -> index in the loaded model (identity except --tiny)
    return lmap(L), nmap(n)


def transcript_ids(p):
    reply = gen[p["prompt_id"]]["samples"][int(p["transcript_i"])]["ids"][:args.max_reply]
    return list(p["prompt_ids"]), list(reply)


# ============================================================== set index / hooks
class SetIndex:
    """Union of neuron columns per (device) layer for several named sets, and each set's (column, coefficient) terms."""

    def __init__(self, sets):  # {name: [(L, n, coef)]} in real coordinates
        self.layers = sorted({dev_neuron(L, n)[0] for s in sets.values() for L, n, _ in s})
        un = {}
        for s in sets.values():
            for L, n, _ in s:
                dl, dn = dev_neuron(L, n)
                un.setdefault(dl, set()).add(dn)
        self.union = {L: sorted(v) for L, v in un.items()}
        self.col = {L: {n: j for j, n in enumerate(v)} for L, v in self.union.items()}
        self.idx = {L: torch.tensor(v, device=dev) for L, v in self.union.items()}
        self.terms = {}
        for name, s in sets.items():
            by = {}
            for L, n, c in s:
                dl, dn = dev_neuron(L, n)
                by.setdefault(dl, ([], []))
                by[dl][0].append(self.col[dl][dn])
                by[dl][1].append(c)
            self.terms[name] = {L: (torch.tensor(j, device=dev), torch.tensor(c, device=dev, dtype=torch.float32))
                                for L, (j, c) in by.items()}
        self.maxL = max(self.layers)

    def hooks(self, pos, store):
        """down_proj pre-hooks storing a[:, pos][..., union] (graph kept when grads are on)."""
        hs = []
        for L in self.layers:
            def pre(mod, a, L=L):
                store[L] = a[0][:, pos][..., self.idx[L]]
            hs.append(layers[L].mlp.down_proj.register_forward_pre_hook(pre))
        return hs

    def T(self, store, name):
        """sum_i c_i a_i, averaged over the stored positions -> [B] float32."""
        tot = 0.0
        for L, (j, c) in self.terms[name].items():
            tot = tot + (store[L][..., j].float() * c).sum(-1)
        return tot.mean(-1)


def forward_T(X, sidx, pos, names, upto=None):
    store = {}
    hs = sidx.hooks(pos, store)
    try:
        run_from(X, (upto or sidx.maxL) + 1)
    finally:
        for h in hs:
            h.remove()
    return {nm: sidx.T(store, nm) for nm in names}


# ============================================================== prep
prep_path = f"{args.out}/prep.json"
if "prep" in PH:
    prep = {}
    norm = lm.norm
    for b, d in SETS.items():
        prep[b] = {}
        for role in ("set", "ctrl"):
            name, ns = d[role]
            abar = np.array([float(fork_act[b][L, n]) for L, n in ns])
            W = torch.stack([layers[dl].mlp.down_proj.weight[:, dn].float() for dl, dn in (dev_neuron(L, n) for L, n in ns)])
            u = (torch.tensor(abar, dtype=torch.float32, device=W.device)[:, None] * W).sum(0)
            un = float(u.norm())
            c = (W @ (u / max(un, 1e-12))).cpu().numpy()
            contrib = abar * c
            top10 = [int(i) for i in np.argsort(-contrib, kind="stable")[:10]]
            sg = np.where(abar >= 0, 1.0, -1.0)
            with torch.no_grad():
                v = torch.tensor(sg, dtype=torch.float32, device=W.device)[:, None] * W
                logits = lm_head(norm(v.to(DT)[:, None, :]))[:, 0].float()
                top = logits.topk(8, -1).indices.cpu().tolist()
            prep[b][role] = {"name": name, "neurons": [[L, n] for L, n in ns], "abar": abar.tolist(), "c": c.tolist(),
                             "contrib": contrib.tolist(), "u_norm": un, "top10": top10,
                             "logit_top": [[tok.decode([t]) for t in row] for row in top]}
            log(f"prep {name}: |u| {un:.3f}, top10 layers {[ns[i][0] for i in top10]}")
    jdump(prep, "prep.json")
prep = json.load(open(prep_path))
TERMS = {prep[b][r]["name"]: [(L, n, c) for (L, n), c in zip(prep[b][r]["neurons"], prep[b][r]["c"])]
         for b in prep for r in ("set", "ctrl")}
SIDX_ALL = SetIndex(TERMS)


def thash(name):
    """Hash of one prepared set definition (name, neurons, coefficients as stored in prep.json); identical to
    terms_hash() in scripts/93_chive_evidence.py. Resumed rows are only reused when it matches."""
    obj = {"name": name, "terms": [[int(L), int(n), c] for L, n, c in TERMS[name]]}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


HASH = {b: {r: thash(prep[b][r]["name"]) for r in ("set", "ctrl")} for b in prep}
PREP_HASH = hashlib.sha256(json.dumps(HASH, sort_keys=True).encode()).hexdigest()[:16]


def bank_dirs():
    """[3, 21, D]: Assistant Axis (P1 bank row 0) + P1's 20 isotropic R32 directions, per block 16/32/48."""
    if args.tiny:
        g = torch.Generator().manual_seed(1)
        return torch.nn.functional.normalize(torch.randn(3, 21, DM, generator=g), dim=-1).to(dev)
    bank = np.load(f"{args.p1}/bank.npz")
    R32 = np.load(f"{args.p1}/random_dirs.npz")["R32"]
    return torch.tensor(np.stack([np.concatenate([bank[f"b{b}"][:1], R32]) for b in AX_BLOCKS]),
                        dtype=torch.float32, device=dev)


DIRS = bank_dirs()


@torch.no_grad()
def capture(ids, pos, sidx=None, blocks=()):
    """No-grad forward of one sequence: union neuron columns [npos, k] (if sidx) and block outputs projected on DIRS
    [npos, 3, 21] (if blocks)."""
    store, hs, bo = {}, [], {}
    pos_t = torch.tensor(pos, device=dev)
    if sidx is not None:
        hs += sidx.hooks(pos_t, store)
    for j, b in enumerate(blocks):
        def fh(mod, a, o, j=j):
            x = o[0] if isinstance(o, tuple) else o
            bo[j] = x[0, pos_t].float() @ DIRS[j].T
        hs.append(layers[b].register_forward_hook(fh))
    try:
        upto = max([sidx.maxL if sidx is not None else 0] + list(blocks))
        run_from(embed(ids), upto + 1)
    finally:
        for h in hs:
            h.remove()
    cols = None
    if sidx is not None:
        cols = torch.cat([store[L][0].float() for L in sidx.layers], -1).cpu().numpy()  # [npos, k], union order
    proj = torch.stack([bo[j] for j in range(len(blocks))], 1).cpu().numpy() if blocks else None
    return cols, proj


def union_keys(sidx):
    return [(L, n) for L in sidx.layers for n in sidx.union[L]]


# ============================================================== acts
if "acts" in PH:
    ukeys = union_keys(SIDX_ALL)  # device coordinates
    ucol = {kk: j for j, kk in enumerate(ukeys)}
    reals = sorted({(L, n) for s in TERMS.values() for L, n, _ in s})
    s1 = np.zeros(len(ukeys)); s2 = np.zeros(len(ukeys)); cnt = 0
    acts, meta, axr = {}, {}, {}
    for k, p in enumerate(pool):
        pids, rids = transcript_ids(p)
        P, R = len(pids), len(rids)
        cols, proj = capture(pids + rids, list(range(P - 1, P + R)), SIDX_ALL, AXB)
        axr[p["prompt_id"]] = proj.astype(np.float32)  # [R+1, 3, 21]
        x = cols[:R].astype(np.float64)  # states choosing reply tokens 0..R-1
        s1 += x.sum(0); s2 += (x ** 2).sum(0); cnt += R
        b = p["behavior_id"]
        if b in SETS:
            ns = prep[b]["set"]["neurons"] + prep[b]["ctrl"]["neurons"]
            sel = [ucol[dev_neuron(L, n)] for L, n in ns]
            acts[p["prompt_id"]] = cols[:, sel].astype(np.float16)
            meta[p["prompt_id"]] = {"neurons": ns, "n_set": 100, "R": R, "P": P, "transcript_i": int(p["transcript_i"]),
                                    "terms_hash": HASH[b]}
        if k % 20 == 0:
            log(f"acts {k + 1}/{len(pool)} R={R}")
    mu = s1 / cnt
    sd = np.sqrt(np.maximum(s2 / cnt - mu ** 2, 1e-12))
    stats = {"n_states": int(cnt), "convention": "state j = position P-1+j chooses reply token j", "prep_hash": PREP_HASH,
             "neurons": {f"{L}|{n}": [float(mu[ucol[dev_neuron(L, n)]]), float(sd[ucol[dev_neuron(L, n)]])]
                         for L, n in reals}}
    jdump(stats, "stats.json")
    jdump(meta, "acts_meta.json")
    np.savez_compressed(f"{args.out}/acts.npz", **acts)
    np.savez_compressed(f"{args.out}/axis_reply.npz", **axr)
    # fork replies (axis arm, fork-window option)
    axf = {}
    for p in pool:
        rid = p.get("fork_rid")
        if not rid:
            continue
        pid, i = rid.split("|")
        rr = gen[pid]["samples"][int(i)]["ids"][:int(forks[rid]["L"])]
        P = len(p["prompt_ids"])
        _, proj = capture(list(p["prompt_ids"]) + list(rr), list(range(P - 1, P + len(rr))), None, AXB)
        axf[rid] = proj.astype(np.float32)
    np.savez_compressed(f"{args.out}/axis_fork.npz", **axf)
    # T of every set at the last token of all 403 stage-A prompts
    names = list(TERMS)
    pids403 = sorted(gen)
    if args.limit:
        pids403 = pids403[:max(args.limit, 8)]
    T403 = np.zeros((len(pids403), len(names)), np.float32)
    R403 = np.zeros((len(pids403), 3, DM), np.float16)  # unbatched last-token residuals, blocks 16/32/48
    with torch.no_grad():
        for k, pid in enumerate(pids403):
            ids = list(gen[pid]["prompt_ids"])
            bst, bh = {}, []
            for j, b in enumerate(AXB):
                def fh(mod, a, o, j=j):
                    bst[j] = (o[0] if isinstance(o, tuple) else o)[0, -1].float()
                bh.append(layers[b].register_forward_hook(fh))
            try:
                out = forward_T(embed(ids), SIDX_ALL, torch.tensor([len(ids) - 1], device=dev), names,
                                upto=max(SIDX_ALL.maxL, max(AXB)))
            finally:
                for h in bh:
                    h.remove()
            T403[k] = [float(out[nm][0]) for nm in names]
            R403[k] = torch.stack([bst[j] for j in range(3)]).cpu().numpy()
    np.savez(f"{args.out}/T403.npz", pids=np.array(pids403), names=np.array(names), T=T403, prep_hash=PREP_HASH,
             behavior=np.array([gen[p]["behavior_id"] for p in pids403]), split=np.array([gen[p]["split"] for p in pids403]))
    np.savez_compressed(f"{args.out}/resid403.npz", pids=np.array(pids403), blocks=np.array(AX_BLOCKS), resid=R403)
    log(f"acts done: {len(acts)} transcripts with sets, {len(axr)} axis replies, {len(axf)} fork replies, T403 {T403.shape}")

# ============================================================== axisref
P1_QUESTIONS = [
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
]  # = scripts/70_persona_p1_gpu.py QUESTIONS (benign.jsonl replies)
if "axisref" in PH:
    items = [("p1", q) for q in P1_QUESTIONS]
    items += [("induction_benign", json.loads(l)["prompt"]) for l in open(f"{args.promptsets}/induction_benign.jsonl")]
    items += [("capability_mini", json.loads(l)["prompt"]) for l in open(f"{args.promptsets}/capability_mini.jsonl")]
    if args.limit:
        items = items[:args.limit * 4]
    resid = np.zeros((len(items), 3, DM), np.float16)
    for k, (_, text) in enumerate(items):
        ids = template_user_message(text, tok)
        store, hs = {}, []
        for j, b in enumerate(AXB):
            def fh(mod, a, o, j=j):
                x = o[0] if isinstance(o, tuple) else o
                store[j] = x[0, -1].float()
            hs.append(layers[b].register_forward_hook(fh))
        try:
            with torch.no_grad():
                run_from(embed(ids), max(AXB) + 1)
        finally:
            for h in hs:
                h.remove()
        resid[k] = torch.stack([store[j] for j in range(3)]).cpu().numpy()
    last_proj = np.einsum("nbd,bkd->nbk", resid.astype(np.float32), DIRS.cpu().numpy())
    reply_proj = []
    for r in map(json.loads, open(f"{args.p1}/benign.jsonl")):
        if r["style"] != "default":
            continue
        P, R = len(r["prompt_ids"]), len(r["ids"])
        _, proj = capture(r["prompt_ids"] + r["ids"], list(range(P - 1, P + R - 1)), None, AXB)
        reply_proj.append(proj)
    np.savez_compressed(f"{args.out}/axis_benign.npz", resid_last=resid, last_proj=last_proj.astype(np.float32),
                        reply_proj=np.concatenate(reply_proj).astype(np.float32), source=np.array([s for s, _ in items]),
                        blocks=np.array(AX_BLOCKS), dirs="row 0 = Assistant Axis (bank.npz), rows 1-20 = R32")
    log(f"axisref done: {len(items)} benign prompts, {sum(len(x) for x in reply_proj)} benign reply tokens")

# ============================================================== integrated gradients
CONTENT_START, SUFFIX_LEN = 3, 9  # "<|im_start|>user\n" | content | "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def ig(ids, P, target_pos, names, steps):
    """IG of each T[name] w.r.t. the embeddings of the user-message tokens (ids[3:P-9]); baseline = pad embedding there.
    Trapezoid rule on the uniform grid alpha_s = s / steps, s = 0..steps; when completeness fails the grid is doubled
    and the gradients already computed are reused (the new points are the midpoints), per target, until --max_steps.
    Returns {name: {"attr": [n_content], "T_x", "T_base", "steps", "rel_err", "abs_err_frac", "ok"}}."""
    lo, hi = CONTENT_START, P - SUFFIX_LEN
    with torch.no_grad():
        e = embed(ids)[0].float()  # [T, D]
        b = e.clone()
        b[lo:hi] = embed([PAD_ID])[0, 0].float()
        delta = e - b
    pos = torch.tensor(target_pos, device=dev)
    sidx = SetIndex({nm: TERMS[nm] for nm in names})
    with torch.no_grad():
        ends = forward_T(torch.stack([e, b]).to(DT), sidx, pos, names)
    Tn = len(ids)

    def grads_at(alphas, nms):
        """sum over the given alphas of dT/dX[lo:hi] for every name in nms -> {name: [hi-lo, DM]} (OOM halves rows)."""
        G = {nm: torch.zeros(hi - lo, DM, device=dev) for nm in nms}
        rows = max(1, min(len(alphas), args.tok_budget // Tn))
        s = 0
        while s < len(alphas):
            al = alphas[s:s + rows].to(dev)
            try:
                X = (b[None] + al[:, None, None] * delta[None]).to(DT).requires_grad_(True)
                Tv = forward_T(X, sidx, pos, nms)
                gs = [torch.autograd.grad(Tv[nm].sum(), X, retain_graph=j < len(nms) - 1)[0][:, lo:hi].float().sum(0)
                      for j, nm in enumerate(nms)]
                for nm, g in zip(nms, gs):  # only after every target succeeded (an OOM retry must not double-count)
                    G[nm] += g
                del X, Tv, gs
                s += len(al)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as ex:  # cuBLAS reports OOM as CUBLAS_STATUS_ALLOC_FAILED
                if not isinstance(ex, torch.cuda.OutOfMemoryError) and not any(
                        k in str(ex) for k in ("out of memory", "CUBLAS_STATUS_ALLOC_FAILED", "CUDA_ERROR_OUT_OF_MEMORY")):
                    raise
                X = Tv = gs = None
                torch.cuda.empty_cache()
                if rows == 1:
                    raise
                rows = max(1, rows // 2)
                log(f"OOM at T={Tn}: rows -> {rows}")
        return G

    res = {}
    todo = list(names)
    G_end = grads_at(torch.tensor([0.0, 1.0]), todo)           # the two endpoints (weight 1/2 each)
    G_int = {nm: torch.zeros(hi - lo, DM, device=dev) for nm in todo}
    K = steps
    G_int_add = grads_at((torch.arange(1, K, dtype=torch.float32)) / K, todo)   # interior grid points 1..K-1
    for nm in todo:
        G_int[nm] += G_int_add[nm]
    while True:
        still = []
        for nm in todo:
            attr = (delta[lo:hi] * (G_end[nm] / 2 + G_int[nm]) / K).sum(-1)
            tx, tb = float(ends[nm][0]), float(ends[nm][1])
            dT = tx - tb
            err = abs(float(attr.sum()) - dT)
            rel = err / max(abs(dT), 1e-12)
            absf = err / max(float(attr.abs().sum()), 1e-12)
            res[nm] = {"attr": attr.cpu().tolist(), "T_x": tx, "T_base": tb, "steps": K, "rel_err": rel,
                       "abs_err_frac": absf, "ok": rel <= args.tol, "rule": "trapezoid"}
            if rel > args.tol:
                still.append(nm)
        if not still or K * 2 > args.max_steps:
            return res
        mids = (torch.arange(K, dtype=torch.float32) + 0.5) / K     # the new points of the doubled grid
        G_add = grads_at(mids, still)
        for nm in still:
            G_int[nm] += G_add[nm]
        K *= 2
        todo = still


def fd_check(ids, P, target_pos, name, eps=0.05, n=3):
    """Central finite difference of T along random directions on the user-message embeddings vs gradient . d."""
    lo, hi = CONTENT_START, P - SUFFIX_LEN
    pos = torch.tensor(target_pos, device=dev)
    sidx = SetIndex({name: TERMS[name]})
    e = embed(ids)[0].float()
    X = e[None].to(DT).requires_grad_(True)
    g = torch.autograd.grad(forward_T(X, sidx, pos, [name])[name].sum(), X)[0][0].float()
    out = []
    gen_ = torch.Generator().manual_seed(7)
    for _ in range(n):
        d = torch.zeros_like(e)
        r = torch.randn(hi - lo, DM, generator=gen_).to(dev)
        d[lo:hi] = r / r.norm(dim=-1, keepdim=True) * e[lo:hi].norm(dim=-1, keepdim=True) * eps
        with torch.no_grad():
            tp = forward_T(torch.stack([e + d, e - d]).to(DT), sidx, pos, [name])[name]
        out.append({"fd": float(tp[0] - tp[1]) / 2, "lin": float((g * d).sum())})
    return out


def target_positions(P, R):
    return [P - 1] if args.positions == "last" else list(range(P - 1, P + R - 1))


if "ig" in PH:
    path = f"{args.out}/attrib.jsonl" if args.positions == "last" else f"{args.out}/attrib_reply.jsonl"

    def ig_key(pid, b, ti):  # a rebuilt prep.json (or a new transcript for reply targets) never reuses stale rows
        return (pid, HASH[b]["set"], HASH[b]["ctrl"], args.positions, ti if args.positions == "reply" else None)
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if r["behavior_id"] in HASH and r.get("terms_hash") == HASH[r["behavior_id"]]:
                done.add(ig_key(r["prompt_id"], r["behavior_id"], int(r["transcript_i"])))
    fds = []
    with open(path, "a") as fh:
        SI, SN = (int(x) for x in args.ig_shard.split("/"))
        for k, p in enumerate(pool):
            b = p["behavior_id"]
            if b not in SETS or ig_key(p["prompt_id"], b, int(p["transcript_i"])) in done or k % SN != SI:
                continue
            t0 = time.time()
            pids, rids = transcript_ids(p)
            ids = pids if args.positions == "last" else pids + rids
            P = len(pids)
            assert ids[P - 4:P] == [248068, 271, 248069, 271] or args.tiny
            names = [prep[b]["set"]["name"], prep[b]["ctrl"]["name"]]
            res = ig(ids, P, target_positions(P, len(rids)), names, args.steps)
            row = {"prompt_id": p["prompt_id"], "behavior_id": b, "split": p["split"],
                   "transcript_i": int(p["transcript_i"]), "positions": args.positions, "P": P, "terms_hash": HASH[b],
                   "content": [CONTENT_START, P - SUFFIX_LEN], "content_ids": pids[CONTENT_START:P - SUFFIX_LEN],
                   "baseline_id": PAD_ID, "set": {"name": names[0], **res[names[0]]},
                   "ctrl": {"name": names[1], **res[names[1]]}, "secs": round(time.time() - t0, 1)}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if len(fds) < 2:
                fds.append({"prompt_id": p["prompt_id"], "checks": fd_check(ids, P, target_positions(P, len(rids)), names[0])})
            log(f"ig {k + 1}/{len(pool)} {b[:12]} T={len(ids)} steps {res[names[0]]['steps']} "
                f"rel_err set {res[names[0]]['rel_err']:.3f} ctrl {res[names[1]]['rel_err']:.3f} {row['secs']}s")
    if fds:
        jdump(fds, "fd.json")

# ============================================================== occlusion check (non-pool prompts only)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def unit_of_tokens(text, units):
    """token index -> unit index (max char overlap; whitespace-only tokens go to the nearest unit)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    out = []
    for s, e in enc["offset_mapping"]:
        ov = [max(0, min(e, u["end"]) - max(s, u["start"])) for u in units]
        if max(ov) > 0:
            out.append(int(np.argmax(ov)))
        else:
            mid = (s + e) / 2
            out.append(int(np.argmin([min(abs(mid - u["start"]), abs(mid - u["end"])) for u in units])))
    return enc["input_ids"], out


if "occl" in PH:
    poolset = {json.loads(l)["prompt_id"] for l in open(args.pool)}  # the full pool, even with --limit
    cand = {}
    for pid, g in gen.items():
        b = g["behavior_id"]
        if b in SETS and pid not in poolset:
            user = tok.decode(g["prompt_ids"][CONTENT_START:-SUFFIX_LEN])
            if len(split_units(user)) >= 3:
                cand.setdefault(b, []).append(pid)
    sel = [pid for b in sorted(cand) for pid in sorted(cand[b], key=lambda x: hashlib.sha256(x.encode()).hexdigest())[:args.n_occl]]
    if args.limit:
        sel = sel[:args.limit]
    path = f"{args.out}/occl.jsonl"
    done = {(json.loads(l)["prompt_id"], json.loads(l).get("terms_hash")) for l in open(path)} if os.path.exists(path) else set()
    with open(path, "a") as fh:
        for pid in sel:
            if (pid, HASH[gen[pid]["behavior_id"]]["set"]) in done:
                continue
            g = gen[pid]
            b, ids = g["behavior_id"], list(g["prompt_ids"])
            P = len(ids)
            user = tok.decode(ids[CONTENT_START:P - SUFFIX_LEN])
            units = split_units(user)
            cids, t2u = unit_of_tokens(user, units)
            if cids != ids[CONTENT_START:P - SUFFIX_LEN]:
                log(f"occl skip {pid[:8]}: offsets do not reproduce the content ids")
                continue
            name = prep[b]["set"]["name"]
            r = ig(ids, P, [P - 1], [name], args.steps)[name]
            ig_u = np.zeros(len(units))
            for t, u in enumerate(t2u):
                ig_u[u] += r["attr"][t]
            sidx = SetIndex({name: TERMS[name]})
            with torch.no_grad():
                e = embed(ids)[0]
                padv = embed([PAD_ID])[0, 0]
                Xs = []
                for ui in range(len(units)):
                    x = e.clone()
                    for t, u in enumerate(t2u):
                        if u == ui:
                            x[CONTENT_START + t] = padv
                    Xs.append(x)
                tx = r["T_x"]
                d_pad = [tx - float(forward_T(x[None], sidx, torch.tensor([P - 1], device=dev), [name])[name][0])
                         for x in Xs]
                d_del = []
                for u in units:
                    if user.count(u["text"]) != 1:
                        d_del.append(None)
                        continue
                    ids2 = template_user_message(apply_edit(user, "delete", u["text"], ""), tok)
                    d_del.append(tx - float(forward_T(embed(ids2), sidx, torch.tensor([len(ids2) - 1], device=dev),
                                                      [name])[name][0]))
            ok = [i for i, v in enumerate(d_del) if v is not None]
            row = {"prompt_id": pid, "behavior_id": b, "set": name, "terms_hash": HASH[b]["set"], "n_units": len(units),
                   "ig_units": ig_u.tolist(),
                   "d_pad": d_pad, "d_del": d_del, "ig_rel_err": r["rel_err"], "ig_steps": r["steps"],
                   "rho_pad": spearman(ig_u, d_pad), "rho_del": spearman(ig_u[ok], [d_del[i] for i in ok])}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            log(f"occl {pid[:8]} {b[:12]} units {len(units)} rho_pad {row['rho_pad']} rho_del {row['rho_del']}")

# ============================================================== path check
# Gradient vs forward-only slopes along the IG path for the prompts with the largest |dT|: T at K+1 grid points (no
# grad, twice in different batch compositions to measure bf16 forward noise) against grad . delta / K at the K
# midpoints. Separates "gradients are wrong" (steps disagree beyond the forward noise) from "the path is too
# curved for the step count" (steps agree, sums still differ) from "IG bookkeeping bug" (sum_grad matches dT here
# while the attrib rows do not).
if "pathcheck" in PH:
    K = args.path_k
    arows = [json.loads(l) for l in open(f"{args.out}/attrib.jsonl")]
    arows.sort(key=lambda r: -abs(r["set"]["T_x"] - r["set"]["T_base"]))
    out = []
    for r in arows[:args.path_n]:
        p = next(q for q in pool if q["prompt_id"] == r["prompt_id"])
        b = p["behavior_id"]
        pids, rids = transcript_ids(p)
        ids, P = pids, len(pids)
        names = [prep[b]["set"]["name"], prep[b]["ctrl"]["name"]]
        lo, hi = CONTENT_START, P - SUFFIX_LEN
        pos = torch.tensor(target_positions(P, len(rids)), device=dev)
        sidx = SetIndex({nm: TERMS[nm] for nm in names})
        with torch.no_grad():
            e = embed(ids)[0].float()
            bb = e.clone()
            bb[lo:hi] = embed([PAD_ID])[0, 0].float()
            delta = e - bb
        Tn = len(ids)
        rows = max(1, min(K + 1, args.tok_budget // Tn))

        def T_at(alphas):
            vals = {nm: [] for nm in names}
            for s in range(0, len(alphas), rows):
                al = alphas[s:s + rows].to(dev)
                with torch.no_grad():
                    Tv = forward_T((bb[None] + al[:, None, None] * delta[None]).to(DT), sidx, pos, names)
                for nm in names:
                    vals[nm] += Tv[nm].float().cpu().tolist()
            return vals
        grid = torch.arange(K + 1, dtype=torch.float32) / K
        Tg, Tg2 = T_at(grid), T_at(grid.flip(0))
        mids = (torch.arange(K, dtype=torch.float32) + 0.5) / K
        gd = {nm: [] for nm in names}
        grows = max(1, min(K, (args.tok_budget // 2) // Tn))
        for s in range(0, K, grows):
            al = mids[s:s + grows].to(dev)
            X = (bb[None] + al[:, None, None] * delta[None]).to(DT).requires_grad_(True)
            Tv = forward_T(X, sidx, pos, names)
            for j, nm in enumerate(names):
                g = torch.autograd.grad(Tv[nm].sum(), X, retain_graph=j < len(names) - 1)[0][:, lo:hi].float()
                gd[nm] += (g * delta[lo:hi][None]).sum((1, 2)).cpu().tolist()
            del X, Tv
            torch.cuda.empty_cache()
        rec = {"prompt_id": r["prompt_id"], "behavior_id": b, "K": K, "T_len": Tn, "content_tokens": hi - lo}
        for k_, nm in enumerate(names):
            fd = [Tg[nm][i + 1] - Tg[nm][i] for i in range(K)]
            pred = [x / K for x in gd[nm]]
            rebatched = Tg2[nm][::-1]
            noise = [abs(a - c) for a, c in zip(Tg[nm], rebatched)]
            rec[nm] = {"T_grid": Tg[nm], "T_grid_rebatched": rebatched, "fd_steps": fd, "grad_steps": pred,
                       "sum_fd": sum(fd), "sum_grad": sum(pred), "dT": Tg[nm][-1] - Tg[nm][0],
                       "fwd_noise_max": max(noise), "attrib_rel_err": r["set" if k_ == 0 else "ctrl"]["rel_err"],
                       "attrib_dT": r["set" if k_ == 0 else "ctrl"]["T_x"] - r["set" if k_ == 0 else "ctrl"]["T_base"]}
            log(f"pathcheck {r['prompt_id'][:8]} {nm}: dT {rec[nm]['dT']:.2f} sum_grad {rec[nm]['sum_grad']:.2f} "
                f"fwd_noise_max {max(noise):.2f} max|fd-grad| step {max(abs(a - c) for a, c in zip(fd, pred)):.2f}")
        out.append(rec)
    jdump(out, "pathcheck.json")

# ============================================================== gate + summary
if "gate" in PH:
    rng = np.random.default_rng(0)
    out = {"tol": args.tol, "positions": args.positions, "gate": {}, "rule": (
        "per set: sd over the behavior's prompts among the 403 stage-A prompts of T_set vs T_ctrl; pass if the 95% "
        "bootstrap CI (2,000 prompt resamples) of sd_set / sd_ctrl is above 1. Pre-registered; reported, not retuned.")}
    z = np.load(f"{args.out}/T403.npz")
    names, Tm, beh = list(z["names"]), z["T"], z["behavior"]
    poolids = {p["prompt_id"] for p in pool}
    inpool = np.array([p in poolids for p in z["pids"]])
    for b in SETS:
        for scope, m in (("all403", beh == b), ("pool", (beh == b) & inpool)):
            if m.sum() < 3:
                continue
            ts, tc = Tm[m, names.index(prep[b]["set"]["name"])], Tm[m, names.index(prep[b]["ctrl"]["name"])]
            idx = np.arange(m.sum())
            bs = []
            for _ in range(2000):
                ii = rng.choice(idx, len(idx))
                bs.append(ts[ii].std() / max(tc[ii].std(), 1e-12))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out["gate"].setdefault(b, {})[scope] = {"n": int(m.sum()), "sd_set": float(ts.std()),
                                                   "sd_ctrl": float(tc.std()), "ratio": float(ts.std() / max(tc.std(), 1e-12)),
                                                   "ci95": [float(lo), float(hi)], "pass": bool(lo > 1)}
    path = f"{args.out}/attrib.jsonl" if args.positions == "last" else f"{args.out}/attrib_reply.jsonl"
    if os.path.exists(path):
        rows = [r for r in map(json.loads, open(path))  # current set definitions only (stale resumed rows ignored)
                if r["behavior_id"] in HASH and r.get("terms_hash") == HASH[r["behavior_id"]]]
        for role in ("set", "ctrl"):
            rel = np.array([r[role]["rel_err"] for r in rows])
            out[f"completeness_{role}"] = {"n": len(rows), "ok": int(sum(r[role]["ok"] for r in rows)),
                                           "median_rel_err": float(np.median(rel)), "max_rel_err": float(rel.max()),
                                           "steps": {str(s): int(sum(r[role]["steps"] == s for r in rows))
                                                     for s in sorted({r[role]["steps"] for r in rows})}}
    if os.path.exists(f"{args.out}/fd.json"):
        out["fd"] = json.load(open(f"{args.out}/fd.json"))
    if os.path.exists(f"{args.out}/occl.jsonl"):
        oc = [r for r in map(json.loads, open(f"{args.out}/occl.jsonl"))
              if r["behavior_id"] in HASH and r.get("terms_hash") == HASH[r["behavior_id"]]["set"]]
        for k in ("rho_pad", "rho_del"):
            v = [r[k] for r in oc if r[k] is not None]
            out[f"occl_{k}"] = {"n": len(v), "median": float(np.median(v)) if v else None,
                                "mean": float(np.mean(v)) if v else None}
    jdump(out, "checks.json")
    log("checks:", json.dumps({k: v for k, v in out.items() if k != "fd"})[:1500])
log("done")
