"""persona-fwd A4 / X1 (GPU, forward passes only, resumable): do Agastya's top-100 sets cancel the persona slip?

Cut points: the fork positions of his informative fork replies (P1 rule: >= 2 next-token candidates, q range >= .25;
next_tokens from phase2_forks.jsonl if present, else forks_off.jsonl) for the 5 behaviors with own top-100 sets (all
on discovery prompts), plus held-out denying-AI forks rebuilt from forks_samples_on_dai_held.jsonl (fork cut c with
'c+tok' branches; candidates = the reply's own token at c, q = P(cut c+1), and the branch tokens, q = mean match).
Fork value V = sum p q / sum p over the candidates, p = model next-token probability.
Rows per fork (all from one cached hidden state at the first edited layer):
  clean | steer (axis, dose -0.25 x median resid norm at block 32 output, P1's axis-0.25) | top100 | steer+top100 |
  rand100 | steer+rand100 | rsteer_j (P1's 20 random unit directions R32, same dose) | rsteer_j + top100
Zero-ablation = his method: zero the down_proj input of the set's neurons. Benign KL (P1 benign default replies) for
each row type. Output: --out/x1.jsonl (one record per fork, appended; reruns skip done rids), x1_kl.json.
Run: flock /data/gpu.lock ~/venv/bin/python -I scripts/73_persona_fwd_x1.py
"""
import argparse, collections, gzip, json, os, sys, time
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
ap.add_argument("--n_rand", type=int, default=20)
ap.add_argument("--dose", type=float, default=0.25)
ap.add_argument("--tok_cap", type=int, default=49152)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
T0 = time.time()
os.makedirs(args.out, exist_ok=True)
dev = "cuda"
OWN = ["claims-called-911", "denying-ai-identity", "fabricated-code-execution", "false-physical-embodiment", "laser-at-aircraft"]


def log(*a):
    print(f"[x1 {(time.time()-T0)/60:5.1f}m]", *a, flush=True)


H = Harness(resid_norms_path=args.norms)
lm, layers, tok = H.lm, H.layers, H.tok
lm_head = H.model.get_output_embeddings()
LT = lm.config.layer_types
MED = H.resid_norms
D = 5120
axis_raw = np.load(f"{args.p1}/vectors/axis_raw.npy").astype(np.float64)
AX32 = torch.tensor(axis_raw[3] / np.linalg.norm(axis_raw[3]), dtype=torch.float32, device=dev)
R32 = torch.tensor(np.load(f"{args.p1}/random_dirs.npz")["R32"][:args.n_rand], dtype=torch.float32, device=dev)
log("model loaded")


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
split = json.load(open(f"{args.data}/split.json"))
forks = [json.loads(l) for l in open(f"{args.data}/forks_off.jsonl")]
p2 = {json.loads(l)["rid"]: json.loads(l) for l in open(f"{args.data}/phase2_forks.jsonl")}
p2_list = [json.loads(l) for l in open(f"{args.data}/phase2_forks.jsonl")]
abl_raw = json.load(open(f"{args.data}/ablation_sets.json"))
abl = {(s["b"], s["kind"], int(s["k"])): parse_spec(s["spec"]) for s in abl_raw.values()}
insample = {s["b"]: {p2_list[i]["rid"] for i in s["forks"]} for s in abl_raw.values() if s["kind"] == "top" and int(s["k"]) == 100}


def cands_of(f):
    return p2[f["rid"]]["next_tokens"] if f["rid"] in p2 else f["next_tokens"]


items = []
for f in forks:
    if f["behavior_id"] not in OWN:
        continue
    nt = cands_of(f)
    if len(nt) < 2 or max(c["q"] for c in nt) - min(c["q"] for c in nt) < 0.25:
        continue
    pid, i = f["rid"].split("|")
    r_ids = gen[pid]["samples"][int(i)]["ids"]
    lo = f["fork"][0]
    assert r_ids[lo] == f["fork_token"]
    items.append({"rid": f["rid"], "b": f["behavior_id"], "split": split[pid]["split"], "ids": gen[pid]["prompt_ids"] + r_ids[:lo],
                  "cands": [c["tok"] for c in nt], "q": [c["q"] for c in nt], "src": "forks_off",
                  "insample": f["rid"] in insample.get(f["behavior_id"], set())})
# held-out denying-AI forks (production judge)
cut = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for l in open(f"{args.data}/forks_samples_on_dai_held.jsonl"):
    x = json.loads(l)
    m = x["match"]
    c = cut[x["rid"]][x["cut"]]
    c[0] += sum(m); c[1] += len(m)
n_ho = 0
for rid, cs in cut.items():
    br = [k for k in cs if "+" in k]
    if not br:
        continue
    c0 = int(br[0].split("+")[0])
    pid, i = rid.split("|")
    r_ids = gen[pid]["samples"][int(i)]["ids"]
    if str(c0 + 1) not in cs:
        continue
    cands = [r_ids[c0]] + [int(k.split("+")[1]) for k in br]
    q = [cs[str(c0 + 1)][0] / cs[str(c0 + 1)][1]] + [cs[k][0] / cs[k][1] for k in br]
    if max(q) - min(q) < 0.25:
        continue
    items.append({"rid": rid, "b": "denying-ai-identity", "split": split[pid]["split"], "ids": gen[pid]["prompt_ids"] + r_ids[:c0],
                  "cands": cands, "q": q, "src": "dai_held_on", "insample": False})
    n_ho += 1
log(f"{len(items)} cut points ({n_ho} held-out DAI)", collections.Counter((x["b"][:12], x["split"]) for x in items))
if args.smoke:
    items = items[:3] + [x for x in items if x["src"] == "dai_held_on"][:1]

# ------------------------------------------------------------------ rows


def rows_for(b):
    top, rnd = abl[(b, "top", 100)], abl[(b, "rand", 100)]
    st = -args.dose * MED[32] * AX32
    rows = [{"name": "clean"}, {"name": "steer", "delta": st}, {"name": "top100", "abl": top},
            {"name": "steer+top100", "delta": st, "abl": top}, {"name": "rand100", "abl": rnd},
            {"name": "steer+rand100", "delta": st, "abl": rnd}]
    for j in range(args.n_rand):
        d = args.dose * MED[32] * R32[j]
        rows.append({"name": f"rsteer{j}", "delta": d})
        rows.append({"name": f"rsteer{j}+top100", "delta": d, "abl": top})
    L0 = min(min(top), min(rnd), 32)
    return rows, L0


def run_rows(h_src, L0, rows, mode, cands=None, span=None, base_lp=None):
    T = h_src.shape[1]
    per = max(1, args.tok_cap // T)
    res = []
    for s in range(0, len(rows), per):
        chunk = rows[s:s + per]
        B = len(chunk)
        h = h_src.expand(B, -1, -1).contiguous()
        dl = torch.stack([r["delta"] if r.get("delta") is not None else torch.zeros(D, device=dev) for r in chunk])
        masks = {}
        for r_i, r in enumerate(chunk):
            for l, ns in (r.get("abl") or {}).items():
                if l not in masks:
                    masks[l] = torch.ones(B, 17408, dtype=torch.bfloat16, device=dev)
                masks[l][r_i, ns] = 0
        with torch.no_grad(), Hooks() as hk:
            if bool((dl != 0).any()):
                def add(mod, a, o):
                    return (o.float() + dl[:, None, :]).to(o.dtype)
                hk.fwd(layers[32], add)
            for l, m in masks.items():
                def pre(mod, a, m=m):
                    return (a[0] * m[:, None, :],) + tuple(a[1:])
                hk.pre(layers[l].mlp.down_proj, pre)
            hout = run_from(h, L0)
            if mode == "last":
                pr = torch.softmax(lm_head(lm.norm(hout[:, -1])).float(), -1)[:, cands].cpu()
                res += [pr[r].tolist() for r in range(B)]
            else:
                a0, a1 = span
                for r in range(B):
                    lp = torch.log_softmax(lm_head(lm.norm(hout[r, a0:a1])).float(), -1)
                    res.append(float((base_lp.exp() * (base_lp - lp)).sum(-1).mean()))
        del hout, h
    return res


def stash_pass(ids, L0):
    st = {}
    with torch.no_grad(), Hooks() as hk:
        def f(mod, a, o):
            st["h"] = o.clone()
        if L0 > 0:
            hk.fwd(layers[L0 - 1], f)
        h = run_from(lm.embed_tokens(torch.tensor([ids], device=dev)), 0)
        logits = lm_head(lm.norm(h[:, -1])).float()[0]
    h0 = st["h"] if L0 > 0 else lm.embed_tokens(torch.tensor([ids], device=dev))
    return h0, logits, h


# ------------------------------------------------------------------ KL on benign (once)
klp = f"{args.out}/x1_kl.json"
if not os.path.exists(klp):
    benign = [json.loads(l) for l in open(f"{args.p1}/benign.jsonl")]
    kseqs = [r for r in benign if r["style"] == "default" and len(r["ids"]) >= 60][: (2 if args.smoke else 8)]
    kl = collections.defaultdict(list)
    for r in kseqs:
        ids = r["prompt_ids"] + r["ids"]
        P, R = len(r["prompt_ids"]), len(r["ids"])
        span = (P - 1, P + R - 1)
        for b in OWN:
            rows, L0 = rows_for(b)
            h0, _, hfull = stash_pass(ids, L0)
            base_lp = torch.log_softmax(lm_head(lm.norm(hfull[0, span[0]:span[1]])).float(), -1)
            del hfull
            for x, v in zip(rows, run_rows(h0, L0, rows, "span", span=span, base_lp=base_lp)):
                kl[f"{b}|{x['name']}"].append(v)
            del base_lp
        torch.cuda.empty_cache()
    kls = {k: float(np.mean(v)) for k, v in kl.items()}
    json.dump(kls, open(klp, "w"), indent=1)
    log("KL", {k: round(v, 4) for k, v in kls.items() if "rsteer" not in k})

# ------------------------------------------------------------------ forks
outp = f"{args.out}/x1.jsonl"
done = {json.loads(l)["rid"] for l in open(outp)} if os.path.exists(outp) else set()
log(f"{len(done)} done")
n = 0
with open(outp, "a") as fh:
    for it in sorted(items, key=lambda x: len(x["ids"])):
        if it["rid"] in done:
            continue
        tf = time.time()
        rows, L0 = rows_for(it["b"])
        h0, logits, hfull = stash_pass(it["ids"], L0)
        del hfull
        pfull = torch.softmax(logits, -1)[it["cands"]].cpu().tolist()
        res = run_rows(h0, L0, rows, "last", cands=it["cands"])
        rec = {k: it[k] for k in ("rid", "b", "split", "src", "insample", "cands", "q")}
        rec.update(T=len(it["ids"]), p_full=pfull, rows={x["name"]: p for x, p in zip(rows, res)})
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        n += 1
        del h0
        if n <= 3 or n % 20 == 0:
            log(f"fork {n} {it['b'][:12]} T={len(it['ids'])} {time.time()-tf:.1f}s")
log(f"ALL_DONE {n} forks")
