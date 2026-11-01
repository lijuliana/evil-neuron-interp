"""Eval-A capture pass (runs/monitors/SPEC.md, "Compute"): one teacher-forced forward per sequence collects everything the
cheap monitors and the lens / O-lens readers need, at every position they score.

Positions (rows): data/olens/seqs.jsonl kinds prompt (c = 0) and reply (every measured cut point incl. both endpoints,
plus the dense c_fork-40..c_fork+20 windows of the 46 sharp forks), then every UNIQUE position (canon[i] == sid) of
runs/monitors/seqs_b.jsonl (kind cont) and seqs_bh.jsonl (kind conth). State for cut c sits at P + c - 1; (b) states
at P + cut + k - 1 (k = 1, 4), i.e. the state whose input token is cont[k-1].
Per row:
  resid_emb.npy, resid_<L>.npy  [N, 5120] fp16: embeddings (= HF hidden_states[0]) and block L = output of layers[L]
                                (= hidden_states[L+1]) for L in 16, 24, 32, 40, 42, 44, 48, 56
  neurons.npy                   [N, U] fp16: down_proj inputs (silu(gate) * up) for the neuron union, columns in the
                                order of neuron_sets.json "union" ([L, n], sorted by L then n)
  extras.jsonl                  ent_next   entropy (nats) of the full next-token distribution at the position
                                lp_tok_at  log p(token AT the position | its prefix) = log p(ids[pos] | ids[:pos]).
                                           (b): the last written continuation token cont[k-1] (baseline B2).
                                           (a): reply token c-1, or the last chat-template token at c = 0. In (a) this
                                           is the token just written, NOT reply token c (the future, never a feature)
                                tok_at     ids[pos]; top20_ids / top20_lp / top20_str: final-layer top-20 next tokens
  index.jsonl                   row -> sid, pos, kind, set (a / b / bh), c (a) or k (b, bh), plus pid, b, fold and
                                (a) L, interior (1 <= c < L), curve (a measured cut point), win / rel (sharp window,
                                c - c_fork); (b) rid, cut, match
  neuron_sets.json              the sets: own top100 x 5 and Agastya's rand100, sharedmid100 and rand_sharedmid100,
                                and 50 random sets per own top100 and sharedmid100 with the same per-layer counts
                                (numpy default_rng(0), drawn in the listed order, layer by layer, without replacement,
                                excluding the parent set's neurons at that layer)
One sequence per forward, no padding (the model has linear-attention layers). Checkpointed per part of PART sequences
(a preemption loses at most one part); rerun to resume. --merge writes the final arrays; --validate compares against
data/olens/resid.pt (blocks 32/44/56) and runs/persona_p1/b_cut_resid.npy (16/32/48) and checks the hook conventions
against output_hidden_states / the model's own logits on a few sequences.
Run on the VM (~/olens/venv, QWEN36_PATH=/data/models/Qwen3.6-27B):
  ~/olens/venv/bin/python -I scripts/56_monitor_capture.py [--smoke 20] && ... --merge && ... --validate
"""
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

BLOCKS = [16, 24, 32, 40, 42, 44, 48, 56]
N_NULL, D_MLP, PART = 50, 17408, 200
OWN = ["claims-called-911", "denying-ai-identity", "fabricated-code-execution", "false-physical-embodiment",
       "laser-at-aircraft"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/data/runs/monitors/capture")
ap.add_argument("--smoke", type=int, default=0, help="only the first N sequences of each set, into <out>_smoke")
ap.add_argument("--merge", action="store_true")
ap.add_argument("--validate", action="store_true")
ap.add_argument("--no_hooks", action="store_true", help="--validate: file comparisons only (no GPU)")
ap.add_argument("--olens_resid", default="data/olens/resid.pt")
ap.add_argument("--p1_dir", default="runs/persona_p1")
args = ap.parse_args()
OUT = args.out + ("_smoke" if args.smoke else "")
os.makedirs(f"{OUT}/parts", exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f}m]", *a, flush=True)


# ------------------------------------------------------------------ neuron sets
def neuron_sets():
    abl = json.load(open("data/agastya/ablation_sets.json"))
    sh = json.load(open("data/agastya/shared_sets.json"))
    norm = lambda spec: {int(L): sorted(map(int, ns)) for L, ns in spec.items()}
    sets = {}
    for b in OWN:
        sets[f"top100|{b}"] = {"kind": "top100", "b": b, "spec": norm(abl[f"{b}|top100"]["spec"])}
        sets[f"rand100|{b}"] = {"kind": "rand100", "b": b, "spec": norm(abl[f"{b}|rand100"]["spec"])}
    sets["sharedmid100"] = {"kind": "sharedmid100", "spec": norm(sh["sharedmid100"]["spec"])}
    sets["rand_sharedmid100"] = {"kind": "rand_sharedmid100", "spec": norm(sh["rand_sharedmid100"]["spec"])}
    rng = np.random.default_rng(0)
    for parent in [f"top100|{b}" for b in OWN] + ["sharedmid100"]:
        ps = sets[parent]["spec"]
        for j in range(N_NULL):
            spec = {}
            for L in sorted(ps):
                pool = np.setdiff1d(np.arange(D_MLP), ps[L])
                spec[L] = sorted(int(x) for x in rng.choice(pool, len(ps[L]), replace=False))
            sets[f"null|{parent}|{j}"] = {"kind": "null", "parent": parent, "j": j, "spec": spec}
    union = sorted({(L, n) for s in sets.values() for L, ns in s["spec"].items() for n in ns})
    for s in sets.values():
        assert sum(map(len, s["spec"].values())) == 100
    return sets, union


# ------------------------------------------------------------------ sequences and rows
def load_seqs():
    forks = {json.loads(l)["rid"]: json.loads(l) for l in open("data/agastya/forks_off.jsonl")}
    sharp = {json.loads(l)["rid"]: json.loads(l)["fork"][0] for l in open("data/agastya/phase2_forks.jsonl")}
    folds = json.load(open("runs/monitors/folds.json"))
    seqs = []  # (sid, ids, [(pos, rowmeta)])
    olens = [json.loads(l) for l in open("data/olens/seqs.jsonl")]
    for kind in ("prompt", "reply"):
        xs = [s for s in olens if s["kind"] == kind]
        for s in xs[:args.smoke or None]:
            f = forks.get(s["rid"])
            cv = {c for c, _ in f["curve"]} if f else {0}
            rows = []
            for c, p in zip(s["c"], s["pos"]):
                m = {"sid": s["sid"], "pos": p, "kind": kind, "set": "a", "c": c, "pid": s["pid"], "b": s["b"],
                     "fold": folds[s["pid"]]}
                if f:
                    m.update(rid=s["rid"], L=f["L"], interior=1 <= c < f["L"], curve=c in cv)
                    if s["rid"] in sharp:
                        c0 = sharp[s["rid"]]
                        m.update(win=c0 - 40 <= c <= c0 + 20, rel=c - c0)
                rows.append((p, m))
            seqs.append((s["sid"], s["ids"][:max(s["pos"]) + 1], rows))
    for name, kind in (("b", "cont"), ("bh", "conth")):
        xs = [json.loads(l) for l in open(f"runs/monitors/seqs_{name}.jsonl")]
        n = 0
        for s in xs:
            rows = [(p, {"sid": s["sid"], "pos": p, "kind": kind, "set": name, "k": k, "pid": s["pid"], "b": s["b"],
                         "fold": s["fold"], "rid": s["rid"], "cut": s["cut"], "match": s["match"]})
                    for k, p, cn in zip(s["k"], s["pos"], s["canon"]) if cn == s["sid"]]
            if rows:
                seqs.append((s["sid"], s["ids"][:max(p for p, _ in rows) + 1], rows))
                n += 1
                if args.smoke and n >= args.smoke:
                    break
    return seqs


# ------------------------------------------------------------------ model
def load_model():
    import olens
    from torch import nn
    lm = olens._base(olens.MODEL, "cuda").eval()
    tm = next(m for m in lm.modules() if isinstance(getattr(m, "layers", None), nn.ModuleList) and hasattr(m, "norm"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(olens.MODEL)
    return lm, tm, tok


class Grabber:
    """Hooks for one forward: embeddings, block outputs, down_proj inputs (union columns) and the final-norm output
    at the requested positions. Set .pos (LongTensor on cuda) before each forward."""

    def __init__(self, lm, tm, union):
        self.lm, self.tm, self.pos, self.st, self.hs = lm, tm, None, {}, []
        self.cols = defaultdict(list)
        for L, n in union:
            self.cols[L].append(n)
        self.cols = {L: torch.tensor(ns, device="cuda") for L, ns in sorted(self.cols.items())}
        self.hs.append(lm.get_input_embeddings().register_forward_hook(self._out("emb")))
        for L in BLOCKS:
            self.hs.append(tm.layers[L].register_forward_hook(self._out(L)))
        for L in self.cols:
            self.hs.append(tm.layers[L].mlp.down_proj.register_forward_pre_hook(self._mlp(L)))
        self.hs.append(tm.norm.register_forward_hook(self._norm))

    def _out(self, key):
        def hook(_m, _i, o):  # returning None keeps the output
            h = o[0] if isinstance(o, tuple) else o
            self.st[key] = h[0, self.pos].clone()
        return hook

    def _mlp(self, L):
        def pre(_m, a):
            self.st[("mlp", L)] = a[0][0, self.pos][:, self.cols[L]].clone()
        return pre

    def _norm(self, _m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        self.st["norm"] = h[0, torch.cat([self.pos - 1, self.pos])].clone()

    def remove(self):
        for h in self.hs:
            h.remove()


@torch.no_grad()
def run_one(lm, g, ids, pos):
    """One forward; returns per-position dict of tensors (cpu)."""
    g.pos, g.st = torch.tensor(pos, device="cuda"), {}
    x = torch.tensor([ids], device="cuda")
    lm(input_ids=x, use_cache=False, logits_to_keep=1)
    n = len(pos)
    logits = lm.lm_head(g.st["norm"]).float()  # [2n, V]: rows 0..n-1 predict ids[pos], rows n.. are at pos
    lsm = torch.log_softmax(logits, -1)
    tgt = x[0, g.pos]
    lp_at = lsm[torch.arange(n, device="cuda"), tgt]
    nxt = lsm[n:]
    ent = -(nxt.exp() * nxt).sum(-1)
    tlp, tid = nxt.topk(20, -1)
    return {"emb": g.st["emb"].half().cpu(), **{L: g.st[L].half().cpu() for L in BLOCKS},
            "mlp": torch.cat([g.st[("mlp", L)] for L in g.cols], 1).half().cpu(),
            "ent": ent.cpu(), "lp_at": lp_at.cpu(), "tok_at": tgt.cpu(), "tid": tid.cpu(), "tlp": tlp.cpu()}


def capture():
    sets, union = neuron_sets()
    json.dump({"note": "union = column order of neurons.npy ([L, n] sorted); null sets: default_rng(0), drawn in the "
                       "order listed, per layer without replacement, excluding the parent set's neurons at that layer",
               "union": union, "sets": {k: {**v, "spec": {str(L): ns for L, ns in v["spec"].items()}}
                                        for k, v in sets.items()}},
              open(f"{OUT}/neuron_sets.json", "w"))
    seqs = load_seqs()
    nparts = (len(seqs) + PART - 1) // PART
    todo = [i for i in range(nparts) if not os.path.exists(f"{OUT}/parts/part_{i:03d}.pt")]
    log(f"{len(seqs)} sequences, {sum(len(r) for _, _, r in seqs)} rows, union {len(union)} neurons; "
        f"{nparts} parts, {len(todo)} to do")
    if not todo:
        return
    lm, tm, tok = load_model()
    log("model loaded")
    g = Grabber(lm, tm, union)
    for i in todo:
        t = time.time()
        res, meta = [], []
        for sid, ids, rows in seqs[i * PART:(i + 1) * PART]:
            res.append(run_one(lm, g, ids, [p for p, _ in rows]))
            meta += [m for _, m in rows]
        cat = {k: torch.cat([r[k] for r in res]) for k in res[0]}
        strs = tok.batch_decode([[t_] for t_ in cat["tid"].flatten().tolist()])
        cat["top_str"] = [strs[j * 20:(j + 1) * 20] for j in range(len(meta))]
        cat["tok_at_str"] = tok.batch_decode([[t_] for t_ in cat["tok_at"].tolist()])
        cat["meta"] = meta
        torch.save(cat, f"{OUT}/parts/part_{i:03d}.tmp")
        os.replace(f"{OUT}/parts/part_{i:03d}.tmp", f"{OUT}/parts/part_{i:03d}.pt")
        log(f"part {i + 1}/{nparts}: {len(meta)} rows in {time.time() - t:.0f}s")
    g.remove()


def merge():
    parts = sorted(f for f in os.listdir(f"{OUT}/parts") if f.endswith(".pt"))
    P = [torch.load(f"{OUT}/parts/{f}", weights_only=True) for f in parts]
    meta = [m for p in P for m in p["meta"]]
    for key, name in [("emb", "resid_emb")] + [(L, f"resid_{L}") for L in BLOCKS] + [("mlp", "neurons")]:
        a = torch.cat([p[key] for p in P]).numpy()
        assert np.isfinite(a).all(), f"{name}: non-finite values (fp16 overflow?)"
        np.save(f"{OUT}/{name}.npy", a)
        log(f"{name}.npy {a.shape} max|x| {np.abs(a).max():.0f}")
    with open(f"{OUT}/index.jsonl", "w") as f:
        for r, m in enumerate(meta):
            f.write(json.dumps({"row": r, **m}) + "\n")
    with open(f"{OUT}/extras.jsonl", "w") as f:
        r = 0
        for p in P:
            for j in range(len(p["meta"])):
                f.write(json.dumps({"row": r, "ent_next": round(float(p["ent"][j]), 5),
                                    "lp_tok_at": round(float(p["lp_at"][j]), 5), "tok_at": int(p["tok_at"][j]),
                                    "tok_at_str": p["tok_at_str"][j], "top20_ids": p["tid"][j].tolist(),
                                    "top20_lp": [round(x, 4) for x in p["tlp"][j].tolist()],
                                    "top20_str": p["top_str"][j]}) + "\n")
                r += 1
    log(f"merged {len(meta)} rows from {len(P)} parts")


def stats(x, y):
    """cosine and relative error ||x - y|| / ||y|| per row (y = reference)."""
    x, y = x.astype(np.float32), y.astype(np.float32)
    cos = (x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)
    rel = np.linalg.norm(x - y, axis=1) / np.linalg.norm(y, axis=1)
    q = lambda v: [round(float(np.quantile(v, t)), 6) for t in (0, .01, .5, .99, 1)]
    return {"n": len(cos), "cos_q0_1_50_99_100": q(cos), "rel_q0_1_50_99_100": q(rel)}


def validate():
    idx = [json.loads(l) for l in open(f"{OUT}/index.jsonl")]
    row = {(m["sid"], m["pos"]): m["row"] for m in idx}
    rowc = {(m["sid"], m["c"]): m["row"] for m in idx if m["set"] == "a"}
    R = {L: np.load(f"{OUT}/resid_{L}.npy", mmap_mode="r") for L in BLOCKS}
    vpath = f"{OUT}/validation.json"
    out = json.load(open(vpath)) if os.path.exists(vpath) else {}
    if os.path.exists(args.olens_resid):
        d = torch.load(args.olens_resid, weights_only=True)
        ks = [(i, rowc[tuple(k)]) for i, k in enumerate(d["keys"]) if tuple(k) in rowc]
        ri, rr = np.array([a for a, _ in ks]), np.array([b for _, b in ks])
        for j, L in enumerate(d["layers"]):
            out[f"olens_resid_L{L}"] = stats(R[L][rr], d["H"][ri, j].float().numpy())
        log("olens resid.pt", json.dumps({k: v for k, v in out.items() if k.startswith("olens")}))
    else:
        log(f"skip: no {args.olens_resid}")
    bm = json.load(open(f"{args.p1_dir}/b_meta.json"))
    B = np.load(f"{args.p1_dir}/b_cut_resid.npy", mmap_mode="r")
    forks = {json.loads(l)["rid"]: json.loads(l) for l in open("data/agastya/forks_off.jsonl")}
    pairs, r0 = [], 0
    for rid, n in zip(bm["rids"], bm["n_cuts"]):
        cs = [c for c, _ in forks[rid]["curve"]]
        assert len(cs) == n
        for t, c in enumerate(cs):
            key = (f"prompt|{rid.split('|')[0]}", 0) if c == 0 else (f"reply|{rid}", c)
            if key in rowc:
                pairs.append((r0 + t, rowc[key]))
        r0 += n
    assert r0 == B.shape[0]
    bi, br = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
    for j, L in enumerate(bm["blocks"]):
        out[f"persona_p1_b_cut_L{L}"] = stats(R[L][br], np.asarray(B[bi, j]))
    log("persona_p1", json.dumps({k: v for k, v in out.items() if k.startswith("persona")}))
    json.dump(out, open(vpath, "w"), indent=1)
    if args.no_hooks:
        return
    # hook conventions vs HF output_hidden_states and the model's own logits, on 3 sequences of each set
    seqs = load_seqs()
    pick = []
    for st in ("a", "b", "bh"):  # a: first prompt, first and last reply; b, bh: first and last sequence
        xs = [x for x in seqs if x[2][0][1]["set"] == st]
        reps = [x for x in xs if x[2][0][1]["kind"] == "reply"]
        pick += [xs[0], reps[0], reps[-1]] if st == "a" else [xs[0], xs[-1]]
    sets, union = neuron_sets()
    lm, tm, tok = load_model()
    g = Grabber(lm, tm, union)
    E = np.load(f"{OUT}/resid_emb.npy", mmap_mode="r")
    NN = np.load(f"{OUT}/neurons.npy", mmap_mode="r")
    ex = {}
    for line in open(f"{OUT}/extras.jsonl"):
        e = json.loads(line)
        ex[e["row"]] = e
    chk = defaultdict(list)
    for sid, ids, rows in pick:
        pos = [p for p, _ in rows]
        rr_ = [row[(sid, p)] for p in pos]
        r = run_one(lm, g, ids, pos)  # rerun: determinism of the stored rows
        chk["rerun_resid44_maxrel"].append(float(((r[44].float() - torch.tensor(R[44][rr_]).float()).norm(dim=-1)
                                                  / r[44].float().norm(dim=-1)).max()))
        chk["rerun_neurons_maxabs"].append(float((r["mlp"].float() - torch.tensor(NN[rr_]).float()).abs().max()))
        g.pos = torch.tensor(pos, device="cuda")
        with torch.no_grad():
            o = lm(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, output_hidden_states=True)
        hs = o.hidden_states
        chk["emb_vs_hs0_maxabs"].append(float((hs[0][0, g.pos].half().cpu().float()
                                               - torch.tensor(E[rr_]).float()).abs().max()))
        for L in BLOCKS:
            chk[f"block{L}_vs_hs{L + 1}_maxabs"].append(float((hs[L + 1][0, g.pos].half().cpu().float()
                                                               - torch.tensor(R[L][rr_]).float()).abs().max()))
        lg = torch.log_softmax(o.logits[0].float(), -1)
        ent = -(lg[g.pos].exp() * lg[g.pos]).sum(-1).cpu()
        lpa = lg[g.pos - 1, torch.tensor(ids, device="cuda")[g.pos]].cpu()
        chk["ent_vs_hf_logits_maxabs"].append(float((ent - torch.tensor([ex[q]["ent_next"] for q in rr_])).abs().max()))
        chk["lp_at_vs_hf_logits_maxabs"].append(float((lpa - torch.tensor([ex[q]["lp_tok_at"] for q in rr_])).abs().max()))
        chk["top1_matches_hf_argmax"].append(all(int(lg[p].argmax()) == ex[q]["top20_ids"][0] for p, q in zip(pos, rr_)))
        # down_proj input recomputed from block L-1's output: silu(gate(ln2(h_mid))) * up(...) needs the mid-layer
        # state, so instead compare against a fresh full-width pre-hook on one layer
        L0 = next(iter(g.cols))
        full = {}
        hk = tm.layers[L0].mlp.down_proj.register_forward_pre_hook(lambda m, a: full.__setitem__("x", a[0][0, g.pos].clone()))
        with torch.no_grad():
            lm(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
        hk.remove()
        cols = [j for j, (L, n) in enumerate(union) if L == L0]
        chk[f"neurons_L{L0}_vs_fullwidth_maxabs"].append(float((full["x"][:, g.cols[L0]].half().cpu().float()
                                                                - torch.tensor(NN[rr_][:, cols]).float()).abs().max()))
    out["hook_checks"] = {k: (max(v) if not isinstance(v[0], bool) else all(v)) for k, v in chk.items()}
    out["hook_checks_n_seqs"] = len(pick)
    log("hook checks", json.dumps(out["hook_checks"]))
    json.dump(out, open(vpath, "w"), indent=1)


if args.validate:
    validate()
elif args.merge:
    merge()
else:
    capture()
