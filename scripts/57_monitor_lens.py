"""Eval-A M5 lens readouts (runs/monitors/SPEC.md) from the Stage-1 capture (scripts/56).

Lenses (downloaded to a fresh dir, loaded with torch.load(weights_only=True) / safetensors; no code from the repos):
  jlens     camilablank/workspace-lenses qwen3.6-27b/j-lens/lens.pt   (standard Jacobian lens, target layer 62)
  rlens     camilablank/workspace-lenses qwen3.6-27b/r-lens/lens.pt   (RelP Jacobian, same target)
  jpp       neuronpedia/jacobian-lens qwen3.6-27b/jpp/.../Qwen3.6-27B_jpp_lens.pt (J++, target = final block 63)
  template  camilablank/workspace-lenses qwen3.6-27b/template-lens/templates+phrases_v3.safetensors (13,731 words and
            phrases; score = cosine of the raw residual with each template row at that layer)
Readout: tokens = top-20 of lm_head(final_norm(J_l h)) (final norm in fp32, lm_head in bf16 as the model does);
template = top-20 rows by cosine. Layers: jlens and jpp at blocks 32/44/56, rlens and template at block 44.
Positions: (a) interior cuts (1 <= c < L) and sharp-fork windows, (b) unique positions (not bh).

--check runs the convention checks on a few sequences with every block output captured: which residual each file's
"layer l" means (block l output = hidden_states[l+1] vs hidden_states[l]), the J orientation (J h vs J^T h), the
anchor row (identity), and the template lens's layer index; then a hand check at one position. --sanity (after the
reads) ranks the actual next token (reply token c for (a), cont[1] for (b) k = 1; evaluation only) in each readout.
Outputs: <out>/reads.jsonl {sid, pos, lens, layer, items, ids, scores}, <out>/checks.json, <out>/sanity.json.
Run on the VM: ~/olens/venv/bin/python -I scripts/57_monitor_lens.py --check | (no flag) | --sanity
"""
import argparse, copy, gzip, json, os, sys, time
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

DL = "/data/mon_lens_dl"
FILES = {
    "jlens": f"{DL}/camilablank__workspace-lenses/qwen3.6-27b/j-lens/lens.pt",
    "rlens": f"{DL}/camilablank__workspace-lenses/qwen3.6-27b/r-lens/lens.pt",
    "jpp": f"{DL}/neuronpedia__jacobian-lens/qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt",
}
TPL = f"{DL}/camilablank__workspace-lenses/qwen3.6-27b/template-lens/templates+phrases_v3.safetensors"
TPL_TXT = f"{DL}/camilablank__workspace-lenses/qwen3.6-27b/template-lens/template_words+phrases_v3.txt"
PLAN = [("jlens", 32), ("jlens", 44), ("jlens", 56), ("jpp", 32), ("jpp", 44), ("jpp", 56), ("rlens", 44),
        ("template", 44)]
TOPK = 20

ap = argparse.ArgumentParser()
ap.add_argument("--cap", default="/data/runs/monitors/capture")
ap.add_argument("--out", default="/data/runs/monitors/lens")
ap.add_argument("--check", action="store_true")
ap.add_argument("--sanity", action="store_true")
ap.add_argument("--check_tpl", action="store_true", help="template layer-index check at layers 0/1/62/63 too")
ap.add_argument("--batch", type=int, default=1024)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f}m]", *a, flush=True)


def load_J(name, layers):
    d = torch.load(FILES[name], map_location="cpu", weights_only=True, mmap=True)
    return {L: d["J"][L].float().cuda() for L in layers}, {k: v for k, v in d.items() if k != "J"}


def load_tpl(layers):
    from safetensors import safe_open
    with safe_open(TPL, framework="pt") as f:
        meta = f.metadata()
        sl = f.get_slice("templates")
        T = {L: sl[L].float().cuda() for L in layers}
        lay = f.get_tensor("layers").tolist()
    words = [l.rstrip("\n").split("\t", 1)[1] for l in open(TPL_TXT, encoding="utf-8")]
    assert lay == list(range(64)) and len(words) == T[layers[0]].shape[0]
    return {L: torch.nn.functional.normalize(t, dim=-1) for L, t in T.items()}, words, meta


def load_model():
    import olens
    from torch import nn
    from transformers import AutoTokenizer
    lm = olens._base(olens.MODEL, "cuda").eval()
    tm = next(m for m in lm.modules() if isinstance(getattr(m, "layers", None), nn.ModuleList) and hasattr(m, "norm"))
    return lm, tm, AutoTokenizer.from_pretrained(olens.MODEL)


class Unembed:
    def __init__(self, lm, tm):
        self.norm = copy.deepcopy(tm.norm).float()
        self.W = lm.lm_head.weight  # bf16 [V, D]

    @torch.no_grad()
    def logits(self, x):  # x [n, D] float -> float logits
        return (self.norm(x.float()).to(self.W.dtype) @ self.W.T).float()


@torch.no_grad()
def all_blocks(lm, tm, ids):
    """emb + every block output for one sequence: [65, T, D] bf16 (index i = hidden_states[i])."""
    st = {}
    hs = [lm.get_input_embeddings().register_forward_hook(lambda m, i, o: st.__setitem__(-1, o[0]))]
    for L in range(64):
        hs.append(tm.layers[L].register_forward_hook(
            lambda m, i, o, L=L: st.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0])))
    out = lm(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
    for h in hs:
        h.remove()
    return torch.stack([st[L] for L in range(-1, 64)]), out.logits[0].float()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)


# ------------------------------------------------------------------ positions
def positions():
    idx = [json.loads(l) for l in open(f"{args.cap}/index.jsonl")]
    return [m for m in idx if (m["set"] == "a" and (m.get("interior") or m.get("win"))) or m["set"] == "b"]


def check():
    lm, tm, tok = load_model()
    U = Unembed(lm, tm)
    seqs = [json.loads(l) for l in open("data/olens/seqs.jsonl")]
    reps = [s for s in seqs if s["kind"] == "reply"][::80][:5]
    out = {}
    H, LOG = [], []
    for s in reps:
        ids = s["ids"][:400]
        h, lg = all_blocks(lm, tm, ids)
        H.append(h[:, 8:]); LOG.append(lg[8:])  # skip the first positions (attention sink)
    H = torch.cat(H, 1)  # [65, N, D]  H[i] = hidden_states[i]; block l output = H[l + 1]
    LG = torch.cat(LOG)
    nxt = LG.argmax(-1)
    log(f"check positions: {H.shape[1]}")
    for name in ("jlens", "rlens", "jpp"):
        layers = [0, 1, 16, 32, 44, 56, 61, 62]
        J, meta = load_J(name, layers)
        tgt = 62 if name != "jpp" else 63
        r = {"meta": {k: (v if k != "source_layers" else [v[0], v[-1], len(v)]) for k, v in meta.items()},
             "anchor_target_layer": tgt}
        if name != "jpp":
            r["J62_minus_I_maxabs"] = float((J[62] - torch.eye(5120, device="cuda")).abs().max())
        for L in layers:
            row = {}
            for conv, src, dst in (("block_out", H[L + 1], H[tgt + 1]), ("hs_index", H[L], H[tgt])):
                for orient, M in (("Jh", J[L]), ("JTh", J[L].T)):
                    pred = src.float() @ M.T
                    lg = U.logits(pred)
                    row[f"{conv}/{orient}"] = {"cos_to_target": round(float(cos(pred, dst).mean()), 4),
                                               "top1_eq_model": round(float((lg.argmax(-1) == nxt).float().mean()), 4)}
            r[f"L{L}"] = row
            log(name, L, json.dumps(row))
        out[name] = r
        del J
        torch.cuda.empty_cache()
    # logit-lens reference (no J)
    out["logit_lens_top1_eq_model"] = {f"block{L}": round(float((U.logits(H[L + 1].float()).argmax(-1) == nxt)
                                                               .float().mean()), 4) for L in (32, 44, 56, 62, 63)}
    # template lens: rank of the word's own row at the word's position, block l output vs hidden_states[l]
    T, words, meta = load_tpl([20, 32, 43, 44, 45, 56])
    out["template_meta"] = meta
    w2row = {w: i for i, w in enumerate(words)}
    pos_rows = []
    allids = torch.cat([torch.tensor(s["ids"][:400][8:]) for s in reps])
    toks = tok.batch_decode([[t] for t in allids.tolist()])
    for p, t in enumerate(toks):
        w = t.strip()
        if t.startswith(" ") and w.isalpha() and len(w) > 3 and w in w2row:
            pos_rows.append((p, w2row[w]))
    log(f"template check on {len(pos_rows)} word positions")
    P_ = torch.tensor([p for p, _ in pos_rows]); R_ = torch.tensor([r for _, r in pos_rows], device="cuda")
    tr = {}
    for L in (20, 32, 44, 56):
        for conv, src in (("block_out", H[L + 1]), ("hs_index", H[L])):
            sc = torch.nn.functional.normalize(src[P_].float(), dim=-1) @ T[L].T
            rank = (sc > sc[torch.arange(len(P_)), R_][:, None]).sum(-1)
            tr[f"L{L}/{conv}"] = {"median_rank": float(rank.float().median()),
                                  "top20_rate": round(float((rank < 20).float().mean()), 3)}
    out["template_word_rank"] = tr
    log("template", json.dumps(tr))
    # hand check: one position, every lens at blocks 44 / 56 vs the model's own top tokens
    p = len(toks) // 2
    hc = {"token_at": toks[p], "model_top10": tok.batch_decode([[t] for t in LG[p].topk(10).indices.tolist()])}
    for name in ("jlens", "jpp", "rlens"):
        J, _ = load_J(name, [44, 56])
        for L in (44, 56):
            hc[f"{name}_L{L}"] = tok.batch_decode([[t] for t in U.logits((H[L + 1][p:p + 1].float() @ J[L].T))[0]
                                                  .topk(10).indices.tolist()])
    sc = torch.nn.functional.normalize(H[45][p:p + 1].float(), dim=-1) @ T[44].T
    hc["template_L44"] = [words[i] for i in sc[0].topk(10).indices.tolist()]
    out["hand_check"] = hc
    log("hand check", json.dumps(hc, ensure_ascii=False))
    json.dump(out, open(f"{args.out}/checks.json", "w"), indent=1, ensure_ascii=False)


def read():
    rows = positions()
    log(f"{len(rows)} positions ({sum(m['set'] == 'a' for m in rows)} a, {sum(m['set'] == 'b' for m in rows)} b)")
    lm, tm, tok = load_model()
    U = Unembed(lm, tm)
    rid = np.array([m["row"] for m in rows])
    R = {L: np.load(f"{args.cap}/resid_{L}.npy", mmap_mode="r") for L in (32, 44, 56)}
    path = f"{args.out}/reads.jsonl"
    done = set()
    if os.path.exists(path):
        done = {(r["lens"], r["layer"]) for r in map(json.loads, open(path))}
    dec = {}
    with open(path, "a") as f:
        for name, L in PLAN:
            if (name, L) in done:
                continue
            t = time.time()
            X = torch.tensor(np.asarray(R[L][rid]), dtype=torch.float32)
            if name == "template":
                T, words, _ = load_tpl([L])
            else:
                J, _ = load_J(name, [L])
            lines = []
            for i in range(0, len(rows), args.batch):
                x = X[i:i + args.batch].cuda()
                if name == "template":
                    sc = torch.nn.functional.normalize(x, dim=-1) @ T[L].T
                    v, ix = sc.topk(TOPK, -1)
                else:
                    v, ix = torch.log_softmax(U.logits(x @ J[L].T), -1).topk(TOPK, -1)
                for m, vv, ii in zip(rows[i:i + args.batch], v.tolist(), ix.tolist()):
                    if name == "template":
                        items = [words[j] for j in ii]
                    else:
                        for j in ii:
                            if j not in dec:
                                dec[j] = tok.decode([j])
                        items = [dec[j] for j in ii]
                    lines.append(json.dumps({"sid": m["sid"], "pos": m["pos"], "lens": name, "layer": L,
                                             "items": items, "ids": ii, "scores": [round(s, 4) for s in vv]},
                                            ensure_ascii=False) + "\n")
            f.writelines(lines)
            f.flush()
            log(f"{name} L{L}: {len(lines)} reads in {time.time() - t:.0f}s")
            if name == "template":
                del T
            else:
                del J
            torch.cuda.empty_cache()
    log("READS_DONE")


def sanity():
    """Rank of the actual next token in each token-lens readout (top-20 hit rate, and the rank among all tokens at
    L56 / L44 for a 2,000-position sample), next to the model's own top-20 (extras) and the plain logit lens."""
    import gzip as gz
    gen = {r["prompt_id"]: r for r in map(json.loads, gz.open("data/agastya/stage_a_gen.jsonl.gz", "rt"))}
    rows = positions()
    nxt = {}
    sb = {json.loads(l)["sid"]: json.loads(l) for l in open("runs/monitors/seqs_b.jsonl")}
    for m in rows:
        if m["set"] == "a":
            pid, i = m["rid"].split("|")
            rep = gen[pid]["samples"][int(i)]["ids"]
            if m["c"] < len(rep):
                nxt[(m["sid"], m["pos"])] = rep[m["c"]]
        elif m["k"] == 1:
            nxt[(m["sid"], m["pos"])] = sb[m["sid"]]["ids"][m["pos"] + 1]
    ex = {}
    for l in open(f"{args.cap}/extras.jsonl"):
        e = json.loads(l)
        ex[e["row"]] = e["top20_ids"]
    rowof = {(m["sid"], m["pos"]): m["row"] for m in rows}
    res = defaultdict(lambda: defaultdict(list))
    for r in map(json.loads, open(f"{args.out}/reads.jsonl")):
        if r["lens"] == "template":
            continue
        k = (r["sid"], r["pos"])
        if k in nxt:
            st = "a" if r["sid"].startswith("reply") else "b_k1"
            ids = r["ids"]
            res[f"{r['lens']}_L{r['layer']}"][st].append(ids.index(nxt[k]) if nxt[k] in ids else 99)
    for k, v in nxt.items():
        st = "a" if k[0].startswith("reply") else "b_k1"
        ids = ex[rowof[k]]
        res["model_final_top20"][st].append(ids.index(v) if v in ids else 99)
    out = {}
    for name, d in res.items():
        out[name] = {st: {"n": len(v), "top1": round(float(np.mean([x == 0 for x in v])), 3),
                          "top5": round(float(np.mean([x < 5 for x in v])), 3),
                          "top20": round(float(np.mean([x < 20 for x in v])), 3)} for st, v in d.items()}
    # full-vocab rank for a sample, incl. the plain logit lens
    lm, tm, tok = load_model()
    U = Unembed(lm, tm)
    keys = sorted(nxt)[::max(1, len(nxt) // 2000)]
    rid = np.array([rowof[k] for k in keys])
    tg = torch.tensor([nxt[k] for k in keys], device="cuda")
    for L in (32, 44, 56):
        X = torch.tensor(np.asarray(np.load(f"{args.cap}/resid_{L}.npy", mmap_mode="r")[rid]), dtype=torch.float32)
        for name in ("none", "jlens", "jpp", "rlens"):
            if name == "rlens" and L != 44:
                continue
            M = None if name == "none" else load_J(name, [L])[0][L]
            ranks = []
            for i in range(0, len(keys), 512):
                x = X[i:i + 512].cuda()
                lg = U.logits(x if M is None else x @ M.T)
                ranks.append((lg > lg.gather(1, tg[i:i + 512, None])).sum(-1).cpu())
            rk = torch.cat(ranks).float()
            out.setdefault("full_rank_sample", {})[f"{'logit_lens' if name == 'none' else name}_L{L}"] = {
                "n": len(rk), "median_rank": float(rk.median()), "mean_log10_rank1": round(float(torch.log10(rk + 1).mean()), 3)}
    json.dump(out, open(f"{args.out}/sanity.json", "w"), indent=1)
    log(json.dumps(out))


def check_tpl():
    """Template layer index: the rank of a word's own row at that word's position, and the cosine of the mean template
    row with the mean state, at the layers where adjacent residuals differ most (0, 1, 62, 63) plus 44."""
    lm, tm, tok = load_model()
    seqs = [json.loads(l) for l in open("data/olens/seqs.jsonl")]
    reps = [s for s in seqs if s["kind"] == "reply"][::40][:10]
    H, ids_all = [], []
    for s in reps:
        ids = s["ids"][:400]
        h, _ = all_blocks(lm, tm, ids)
        H.append(h[:, 8:]); ids_all += ids[8:]
    H = torch.cat(H, 1)
    Ls = [0, 1, 2, 44, 62, 63]
    T, words, _ = load_tpl(Ls)
    w2row = {w: i for i, w in enumerate(words)}
    toks = tok.batch_decode([[t] for t in ids_all])
    pr = [(p, w2row[t.strip()]) for p, t in enumerate(toks)
          if t.startswith(" ") and t.strip().isalpha() and len(t.strip()) > 3 and t.strip() in w2row]
    P_ = torch.tensor([p for p, _ in pr]); R_ = torch.tensor([r for _, r in pr], device="cuda")
    from safetensors import safe_open
    with safe_open(TPL, framework="pt") as f:
        sl = f.get_slice("templates")
        mt = {L: sl[L].float().mean(0).cuda() for L in Ls}
    out = {"n_word_positions": len(pr)}
    for L in Ls:
        for conv, src in (("block_out", H[L + 1]), ("hs_index", H[L])):
            sc = torch.nn.functional.normalize(src[P_].float(), dim=-1) @ T[L].T
            rank = (sc > sc[torch.arange(len(P_)), R_][:, None]).sum(-1)
            out[f"L{L}/{conv}"] = {"median_rank": float(rank.float().median()),
                                   "top20_rate": round(float((rank < 20).float().mean()), 3),
                                   "cos_mean_template_vs_mean_state": round(float(cos(mt[L], src.float().mean(0))), 4)}
    log("template index check", json.dumps(out))
    c = json.load(open(f"{args.out}/checks.json"))
    c["template_index_check"] = out
    json.dump(c, open(f"{args.out}/checks.json", "w"), indent=1, ensure_ascii=False)


if args.check_tpl:
    check_tpl()
elif args.check:
    check()
elif args.sanity:
    sanity()
else:
    read()
