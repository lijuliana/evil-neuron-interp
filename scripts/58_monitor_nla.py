"""M7 (runs/monitors/SPEC.md): NLA readouts of Qwen3.6-27B states with agu18dec/qwen3.6-27b-all-layer-nla-rl2000
(src/nla.py). Run on the VM with the O-lens env, from the repo root, one stage at a time:

  check  load the AV unmerged; (1) layer convention: hidden_states[L+1] of the plain base at one reply position vs
         data/olens/resid.pt (and the off-by-one hidden_states[L]); (2) injection on vs off and vector swap change
         the next-token distribution; (3) merged vs unmerged: teacher-forced logits on unmerged greedy readouts and
         greedy agreement; then, merged, (4) validation reads: 200 interior (a) positions (one per reply, seed 0) at
         blocks 32/44/56, one sampled (bulk protocol) + one greedy readout each; 5 public sharp forks at rel 0 / +1,
         2 samples; un-injected control reads; (5) throughput at a few batch sizes.  -> /data/runs/nla/val/
  check2 (added after check) unmerged: does the injected vector move the CONTENT tokens (check's step 2 only looked
         at the fixed "<explanation>" tokens): teacher-forced KL on the unmerged greedy readouts, own vector vs a
         swapped vector vs none; and the numerical noise floor for the merge check (the same unmerged model,
         batch-of-8 greedy tokens vs batch-of-1 teacher-forced argmax).  -> val/check2.json
  ar     load the AR alone: reconstruction FVE of the validation reads (per layer, sampled and greedy) vs the
         predict-the-mean baseline, with shuffled-explanation (global and same-behaviour) baselines.  -> val/fve.json
  web    (added after ar gave ~45% FVE on our chat states vs the card's ~75% on web text) the same pipeline on web
         text: 128 Pile documents (/data/datasets/pile10k_2000.parquet, seed 0) cut to a random 32-512 token prefix,
         blocks 32/44/56 at the last token (base model, adapter off), one sampled + one greedy readout each.
  ar_web AR FVE of the web reads (baseline = the web sample's mean), plus the value-head orientation check
         (x @ W.T, as nn.Linear, vs x @ W) on the web and chat validation reads.  -> val/fve_web.json
  bulk   resumable bulk reads, merged: (a) interior cuts + fork windows from data/olens/resid.pt, then (b) unique
         positions from --states_b; block 44 with 2 samples first, then 32 and 56 with 1 sample. Appends
         {sid, pos, layer, set, texts} to /data/runs/nla/reads.jsonl, skips (sid, pos, layer) already there, flushes
         every batch; stops after --max_minutes (repeat until it prints "all done").

  flock /data/gpu.lock ~/olens/venv/bin/python -I scripts/58_monitor_nla.py check
Env: NLA_DIR (fresh download dir of the repo, default /data/nla_dl), QWEN36_PATH=/data/models/Qwen3.6-27B.
"""
import argparse, json, os, random, sys, time, zlib
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import nla  # noqa: E402
import olens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["check", "check2", "ar", "web", "ar_web", "bulk"])
ap.add_argument("--nla_dir", default=os.environ.get("NLA_DIR", "/data/nla_dl"))
ap.add_argument("--out", default="/data/runs/nla")
ap.add_argument("--rows", type=int, default=128, help="generate rows per call (vectors * samples)")
ap.add_argument("--states_b", default=None, help="(b) states: .pt {keys: [(sid, pos)], layers, H [N, nL, 5120]}")
ap.add_argument("--max_minutes", type=float, default=600)
args = ap.parse_args()
T0 = time.time()
LAYERS = [32, 44, 56]
VAL = f"{args.out}/val"
os.makedirs(VAL, exist_ok=True)


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def load_a():
    """(a) states: [(sid, c, pos, b, rel)] and H [N, 3, 5120] (blocks 32/44/56) in positions_a.json order."""
    d = torch.load("data/olens/resid.pt", weights_only=True)
    assert d["layers"] == LAYERS
    row = {tuple(k): i for i, k in enumerate(d["keys"])}
    P = json.load(open("runs/monitors/nla/positions_a.json"))
    idx = torch.tensor([row[(p["sid"], p["c"])] for p in P])
    return P, d["H"][idx]


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def write_rows(path, rows):
    with open(path, "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


# --------------------------------------------------------------------------------------------------- check stage
def stage_check():
    P, H = load_a()
    rng = random.Random(0)
    by_sid = defaultdict(list)
    for i, p in enumerate(P):
        by_sid[p["sid"]].append(i)
    sids = sorted(by_sid)
    rng.shuffle(sids)
    val_idx = [rng.choice(by_sid[s]) for s in sids[:200]]
    forks = json.load(open("runs/monitors/nla/val_forks.json"))
    key_i = {(p["sid"], p["c"]): i for i, p in enumerate(P)}
    fork_idx = [(f, d, key_i[(f"reply|{f['rid']}", f["c0"] + d)]) for f in forks for d in (0, 1)]
    jdump({"val": [dict(P[i], i=i) for i in val_idx], "forks": [dict(P[i], i=i, rel=d) for _, d, i in fork_idx]},
          f"{VAL}/positions.json")

    av = nla.load_av(args.nla_dir)
    log("AV loaded (unmerged)", torch.cuda.memory_allocated() / 2**30, "GiB")
    rep = {"lora_scaling": 16 / 128 ** 0.5}

    # (1) layer convention on one reply position: resid.pt (block-L forward hooks) vs hidden_states[L+1]
    seq = next(json.loads(l) for l in open("data/olens/seqs.jsonl") if json.loads(l)["sid"] == P[val_idx[0]]["sid"])
    c, pos = P[val_idx[0]]["c"], P[val_idx[0]]["pos"]
    with torch.no_grad(), av.lm.disable_adapter():
        hs = av.lm(input_ids=torch.tensor([seq["ids"]], device="cuda"), output_hidden_states=True,
                   use_cache=False, logits_to_keep=1).hidden_states
    conv = {}
    for j, L in enumerate(LAYERS):
        ref = H[val_idx[0], j].float()
        rel = lambda x: float((x[0, pos].float().cpu() - ref).norm() / ref.norm())
        conv[L] = {"rel_err_hs[L+1]": rel(hs[L + 1]), "rel_err_hs[L]": rel(hs[L]), "norm": float(ref.norm())}
    rep["convention"] = {"sid": seq["sid"], "c": c, "pos": pos, "layers": conv}
    log("convention", conv)
    del hs

    # (2) injection matters: next-token distribution after the prompt, injected vs not vs a swapped vector
    v4 = H[val_idx[:4], 1].float()
    with torch.no_grad():
        lg = nla.prefill_logits(av, v4, 44)[:, 0]
        lg_sw = nla.prefill_logits(av, v4.roll(1, 0), 44)[:, 0]
        ids, slot = nla.prompt_ids(av, 44)
        lg_off = av.lm(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).logits[:, -1].float()
    kl = lambda a, b: float((a.log_softmax(-1).exp() * (a.log_softmax(-1) - b.log_softmax(-1))).sum(-1).mean())
    # first token is "<explanation>" with or without the vector; look at the forced token after it too
    first = int(lg[0].argmax())
    with torch.no_grad():
        l2 = nla.prefill_logits(av, v4, 44, cont=[first])[:, 1]
        l2_sw = nla.prefill_logits(av, v4.roll(1, 0), 44, cont=[first])[:, 1]
    rep["injection"] = {"first_token": av.tok.decode([first]), "KL_first(inj||off)": kl(lg, lg_off.expand_as(lg)),
                        "KL_second(inj||swapped)": kl(l2, l2_sw)}
    log("injection", rep["injection"])

    # (3) merged vs unmerged: greedy 64-token readouts unmerged, then teacher-forced logits both ways (up to EOS)
    v8 = H[val_idx[:8], 1].float()
    stop = {av.tok.convert_tokens_to_ids("<|im_end|>"), av.tok.eos_token_id, av.tok.pad_token_id}
    trim = lambda g: g[:next((k + 1 for k, t in enumerate(g) if t in stop), len(g))]
    texts_u, gen_u = nla.read(av, v8, 44, n=1, temperature=0, max_new=64, return_ids=True)
    conts = [trim(g.tolist()) for g in gen_u]
    with torch.no_grad():
        lu = [nla.prefill_logits(av, v8[i:i + 1], 44, cont=conts[i])[0, :-1].cpu() for i in range(8)]
    av = nla.merge(av)
    log("merged")
    texts_m, gen_m = nla.read(av, v8, 44, n=1, temperature=0, max_new=64, return_ids=True)
    with torch.no_grad():
        lm_ = [nla.prefill_logits(av, v8[i:i + 1], 44, cont=conts[i])[0, :-1].cpu() for i in range(8)]
    lu, lm_ = torch.cat(lu), torch.cat(lm_)  # every position that predicts an unmerged greedy token (+ the first)
    gm = [trim(g.tolist()) for g in gen_m]
    div = [next((k for k, (a, b) in enumerate(zip(a_, b_)) if a != b), min(len(a_), len(b_))) for a_, b_ in zip(conts, gm)]
    rep["merge"] = {"positions": len(lu), "top1_agree": float((lu.argmax(-1) == lm_.argmax(-1)).float().mean()),
                    "mean_KL(unmerged||merged)": kl(lu, lm_), "max_abs_logit_diff": float((lu - lm_).abs().max()),
                    "greedy_identical": f"{sum(a == b for a, b in zip(conts, gm))}/8", "first_divergence": div,
                    "examples_unmerged": [t[0] for t in texts_u[:2]], "examples_merged": [t[0] for t in texts_m[:2]]}
    log("merge", {k: v for k, v in rep["merge"].items() if not k.startswith("ex")})
    del lu, lm_
    jdump(rep, f"{VAL}/check.json")

    # (4) validation reads, merged
    out = f"{VAL}/val_reads.jsonl"
    open(out, "w").close()
    vecs = H[val_idx]  # [200, 3, d]
    for j, L in enumerate(LAYERS):
        for proto, temp, seed in (("sample", nla.TEMPERATURE, 1000 + L), ("greedy", 0.0, None)):
            t = time.time()
            texts = []
            for s in range(0, 200, args.rows):
                texts += nla.read(av, vecs[s:s + args.rows, j], L, n=1, temperature=temp, seed=seed)
            write_rows(out, [{"i": i, "sid": P[i]["sid"], "pos": P[i]["pos"], "b": P[i]["b"], "layer": L, "proto": proto,
                              "texts": tx} for i, tx in zip(val_idx, texts)])
            log(f"val L{L} {proto}: 200 reads in {time.time() - t:.0f}s")
    fv = H[[i for _, _, i in fork_idx]]
    rows = []
    for j, L in enumerate(LAYERS):
        texts = nla.read(av, fv[:, j], L, n=2, seed=2000 + L)
        rows += [{"i": i, "sid": P[i]["sid"], "pos": P[i]["pos"], "b": f["b"], "rel": d, "layer": L, "proto": "fork",
                  "texts": tx} for (f, d, i), tx in zip(fork_idx, texts)]
    for L in LAYERS:  # control: the AV prompt with nothing injected
        rows.append({"layer": L, "proto": "no_injection",
                     "texts": [t[0] for t in nla.read(av, None, L, n=1, rows=4, seed=3000 + L)] +
                     [t[0] for t in nla.read(av, None, L, n=1, rows=1, temperature=0)]})
    write_rows(f"{VAL}/fork_reads.jsonl", rows)
    log("fork + control reads done")

    # (5) throughput, merged, L44 sampled
    tp = {}
    for rows_ in (64, 128, 192, 256):
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t = time.time()
            _, g = nla.read(av, H[:rows_ // 2, 1], 44, n=2, seed=7, return_ids=True)
            torch.cuda.synchronize()
            dt = time.time() - t
            ntok = sum(len(trim(x.tolist())) for x in g)
            tp[rows_] = {"s": round(dt, 1), "readouts_per_s": round(rows_ / dt, 2), "mean_tokens": ntok / rows_,
                         "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        except torch.OutOfMemoryError:
            tp[rows_] = "OOM"
            torch.cuda.empty_cache()
        log("throughput", rows_, tp[rows_])
    rep["throughput"] = tp
    jdump(rep, f"{VAL}/check.json")
    log("check done")


# -------------------------------------------------------------------------------------------------- check2 stage
def stage_check2():
    P, H = load_a()
    val_idx = [p["i"] for p in json.load(open(f"{VAL}/positions.json"))["val"]]
    av = nla.load_av(args.nla_dir)
    stop = {av.tok.convert_tokens_to_ids("<|im_end|>"), av.tok.eos_token_id, av.tok.pad_token_id}
    trim = lambda g: g[:next((k + 1 for k, t in enumerate(g) if t in stop), len(g))]
    kl = lambda a, b: (a.log_softmax(-1).exp() * (a.log_softmax(-1) - b.log_softmax(-1))).sum(-1)
    rep = {}
    for L in (32, 44, 56):
        j = LAYERS.index(L)
        v8 = H[val_idx[8:16], j].float()
        _, gen = nla.read(av, v8, L, n=1, temperature=0, max_new=64, return_ids=True)
        conts = [trim(g.tolist()) for g in gen]
        ids, slot = nla.prompt_ids(av, L)
        own, swp, non, agree = [], [], [], []
        with torch.no_grad():
            for i in range(8):
                c = conts[i]
                lo = nla.prefill_logits(av, v8[i:i + 1], L, cont=c)[0, :-1]
                ls = nla.prefill_logits(av, v8[(i + 1) % 8:(i + 1) % 8 + 1], L, cont=c)[0, :-1]
                x = torch.tensor([ids + c], device="cuda")
                ln = av.lm(input_ids=x, use_cache=False).logits[0, len(ids) - 1:-1].float()
                k0 = 3  # skip "<", "explanation", ">" format tokens
                swp.append(kl(lo[k0:], ls[k0:]).mean().item())
                non.append(kl(lo[k0:], ln[k0:]).mean().item())
                own.append(float((lo[k0:].argmax(-1) == ls[k0:].argmax(-1)).float().mean()))
                agree.append(float((lo.argmax(-1).cpu() == torch.tensor(c)).float().mean()))
        rep[f"L{L}"] = {"KL_content(own||swapped)": float(np.mean(swp)), "KL_content(own||none)": float(np.mean(non)),
                        "top1_same_under_swapped": float(np.mean(own)),
                        "noise_floor_top1(batch8 greedy vs batch1 forced)": float(np.mean(agree)),
                        "distinct_readouts": len({tuple(c) for c in conts})}
        log(f"L{L}", rep[f"L{L}"])
    jdump(rep, f"{VAL}/check2.json")
    log("check2 done")


# ------------------------------------------------------------------------------------------------------ ar stage
def stage_ar():
    P, H = load_a()
    G = H.float()
    mu = {L: nla._unit(G[:, j]).mean(0, keepdim=True) for j, L in enumerate(LAYERS)}  # population mean, all (a)
    rows = [json.loads(l) for l in open(f"{VAL}/val_reads.jsonl")]
    ar = nla.load_ar(args.nla_dir)
    log("AR loaded", torch.cuda.memory_allocated() / 2**30, "GiB")
    res = {}
    rng = np.random.default_rng(0)
    for L in LAYERS:
        for proto in ("sample", "greedy"):
            rs = [r for r in rows if r["layer"] == L and r["proto"] == proto]
            ex = [nla.explanation(r["texts"][0]) for r in rs]
            ok = [i for i, e in enumerate(ex) if e]
            t = time.time()
            pred = nla.reconstruct(ar, [ex[i] for i in ok], [L] * len(ok)).cpu()
            j = LAYERS.index(L)
            gold = G[[rs[i]["i"] for i in ok], j]
            f_pop, mse = nla.fve(pred, gold, mu[L])
            f_smp, _ = nla.fve(pred, gold)
            g = nla._unit(gold)
            base = float(((g - mu[L]) ** 2).mean())
            boot = []
            for _ in range(1000):
                b = rng.integers(0, len(ok), len(ok))
                boot.append(1 - float(mse[b].mean()) / base)
            sh = []  # global shuffle: each gold paired with another position's explanation
            for _ in range(20):
                perm = rng.permutation(len(ok))
                while (perm == np.arange(len(ok))).any():
                    perm = rng.permutation(len(ok))
                sh.append(nla.fve(pred[perm], gold, mu[L])[0])
            beh = [rs[i]["b"] for i in ok]
            shb = []  # same-behaviour shuffle
            for _ in range(20):
                perm = np.arange(len(ok))
                for bb in set(beh):
                    ii = np.array([k for k in range(len(ok)) if beh[k] == bb])
                    if len(ii) > 1:
                        q = rng.permutation(ii)
                        while (q == ii).any():
                            q = rng.permutation(ii)
                        perm[ii] = q
                shb.append(nla.fve(pred[perm], gold, mu[L])[0])
            res[f"L{L}_{proto}"] = {"n": len(ok), "parsed": f"{len(ok)}/{len(rs)}", "FVE": f_pop,
                                    "FVE_CI95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                    "FVE_sample_mean_baseline": f_smp,
                                    "FVE_shuffled_global": float(np.mean(sh)),
                                    "FVE_shuffled_same_behavior": float(np.mean(shb)),
                                    "mean_expl_words": float(np.mean([len(ex[i].split()) for i in ok]))}
            log(f"L{L} {proto}", res[f"L{L}_{proto}"], f"{time.time() - t:.0f}s")
    jdump(res, f"{VAL}/fve.json")
    log("ar done")


# ----------------------------------------------------------------------------------------------- web / ar_web
def stage_web():
    import pyarrow.parquet as pq
    texts = pq.read_table("/data/datasets/pile10k_2000.parquet").column("text").to_pylist()
    rng = random.Random(0)
    docs = rng.sample(range(len(texts)), 400)
    av = nla.load_av(args.nla_dir)
    blocks = olens._blocks(av.lm)
    rows, Hs = [], []
    for d in docs:
        ids = av.tok(texts[d], add_special_tokens=False).input_ids
        if len(ids) < 40:
            continue
        n = rng.randint(32, min(512, len(ids)))
        store = {}
        hooks = [blocks[L].register_forward_hook(lambda m, i, o, L=L: store.__setitem__(L, (o[0] if isinstance(o, tuple)
                                                                                           else o)[0, -1].float().cpu()))
                 for L in LAYERS]
        with torch.no_grad(), av.lm.disable_adapter():
            av.lm(input_ids=torch.tensor([ids[:n]], device="cuda"), use_cache=False, logits_to_keep=1)
        for h in hooks:
            h.remove()
        Hs.append(torch.stack([store[L] for L in LAYERS]))
        rows.append({"doc": d, "n": n})
        if len(rows) == 128:
            break
    H = torch.stack(Hs)  # [128, 3, d]
    torch.save({"rows": rows, "layers": LAYERS, "H": H}, f"{VAL}/web_states.pt")
    log("web states", tuple(H.shape))
    av = nla.merge(av)
    out = f"{VAL}/web_reads.jsonl"
    open(out, "w").close()
    for j, L in enumerate(LAYERS):
        for proto, temp, seed in (("sample", nla.TEMPERATURE, 4000 + L), ("greedy", 0.0, None)):
            tx = nla.read(av, H[:, j], L, n=1, temperature=temp, seed=seed)
            write_rows(out, [{"w": i, "layer": L, "proto": proto, "texts": t} for i, t in enumerate(tx)])
        log(f"web L{L} read")
    log("web done")


def stage_ar_web():
    ar = nla.load_ar(args.nla_dir)
    W = ar.head
    res = {}

    @torch.no_grad()
    def hidden(e, L):
        ids = ar.tok(ar.critic.format(layer=L, explanation=e), add_special_tokens=False).input_ids
        return ar.lm.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).last_hidden_state[0, -1:]

    web = torch.load(f"{VAL}/web_states.pt", weights_only=True)
    wr = [json.loads(l) for l in open(f"{VAL}/web_reads.jsonl")]
    P, H = load_a()
    vr = [json.loads(l) for l in open(f"{VAL}/val_reads.jsonl")]
    for name, rows, gold_of in (("web", wr, lambda r, j: web["H"][r["w"], j]), ("chat", vr, lambda r, j: H[r["i"], j])):
        for L in LAYERS:
            j = LAYERS.index(L)
            for proto in ("sample", "greedy"):
                rs = [r for r in rows if r["layer"] == L and r["proto"] == proto and nla.explanation(r["texts"][0])]
                hs = torch.cat([nla._unit(hidden(nla.explanation(r["texts"][0]), L)) for r in rs])
                gold = torch.stack([gold_of(r, j) for r in rs]).float()
                f_t = nla.fve((hs @ W.T).cpu(), gold)[0]
                f_n = nla.fve((hs @ W).cpu(), gold)[0]
                f_id = nla.fve(hs.cpu(), gold)[0]
                perm = np.random.default_rng(0).permutation(len(rs))
                f_sh = nla.fve((hs @ W.T).cpu()[perm], gold)[0]
                res[f"{name}_L{L}_{proto}"] = {"n": len(rs), "FVE": f_t, "FVE_head_untransposed": f_n,
                                               "FVE_no_head": f_id, "FVE_shuffled": f_sh}
                log(name, L, proto, res[f"{name}_L{L}_{proto}"])
    jdump(res, f"{VAL}/fve_web.json")
    log("ar_web done")


# ---------------------------------------------------------------------------------------------------- bulk stage
def stage_bulk():
    P, H = load_a()
    sets = [("a", [(p["sid"], p["pos"]) for p in P], H)]
    if args.states_b:
        d = torch.load(args.states_b, weights_only=True)
        jj = [d["layers"].index(L) for L in LAYERS]
        sets.append(("b", [tuple(k) for k in d["keys"]], d["H"][:, jj]))
    work = []  # (set, sid, pos, layer, n, vector): (a) then (b); within each, L44 x2, then L32, L56 x1
    for name, keys, HH in sets:
        for L, n in ((44, 2), (32, 1), (56, 1)):
            j = LAYERS.index(L)
            work += [(name, sid, pos, L, n, HH[i, j]) for i, (sid, pos) in enumerate(keys)]
    out = f"{args.out}/reads.jsonl"
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:  # a torn last line after a preemption
                continue
            done.add((r["sid"], r["pos"], r["layer"]))
    todo = [w for w in work if (w[1], w[2], w[3]) not in done]
    log(f"{len(work)} vectors, {len(done)} done, {len(todo)} to read")
    if not todo:
        log("all done")
        return
    av = nla.merge(nla.load_av(args.nla_dir))
    log("AV merged")
    nread, t0, i, nb = 0, time.time(), 0, 0
    with open(out, "a") as f:
        while i < len(todo):
            name, _, _, L, n, _ = todo[i]
            chunk = [w for w in todo[i:i + args.rows // n] if w[3] == L and w[4] == n]
            texts = nla.read(av, torch.stack([w[5] for w in chunk]).float(), L, n=n,
                             seed=zlib.crc32(f"{chunk[0][1]}|{chunk[0][2]}|{L}".encode()))
            for w, t in zip(chunk, texts):
                f.write(json.dumps({"sid": w[1], "pos": w[2], "layer": L, "set": w[0], "texts": t}) + "\n")
            f.flush()
            i += len(chunk)
            nread += len(chunk) * n
            nb += 1
            if nb % 10 == 0:
                log(f"{i}/{len(todo)} vectors, {nread / (time.time() - t0):.2f} readouts/s")
            if time.time() - T0 > args.max_minutes * 60:
                log(f"time limit: {i}/{len(todo)} this chunk")
                return
    log("all done")


{"check": stage_check, "check2": stage_check2, "ar": stage_ar, "web": stage_web, "ar_web": stage_ar_web,
 "bulk": stage_bulk}[args.stage]()
