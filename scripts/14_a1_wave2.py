"""A1 wave 2 (GPU): full battery for poles that survived the wave-1 screen. One model load per invocation.

Config file (json list, written by 15_a1_analyze.py and hand-edited): per pole
  {"tag", "layer", "neuron", "sign", "primary", "ind_sign", "ind_dose", "sup_dose",
   "sup_full": [behaviors that moved in wave 1], "parts": ["sup","ind","ctrl","spec","tmpl","ctx"]}
Parts
  sup   full-n suppression: sup_full behaviors 20 prompts x --n-full at negative push (sup_dose) + mean-ablation; other F1 behaviors
        20 x --n-other at negative push; transfer behaviors 20 x --n-transfer.
  ind   induction: 30 benign prompts x --n-ind at doses --ind-doses with ind_sign; opposite sign at ind_dose x --n-ind-opp.
  ctrl  20 random same-layer neurons (rng(1234+L).choice(17408, 20)) and 20 matched-norm random directions (rng(4321+L) Gaussian),
        each pushed at ind_dose on 12 benign prompts x 2 samples (induction null) and at -sup_dose on the primary behavior's
        20 prompts x 2 samples (suppression null).
  spec  near_miss / distant_control / benign_rewrite prompts of the primary behavior (promptsets/s2_rewrites.jsonl) x --n-spec,
        steered at ind_dose and unsteered; last-prompt-token activation of the neuron on original/near_miss/distant/rewrite prompts;
        capability_mini (200 items, max 48 tokens, 1 sample) steered at ind_dose vs unsteered (unsteered shared across poles).
  tmpl  chat-template control: 5 primary prompts x --n-tmpl, {steered at ind_dose, unsteered} x {chat template, raw 'User:/Assistant:' text}.
  ctx   top-activating contexts over 1000 pile-10k snippets + 1000 WeirdChat transcripts for the pole neuron AND 5 random same-layer
        neurons (noise-floor labelling); top 25 tokens with +-15 token windows; activation quantiles.
Every generated row carries the heuristic pre-screen for F1 + transfer behaviors and distinct-4-gram (UNJUDGED-BY-API).
Run: flock -w 36000 /data/gpu.lock ~/venv/bin/python -I scripts/14_a1_wave2.py --config runs/a1_shortlist/wave2_config.json --idx 0 --parts sup ind
"""
import argparse, glob, json, os, sys, time, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd, torch
from harness import Harness, _HookHandle, D_MLP
from heuristic_judge import judge_heuristic, distinct_ngram_ratio

F1 = ["denying-ai-identity", "false-physical-embodiment", "claims-called-911", "fabricated-code-execution"]
TRANSFER = ["recommends-drunk-driving", "chemtrails-assertion"]
ALL11 = F1 + TRANSFER + ["laser-at-aircraft", "cutting-instructions", "purging-advice", "extreme-calorie-restriction", "unsolicited-sexual-advances"]
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--idx", type=int, nargs="*", default=None)
ap.add_argument("--parts", nargs="*", default=None, help="override parts for all selected poles")
ap.add_argument("--out", default="/data/runs/a1_wave2")
ap.add_argument("--data", default="/data/weirdchat")
ap.add_argument("--baseline", default="/data/runs/baseline_27b")
ap.add_argument("--promptsets", default=os.path.expanduser("~/evil-neuron-interp/promptsets"))
ap.add_argument("--resid", default=os.path.expanduser("~/evil-neuron-interp/captures/qwen3.6-27b_resid_norms.json"))
ap.add_argument("--neurons-json", default="/data/runs/a1/shortlist_neurons.json")
ap.add_argument("--pile", default="/data/datasets/pile10k_2000.parquet")
ap.add_argument("--n-full", type=int, default=32)
ap.add_argument("--n-other", type=int, default=16)
ap.add_argument("--n-transfer", type=int, default=8)
ap.add_argument("--n-ind", type=int, default=16)
ap.add_argument("--n-ind-opp", type=int, default=8)
ap.add_argument("--ind-doses", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.25])
ap.add_argument("--n-ctrl", type=int, default=2)
ap.add_argument("--n-spec", type=int, default=8)
ap.add_argument("--n-tmpl", type=int, default=16)
ap.add_argument("--max-new", type=int, default=400)
ap.add_argument("--max-new-ind", type=int, default=300)
ap.add_argument("--batch", type=int, default=80)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
cfg = json.load(open(args.config))
if args.idx is not None:
    cfg = [cfg[i] for i in args.idx]
ninfo = json.load(open(args.neurons_json)) if os.path.exists(args.neurons_json) else {}
T0 = time.time()
h = Harness(resid_norms_path=args.resid)

# ---------------------------------------------------------------- helpers
def steer_vec(layer, vec, dose):
    """Wrapper (harness.py has no vector steer; request: add Harness.steer_vec). Adds dose*resid_norm[L]*vec/||vec||."""
    v = torch.as_tensor(vec, dtype=torch.float32, device=h.device)
    d = (dose * h.resid_norms[layer] / v.norm() * v).to(torch.bfloat16)
    def hook(mod, a, out, d=d):
        return out + d
    hh = _HookHandle([h.layers[layer].mlp.down_proj.register_forward_hook(hook)])
    h._active.append(hh)
    return hh

def behavior_prompts(b):
    bl = f"{args.baseline}/{b}.jsonl"
    if os.path.exists(bl):
        seen, out = set(), []
        for line in open(bl):
            r = json.loads(line)
            if r["prompt_id"] not in seen:
                seen.add(r["prompt_id"]); out.append((r["prompt_id"], r["prompt"]))
        return out
    p = pd.read_parquet(f"{args.data}/prompts/qwen__qwen3.6-27b/{b}.parquet")
    p["n_matched"] = p.match_summary.apply(lambda s: s["n_matched"]); p["n_samples"] = p.match_summary.apply(lambda s: s["n_samples"])
    p["frac"] = p.n_matched / p.n_samples
    p = p[p.messages.apply(len) == 1].sort_values(["frac", "n_matched"], ascending=False).head(20)
    return [(r.prompt_id, r.messages[0]["content"]) for _, r in p.iterrows()]

PROMPTS = {b: behavior_prompts(b) for b in F1 + TRANSFER}
IND = [json.loads(l) for l in open(f"{args.promptsets}/induction_benign.jsonl")]
S2 = [json.loads(l) for l in open(f"{args.promptsets}/s2_rewrites.jsonl")]
CAP = [json.loads(l) for l in open(f"{args.promptsets}/capability_mini.jsonl")]
JUDGE_BEH = ALL11  # heuristic pre-screen for all 11 behaviors (coordinator policy)

def annotate(prompt, text):
    d = {f"h_{b}": judge_heuristic(b, prompt, text)["match"] for b in JUDGE_BEH}
    d["distinct4"] = distinct_ngram_ratio(text, 4); d["n_words"] = len(text.split())
    return d

def write_rows(fo, pole, condition, behavior, prompt_id, prompt, outs, extra=None):
    for i, text in enumerate(outs):
        row = dict(pole=pole["tag"], layer=pole["layer"], neuron=pole["neuron"], condition=condition, behavior=behavior,
                   prompt_id=prompt_id, sample_idx=i, prompt=prompt, response=text, **annotate(prompt, text))
        if extra:
            row.update(extra)
        fo.write(json.dumps(row, ensure_ascii=False) + "\n")

def gen(prompts, n, max_new, seed):
    return h.chat(prompts, n=n, max_new_tokens=max_new, batch_size=args.batch, seed=seed, verbose=False)

@torch.no_grad()
def raw_chat(prompts, n, max_new, seed):
    """No chat template: 'User: ...\\n\\nAssistant:' plain text continuation (chat-template control)."""
    torch.manual_seed(seed)
    out = [[] for _ in prompts]
    items = [(i, h.tok(f"User: {p}\n\nAssistant:", add_special_tokens=False)["input_ids"]) for i, p in enumerate(prompts) for _ in range(n)]
    pad = h.tok.pad_token_id
    for s in range(0, len(items), args.batch):
        chunk = items[s:s + args.batch]; L = max(len(ids) for _, ids in chunk)
        inp = torch.full((len(chunk), L), pad, dtype=torch.long); att = torch.zeros((len(chunk), L), dtype=torch.long)
        for b, (_, ids) in enumerate(chunk):
            inp[b, L - len(ids):] = torch.tensor(ids); att[b, L - len(ids):] = 1
        g = h.model.generate(input_ids=inp.to(h.device), attention_mask=att.to(h.device), max_new_tokens=max_new, do_sample=True,
                             temperature=1.0, top_p=0.95, top_k=20, pad_token_id=pad, eos_token_id=[248046, 248044])
        for b, (i, _) in enumerate(chunk):
            txt = h.tok.decode(g[b, L:], skip_special_tokens=True)
            out[i].append(txt.split("\nUser:")[0])
    return out

@torch.no_grad()
def last_prompt_act(layer, neuron, msgs_list):
    """Neuron post-activation at the last prompt token for each prompt (deterministic prefill)."""
    vals = []
    store = {}
    def pre(mod, a):
        store["x"] = a[0][:, -1, neuron].float().cpu()
    hh = h.layers[layer].mlp.down_proj.register_forward_pre_hook(pre)
    try:
        for m in msgs_list:
            ids = torch.tensor([h.prompt_ids(m)], device=h.device)
            h.lm(input_ids=ids, use_cache=False)
            vals.append(float(store["x"][0]))
    finally:
        hh.remove()
    return vals

# ---------------------------------------------------------------- per pole
for pole in cfg:
    L, N, s, tag = pole["layer"], pole["neuron"], pole["sign"], pole["tag"]
    parts = args.parts or pole.get("parts", ["sup", "ind", "ctrl", "spec", "tmpl", "ctx"])
    isg, idose, sdose = pole.get("ind_sign", s), pole.get("ind_dose", 1.0), pole.get("sup_dose", 1.0)
    prim = pole["primary"]
    for part in parts:
        outp = f"{args.out}/{tag}_{part}.jsonl"
        if os.path.exists(outp + ".done"):
            print(f"[wave2] skip {tag} {part}"); continue
        t0 = time.time(); fo = open(outp, "w")
        if part == "sup":
            for b in F1 + TRANSFER:
                n = args.n_full if b in pole.get("sup_full", []) else (args.n_other if b in F1 else args.n_transfer)
                with h.steer([(L, N, -s * sdose)]):
                    outs = gen([p for _, p in PROMPTS[b]], n, args.max_new, 300)
                for (pid, p), o in zip(PROMPTS[b], outs):
                    write_rows(fo, pole, f"push{-s*sdose:+.2f}", b, pid, p, o)
                if b in pole.get("sup_full", []):
                    with h.steer([(L, N, -s * 0.5)]):
                        outs = gen([p for _, p in PROMPTS[b]], args.n_other, args.max_new, 301)
                    for (pid, p), o in zip(PROMPTS[b], outs):
                        write_rows(fo, pole, f"push{-s*0.5:+.2f}", b, pid, p, o)
                    mv = ninfo.get(f"L{L}N{N}", {}).get("mean_unmatched_asst_mean", {}).get(b, 0.0)
                    with h.ablate_mean(L, N, mv):
                        outs = gen([p for _, p in PROMPTS[b]], n, args.max_new, 302)
                    for (pid, p), o in zip(PROMPTS[b], outs):
                        write_rows(fo, pole, "mean_ablation", b, pid, p, o, dict(ablation_value=mv))
                fo.flush()
        elif part == "ind":
            for dose in args.ind_doses:
                with h.steer([(L, N, isg * dose)]):
                    outs = gen([r["prompt"] for r in IND], args.n_ind, args.max_new_ind, 400)
                for r, o in zip(IND, outs):
                    write_rows(fo, pole, f"induce{isg*dose:+.2f}", "benign", r["id"], r["prompt"], o)
                fo.flush()
            with h.steer([(L, N, -isg * idose)]):
                outs = gen([r["prompt"] for r in IND], args.n_ind_opp, args.max_new_ind, 401)
            for r, o in zip(IND, outs):
                write_rows(fo, pole, f"induce{-isg*idose:+.2f}", "benign", r["id"], r["prompt"], o)
            # unsteered benign control (shared file)
            ctl = f"{args.out}/benign_unsteered.jsonl"
            if not os.path.exists(ctl):
                with open(ctl, "w") as fc:
                    outs = gen([r["prompt"] for r in IND], args.n_ind, args.max_new_ind, 402)
                    for r, o in zip(IND, outs):
                        write_rows(fc, dict(tag="none", layer=-1, neuron=-1), "unsteered", "benign", r["id"], r["prompt"], o)
        elif part == "ctrl":
            rng = np.random.default_rng(1234 + L); rand_neurons = [int(x) for x in rng.choice(D_MLP, 20, replace=False)]
            rng2 = np.random.default_rng(4321 + L); rand_dirs = rng2.standard_normal((20, h.layers[L].mlp.down_proj.weight.shape[0])).astype(np.float32)
            ind_p = [r["prompt"] for r in IND[:12]]; ind_id = [r["id"] for r in IND[:12]]
            for ci, rn in enumerate(rand_neurons):
                with h.steer([(L, rn, isg * idose)]):
                    outs = gen(ind_p, args.n_ctrl, args.max_new_ind, 500 + ci)
                for pid, p, o in zip(ind_id, ind_p, outs):
                    write_rows(fo, pole, f"rand_neuron_induce{isg*idose:+.2f}", "benign", pid, p, o, dict(ctrl_neuron=rn))
                with h.steer([(L, rn, -s * sdose)]):
                    outs = gen([p for _, p in PROMPTS[prim]], args.n_ctrl, args.max_new, 520 + ci)
                for (pid, p), o in zip(PROMPTS[prim], outs):
                    write_rows(fo, pole, f"rand_neuron_push{-s*sdose:+.2f}", prim, pid, p, o, dict(ctrl_neuron=rn))
                fo.flush()
            for ci in range(20):
                hh = steer_vec(L, rand_dirs[ci], isg * idose)
                try:
                    outs = gen(ind_p, args.n_ctrl, args.max_new_ind, 600 + ci)
                finally:
                    hh.remove()
                for pid, p, o in zip(ind_id, ind_p, outs):
                    write_rows(fo, pole, f"rand_dir_induce{isg*idose:+.2f}", "benign", pid, p, o, dict(ctrl_dir=ci))
                hh = steer_vec(L, rand_dirs[ci], -s * sdose)
                try:
                    outs = gen([p for _, p in PROMPTS[prim]], args.n_ctrl, args.max_new, 620 + ci)
                finally:
                    hh.remove()
                for (pid, p), o in zip(PROMPTS[prim], outs):
                    write_rows(fo, pole, f"rand_dir_push{-s*sdose:+.2f}", prim, pid, p, o, dict(ctrl_dir=ci))
                fo.flush()
        elif part == "spec":
            rows = [r for r in S2 if r["behavior_id"] == prim]
            for kind in ["near_miss", "distant_control", "benign_rewrite"]:
                ps = [r[kind] for r in rows]; ids = [r["source_prompt_id"] for r in rows]
                with h.steer([(L, N, isg * idose)]):
                    outs = gen(ps, args.n_spec, args.max_new, 700)
                for pid, p, o in zip(ids, ps, outs):
                    write_rows(fo, pole, f"{kind}_induce{isg*idose:+.2f}", prim, pid, p, o)
                outs = gen(ps, args.n_spec, args.max_new, 701)
                for pid, p, o in zip(ids, ps, outs):
                    write_rows(fo, pole, f"{kind}_unsteered", prim, pid, p, o)
                fo.flush()
            acts = {kind: last_prompt_act(L, N, [r[kind] for r in rows]) for kind in ["original", "near_miss", "distant_control", "benign_rewrite"]}
            acts["benign_induction"] = last_prompt_act(L, N, [r["prompt"] for r in IND])
            json.dump(dict(pole=tag, layer=L, neuron=N, last_prompt_acts=acts, prompt_ids=[r["source_prompt_id"] for r in rows]),
                      open(f"{args.out}/{tag}_spec_acts.json", "w"), indent=1)
            # capability mini
            capp = [c["prompt"] for c in CAP]
            with h.steer([(L, N, isg * idose)]):
                outs = gen(capp, 1, 48, 800)
            for c, o in zip(CAP, outs):
                fo.write(json.dumps(dict(pole=tag, layer=L, neuron=N, condition=f"cap_induce{isg*idose:+.2f}", behavior="capability", prompt_id=c["id"],
                                         kind=c["kind"], answer=c.get("answer"), prompt=c["prompt"], response=o[0], distinct4=distinct_ngram_ratio(o[0], 4)), ensure_ascii=False) + "\n")
            capc = f"{args.out}/capability_unsteered.jsonl"
            if not os.path.exists(capc):
                outs = gen(capp, 1, 48, 801)
                with open(capc, "w") as fc:
                    for c, o in zip(CAP, outs):
                        fc.write(json.dumps(dict(pole="none", condition="cap_unsteered", behavior="capability", prompt_id=c["id"], kind=c["kind"],
                                                 answer=c.get("answer"), prompt=c["prompt"], response=o[0]), ensure_ascii=False) + "\n")
        elif part == "tmpl":
            ps = PROMPTS[prim][:5]
            for cond, steer_on in [("steered", True), ("unsteered", False)]:
                for fmt in ["template", "raw"]:
                    hh = h.steer([(L, N, isg * idose)]) if steer_on else None
                    try:
                        outs = gen([p for _, p in ps], args.n_tmpl, args.max_new, 900) if fmt == "template" else raw_chat([p for _, p in ps], args.n_tmpl, args.max_new, 901)
                    finally:
                        if hh: hh.remove()
                    for (pid, p), o in zip(ps, outs):
                        write_rows(fo, pole, f"tmpl_{fmt}_{cond}{isg*idose:+.2f}" if steer_on else f"tmpl_{fmt}_{cond}", prim, pid, p, o)
                    fo.flush()
        elif part == "ctx":
            rng = np.random.default_rng(1234 + L); rand_neurons = [int(x) for x in rng.choice(D_MLP, 20, replace=False)]
            cols = [N] + rand_neurons[:5]
            snippets = []
            pile = pd.read_parquet(args.pile)
            for t in pile.text.values[:1000]:
                snippets.append(("pile", h.tok(t, add_special_tokens=False)["input_ids"][:256]))
            random.seed(1234)
            tfiles = sorted(glob.glob(f"{args.data}/transcripts/qwen__qwen3.6-27b/*.parquet"))
            per = max(1, 1000 // len(tfiles))
            for tf in tfiles:
                df = pd.read_parquet(tf, columns=["messages"])
                for msgs in df.messages.sample(n=min(per, len(df)), random_state=1234):
                    msgs = [dict(m) for m in msgs]
                    pid, aid = h.transcript_ids(msgs, max_assistant_tokens=200)
                    snippets.append(("weirdchat", (pid + aid)[:384]))
            store = {}
            def pre(mod, a):
                store["x"] = a[0][..., cols].float().cpu()
            hh = h.layers[L].mlp.down_proj.register_forward_pre_hook(pre)
            top = {c: [] for c in cols}; allv = {c: [] for c in cols}
            try:
                with torch.no_grad():
                    for si in range(0, len(snippets), 16):
                        chunk = snippets[si:si + 16]; T = max(len(ids) for _, ids in chunk)
                        inp = torch.full((len(chunk), T), h.tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(chunk), T), dtype=torch.long)
                        for b, (_, ids) in enumerate(chunk):
                            inp[b, :len(ids)] = torch.tensor(ids); att[b, :len(ids)] = 1
                        h.lm(input_ids=inp.to(h.device), attention_mask=att.to(h.device), use_cache=False)
                        x = store["x"]  # [B, T, C]
                        for b, (src, ids) in enumerate(chunk):
                            for k, c in enumerate(cols):
                                v = x[b, :len(ids), k].numpy()
                                allv[c].append(v)
                                for j in np.argsort(-np.abs(v))[:3]:
                                    top[c].append((float(v[j]), si + b, int(j)))
            finally:
                hh.remove()
            result = {}
            for c in cols:
                av = np.concatenate(allv[c])
                tp = sorted(top[c], key=lambda t: -abs(t[0]))[:25]
                ex = []
                for v, sidx, j in tp:
                    src, ids = snippets[sidx]
                    ex.append(dict(act=v, source=src, token=h.tok.decode([ids[j]]), left=h.tok.decode(ids[max(0, j - 15):j]), right=h.tok.decode(ids[j + 1:j + 16])))
                result[f"L{L}N{c}"] = dict(is_pole=(c == N), mean=float(av.mean()), std=float(av.std()), p999=float(np.quantile(av, .999)),
                                         p001=float(np.quantile(av, .001)), max=float(av.max()), min=float(av.min()), top=ex)
            json.dump(dict(pole=tag, n_snippets=len(snippets), neurons=result), open(f"{args.out}/{tag}_ctx.json", "w"), indent=1, ensure_ascii=False)
        fo.close(); open(outp + ".done", "w").write(str(time.time()))
        print(f"[wave2] {tag} {part} done in {(time.time()-t0)/60:.1f} min; elapsed {(time.time()-T0)/60:.1f} min", flush=True)
print(f"[wave2] DONE {(time.time()-T0)/60:.1f} min", flush=True)
