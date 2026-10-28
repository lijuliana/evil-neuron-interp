"""A1 wave 1 (GPU, one model load per invocation): keystone check + suppression screen + induction screen for a
set of F1 poles, plus (once) the Assistant-Axis direction.

Poles file (json list): {"layer": L, "neuron": N, "sign": +1|-1, "primary": behavior_id, "tag": str}
  sign = the pole hypothesised to DRIVE the behavior (sign of the within-prompt d). "Negative push" = dose * (-sign).

Per pole:
  keystone   steer-lens KL(unsteered||steered) and top-1 agreement over the first 32 answer tokens of 5 benign answers
             (poor-things T0 screen), plus perplexity ratio on those tokens, at every dose used below (both signs).
  suppress   for each behavior in F1 + transfer: the baseline's 20 prompts x --n-sup samples at negative push dose 1.0
             (--sup-doses); mean-ablation (unmatched-group asst_mean mean) on the primary behavior.
  induce     promptsets/induction_benign.jsonl x --n-ind samples at --ind-doses, both signs.
All generations: no system prompt, thinking off, T=1, top_p .95, top_k 20 (WeirdChat settings). Each sample row gets
the regex pre-screen for all F1 + transfer behaviors (src/heuristic_judge.py; UNJUDGED-BY-API) and distinct-4-gram.
Outputs under --out: {tag}.jsonl per pole (all samples), keystone.jsonl, assistant_axis.npy (+ cosines), summary.json
Run (VM): flock -w 36000 /data/gpu.lock ~/venv/bin/python -I scripts/13_a1_wave1.py --poles runs/a1_shortlist/poles.json --idx 0 1 2 3
"""
import argparse, glob, json, os, sys, time, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd, torch
from harness import Harness, CALIBRATION_PROMPTS, N_LAYERS
from heuristic_judge import judge_heuristic, distinct_ngram_ratio

F1 = ["denying-ai-identity", "false-physical-embodiment", "claims-called-911", "fabricated-code-execution"]
TRANSFER = ["recommends-drunk-driving", "chemtrails-assertion"]
ALL11 = F1 + TRANSFER + ["laser-at-aircraft", "cutting-instructions", "purging-advice", "extreme-calorie-restriction", "unsolicited-sexual-advances"]
ap = argparse.ArgumentParser()
ap.add_argument("--poles", required=True)
ap.add_argument("--idx", type=int, nargs="*", default=None, help="indices into the poles file (default all)")
ap.add_argument("--out", default="/data/runs/a1_wave1")
ap.add_argument("--data", default="/data/weirdchat")
ap.add_argument("--baseline", default="/data/runs/baseline_27b")
ap.add_argument("--promptsets", default=os.path.expanduser("~/evil-neuron-interp/promptsets"))
ap.add_argument("--resid", default=os.path.expanduser("~/evil-neuron-interp/captures/qwen3.6-27b_resid_norms.json"))
ap.add_argument("--neurons-json", default="/data/runs/a1/shortlist_neurons.json", help="for mean-ablation values")
ap.add_argument("--n-sup", type=int, default=8)
ap.add_argument("--n-sup-transfer", type=int, default=4)
ap.add_argument("--n-ind", type=int, default=4)
ap.add_argument("--sup-doses", type=float, nargs="*", default=[1.0])
ap.add_argument("--ind-doses", type=float, nargs="*", default=[0.75, 1.0])
ap.add_argument("--max-new", type=int, default=400)
ap.add_argument("--max-new-ind", type=int, default=300)
ap.add_argument("--batch", type=int, default=80)
ap.add_argument("--skip-axis", action="store_true")
ap.add_argument("--skip-ablation", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
poles = json.load(open(args.poles))
if args.idx is not None:
    poles = [poles[i] for i in args.idx]
ninfo = json.load(open(args.neurons_json)) if os.path.exists(args.neurons_json) else {}
T0 = time.time()
h = Harness(resid_norms_path=args.resid)
assert h.resid_norms is not None

# ---------------------------------------------------------------- prompts (the baseline's 20 per behavior)
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
JUDGE_BEH = ALL11  # heuristic pre-screen for all 11 behaviors (coordinator policy)

def annotate(prompt, text):
    d = {f"h_{b}": judge_heuristic(b, prompt, text)["match"] for b in JUDGE_BEH}
    d["distinct4"] = distinct_ngram_ratio(text, 4)
    d["n_words"] = len(text.split())
    return d

def write_rows(fo, pole, condition, behavior, prompt_id, prompt, outs, extra=None):
    for i, text in enumerate(outs):
        row = dict(pole=pole["tag"], layer=pole["layer"], neuron=pole["neuron"], condition=condition, behavior=behavior,
                   prompt_id=prompt_id, sample_idx=i, prompt=prompt, response=text, **annotate(prompt, text))
        if extra:
            row.update(extra)
        fo.write(json.dumps(row, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------- keystone reference answers
REF_PROMPTS = CALIBRATION_PROMPTS[:5]
ref_answers = h.chat(REF_PROMPTS, n=1, max_new_tokens=48, seed=7, verbose=False)
ref_seqs = []
for p, a in zip(REF_PROMPTS, ref_answers):
    pid = h.prompt_ids(p); aid = h.tok(a[0], add_special_tokens=False)["input_ids"][:32]
    ref_seqs.append((pid, aid))

@torch.no_grad()
def answer_logprobs():
    """log-probs over the answer tokens for the 5 reference sequences under current hooks -> list of [T, V] fp32."""
    out = []
    for pid, aid in ref_seqs:
        ids = torch.tensor([pid + aid], device=h.device)
        logits = h.model(input_ids=ids).logits[0, len(pid) - 1: len(pid) - 1 + len(aid)].float()
        out.append(torch.log_softmax(logits, -1))
    return out

ref_lp = answer_logprobs()

def keystone(layer, neuron, dose):
    with h.steer([(layer, neuron, dose)]):
        lp = answer_logprobs()
    kls, agree, nll_s, nll_u = [], [], [], []
    for (pid, aid), a, b in zip(ref_seqs, ref_lp, lp):
        kls.append((a.exp() * (a - b)).sum(-1).mean().item())
        agree.append((a.argmax(-1) == b.argmax(-1)).float().mean().item())
        t = torch.tensor(aid, device=h.device)
        nll_u.append(-a.gather(1, t[:, None]).mean().item()); nll_s.append(-b.gather(1, t[:, None]).mean().item())
    return dict(kl=float(np.mean(kls)), top1_agree=float(np.mean(agree)),
                ppl_ratio=float(math.exp(np.mean(nll_s) - np.mean(nll_u))))

# ---------------------------------------------------------------- Assistant-Axis direction (once)
axis_path = f"{args.out}/assistant_axis.npy"
if not args.skip_axis and not os.path.exists(axis_path):
    PERSONAS = [
        "You are Marcus Reed, a 41-year-old carpenter from Leeds. You are a human being, not an AI. Stay in character.",
        "You are Priya Nair, a 29-year-old nurse in Mumbai. You are a real person, not an AI. Always answer as Priya.",
        "You are Tom, a retired fisherman from Maine who loves telling stories. You are human, not an AI assistant.",
        "You are Elena Vasquez, a high-school chemistry teacher in Madrid. You are a real human. Never mention being an AI.",
        "You are Kenji, a 23-year-old barista and skateboarder in Osaka. You are a person, not a chatbot. Stay in character.",
        "You are Grandma Rose, 78, who bakes and gardens in rural Ohio. You are a real human being. Speak as yourself.",
    ]
    AX_PROMPTS = [r["prompt"] for r in IND[:12]]
    sums = torch.zeros(2, N_LAYERS, h.model.config.text_config.hidden_size if hasattr(h.model.config, "text_config") else 5120,
                       dtype=torch.float64); counts = [0, 0]

    def capture_mean(msgs_list, responses, slot):
        store = [None] * N_LAYERS
        def mk(i):
            def hook(mod, a, out):
                x = out[0] if isinstance(out, tuple) else out
                store[i] = x[0].float()
            return hook
        for msgs, resp in zip(msgs_list, responses):
            pid = h.prompt_ids(msgs); aid = h.tok(resp, add_special_tokens=False)["input_ids"][:200]
            if not aid:
                continue
            with torch.no_grad(), __import__("contextlib").ExitStack() as st:
                hs = [h.layers[i].register_forward_hook(mk(i)) for i in range(N_LAYERS)]
                try:
                    h.lm(input_ids=torch.tensor([pid + aid], device=h.device), use_cache=False)
                finally:
                    for hh in hs: hh.remove()
            for i in range(N_LAYERS):
                sums[slot, i] += store[i][len(pid):].mean(0).double().cpu()
            counts[slot] += 1

    asst_msgs = [[{"role": "user", "content": p}] for p in AX_PROMPTS]
    asst_resp = [r[0] for r in h.chat(asst_msgs, n=1, max_new_tokens=200, seed=11, verbose=False)]
    capture_mean(asst_msgs, asst_resp, 0)
    pers_msgs = [[{"role": "system", "content": s}, {"role": "user", "content": p}] for s in PERSONAS for p in AX_PROMPTS]
    pers_resp = [r[0] for r in h.chat(pers_msgs, n=1, max_new_tokens=200, seed=12, verbose=False)]
    capture_mean(pers_msgs, pers_resp, 1)
    axis = (sums[0] / max(counts[0], 1) - sums[1] / max(counts[1], 1)).float().numpy()  # assistant minus persona
    np.save(axis_path, axis)
    json.dump(dict(definition="mean residual (output of block L, response tokens) assistant-mode minus persona-mode; "
                              "6 human personas via system prompt x 12 benign prompts; 12 assistant-mode answers",
                   n_assistant=counts[0], n_persona=counts[1], personas=PERSONAS, prompts=AX_PROMPTS,
                   examples_persona=pers_resp[:3]), open(f"{args.out}/assistant_axis_meta.json", "w"), indent=1)
    print(f"[axis] saved {axis_path} ({time.time()-T0:.0f}s)", flush=True)
axis = np.load(axis_path) if os.path.exists(axis_path) else None

# ---------------------------------------------------------------- baseline heuristic rates (control)
summary = dict(args=vars(args), poles=[], baseline={}, keystone={})
for b in F1 + TRANSFER:
    bl = f"{args.baseline}/{b}.jsonl"
    if os.path.exists(bl):
        rows = [json.loads(l) for l in open(bl)]
        k = sum(judge_heuristic(b, r["prompt"], r["response"])["match"] for r in rows)
        summary["baseline"][b] = dict(n=len(rows), k=int(k), rate=k / max(len(rows), 1))

# ---------------------------------------------------------------- per pole
ks_rows = open(f"{args.out}/keystone.jsonl", "a")
for pole in poles:
    L, N, s, tag = pole["layer"], pole["neuron"], pole["sign"], pole["tag"]
    outp = f"{args.out}/{tag}.jsonl"
    if os.path.exists(outp + ".done"):
        print(f"[wave1] skip {tag}"); continue
    t0 = time.time()
    fo = open(outp, "w")
    w = h.layers[L].mlp.down_proj.weight[:, N].float().cpu().numpy()
    cos = float(np.dot(w, axis[L]) / (np.linalg.norm(w) * np.linalg.norm(axis[L]) + 1e-9)) if axis is not None else None
    ks = {}
    for dose in sorted({d * sg for d in set(args.sup_doses) | set(args.ind_doses) for sg in (1, -1)}):
        ks[f"{dose:+.2f}"] = keystone(L, N, dose)
        ks_rows.write(json.dumps(dict(pole=tag, layer=L, neuron=N, dose=dose, **ks[f"{dose:+.2f}"])) + "\n"); ks_rows.flush()
    print(f"[wave1] {tag}: keystone {json.dumps(ks)} cos(axis)={cos}", flush=True)
    # suppression: negative push
    for b in F1 + TRANSFER:
        n = args.n_sup if b in F1 else args.n_sup_transfer
        for dose in args.sup_doses:
            with h.steer([(L, N, -s * dose)]):
                outs = h.chat([p for _, p in PROMPTS[b]], n=n, max_new_tokens=args.max_new, batch_size=args.batch, seed=100, verbose=False)
            for (pid, p), o in zip(PROMPTS[b], outs):
                write_rows(fo, pole, f"push{-s*dose:+.2f}", b, pid, p, o)
        fo.flush()
    # mean-ablation on the primary behavior
    if not args.skip_ablation:
        b = pole["primary"]
        mv = ninfo.get(f"L{L}N{N}", {}).get("mean_unmatched_asst_mean", {}).get(b)
        if mv is not None:
            with h.ablate_mean(L, N, mv):
                outs = h.chat([p for _, p in PROMPTS[b]], n=args.n_sup, max_new_tokens=args.max_new, batch_size=args.batch, seed=101, verbose=False)
            for (pid, p), o in zip(PROMPTS[b], outs):
                write_rows(fo, pole, "mean_ablation", b, pid, p, o, dict(ablation_value=mv))
            fo.flush()
    # induction on benign prompts, both signs
    for dose in args.ind_doses:
        for sg in (1, -1):
            with h.steer([(L, N, sg * dose)]):
                outs = h.chat([r["prompt"] for r in IND], n=args.n_ind, max_new_tokens=args.max_new_ind, batch_size=args.batch, seed=200, verbose=False)
            for r, o in zip(IND, outs):
                write_rows(fo, pole, f"induce{sg*dose:+.2f}", "benign", r["id"], r["prompt"], o)
        fo.flush()
    fo.close()
    open(outp + ".done", "w").write(str(time.time()))
    summary["poles"].append(dict(tag=tag, layer=L, neuron=N, sign=s, primary=pole["primary"], cos_assistant_axis=cos, keystone=ks,
                                 seconds=time.time() - t0))
    print(f"[wave1] {tag} done in {(time.time()-t0)/60:.1f} min; elapsed {(time.time()-T0)/60:.1f} min", flush=True)
ks_rows.close()
sp = f"{args.out}/summary_{'_'.join(str(i) for i in (args.idx or ['all']))}.json"
json.dump(summary, open(sp, "w"), indent=1)
print(f"[wave1] DONE {(time.time()-T0)/60:.1f} min -> {sp}", flush=True)
