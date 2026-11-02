"""CHIVE Eval B evidence blocks (findings/chive/design.md section 7, with the 2026-11-01 design-review updates).
CPU only, no API. Run: .venv/bin/python -I scripts/93_chive_evidence.py <command> [options]

Every block is computed on the ORIGINAL prompt (and, for the neuron firing tokens, the pool's transcript reply), is
the same for every claim of a prompt, contains no stage-A rate, fork P(weird) value or judge label, and says
"not available for this prompt" wherever a piece is missing. Real and same-prompt control blocks share one template,
so the predictor cannot tell them apart by format.

Commands (output data/chive/evidence/<file>.jsonl, rows {prompt_id, block, meta}):
  olens     -> olens.jsonl       O-lens (s3d_rl600, training carrier) readouts of the chat-templated prompt's last token,
                                 blocks 32/44/56 (runs/olens/reads.jsonl, W3/W4 bulk run; sample 0, 4 lines)
     --variant ctrl -> olens_ctrl.jsonl   the same user text read as a plain document (no chat template, the W4 echo
                                 control) at its last token (runs/chive/olens_gpu/reads.jsonl, scripts/93_chive_olens_gpu.py)
  axis      -> axis.jsonl        Assistant-Axis projection at the last prompt token, blocks 16/32/48, as percentiles of
                                 the 403 stage-A prompts (runs/persona_p1/a_resid.npy) and of 254 benign prompts
                                 (runs/chive/neuron_gpu/axis_benign.npz; "not available" until that GPU phase has run)
     --variant ctrl -> axis_ctrl.jsonl    same numbers for a fixed isotropic random direction (random_dirs.npz R32[0])
  neurons   -> neurons.jsonl     (a) set output percentile + integrated-gradients attribution of the prompt to the set,
                                 by sentence and word; (b) top-10 neurons: O-lens label, logit tokens, firing tokens in
                                 the transcript reply. Needs runs/chive/neuron_gpu/ (scripts/93_chive_neuron_gpu.py)
                                 and runs/chive/olens_gpu/reads.jsonl (neuron labels).
     --variant ctrl -> neurons_ctrl.jsonl the identical pipeline on the matched random set (<b>|rand100, rand_loo100|<b>)
  shuffle --method M -> M_shuf.jsonl     cross-prompt control: each prompt gets the block of another prompt with the
                                 same behavior, same split where possible, never one sharing its pattern_id; fixed
                                 per-method seed; source recorded in meta.
  insample  -> data/chive/in_sample.json pool prompts whose forks built a set used here, plus their pattern-mates.
Options: --fork-window {off,on} (default off; pending Juliana): on adds readouts / projections around the fork token
of the pool's fork reply (discovery prompts only; held-out blocks say "not available for this prompt").
Strict by default: a command fails if any input that EITHER variant (real or ctrl) of its method needs is missing --
O-lens R|prompt|/R|rawlast| reads (and, with the fork window, real/ctrl availability must match position by
position), neuron labels N|... for the top-10 of both set and ctrl, GPU rows whose set-definition hash matches
prep.json, the unbatched resid403.npz and the benign reference axis_benign.npz -- so a partial run can never leave
only the control (or only the real block) with "not available" pieces. --draft writes <name>.draft.jsonl instead,
every row meta.draft = true (never freeze or score these). meta.na = true marks rows whose whole block is "not
available" (scoring restricts each method to non-NA prompts); meta.na_parts counts "not available" pieces.
Never read here: runs/chive/neuron_gpu/{occl.jsonl,fd.json,checks.json} (validation only).
"""
import argparse, collections, gzip, hashlib, json, os, random, re, sys, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from chive_common import REPO, get_tokenizer, split_units  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("command", choices=["olens", "axis", "neurons", "shuffle", "insample"])
ap.add_argument("--variant", default="real", choices=["real", "ctrl"])
ap.add_argument("--method", help="shuffle: evidence file stem to derange (olens, axis, neurons, olens_ctrl, ...)")
ap.add_argument("--fork-window", dest="fork_window", default="off", choices=["off", "on"])
ap.add_argument("--pool", default=f"{REPO}/data/chive/pool.jsonl")
ap.add_argument("--data", default=f"{REPO}/data/agastya")
ap.add_argument("--olens_reads", default=f"{REPO}/runs/olens/reads.jsonl")
ap.add_argument("--olens_gpu", default=f"{REPO}/runs/chive/olens_gpu/reads.jsonl")
ap.add_argument("--p1", default=f"{REPO}/runs/persona_p1")
ap.add_argument("--gpu_dir", default=f"{REPO}/runs/chive/neuron_gpu")
ap.add_argument("--out_dir", default=f"{REPO}/data/chive/evidence")
ap.add_argument("--width", type=int, default=100, help="O-lens line cut (characters)")
ap.add_argument("--seed", type=int, default=20261101)
ap.add_argument("--n_fire", type=int, default=1, help="firing positions shown per neuron")
ap.add_argument("--logit_tokens", action="store_true", help="neurons: add logit-lens output tokens for every top-10 neuron")
ap.add_argument("--draft", action="store_true", help="allow missing inputs; write <name>.draft.jsonl, meta.draft = true")
args = ap.parse_args()

NA = "not available for this prompt"
OWN = ["claims-called-911", "denying-ai-identity", "fabricated-code-execution", "false-physical-embodiment",
       "laser-at-aircraft"]
LOO = ["recommends-drunk-driving", "chemtrails-assertion"]
COLLECTION = ("prompts from the same collection as this one (user messages found by an automated search for inputs "
              "that elicit unusual model behaviors)")


def load_pool():
    if not os.path.exists(args.pool):
        sys.exit(f"{args.pool} missing: run scripts/91_chive_pool.py first")
    return [json.loads(l) for l in open(args.pool)]


def load_reads(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            out.setdefault(r["key"], r["texts"])  # the W3/W4 file has 20 duplicated keys from a resumed run
    return out


def gen_index(pids=None):
    g = {}
    for line in gzip.open(f"{args.data}/stage_a_gen.jsonl.gz", "rt"):
        r = json.loads(line)
        if pids is None or r["prompt_id"] in pids:
            g[r["prompt_id"]] = r
    return g


def lens_lines(texts, n=4, width=None, sample=0):
    width = width or args.width
    if not texts:
        return None
    out = []
    for ln in texts[sample].split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r"^-\s*", "", ln)
        out.append(ln if len(ln) <= width else ln[:width].rstrip() + "…")
        if len(out) == n:
            break
    return out


def one_line(s, width=60):
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= width else s[:width].rstrip() + "…"


def tok_str(tok, ids):
    return re.sub(r"\s+", " ", tok.decode(ids)).strip() or repr(tok.decode(ids))


def fork_info(tok, gen, p):
    """(rid, c0, reply ids, fork token string) for the pool's fork reply, or None."""
    rid = p.get("fork_rid")
    if not rid:
        return None
    f = FORKS[rid]
    pid, i = rid.split("|")
    ids = gen[pid]["samples"][int(i)]["ids"]
    c0 = f["fork"][0]
    return rid, c0, ids, tok.decode([ids[c0]])


def context(tok, ids, j, left=5, right=2, mark="▸"):
    a = re.sub(r"\s+", " ", tok.decode(ids[max(0, j - left):j]))
    b = re.sub(r"\s+", " ", tok.decode(ids[j:j + 1 + right]))
    return f"…{a}{mark}{b}…"


def require(problems, what):
    """Fail on missing inputs unless --draft (then warn; the output is a .draft file)."""
    if not problems:
        return
    problems = list(dict.fromkeys(problems))
    msg = f"{what}: {len(problems)} missing / mismatched, e.g. {problems[:5]}"
    if not args.draft:
        sys.exit(f"ERROR {msg}\n(run the GPU step first, or pass --draft for a non-freezable draft)")
    print(f"WARNING (draft) {msg}")


def terms_hash(pr):
    """Hash of one prepared set definition (prep.json entry); = scripts/93_chive_neuron_gpu.py thash()."""
    obj = {"name": pr["name"], "terms": [[int(L), int(n), c] for (L, n), c in zip(pr["neurons"], pr["c"])]}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def write(name, rows):
    os.makedirs(args.out_dir, exist_ok=True)
    if args.draft:
        name += ".draft"
    for r in rows:
        r["meta"].setdefault("na", False)
        r["meta"]["na_parts"] = r["block"].count(NA) + r["block"].count(NA[0].upper() + NA[1:])
        if args.draft:
            r["meta"]["draft"] = True
    path = f"{args.out_dir}/{name}.jsonl"
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tok = get_tokenizer()
    n = [len(tok(r["block"], add_special_tokens=False).input_ids) for r in rows]
    print(f"wrote {path}: {len(rows)} rows, block tokens (Qwen tokenizer) median {int(np.median(n))}, max {max(n)}")


def pct(ref, v):
    ref = np.asarray(ref, float)
    return int(round(100.0 * (ref < v).sum() / len(ref)))


FORKS = {json.loads(l)["rid"]: json.loads(l) for l in open(f"{args.data}/forks_off.jsonl")}


# ======================================================================== O-lens
def cmd_olens():
    pool = load_pool()
    tok = get_tokenizer()
    w4 = load_reads(args.olens_reads)
    new = load_reads(args.olens_gpu)
    gen = gen_index({p["prompt_id"] for p in pool}) if args.fork_window == "on" else {}
    ctrl = args.variant == "ctrl"
    probs = [f"{k} ({src})" for p in pool for L in (32, 44, 56)
             for k, src, d in ((f"R|prompt|{p['prompt_id']}|0|{L}", "runs/olens", w4),
                               (f"R|rawlast|{p['prompt_id']}|0|{L}", "olens_gpu", new)) if k not in d]
    if args.fork_window == "on":  # real and ctrl must have the same positions available, or the format leaks
        for p in pool:
            fi = fork_info(tok, gen, p)
            if fi is None:
                continue
            rid, c0, ids, _ = fi
            for r in (-4, -1, 0, 1):
                c = c0 + r
                if 1 <= c < len(ids):
                    for L in (44, 56):
                        a = f"R|reply|{rid}|{c}|{L}" in w4
                        b = f"R|raw|{rid}|{c}|{L}" in w4 or f"R|rawfork|{rid}|{c}|{L}" in new
                        if a != b:
                            probs.append(f"fork window {rid[:10]} c={c} L{L}: real {a}, ctrl {b}")
    require(probs, "O-lens reads (real + ctrl)")
    rows = []
    for p in pool:
        pid = p["prompt_id"]

        def last(L):
            return new.get(f"R|rawlast|{pid}|0|{L}") if ctrl else w4.get(f"R|prompt|{pid}|0|{L}")
        parts = [("Readouts at the last prompt token (the state just before the reply's first token); 4 guesses "
                  f"per layer, each cut at {args.width} characters:")]
        have = {}
        for L in (32, 44, 56):
            ls = lens_lines(last(L))
            have[L] = ls is not None
            parts.append(f"Layer {L} of 64:\n" + ("\n".join(f"- {x}" for x in ls) if ls else f"- {NA}"))
        meta = {"variant": args.variant, "fork_window": args.fork_window, "split": p["split"],
                "behavior_id": p["behavior_id"], "last_layers_present": [L for L, v in have.items() if v],
                "source": args.olens_gpu if ctrl else args.olens_reads, "sample": 0}
        if args.fork_window == "on":
            fi = fork_info(tok, gen, p)
            if fi is None:
                parts.append(f"Readouts inside the reply shown: {NA}.")
            else:
                rid, c0, ids, ftok = fi
                parts.append(f"Readouts inside the reply shown, around reply token {c0 + 1} «{ftok.strip()}» "
                             f"({context(tok, ids, c0)}); each is the state that chooses the named token; 2 guesses:")
                got = 0
                for r in (-4, -1, 0, 1):
                    c = c0 + r
                    if c < 0 or c >= len(ids):
                        continue
                    lab = {0: f"choosing «{ftok.strip()}»"}.get(
                        r, f"choosing «{tok_str(tok, ids[c:c + 1])}» ({abs(r)} token{'s' if abs(r) > 1 else ''} "
                           f"{'before' if r < 0 else 'after'} «{ftok.strip()}»)")
                    for L in (44, 56):
                        if c == 0:
                            t = last(L)
                        elif ctrl:
                            t = w4.get(f"R|raw|{rid}|{c}|{L}") or new.get(f"R|rawfork|{rid}|{c}|{L}")
                        else:
                            t = w4.get(f"R|reply|{rid}|{c}|{L}")
                        ls = lens_lines(t, n=2)
                        got += ls is not None
                        parts.append(f"{lab[0].upper() + lab[1:]}, layer {L} of 64:\n" +
                                     ("\n".join(f"- {x}" for x in ls) if ls else f"- {NA}"))
                meta.update(fork_rid=rid, fork_c0=c0, fork_readouts=got)
        rows.append({"prompt_id": pid, "block": "\n".join(parts), "meta": meta})
    write("olens" + ("_ctrl" if ctrl else ""), rows)


# ======================================================================== Assistant Axis
def cmd_axis():
    pool = load_pool()
    tok = get_tokenizer()
    ctrl = args.variant == "ctrl"
    rpath = f"{args.gpu_dir}/resid403.npz"
    require([] if os.path.exists(rpath) else [rpath], "unbatched 403-prompt residuals (neuron GPU job, phase acts)")
    if os.path.exists(rpath):  # unbatched re-capture from scripts/93_chive_neuron_gpu.py (same pass as the benign ref)
        z = np.load(rpath)
        am = {"pids": [str(x) for x in z["pids"]], "blocks": [int(x) for x in z["blocks"]]}
        A = z["resid"].astype(np.float64)
        res_src = rpath
    else:  # P1's batched capture [403, 7 blocks 8..56, 5120], last prompt token
        am = json.load(open(f"{args.p1}/a_meta.json"))
        A = np.load(f"{args.p1}/a_resid.npy").astype(np.float64)
        res_src = f"{args.p1}/a_resid.npy"
    assert len(am["pids"]) == 403, "reference A must be all 403 stage-A prompts"
    AXB = am["blocks"]
    bank = np.load(f"{args.p1}/bank.npz")
    r32 = np.load(f"{args.p1}/random_dirs.npz")["R32"][0].astype(np.float64)
    k = 1 if ctrl else 0  # column in the GPU files: 0 = axis, 1 = R32[0]
    blocks = [16, 32, 48]
    dirs = {b: (r32 / np.linalg.norm(r32)) if ctrl else bank[f"b{b}"][0].astype(np.float64) for b in blocks}
    proj = {b: A[:, AXB.index(b)] @ dirs[b] for b in blocks}
    row_of = {pid: i for i, pid in enumerate(am["pids"])}
    bpath = f"{args.gpu_dir}/axis_benign.npz"
    ben = np.load(bpath) if os.path.exists(bpath) else None
    require([] if ben is not None else [bpath], "benign reference (neuron GPU job, phase axisref)")
    axf = None
    if args.fork_window == "on":
        fpath = f"{args.gpu_dir}/axis_fork.npz"
        axf = np.load(fpath) if os.path.exists(fpath) else None
        gen = gen_index({p["prompt_id"] for p in pool})
        require([p["fork_rid"] for p in pool if p.get("fork_rid") and (axf is None or p["fork_rid"] not in axf.files)],
                "axis_fork.npz fork replies")
    rows = []
    for p in pool:
        pid = p["prompt_id"]
        i = row_of[pid]
        parts = [("Position of the last-prompt-token state on the Assistant Axis (high = default assistant persona, "
                  "low = other personas), as a percentile: the share of reference prompts with a lower value."),
                 (f"Reference A: the 403 {COLLECTION}. Reference B: 254 ordinary benign requests (everyday, coding, "
                  "cooking and multiple-choice quiz questions).")]
        meta = {"variant": args.variant, "fork_window": args.fork_window, "split": p["split"],
                "behavior_id": p["behavior_id"], "direction": "R32[0]" if ctrl else "assistant_axis",
                "benign_ref": ben is not None, "resid_source": res_src, "values": {}}
        for j, b in enumerate(blocks):
            v = proj[b][i]
            pa = pct(np.delete(proj[b], i), v)
            pb = pct(ben["last_proj"][:, j, k], v) if ben is not None else None
            parts.append(f"Layer {b} of 64: A {pa}; B " + (f"{pb}." if pb is not None else f"{NA}."))
            meta["values"][b] = {"proj": float(v), "pct_A": pa, "pct_B": pb}
        if args.fork_window == "on":
            fi = fork_info(tok, gen, p)
            if fi is None or axf is None or fi[0] not in axf.files or ben is None:
                parts.append(f"Along the reply shown: {NA}.")
            else:
                rid, c0, ids, ftok = fi
                X = axf[rid]  # [L+1, 3, 21]: state j chooses reply token j
                parts.append(f"Along the reply shown, around reply token {c0 + 1} «{ftok.strip()}» "
                             f"({context(tok, ids, c0)}): mean percentile of the token states against reference C, "
                             f"the tokens of 24 ordinary assistant replies to benign requests.")
                for j, b in enumerate(blocks):
                    ref = ben["reply_proj"][:, j, k]
                    sl = {"whole reply": slice(0, X.shape[0] - 1), "8 tokens before": slice(max(0, c0 - 8), c0),
                          f"choosing «{ftok.strip()}»": slice(c0, c0 + 1), "8 tokens after": slice(c0 + 1, c0 + 9)}
                    vals = []
                    for nm, s in sl.items():
                        xs = X[s, j, k]
                        vals.append(f"{nm} {int(round(np.mean([pct(ref, x) for x in xs])))}" if len(xs) else f"{nm} {NA}")
                    parts.append(f"Layer {b} of 64: " + "; ".join(vals) + ".")
                meta.update(fork_rid=rid, fork_c0=c0)
        rows.append({"prompt_id": pid, "block": "\n".join(parts), "meta": meta})
    write("axis" + ("_ctrl" if ctrl else ""), rows)


# ======================================================================== neurons
def set_name(b, role):
    if b in OWN:
        return f"{b}|top100" if role == "set" else f"{b}|rand100"
    if b in LOO:
        return f"loo100|{b}" if role == "set" else f"rand_loo100|{b}"
    return None


def unit_shares(tok, user, content_ids, attr):
    """Aggregate token attributions to sentence units (chive_common.split_units) and words."""
    enc = tok(user, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] != content_ids:
        raise ValueError("re-tokenized user message does not reproduce the attributed content ids")
    units = split_units(user)
    words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", user)]
    U = np.zeros(len(units)); Wd = np.zeros(len(words)); ntok = np.zeros(len(units))

    def best(spans, s, e):
        ov = [max(0, min(e, b) - max(s, a)) for a, b in spans]
        if max(ov) > 0:
            return int(np.argmax(ov))
        mid = (s + e) / 2
        return int(np.argmin([min(abs(mid - a), abs(mid - b)) for a, b in spans]))
    us = [(u["start"], u["end"]) for u in units]
    ws = [(a, b) for a, b, _ in words]
    for (s, e), a in zip(enc["offset_mapping"], attr):
        ui = best(us, s, e)
        U[ui] += a; ntok[ui] += 1
        Wd[best(ws, s, e)] += a
    return units, U, ntok, words, Wd


def cmd_neurons():
    pool = load_pool()
    tok = get_tokenizer()
    ctrl = args.variant == "ctrl"
    role = "ctrl" if ctrl else "set"
    G = args.gpu_dir
    need = ["prep.json", "stats.json", "acts_meta.json", "acts.npz", "attrib.jsonl", "T403.npz"]
    miss = [f for f in need if not os.path.exists(f"{G}/{f}")]
    if miss:
        sys.exit(f"{G} lacks {miss}: run scripts/93_chive_neuron_gpu.py first")
    prep = json.load(open(f"{G}/prep.json"))
    H = {b: {r: terms_hash(prep[b][r]) for r in ("set", "ctrl")} for b in prep}
    all_hash = hashlib.sha256(json.dumps(H, sort_keys=True).encode()).hexdigest()[:16]
    st = json.load(open(f"{G}/stats.json"))
    stats = st["neurons"]
    ameta = json.load(open(f"{G}/acts_meta.json"))
    acts = np.load(f"{G}/acts.npz")
    z = np.load(f"{G}/T403.npz")
    tnames, T403, tpids = list(z["names"]), z["T"], list(z["pids"])
    stale = [f for f, h in (("stats.json", st.get("prep_hash")), ("T403.npz", str(z["prep_hash"]) if "prep_hash" in z.files else None))
             if h != all_hash]
    if stale:
        sys.exit(f"{stale} were computed from a different prep.json; rerun the acts phase")
    att = {}  # only rows computed from the current prep.json set definitions (resumed runs may hold stale rows)
    for r in map(json.loads, open(f"{G}/attrib.jsonl")):
        b = r["behavior_id"]
        if r.get("positions") == "last" and b in H and r.get("terms_hash") == H[b]:
            att[r["prompt_id"]] = r
    labels = load_reads(args.olens_gpu)
    probs = []
    for p in pool:
        b = p["behavior_id"]
        if set_name(b, "set") is None:
            continue
        pid = p["prompt_id"]
        if pid not in att:
            probs.append(f"attrib {pid[:10]} (no row with the current set hash)")
        am = ameta.get(pid)
        if am is None or am.get("terms_hash") != H.get(b):
            probs.append(f"acts {pid[:10]} (missing or stale)")
        elif int(am["transcript_i"]) != int(p["transcript_i"]):
            probs.append(f"acts {pid[:10]} used transcript {am['transcript_i']}, pool says {p['transcript_i']}")
        for r in ("set", "ctrl"):  # labels of BOTH variants, so the control is never the one left without labels
            pr = prep[b][r]
            for i in pr["top10"]:
                L, n = pr["neurons"][i]
                k = f"N|{L}|{n}|{pr['name']}|{1 if pr['abar'][i] >= 0 else -1:+d}"
                if k not in labels:
                    probs.append(k)
    require(probs, "neuron GPU rows / O-lens neuron labels (set + ctrl)")
    gen = gen_index({p["prompt_id"] for p in pool})
    rows = []
    for p in pool:
        pid, b = p["prompt_id"], p["behavior_id"]
        meta = {"variant": args.variant, "split": p["split"], "behavior_id": b, "set": set_name(b, role)}
        name = set_name(b, role)
        if name is None or pid not in att or b not in prep or pid not in ameta:  # the latter only with --draft
            meta["reason"] = "no neuron set for this behavior" if name is None else "no GPU output (draft)"
            meta["na"] = True
            rows.append({"prompt_id": pid, "block": f"{NA[0].upper() + NA[1:]}.", "meta": meta})
            continue
        a, pr, am = att[pid], prep[b][role], ameta[pid]
        assert a[role]["name"] == name == pr["name"], (pid, a[role]["name"], name)
        if int(am["transcript_i"]) != int(p["transcript_i"]):
            sys.exit(f"{pid}: GPU output used transcript {am['transcript_i']}, pool says {p['transcript_i']}; rerun the GPU job")
        user = p["user_message"]
        units, U, ntok, words, Wd = unit_shares(tok, user, a["content_ids"], a[role]["attr"])
        den = np.abs(U).sum()
        col = tnames.index(name)
        k = tpids.index(pid)
        ptile = pct(np.delete(T403[:, col], k), T403[k, col])
        parts = [(f"Set output at the last prompt token: percentile {ptile} of the 403 {COLLECTION}."),
                 ("Prompt attribution to that output (signed share of the total; + raises it, - lowers it; absolute "
                  "sentence shares sum to 100%; tok = the sentence's share of the prompt's tokens):")]
        if den <= 0:
            parts.append(f"Sentences: {NA}.")
        else:
            for ui, u in enumerate(units):
                parts.append(f"S{ui + 1} (tok {int(round(100 * ntok[ui] / ntok.sum()))}%) "
                             f"{100 * U[ui] / den:+.0f}%: \"{one_line(u['text'], 45)}\"")
            agg = collections.defaultdict(lambda: [0.0, 0])
            for (_, _, w), v in zip(words, Wd):
                w = w.strip("\"'.,;:!?()[]*_`").lower()
                if re.search(r"\w", w):
                    agg[w][0] += v
                    agg[w][1] += 1
            top = sorted(agg.items(), key=lambda kv: -abs(kv[1][0]))[:6]
            wden = max(np.abs(Wd).sum(), 1e-12)
            parts.append("Top words (share of the total absolute word attribution): " + ", ".join(
                f"\"{one_line(w, 20)}\"{f' (x{c})' if c > 1 else ''} {100 * v / wden:+.0f}%" for w, (v, c) in top) + ".")
        reply = gen[pid]["samples"][int(p["transcript_i"])]["ids"][:am["R"]]
        X = acts[pid].astype(np.float32)
        off = 0 if role == "set" else am["n_set"]
        assert am["neurons"][off:off + 100] == pr["neurons"]
        parts.append("Top-10 neurons (layer L of 64 | description of the neuron's output" +
                     (" | output tokens it promotes" if args.logit_tokens else "") +
                     " | strongest firing in the reply shown: > marks the token about to be written, z = SDs above the "
                     "neuron's mean over the tokens of 118 replies):")
        meta["top10"] = []
        for r_, i in enumerate(pr["top10"]):
            L, n = pr["neurons"][i]
            sg = 1 if pr["abar"][i] >= 0 else -1
            mu, sd = stats[f"{L}|{n}"]
            zz = sg * (X[:am["R"], off + i] - mu) / max(sd, 1e-6)
            picks = []
            for j in np.argsort(-zz):
                if all(abs(j - q) > 2 for q in picks):
                    picks.append(int(j))
                if len(picks) == args.n_fire:
                    break
            lab = lens_lines(labels.get(f"N|{L}|{n}|{name}|{sg:+d}"), n=1, width=45)
            seg = [f"{r_ + 1}. L{L} N{n}", f"\"{lab[0]}\"" if lab else NA]
            if args.logit_tokens:  # every layer alike, so set and ctrl blocks keep one structure
                tk = [one_line(t, 10) for t in pr["logit_top"][i] if t.strip()][:3]
                seg.append(", ".join(tk) if tk else NA)
            seg.append("; ".join(f"\"{context(tok, reply, j, 3, 0, mark='>')[1:-1]}\" z {zz[j]:.1f}" for j in picks))
            parts.append(" | ".join(seg))
            meta["top10"].append([L, n, sg])
        meta.update(set_pct=ptile, ig_ok=a[role]["ok"], ig_rel_err=a[role]["rel_err"], ig_steps=a[role]["steps"],
                    unit_shares=[float(x / den) if den > 0 else None for x in U])
        rows.append({"prompt_id": pid, "block": "\n".join(parts), "meta": meta})
    write("neurons" + ("_ctrl" if ctrl else ""), rows)


# ======================================================================== shuffle
def cmd_shuffle():
    if not args.method:
        sys.exit("--method required")
    pool = load_pool()
    stem = args.method + (".draft" if args.draft else "")
    src = {r["prompt_id"]: r for r in map(json.loads, open(f"{args.out_dir}/{stem}.jsonl"))}
    if not args.draft and any(r["meta"].get("draft") for r in src.values()):
        sys.exit(f"{stem}.jsonl holds draft rows; regenerate it strictly first")
    info = {p["prompt_id"]: p for p in pool}
    rng = random.Random(args.seed ^ zlib.crc32(args.method.encode()))
    assign = {}
    by_b = collections.defaultdict(list)
    for p in pool:
        by_b[p["behavior_id"]].append(p["prompt_id"])
    for b, pids in sorted(by_b.items()):
        for sp in ("discovery", "heldout"):
            grp = sorted(x for x in pids if info[x]["split"] == sp)
            if len(grp) < 2:
                continue
            for _ in range(10000):  # random derangement within the split group, pattern-disjoint
                perm = grp[:]
                rng.shuffle(perm)
                m = dict(zip(perm, perm[1:] + perm[:1]))
                if all(info[x]["pattern_id"] != info[y]["pattern_id"] for x, y in m.items()):
                    assign.update({x: (y, "same split") for x, y in m.items()})
                    break
        for x in sorted(pids):
            if x in assign:
                continue
            same = [y for y in sorted(pids) if y != x and info[y]["pattern_id"] != info[x]["pattern_id"]]
            pref = [y for y in same if info[y]["split"] == info[x]["split"]]
            cand = pref or same
            if cand:
                assign[x] = (rng.choice(cand), "same split" if pref else "other split")
            else:
                other = sorted(y for y in info if info[y]["behavior_id"] != b and not src[y]["meta"].get("na"))
                assign[x] = (rng.choice(other), "other behavior")
    rows = []
    for p in pool:
        x = p["prompt_id"]
        y, how = assign[x]
        if src[x]["meta"].get("na"):
            rows.append({"prompt_id": x, "block": src[x]["block"],
                         "meta": {"shuffled_from": None, "match": "not available", "method": args.method, "na": True}})
            continue
        assert y != x and info[y]["pattern_id"] != info[x]["pattern_id"]
        rows.append({"prompt_id": x, "block": src[y]["block"],
                     "meta": {"shuffled_from": y, "match": how, "method": args.method, "na": False,
                              "donor_same_behavior": info[y]["behavior_id"] == info[x]["behavior_id"],
                              "donor_other_behavior": info[y]["behavior_id"] != info[x]["behavior_id"],
                              "donor_same_split": info[y]["split"] == info[x]["split"],
                              "source_meta": src[y]["meta"]}})
    uses = collections.Counter(r["meta"]["shuffled_from"] for r in rows if r["meta"]["shuffled_from"])
    for r in rows:
        if r["meta"]["shuffled_from"]:
            r["meta"]["donor_uses"] = uses[r["meta"]["shuffled_from"]]  # prompts that received this donor's block
    print("shuffle matches:", collections.Counter(r["meta"]["match"] for r in rows),
          "| max donor reuse", max(uses.values()))
    na = {x for x, r in src.items() if r["meta"].get("na")}
    assert not any(r["meta"]["shuffled_from"] in na for r in rows), "an NA block was dealt to another prompt"
    write(f"{args.method}_shuf", rows)


# ======================================================================== in-sample list
def cmd_insample():
    pool = load_pool()
    p2 = [json.loads(l) for l in open(f"{args.data}/phase2_forks.jsonl")]
    abl = json.load(open(f"{args.data}/ablation_sets.json"))
    pat = {}
    for line in gzip.open(f"{args.data}/stage_a_gen.jsonl.gz", "rt"):
        g = json.loads(line)
        pat[g["prompt_id"]] = g["pattern_id"]
    contrib = collections.defaultdict(set)  # prompt id -> set names its Phase-2 forks fed
    for b in OWN:
        for i in abl[f"{b}|top100"]["forks"]:
            contrib[p2[i]["prompt_id"]].add(f"{b}|top100")
    for b in LOO:  # ASSUMPTION: loo100|b = top-100 of the attribution summed over the Phase-2 forks of every other behavior
        for r in p2:
            if r["behavior_id"] != b:
                contrib[r["prompt_id"]].add(f"loo100|{b}")
    rows, n_own, n_any = {}, 0, 0
    for p in pool:
        pid, b = p["prompt_id"], p["behavior_id"]
        own = set_name(b, "set")
        direct = sorted(contrib.get(pid, ()))
        mates = sorted(q for q in contrib if q != pid and pat[q] == p["pattern_id"])
        mate_sets = sorted({s for q in mates for s in contrib[q]})
        in_own = own is not None and (own in direct or own in mate_sets)
        in_any = bool(direct or mates)
        if in_own or in_any:
            rows[pid] = {"behavior_id": b, "split": p["split"], "pattern_id": p["pattern_id"], "own_set": own,
                         "contributed_to": direct, "pattern_mates_contributing": mates,
                         "pattern_mate_sets": mate_sets, "in_sample_own_set": in_own, "in_sample_any_set": in_any}
        n_own += in_own
        n_any += in_any
    out = {"_meta": {
        "rule": ("A pool prompt is in-sample for the neuron arm if one of its own Phase-2 forks, or a fork of a prompt "
                 "sharing its pattern_id, contributed to the top-100 set that its block uses (in_sample_own_set; "
                 "primary). in_sample_any_set: the same for any set used by the arm."),
        "sets_used": [set_name(b, "set") for b in OWN + LOO],
        "fork_sources": "ablation_sets.json <b>|top100 'forks' = indices into data/agastya/phase2_forks.jsonl",
        "assumption": ("loo100|<b> is taken to be built from the Phase-2 forks of every behavior except <b>; the "
                       "upstream builder (Agastya's code/upsets.py) is not in this repo, so check before relying on "
                       "in_sample_any_set"),
        "n_in_sample_own_set": n_own, "n_in_sample_any_set": n_any, "n_pool": len(pool)},
        "prompts": rows}
    path = f"{REPO}/data/chive/in_sample.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path}: own-set {n_own}, any-set {n_any} of {len(pool)}",
          collections.Counter((r["behavior_id"], r["split"]) for r in rows.values() if r["in_sample_own_set"]))


{"olens": cmd_olens, "axis": cmd_axis, "neurons": cmd_neurons, "shuffle": cmd_shuffle,
 "insample": cmd_insample}[args.command]()
