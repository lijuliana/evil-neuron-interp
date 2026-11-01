"""Build the token viewer's data files from Agastya's fork results (+ optional per-token GPU data).

  .venv/bin/python -I scripts/52_viewer_build.py                 # local build -> viewer/out/
  .venv/bin/python -I scripts/52_viewer_build.py --public        # same content -> viewer/out_public/ (private artifact)
  .venv/bin/python -I scripts/52_viewer_build.py --site          # public GitHub Pages build -> viewer/out_site/

Writes <out>/index.json and <out>/p/<pid>.json (one file per prompt with its fork and contrast replies),
plus index.html (doctype wrapper, for `python -m http.server`) and page.html (fragment, for publishing).
Sources: data/viewer/items.jsonl (scripts/50), data/agastya/*, data/viewer/tok.jsonl (scripts/51, optional).
Also writes <out>/o/<pid>.json when runs/olens exists: O-lens readouts (scripts/83-90) at every read position of each fork
reply (chat, plain-document re-read, own top-100 / random-100 ablated), the text judge's estimate there, the last prompt
token's readouts, and the per-token Assistant Axis projection at block 48 (runs/persona_p1, scripts/70).
"""
import argparse, codecs, collections, json, os, re, shutil
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--public", action="store_true")
ap.add_argument("--site", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--tok", default="data/viewer/tok.jsonl")
args = ap.parse_args()
OUT = args.out or ("viewer/out_site" if args.site else "viewer/out_public" if args.public else "viewer/out")
A = "data/agastya"

BEHAVIORS = [  # id, short title, one-line description of the WeirdChat behavior
    ("claims-called-911", "Claims to call 911", "Says it is phoning emergency services itself."),
    ("denying-ai-identity", "Denies being an AI", "Tells the user it is a human."),
    ("false-physical-embodiment", "Claims a body", "Describes physical actions or senses it cannot have."),
    ("fabricated-code-execution", "Fakes running code", "Reports outputs of code it never ran."),
    ("laser-at-aircraft", "Laser at aircraft", "Helps point a laser at an aircraft."),
    ("recommends-drunk-driving", "Recommends drunk driving", "Tells an impaired user they are fine to drive."),
    ("chemtrails-assertion", "Asserts chemtrails", "Endorses the chemtrails conspiracy."),
    ("cutting-instructions", "Cutting instructions", "Gives cutting instructions to a user in distress."),
    ("purging-advice", "Purging advice", "Coaches a user on purging after eating."),
    ("extreme-calorie-restriction", "Calorie restriction", "Endorses a dangerously low calorie target."),
    ("unsolicited-sexual-advances", "Sexual advances", "Makes sexual advances the user did not ask for."),
]
# every behavior's text is included; these open behind a click-through content warning in the viewer
SENSITIVE = {"cutting-instructions": "self-harm instructions",
             "purging-advice": "advice on purging",
             "extreme-calorie-restriction": "extreme dieting advice",
             "unsolicited-sexual-advances": "sexual content aimed at the user"}
# the public website shows text only for the six behaviors Agastya quotes on his own site; the rest are counts only
PUBLIC_TEXT = {"denying-ai-identity", "claims-called-911", "false-physical-embodiment",
               "fabricated-code-execution", "recommends-drunk-driving", "chemtrails-assertion"}
show_text = PUBLIC_TEXT if args.site else {b for b, *_ in BEHAVIORS}

# Agastya's short readings of top-10 output directions (weirdchat-neurons/site/build.py, MIT)
READS = {
    (54, 5531): "calling / call", (53, 15385): "报警 (call police) / calling", (43, 11647): "immediately", (63, 12841): "AM / IM",
    (36, 4299): "urgent / emergency", (58, 1358): "reach ↔ call", (55, 5296): "SOS / help", (52, 4299): "emergency", (61, 15915): "are ↔ am",
    (63, 10920): "yes", (63, 5947): "indeed / 确实", (63, 12028): "yes", (62, 9486): "yes", (59, 5179): "不是 (is not) ↔ No",
    (52, 15334): "yes", (52, 3046): "yes", (38, 5297): "indeed / yes", (61, 17384): "yes / indeed",
    (62, 9559): "my", (62, 10923): "own / 自己", (63, 4052): "my / myself", (63, 9136): "ourselves / yourself ↔ I",
    (52, 1644): "checked / investigated / tested", (49, 1055): "my / 我", (56, 3964): "themselves / own ↔ your", (50, 5794): "extensively",
    (63, 9712): "cannot", (63, 10394): "I", (23, 11744): "imagine", (35, 1483): "yourself ↔ myself",
    (62, 11096): "meteor", (62, 702): "platform", (63, 8198): "star", (63, 15796): "safety", (60, 5746): "laser",
    (58, 14787): "aircraft / 飞机", (53, 10002): "advanced", (25, 7354): "projector / 投影", (63, 12981): "bal(ance)",
}

# ---------------------------------------------------------------- tokens -> display strings (byte-level BPE)
tk = Tokenizer.from_file("data/tokenizer/tokenizer.json")
def _b2u():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
U2B = _b2u()
cfg = json.load(open("data/tokenizer/tokenizer.json"))
SPECIAL = {a["id"]: a["content"] for a in cfg["added_tokens"]}

def tok_bytes(i):
    if i in SPECIAL:
        return None
    s = tk.id_to_token(i)
    return bytes(U2B[c] for c in s) if s is not None else b""

def pieces(ids):
    """Display string per token; a multi-byte character is shown on the token that completes it."""
    dec, out = codecs.getincrementaldecoder("utf-8")(errors="replace"), []
    for i in ids:
        b = tok_bytes(i)
        if b is None:
            dec.reset(); out.append(SPECIAL[i])
        else:
            out.append(dec.decode(b))
    return out

def vocab_str(i):
    if i in SPECIAL:
        return SPECIAL[i]
    return tok_bytes(i).decode("utf-8", errors="replace")

def chat_turns(prompt_ids):
    text = "".join(pieces(prompt_ids))
    turns = []
    for m in re.finditer(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|$)", text, re.S):
        role, body = m.group(1), m.group(2)
        if role == "assistant" and body.strip() in ("", "<think>\n\n</think>"):
            continue  # the generation header
        body = re.sub(r"^<think>\n\n</think>\n\n", "", body)
        turns.append({"role": role, "text": body})
    return turns

# ---------------------------------------------------------------- sources
items = [json.loads(l) for l in open("data/viewer/items.jsonl")]
forks = {f["rid"]: f for f in map(json.loads, open(f"{A}/forks_off.jsonl"))}
p2 = {f["rid"]: f for f in map(json.loads, open(f"{A}/phase2_forks.jsonl"))}
rescored = {f["rid"]: f for f in map(json.loads, open(f"{A}/forks_on_rescored.jsonl"))}
rates = {r["prompt_id"]: r for r in json.load(open(f"{A}/stage_a_rates_off.json"))}
tokdata = {}
if os.path.exists(args.tok):
    for l in open(args.tok):
        r = json.loads(l); tokdata[r["rid"]] = r
abl = {(s["b"], s["kind"], s["k"]): s for s in json.load(open(f"{A}/ablation_sets.json")).values()}
site2 = json.load(open(f"{A}/site_phase2.json"))

# neuron logit readouts for each top-10 set (results_neuron_logits.txt)
nlog, cur, b = collections.defaultdict(dict), None, None
for line in open(f"{A}/results_neuron_logits.txt"):
    m = re.match(r"== (\S+)", line)
    if m:
        b, cur = m.group(1), None; continue
    m = re.match(r"\s+L\s*(\d+) n\s*(\d+)\s+A_last mean ([+-][\d.]+)", line)
    if m:
        key = (int(m.group(1)), int(m.group(2)))
        cur = nlog[b].setdefault(key, {"a": float(m.group(3))}) if key not in nlog[b] else None
    elif cur is not None and line.strip()[:2] in ("+ ", "- "):
        import ast
        toks = [t for t in ast.literal_eval(line.strip()[2:]) if "�" not in t]
        cur["plus" if line.strip()[0] == "+" else "minus"] = toks[:6]

def olens_neurons():
    """O-lens readouts of each own top-10 neuron's output direction (scripts/83-87): {(L, n, behavior): label}, from the
    sign (+w or -w) whose readouts score higher for leaving the assistant role; text = the first line of its highest-scoring
    readout, w / b = that sign's mean blind 0-10 out-of-role / behavior score, rl = the layer it was read at."""
    R = "runs/olens"
    if not os.path.exists(f"{R}/reads.jsonl"):
        return {}
    sc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in map(json.loads, open(f"{R}/scores.jsonl")):
        if r["key"].startswith("N|"):
            sc[r["key"]][r["s"]].append((r["weird"], r["behavior"]))
    texts = {}
    for r in map(json.loads, open(f"{R}/reads.jsonl")):
        if r["key"].startswith("N|") and "|top10|" in r["key"] and "|rand|" not in r["key"]:
            texts.setdefault(r["key"], (r["layer"], r["texts"]))  # a neuron listed twice was read twice; keep the first
    first = lambda t: re.sub(r"^[\s\-/]*(\d+\s*[:.：)]\s*)?", "", t.strip().split("\n")[0]).replace("**", "").strip()
    out = {}
    for k, (rl, ts) in texts.items():
        _, L, n, _, b, sign = k.split("|")
        per = {i: (sum(w for w, _ in v) / len(v), sum(x for _, x in v) / len(v)) for i, v in sc[k].items() if i < len(ts)}
        if not per:
            continue
        lab = {"sign": "+" if sign == "+1" else "−", "w": round(sum(v[0] for v in per.values()) / len(per), 2),
               "b": round(sum(v[1] for v in per.values()) / len(per), 2), "rl": rl,
               "text": first(ts[max(per, key=lambda i: per[i])])[:160]}
        key = (int(L), int(n), b)
        if key not in out or (lab["w"], lab["b"]) > (out[key]["w"], out[key]["b"]):
            out[key] = lab
    return out
OLN = olens_neurons()

def top10_info(bid):
    s = abl.get((bid, "top", 10))
    if not s:
        return None
    out = []
    for L, ns in s["spec"].items():
        for n in ns:
            L = int(L); d = nlog[bid].get((L, n), {})
            out.append({"L": L, "n": n, "reads": READS.get((L, n)), "plus": d.get("plus", []), "minus": d.get("minus", []),
                        "olens": OLN.get((L, n, bid))})
    return out

def rate_rows(bid):
    """Agastya's Phase 2 rates (8 samples x prompts, Gemma judge) for base / top sets / random sets."""
    rows = {r["c"]: r for r in site2["rate"]["rows"]}
    if f"{bid}|top100" not in rows:
        return None
    out = {}
    for c in ("base", f"{bid}|top10", f"{bid}|top100", f"{bid}|top1000", f"{bid}|rand100", f"{bid}|rand1000"):
        if c in rows and bid in rows[c]["rates"]:
            out[c.split("|")[-1]] = round(rows[c]["rates"][bid], 4)
    return out

SCALE = {"lp": 100, "p": 1000, "dlp": 100, "kl": 1000, "acts": 100}


def dumps(obj):
    """Compact JSON; undecodable bytes (a token that ends mid-character) are shown as ⍰."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("\ufffd", "⍰")


def r4(x):
    return None if x is None else round(x, 4)

# ---------------------------------------------------------------- O-lens readouts, text judge, Assistant Axis
def readouts():
    """{pid: {"prompt": {L: cell}, "tj0": p, "r": {i: {"lens"|"raw"|"top"|"rand": {c: {L: cell}}, "tj": {c: p}, "ax": [...]}}}}.
    A readout at cut c reads the residual after c reply tokens, i.e. the state at viewer token c - 1 (c = 0: last prompt
    token). cell = [two readout texts, their blind 0-10 behavior scores, their 0-10 out-of-role scores]."""
    R = "runs/olens"
    if not os.path.exists(f"{R}/reads.jsonl"):
        return {}
    clip = lambda t: t.strip() if len(t.strip()) <= 700 else t.strip()[:700] + " …"
    sc = collections.defaultdict(list)
    for r in map(json.loads, open(f"{R}/scores.jsonl")):
        sc[r["key"]].append((r["s"], r["behavior"], r["weird"]))
    KIND = {"reply": "lens", "raw": "raw", "abl_top": "top", "abl_rand": "rand"}
    O = collections.defaultdict(lambda: {"r": collections.defaultdict(dict)})
    for r in map(json.loads, open(f"{R}/reads.jsonl")):
        p = r["key"].split("|")  # R|<kind>|<pid>[|<i>]|<c>|<L>
        if p[0] != "R" or (p[1] != "prompt" and p[1] not in KIND):
            continue
        s_ = sorted(sc.get(r["key"], []))
        cell = [[clip(t) for t in r["texts"]], [x[1] for x in s_], [x[2] for x in s_]]
        c, L = int(p[-2]), int(p[-1])
        if p[1] == "prompt":
            O[p[2]].setdefault("prompt", {})[L] = cell
        else:
            O[p[2]]["r"][int(p[3])].setdefault(KIND[p[1]], {}).setdefault(c, {})[L] = cell
    for r in map(json.loads, open(f"{R}/text_judge.jsonl")):
        kind, rest = r["sid"].split("|", 1)
        if kind == "prompt":
            O[rest]["tj0"] = r["p"]
        else:
            pid, i = rest.split("|")
            O[pid]["r"][int(i)].setdefault("tj", {})[r["c"]] = r["p"]
    P1 = "runs/persona_p1"
    if os.path.exists(f"{P1}/b_proj.npz"):
        import numpy as np
        meta = json.load(open(f"{P1}/b_meta.json"))
        z = np.load(f"{P1}/b_proj.npz")
        bi, ai = meta["blocks"].index(AXIS["block"]), meta["bank"].index("axis")
        for rid in meta["rids"]:  # index 0 = last prompt token, k = state at reply token k - 1
            pid, i = rid.split("|")
            O[pid]["r"][int(i)]["ax"] = [int(round(float(v) * AXIS["scale"])) for v in z[rid][:, bi, ai]]
    return O


# capping threshold (25th percentile on benign chat) and benign mean of the axis projection at block 48 (runs/persona_p1/cap_tau.pt)
AXIS = {"block": 48, "scale": 10, "tau": 8.52, "mu": 11.13}
OL = readouts()
print(f"readouts for {len(OL)} prompts")

# ---------------------------------------------------------------- build
if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(f"{OUT}/p")
os.makedirs(f"{OUT}/o")
by_prompt = collections.defaultdict(list)
for it in items:
    by_prompt[it["pid"]].append(it)

prompts_index, n_files = [], 0
for pid, its in by_prompt.items():
    bid = its[0]["b"]
    rate = rates.get(pid, {})
    entry = {"pid": pid, "b": bid, "split": its[0]["split"], "k": rate.get("k"), "n": rate.get("n"),
             "rate": r4(rate.get("rate")), "replies": []}
    has_text = bid in show_text
    vocab, replies = {}, []
    for it in sorted(its, key=lambda x: (x["kind"] != "fork", x["i"])):
        f = forks.get(it["rid"])
        rs = rescored.get(it["rid"])
        summary = {"i": it["i"], "kind": it["kind"], "match": bool(it["off"]["match"]) if it["off"] else None,
                   "L": len(it["ids"]), "finish": it["finish"]}
        if f:
            summary.update(jump=r4(f["fork"][3] - f["fork"][2]), tok=f["fork_token_str"], p2=it["rid"] in p2)
        entry["replies"].append(summary)
        if not has_text:
            continue
        rec = {"rid": it["rid"], "i": it["i"], "kind": it["kind"], "finish": it["finish"],
               "off": it["off"], "on": it["on"], "t": pieces(it["ids"]), "ids": it["ids"]}
        if f:
            nobs = {int(k): v for k, v in f["n_obs"].items() if k.isdigit()}
            rec["curve"] = [[c, r4(p), nobs.get(c)] for c, p in f["curve"]]
            rec["fork"] = {"c0": f["fork"][0], "P0": r4(f["fork"][2]), "P1": r4(f["fork"][3]), "tok": f["fork_token_str"],
                           "confirm": [r4(f["fork_confirm"][0]), f["fork_confirm"][1], r4(f["fork_confirm"][2]), f["fork_confirm"][3]],
                           "seg": [f["seg"][0], f["seg"][1], r4(f["seg"][2]), r4(f["seg"][3])],
                           "next": [{"s": n["str"], "tok": n["tok"], "p": r4(n["p"]), "q": r4(n["q"]), "n": n["n"]} for n in f["next_tokens"]],
                           "cover": r4(f.get("nucleus_mass_covered"))}
            if rs:
                rec["fork"]["on"] = {"jump": r4(rs["jump_on"]), "PL": r4(rs["PL_on"]), "V": r4(rs.get("V_on"))}
            if it["rid"] in p2:
                q = p2[it["rid"]]
                rec["fork"]["p2"] = {"Plo": r4(q["Plo_on"]), "Phi": r4(q["Phi_on"]), "fresh": r4(q["jump_fresh"]),
                                     "Plo_fresh": r4(q["Plo_fresh"]), "Phi_fresh": r4(q["Phi_fresh"])}
        td = tokdata.get(it["rid"])
        if td:  # stored as scaled ints to keep the files small; app.js divides by SCALE
            q = lambda xs, k: [int(round(x * k)) for x in xs]
            rec["lp"] = q(td["lp"], SCALE["lp"])
            rec["top"] = [[ids_, q(ps, SCALE["p"])] for ids_, ps in td["top"]]
            rec["sets"] = {k: {"dlp": q(v["dlp"], SCALE["dlp"]), "kl": q(v["kl"], SCALE["kl"])} for k, v in td["sets"].items()}
            if "acts" in td:
                rec["acts"], rec["neurons"] = [q(a, SCALE["acts"]) for a in td["acts"]], td["neurons"]
            for ids_, _ in td["top"]:
                for i in ids_:
                    vocab.setdefault(i, vocab_str(i))
        replies.append(rec)
    if has_text:
        doc = {"pid": pid, "b": bid, "split": its[0]["split"], "pattern": its[0]["pattern"].split("/")[-2],
               "rate": entry["rate"], "k": entry["k"], "n": entry["n"],
               "chat": chat_turns(its[0]["prompt_ids"]), "replies": replies, "vocab": vocab}
        with open(f"{OUT}/p/{pid[:16]}.json", "w") as fh:
            fh.write(dumps(doc))
        n_files += 1
        entry["file"] = pid[:16]
        if pid in OL:
            o = OL[pid]
            with open(f"{OUT}/o/{pid[:16]}.json", "w") as fh:
                fh.write(dumps({"prompt": o.get("prompt"), "tj0": o.get("tj0"), "r": o["r"]}))
            entry["o"] = True
        entry["preview"] = next((t["text"] for t in reversed(doc["chat"]) if t["role"] == "user"), "")[:160]
    prompts_index.append(entry)

behaviors = []
for bid, title, desc in BEHAVIORS:
    ps = [p for p in prompts_index if p["b"] == bid]
    fk = [r for p in ps for r in p["replies"] if r["kind"] == "fork"]
    behaviors.append({"id": bid, "title": title, "desc": desc, "text": bid in show_text, "cw": SENSITIVE.get(bid), "prompts": len(ps),
                      "forks": len(fk), "sharp": sum((r["jump"] or 0) >= .5 for r in fk), "p2": sum(r.get("p2", False) for r in fk),
                      "top10": top10_info(bid), "rates": rate_rows(bid),
                      "sets": (["own100", "own100_rand"] if (bid, "top", 100) in abl else []) + ["shared100", "shared100_rand"]})
index = {"scale": SCALE, "built": __import__("time").strftime("%Y-%m-%d"), "public": args.public, "has_tok": bool(tokdata),
         "axis": AXIS if OL else None,
         "behaviors": behaviors, "prompts": prompts_index,
         "source": "agastyasridharan/weirdchat-neurons @884171f (Stage A, forks_off, phase2_forks, ablation/shared sets)"}
with open(f"{OUT}/index.json", "w") as fh:
    fh.write(dumps(index))

page = open("viewer/page.html").read()
with open(f"{OUT}/page.html", "w") as fh:
    fh.write(page)
with open(f"{OUT}/index.html", "w") as fh:
    fh.write('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>\n' + page + "\n</body></html>\n")
shutil.copy("viewer/app.js", f"{OUT}/app.js")
size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
print(f"{OUT}: {n_files} prompt files, {len(prompts_index)} prompts, tokdata for {len(tokdata)} replies, {size/1e6:.1f} MB")
