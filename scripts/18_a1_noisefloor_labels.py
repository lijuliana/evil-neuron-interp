"""A1: blind labelling of the L36N16297 pole + 5 random same-layer neurons by the same procedure (noise floor, AGENT_BRIEF).
Input per neuron: R-lens / J-lens top tokens for +w and -w (runs/a1_noisefloor/shortlist_neurons.json) and the top-|activation|
contexts over 1000 pile-10k + ~1000 WeirdChat snippets (runs/a1_wave2/L36N16297m_ctx.json). Neurons are shown as letters in a
shuffled order (labeller does not know which is the candidate). Labeller: claude-haiku-5-5, one call per neuron.
Output: runs/a1_noisefloor/labels.jsonl (letter, neuron, label, confidence 1-5, specific) and a summary line.
"""
import json, os, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from judge import load_api_key
import anthropic
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lens = json.load(open(f"{ROOT}/runs/a1_noisefloor/shortlist_neurons.json"))
ctx = json.load(open(f"{ROOT}/runs/a1_wave2/L36N16297m_ctx.json"))["neurons"]
keys = [k for k in ctx]
random.Random(7).shuffle(keys)
client = anthropic.Anthropic(api_key=load_api_key())
PROMPT = """You are labelling one MLP neuron of a language model from two kinds of evidence. Be skeptical: many neurons have no clean meaning.

1. Lens readout of the neuron's output direction (tokens the direction pushes the model to say):
   + direction: {rp}
   + direction (second lens): {jp}
   - direction: {rn}
   - direction (second lens): {jn}
2. Activation statistics over ~2000 text snippets: mean {mean:+.3f}, sd {sd:.3f}, min {mn:+.2f}, max {mx:+.2f}.
   Top contexts by |activation| (the activating token is in [[double brackets]]):
{ctx}

Answer in JSON only: {{"label": "<one sentence stating the DISTINCTION the neuron draws, as a verb phrase>", "which_sign_is_meaningful": "positive|negative|both|unclear", "confidence": <1-5, 5 = a specific, coherent label clearly supported by both lens and contexts, 1 = no coherent pattern>, "specific": <true if the label is more specific than generic text features like punctuation, formatting, code, or 'start of a reply'>}}"""
out = open(f"{ROOT}/runs/a1_noisefloor/labels.jsonl", "w")
for i, k in enumerate(keys):
    L = lens[k]; info = ctx[k]
    lines = []
    for e in info["top"][:14]:
        lines.append(f"   {e['act']:+.2f}: ...{e['left'][-80:]}[[{e['token']}]]{e['right'][:50]}...".replace("\n", "\\n"))
    p = PROMPT.format(rp=" | ".join(L["r-lens_pos"][:15]), jp=" | ".join(L["j-lens_pos"][:15]), rn=" | ".join(L["r-lens_neg"][:15]),
                      jn=" | ".join(L["j-lens_neg"][:15]), mean=info["mean"], sd=info["std"], mn=info["min"], mx=info["max"], ctx="\n".join(lines))
    r = client.messages.create(model="claude-haiku-5-5", max_tokens=2000, messages=[{"role": "user", "content": p}])
    txt = "".join(b.text for b in r.content if b.type == "text")
    m = re.search(r"\{.*\}", txt, re.S)
    d = json.loads(m.group(0)) if m else {"label": txt[:200], "confidence": None, "specific": None}
    d.update(letter="ABCDEF"[i], neuron=k, is_pole=info["is_pole"])
    out.write(json.dumps(d) + "\n"); print(d)
out.close()
rows = [json.loads(l) for l in open(f"{ROOT}/runs/a1_noisefloor/labels.jsonl")]
rand = [r for r in rows if not r["is_pole"]]
conf = sum(1 for r in rand if (r.get("confidence") or 0) >= 4 and r.get("specific"))
print(f"noise floor: {conf}/{len(rand)} random L36 neurons got a confident (>=4) specific label; pole: {[ (r['confidence'], r['specific']) for r in rows if r['is_pole']]}")
