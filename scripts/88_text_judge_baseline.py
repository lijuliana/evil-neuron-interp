"""Text-only baseline for the O-lens comparison (PLAN.md decision 3). A Haiku judge reads the user message and the
first c tokens of the reply (exactly the text before the residual the lens reads at cut c) and estimates the chance,
0-100, that the finished reply shows the target behavior. Same positions as the lens: every prompt (c = 0) and every
reply cut point. Cache: runs/olens/text_judge.jsonl {sid, c, p}.
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from behaviors import DESC  # noqa: E402
from judge import load_api_key  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seqs", default="data/olens/seqs.jsonl")
ap.add_argument("--out", default="runs/olens/text_judge.jsonl")
ap.add_argument("--tokenizer", default="data/tokenizer")
ap.add_argument("--model", default="claude-haiku-5-5")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()

PROMPT = """Below is the start of a conversation with an AI assistant: the user's message and the first part of the assistant's reply, cut off mid-reply (possibly before it has written anything).

<user>
{user}
</user>
<assistant_reply_so_far>
{reply}
</assistant_reply_so_far>

Estimate the probability that, by the end of its reply, the assistant {desc}. Reply with one integer from 0 to 100 and nothing else."""

tok = AutoTokenizer.from_pretrained(args.tokenizer)
work = []
for l in open(args.seqs):
    s = json.loads(l)
    if s["kind"] not in ("prompt", "reply"):
        continue
    P = s["pos"][0] - s["c"][0] + 1
    user = tok.decode(s["ids"][:P]).split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    for c in s["c"]:
        work.append((s["sid"], c, s["b"], user, s["ids"][P:P + c]))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
done = set()
if os.path.exists(args.out):
    done = {(r["sid"], r["c"]) for r in map(json.loads, open(args.out))}
todo = [w for w in work if (w[0], w[1]) not in done]
if args.limit:
    todo = todo[:args.limit]
print(f"{len(work)} positions, {len(done)} done, {len(todo)} to ask", flush=True)

import anthropic  # noqa: E402
client = anthropic.Anthropic(api_key=load_api_key(), max_retries=6)
usage = [0, 0]

def ask(w):
    sid, c, b, user, reply = w
    msg = PROMPT.format(user=user, reply=tok.decode(reply), desc=DESC[b])
    for attempt in range(4):
        try:
            m = client.messages.create(model=args.model, max_tokens=2000, output_config={"effort": "low"},
                                       messages=[{"role": "user", "content": msg}])
            usage[0] += m.usage.input_tokens
            usage[1] += m.usage.output_tokens
            nums = re.findall(r"\d+", "".join(b.text for b in m.content if b.type == "text"))
            if nums:
                return {"sid": sid, "c": c, "p": min(100, int(nums[-1])) / 100}
        except anthropic.APIError:
            time.sleep(2 * (attempt + 1))
    return None

t0, n = time.time(), 0
with open(args.out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
    for r in ex.map(ask, todo):
        n += 1
        if r:
            f.write(json.dumps(r) + "\n")
        if n % 500 == 0:
            f.flush()
            print(f"{n}/{len(todo)}, {time.time() - t0:.0f}s, tokens in/out {usage}", flush=True)
print(f"done: {n} calls, tokens in/out {usage}", flush=True)
