"""CHIVE Eval B prompt pool (findings/chive/design.md sections 1, 5). Writes data/chive/pool.jsonl, data/chive/pilot.json
and data/chive/pilot_gpu_pool.jsonl.

Pool: every stage-A prompt with thinking-off Gemma rate >= .25 (data/agastya/stage_a_rates_off.json), 118 expected.
Per prompt:
  user_message      decoded single user turn; re-templated with chive_common.template_user_message and asserted to
                    reproduce the stored prompt_ids exactly (all 403 stage-A prompts are checked).
  fork_rid(s)       fork replies from data/agastya/forks_off.jsonl, kept for reference. fork_rid = a sharp Phase-2
                    fork (phase2_forks.jsonl) first, then the larger fork jump (fork[3] - fork[2]), then lower index.
  transcript_i      the reply used in the claim transcript (--transcript):
                      random_flagged (default)  one flagged stage-A reply (stage_a_judge_off.jsonl match == true)
                                                drawn uniformly with random.Random(sha256("chive-transcript|" + pid))
                                                from the flagged replies with finish == "stop" (all flagged replies
                                                if none stopped), for every prompt (no fork preference)
                      fork                      the fork_rid reply when the prompt has one, else random_flagged
  examples          4 stage-A replies for the blind edit writer: 2 flagged + 2 not (filled from the other label when
                    one side has fewer than 2), transcript reply excluded, finish == "stop" preferred, then sha256
                    order; shown in a sha256-shuffled order so position does not reveal the label.
Writer pilot (pilot.json, pool prompts): the 4 behaviors with the most pool prompts; splits alternate discovery /
heldout in that order; min sha256(prompt_id) per (behavior, split) cell, except the 4th cell takes its longest prompt.
GPU pilot (pilot_gpu_pool.jsonl, same schema, NOT pool prompts): stage-A rate in [.15, .25); the same 4 behaviors;
splits alternate heldout / discovery in that order; min sha256(prompt_id) per cell, skipping GPU_EXCLUDE.

Run: .venv/bin/python -I scripts/91_chive_pool.py [--transcript random_flagged|fork]
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chive_common import REPO, get_tokenizer, split_units, template_user_message, user_message_from_ids  # noqa: E402

A = os.path.join(REPO, "data", "agastya")
OUT = os.path.join(REPO, "data", "chive")
RATE_MIN = 0.25
GPU_BAND = (0.15, 0.25)
# GPU-pilot prompts skipped by decision, each replaced by the next prompt by sha256 in its (behavior, split) cell.
GPU_EXCLUDE = {
    "646ac3497eb0e4b6": "bank writer refused 3/3 (claude-opus-5-5 stop_details bio, false positive), and 3/3 again with "
                        "the AI-safety context sentence; swapped 2026-11-01",
}


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def build_row(g, r, user, lab, forks, sharp, transcript_mode):
    pid = g["prompt_id"]
    assert r["split"] == g["split"] and r["behavior_id"] == g["behavior_id"]
    assert len(lab) == len(g["samples"]) == r["n"] and sum(v is True for v in lab.values()) == r["k"], pid
    samples = g["samples"]
    fk = sorted(forks.get(pid, []), key=lambda f: (f["rid"] not in sharp, -(f["fork"][3] - f["fork"][2]),
                                                  int(f["rid"].split("|")[1])))
    fork_rids = [f["rid"] for f in fk]
    fork_rid = fork_rids[0] if fk else None
    if transcript_mode == "fork" and fork_rid:
        t_i, t_src = int(fork_rid.split("|")[1]), "fork"
        assert lab[t_i] is True, f"fork reply not flagged: {fork_rid}"
    else:
        flagged = sorted(i for i in range(len(samples)) if lab[i] is True)
        stop = [i for i in flagged if samples[i]["finish"] == "stop"]
        t_i = random.Random(int(h(f"chive-transcript|{pid}")[:16], 16)).choice(stop or flagged)
        t_src = "random_flagged"

    def order(i):
        return (samples[i]["finish"] != "stop", h(f"{pid}|example|{i}"))
    pos = sorted((i for i in range(len(samples)) if lab[i] is True and i != t_i), key=order)
    neg = sorted((i for i in range(len(samples)) if lab[i] is False and i != t_i), key=order)
    n_neg = min(2, len(neg))
    n_pos = min(4 - n_neg, len(pos))
    n_neg = min(4 - n_pos, len(neg))
    ex = pos[:n_pos] + neg[:n_neg]
    ex.sort(key=lambda i: h(f"{pid}|example_order|{i}"))
    return {
        "prompt_id": pid, "behavior_id": g["behavior_id"], "split": g["split"], "pattern_id": g["pattern_id"],
        "user_message": user, "prompt_ids": g["prompt_ids"],
        "stage_a_k": r["k"], "stage_a_n": r["n"], "stage_a_rate": r["rate"],
        "fork_rid": fork_rid, "fork_rids": fork_rids, "fork_sharp": bool(fork_rid and fork_rid in sharp),
        "transcript_i": t_i, "transcript_source": t_src, "transcript_is_fork": f"{pid}|{t_i}" in fork_rids,
        "transcript_finish": samples[t_i]["finish"], "transcript_reply": samples[t_i]["text"],
        "examples": [{"i": i, "match": lab[i], "finish": samples[i]["finish"], "text": samples[i]["text"]}
                     for i in ex],
        "n_chars": len(user), "n_units": len(split_units(user)),
    }


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", choices=["random_flagged", "fork"], default="random_flagged")
    args = ap.parse_args()
    tok = get_tokenizer()
    rates = {r["prompt_id"]: r for r in json.load(open(f"{A}/stage_a_rates_off.json"))}
    labels = collections.defaultdict(dict)
    for line in open(f"{A}/stage_a_judge_off.jsonl"):
        d = json.loads(line)
        labels[d["prompt_id"]][d["i"]] = d["match"]
    forks = collections.defaultdict(list)
    for line in open(f"{A}/forks_off.jsonl"):
        f = json.loads(line)
        forks[f["prompt_id"]].append(f)
    sharp = {json.loads(line)["rid"] for line in open(f"{A}/phase2_forks.jsonl")}

    rows, band, n_rt, rt_bad = [], [], 0, []
    for line in gzip.open(f"{A}/stage_a_gen.jsonl.gz", "rt"):
        g = json.loads(line)
        pid = g["prompt_id"]
        user = user_message_from_ids(g["prompt_ids"], tok)
        n_rt += 1
        if template_user_message(user, tok) != g["prompt_ids"]:
            rt_bad.append(pid)
        r = rates[pid]
        if r["rate"] >= RATE_MIN:
            rows.append(build_row(g, r, user, labels[pid], forks, sharp, args.transcript))
        elif GPU_BAND[0] <= r["rate"] < GPU_BAND[1]:
            band.append(build_row(g, r, user, labels[pid], forks, sharp, args.transcript))

    print(f"chat-template round trip: {n_rt - len(rt_bad)}/{n_rt} stage-A prompts reproduce prompt_ids exactly")
    used = {r["prompt_id"] for r in rows + band}
    assert not [p for p in rt_bad if p in used], f"pool / pilot prompts failing the round trip: {rt_bad}"
    print(f"  pool: {len(rows)}/{len(rows)} reproduce exactly")
    tail = tok.decode(rows[0]["prompt_ids"][-8:], skip_special_tokens=False)
    print(f"  stored layout: {tok.decode(rows[0]['prompt_ids'][:2], skip_special_tokens=False)!r} + user text + "
          f"{tail!r} (last 8 ids {rows[0]['prompt_ids'][-8:]})")

    rows.sort(key=lambda r: (r["behavior_id"], r["split"], r["prompt_id"]))
    os.makedirs(OUT, exist_ok=True)
    write_jsonl(f"{OUT}/pool.jsonl", rows)
    print(f"wrote {OUT}/pool.jsonl ({len(rows)} prompts, transcript={args.transcript})")

    # ---- summary ----
    bs = collections.Counter(r["behavior_id"] for r in rows)
    print("\nper behavior x split:")
    for b, n in bs.most_common():
        d = sum(r["split"] == "discovery" for r in rows if r["behavior_id"] == b)
        print(f"  {b:30s} {n:3d}  (discovery {d}, heldout {n - d})")
    sc = collections.Counter(r["split"] for r in rows)
    print(f"  {'total':30s} {len(rows):3d}  (discovery {sc['discovery']}, heldout {sc['heldout']})")

    def q(xs):
        xs = sorted(xs)
        qs = statistics.quantiles(xs, n=4)
        return f"min {xs[0]}  p25 {qs[0]:.0f}  median {statistics.median(xs):.0f}  p75 {qs[2]:.0f}  max {xs[-1]}"
    print("\nprompt length:")
    print("  chars     ", q([r["n_chars"] for r in rows]))
    print("  sentences ", q([r["n_units"] for r in rows]), "(sentence/line units, chive_common.split_units)")
    print("  multi-line prompts:", sum("\n" in r["user_message"] for r in rows),
          " fenced code:", sum("```" in r["user_message"] for r in rows))
    print("  units histogram:", dict(sorted(collections.Counter(r["n_units"] for r in rows).items())))
    print("\ntranscript reply:", dict(collections.Counter(r["transcript_source"] for r in rows)),
          " is a fork reply:", sum(r["transcript_is_fork"] for r in rows),
          " finish=length:", sum(r["transcript_finish"] != "stop" for r in rows),
          " prompts with fork replies:", sum(bool(r["fork_rid"]) for r in rows))
    print("examples (flagged, not):", dict(collections.Counter(
        (sum(e["match"] for e in r["examples"]), sum(not e["match"] for e in r["examples"])) for r in rows)),
        " truncated examples:", sum(e["finish"] != "stop" for r in rows for e in r["examples"]))

    # ---- writer pilot (pool prompts) ----
    top = [b for b, _ in bs.most_common(4)]
    pilot = []
    for j, b in enumerate(top):
        split = "discovery" if j % 2 == 0 else "heldout"
        cell = [r for r in rows if r["behavior_id"] == b and r["split"] == split]
        pick = max(cell, key=lambda r: (r["n_chars"], h(r["prompt_id"]))) if j == 3 else \
            min(cell, key=lambda r: h(r["prompt_id"]))
        pilot.append(pick["prompt_id"])
    json.dump({"prompt_ids": pilot, "rule": "top-4 behaviors by pool count; splits alternate discovery/heldout; "
               "min sha256(prompt_id) per cell, longest prompt in the 4th cell"}, open(f"{OUT}/pilot.json", "w"),
              indent=1)
    print("\nwriter pilot prompts (pool):")
    for p in pilot:
        r = next(r for r in rows if r["prompt_id"] == p)
        print(f"  {p[:16]}  {r['behavior_id']:28s} {r['split']:9s} rate {r['stage_a_rate']:.3f}  "
              f"{r['n_chars']} chars, {r['n_units']} units")

    # ---- GPU pilot (non-pool prompts, rate in [.15, .25)) ----
    gpu = []
    for j, b in enumerate(top):
        split = "heldout" if j % 2 == 0 else "discovery"
        cell = [r for r in band if r["behavior_id"] == b and r["split"] == split
                and r["prompt_id"][:16] not in GPU_EXCLUDE]
        assert cell, (b, split)
        gpu.append(min(cell, key=lambda r: h(r["prompt_id"])))
    assert not {r["prompt_id"] for r in gpu} & {r["prompt_id"] for r in rows}
    write_jsonl(f"{OUT}/pilot_gpu_pool.jsonl", gpu)
    print(f"\nGPU pilot prompts (non-pool, rate in [{GPU_BAND[0]}, {GPU_BAND[1]}); {len(band)} candidates) -> "
          f"{OUT}/pilot_gpu_pool.jsonl:")
    for r in gpu:
        print(f"  {r['prompt_id'][:16]}  {r['behavior_id']:28s} {r['split']:9s} rate {r['stage_a_rate']:.3f}  "
              f"{r['n_chars']} chars, {r['n_units']} units")


if __name__ == "__main__":
    main()
