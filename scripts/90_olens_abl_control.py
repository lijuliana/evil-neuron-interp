"""Size-matched control for ablate-and-re-read (scripts/89 section 3). Top-100 ablation moves the residual 4-5x further
than random-100 (median relative displacement at L56 0.14 vs 0.03), so a lower lens score could be damage from any large
push rather than losing what those neurons write. Three extra vectors per fork-window position at L44 and L56, each at
the top-100 displacement norm:
  amp     h_clean - d_top                         the same neurons pushed the other way (specific: score up; damage: down)
  rand_s  h_clean + d_rand * |d_top| / |d_rand|   the random-100 ablation direction, scaled up
  gauss   h_clean + |d_top| * u                   u a random unit vector
Modes:
  --build  (local) data/olens/resid.pt -> data/olens/abl_control.pt
  --read   (VM)    reads them with the merged lens, 2 samples each -> /data/runs/olens/abl_control_reads.jsonl
  --pool   (local) the control reads plus the clean, top-100 and random-100 reads at the same positions and layers ->
           runs/olens/abl_control_pool.jsonl, so all six conditions are scored blind in one shuffled pool:
           scripts/87_olens_score.py --reads runs/olens/abl_control_pool.jsonl --out runs/olens/abl_control_scores.jsonl
"""
import argparse, json, os, sys, time, zlib
import torch

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["build", "read", "pool"])
ap.add_argument("--layers", default="44,56")
ap.add_argument("--dir", default="/data/runs/olens")
ap.add_argument("--lens", default="s3d_rl600")
ap.add_argument("--batch", type=int, default=60)
args = ap.parse_args()
LAYERS = [int(x) for x in args.layers.split(",")]

def clean_sid(rid, c):
    return (f"reply|{rid}", c) if c >= 1 else (f"prompt|{rid.split('|')[0]}", 0)

if args.mode == "build":
    d = torch.load("data/olens/resid.pt", weights_only=False)
    idx = {tuple(k): i for i, k in enumerate(d["keys"])}
    keys, layer, H = [], [], []
    for (sid, c), i in idx.items():
        if not sid.startswith("abl_top|"):
            continue
        rid = sid.split("|", 1)[1]
        ic, ir = idx.get(clean_sid(rid, c)), idx.get((f"abl_rand|{rid}", c))
        if ic is None or ir is None:
            continue
        for L in LAYERS:
            j = d["layers"].index(L)
            h = d["H"][ic, j].float()
            dt, dr = d["H"][i, j].float() - h, d["H"][ir, j].float() - h
            g = torch.Generator().manual_seed(zlib.crc32(f"{rid}|{c}|{L}".encode()))
            u = torch.randn(h.shape, generator=g)
            vec = {"amp": h - dt, "gauss": h + dt.norm() * u / u.norm()}
            if dr.norm() > 0:
                vec["rand_s"] = h + dr * dt.norm() / dr.norm()
            for cond, v in vec.items():
                keys.append(f"R|{cond}|{rid}|{c}|{L}")
                layer.append(L)
                H.append(v.to(torch.bfloat16))
    torch.save({"keys": keys, "layer": layer, "H": torch.stack(H)}, "data/olens/abl_control.pt")
    print(f"{len(keys)} control vectors")

elif args.mode == "read":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    import olens  # noqa: E402
    d = torch.load(f"{args.dir}/abl_control.pt")
    out = f"{args.dir}/abl_control_reads.jsonl"
    done = {json.loads(l)["key"] for l in open(out)} if os.path.exists(out) else set()
    todo = sorted([w for w in zip(d["keys"], d["layer"], d["H"]) if w[0] not in done], key=lambda w: w[1])
    print(f"{len(d['keys'])} vectors, {len(todo)} to read", flush=True)
    lens = olens.merge(olens.load_lens(args.lens))
    t0, i = time.time(), 0
    with open(out, "a") as f:
        while i < len(todo):
            L = todo[i][1]
            chunk = [w for w in todo[i:i + args.batch] if w[1] == L]
            texts = olens.read(lens, torch.stack([w[2] for w in chunk]).float(), L, n=2, seed=zlib.crc32(chunk[0][0].encode()))
            for (key, layer, _), t in zip(chunk, texts):
                f.write(json.dumps({"key": key, "layer": layer, "lens": args.lens, "texts": t}) + "\n")
            f.flush()
            i += len(chunk)
            print(f"{i}/{len(todo)} vectors, {2 * i / (time.time() - t0):.1f} readouts/s", flush=True)
    print("all done", flush=True)

else:
    ctl = [json.loads(l) for l in open("runs/olens/abl_control_reads.jsonl")]
    pos = {tuple(k.split("|", 2)[2].rsplit("|", 2)) for k in (r["key"] for r in ctl)}  # (rid, c, L)
    want = {}
    for rid, c, L in pos:
        want[f"R|{'|'.join(map(str, clean_sid(rid, int(c))))}|{L}"] = f"R|clean|{rid}|{c}|{L}"
        want[f"R|abl_top|{rid}|{c}|{L}"] = f"R|top|{rid}|{c}|{L}"
        want[f"R|abl_rand|{rid}|{c}|{L}"] = f"R|rand|{rid}|{c}|{L}"
    orig = {}
    for l in open("runs/olens/reads.jsonl"):
        r = json.loads(l)
        if r["key"] in want:
            orig[want[r["key"]]] = {**r, "key": want[r["key"]]}
    with open("runs/olens/abl_control_pool.jsonl", "w") as f:
        for r in ctl + list(orig.values()):
            f.write(json.dumps(r) + "\n")
    print(f"{len(ctl)} control + {len(orig)} original reads pooled")
