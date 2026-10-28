#!/usr/bin/env python3
"""Rebuild findings/REGISTRY.md from the YAML frontmatter of every neuron card."""
import glob, os, re, sys
try:
    import yaml
except ImportError:
    yaml = None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for path in sorted(glob.glob(os.path.join(ROOT, "findings/neurons/**/*.md"), recursive=True)):
    if path.endswith("TEMPLATE.md"):
        continue
    txt = open(path).read()
    m = re.match(r"^---\n(.*?)\n---", txt, re.S)
    if not m:
        continue
    fm = yaml.safe_load(m.group(1)) if yaml else {}
    rel = os.path.relpath(path, ROOT)
    rows.append((fm.get("model",""), fm.get("layer",""), fm.get("neuron",""), fm.get("pole",""), fm.get("label",""),
                 fm.get("verdict",""), fm.get("confidence",""), fm.get("hub_score",""),
                 ", ".join(fm.get("behaviors_affected") or []), fm.get("red_team",""), f"[{os.path.basename(path)}]({rel})"))
order = {"hub":0,"lever":1,"candidate":2,"word-writer":3,"null-result":4}
rows.sort(key=lambda r: (order.get(r[5], 9), -(r[7] or 0) if isinstance(r[7], int) else 0))
out = ["# Neuron registry (auto-built by scripts/build_registry.py from findings/neurons/**/*.md)", "",
       "| model | L | N | pole | label | verdict | confidence | hub_score | behaviors | red_team | card |", "|---|---|---|---|---|---|---|---|---|---|---|"]
out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
open(os.path.join(ROOT, "findings/REGISTRY.md"), "w").write("\n".join(out) + "\n")
print(f"{len(rows)} cards")
