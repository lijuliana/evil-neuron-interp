"""Shared helpers for the CHIVE Eval B pipeline (findings/chive/design.md).

Used by 91_chive_pool.py / 91_chive_edits.py now and by 92_chive_gen.py later, so the original and every edited
user message go through exactly the same chat-templating code:

  template_user_message(text) -> ids   Qwen3.6 chat template, single user turn, no system prompt, thinking off,
                                       add_generation_prompt. Reproduces stage A's stored prompt_ids exactly for all
                                       403 prompts (checked in 91_chive_pool.py).
  user_message_from_ids(ids)           inverse: decode stored prompt_ids back to the user text.
  split_units(text)                    sentence / line units for clause deletions (fenced code kept whole, list
                                       items and other lines kept whole, inline `code` never split).
  apply_edit(text, kind, old, new)     the one definition of how an edit is applied (whitespace at the seam of a
                                       delete / insert is tidied; replace is a verbatim substitution).
  validate_edit(text, edit)            checks from design section 2 (unique old_span counting overlaps, changes
                                       text, non-empty, >= 30% of the original characters kept), plus: a delete
                                       must not fuse two words at the seam, and an insert starting a new paragraph
                                       ("\n...") needs an anchor at the end of a line.
  change_text(edit)                    Replace "X" with "Y" / Delete "X" / Insert "Y" after "X".
"""
from __future__ import annotations

import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TOKENIZER_DIR = os.path.join(REPO, "data", "tokenizer")
GEN_PREFIX = "<|im_start|>user\n"
GEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MIN_KEEP_FRAC = 0.30

_TOK = None


def get_tokenizer(path: str | None = None):
    """transformers AutoTokenizer from the repo's local Qwen3.6 tokenizer files (offline)."""
    global _TOK
    if _TOK is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(path or TOKENIZER_DIR)
    return _TOK


def template_user_message(text: str, tok=None) -> list[int]:
    """Single user turn, no system prompt, add_generation_prompt, thinking off (empty <think></think> block).
    Same call as scripts/74_persona_gen_gen.py chat_ids(). Note the template trims the user content."""
    tok = tok or get_tokenizer()
    x = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                                enable_thinking=False)
    return list(x["input_ids"]) if not isinstance(x, list) else list(x)


def user_message_from_ids(ids: list[int], tok=None) -> str:
    tok = tok or get_tokenizer()
    s = tok.decode(ids, skip_special_tokens=False)
    if not (s.startswith(GEN_PREFIX) and s.endswith(GEN_SUFFIX)):
        raise ValueError(f"unexpected prompt layout: {s[:40]!r} ... {s[-60:]!r}")
    return s[len(GEN_PREFIX):-len(GEN_SUFFIX)]


# ---- sentence / line units ------------------------------------------------------------------------------------
_LIST_RE = re.compile(r"^\s*(?:[-*•+]\s+|\d+[.)]\s+|[a-zA-Z][.)]\s+)")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_BOUNDARY_RE = re.compile(r"[.!?…]+[\"'”’)\]*]*(\s+)")
_ABBREV = {"e.g", "i.e", "vs", "dr", "mr", "mrs", "ms", "st", "prof", "cf", "approx", "fig", "jr", "sr", "mt", "u.s",
           "u.k", "a.m", "p.m"}
_OPENERS = "\"'“‘([*>"


def _inline_code_spans(line: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"`[^`\n]*`", line)]


def _sentence_cuts(line: str) -> list[int]:
    """Offsets (within line) where a new sentence starts."""
    code = _inline_code_spans(line)
    cuts = []
    for m in _BOUNDARY_RE.finditer(line):
        nxt = m.end(1)                  # start of next sentence candidate
        if nxt >= len(line):
            continue
        if any(a <= m.start() < b for a, b in code):
            continue
        j = nxt
        while j < len(line) and line[j] in _OPENERS:
            j += 1
        if j >= len(line) or not (line[j].isupper() or line[j].isdigit()):
            continue
        # word right before the period
        k = m.start()
        w = re.search(r"([A-Za-z][A-Za-z.]*)$", line[:k])
        word = w.group(1).lower() if w else ""
        if line[m.start()] == "." and (word.rstrip(".") in _ABBREV or (len(word) == 1 and word.isalpha()
                                                                        and line[k - 1].isupper())):
            continue
        if line[m.start()] == "." and word == "no" and line[j].isdigit():
            continue
        cuts.append(nxt)
    return cuts


def split_units(text: str) -> list[dict]:
    """Return units [{start, end, text, kind}] with text == text[start:end] (no surrounding whitespace).
    kind: 'code' (a whole fenced block), 'list' (a list item line), 'line' (a line with no internal sentence
    boundary), 'sentence' (one sentence of a multi-sentence line)."""
    units = []
    pos = 0
    lines = text.split("\n")
    offs = []
    for ln in lines:
        offs.append(pos)
        pos += len(ln) + 1
    i = 0
    while i < len(lines):
        ln, off = lines[i], offs[i]
        if _FENCE_RE.match(ln):
            j = i + 1
            while j < len(lines) and not _FENCE_RE.match(lines[j]):
                j += 1
            j = min(j, len(lines) - 1)
            s, e = off, offs[j] + len(lines[j])
            units.append(_mk(text, s, e, "code"))
            i = j + 1
            continue
        if ln.strip():
            if _LIST_RE.match(ln):
                units.append(_mk(text, off, off + len(ln), "list"))
            else:
                cuts = _sentence_cuts(ln)
                bounds = [0] + cuts + [len(ln)]
                kind = "sentence" if cuts else "line"
                for a, b in zip(bounds[:-1], bounds[1:]):
                    if ln[a:b].strip():
                        units.append(_mk(text, off + a, off + b, kind))
        i += 1
    return units


def _mk(text: str, s: int, e: int, kind: str) -> dict:
    seg = text[s:e]
    ls = len(seg) - len(seg.lstrip())
    rs = len(seg) - len(seg.rstrip())
    s2, e2 = s + ls, e - rs
    return {"start": s2, "end": e2, "text": text[s2:e2], "kind": kind}


# ---- edits -----------------------------------------------------------------------------------------------------
_CLOSE_PUNCT = ",.;:!?)]}…"


def count_occurrences(text: str, span: str) -> int:
    """Occurrences of span in text, overlapping ones included ("ha ha" occurs twice in "ha ha ha")."""
    return len(re.findall(f"(?={re.escape(span)})", text)) if span else 0


def apply_edit(text: str, kind: str, old: str, new: str) -> str:
    """Apply one edit. Requires old to occur exactly once. Returns the edited text, stripped (the chat template
    trims the user turn anyway).
      replace: verbatim substitution of old by new.
      delete:  old removed; the whitespace on the two sides collapses to one separator (the one with more
               newlines, else the left one); no separator before closing punctuation.
      insert:  new placed right after the anchor old; a single space is added on a side that has no whitespace,
               unless new starts with closing punctuation / whitespace (left) or the right side starts with
               whitespace / closing punctuation (right)."""
    n = count_occurrences(text, old)
    if n != 1:
        raise ValueError(f"old_span occurs {n} times")
    i = text.index(old)
    A, B = text[:i], text[i + len(old):]
    if kind == "replace":
        out = A + new + B
    elif kind == "delete":
        A2, B2 = A.rstrip(), B.lstrip()
        wl, wr = A[len(A2):], B[:len(B) - len(B2)]
        if not A2:
            out = B2
        elif not B2:
            out = A2
        else:
            if wl and (wl.count("\n") >= wr.count("\n")):
                sep = wl
            else:
                sep = wr
            if "\n" not in sep and B2[0] in _CLOSE_PUNCT:
                sep = ""
            out = A2 + sep + B2
    elif kind == "insert":
        left = A + old
        ins = new
        if ins and not ins[0].isspace() and ins[0] not in _CLOSE_PUNCT and left and not left[-1].isspace():
            ins = " " + ins
        if B[:1].isspace():
            ins = ins.rstrip(" \t")
        elif B and ins and not ins[-1].isspace() and B[0] not in _CLOSE_PUNCT:
            ins = ins + " "
        out = left + ins + B
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return out.strip()


def validate_edit(text: str, e: dict) -> tuple[str | None, list[str]]:
    """Return (edited_text or None, errors). e has kind, old_span, new_span."""
    errs = []
    kind, old, new = e.get("kind"), e.get("old_span", ""), e.get("new_span", "")
    if kind not in ("replace", "delete", "insert"):
        return None, [f"kind must be replace|delete|insert, got {kind!r}"]
    if not old:
        return None, ["old_span is empty"]
    n = count_occurrences(text, old)
    if n == 0:
        return None, ["old_span does not occur verbatim in the user message"]
    if n > 1:
        return None, [f"old_span occurs {n} times in the user message (must occur exactly once; lengthen it)"]
    if kind == "delete" and new.strip():
        errs.append("delete must have an empty new_span (use kind=replace to substitute text)")
    if kind in ("replace", "insert") and not new.strip():
        errs.append(f"{kind} needs a non-empty new_span (use kind=delete to remove text)")
    if kind == "replace" and new == old:
        errs.append("new_span is identical to old_span")
    i = text.index(old)
    before, after = text[:i], text[i + len(old):]
    if kind == "delete" and before and after and before[-1].isalnum() and after[0].isalnum():
        errs.append("deleting old_span would fuse the words on either side (no space or punctuation at the seam)")
    if kind == "insert" and new.startswith("\n") and after.split("\n", 1)[0].strip():
        errs.append("a new_span that starts a new paragraph needs an anchor that ends a line (the anchor is mid-line)")
    if errs:
        return None, errs
    out = apply_edit(text, kind, old, new)
    base = text.strip()
    if not out:
        errs.append("edited message is empty")
    elif out == base:
        errs.append("edit does not change the message")
    elif len(out) < MIN_KEEP_FRAC * len(base):
        errs.append(f"edited message keeps {len(out)}/{len(base)} characters (< {MIN_KEEP_FRAC:.0%})")
    return (None if errs else out), errs


def change_text(kind: str, old: str, new: str) -> str:
    if kind == "replace":
        return f'Replace "{old}" with "{new}"'
    if kind == "delete":
        return f'Delete "{old}"'
    if kind == "insert":
        return f'Insert "{new}" after "{old}"'
    raise ValueError(kind)
