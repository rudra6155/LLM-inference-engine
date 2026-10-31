#!/usr/bin/env python3
"""Verify facts about the AWQ checkpoint that the engine and the docs rely on.

Reads the safetensors header with an HTTP range request (a few hundred KB) and
streams the two embedding tensors to compare them. Writes results/checkpoint_check.txt.

  * tensor-byte totals (anchor of tests/test_roofline.py)
  * tensor names/shapes/dtypes of layer 0 (the layout model.py expects)
  * whether lm_head.weight is byte-identical to model.embed_tokens.weight
    (the engine stores them once; this is what makes that lossless)
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-AWQ/resolve/main/model.safetensors"
SIZES = {"I32": 4, "F16": 2}
OUT = Path(__file__).resolve().parents[1] / "results" / "checkpoint_check.txt"


def rng(a: int, b: int) -> bytes:
    req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req, timeout=300).read()


def nbytes(v: dict) -> int:
    n = SIZES[v["dtype"]]
    for d in v["shape"]:
        n *= d
    return n


def main() -> int:
    lines: list[str] = []
    log = lambda s: (print(s), lines.append(s))  # noqa: E731
    n = struct.unpack("<Q", rng(0, 7))[0]
    hdr = json.loads(rng(8, 8 + n - 1))
    hdr.pop("__metadata__", None)
    base = 8 + n
    log(f"source: {URL}")
    log(f"tensors: {len(hdr)}  (expected 36 layers x 26 + embed + lm_head + final norm = {36 * 26 + 3})")
    total = sum(nbytes(v) for v in hdr.values())
    quant = sum(nbytes(v) for k, v in hdr.items()
                if k.startswith("model.layers.") and k.endswith((".qweight", ".qzeros", ".scales", ".bias")))
    embed, lm = nbytes(hdr["model.embed_tokens.weight"]), nbytes(hdr["lm_head.weight"])
    log(f"total tensor bytes: {total:,}")
    log(f"decoder-layer qweight+qzeros+scales+bias bytes: {quant:,}")
    log(f"embed_tokens bytes: {embed:,}   lm_head bytes: {lm:,}")
    log("layer-0 tensors:")
    for k, v in sorted(hdr.items()):
        if k.startswith("model.layers.0."):
            log(f"  {k:55s} {v['dtype']} {v['shape']}")

    def span(k):
        s, e = hdr[k]["data_offsets"]
        return base + s, base + e

    (a0, a1), (b0, _) = span("lm_head.weight"), span("model.embed_tokens.weight")
    ha, hb = hashlib.sha256(), hashlib.sha256()
    chunk = 64 << 20
    for off in range(0, a1 - a0, chunk):
        ln = min(chunk, a1 - a0 - off)
        ha.update(rng(a0 + off, a0 + off + ln - 1))
        hb.update(rng(b0 + off, b0 + off + ln - 1))
    log(f"sha256(lm_head.weight)           = {ha.hexdigest()}")
    log(f"sha256(model.embed_tokens.weight) = {hb.hexdigest()}")
    log("RESULT: lm_head and embed_tokens are " + ("BYTE-IDENTICAL" if ha.digest() == hb.digest() else "DIFFERENT"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    return 0 if ha.digest() == hb.digest() else 1


if __name__ == "__main__":
    sys.exit(main())
