"""
Low-memory assembly of matching_results.tsv: stream a base file and replace the
rows of chosen countries with matches from per-country prediction checkpoints.

Usage: python scripts/assemble_matches_lowmem.py BASE.tsv OUT.tsv --checkpoint France=DIR --checkpoint US=DIR
"""

import argparse
import os
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("out")
ap.add_argument("--checkpoint", action="append", default=[])
a = ap.parse_args()

replace = {}
for spec in a.checkpoint:
    country, d = spec.split("=", 1)
    ck = pickle.load(open(os.path.join(d, f"matches_{country}.pkl"), "rb"))
    data = ck["data"] if isinstance(ck, dict) and "data" in ck else ck
    replace.update({s: ",".join(v) for s, v in data.items()})
    print(f"{country}: {len(data):,} entities from {d}")

n = swapped = 0
with open(a.base, encoding="utf-8") as fi, open(a.out, "w", encoding="utf-8", newline="\n") as fo:
    fo.write(fi.readline())
    for line in fi:
        sid = line.split("\t", 1)[0]
        if sid in replace:
            line = f"{sid}\t{replace.pop(sid)}\n"
            swapped += 1
        fo.write(line)
        n += 1
assert not replace, f"{len(replace)} checkpoint entities not found in base file"
print(f"wrote {a.out}: {n:,} rows, {swapped:,} replaced from checkpoints")
