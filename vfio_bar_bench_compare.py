#!/usr/bin/env python3
# Compare two vfio_bar_bench result directories (A = baseline, B = candidate).
# Each directory holds the --json outputs of the same runs (same file names).
# Prints the median (p50 for per-op metrics) and p90 of every metric, and the
# relative change of B vs A.
#
# usage: vfio_bar_bench_compare.py A_DIR B_DIR [METRIC_SUBSTRING,...]
import json, os, sys

def load(p):
    d = json.load(open(p)); out = {}
    for r in d.get("results", []):
        case = r["mode"] + "".join(f" {k}={r[k]}" for k in ("variant", "layout", "threads", "stride") if k in r)
        for k, v in r.items():
            if isinstance(v, dict) and ("median" in v or "p50" in v):
                out[(case, k)] = (v.get("median", v.get("p50")), v.get("p90"), v["unit"])
    return d, out

a, b = sys.argv[1:3]
keys = sys.argv[3].split(",") if len(sys.argv) > 3 else None
for f in sorted(x for x in os.listdir(a) if x.endswith(".json") and os.path.exists(os.path.join(b, x))):
    da, A = load(os.path.join(a, f)); db, B = load(os.path.join(b, f))
    print(f"== {f}: A={da['uname']['release']}  B={db['uname']['release']}")
    for key, (ma, pa, u) in A.items():
        if keys and not any(k in key[1] for k in keys): continue
        if key not in B or ma is None or B[key][0] is None: continue
        mb, pb = B[key][0], B[key][1]
        d = (mb - ma) / ma * 100 if ma else float("nan")
        p = f"p90 {pa:10.2f} {pb:10.2f}" if pa is not None and pb is not None else ""
        print(f"  {key[0]:40} {key[1]:28} {ma:12.2f} {mb:12.2f} {d:+7.1f}% {u:4} {p}")
