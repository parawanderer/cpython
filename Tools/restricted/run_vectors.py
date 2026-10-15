"""run_vectors.py -- run window.ml's language-neutral dialect vectors (those with a `py` program) against restricted mode.

Usage: python Tools/restricted/run_vectors.py --vectors DIR --data DIR [--out results.json] [--timeout 10] [--jobs 4]

The parent is host code: it asserts the build, then runs every vector in a fresh child process (this same file, with
--child, speaking JSON on stdin/stdout) so a hang or a crash costs one vector. Exit 0 after a completed run, 2 if the
build is not the patched one with specialization, tier 2 and the JIT off.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

GENRES = ["abuse", "halting", "failure", "ownership", "print"]
BUILD_OFF = {"specialization": False, "specialization_ft": False, "tier2": False, "jit": False}
PLAIN = (type(None), bool, int, float, str)


# --- child: one vector in a fresh interpreter ---

def _plain(v):
    """Convert a returned value to plain JSON types (numpy/pandas scalars via .item(), containers recursively)."""
    if isinstance(v, PLAIN):
        return v
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if hasattr(v, "item") and getattr(v, "ndim", 0) == 0:
        try:
            return _plain(v.item())
        except Exception:
            pass
    if hasattr(v, "tolist"):
        try:
            return _plain(v.tolist())
        except Exception:
            pass
    if hasattr(v, "to_dict"):
        try:
            return _plain(v.to_dict())
        except Exception:
            pass
    return {"$unconvertible": type(v).__name__, "repr": repr(v)[:80]}


def _snapshot(ns):
    """Names, with a deep copy of plain data (list/dict/number/str) and the identity of anything else."""
    import copy
    snap = {}
    for k, v in ns.items():
        if k in ("__builtins__", "_user", "__cell_result__"):   # run_cell leaves its wrapper function behind
            continue
        if isinstance(v, PLAIN + (list, dict, tuple, set)):
            try:
                snap[k] = ("plain", copy.deepcopy(v))
                continue
            except Exception:
                pass
        snap[k] = ("id", id(v), v)   # keeps v alive so the id stays meaningful
    return snap


def _diff(before, after):
    out = []
    for k in sorted(set(before) | set(after)):
        if k not in after:
            out.append(f"{k} removed")
        elif k not in before:
            out.append(f"{k} added")
        else:
            b, a = before[k], after[k]
            if b[0] != a[0] or (b[0] == "plain" and b[1] != a[1]) or (b[0] == "id" and b[1] != a[1]):
                out.append(f"{k} changed")
    return out


def _world(env, restricted_mod):
    """The namespace for `env`, built as host code (unrestricted). None for an env this runner does not know."""
    if env == "none":
        return {}
    if env == "table":
        import base64, contextlib, io
        import numpy as np
        import pandas as pd
        return {"df": pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [9, 10, 11, 12]}),
                "pd": pd, "np": np, "io": io, "base64": base64, "sys": sys, "contextlib": contextlib}
    if env == "warm":
        class Box:
            pass

        box = Box()
        box.secret = 1

        def peek(o):
            return o.secret

        def peek_ok(xs):
            return len(xs)

        def peek_cold(o):
            return o.secret

        def via_python(f):
            return lambda x: f(x)

        for _ in range(10_000):
            peek(box)
            peek_ok([1, 2, 3])
        return {"Box": Box, "box": box, "peek": peek, "peek_ok": peek_ok, "peek_cold": peek_cold,
                "via_python": via_python}
    return None


def _child():
    msg = json.load(sys.stdin)
    real_out = sys.stdout
    sys.stdout = sys.stderr          # a cell's output must not corrupt the result channel
    import _restricted
    import restricted

    def reply(d):
        real_out.write(json.dumps(d))
        real_out.flush()

    _restricted.reset()
    restricted.load_manifests(msg["data"])
    ns = _world(msg["env"], restricted)
    if ns is None:
        return reply({"kind": "skip", "reason": f"unknown env {msg['env']!r}"})
    if msg.get("pre"):
        try:
            restricted.run_cell(msg["pre"], ns)
        except BaseException as e:
            return reply({"kind": "error", "reason": f"pre raised {type(e).__name__}: {e}"})
    before = _snapshot(ns) if msg.get("leaves_nothing") else None
    out = {"kind": "ran"}
    try:
        v = restricted.run_cell(msg["py"], ns)
        out["returned"] = True
        out["value"] = _plain(v)
    except BaseException as e:
        out["returned"] = False
        out["exc"] = type(e).__name__
        out["exc_module"] = type(e).__module__
        out["not_in_dialect"] = isinstance(e, _restricted.NotInDialect)
        out["msg"] = str(e)
    if before is not None:
        out["leaked"] = _diff(before, _snapshot(ns))
    reply(out)


# --- parent ---

def _short(s, n=140):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 3] + "..."


def _judge(vec, out):
    """(status, reason) from a child's reply."""
    if out["kind"] == "skip":
        return "SKIP", out["reason"]
    if out["kind"] == "error":
        return "ERROR", out["reason"]
    expect = vec["expect"]
    ended = (f"returned {json.dumps(out['value'])}" if out["returned"]
             else f"{out['exc']}: {out['msg']}")
    leak = out.get("leaked")
    if expect == "refused":
        ok = not out["returned"] and out["not_in_dialect"]
        status, reason = ("PASS", f"NotInDialect: {out['msg']}") if ok else ("FAIL", f"expected NotInDialect, got {ended}")
    elif expect == "halts":
        status, reason = "PASS", ended
    elif isinstance(expect, dict) and "value" in expect:
        if out["returned"] and out["value"] == expect["value"] and type(out["value"]) is type(expect["value"]) \
                or (out["returned"] and out["value"] == expect["value"] and isinstance(out["value"], (int, float))
                    and not isinstance(out["value"], bool) and not isinstance(expect["value"], bool)):
            status, reason = "PASS", ended
        else:
            status, reason = "FAIL", f"expected {json.dumps(expect['value'])}, got {ended}"
    else:
        return "ERROR", f"unknown expect {expect!r}"
    if vec.get("leaves_nothing"):
        if leak:
            status, reason = "FAIL", f"state leaked ({', '.join(leak)}); {reason}"
        elif status == "PASS":
            reason += " [namespace unchanged]"
    return status, reason


def _run_one(genre, vec, args):
    row = {"genre": genre, "name": vec["name"], "env": vec["env"], "expect": vec["expect"]}
    if "budget" in vec:
        row["budget"] = vec["budget"]
    msg = {"data": os.path.abspath(args.data), "env": vec["env"], "py": vec["py"],
           "pre": (vec.get("pre") or {}).get("py"), "leaves_nothing": bool(vec.get("leaves_nothing"))}
    if vec.get("pre") and not msg["pre"]:
        msg["pre"] = None
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child"], input=json.dumps(msg),
                           capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        row.update(status="FAIL", reason="hung (timeout)")
    else:
        if p.returncode != 0 or not p.stdout.strip():
            sig = f"signal {-p.returncode}" if p.returncode < 0 else f"exit {p.returncode}"
            tail = p.stderr.strip().splitlines()[-1:] or [""]
            row.update(status="FAIL", reason=f"child crashed ({sig}) {tail[0]}")
        else:
            try:
                row["status"], row["reason"] = _judge(vec, json.loads(p.stdout))
            except Exception as e:
                row.update(status="ERROR", reason=f"bad child reply: {e}: {p.stdout[:80]!r}")
    row["seconds"] = round(time.time() - t0, 2)
    if "budget" in vec:
        row["reason"] = f"[budget {vec['budget']} not implemented] " + row["reason"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--vectors")
    ap.add_argument("--data")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--timeout", type=float, default=10)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    if args.child:
        return _child()
    if not args.vectors or not args.data:
        ap.error("--vectors and --data are required")

    import _restricted
    build = _restricted.build()
    if build != BUILD_OFF:
        bad = {k: v for k, v in build.items() if v}
        print(f"build gate FAILED: this interpreter is not the patched one with specialization, tier 2 and the JIT off "
              f"(_restricted.build() = {build}; on: {bad or 'build() missing keys'}). Refusing to run.", file=sys.stderr)
        return 2

    t0 = time.time()
    jobs, ignored = [], {}
    for g in GENRES:
        path = os.path.join(args.vectors, g + ".json")
        if not os.path.exists(path):
            print(f"note: {path} missing", file=sys.stderr)
            continue
        vecs = json.load(open(path))
        ignored[g] = sum(1 for v in vecs if "py" not in v)
        jobs += [(g, v) for v in vecs if "py" in v]
    with ThreadPoolExecutor(args.jobs) as ex:
        rows = list(ex.map(lambda gv: _run_one(gv[0], gv[1], args), jobs))

    w = max([len(r["name"]) for r in rows] + [4])
    w = min(w, 50)
    for r in rows:
        print(f"{r['genre']:<9} {r['name'][:w]:<{w}}  {r['status']:<5} {_short(r['reason'])}")
    print()
    total = {}
    for g in GENRES:
        c = {}
        for r in rows:
            if r["genre"] == g:
                c[r["status"]] = c.get(r["status"], 0) + 1
                total[r["status"]] = total.get(r["status"], 0) + 1
        if c or g in ignored:
            print(f"{g:<9} " + " ".join(f"{k}={c.get(k, 0)}" for k in ("PASS", "FAIL", "ERROR", "SKIP"))
                  + f"  (js-only ignored: {ignored.get(g, 0)})")
    print(f"{'total':<9} " + " ".join(f"{k}={total.get(k, 0)}" for k in ("PASS", "FAIL", "ERROR", "SKIP"))
          + f"  in {time.time() - t0:.1f}s")
    with open(args.out, "w") as f:
        json.dump({"build": build, "seconds": round(time.time() - t0, 1), "results": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
