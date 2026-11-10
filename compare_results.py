"""Compare two results.json files (e.g. the committed run and your rerun).
Usage:  python compare_results.py old/results.json new/results.json [--tol 1e-9]
Prints every numeric field that differs by more than the tolerance; exit code 1 if any does."""
import json
import sys


def walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k == "runtime_s":
                continue
            if k not in a or k not in b:
                yield path + "/" + k, a.get(k, "<missing>"), b.get(k, "<missing>")
            else:
                yield from walk(a[k], b[k], path + "/" + k)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from walk(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > TOL * max(1.0, abs(a), abs(b)):
            yield path, a, b
    elif a != b:
        yield path, a, b


if __name__ == "__main__":
    args = sys.argv[1:]
    TOL = float(args[args.index("--tol") + 1]) if "--tol" in args else 1e-9
    files = [x for x in args if x.endswith(".json")]
    a, b = (json.load(open(f, encoding="utf-8")) for f in files[:2])
    diffs = list(walk(a, b))
    for p, x, y in diffs[:50]:
        print(f"{p}: {x}  vs  {y}")
    print(f"{len(diffs)} field(s) differ (relative tolerance {TOL:g})")
    sys.exit(1 if diffs else 0)
