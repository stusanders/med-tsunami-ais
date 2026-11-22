"""T27: run the equivalence tests the demo cites and record their results in outputs/.

    PYTHONPATH=src ../med-tsunami-ais/.venv/bin/python scripts/demo/record_checks.py

Writes outputs/demo_data/checks.json: one entry per test with pass/fail, the pytest summary
line, git commit and timestamp, so scenes can show a tick only for checks a file confirms.
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "demo_data" / "checks.json"

TESTS = {
    "scoring_tensor_equals_loop": "tests/test_scoring.py::test_tensor_equals_loop",
    "solver_batched_equals_reference": "tests/test_solver_equivalence.py",
}


def run(nodeid):
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", nodeid],
                       cwd=REPO, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.strip().splitlines() if ln.strip()]
    return {"nodeid": nodeid, "passed": p.returncode == 0, "summary": lines[-1] if lines else ""}


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                            text=True).stdout.strip()
    res = {"date": datetime.now().isoformat(timespec="minutes"), "commit": commit,
           "machine": "MacBook arm64 CPU (NumPy backend)", "tests": {}}
    for key, nodeid in TESTS.items():
        res["tests"][key] = run(nodeid)
        print(key, res["tests"][key]["summary"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    # one-line record that scene 7's scanner understands
    t = res["tests"]["scoring_tensor_equals_loop"]
    (OUT.parent / "checks.md").write_text(
        f"test_tensor_equals_loop {'passed' if t['passed'] else 'FAILED'} ({t['summary']}; commit {commit}, "
        f"{res['date']})\n")
    print(OUT)


if __name__ == "__main__":
    main()
