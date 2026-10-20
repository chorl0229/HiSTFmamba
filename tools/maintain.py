"""Run the repository's local maintenance checks from any working directory."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", action="store_true", help="Skip PyTorch component tests.")
    args = parser.parse_args()
    commands = [
        ["-m", "ruff", "check", "."],
        ["-m", "ruff", "format", "--check", "."],
        ["tools/check_release.py"],
        ["tools/sync_publication.py", "--check"],
        ["-c", "from cffconvert.cli.cli import cli; cli()", "--validate"],
        ["tools/check_docs.py"],
        ["-m", "unittest", "discover", "-s", "tests", "-p", "test_maintenance.py", "-v"],
    ]
    if not args.static:
        commands.append(["-m", "unittest", "discover", "-s", "tests", "-p", "test_core.py", "-v"])
    failures = []
    for command in commands:
        print(f"\n> python {' '.join(command)}", flush=True)
        if subprocess.run([sys.executable, *command], cwd=ROOT).returncode:
            failures.append(command)
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed.")
    print("\nAll requested maintenance checks passed.")


if __name__ == "__main__":
    main()
