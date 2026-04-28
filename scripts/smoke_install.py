"""
Build artifact smoke test for CI.

Creates a clean virtual environment, installs the freshly built wheel, then
checks the CLI entry point and Python SDK import path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".tmp" / "smoke-venv"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def latest_wheel() -> Path:
    wheels = sorted((ROOT / "dist").glob("memk-*.whl"))
    if not wheels:
        raise RuntimeError("No memk wheel found in dist/. Run python -m build first.")
    return wheels[-1]


def main() -> None:
    if VENV.exists():
        shutil.rmtree(VENV)

    VENV.parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "venv", str(VENV)])

    python = str(venv_python())
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", str(latest_wheel())])
    run([python, "-m", "memk.cli.main", "--help"])
    run([python, "-m", "memk.cli.main", "add", "--help"])
    run([python, "-m", "memk.cli.main", "search", "--help"])
    run([
        python,
        "-c",
        (
            "from memk.sdk import MemoryKernelClient; "
            "mk = MemoryKernelClient(daemon_url='http://127.0.0.1:9'); "
            "print(type(mk).__name__)"
        ),
    ])


if __name__ == "__main__":
    main()
