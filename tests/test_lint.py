"""Lint and type-checking tests.

These run ruff and ty against the source tree so that CI catches violations
even without a pre-commit hook installed.

Fix commands:
  ruff check:   uv run ruff check --fix src/ tests/
  ruff format:  uv run ruff format src/ tests/
  ty:           uv run ty check src/
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def test_ruff_check():
    result = _run([sys.executable, "-m", "ruff", "check", "src/", "tests/"])
    assert result.returncode == 0, (
        "ruff check found violations. Fix with:\n"
        "  uv run ruff check --fix src/ tests/\n\n" + result.stdout
    )


def test_ruff_format():
    result = _run([sys.executable, "-m", "ruff", "format", "--check", "src/", "tests/"])
    assert result.returncode == 0, (
        "ruff format found unformatted files. Fix with:\n"
        "  uv run ruff format src/ tests/\n\n" + result.stdout
    )


def test_ty():
    result = _run([sys.executable, "-m", "ty", "check", "src/"])
    assert result.returncode == 0, (
        "ty found type errors. Fix with:\n"
        "  uv run ty check src/\n\n" + result.stdout + result.stderr
    )
