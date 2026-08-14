"""
selftest_io.py

Output-path guard for the synthetic self-tests.

Every script in this repository runs a self-contained synthetic self-test when
its production input argument is omitted. Those self-tests used to honour the
same `--out` default as a real run, so a bare `python projection_analysis.py`
would silently overwrite the production `projection_flow.parquet` with synthetic
test output. The large pipeline outputs are gitignored and regenerable only from
`chess_scape_output/`, so that overwrite is unrecoverable on a machine that does
not hold the CHESS-SCAPE data.

Self-test writes are therefore redirected here into `selftest/`, keeping the
same filename. Production runs are untouched: `redirect()` is only ever called
on the self-test branch.
"""
from __future__ import annotations

from pathlib import Path

SELFTEST_DIR = Path("selftest")


def redirect(path: str | Path, selftest: bool) -> Path:
    """Return `path` unchanged for a production run, or the same filename inside
    `selftest/` when `selftest` is True."""
    p = Path(path)
    if not selftest:
        return p
    SELFTEST_DIR.mkdir(exist_ok=True)
    return SELFTEST_DIR / p.name


def announce(paths) -> None:
    """One-line reminder of where self-test output went."""
    names = ", ".join(sorted({str(Path(p).parent) for p in paths}))
    print(f"\n[self-test] synthetic output written under {names}/ -- these are "
          f"NOT results and must not be pasted into the manuscript.")
