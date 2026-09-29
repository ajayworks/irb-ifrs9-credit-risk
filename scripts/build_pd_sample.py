#!/usr/bin/env python
"""Build the PD development sample: annual December snapshots for every vintage.

Processes one vintage at a time (loans are independent), all sharing the dataset's
observation cut-off, and writes one Parquet file per vintage.

    python scripts/build_pd_sample.py              # every vintage found
    python scripts/build_pd_sample.py 2006 2007    # selected vintages
    python scripts/build_pd_sample.py --out DIR    # somewhere other than the default

Default output: data/processed/pd_sample/ (git-ignored - it is derived from licensed data).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hcr.config import REPO_ROOT                       # noqa: E402
from hcr.data import freddie as F                      # noqa: E402
from hcr.default_def import panel as P                 # noqa: E402
from hcr.pd import sample as S                         # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "pd_sample"


def build_vintage(year: int, cut_off: str, out_dir: Path) -> tuple[int, float]:
    db = Path(tempfile.gettempdir()) / f"hcr_sample_{year}.duckdb"
    for suffix in ("", ".wal"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    con = P.connect(db)
    try:
        F.load_freddie(con, years=[year])
        con.execute("DROP TABLE orig_raw; DROP TABLE perf_raw; CHECKPOINT")
        P.build_panel(con, "baseline", data_end_period=cut_off)
        S.build_snapshots(con)
        target = out_dir / f"vintage_{year}.parquet"
        con.execute(f"COPY pd_snapshots TO '{target}' (FORMAT PARQUET)")
        rows, rate = con.execute("SELECT COUNT(*), AVG(target) FROM pd_snapshots").fetchone()
        return rows, rate or 0.0
    finally:
        con.close()
        for suffix in ("", ".wal"):
            Path(str(db) + suffix).unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("years", nargs="*", type=int)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    found = F.vintages()
    years = args.years or found
    missing = set(years) - set(found)
    if missing:
        raise SystemExit(f"No files for vintages {sorted(missing)}")
    cut_off = F.data_end_period()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(years)} vintages, observation cut-off {cut_off}, writing to {args.out}")

    total = 0
    for y in years:
        t = time.time()
        rows, rate = build_vintage(y, cut_off, args.out)
        total += rows
        print(f"  {y}: {rows:>9,} snapshots, one-year default rate {rate:.4%} "
              f"({time.time() - t:.0f}s)", flush=True)
    print(f"Done: {total:,} snapshots.")


if __name__ == "__main__":
    main()
