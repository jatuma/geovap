"""uv run python -m mapping.cli.build_store [--workers 8]"""
from __future__ import annotations

import argparse
import time

from ..cloud_store import build_store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t = time.time()
    build_store(workers=a.workers)
    print(f"done in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
