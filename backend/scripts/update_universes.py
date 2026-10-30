"""Refresh the index universes in app/universes.json from Wikipedia's constituent tables.

    cd backend && .venv/bin/python scripts/update_universes.py

These are *today's* members. Backtests on them carry survivorship bias: names that were dropped
or delisted along the way are missing, and today's members are partly members because they did well.
"""

import io
import json
import sys
import time
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "app" / "universes.json"
SOURCES = {
    "S&P 500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
    "S&P 400 mid caps": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol"),
    "S&P 600 small caps": ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Symbol"),
}


def _get(url: str, tries: int = 4) -> str:
    for k in range(tries):
        try:
            return urlopen(Request(url, headers={"User-Agent": "candlecast-universe-refresh"}), timeout=120).read().decode()
        except (TimeoutError, OSError):
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise AssertionError


def fetch(url: str, col: str) -> list[str]:
    html = _get(url)
    for t in pd.read_html(io.StringIO(html)):
        if col in t.columns and len(t) > 50:
            # Yahoo writes share classes with a dash (BRK-B), Wikipedia with a dot (BRK.B)
            return sorted({str(s).strip().replace(".", "-") for s in t[col] if str(s).strip()})
    raise ValueError(f"No '{col}' table on {url}")


def main() -> None:
    out = {"as_of": date.today().isoformat(), "source": "Wikipedia constituent tables", "universes": {}}
    for name, (url, col) in SOURCES.items():
        out["universes"][name] = fetch(url, col)
        print(f"{name}: {len(out['universes'][name])}", file=sys.stderr)
    OUT.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
