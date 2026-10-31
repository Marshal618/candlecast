"""SEC EDGAR financial statements, as filed, for point-in-time accounting signals.

Every 10-K and 10-Q since about 2009 is tagged in XBRL, and the SEC's free companyfacts API
returns each reported value with the date its filing was accepted. That date is what makes these
backtestable: a rebalance only sees values from filings dated before it (see
fundamentals._cutoffs). Later filings repeat earlier periods as comparatives, sometimes restated;
the latest value known by a rebalance is used, which is what an investor could have looked up then.

Kept per company (epoch seconds of the filing date, of the period end, and the value):
- assets, equity: balance-sheet totals at each quarter end
- gross, net_income, cfo: full fiscal-year gross profit, net income and operating cash flow
  (periods of 350-380 days). Gross profit is derived as revenue minus cost of revenue where a
  company doesn't tag it. Banks and insurers report neither, so they have no profitability.
- shares: shares outstanding from each filing's cover page (as reported; fundamentals.py
  split-adjusts them with Yahoo's split history).

A company that reorganizes under a new holding company gets a new registrant ID, and its history
stays under the old one. PREDECESSORS lists the known cases so their filings are merged.

The SEC requires a contact in the User-Agent of every request (KD_SEC_CONTACT) and allows ten
requests a second; this stays under that. Each company's facts are cached for a week.
"""

import gzip
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
TTL = 7 * 86400
MAX_RATE = 8  # requests per second, under the SEC's limit of 10
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}

# Concepts in order of preference; the first one a filing tags wins for that period
INSTANT = {
    "assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}
ANNUAL = {
    "gross": ["GrossProfit"],
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "cost": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
}
KEEP = ("assets", "equity", "gross", "net_income", "cfo", "shares")
# Ticker -> earlier registrant IDs whose filings are this company's history
PREDECESSORS = {"XOM": [34088]}  # ExxonMobil Holdings Corp replaced Exxon Mobil Corp as registrant in 2026


def _epoch(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _series(gaap: dict, concepts: list[str], annual: bool, unit: str = "USD") -> dict[tuple[int, int], float]:
    """(filed, end) -> value, from the first concept that has each (filing, period)."""
    out: dict[tuple[int, int], float] = {}
    for c in concepts:
        for row in gaap.get(c, {}).get("units", {}).get(unit, []):
            if row.get("form") not in FORMS or "filed" not in row or "end" not in row:
                continue
            if annual:
                if "start" not in row:
                    continue
                days = (_epoch(row["end"]) - _epoch(row["start"])) / 86400
                if not 350 <= days <= 380:
                    continue
            elif "start" in row:
                continue
            out.setdefault((_epoch(row["filed"]), _epoch(row["end"])), float(row["val"]))
    return out


def extract(*payloads: dict) -> dict[str, list[list[float]]]:
    """Compact point-in-time series from companyfacts payloads (current registrant first, then
    predecessors): name -> [[filed, end, value]], by filed."""
    s: dict[str, dict] = {k: {} for k in [*INSTANT, *ANNUAL, "shares"]}
    for facts in payloads:
        gaap = facts.get("facts", {}).get("us-gaap", {})
        for k, v in INSTANT.items():
            s[k] = {**_series(gaap, v, annual=False), **s[k]}
        for k, v in ANNUAL.items():
            s[k] = {**_series(gaap, v, annual=True), **s[k]}
        dei = facts.get("facts", {}).get("dei", {})
        s["shares"] = {**_series(dei, ["EntityCommonStockSharesOutstanding"], annual=False, unit="shares"), **s["shares"]}
    for key, rev in s["revenue"].items():
        if key not in s["gross"] and key in s["cost"]:
            s["gross"][key] = rev - s["cost"][key]
    return {k: sorted([f, e, v] for (f, e), v in s[k].items()) for k in KEEP}


class EdgarClient:
    """Rate-limited companyfacts fetches; needs a contact for the SEC's User-Agent rule."""

    def __init__(self, contact: str):
        self.agent = f"Candlecast research {contact}"
        self._lock = threading.Lock()
        self._next = 0.0
        self._ciks: dict[str, int] | None = None

    def _get(self, url: str) -> dict:
        with self._lock:
            wait = self._next - time.monotonic()
            self._next = max(self._next, time.monotonic()) + 1 / MAX_RATE
        if wait > 0:
            time.sleep(wait)
        req = Request(url, headers={"User-Agent": self.agent, "Accept-Encoding": "gzip"})
        with urlopen(req, timeout=60) as r:
            body = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
        return json.loads(body)

    def cik(self, symbol: str) -> int | None:
        if self._ciks is None:
            # Yahoo and the SEC both write share classes with a dash (BRK-B)
            self._ciks = {v["ticker"].upper(): int(v["cik_str"]) for v in self._get(TICKERS_URL).values()}
        return self._ciks.get(symbol)

    def fetch(self, symbol: str) -> dict:
        cik = self.cik(symbol)
        if cik is None:
            return {}  # not an SEC registrant: a fund, crypto, a foreign listing, or a renamed ticker
        payloads = []
        for c in [cik, *PREDECESSORS.get(symbol, [])]:
            try:
                payloads.append(self._get(FACTS_URL.format(cik=c)))
            except OSError as e:
                if getattr(e, "code", None) != 404:  # 404: files no XBRL financials (most ETFs)
                    raise
        return extract(*payloads) if payloads else {}


class CachedEdgar:
    """Per-symbol EDGAR series, refreshed weekly. Without a contact it returns nothing."""

    def __init__(self, client: EdgarClient | None, cache_path: Path):
        self.client = client
        self.cache_path = cache_path
        self._lock = threading.Lock()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(cache_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS edgar (symbol TEXT PRIMARY KEY, fetched_at REAL, payload TEXT)")

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def get(self, symbol: str) -> dict:
        symbol = symbol.strip().upper()
        with self._lock, sqlite3.connect(self.cache_path) as conn:
            row = conn.execute("SELECT fetched_at, payload FROM edgar WHERE symbol=?", (symbol,)).fetchone()
        if row and time.time() - row[0] < TTL:
            return json.loads(row[1])
        if self.client is None:
            return json.loads(row[1]) if row else {}
        try:
            payload = self.client.fetch(symbol)
        except Exception:
            # A failed refresh keeps the stale copy
            return json.loads(row[1]) if row else {}
        with self._lock, sqlite3.connect(self.cache_path) as conn:
            conn.execute("INSERT OR REPLACE INTO edgar VALUES (?, ?, ?)", (symbol, time.time(), json.dumps(payload)))
        return payload
