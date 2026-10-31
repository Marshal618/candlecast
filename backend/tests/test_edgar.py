import numpy as np
import pytest

from app.services.edgar import extract
from app.services.fundamentals import DAY, Fundamentals, edgar_scores, known, symbol_scores
from app.services.edgar import _epoch


def _row(end, val, filed, start=None, form="10-K"):
    r = {"end": end, "val": val, "filed": filed, "form": form}
    if start:
        r["start"] = start
    return r


def _facts(gaap, dei=None):
    return {"facts": {"us-gaap": {k: {"units": {"USD": v}} for k, v in gaap.items()},
                      "dei": {k: {"units": {"shares": v}} for k, v in (dei or {}).items()}}}


def test_extract_keeps_full_years_and_derives_gross_profit():
    f = _facts({
        "Revenues": [_row("2021-12-31", 100, "2022-02-20", "2021-01-01"),
                     _row("2021-09-30", 70, "2021-11-01", "2021-01-01", "10-Q")],  # 9 months: dropped
        "CostOfRevenue": [_row("2021-12-31", 60, "2022-02-20", "2021-01-01")],
        "NetIncomeLoss": [_row("2021-12-31", 9, "2022-02-20", "2021-01-01"),
                          _row("2021-12-31", 9, "2022-03-01", "2021-01-01", "DEF 14A")],  # proxy: dropped
        "Assets": [_row("2021-12-31", 500, "2022-02-20"), _row("2021-12-31", 999, "2022-02-20", "2021-01-01")],
    })
    out = extract(f)
    assert out["gross"] == [[_epoch("2022-02-20"), _epoch("2021-12-31"), 40.0]]
    assert len(out["net_income"]) == 1
    assert out["assets"] == [[_epoch("2022-02-20"), _epoch("2021-12-31"), 500.0]]  # a duration row isn't a balance


def test_current_registrant_wins_over_predecessor_for_the_same_filing():
    new = _facts({"Assets": [_row("2026-03-31", 2, "2026-05-01")]})
    old = _facts({"Assets": [_row("2026-03-31", 1, "2026-05-01"), _row("2020-12-31", 7, "2021-02-01")]})
    assert [r[2] for r in extract(new, old)["assets"]] == [7.0, 2.0]


def test_known_uses_only_earlier_filings_and_their_latest_restatement():
    s = np.array([[10, 5, 1.0], [20, 5, 1.5], [30, 8, 2.0]])  # filed, end, value
    assert known(s, 15) == (5, 1.0)
    assert known(s, 25) == (5, 1.5)  # period 5 restated in the filing at 20
    assert known(s, 30) == (5, 1.5)  # filed *at* the cut isn't known yet
    assert known(s, 31) == (8, 2.0)
    assert known(s, 31, end=5, tol=1) == (5, 1.5)
    assert np.isnan(known(s, 5)[0])


def _company(cut):
    """A company with two years of filings, all public well before `cut`."""
    y = 365 * DAY
    end, prev = cut - 100 * DAY, cut - 100 * DAY - y
    filed = lambda e: e + 40 * DAY  # noqa: E731
    return {
        "assets": np.array([[filed(prev), prev, 800.0], [filed(end), end, 1000.0]]),
        "equity": np.array([[filed(end), end, 400.0]]),
        "gross": np.array([[filed(end), end, 300.0]]),
        "net_income": np.array([[filed(end), end, 50.0]]),
        "cfo": np.array([[filed(end), end, 80.0]]),
        # Cover-page counts are dated near the filing, so quarterly ones sit close to any date
        "shares": np.array([[cut - 385 * DAY, cut - 385 * DAY, 110.0], [cut - 20 * DAY, cut - 20 * DAY, 100.0]]),
    }


def test_accounting_signals():
    cut = 2_000_000_000
    s = edgar_scores(Fundamentals.empty().with_facts(_company(cut)), cut, shares=100.0, close=8.0)
    assert s["profitability"] == pytest.approx(0.3)
    assert s["investment"] == pytest.approx(-0.25)
    assert s["accruals"] == pytest.approx(-(50 - 80) / 1000)
    assert s["book"] == pytest.approx(400 / 800)


def test_stale_balance_sheet_scores_nothing():
    cut = 2_000_000_000
    s = edgar_scores(Fundamentals.empty().with_facts(_company(cut)), cut + 200 * DAY, shares=100.0, close=8.0)
    assert all(np.isnan(v) for v in s.values())


def test_cover_shares_are_split_adjusted_and_drive_issuance():
    cut = 2_000_000_000
    facts = _company(cut)
    split_at = cut - 300 * DAY  # a 2-for-1 between last year's cover and this year's
    f = Fundamentals.from_json({"earnings": [[cut - 30 * DAY, 1.0, 0.0]], "splits": [[split_at, 2.0]]}).with_facts(facts)
    sc = symbol_scores(f, cut, "1d", close=4.0)
    # 110 old shares are 220 on today's basis; now 100 (the cover after the split is already post-split)
    assert sc["issuance"] == pytest.approx(-np.log(100 / 220))
    assert sc["book"] == pytest.approx(400 / (100 * 4.0))
