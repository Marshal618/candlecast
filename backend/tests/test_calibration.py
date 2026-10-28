"""Conformal band calibration: finite-sample quantile, coverage, honest sequential check."""

import sqlite3

import numpy as np
import pytest

from app.services import calibration as cal


def _windows(n, h=3, true_sd=0.03, band_sd=0.01, seed=0):
    """Bands built for sd=1% while outcomes have sd=3%: badly overconfident, like Kronos."""
    rng = np.random.default_rng(seed)
    z = 1.2816  # p90 of a standard normal
    out = []
    for _ in range(n):
        y = rng.normal(0, true_sd, h)
        out.append({"lo_path": [-z * band_sd] * h, "hi_path": [z * band_sd] * h, "actual_path": y.tolist()})
    return out


def test_conformal_quantile_finite_sample_level():
    s = np.arange(1, 11, dtype=float)  # n=10 -> ceil(11*0.8)/10 = 0.9 quantile -> 9
    assert cal.conformal_quantile(s, 0.8) == 9.0


def test_widens_overconfident_bands_to_target_out_of_sample():
    w = _windows(400)
    c = cal.calibrate(w)
    assert c["raw_coverage"] < 0.4
    assert c["calibrated_coverage"] == pytest.approx(0.8, abs=0.06)
    assert c["q"][-1] > 0 and c["widening"] > 0


def test_narrows_underconfident_bands():
    c = cal.calibrate(_windows(400, true_sd=0.005, band_sd=0.03))
    assert c["raw_coverage"] > 0.95 and c["q"][-1] < 0
    assert c["calibrated_coverage"] == pytest.approx(0.8, abs=0.08)


def test_needs_enough_windows():
    assert cal.calibrate(_windows(cal.MIN_HISTORY)) is None


def test_apply_never_crosses_bands():
    lo, hi = cal.apply(100.0, np.array([99.0, 98.0]), np.array([101.0, 102.0]), [-5.0])
    assert (lo <= hi).all()
    lo, hi = cal.apply(100.0, np.array([99.0]), np.array([101.0]), [0.02])
    assert lo[0] == pytest.approx(97.0) and hi[0] == pytest.approx(103.0)


def test_journal_migrates_old_schema_and_stores_calibration(tmp_path):
    from app.services.journal import Journal

    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as conn:  # a journal from before calibration existed
        conn.execute("CREATE TABLE forecasts (id INTEGER PRIMARY KEY, symbol TEXT)")
    j = Journal(path, market=None)
    with sqlite3.connect(path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(forecasts)")}
    assert {"cal_p10_final", "cal_p90_final"} <= cols
    j.record_calibration("spy", "1d", "mini", 5, {"q": [0.01] * 5, "target": 0.8})
    assert j.get_calibration("SPY", "1d", "mini", 5)["q"] == [0.01] * 5
    assert j.get_calibration("SPY", "1d", "mini", 10) is None


def test_scorecard_reports_calibrated_coverage():
    from app.services.journal import scorecard

    rows = [{"last_close": 100.0, "actual_final": 100.0 + d, "pred_final": 100.0, "p_up": 0.5,
             "p10_final": 99.5, "p90_final": 100.5, "cal_p10_final": 97.0, "cal_p90_final": 103.0}
            for d in (-2, -1, 1, 2, 4)]
    c = scorecard(rows)["coverage_calibrated"]
    assert c == {"n": 5, "raw": 0.0, "calibrated": 0.8}
