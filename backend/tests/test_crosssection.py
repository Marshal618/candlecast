import numpy as np
import pandas as pd
import pytest

from app.services.crosssection import _ranks, _rowwise_corr, bucket_weights, run_crosssection, sector_report

HORIZON = 5


def _universe(n_symbols=6, bars=400, seed=0, crypto=False):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=bars, tz="America/New_York")
    frames = {}
    for k in range(n_symbols):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, bars)))
        frames[f"S{k}"] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6, "amount": 1e8},
            index=idx,
        )
    if crypto:
        cidx = pd.date_range(idx[0].date(), idx[-1].date(), freq="D", tz="UTC")
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(cidx))))
        frames["BTC"] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1.0, "amount": 1.0}, index=cidx
        )
    return frames


class PeekingForecaster:
    """Returns each context's true future close (sign=1) or its mirror (sign=-1), to test the scoring."""

    def __init__(self, frames, sign=1.0):
        self.frames, self.sign, self.contexts = frames, sign, []

    def sample_paths(self, contexts, future_index, model, samples, temperature, top_p):
        self.contexts += contexts
        out = np.zeros((len(contexts), samples, len(future_index[0]), 6))
        for i, c in enumerate(contexts):
            f = next(f for f in self.frames.values() if c.index[-1] in f.index and f["close"].loc[c.index[-1]] == c["close"].iloc[-1])
            p = f.index.get_loc(c.index[-1])
            r = f["close"].iloc[p + HORIZON] / f["close"].iloc[p] - 1
            out[i, :, :, 3] = c["close"].iloc[-1] * (1 + self.sign * r)
        return out


def _run(frames, fc, **kw):
    args = dict(interval="1d", models=["mini"], lookback=64, horizon=HORIZON, periods=12, buckets=3,
                samples=2, temperature=1.0, top_p=0.9, cost_bps=0.0)
    return run_crosssection(fc, frames, **{**args, **kw})


def test_perfect_ranking_scores_rank_ic_one_and_monotone_buckets():
    frames = _universe()
    res = _run(frames, PeekingForecaster(frames))
    sig = res["signals"][0]
    assert sig["name"] == "mini" and sig["kind"] == "model"
    assert sig["rank_ic"]["mean"] == pytest.approx(1.0)
    b = sig["bucket_returns"]
    assert b[0] < b[1] < b[2]
    assert sig["long_short"]["hit_rate"] == 1.0
    assert {s["name"] for s in res["signals"]} == {"mini", "momentum", "reversal"}


def test_inverted_ranking_scores_rank_ic_minus_one():
    frames = _universe()
    sig = _run(frames, PeekingForecaster(frames, sign=-1.0))["signals"][0]
    assert sig["rank_ic"]["mean"] == pytest.approx(-1.0)
    assert sig["long_short"]["net_mean"] < 0


def test_contexts_end_at_rebalance_dates_and_never_see_the_future():
    frames = _universe()
    fc = PeekingForecaster(frames)
    res = _run(frames, fc)
    origins = {pd.Timestamp(t, unit="s").date() for t in res["origin_times"]}
    assert len(fc.contexts) == res["periods"] * len(frames)
    assert all(len(c) == 64 for c in fc.contexts)
    assert {c.index[-1].date() for c in fc.contexts} == origins
    # Non-overlapping: rebalances are exactly `horizon` trading days apart
    days = pd.bdate_range(min(origins), max(origins))
    assert len(days) == (res["periods"] - 1) * HORIZON + 1


def test_costs_reduce_long_short_returns():
    frames = _universe()
    free = _run(frames, PeekingForecaster(frames))["signals"][0]["long_short"]
    paid = _run(frames, PeekingForecaster(frames), cost_bps=50)["signals"][0]["long_short"]
    assert paid["net_mean"] < free["net_mean"]
    assert paid["gross_mean"] == pytest.approx(free["gross_mean"])


def test_mixed_calendars_rebalance_on_shared_days():
    frames = _universe(crypto=True)
    res = _run(frames, PeekingForecaster(frames))
    assert "BTC" in res["symbols"]
    assert all(pd.Timestamp(t, unit="s").weekday() < 5 for t in res["origin_times"])


def test_bucket_weights_are_dollar_neutral():
    score = np.random.default_rng(1).normal(size=(10, 7))
    b, w = bucket_weights(score, 3)
    assert np.allclose(w.sum(axis=1), 0)
    assert np.allclose(np.abs(w).sum(axis=1), 2)
    assert all(set(row) == {0, 1, 2} for row in b)


def test_too_many_buckets_or_too_little_history_is_rejected():
    frames = _universe(n_symbols=4)
    with pytest.raises(ValueError, match="buckets"):
        _run(frames, PeekingForecaster(frames), buckets=5)
    short = _universe(n_symbols=4, bars=80)
    with pytest.raises(ValueError, match="rebalance dates"):
        _run(short, PeekingForecaster(short))


def test_residual_strips_out_a_factor_the_model_copies():
    frames = _universe()

    class ReversalCopycat(PeekingForecaster):
        def sample_paths(self, contexts, future_index, model, samples, temperature, top_p):
            out = np.zeros((len(contexts), samples, len(future_index[0]), 6))
            for i, c in enumerate(contexts):
                past = c["close"].iloc[-1] / c["close"].iloc[-1 - HORIZON] - 1
                out[i, :, :, 3] = c["close"].iloc[-1] * (1 - past)
            return out

    sig = _run(frames, ReversalCopycat(frames))["signals"][0]
    assert sig["corr"]["reversal"] == pytest.approx(1.0)
    assert sig["rank_ic_residual"]["mean"] == 0.0


def test_recent_listing_is_dropped_instead_of_shrinking_the_study():
    frames = _universe(n_symbols=5)
    frames["NEW"] = frames["S0"].iloc[-90:].copy()  # listed 90 bars ago
    res = _run(frames, PeekingForecaster(frames), periods=40)
    assert res["dropped"] == ["NEW"] and "NEW" not in res["symbols"]
    assert res["periods"] == 40


def test_interrupted_run_resumes_from_cached_chunks(tmp_path, monkeypatch):
    from app.services import crosssection
    from app.services.crosssection import ScoreCache

    monkeypatch.setattr(crosssection, "CHUNK", 10)
    frames = _universe(n_symbols=6)
    cache = ScoreCache(tmp_path / "r.sqlite3")

    class Crashes(PeekingForecaster):
        def sample_paths(self, contexts, *a):
            if len(self.contexts) >= 30:
                raise RuntimeError("killed")
            return super().sample_paths(contexts, *a)

    with pytest.raises(RuntimeError):
        _run(frames, Crashes(frames), cache=cache)
    assert crosssection.progress.snapshot()["total"] == 0  # cleared even on failure
    fc = PeekingForecaster(frames)
    _run(frames, fc, cache=cache)
    assert len(fc.contexts) == 12 * 6 - 30  # only the chunks that never finished


def _sector_universe():
    """Two sectors whose members share a strong common move, plus a little idiosyncratic noise."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2022-01-03", periods=400, tz="America/New_York")
    common = {g: rng.normal(0, 0.03, len(idx)) for g in "AB"}
    frames, sectors = {}, {}
    for k in range(8):
        g = "AB"[k % 2]
        close = 100 * np.exp(np.cumsum(common[g] + rng.normal(0, 0.005, len(idx))))
        frames[f"S{k}"] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1e6, "amount": 1e8}, index=idx
        )
        sectors[f"S{k}"] = f"Sector {g}"
    return frames, sectors


def test_sector_bet_scores_well_overall_but_zero_within_sectors():
    rng = np.random.default_rng(3)
    sectors = np.array(["A"] * 6 + ["B"] * 6)
    move = rng.normal(0, 0.05, (30, 2))
    fwd = move[:, (sectors == "B").astype(int)] + rng.normal(0, 0.005, (30, 12))
    bet = move[:, (sectors == "B").astype(int)]  # knows each sector's move, nothing within it
    rep = sector_report(bet, fwd, sectors, buckets=2, cost_bps=0, periods_per_year=12)
    assert _rowwise_corr(_ranks(bet), _ranks(fwd)).mean() > 0.5
    assert rep["rank_ic"]["mean"] == 0.0  # constant within each sector: no ranking left
    assert set(rep["by_sector"]) == {"A", "B"} and abs(rep["by_sector"]["A"]["mean"]) < 0.2
    # The plain long-short is all-in on one sector each period, net zero on average only by luck
    assert sum(rep["exposure"].values()) == pytest.approx(0)


def test_research_reports_sectors_when_symbols_are_classified():
    frames, sectors = _sector_universe()
    res = _run(frames, PeekingForecaster(frames), sectors=sectors, buckets=2, periods=20)
    sig = res["signals"][0]
    assert res["sectors"] == {"Sector A": 4, "Sector B": 4}
    assert sig["sector"]["rank_ic"]["mean"] == pytest.approx(1.0)
    assert set(sig["sector"]["exposure"]) == {"Sector A", "Sector B"}


def test_sector_checks_need_classified_symbols():
    frames = _universe()
    res = _run(frames, PeekingForecaster(frames), sectors={"S0": "Tech"})
    assert res["sectors"] is None and "sector" not in res["signals"][0]


def test_breakeven_cost_zeroes_the_net_edge_and_decay_starts_at_rank_ic():
    frames = _universe()
    sig = _run(frames, PeekingForecaster(frames))["signals"][0]
    be = sig["long_short"]["breakeven_bps"]
    assert be > 0
    at_be = _run(frames, PeekingForecaster(frames), cost_bps=be)["signals"][0]
    assert at_be["long_short"]["net_mean"] == pytest.approx(0, abs=1e-12)
    assert sig["ic_decay"][0] == pytest.approx(sig["rank_ic"]["mean"])
    assert len(sig["ic_decay"]) == 4 and -1 <= sig["rank_autocorr"] <= 1
    inverted = _run(frames, PeekingForecaster(frames, sign=-1.0))["signals"][0]
    assert inverted["long_short"]["breakeven_bps"] is None
