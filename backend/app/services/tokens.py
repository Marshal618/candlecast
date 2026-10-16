"""Post-hoc analysis of how Kronos tokenizes and reads candles.

Kronos is an LLM over candles: a BSQ tokenizer turns each z-scored OHLCV bar into a coarse
s1 token and a fine s2 token (1024 each), and a causal transformer predicts the next pair.
That makes standard LLM inspection tools applicable:

- surprisal: -log2 p(actual token | past), one teacher-forced pass (attention is causal)
- next-token distribution: entropy and top candidates for the bar after the window
- reconstruction: decode coarse-only vs coarse+fine tokens back to prices
- token semantics: group historical bars by s1 token and measure which candle properties the
  token explains (eta^2), plus a per-token prototype (mean return, P(up), level in window)
- attribution: occlude blocks of bars (flatten them) and measure how the next-token
  distribution moves; blocks rather than single bars, since lagged prices are collinear

All of it is deterministic: no sampling.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model.kronos import calc_time_stamps

PRICE = ["open", "high", "low", "close"]
COLS = PRICE + ["volume", "amount"]
SEMANTIC_FEATURES = ["level", "volume", "return", "body", "range"]
TOP_K = 5
MAX_HISTORY_CHUNKS = 12


def _prepare(df: pd.DataFrame, clip: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror KronosPredictor's preprocessing: amount column, per-window z-score, clip."""
    df = df.copy()
    df["amount"] = df["volume"] * df[PRICE].mean(axis=1)
    x = df[COLS].to_numpy(np.float32)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    z = np.clip((x - mu) / (sd + 1e-5), -clip, clip)
    stamps = calc_time_stamps(pd.Series(df.index)).to_numpy(np.float32)
    return z, stamps, mu, sd


def _bar_features(df: pd.DataFrame, z: np.ndarray) -> np.ndarray:
    """(bars, 5): close level z, volume z, return, body, range (last three as % of prev close)."""
    c = df["close"].to_numpy()
    prev = np.concatenate([[df["open"].iloc[0]], c[:-1]])
    o, h, lo = df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    return np.column_stack([z[:, 3], z[:, 4], c / prev - 1, (c - o) / prev, (h - lo) / prev])


def _eta2(values: np.ndarray, groups: np.ndarray) -> float:
    """Share of a feature's variance explained by token identity (one-way ANOVA eta^2)."""
    total = np.sum((values - values.mean()) ** 2)
    if total == 0:
        return 0.0
    between = 0.0
    for g in np.unique(groups):
        v = values[groups == g]
        between += len(v) * (v.mean() - values.mean()) ** 2
    return float(between / total)


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _coarse_decode(tok, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """Decode from s1 bits only, via the tokenizer's coarse path (see KronosTokenizer.forward)."""
    bits = tok.indices_to_bits([s1, s2], half=True)[..., : tok.s1_bits]
    z = tok.post_quant_embed_pre(bits)
    for layer in tok.decoder:
        z = layer(z)
    return tok.head(z)


def _next_s1_probs(model, s1, s2, stamps) -> torch.Tensor:
    logits, _ = model.decode_s1(s1, s2, stamps)
    return F.softmax(logits[:, -1, :].float(), dim=-1)


def _occlude(ctx: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Replace bars [start, end) with flat candles at the preceding close and median volume."""
    out = ctx.copy()
    level = ctx["close"].iloc[start - 1] if start > 0 else ctx["open"].iloc[0]
    for col in PRICE:
        out.iloc[start:end, out.columns.get_loc(col)] = level
    out.iloc[start:end, out.columns.get_loc("volume")] = float(ctx["volume"].median())
    return out


@torch.no_grad()
def inspect(predictor, ctx: pd.DataFrame, history: pd.DataFrame, segments: int = 16) -> dict:
    """`ctx` is the window Kronos would forecast from; `history` (ending with ctx) gives token semantics."""
    tok, model, dev, clip = predictor.tokenizer, predictor.model, predictor.device, predictor.clip
    L = len(ctx)

    def tensors(z, st):
        return torch.from_numpy(z)[None].to(dev), torch.from_numpy(st)[None].to(dev)

    z, st, mu, sd = _prepare(ctx, clip)
    x, stamps = tensors(z, st)
    s1, s2 = tok.encode(x, half=True)

    # Reconstruction fidelity, in price space
    def close_of(recon):
        return recon[0, :, 3].float().cpu().numpy() * (sd[3] + 1e-5) + mu[3]

    actual_close = ctx["close"].to_numpy()
    recon_full = close_of(tok.decode([s1, s2], half=True))
    recon_coarse = close_of(_coarse_decode(tok, s1, s2))

    # Teacher-forced surprisal: logits at t score the actual token at t+1
    s1_logits, hidden = model.decode_s1(s1, s2, stamps)
    lp1 = F.log_softmax(s1_logits.float(), dim=-1)
    s1_next = torch.cat([s1[:, 1:], s1[:, -1:]], dim=1)
    lp2 = F.log_softmax(model.decode_s2(hidden, s1_next).float(), dim=-1)
    ln2 = np.log(2)
    sur1 = (-lp1[0, :-1].gather(-1, s1[0, 1:, None]).squeeze(-1) / ln2).cpu().numpy()
    sur2 = (-lp2[0, :-1].gather(-1, s2[0, 1:, None]).squeeze(-1) / ln2).cpu().numpy()
    probs = lp1[0, -1].exp().cpu().numpy()

    # Token semantics from history, chunked the way the model sees windows (each z-scored alone)
    hist = history.iloc[-(MAX_HISTORY_CHUNKS * L) :]
    chunks = [hist.iloc[i : i + L] for i in range(len(hist) % L, len(hist) - L + 1, L)]
    prepared = [_prepare(c, clip) for c in chunks]
    hz = torch.from_numpy(np.stack([p[0] for p in prepared])).to(dev)
    h1, _ = tok.encode(hz, half=True)
    hs1 = h1.cpu().numpy().reshape(-1)
    feats = np.concatenate([_bar_features(c, p[0]) for c, p in zip(chunks, prepared)])
    # Drop each chunk's first bar: its "previous close" is outside the chunk
    keep = np.ones(len(hs1), bool)
    keep[::L] = False
    hs1, feats = hs1[keep], feats[keep]

    n_groups = len(np.unique(hs1))
    semantics = {
        "bars": int(len(hs1)),
        "tokens_used": n_groups,
        # eta^2 expected by chance with this many groups; real signal has to clear it
        "chance": float((n_groups - 1) / max(len(hs1) - 1, 1)),
        "explained": {name: _eta2(feats[:, i], hs1) for i, name in enumerate(SEMANTIC_FEATURES)},
    }
    protos: dict[int, dict] = {}
    for t in np.unique(hs1):
        f = feats[hs1 == t]
        protos[int(t)] = {
            "count": int(len(f)),
            "mean_return": float(f[:, 2].mean()),
            "p_up": float((f[:, 2] > 0).mean()),
            "level": float(f[:, 0].mean()),
            "volume": float(f[:, 1].mean()),
        }

    def implied(p: np.ndarray) -> tuple[float | None, float | None, float]:
        """Next-bar return and P(up) implied by the token distribution and token prototypes."""
        ids = np.array(list(protos))
        mass = p[ids]
        if mass.sum() == 0:
            return None, None, 0.0
        w = mass / mass.sum()
        ret = float(np.dot(w, [protos[i]["mean_return"] for i in ids]))
        up = float(np.dot(w, [protos[i]["p_up"] for i in ids]))
        return ret, up, float(mass.sum())

    base_ret, base_up, coverage = implied(probs)
    top = np.argsort(probs)[::-1][:TOP_K]

    # Occlusion attribution over contiguous blocks
    segments = max(1, min(segments, L // 4))
    edges = np.linspace(0, L, segments + 1).astype(int)
    occluded = [_prepare(_occlude(ctx, a, b), clip) for a, b in zip(edges[:-1], edges[1:])]
    oz = torch.from_numpy(np.stack([o[0] for o in occluded])).to(dev)
    ost = torch.from_numpy(np.stack([o[1] for o in occluded])).to(dev)
    o1, o2 = tok.encode(oz, half=True)
    oprobs = _next_s1_probs(model, o1, o2, ost).cpu().numpy()
    base = np.clip(probs, 1e-12, None)
    attribution = []
    for (a, b), q in zip(zip(edges[:-1], edges[1:]), oprobs):
        kl = float(np.sum(base * (np.log2(base) - np.log2(np.clip(q, 1e-12, None)))))
        r, _, _ = implied(q)
        attribution.append(
            {
                "start": int(ctx.index[a].timestamp()),
                "end": int(ctx.index[b - 1].timestamp()),
                "kl_bits": kl,
                # Removing this block moved the implied next-bar return by -delta, so the block
                # itself contributes +delta
                "contribution": None if r is None or base_ret is None else base_ret - r,
            }
        )

    times = [int(t.timestamp()) for t in ctx.index]
    s1_np, s2_np = s1[0].cpu().numpy(), s2[0].cpu().numpy()
    mean_sur = float(sur1.mean() + sur2.mean())
    return {
        "lookback_used": L,
        "vocab": {"s1": 2**tok.s1_bits, "s2": 2**tok.s2_bits},
        "bars": [
            {
                "time": times[i],
                "s1": int(s1_np[i]),
                "s2": int(s2_np[i]),
                "surprisal": None if i == 0 else float(sur1[i - 1] + sur2[i - 1]),
                "recon_full": float(recon_full[i]),
                "recon_coarse": float(recon_coarse[i]),
            }
            for i in range(L)
        ],
        "summary": {
            "unique_s1": int(len(np.unique(s1_np))),
            "unique_s2": int(len(np.unique(s2_np))),
            "usage_entropy_bits": _entropy_bits(np.bincount(s1_np) / L),
            "mean_surprisal_bits": mean_sur,
            "perplexity": float(2**mean_sur),
            "recon_mae_full_pct": float(np.mean(np.abs(recon_full - actual_close) / actual_close)),
            "recon_mae_coarse_pct": float(np.mean(np.abs(recon_coarse - actual_close) / actual_close)),
        },
        "next": {
            "entropy_bits": _entropy_bits(probs),
            "max_entropy_bits": float(tok.s1_bits),
            "implied_return": base_ret,
            "implied_p_up": base_up,
            "coverage": coverage,
            "top": [{"s1": int(t), "p": float(probs[t]), "prototype": protos.get(int(t))} for t in top],
        },
        "semantics": semantics,
        "attribution": attribution,
    }
