"""Technical-indicator computation + look-ahead validation.

Ported from TradeSim (`tradesim_handoff_seed/src/tradesim/data/indicators.py`) — the
clean, causal feature pipeline that produced the 71-column reference schema, including
the divergence and S/R features that map onto our residual/sweep edge thesis (vault
"Trading Strategies" / "Simulated Market").

The look-ahead guard (`validate_no_lookahead`) is the non-negotiable discipline: it
recomputes indicators on truncated history and asserts the value at row *i* is unchanged,
proving features are causal — any new BSC feature must pass it before entering a dataset.

Adapted from source: `IndicatorConfig` inlined as a stdlib dataclass (no pydantic),
`loguru` → stdlib `logging`, and the look-ahead guard tolerates a missing `quote_volume`
column (our GeckoTerminal OHLCV has none). Uses the `ta` library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

logger = logging.getLogger(__name__)


@dataclass
class IndicatorConfig:
    """Indicator periods (TradeSim converged defaults)."""

    sma_periods: list[int] = field(default_factory=lambda: [20, 50])
    ema_periods: list[int] = field(default_factory=lambda: [12, 26])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth_k: int = 3
    willr_period: int = 14
    roc_period: int = 12
    bbands_period: int = 20
    bbands_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    hvol_period: int = 20
    vol_sma_period: int = 20
    return_periods: list[int] = field(default_factory=lambda: [1, 5, 15, 60])


class IndicatorComputer:
    """Computes all technical indicators on OHLCV data."""

    def __init__(self, config: IndicatorConfig | None = None):
        self.config = config or IndicatorConfig()

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append all configured indicators to an OHLCV frame (original preserved).

        Args:
            df: columns [timestamp, open, high, low, close, volume].
        """
        df = df.copy()
        cfg = self.config

        if len(df) < 2:
            logger.warning("Not enough data to compute indicators")
            return df

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # --- Trend ---
        for period in cfg.sma_periods:
            df[f"sma_{period}"] = SMAIndicator(close=close, window=period).sma_indicator()
        for period in cfg.ema_periods:
            df[f"ema_{period}"] = EMAIndicator(close=close, window=period).ema_indicator()

        macd = MACD(close=close, window_fast=cfg.macd_fast, window_slow=cfg.macd_slow,
                    window_sign=cfg.macd_signal)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        df["adx"] = ADXIndicator(high=high, low=low, close=close, window=cfg.adx_period).adx()

        # --- Momentum ---
        df["rsi_14"] = RSIIndicator(close=close, window=cfg.rsi_period).rsi()
        stoch = StochasticOscillator(high=high, low=low, close=close,
                                     window=cfg.stoch_k, smooth_window=cfg.stoch_d)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        df["willr"] = WilliamsRIndicator(high=high, low=low, close=close,
                                         lbp=cfg.willr_period).williams_r()
        df["roc_12"] = ROCIndicator(close=close, window=cfg.roc_period).roc()

        # --- Volatility ---
        bb = BollingerBands(close=close, window=cfg.bbands_period, window_dev=cfg.bbands_std)
        bb_upper, bb_lower, bb_mid = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
        bb_range = bb_upper - bb_lower
        df["bb_pctb"] = (close - bb_lower) / bb_range.replace(0, np.nan)
        df["bb_width"] = bb_range / bb_mid.replace(0, np.nan)
        df["atr_14"] = AverageTrueRange(high=high, low=low, close=close,
                                        window=cfg.atr_period).average_true_range()
        log_returns = np.log(close / close.shift(1))
        df["hvol_20"] = log_returns.rolling(window=cfg.hvol_period).std() * np.sqrt(cfg.hvol_period)

        # --- Volume ---
        df["obv"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        df["vwap"] = self._compute_vwap(df)
        vol_sma = SMAIndicator(close=volume, window=cfg.vol_sma_period).sma_indicator()
        df["vol_sma_ratio"] = volume / vol_sma.replace(0, np.nan)

        # --- Derived / dynamics / regime ---
        df = self._compute_custom_features(df)
        df = self._compute_divergence_features(df)
        df = self._compute_indicator_dynamics(df)
        df = self._compute_leading_signals(df)
        df = self._compute_support_resistance(df)
        df = self._compute_regime_label(df)
        return df

    def _compute_vwap(self, df: pd.DataFrame) -> pd.Series:
        """VWAP, reset daily."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_volume = typical_price * df["volume"]
        dates = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
        cum_tp_vol = tp_volume.groupby(dates).cumsum()
        cum_vol = df["volume"].groupby(dates).cumsum()
        return cum_tp_vol / cum_vol.replace(0, np.nan)

    def _compute_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        eps = 1e-10
        if "vwap" in df.columns and "atr_14" in df.columns:
            df["vwap_distance"] = (df["close"] - df["vwap"]) / df["atr_14"].replace(0, np.nan)
        candle_range = df["high"] - df["low"]
        df["candle_body_ratio"] = (df["close"] - df["open"]) / candle_range.clip(lower=eps)
        body_top = df[["open", "close"]].max(axis=1)
        body_bottom = df[["open", "close"]].min(axis=1)
        body_size = (body_top - body_bottom).clip(lower=eps)
        df["upper_wick"] = (df["high"] - body_top) / body_size
        df["lower_wick"] = (body_bottom - df["low"]) / body_size
        for period in cfg.return_periods:
            df[f"ret_{period}"] = df["close"].pct_change(periods=period)
        return df

    def _compute_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classic floor pivots from the previous hour, forward-filled to 1-minute rows."""
        eps = 1e-10
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        hourly = df.copy()
        hourly.index = ts
        hourly_ohlc = hourly.resample("1h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
        }).dropna()

        h, l, c = hourly_ohlc["high"].shift(1), hourly_ohlc["low"].shift(1), hourly_ohlc["close"].shift(1)
        pp = (h + l + c) / 3.0
        pivots = pd.DataFrame({
            "sr_pivot": pp, "sr_support1": 2.0 * pp - h, "sr_support2": pp - (h - l),
            "sr_resist1": 2.0 * pp - l, "sr_resist2": pp + (h - l),
        }, index=hourly_ohlc.index)
        pivots_1m = pivots.reindex(ts, method="ffill")
        for col in ["sr_pivot", "sr_support1", "sr_support2", "sr_resist1", "sr_resist2"]:
            df[col] = pivots_1m[col].values

        close = df["close"].values
        atr = df["atr_14"].values + eps
        s1, s2 = df["sr_support1"].values, df["sr_support2"].values
        r1, r2 = df["sr_resist1"].values, df["sr_resist2"].values

        dist_s1, dist_s2 = (close - s1) / atr, (close - s2) / atr
        df["sr_dist_support"] = np.clip(np.where(
            (dist_s1 > 0) & (dist_s2 > 0), np.minimum(dist_s1, dist_s2),
            np.where(dist_s1 > 0, dist_s1, dist_s2)), -10, 10)
        dist_r1, dist_r2 = (r1 - close) / atr, (r2 - close) / atr
        df["sr_dist_resist"] = np.clip(np.where(
            (dist_r1 > 0) & (dist_r2 > 0), np.minimum(dist_r1, dist_r2),
            np.where(dist_r1 > 0, dist_r1, dist_r2)), -10, 10)
        df["sr_position"] = np.clip((close - s1) / (r1 - s1 + eps), -0.5, 1.5)
        return df

    def _compute_regime_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label each candle 1/0/-1 (bull/neutral/bear) by SMA(50) level + slope."""
        close = df["close"]
        sma50 = df["sma_50"] if "sma_50" in df.columns else close.rolling(50).mean()
        sma_slope = self._fast_rolling_slope(sma50.values, 20)
        price_above = close > sma50
        slope_positive = pd.Series(sma_slope) > 0
        regime = np.zeros(len(df), dtype=np.int8)
        regime[(price_above.values) & (slope_positive.values)] = 1
        regime[(~price_above.values) & (~slope_positive.values)] = -1
        df["regime"] = regime
        return df

    @staticmethod
    def _fast_rolling_slope(arr: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling linear-regression slope (no scipy)."""
        out = np.full(len(arr), np.nan)
        x = np.arange(window, dtype=np.float64)
        x_mean = x.mean()
        x_var = np.sum((x - x_mean) ** 2)
        if x_var == 0:
            return out
        for i in range(window - 1, len(arr)):
            y = arr[i - window + 1: i + 1]
            if np.isnan(y).any():
                continue
            out[i] = np.sum((x - x_mean) * (y - y.mean())) / x_var
        return out

    def _compute_divergence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-vs-indicator divergence (indicator slope minus price slope, normalized)."""
        eps = 1e-10
        close = df["close"].values
        atr = df["atr_14"].values
        for w in [10, 20]:
            price_slope_norm = self._fast_rolling_slope(close, w) / (atr + eps)
            macd_hist = df["macd_hist"].values
            macd_std = pd.Series(macd_hist).rolling(w).std().values + eps
            df[f"div_macd_hist_{w}"] = self._fast_rolling_slope(macd_hist, w) / macd_std - price_slope_norm
            df[f"div_rsi_{w}"] = self._fast_rolling_slope(df["rsi_14"].values, w) / 10.0 - price_slope_norm
            obv_diff = df["obv"].diff().values
            obv_std = pd.Series(obv_diff).rolling(w).std().values + eps
            df[f"div_obv_{w}"] = self._fast_rolling_slope(np.nan_to_num(obv_diff), w) / obv_std - price_slope_norm
            df[f"div_stoch_{w}"] = self._fast_rolling_slope(df["stoch_k"].values, w) / 10.0 - price_slope_norm
            roc = df["roc_12"].values
            roc_std = pd.Series(roc).rolling(w).std().values + eps
            df[f"div_roc_{w}"] = self._fast_rolling_slope(roc, w) / roc_std - price_slope_norm
        return df

    def _compute_indicator_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicator slopes, crossover rates, volume trend."""
        eps = 1e-10
        w = 5
        macd_hist = df["macd_hist"].values
        df["macd_hist_slope"] = self._fast_rolling_slope(macd_hist, w) / (
            pd.Series(macd_hist).rolling(20).std().values + eps)
        df["rsi_slope"] = self._fast_rolling_slope(df["rsi_14"].values, w) / 10.0
        obv_diff = df["obv"].diff().values
        df["obv_slope"] = self._fast_rolling_slope(np.nan_to_num(obv_diff), w) / (
            pd.Series(obv_diff).rolling(20).std().values + eps)
        df["stoch_k_slope"] = self._fast_rolling_slope(df["stoch_k"].values, w) / 10.0
        df["bb_pctb_slope"] = self._fast_rolling_slope(np.nan_to_num(df["bb_pctb"].values), w)
        df["macd_cross_rate"] = df["macd_hist"] - df["macd_hist"].shift(1)
        df["stoch_cross"] = df["stoch_k"] - df["stoch_d"]
        vol = df["volume"].values
        vol_mean = pd.Series(vol).rolling(20).mean().values + eps
        df["vol_trend"] = self._fast_rolling_slope(vol, 10) / vol_mean
        return df

    def _compute_leading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Buy/sell pressure, pressure divergence, ATR squeeze, relative volume."""
        eps = 1e-10
        candle_range = (df["high"] - df["low"]).clip(lower=eps)
        buy_pressure = df["volume"] * (df["close"] - df["low"]) / candle_range
        sell_pressure = df["volume"] * (df["high"] - df["close"]) / candle_range
        pressure_ratio = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure + eps)
        df["pressure_ratio"] = pressure_ratio
        atr = df["atr_14"].values + eps
        pr_slope = self._fast_rolling_slope(pressure_ratio.values, 20)
        price_slope = self._fast_rolling_slope(df["close"].values, 20)
        df["div_pressure_20"] = pr_slope - price_slope / atr
        atr_slope = self._fast_rolling_slope(df["atr_14"].values, 20)
        atr_mean = pd.Series(df["atr_14"].values).rolling(20).mean().values + eps
        df["div_atr_20"] = atr_slope / atr_mean
        vol_mean_60 = pd.Series(df["volume"].values).rolling(60).mean().values + eps
        df["rvol_60"] = df["volume"].values / vol_mean_60
        return df

    def validate_no_lookahead(self, df: pd.DataFrame,
                              sample_indices: list[int] | None = None) -> bool:
        """Assert indicators at row i depend only on rows 0..i (causal).

        Recomputes indicators on `df[:i+1]` for sampled `i` and checks they match `df[i]`.
        """
        if sample_indices is None:
            n = len(df)
            sample_indices = [i for i in [n // 2, n // 2 + 100, n // 2 + 500, 3 * n // 4, n - 1] if i < n]

        base_cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
        original_cols = [c for c in base_cols if c in df.columns]  # quote_volume optional (DEX data)
        indicator_cols = [
            c for c in df.columns
            if c not in original_cols and not c.startswith("is_")
            and not c.startswith("gap_") and c != "session_id"
        ]

        all_pass = True
        for idx in sample_indices:
            partial = self.compute_all(df[original_cols].iloc[: idx + 1].copy())
            for col in indicator_cols:
                if col not in partial.columns:
                    continue
                full_val = df[col].iloc[idx]
                part_val = partial[col].iloc[-1]
                if pd.isna(full_val) and pd.isna(part_val):
                    continue
                if pd.isna(full_val) or pd.isna(part_val):
                    logger.warning(f"Lookahead FAIL idx={idx} col={col}: one is NaN")
                    all_pass = False
                    continue
                if abs(full_val - part_val) > 1e-6:
                    logger.warning(f"Lookahead FAIL idx={idx} col={col}: "
                                   f"full={full_val:.6f} partial={part_val:.6f}")
                    all_pass = False
        if all_pass:
            logger.info("Lookahead validation PASSED for all sampled indices")
        return all_pass
