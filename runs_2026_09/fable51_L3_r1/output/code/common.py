"""Shared pieces for the load-forecasting bake-off.

Everything the six model scripts have in common lives here: reading the
data, the train / validation / test split, the rolling validation origins,
the scoring functions, the colour assigned to each model, and the small
on-disk cache that lets the orchestrator assemble figures without
re-running every model.
"""

from __future__ import annotations

import json
import time
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import holidays
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN_DIR: Path = Path(__file__).resolve().parent.parent
DATA_FILE: Path = RUN_DIR / "opsd_de_load.csv"
FORECAST_DIR: Path = RUN_DIR / "forecasts"
FIGURE_DIR: Path = RUN_DIR / "figures"

TARGET: str = "DE_load_actual_entsoe_transparency"
SEED: int = 42
HORIZON: int = 168

TRAIN_START: pd.Timestamp = pd.Timestamp("2015-01-01 00:00")
TRAIN_END: pd.Timestamp = pd.Timestamp("2019-09-30 23:00")
VAL_START: pd.Timestamp = pd.Timestamp("2019-10-01 00:00")
VAL_END: pd.Timestamp = pd.Timestamp("2019-12-31 23:00")
TEST_START: pd.Timestamp = pd.Timestamp("2020-01-01 00:00")
TEST_END: pd.Timestamp = pd.Timestamp("2020-01-07 23:00")

# Seven forecast origins inside the validation window, two weeks apart.
# Every model with tunable settings is scored the same way: parameters are
# estimated on Train only, then from each origin the model forecasts the
# next 168 hours and the seven MAPEs are averaged. The last origin
# (24 December) covers the Christmas week, which is the closest thing the
# validation window has to the New Year holiday in the test week.
VALIDATION_ORIGINS: list[pd.Timestamp] = [
    VAL_START + pd.Timedelta(days=14 * k) for k in range(7)
]

# Quantile levels needed for the 80 % and 95 % intervals and the pinball
# losses. Every probabilistic model reports exactly these five.
QUANTILES: tuple[float, ...] = (0.025, 0.1, 0.5, 0.9, 0.975)

MODEL_ORDER: list[str] = [
    "naive",
    "sarima",
    "prophet",
    "lightgbm",
    "nbeats",
    "patchtst",
]
MODEL_LABELS: dict[str, str] = {
    "naive": "Seasonal naive (t-168 h)",
    "sarima": "SARIMA",
    "prophet": "Prophet (DE holidays)",
    "lightgbm": "LightGBM (features)",
    "nbeats": "N-BEATS",
    "patchtst": "TSMixer (PatchTST slot)",
}
# Fixed tab10 slots, one per model, used in every figure. Red is kept free
# because it marks the test window in the overview figure.
_TAB10 = plt.get_cmap("tab10")
MODEL_COLORS: dict[str, tuple[float, float, float, float]] = {
    "naive": _TAB10(7),
    "sarima": _TAB10(0),
    "prophet": _TAB10(1),
    "lightgbm": _TAB10(2),
    "nbeats": _TAB10(4),
    "patchtst": _TAB10(9),
}
OBSERVED_COLOR: str = "black"
TEST_WINDOW_COLOR: str = "tab:red"

TS_FIGSIZE: tuple[float, float] = (11.0, 6.0)
SQUARE_FIGSIZE: tuple[float, float] = (6.0, 6.0)
DPI: int = 300


def set_seeds(seed: int = SEED) -> None:
    """Seed every random number generator the study touches."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def apply_figure_style() -> None:
    """Matplotlib defaults shared by all figures: sans-serif, light grid."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": DPI,
            "figure.dpi": 100,
        }
    )


def load_series() -> pd.Series:
    """Read the CSV and return the load as an hourly series indexed in UTC.

    The index is made timezone-naive (still UTC) because several libraries
    downstream refuse timezone-aware indexes. The function also re-checks the
    three data guarantees stated in the task: 50,400 rows, no missing
    values, and no gaps in the hourly grid.
    """
    df = pd.read_csv(DATA_FILE, parse_dates=["utc_timestamp"])
    s = df.set_index("utc_timestamp")[TARGET].astype(float)
    s.index = s.index.tz_convert("UTC").tz_localize(None)
    s.index.name = "utc_timestamp"
    if len(s) != 50400:
        raise ValueError(f"expected 50,400 rows, found {len(s)}")
    if s.isna().any():
        raise ValueError("load series contains NaN")
    steps = s.index.to_series().diff().dropna().unique()
    if len(steps) != 1 or steps[0] != pd.Timedelta(hours=1):
        raise ValueError("load series is not a gap-free hourly grid")
    s = s.asfreq("h")
    return s


def train_slice(s: pd.Series) -> pd.Series:
    return s.loc[TRAIN_START:TRAIN_END]


def val_slice(s: pd.Series) -> pd.Series:
    return s.loc[VAL_START:VAL_END]


def train_val_slice(s: pd.Series) -> pd.Series:
    return s.loc[TRAIN_START:VAL_END]


def test_slice(s: pd.Series) -> pd.Series:
    return s.loc[TEST_START:TEST_END]


def history_before(s: pd.Series, origin: pd.Timestamp) -> pd.Series:
    """All observations strictly before `origin` (the forecast context)."""
    return s.loc[: origin - pd.Timedelta(hours=1)]


def horizon_index(origin: pd.Timestamp, horizon: int = HORIZON) -> pd.DatetimeIndex:
    return pd.date_range(origin, periods=horizon, freq="h", name="utc_timestamp")


# ---------------------------------------------------------------------------
# Calendar features derived from the timestamp alone
# ---------------------------------------------------------------------------

_DE_HOLIDAYS = holidays.Germany(years=range(2014, 2022))


def local_time_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert a naive-UTC index to Berlin local clock time (naive)."""
    return index.tz_localize("UTC").tz_convert("Europe/Berlin").tz_localize(None)


def is_german_holiday(index: pd.DatetimeIndex) -> np.ndarray:
    """1 for hours that fall on a German federal public holiday (local date)."""
    local = local_time_index(index)
    return np.array([int(d in _DE_HOLIDAYS) for d in local.date], dtype=np.int8)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour, weekday, month, weekend and holiday flags, in local Berlin time.

    Electricity demand follows the local clock (people wake at 07:00 Berlin
    time whatever UTC says), so the calendar is read in local time. This is
    still derived purely from the timestamp, no outside data is consulted.
    """
    local = local_time_index(index)
    out = pd.DataFrame(index=index)
    out["hour"] = local.hour
    out["day_of_week"] = local.dayofweek
    out["month"] = local.month
    out["is_weekend"] = (local.dayofweek >= 5).astype(np.int8)
    out["is_public_holiday_de"] = is_german_holiday(index)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def mape(actual: pd.Series, forecast: pd.Series) -> float:
    a = actual.to_numpy(dtype=float)
    f = forecast.reindex(actual.index).to_numpy(dtype=float)
    return float(np.mean(np.abs(a - f) / np.abs(a)) * 100.0)


def rmse(actual: pd.Series, forecast: pd.Series) -> float:
    a = actual.to_numpy(dtype=float)
    f = forecast.reindex(actual.index).to_numpy(dtype=float)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    a = actual.to_numpy(dtype=float)
    f = forecast.reindex(actual.index).to_numpy(dtype=float)
    return float(np.mean(np.abs(a - f)))


def per_day_mape(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    """MAPE for each UTC calendar day in the window."""
    a = actual.to_numpy(dtype=float)
    f = forecast.reindex(actual.index).to_numpy(dtype=float)
    ape = pd.Series(np.abs(a - f) / np.abs(a) * 100.0, index=actual.index)
    out = ape.groupby(ape.index.normalize()).mean()
    out.index = pd.DatetimeIndex(out.index, name="day")
    return out


def coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    a = actual.to_numpy(dtype=float)
    lo = lower.reindex(actual.index).to_numpy(dtype=float)
    hi = upper.reindex(actual.index).to_numpy(dtype=float)
    return float(np.mean((a >= lo) & (a <= hi)))


def pinball_loss(actual: pd.Series, quantile_forecast: pd.Series, q: float) -> float:
    """Average pinball (quantile) loss in MW for one quantile level."""
    a = actual.to_numpy(dtype=float)
    f = quantile_forecast.reindex(actual.index).to_numpy(dtype=float)
    diff = a - f
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def rolling_origin_mape(
    s: pd.Series,
    forecast_fn: Any,
    origins: list[pd.Timestamp] | None = None,
) -> tuple[float, list[dict[str, float | str]]]:
    """Average 168-hour-ahead MAPE over the validation origins.

    `forecast_fn(history, origin)` must return a point forecast indexed on
    the 168 hours starting at `origin`, using nothing after `origin - 1 h`.
    """
    origins = VALIDATION_ORIGINS if origins is None else origins
    rows: list[dict[str, float | str]] = []
    for origin in origins:
        idx = horizon_index(origin)
        fc = forecast_fn(history_before(s, origin), origin)
        rows.append({"origin": origin.strftime("%Y-%m-%d"), "mape_pct": mape(s.loc[idx], fc)})
    return float(np.mean([r["mape_pct"] for r in rows])), rows


# ---------------------------------------------------------------------------
# Result container and on-disk cache
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    """Everything the orchestrator needs from one model."""

    name: str
    point: pd.Series
    quantiles: dict[float, pd.Series] | None
    hyperparameters: dict[str, Any]
    runtime_seconds: float
    validation_mape_pct: float | None
    validation_table: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self) -> None:
        FORECAST_DIR.mkdir(exist_ok=True)
        frame = pd.DataFrame({"point": self.point})
        if self.quantiles is not None:
            for q, series in self.quantiles.items():
                frame[f"q{q:g}"] = series.reindex(self.point.index)
        frame.index.name = "utc_timestamp"
        frame.to_csv(FORECAST_DIR / f"{self.name}.csv", float_format="%.3f")
        meta = {
            "name": self.name,
            "hyperparameters": self.hyperparameters,
            "runtime_seconds": self.runtime_seconds,
            "validation_mape_pct": self.validation_mape_pct,
            "validation_table": self.validation_table,
            "notes": self.notes,
            "extra": self.extra,
        }
        (FORECAST_DIR / f"{self.name}.json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, name: str) -> ModelResult:
        frame = pd.read_csv(FORECAST_DIR / f"{name}.csv", parse_dates=["utc_timestamp"], index_col="utc_timestamp")
        meta = json.loads((FORECAST_DIR / f"{name}.json").read_text())
        quantiles: dict[float, pd.Series] | None = None
        qcols = [c for c in frame.columns if c.startswith("q")]
        if qcols:
            quantiles = {float(c[1:]): frame[c] for c in qcols}
        return cls(
            name=name,
            point=frame["point"],
            quantiles=quantiles,
            hyperparameters=meta["hyperparameters"],
            runtime_seconds=meta["runtime_seconds"],
            validation_mape_pct=meta["validation_mape_pct"],
            validation_table=meta["validation_table"],
            notes=meta["notes"],
            extra=meta.get("extra", {}),
        )


def sort_quantiles(quantiles: dict[float, pd.Series]) -> dict[float, pd.Series]:
    """Make the quantile curves non-crossing by sorting per time step.

    Quantile models fitted independently (or one network with several
    quantile outputs) can produce a 0.9 quantile below the 0.5 quantile at
    a few hours. Sorting the values at each hour is the standard fix.
    """
    levels = sorted(quantiles)
    stacked = np.column_stack([quantiles[q].to_numpy(dtype=float) for q in levels])
    stacked.sort(axis=1)
    index = quantiles[levels[0]].index
    return {q: pd.Series(stacked[:, i], index=index) for i, q in enumerate(levels)}


def silence_warnings() -> None:
    warnings.filterwarnings("ignore")
    import logging

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("darts").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Shared training loop for the two darts deep-learning models
# ---------------------------------------------------------------------------

# The series is divided by this constant before training so that the
# network sees numbers near 1. A plain division keeps MAPE unchanged
# (percent errors do not care about units), so the validation MAPE the
# trainer reports is the MAPE in megawatts.
DEEP_SCALE: float = 50_000.0


def run_darts_torch_model(
    name: str,
    model_cls: Any,
    fixed_kwargs: dict[str, Any],
    candidates: list[dict[str, Any]],
    s: pd.Series,
    max_epochs: int = 30,
    patience: int = 3,
    batch_size: int = 256,
) -> ModelResult:
    """Select by validation MAPE with early stopping, refit on Train+Val, forecast.

    For each candidate setting the network is trained on Train for at most
    `max_epochs` epochs; after every epoch the MAPE over all 168-hour
    windows in the validation period is measured and training stops once
    it has not improved for `patience` epochs. (That per-epoch MAPE is
    computed on random draws from the network's quantile head, so it is a
    little pessimistic; it is only used to decide when to stop.) The
    stopped network is then scored exactly like every other model: its
    median forecast from each of the seven validation origins, averaged.
    The candidate with the lowest seven-origin MAPE wins, together with
    the epoch at which its early-stopping metric was best. The winner is
    then trained from scratch on Train+Val for exactly that many epochs
    (there is no validation data left to stop on, so the epoch count
    itself is the selected hyperparameter) and asked for the five
    quantiles of the test week.
    """
    import pytorch_lightning as pl
    import torch
    from darts import TimeSeries
    from darts.utils.likelihood_models import QuantileRegression
    from pytorch_lightning.callbacks import EarlyStopping
    from torchmetrics import MeanAbsolutePercentageError

    monitor = "val_MeanAbsolutePercentageError"

    class EpochRecorder(pl.Callback):
        def __init__(self) -> None:
            self.history: list[float] = []

        def on_validation_end(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
            if trainer.sanity_checking:
                return
            value = trainer.callback_metrics.get(monitor)
            if value is not None:
                self.history.append(float(value))

    def to_ts(series: pd.Series) -> Any:
        return TimeSeries.from_series((series / DEEP_SCALE).astype(np.float32))

    def quantile_frame(model: Any, history: pd.Series) -> pd.DataFrame:
        """Five quantiles (MW) for the 168 hours after `history`."""
        pred = model.predict(HORIZON, series=to_ts(history), predict_likelihood_parameters=True, verbose=False)
        frame = pred.to_dataframe() * DEEP_SCALE
        out = pd.DataFrame(index=horizon_index(history.index[-1] + pd.Timedelta(hours=1)))
        for q in QUANTILES:
            col = [c for c in frame.columns if c.endswith(f"_q{q:.3f}")][0]
            out[q] = frame[col].to_numpy(dtype=float)
        return out

    def build(cand: dict[str, Any], n_epochs: int, callbacks: list[Any]) -> Any:
        set_seeds()
        return model_cls(
            input_chunk_length=HORIZON,
            output_chunk_length=HORIZON,
            n_epochs=n_epochs,
            batch_size=batch_size,
            likelihood=QuantileRegression(list(QUANTILES)),
            torch_metrics=MeanAbsolutePercentageError(),
            optimizer_kwargs={"lr": cand["lr"]},
            random_state=SEED,
            pl_trainer_kwargs={
                "accelerator": "cpu",
                "callbacks": callbacks,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            },
            **fixed_kwargs,
            **{k: v for k, v in cand.items() if k != "lr"},
        )

    t0 = time.perf_counter()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    train_ts = to_ts(train_slice(s))
    # The validation series carries the last week of Train in front so that
    # the first validation window has a full 168-hour input.
    val_ts = to_ts(s.loc[TRAIN_END - pd.Timedelta(hours=HORIZON - 1) : VAL_END])

    table: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cand in candidates:
        recorder = EpochRecorder()
        stopper = EarlyStopping(monitor=monitor, patience=patience, mode="min", min_delta=1e-4)
        model = build(cand, max_epochs, [stopper, recorder])
        tc = time.perf_counter()
        model.fit(train_ts, val_series=val_ts)
        hist = recorder.history
        best_epoch = int(np.argmin(hist)) + 1
        val_mape, origin_rows = rolling_origin_mape(s, lambda history, origin: quantile_frame(model, history)[0.5])
        row = {
            **cand,
            "epochs_run": len(hist),
            "best_epoch": best_epoch,
            "val_mape_pct": val_mape,
            "val_mape_by_origin_pct": [round(float(r["mape_pct"]), 3) for r in origin_rows],
            "early_stopping_val_mape_pct": float(min(hist)) * 100.0,
            "early_stopping_mape_by_epoch_pct": [round(v * 100.0, 3) for v in hist],
            "fit_seconds": round(time.perf_counter() - tc, 1),
        }
        table.append(row)
        print(
            f"{name}: {cand} best epoch {best_epoch}, early-stopping MAPE {row['early_stopping_val_mape_pct']:.2f} %, "
            f"seven-origin median MAPE {val_mape:.2f} % ({row['fit_seconds']} s)"
        )
        if best is None or row["val_mape_pct"] < best["val_mape_pct"]:
            best = row

    assert best is not None
    chosen = {k: v for k, v in best.items() if k in candidates[0]}
    model = build(chosen, best["best_epoch"], [])
    model.fit(to_ts(train_val_slice(s)))
    frame = quantile_frame(model, history_before(s, TEST_START))
    quantiles = sort_quantiles({q: frame[q] for q in QUANTILES})
    return ModelResult(
        name=name,
        point=quantiles[0.5].copy(),
        quantiles=quantiles,
        hyperparameters={
            **{k: v for k, v in fixed_kwargs.items() if not callable(v)},
            **chosen,
            "input_chunk_length": HORIZON,
            "output_chunk_length": HORIZON,
            "batch_size": batch_size,
            "n_epochs_refit": best["best_epoch"],
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "likelihood": f"QuantileRegression{list(QUANTILES)}",
            "scale_divisor_mw": DEEP_SCALE,
        },
        runtime_seconds=time.perf_counter() - t0,
        validation_mape_pct=best["val_mape_pct"],
        validation_table=table,
        extra={"model_class": model_cls.__name__},
    )
