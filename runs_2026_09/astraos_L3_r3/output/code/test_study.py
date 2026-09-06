from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PACKAGE = types.ModuleType("study")
PACKAGE.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules.setdefault("study", PACKAGE)

from study.common import load_data, mape, quantile_diagnostics, score, validation_blocks  # noqa: E402
from study.lightgbm_features import FEATURES, feature_frame, recursive_forecast  # noqa: E402


def test_verified_data_and_split_contract() -> None:
    data = load_data()
    assert len(data) == 50400
    assert len(data.loc[:"2019-09-30 23:00"]) == 41616
    assert len(data.loc["2019-10-01":"2019-12-31 23:00"]) == 2208
    assert len(data.loc["2020-01-01":"2020-01-07 23:00"]) == 168
    assert data.index[0].tzinfo is not None


def test_every_validation_hour_scored_once() -> None:
    data = load_data().loc[:"2019-12-31 23:00"]
    blocks = list(validation_blocks(data))
    assert [len(target) for _, target in blocks] == [168] * 13 + [24]
    index = blocks[0][1].index[:0]
    for history, target in blocks:
        assert history.index[-1] + pd.Timedelta(hours=1) == target.index[0]
        index = index.append(target.index)
    assert index.equals(data.loc["2019-10-01":].index)
    assert index.is_unique


def test_metric_formulas_and_units() -> None:
    actual = np.array([100.0, 200.0])
    pred = np.array([90.0, 240.0])
    assert mape(actual, pred) == pytest.approx(15.0)
    result = score(actual, pred)
    assert result["mae_test_mw"] == pytest.approx(25.0)
    assert result["rmse_test_mw"] == pytest.approx(np.sqrt(850.0))
    q = np.tile([80.0, 90.0, 100.0, 110.0, 120.0], (2, 1))
    diagnostics = quantile_diagnostics(np.array([100.0, 115.0]), q)
    assert diagnostics["winner_coverage_80pct"] == 0.5
    assert diagnostics["winner_coverage_95pct"] == 1.0
    assert diagnostics["winner_pinball_loss_q50"] == 3.75


def test_rolling_features_exclude_current_and_future_targets() -> None:
    index = pd.date_range("2017-01-01", periods=9000, freq="h", tz="UTC")
    series = pd.Series(np.arange(9000, dtype=float) + 1000, index=index)
    first = feature_frame(series)
    changed = series.copy()
    changed.iloc[8800:] = -999999.0
    second = feature_frame(changed)
    np.testing.assert_array_equal(first.iloc[8800].values, second.iloc[8800].values)
    assert first.iloc[8800]["rolling_mean_24h"] == series.iloc[8776:8800].mean()
    assert first.iloc[8800]["lag_8760h"] == series.iloc[40]
    assert first.iloc[8800]["rolling_std_168h"] == pytest.approx(series.iloc[8632:8800].std(ddof=0))


class Lag24Predictor:
    def predict(self, matrix: np.ndarray, **kwargs: object) -> np.ndarray:
        return matrix[:, FEATURES.index("lag_24h")] + 1.0


def test_recursive_forecast_uses_its_own_unobserved_lags() -> None:
    index = pd.date_range("2017-01-01", periods=9000, freq="h", tz="UTC")
    history = pd.Series(np.arange(9000, dtype=float) + 1000, index=index)
    future = pd.date_range(index[-1] + pd.Timedelta(hours=1), periods=168, freq="h")
    prediction = recursive_forecast({0.5: Lag24Predictor()}, history, future)
    np.testing.assert_array_equal(prediction[:24, 0], history.iloc[-24:].values + 1.0)
    np.testing.assert_array_equal(prediction[24:48, 0], prediction[:24, 0] + 1.0)
    with pytest.raises(ValueError, match="immediately"):
        recursive_forecast({0.5: Lag24Predictor()}, history, future + pd.Timedelta(hours=1))


def test_deep_validation_is_no_grad_and_matches_weekly_naive() -> None:
    import torch
    from study.deep_common import RollingMapeStop

    class RepeatWeek(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device = torch.device("cpu")
            self.grad_enabled: bool | None = None

        def forward(self, inputs: tuple[torch.Tensor, None, None]) -> torch.Tensor:
            self.grad_enabled = torch.is_grad_enabled()
            return inputs[0].unsqueeze(-1).repeat(1, 1, 1, 5)

    data = load_data().loc[:"2019-12-31 23:00"]
    callback = RollingMapeStop(data, 50000.0, "test")
    model = RepeatWeek()
    trainer = types.SimpleNamespace(current_epoch=0, should_stop=False)
    callback.on_train_epoch_end(trainer, model)
    actual, prediction = [], []
    for history, target in validation_blocks(data):
        actual.extend(target.values)
        prediction.extend(history.iloc[-168:].values[: len(target)])
    assert callback.records[0]["mape_validation_pct"] == pytest.approx(
        mape(np.array(actual), np.array(prediction)), abs=1e-5
    )
    assert model.grad_enabled is False
    assert model.training is True
    assert callback.best_epoch == 1
    for epoch in range(1, 6):
        trainer.current_epoch = epoch
        callback.on_train_epoch_end(trainer, model)
    assert trainer.should_stop is True


def test_sarima_extension_keeps_training_variance_parameter() -> None:
    from study.sarima import fit, weekly_fourier

    index = pd.date_range("2015-01-01", periods=1000, freq="h", tz="UTC")
    rng = np.random.default_rng(2026)
    series = pd.Series(
        50000 + 5000 * np.sin(np.arange(1000) * 2 * np.pi / 24) + rng.normal(0, 500, 1000),
        index=index,
    )
    fitted, _ = fit(series.iloc[:832], (1, 0, 0), 50000.0)
    assert "sigma2" in fitted.param_names
    assert fitted.model.concentrate_scale is False
    extended = fitted.extend(series.iloc[832:].values / 50000.0, exog=weekly_fourier(index[832:]))
    np.testing.assert_array_equal(extended.params, fitted.params)
    assert np.isfinite(
        extended.get_forecast(
            1, exog=weekly_fourier(pd.DatetimeIndex([index[-1] + pd.Timedelta(hours=1)]))
        ).predicted_mean
    ).all()


def test_lightgbm_validation_and_final_use_identical_quantile_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from study import lightgbm_features
    from study.common import QUANTILES

    requested: list[list[float]] = []

    class ConstantBooster:
        def __init__(self, level: float) -> None:
            self.level = level

        def predict(self, matrix: np.ndarray) -> np.ndarray:
            crossing_quantiles = {0.025: 46000, 0.1: 47000, 0.5: 45000, 0.9: 49000, 0.975: 50000}
            return np.full(len(matrix), crossing_quantiles[self.level])

        def feature_importance(self, importance_type: str) -> np.ndarray:
            return np.ones(len(FEATURES))

    def fake_fit(
        series: pd.Series, config: dict[str, int], levels: list[float]
    ) -> dict[float, ConstantBooster]:
        requested.append(levels)
        return {level: ConstantBooster(level) for level in levels}

    monkeypatch.setattr(lightgbm_features, "fit_models", fake_fit)
    data = load_data().loc[:"2019-12-31 23:00"]
    result = lightgbm_features.run(data)
    assert requested == [QUANTILES, QUANTILES, QUANTILES]
    expected = mape(data.loc["2019-10-01":].to_numpy(), np.full(2208, 47000.0))
    assert all(
        record["mape_validation_pct"] == pytest.approx(expected) for record in result.validation
    )
    np.testing.assert_array_equal(result.point, np.full(168, 47000.0))
