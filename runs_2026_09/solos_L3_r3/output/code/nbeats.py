from __future__ import annotations

from typing import Any

import pandas as pd
from darts.models import NBEATSModel
from darts.utils.likelihood_models import QuantileRegression

from common import ForecastResult, QUANTILES
from deep_common import run_deep_model

NBEATS_PARAMETERS: dict[str, Any] = {
    "generic_architecture": True,
    "num_stacks": 30,
    "num_blocks": 1,
    "num_layers": 4,
    "layer_widths": 256,
    "expansion_coefficient_dim": 5,
    "trend_polynomial_degree": 2,
    "dropout": 0.0,
    "activation": "ReLU",
}


def _make_nbeats(common_parameters: dict[str, Any]) -> NBEATSModel:
    return NBEATSModel(
        **common_parameters,
        **NBEATS_PARAMETERS,
        likelihood=QuantileRegression(quantiles=list(QUANTILES)),
    )


def run_nbeats(
    train: pd.Series,
    validation: pd.Series,
    train_validation: pd.Series,
    test_index: pd.DatetimeIndex,
) -> ForecastResult:
    return run_deep_model(
        name="nbeats",
        model_factory=_make_nbeats,
        architecture_parameters={
            "model_class": "darts.models.NBEATSModel",
            **NBEATS_PARAMETERS,
        },
        train=train,
        validation=validation,
        train_validation=train_validation,
        test_index=test_index,
    )
