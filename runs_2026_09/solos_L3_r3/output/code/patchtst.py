from __future__ import annotations

from typing import Any

import pandas as pd
from darts.models import TSMixerModel
from darts.utils.likelihood_models import QuantileRegression

from common import ForecastResult, QUANTILES
from deep_common import run_deep_model

TSMIXER_PARAMETERS: dict[str, Any] = {
    "hidden_size": 64,
    "ff_size": 64,
    "num_blocks": 2,
    "activation": "ReLU",
    "dropout": 0.1,
    "norm_type": "LayerNorm",
    "normalize_before": False,
}


def _make_tsmixer(common_parameters: dict[str, Any]) -> TSMixerModel:
    return TSMixerModel(
        **common_parameters,
        **TSMIXER_PARAMETERS,
        likelihood=QuantileRegression(quantiles=list(QUANTILES)),
    )


def run_patchtst_substitute(
    train: pd.Series,
    validation: pd.Series,
    train_validation: pd.Series,
    test_index: pd.DatetimeIndex,
) -> ForecastResult:
    return run_deep_model(
        name="patchtst",
        model_factory=_make_tsmixer,
        architecture_parameters={
            "requested_model": "PatchTSTModel",
            "substitution": "darts.models.TSMixerModel",
            "substitution_reason": "Darts 0.41.0 does not expose PatchTSTModel",
            **TSMIXER_PARAMETERS,
        },
        train=train,
        validation=validation,
        train_validation=train_validation,
        test_index=test_index,
    )
