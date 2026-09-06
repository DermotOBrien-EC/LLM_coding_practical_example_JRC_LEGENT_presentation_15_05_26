from __future__ import annotations

import pandas as pd
from darts.models import TransformerModel

from .common import ForecastResult
from .deep_common import run_deep


def run(pretest: pd.Series) -> ForecastResult:
    # Darts 0.41.0 has no PatchTSTModel; TSMixer is an MLP, not a transformer.
    return run_deep(
        pretest,
        "patchtst",
        TransformerModel,
        {
            "d_model": 32,
            "nhead": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 64,
            "dropout": 0.1,
        },
    )
