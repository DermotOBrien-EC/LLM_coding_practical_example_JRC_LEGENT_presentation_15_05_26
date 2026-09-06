from __future__ import annotations

import pandas as pd
from darts.models import TransformerModel

from common import ModelResult
from study_nbeats import run_neural


def run(train: pd.Series, pretest: pd.Series, future: pd.DatetimeIndex) -> ModelResult:
    # TSMixer mixes with MLPs, so it would not test the requested transformer class.
    architecture = {"d_model": 32, "nhead": 4, "num_encoder_layers": 2,
                    "num_decoder_layers": 2, "dim_feedforward": 128, "dropout": 0.1}
    return run_neural("patchtst", TransformerModel, architecture, train, pretest, future)
