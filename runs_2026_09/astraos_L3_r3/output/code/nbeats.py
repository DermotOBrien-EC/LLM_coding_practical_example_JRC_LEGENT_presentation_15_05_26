from __future__ import annotations

import pandas as pd
from darts.models import NBEATSModel

from .common import ForecastResult
from .deep_common import run_deep


def run(pretest: pd.Series) -> ForecastResult:
    return run_deep(
        pretest,
        "nbeats",
        NBEATSModel,
        {
            "generic_architecture": True,
            "num_stacks": 30,
            "num_blocks": 1,
            "num_layers": 4,
            "layer_widths": 64,
        },
    )
