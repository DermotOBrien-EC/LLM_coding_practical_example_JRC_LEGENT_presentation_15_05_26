from __future__ import annotations

import pandas as pd
from .common import ModelResult
from .neural import run_neural


def run(pretest: pd.Series) -> ModelResult:
    return run_neural("patchtst", pretest)
