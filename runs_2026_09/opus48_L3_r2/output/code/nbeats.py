"""N-BEATS: a deep MLP-based univariate forecaster, via darts.

N-BEATS learns to forecast purely from the recent shape of the series. It reads
a window of the last 168 hours (one week) and produces the next 168 hours in one
shot. Internally it is a deep stack of fully-connected blocks that each explain
a piece of the signal and pass the leftover to the next block. It sees no
calendar features and no holiday flags, only the numbers, so it has to rediscover
the daily and weekly rhythm from the data itself.

We attach a quantile-regression head so the model outputs a spread of quantiles,
which gives us prediction intervals for free. Training length is chosen by early
stopping on the validation set; the final model is refit on Train+Validation.
"""

from __future__ import annotations

import warnings

import common as C

warnings.filterwarnings("ignore", category=UserWarning)


def _make_model(n_epochs: int, callbacks: list):
    # darts pulls in the prophet wrapper when its model registry loads; make
    # sure that import finds the real library, not our prophet.py.
    C.ensure_real_prophet()
    from darts.models import NBEATSModel
    from darts.utils.likelihood_models import QuantileRegression

    return NBEATSModel(
        input_chunk_length=C.HORIZON,
        output_chunk_length=C.HORIZON,
        n_epochs=n_epochs,
        batch_size=1024,
        random_state=C.SEED,
        likelihood=QuantileRegression(quantiles=C.QUANTILE_LEVELS),
        pl_trainer_kwargs={
            "accelerator": C.TORCH_ACCELERATOR,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
            "callbacks": callbacks,
        },
    )


def run(bundle: C.DataBundle) -> C.ForecastResult:
    return C.deep_forecast("nbeats", bundle, _make_model)


if __name__ == "__main__":
    b = C.build_bundle()
    res = run(b)
    print("nbeats selected_epochs:", res.hyperparameters["selected_epochs"],
          "val MAPE:", res.val_mape)
    print("nbeats test MAPE:", C.mape(b.test.to_numpy(), res.point))
