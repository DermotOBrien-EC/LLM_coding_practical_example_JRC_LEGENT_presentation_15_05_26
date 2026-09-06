"""PatchTST slot, served by darts TSMixer.

The prompt asks for PatchTST "via darts.models.TSMixerModel or
darts.models.PatchTSTModel if the installed darts version exposes it; otherwise
substitute another transformer-based univariate model". The installed darts
(0.41.0) does not expose PatchTSTModel, so we use TSMixerModel, which the prompt
lists as the first acceptable option. TSMixer is a modern all-MLP mixing
architecture for multivariate/long-horizon forecasting; like N-BEATS it reads a
week of history and emits the next week directly, learning the temporal
structure from the values alone.

The substitution is recorded here and in transcript.md. Everything else (chunk
lengths, epoch budget, early stopping, quantile head for intervals) matches the
N-BEATS setup so the two deep models are compared on equal footing.
"""

from __future__ import annotations

import warnings

import common as C

warnings.filterwarnings("ignore", category=UserWarning)


def _make_model(n_epochs: int, callbacks: list):
    C.ensure_real_prophet()
    from darts.models import TSMixerModel
    from darts.utils.likelihood_models import QuantileRegression

    return TSMixerModel(
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
    res = C.deep_forecast("patchtst", bundle, _make_model)
    res.hyperparameters["actual_model"] = "TSMixerModel"
    res.hyperparameters["substitution_reason"] = (
        "darts 0.41.0 does not expose PatchTSTModel"
    )
    return res


if __name__ == "__main__":
    b = C.build_bundle()
    res = run(b)
    print("patchtst(TSMixer) selected_epochs:", res.hyperparameters["selected_epochs"],
          "val MAPE:", res.val_mape)
    print("patchtst test MAPE:", C.mape(b.test.to_numpy(), res.point))
