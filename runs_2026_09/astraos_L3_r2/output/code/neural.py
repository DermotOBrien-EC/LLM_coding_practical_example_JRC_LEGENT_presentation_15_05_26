from __future__ import annotations

import gc
import logging
from time import perf_counter
from typing import Any
import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.models import NBEATSModel, TransformerModel
from darts.utils.likelihood_models import GaussianLikelihood
from pytorch_lightning import Callback, Trainer, seed_everything
from .common import ModelResult, ROOT, SEED, TRAIN_END, gaussian_quantiles, mape, validation_blocks


class WeeklyValidation(Callback):
    def __init__(self, pretest: pd.Series, scale: float) -> None:
        super().__init__()
        blocks = validation_blocks(pretest)
        self.inputs = np.stack([history.iloc[-168:].to_numpy() / scale for history, _ in blocks])[
            :, :, None
        ]
        self.targets = [target.to_numpy() for _, target in blocks]
        self.scale = scale
        self.best = float("inf")
        self.best_epoch = 0
        self.stale = 0
        self.history: list[dict[str, Any]] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Any) -> None:
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            context = torch.as_tensor(self.inputs, device=pl_module.device, dtype=pl_module.dtype)
            raw = pl_module._produce_train_output((context, None, None, None, None))
            means = raw[:, :, 0, 0].detach().cpu().numpy() * self.scale
        pl_module.train(was_training)
        actual = np.concatenate(self.targets)
        predicted = np.concatenate(
            [mean[: len(target)] for mean, target in zip(means, self.targets)]
        )
        value = mape(actual, predicted)
        epoch = trainer.current_epoch + 1
        self.history.append({"epoch": epoch, "mape_pct": value})
        print(f"VALIDATION epoch={epoch} mape_pct={value:.5f}", flush=True)
        if value < self.best:
            self.best, self.best_epoch, self.stale = value, epoch, 0
        else:
            self.stale += 1
        if self.stale >= 4:
            trainer.should_stop = True


def build(name: str, epochs: int, callback: WeeklyValidation | None = None) -> Any:
    kwargs: dict[str, Any] = dict(
        input_chunk_length=168,
        output_chunk_length=168,
        n_epochs=epochs,
        batch_size=64,
        random_state=SEED,
        likelihood=GaussianLikelihood(),
        optimizer_kwargs={"lr": 0.001},
        save_checkpoints=False,
        force_reset=True,
        work_dir=str(ROOT / ".cache" / "darts"),
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "deterministic": True,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "callbacks": [] if callback is None else [callback],
            "gradient_clip_val": 1.0,
        },
    )
    if name == "nbeats":
        return NBEATSModel(**kwargs)
    return TransformerModel(
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.1,
        **kwargs,
    )


def run_neural(name: str, pretest: pd.Series) -> ModelResult:
    start = perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    train = pretest.loc[: TRAIN_END - pd.Timedelta(hours=1)]
    scale = float(train.mean())
    callback = WeeklyValidation(pretest, scale)
    seed_everything(SEED, workers=True)
    candidate = build(name, 30, callback)
    candidate.fit(
        TimeSeries.from_series((train / scale).astype(np.float32)),
        stride=24,
        dataloader_kwargs={"num_workers": 0, "pin_memory": False},
        verbose=False,
    )
    selected_epochs = callback.best_epoch
    if selected_epochs < 1:
        raise RuntimeError("Neural validation failed to select an epoch")
    validation = callback.history
    del candidate
    gc.collect()
    seed_everything(SEED, workers=True)
    final_scale = float(pretest.mean())
    final = build(name, selected_epochs)
    scaled = TimeSeries.from_series((pretest / final_scale).astype(np.float32))
    final.fit(
        scaled, stride=24, dataloader_kwargs={"num_workers": 0, "pin_memory": False}, verbose=False
    )
    parameters = final.predict(
        168, series=scaled, predict_likelihood_parameters=True, verbose=False
    ).values()
    if parameters.shape != (168, 2):
        raise ValueError(f"Unexpected Gaussian parameters: {parameters.shape}")
    point, sigma = parameters[:, 0] * final_scale, parameters[:, 1] * final_scale
    # Pin the custom early-stopping evaluator against Darts' public forecast path.
    final.model.eval()
    with torch.no_grad():
        context = torch.tensor(
            (pretest.iloc[-168:].to_numpy() / final_scale)[None, :, None], dtype=torch.float32
        )
        direct = (
            final.model._produce_train_output((context, None, None, None, None))[0, :, 0, 0].numpy()
            * final_scale
        )
    np.testing.assert_allclose(point, direct, rtol=1e-5, atol=0.05)
    params: dict[str, Any] = {
        "actual_class": type(final).__name__,
        "input_chunk_length": 168,
        "output_chunk_length": 168,
        "max_epochs": 30,
        "selected_epochs": selected_epochs,
        "early_stopping": "hour-weighted rolling-origin validation MAPE; patience=4 epochs",
        "training_stride_hours": 24,
        "batch_size": 64,
        "learning_rate": 0.001,
        "gradient_clip_val": 1.0,
        "seed": SEED,
        "likelihood": "GaussianLikelihood",
        "scaling": "divide by fitting-partition mean, no centering",
        "accelerator": "cpu",
        "threads": 4,
    }
    if name == "nbeats":
        params.update(
            {
                "generic_architecture": True,
                "num_stacks": 30,
                "num_blocks": 1,
                "num_layers": 4,
                "layer_widths": 256,
            }
        )
    else:
        params.update(
            {
                "substitution": "PatchTSTModel unavailable in Darts 0.41.0; TransformerModel used, not TSMixer",
                "d_model": 32,
                "nhead": 4,
                "num_encoder_layers": 2,
                "num_decoder_layers": 1,
                "dim_feedforward": 64,
                "dropout": 0.1,
            }
        )
    diagnostics = {
        "epochs_evaluated": len(validation),
        "best_validation_mape_pct": callback.best,
        "training_windows": (len(train) - 336) // 24 + 1,
        "refit_windows": (len(pretest) - 336) // 24 + 1,
        "custom_evaluator_matches_public_api": True,
    }
    return ModelResult(
        name,
        point,
        gaussian_quantiles(point, sigma),
        perf_counter() - start,
        params,
        validation,
        diagnostics,
    )
