from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Callback, Trainer, seed_everything
from darts.utils.likelihood_models.torch import QuantileRegression

from .common import (
    QUANTILES,
    ROOT,
    SEED,
    TRAIN_END,
    ForecastResult,
    as_darts,
    mape,
    validation_blocks,
)

MAX_EPOCHS = 2
STRIDE = 24


class RollingMapeStop(Callback):
    def __init__(self, pretest: pd.Series, scale: float, name: str) -> None:
        self.name = name
        self.scale = scale
        blocks = list(validation_blocks(pretest))
        self.inputs = np.stack([history.iloc[-168:].to_numpy() / scale for history, _ in blocks])[
            :, :, None
        ].astype(np.float32)
        self.actual = np.concatenate([target.to_numpy() for _, target in blocks])
        self.lengths = [len(target) for _, target in blocks]
        self.records: list[dict[str, Any]] = []
        self.best = float("inf")
        self.best_epoch = 0
        self.patience_reference = float("inf")
        self.stale = 0
        self.last_predictions: np.ndarray | None = None

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Any) -> None:
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            inputs = torch.tensor(self.inputs, device=pl_module.device)
            # Installed Darts 0.41 uses (past target, future covariates, static covariates).
            raw = pl_module((inputs, None, None)).cpu().numpy()[:, :, 0, :]
        pl_module.train(was_training)
        quantiles = np.sort(raw, axis=-1) * self.scale
        point = np.concatenate([row[:length, 2] for row, length in zip(quantiles, self.lengths)])
        value = mape(self.actual, point)
        epoch = int(trainer.current_epoch) + 1
        self.last_predictions = point
        self.records.append({"epoch": epoch, "mape_validation_pct": value})
        if value < self.best:
            self.best, self.best_epoch = value, epoch
        if value < self.patience_reference - 0.01:
            self.patience_reference, self.stale = value, 0
        else:
            self.stale += 1
        print(f"{self.name} epoch={epoch} validation_MAPE={value:.4f}%", flush=True)
        if epoch >= 6 and self.stale >= 5:
            trainer.should_stop = True


def model_kwargs(name: str, epochs: int, callbacks: list[Callback]) -> dict[str, Any]:
    return {
        "input_chunk_length": 168,
        "output_chunk_length": 168,
        "n_epochs": epochs,
        "batch_size": 32,
        "likelihood": QuantileRegression(QUANTILES),
        "random_state": SEED,
        "optimizer_kwargs": {"lr": 0.001},
        "save_checkpoints": False,
        "force_reset": True,
        "model_name": name,
        "work_dir": str(ROOT / "artifacts" / "torch"),
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "deterministic": True,
            "callbacks": callbacks,
            "default_root_dir": str(ROOT / "artifacts" / "torch"),
            "num_sanity_val_steps": 0,
        },
    }


def run_deep(
    pretest: pd.Series, name: str, factory: Callable[..., Any], architecture: dict[str, Any]
) -> ForecastResult:
    start = perf_counter()
    torch.set_num_threads(4)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    seed_everything(SEED, workers=True, verbose=False)
    train = pretest.loc[:TRAIN_END]
    scale = float(train.mean())
    callback = RollingMapeStop(pretest, scale, name)
    candidate = factory(
        **model_kwargs(name + "_validation", MAX_EPOCHS, [callback]), **architecture
    )
    candidate.fit(
        as_darts(train, scale),
        stride=STRIDE,
        verbose=False,
        dataloader_kwargs={"num_workers": 0, "pin_memory": False},
    )
    # Independently check the callback against the public forecast path, not a second forward call.
    contexts = [as_darts(history.iloc[-168:], scale) for history, _ in validation_blocks(pretest)]
    public = candidate.predict(
        168, series=contexts, predict_likelihood_parameters=True, verbose=False
    )
    point = np.concatenate(
        [
            np.sort(forecast.values(), axis=1)[:length, 2] * scale
            for forecast, length in zip(public, callback.lengths)
        ]
    )
    np.testing.assert_allclose(point, callback.last_predictions, rtol=1e-5, atol=0.5)
    selected_epochs = callback.best_epoch
    if selected_epochs < 1:
        raise RuntimeError("No validation epoch was scored")
    del candidate
    seed_everything(SEED, workers=True, verbose=False)
    final_scale = float(pretest.mean())
    final = factory(**model_kwargs(name + "_refit", selected_epochs, []), **architecture)
    final.fit(
        as_darts(pretest, final_scale),
        stride=STRIDE,
        verbose=False,
        dataloader_kwargs={"num_workers": 0, "pin_memory": False},
    )
    forecast = final.predict(168, predict_likelihood_parameters=True, verbose=False)
    quantiles = np.sort(forecast.values(), axis=1).astype(float) * final_scale
    return ForecastResult(
        name=name,
        point=quantiles[:, 2],
        quantiles=quantiles,
        runtime_seconds=perf_counter() - start,
        hyperparameters={
            **architecture,
            "implementation": factory.__name__,
            "input_chunk_length": 168,
            "output_chunk_length": 168,
            "selected_epochs": selected_epochs,
            "maximum_epochs": MAX_EPOCHS,
            "selection_epochs_run": len(callback.records),
            "early_stop_patience": 5,
            "early_stop_min_delta_mape_pct": 0.01,
            "training_stride_hours": STRIDE,
            "batch_size": 32,
            "learning_rate": 0.001,
            "quantiles": QUANTILES,
            "scaling": "divide by fit-partition mean",
            "seed": SEED,
            "accelerator": "cpu",
            "torch_threads": 4,
        },
        validation=callback.records,
        diagnostics={
            "public_api_validation_agreement": True,
            "interval_caveat": "Direct multi-horizon marginal quantile regression; sorted to prevent crossing; no post-hoc calibration.",
        },
    )
