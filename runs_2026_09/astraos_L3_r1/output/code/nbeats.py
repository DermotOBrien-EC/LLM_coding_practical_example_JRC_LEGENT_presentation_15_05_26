from __future__ import annotations

import gc
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning import Callback, LightningModule, Trainer, seed_everything

from common import HORIZON, ModelResult, QUANTILES, ROOT, SEED, mape, validation_blocks


class ValidationMAPE(Callback):
    def __init__(self, train: pd.Series, pretest: pd.Series, scale: float) -> None:
        super().__init__()
        blocks = validation_blocks(pretest, len(train))
        self.context = torch.tensor(np.stack([h.iloc[-HORIZON:].to_numpy() / scale for h, _ in blocks]),
                                    dtype=torch.float32).unsqueeze(-1)
        self.targets = pretest.iloc[len(train):].to_numpy()
        self.lengths = [len(target) for _, target in blocks]
        self.scale = scale
        self.best_mape = float("inf")
        self.best_epoch = 0
        self.best_prediction = np.empty(0)
        self.history: list[dict[str, Any]] = []
        self.stale = 0
        self.significant_best = float("inf")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        was_training = pl_module.training
        pl_module.eval()
        with torch.inference_mode():
            output = pl_module((self.context.to(pl_module.device), None, None))
            q = np.sort(output[:, :, 0, :].detach().cpu().numpy(), axis=2) * self.scale
        pl_module.train(was_training)
        point = np.concatenate([q[i, :length, 2] for i, length in enumerate(self.lengths)])
        score = mape(self.targets, point)
        epoch = trainer.current_epoch + 1
        self.history.append({"epoch": epoch, "validation_mape_pct": score})
        print(f"{type(pl_module).__name__} epoch={epoch} validation MAPE={score:.3f}%", flush=True)
        if score < self.best_mape:
            self.best_mape = score
            self.best_epoch = epoch
            self.best_prediction = point.copy()
        if score < self.significant_best - 0.01:
            self.significant_best = score
            self.stale = 0
        else:
            self.stale += 1
        if self.stale >= 5:
            trainer.should_stop = True


def make_model(model_class: Any, epochs: int, callbacks: list[Callback], name: str,
               architecture: dict[str, Any]) -> Any:
    return model_class(
        input_chunk_length=168, output_chunk_length=168,
        n_epochs=epochs, batch_size=32, random_state=SEED,
        likelihood=QuantileRegression(quantiles=QUANTILES),
        optimizer_kwargs={"lr": 0.001}, save_checkpoints=False,
        force_reset=True, work_dir=str(ROOT / ".cache" / "darts"), model_name=name,
        pl_trainer_kwargs={"accelerator": "cpu", "devices": 1,
                           "enable_progress_bar": False, "enable_model_summary": False,
                           "logger": False, "enable_checkpointing": False,
                           "deterministic": True, "gradient_clip_val": 1.0,
                           "callbacks": callbacks, "num_sanity_val_steps": 0},
        **architecture,
    )


def series(y: pd.Series, scale: float) -> TimeSeries:
    return TimeSeries.from_series((y / scale).astype(np.float32), freq="h")


def run_neural(name: str, model_class: Any, architecture: dict[str, Any],
               train: pd.Series, pretest: pd.Series, future: pd.DatetimeIndex) -> ModelResult:
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    seed_everything(SEED, workers=True, verbose=False)
    train_scale = float(train.mean())
    callback = ValidationMAPE(train, pretest, train_scale)
    model = make_model(model_class, 30, [callback], f"{name}_selection", architecture)
    model.fit(series(train, train_scale), stride=168, verbose=False,
              dataloader_kwargs={"num_workers": 0})
    selected_epochs = callback.best_epoch
    if selected_epochs < 1:
        raise RuntimeError("Neural validation callback did not run.")
    del model
    gc.collect()
    # This is a fresh fit, not continued optimization on validation targets.
    seed_everything(SEED, workers=True, verbose=False)
    final_scale = float(pretest.mean())
    final = make_model(model_class, selected_epochs, [], f"{name}_final", architecture)
    final_series = series(pretest, final_scale)
    final.fit(final_series, stride=168, verbose=False, dataloader_kwargs={"num_workers": 0})
    distribution = final.predict(len(future), series=final_series,
                                 predict_likelihood_parameters=True, num_samples=1, verbose=False)
    if not distribution.time_index.equals(future):
        raise ValueError("Darts returned an unexpected test forecast index.")
    quantiles = np.sort(distribution.values(), axis=1).astype(float) * final_scale
    # The callback's fast forward path must agree with the public Darts predictor.
    final.model.eval()
    context = torch.tensor(final_series.values()[-168:], dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        raw = final.model((context, None, None))[0, :, 0, :].cpu().numpy()
    np.testing.assert_allclose(np.sort(raw, axis=1) * final_scale, quantiles, rtol=2e-5, atol=0.1)
    params = {"actual_class": model_class.__name__, "input_chunk_length": 168,
              "output_chunk_length": 168, "max_epochs": 30, "selected_epochs": selected_epochs,
              "early_stopping_metric": "validation MAPE of rearranged median, all 2208 hours",
              "early_stopping_patience": 5, "early_stopping_min_delta_pct": 0.01,
              "training_stride_hours": 168, "batch_size": 32, "learning_rate": 0.001,
              "gradient_clip_val": 1.0, "likelihood": "QuantileRegression",
              "quantiles": QUANTILES, "seed": SEED, "accelerator": "cpu", "threads": 4,
              "train_scale_mw": train_scale, "final_scale_mw": final_scale,
              "final_training_windows": (len(pretest) - 336) // 168 + 1,
              **architecture}
    if name == "nbeats":
        params.update({"generic_architecture": True, "num_stacks": 30, "num_blocks": 1,
                       "num_layers": 4, "layer_widths": 256, "expansion_coefficient_dim": 5})
    else:
        params["substitution"] = "Darts 0.41.0 has no PatchTSTModel; TransformerModel is an actual transformer, not TSMixer."
    result = ModelResult(name, quantiles[:, 2], quantiles, callback.best_prediction,
                         params, callback.history, time.perf_counter() - start,
                         {"callback_public_predict_equivalence": True})
    del final
    gc.collect()
    return result


def run(train: pd.Series, pretest: pd.Series, future: pd.DatetimeIndex) -> ModelResult:
    return run_neural("nbeats", NBEATSModel, {}, train, pretest, future)
