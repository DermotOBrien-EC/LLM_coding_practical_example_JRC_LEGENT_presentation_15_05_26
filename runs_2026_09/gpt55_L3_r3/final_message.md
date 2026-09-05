Plan before implementation, per your workflow rule:

1. **Inspect inputs and environment**
   - Read `opsd_de_load.csv` shape and timestamps.
   - Check available packages using `../../.venv/bin/python`.
   - Capture `uname -a` for `transcript.md`.

2. **Create code structure**
   - Write:
     - `code/common.py`
     - `code/naive.py`
     - `code/sarima.py`
     - `code/prophet.py`
     - `code/lightgbm_features.py`
     - `code/nbeats.py`
     - `code/patchtst.py`
     - `code/forecast.py`

3. **Implement shared utilities in `code/common.py`**
   - Data loading and validation.
   - Fixed train, validation, and test split helpers.
   - German holiday indicator.
   - Metrics, daily MAPE, pinball loss, interval coverage.
   - Shared model colours and figure style.

4. **Implement models**
   - Naive: weekly lag.
   - SARIMA: small validation search over explicit SARIMAX orders with daily seasonality.
   - Prophet: Darts Prophet with German holidays, or direct Prophet-compatible fallback if needed.
   - LightGBM: engineered calendar, lag, and rolling features, plus quantile models.
   - N-BEATS: Darts model with 168 input and 168 output chunks, early stopping if the installed API supports it.
   - PatchTST: use `PatchTSTModel` if present, otherwise substitute a Darts transformer-like univariate model and record the substitution.

5. **Run final workflow in `code/forecast.py`**
   - Fit on train.
   - Select hyperparameters on validation.
   - Refit chosen settings on train plus validation.
   - Forecast the 168-hour test window.
   - Save intermediate forecasts only if needed inside the allowed output tree.

6. **Generate outputs**
   - `metrics.json`
   - `metrics.csv`
   - Figures `01` to `06`
   - `07_feature_importance.png` if LightGBM is top 2
   - `08_decomposition.png` if Prophet is top 2
   - `transcript.md`

7. **Verification**
   - Run the orchestrator with `../../.venv/bin/python code/forecast.py`.
   - Check all required files exist.
   - Validate `metrics.json` schema shape.
   - Check no test-window leakage in code.
   - Check no forbidden external inputs.
   - Grep my added prose for em dashes before finalising.

Approve this plan and I will implement it.