"""Ridge-strength selection on rolling winter origins, with Jan 1-7 windows held out."""

from __future__ import annotations

import numpy as np
import pandas as pd

import fastridge
import forecast as fc
import loadfc


def sweep(alphas: list[float]) -> pd.DataFrame:
    s = loadfc.load_series()
    A = loadfc.design_matrix(loadfc.build_features(s.index)).to_numpy(float)
    t = np.log(s.to_numpy(float))

    # Winter origins, Jan 1-8 excluded so the reported January folds stay out of selection.
    roll = [
        ts
        for ts in pd.date_range("2016-04-01", "2020-09-01", freq="D", tz=loadfc.TZ)
        if ts.month in (11, 12, 1, 2) and not (ts.month == 1 and ts.day <= 8)
    ]
    jan = [pd.Timestamp(y, 1, 1, tz=loadfc.TZ) for y in (2017, 2018, 2019)]
    rows = []
    for alpha in alphas:
        fc.ALPHA = alpha
        for level_adjust in (True, False):
            path = fastridge.MomentPath(A, t)
            res: dict[pd.Timestamp, float] = {}
            for o in sorted(set(roll) | set(jan)):
                p = _predict(s, A, path, o, level_adjust)
                if p is None:
                    continue
                a = s.reindex(p.index)
                res[o] = float(np.mean(np.abs(p.to_numpy() / a.to_numpy() - 1)) * 100)
            rows.append(
                {
                    "alpha": alpha,
                    "level_adj": level_adjust,
                    "winter_MAPE": float(np.mean([res[o] for o in roll if o in res])),
                    "winter_n": sum(1 for o in roll if o in res),
                    "jan_MAPE_holdout": float(np.mean([res[o] for o in jan if o in res])),
                }
            )
    return pd.DataFrame(rows)


def _predict(
    s: pd.Series,
    A: np.ndarray,
    path: fastridge.MomentPath,
    origin: pd.Timestamp,
    level_adjust: bool,
) -> pd.Series | None:
    n = int((s.index < origin).sum())
    lo = int(np.searchsorted(s.index, origin))
    if lo + fc.HOURS > len(s) or n < 24 * 400:
        return None
    fit = fastridge.fit_from_moments(path.upto(n), fc.ALPHA)
    pred = fit.predict(A[lo : lo + fc.HOURS])
    if level_adjust:
        pred = pred * np.exp(fc._level_offset(s, A, fit, origin))
    return pd.Series(pred, index=s.index[lo : lo + fc.HOURS], dtype=float)


if __name__ == "__main__":
    df = sweep([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
    pd.set_option("display.width", 200)
    print("=== ridge sweep: selection on winter origins, Jan 1-7 folds held out ===")
    print(df.sort_values("winter_MAPE").round(3).to_string(index=False))
    df.to_csv("tuning_results.csv", index=False)
