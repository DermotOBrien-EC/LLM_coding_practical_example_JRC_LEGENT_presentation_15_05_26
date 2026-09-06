Done. `forecast_de_load.py` runs end to end in ~3 minutes.

## How accurate it was

**MAPE 4.14%**, MAE 2,192 MW, RMSE 2,482 MW over the 168 hours. Bias +455 MW (a slight over-forecast). Worst single hour was off by 4,474 MW.

| Model | MAPE | MAE (MW) |
|---|---|---|
| **LightGBM (calendar features)** | **4.14%** | **2,192** |
| Naive: same hour last week | 12.69% | 7,187 |
| Naive: same hour last year | 9.17% | 4,897 |

![forecast](forecast_2020_week1.png)

## What makes this week hard, and what handled it

New Year's Day averaged 44 GW against ~63 GW for a normal January Wednesday, and load doesn't recover until Jan 7. Day-of-week features alone can't express that, which is why the last-week naive baseline is off by 12.7% — its reference window is Christmas.

Three features carry the week: a national holiday flag, a **population-weighted regional holiday share** (Epiphany on Jan 6 is a holiday in BW/BY/ST, 32% of Germany), and a **Christmas–New Year day-offset categorical** that lets the model learn the shape of the industrial shutdown day by day rather than treating Jan 2 as an ordinary Thursday.

## On the accuracy number

Two things worth knowing before you trust 4.14%:

- **It's a hard week, and it's exactly as hard as predicted.** A rolling-origin backtest over 52 weekly refits in 2019 averaged 2.83%. The first-week-of-January 2019 analogue scored **4.07%** — this week's 4.14% lands +0.08 pp from it. The model degraded on schedule rather than surprising us. 4.14% sits at the 90th percentile of 2019's weeks.
- **The forecast is genuinely 168 hours ahead.** No value from inside the horizon is an input. I offered the model a lag-based feature set (lags ≥168 h) as an alternative, and the backtest picked calendar-only (2.83% vs 2.88%) — selection happened on 2019 alone, never on the test week.

Biggest missing driver is **temperature**. Load is strongly heating-dependent in January and the dataset is load-only, so every weather deviation goes into the residual. That's the first thing I'd add.

## Codex review

I ran a read-only Codex review (`codex exec` is blocked in this sandbox; the `codex-job.sh` companion worked, but its own seatbelt couldn't read files, so I inlined the source). No leakage or blocker found. Six findings, all evaluated:

- **Accepted (4):** the gap check compared index *length*, so an off-grid timestamp with equal row count passed — now `index.equals()`; `score()` compared positionally, so a mis-ordered prediction scored silently — now asserts index equality; the backtest's NaN-skip could compare model means over different origin counts — now raises, plus an explicit coverage assert; and the report **hardcoded** "in line with it" and "degraded about as much as expected" regardless of the numbers, which would have printed even if the week had scored 20% — now prints the measured delta and percentile neutrally, with a caveat that the backtest mean is a *selection* statistic.
- **Partially accepted (1):** Codex read `_yearend_offset` as including Jan 15 by mistake. The codes are collision-free and the window boundary is arbitrary; the actual error was my docstring saying "14 January" for offset +14, which is Jan 15. Fixed the docstring, not the logic — changing behaviour would have invalidated the backtest for no gain.
- **Noted, not changed (1):** zero-denominator MAPE. Minimum load in this series is 31,307 MW, so it's unreachable here.

I proved the two new guards go red on exactly the inputs Codex named and green on valid data. Numbers are unchanged after the fixes. `ruff check`/`format` clean; `mypy --strict` reports 8 errors, all missing third-party stubs (`pandas-stubs` not installed, `holidays`/`matplotlib` untyped) — the two real type errors it found are fixed.

Also worth flagging: LightGBM's default `n_jobs=-1` was **10× slower** than `n_jobs=1` on this data (19 s vs 1.9 s per fit) — thread overhead dominates at 35k rows. That's what made the 104-fit backtest practical.