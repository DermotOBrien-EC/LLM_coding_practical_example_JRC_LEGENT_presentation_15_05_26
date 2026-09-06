# Focused post-review verification

Independent Claude peer, 2026-09-06. Scope was only the report/table,
calendar test assertions, literal-character guard, and correction of the
2019 fold claim. No further model search or training changes were requested.

## Peer response

Delta verified; all four items confirmed, and the coordinator's correction
of my 2019 claim stands.

**Correction accepted.** My finding 1 said the blend "never wins a January
fold outright". That was wrong: in 2019 the blend's MAE 1391.37 is lower
than lag 1434.75 and calendar 1441.69 (metrics rows 17-19), so it wins
that fold. My "anti-correlated fold errors" phrasing was also loose: the
component biases share signs in every January fold (2017: -3152 / -947;
2018: +1107 / +3082; 2019: -798 / -982), so the blend's gain comes from
the components' error magnitudes alternating, not from opposite-signed
errors cancelling. The report's replacement wording ("lowest-error
component changes across years ... selected for its three-year mean, not
because it dominates every fold") is accurate.

**(1) Table, forecast_report.md:107-115, generated at
forecast_load.py:274-279.** Pivot over the three learned candidates on
new_year folds only; the nine values match `backtest_metrics.csv` rows
5-7, 11-13, 17-19 rounded to two decimals. Prose below it is consistent
with the numbers.

**(2) Epiphany caveat, forecast_report.md:63-65.** States weekday examples
are scarce and down-weighted, no count claimed. Accurate against my
earlier check (weekday Jan 6 in 2015, 2016, 2017 only; the lag model
additionally drops the first 672 hours, i.e. Jan 2015).

**(3) Tests, test_forecast_load.py:85-86.** `christmas_distance` [3, 7, 9]
for local Dec 28, Jan 1 (from 2019-12-31 23:00Z), Jan 3 is correct by
hand; `holiday_kind == 2` at local Jan 1 is correct. 11 tests pass,
ruff clean, both run by me.

**(4) Guard, forecast_load.py:425.** `chr(0x2014)`; no literal em dash
remains in `forecast_load.py`, and `forecast_report.md` contains none.

`forecast.csv` SHA-256 `8f7b23e8...c5c4232` matches the coordinator's
value, so the point forecast is unchanged by the delta.

## Author's final disposition

Accepted the narrow verification and the correction of the 2019 result.
No material blockers remain. The report deliberately makes no causal
claim about error correlation or cancellation: aggregate fold biases alone
do not establish the relationship between the two models' hourly errors.
The point forecast, model choice, and historical validation metrics are
unchanged. Runtime and artifact checks pass; the missing pandas type stubs
remain the explicitly disclosed static-check limitation in `verification.md`.
