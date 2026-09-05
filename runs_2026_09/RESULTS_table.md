| Run | Model | Headline model class | Test MAPE % | Test-window audit | Models fitted | Held-out validation | Intervals | Write-up | Figures | .py files | Read AGENTS.md | Turns | Wall min | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fable51_L1_r1 | Fable 5.1 | LightGBM (calendar + holiday + lag features) | 3.96 | test_selected (3) | 3 | no | yes | no | 1 | 1 | n/a | 15 | 6 | complete |
| fable51_L1_r2 | Fable 5.1 | LightGBM (calendar + holiday + lag features, quantile band) | 3.28 | test_selected (3) | 3 | no | yes | no | 1 | 1 | n/a | 17 | 16 | complete |
| fable51_L1_r3 | Fable 5.1 | LightGBM 3-seed ensemble (after a post-test feature change) | 4.49 | leaked | 4 | yes | no | no | 1 | 3 | n/a | 21 | 30 | complete |
| fable51_L2_r1 | Fable 5.1 | LightGBM (calendar + holiday + weekly-lag features) | 2.30 | test_selected (3) | 3 | no | no | yes | 1 | 1 | yes | 33 | 30 | complete |
| fable51_L2_r2 | Fable 5.1 | LightGBM (calendar + holiday + lag features), pre-final | 5.35 | test_selected (3) | 3 | no | no | no | 1 | 2 | yes | 27 | 44 | incomplete |
| fable51_L2_r3 | Fable 5.1 | HistGradientBoostingRegressor (scikit-learn; calendar + holiday + lag features) | 3.21 | test_selected (3) | 3 | no | no | no | 1 | 1 | yes | 42 | 24 | complete |
| opus5_L1_r1 | Opus 5 | Ridge regression on log load, calendar-only features, level correction, empirical 80% band | 3.07 | leaked | 5 | yes | yes | yes | 1 | 8 | n/a | 63 | 93 | complete |
| opus5_L2_r1 | Opus 5 | LightGBM with a workday-run feature added after seeing the target-week error (no forecast file) | *3.40* | leaked | 3 | yes | no | no | 2 | 2 | yes | 89 | 81 | incomplete |
| opus5_L3_r1 | Opus 5 | LightGBM (calendar + holiday + lag features, recursive 168-step), winner of six | 5.46 | test_selected (6) | 6 | yes | yes | yes | 8 | 8 | yes | 137 | 108 | complete |
| fable51_L3_r1 | Fable 5.1 | LightGBM (calendar + holiday + lag and rolling features, recursive 168-step), winner of six | 5.03 | test_selected (6) | 6 | yes | yes | yes | 8 | 8 | yes | 110 | 97 | complete |
| fable51_L3_r2 | Fable 5.1 | LightGBM (calendar + holiday + lag and rolling features, recursive 168-step), winner of six | 4.99 | test_selected (6) | 6 | yes | yes | yes | 8 | 8 | yes | 134 | 126 | complete |
| fable51_L3_r3 | Fable 5.1 | LightGBM (calendar + holiday + lag and rolling features, recursive 168-step), winner of six | 5.53 | leaked (supplement only; headline clean) | 6 | yes | yes | yes | 7 | 9 | yes | 108 | 129 | complete |
| astra_L1_r1 | GPT-6-Astra | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | n/a | 4 | 1 | complete |
| astra_L1_r2 | GPT-6-Astra | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | n/a | 4 | 1 | complete |
| astra_L1_r3 | GPT-6-Astra | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | n/a | 4 | 1 | complete |
| astra_L2_r1 | GPT-6-Astra | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 5 | 1 | complete |
| astra_L2_r2 | GPT-6-Astra | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 5 | 1 | complete |
| astra_L2_r3 | GPT-6-Astra | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 4 | 1 | complete |
| astra_L3_r1 | GPT-6-Astra | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | no | 1 | 0 | complete |
| astra_L3_r2 | GPT-6-Astra | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | no | 1 | 0 | complete |
| astra_L3_r3 | GPT-6-Astra | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 4 | 1 | complete |
| sol_L1_r1 | GPT-5.6-Sol | LightGBM ensemble (three variants; calendar, holiday, level, weekly-lag and prior-year features) | 3.56 | final_scoring_only | 11 | yes | no | no | 0 | 1 | n/a | 30 | 9 | complete |
| sol_L1_r2 | GPT-5.6-Sol | calendar-analog and ridge ensemble (60/40; weights from 2017 to 2019 January weeks) | 2.41 | leaked (selection stage; headline features clean) | 3 | yes | no | no | 0 | 0 | n/a | 43 | 22 | complete |
| sol_L1_r3 | GPT-5.6-Sol | calendar-aware ridge and year-over-year analog ensemble | 3.45 | final_scoring_only | 3 | yes | no | no | 0 | 0 | n/a | 37 | 12 | complete |
| sol_L2_r1 | GPT-5.6-Sol | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 8 | 1 | complete |
| sol_L2_r2 | GPT-5.6-Sol | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 4 | 0 | complete |
| sol_L2_r3 | GPT-5.6-Sol | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 12 | 2 | complete |
| sol_L3_r1 | GPT-5.6-Sol | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 35 | 6 | complete |
| sol_L3_r2 | GPT-5.6-Sol | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 43 | 11 | complete |
| sol_L3_r3 | GPT-5.6-Sol | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 33 | 6 | complete |
| gpt55_L1_r1 | GPT-5.5 | ridge regression on log load with recursive lags, calendar features and an hour-of-week residual correction | 4.20 | final_scoring_only | 1 | no | no | no | 0 | 0 | n/a | 10 | 2 | complete |
| gpt55_L1_r2 | GPT-5.5 | regularised log-load regression (trend, hour, weekday, weekday-hour, annual seasonality, first-week-of-January effects) | 4.58 | final_scoring_only | 1 | yes | no | no | 0 | 1 | n/a | 13 | 4 | complete |
| gpt55_L1_r3 | GPT-5.5 | calendar and seasonal baseline (recent same-weekday-hour load blended with 2017 to 2019 January analogues) | 9.48 | leaked (intermediate file only; final forecast clean) | 1 | no | no | no | 0 | 0 | n/a | 22 | 5 | complete |
| gpt55_L2_r1 | GPT-5.5 | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 6 | 1 | complete |
| gpt55_L2_r2 | GPT-5.5 | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 15 | 2 | complete |
| gpt55_L2_r3 | GPT-5.5 | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 8 | 1 | complete |
| gpt55_L3_r1 | GPT-5.5 | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 14 | 4 | complete |
| gpt55_L3_r2 | GPT-5.5 | LightGBM (calendar, holiday, lag and rolling features), winner of six | *3.77* | leaked (headline) | 6 | yes | yes | yes | 8 | 8 | yes | 94 | 61 | complete |
| gpt55_L3_r3 | GPT-5.5 | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | no | 8 | 1 | complete |
