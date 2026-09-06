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
| sol_L1_r1 | GPT-5.6-Sol | LightGBM ensemble (three variants; calendar, holiday, level, weekly-lag and prior-year features) | 3.56 | indeterminate | 11 | yes | no | no | 0 | 1 | n/a | 30 | 9 | complete |
| sol_L1_r2 | GPT-5.6-Sol | calendar-analog and ridge ensemble (60/40; weights from 2017 to 2019 January weeks) | 2.41 | leaked (selection stage; headline features clean) | 3 | yes | no | no | 0 | 0 | n/a | 43 | 22 | complete |
| sol_L1_r3 | GPT-5.6-Sol | calendar-aware ridge and year-over-year analog ensemble | 3.45 | final_scoring_only | 3 | yes | no | no | 0 | 0 | n/a | 37 | 12 | complete |
| sol_L2_r1 | GPT-5.6-Sol | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 8 | 1 | complete |
| sol_L2_r2 | GPT-5.6-Sol | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 4 | 0 | complete |
| sol_L2_r3 | GPT-5.6-Sol | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 12 | 2 | complete |
| sol_L3_r1 | GPT-5.6-Sol | none (asked a clarifying question and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 35 | 6 | complete |
| sol_L3_r2 | GPT-5.6-Sol | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 43 | 11 | complete |
| sol_L3_r3 | GPT-5.6-Sol | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 33 | 6 | complete |
| gpt55_L1_r1 | GPT-5.5 | ridge regression on log load with recursive lags, calendar features and an hour-of-week residual correction | 4.20 | final_scoring_only | 1 | no | no | no | 0 | 0 | n/a | 10 | 2 | complete |
| gpt55_L1_r2 | GPT-5.5 | regularised log-load regression (trend, hour, weekday, weekday-hour, annual seasonality, first-week-of-January effects) | 4.58 | indeterminate | 1 | yes | no | no | 0 | 1 | n/a | 13 | 4 | complete |
| gpt55_L1_r3 | GPT-5.5 | calendar and seasonal baseline (recent same-weekday-hour load blended with 2017 to 2019 January analogues) | 9.48 | leaked (intermediate file only; final forecast clean) | 1 | no | no | no | 0 | 0 | n/a | 22 | 5 | complete |
| gpt55_L2_r1 | GPT-5.5 | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 6 | 1 | complete |
| gpt55_L2_r2 | GPT-5.5 | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 15 | 2 | complete |
| gpt55_L2_r3 | GPT-5.5 | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 8 | 1 | complete |
| gpt55_L3_r1 | GPT-5.5 | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | yes | 14 | 4 | complete |
| gpt55_L3_r2 | GPT-5.5 | LightGBM (calendar, holiday, lag and rolling features), winner of six | *3.77* | leaked (headline) | 6 | yes | yes | yes | 8 | 8 | yes | 94 | 61 | complete |
| gpt55_L3_r3 | GPT-5.5 | none (presented a plan and stopped) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 0 | no | 8 | 1 | complete |
| astraos_L1_r1 | GPT-6-Astra + note | LightGBM (calendar and holiday features), chosen over five candidates on three historical January hold-outs | 2.71 | final_scoring_only | 6 | yes | yes | yes | 0 | 3 | n/a | 65 | 27 | complete |
| astraos_L1_r2 | GPT-6-Astra + note | 50:50 blend of ridge and HistGradientBoosting, chosen over four candidates on pre-2020 rolling backtests | 3.63 | final_scoring_only | 5 | yes | yes | yes | 1 | 3 | n/a | 61 | 22 | complete |
| astraos_L1_r3 | GPT-6-Astra + note | 50:50 blend of two LightGBM variants, chosen over four baselines on pre-2020 January backtests | 4.47 | final_scoring_only | 6 | yes | no | yes | 0 | 4 | n/a | 65 | 29 | complete |
| astraos_L2_r1 | GPT-6-Astra + note | HistGradientBoosting (calendar features), chosen on five pre-2020 validation weeks | 4.79 | test_selected (3) | 3 | yes | no | yes | 2 | 2 | yes | 82 | 27 | complete |
| astraos_L2_r2 | GPT-6-Astra + note | HistGradientBoosting (calendar only), chosen on five historical validation weeks | 2.80 | test_selected (2) | 2 | yes | no | yes | 2 | 2 | yes | 63 | 37 | complete |
| astraos_L2_r3 | GPT-6-Astra + note | HistGradientBoosting (calendar only), chosen on six pre-2020 validation weeks | 5.08 | test_selected (4) | 4 | yes | no | yes | 2 | 2 | yes | 55 | 17 | complete |
| astraos_L3_r1 | GPT-6-Astra + note | LightGBM, winner of six by test MAPE as the prompt orders | 3.96 | test_selected (6) | 6 | yes | yes | yes | 8 | 13 | yes | 119 | 44 | complete |
| gpt55os_L2_r1 | GPT-5.5 + note | ridge regression on calendar and lag features, penalty chosen on the test week | 4.69 | leaked (selection stage; headline features clean) | 4 | yes | no | no | 1 | 1 | yes | 35 | 10 | complete |
| gpt55os_L2_r2 | GPT-5.5 + note | ridge regression on calendar features | 8.50 | test_selected (2) | 2 | no | no | no | 1 | 1 | yes | 34 | 7 | complete |
| gpt55os_L2_r3 | GPT-5.5 + note | ridge regression on calendar and lag features | 4.13 | final_scoring_only | 1 | no | no | no | 1 | 1 | yes | 36 | 10 | complete |
| gpt55os_L3_r3 | GPT-5.5 + note | LightGBM, winner of six by test MAPE | 5.46 | test_selected (8) | 6 | yes | yes | yes | 8 | 9 | yes | 145 | 104 | complete |
| opus47_L1_r1 | Opus 4.7 | LightGBM (calendar, holiday and bridge-period features) | 3.60 | final_scoring_only | 1 | no | no | no | 0 | 1 | n/a | 16 | 4 | complete |
| opus47_L1_r2 | Opus 4.7 | multiplicative day-of-week and hour profile with calendar-day factors and a December level correction | 3.32 | leaked (headline) | 3 | no | no | no | 0 | 1 | n/a | 12 | 3 | complete |
| opus47_L1_r3 | Opus 4.7 | weekday-weighted day-of-year and hour historical average with an annual trend | 7.73 | final_scoring_only | 1 | no | no | no | 0 | 1 | n/a | 10 | 2 | complete |
| opus47_L2_r1 | Opus 4.7 | LightGBM (calendar, holiday and 168 h lag features) | *3.47* | final_scoring_only | 1 | no | no | no | 1 | 1 | yes | 12 | 4 | complete |
| opus47_L2_r2 | Opus 4.7 | LightGBM (calendar features and 168, 336 and 504 h lags) | *2.87* | final_scoring_only | 1 | no | no | no | 1 | 1 | yes | 15 | 4 | complete |
| opus47_L2_r3 | Opus 4.7 | LightGBM (calendar features and 168, 336 and 8,760 h lags) | *3.13* | test_selected (2) | 2 | no | no | no | 1 | 1 | yes | 17 | 5 | complete |
| opus48_L1_r1 | Opus 4.8 | LightGBM (calendar, holiday, Christmas-window features and 7, 14 and 364 day lags) | 2.92 | test_selected (2) | 2 | no | no | no | 1 | 1 | n/a | 7 | 4 | complete |
| opus48_L1_r2 | Opus 4.8 | HistGradientBoosting (calendar and holiday features, 168 h and 364 d lags) | 2.80 | test_selected (3) | 3 | no | no | no | 1 | 1 | n/a | 21 | 9 | complete |
| opus48_L1_r3 | Opus 4.8 | HistGradientBoosting (calendar and holiday features only) | 4.41 | test_selected (3) | 3 | yes | no | no | 1 | 3 | n/a | 71 | 20 | complete |
| opus48_L2_r1 | Opus 4.8 | HistGradientBoosting (calendar and holiday features, 168 and 336 h lags) | 2.68 | test_selected (2) | 2 | no | no | no | 1 | 1 | yes | 20 | 7 | complete |
| opus48_L2_r2 | Opus 4.8 | HistGradientBoosting (calendar features, 1 and 2 week and 364 day lags) | *3.60* | test_selected (2) | 2 | no | no | no | 1 | 1 | yes | 20 | 7 | complete |
| opus48_L2_r3 | Opus 4.8 | HistGradientBoosting (calendar features only) | *3.57* | test_selected (2) | 2 | no | no | no | 1 | 1 | yes | 15 | 5 | complete |
| opus48_L3_r1 | Opus 4.8 | LightGBM, winner of six by test MAPE as the prompt orders | *4.83* | test_selected (9) | 6 | yes | yes | yes | 8 | 8 | yes | 101 | 110 | complete |
| opus48_L3_r2 | Opus 4.8 | LightGBM, winner of six by test MAPE as the prompt orders | *4.61* | test_selected (6) | 6 | yes | yes | yes | 8 | 8 | yes | 99 | 62 | complete |
| opus5_L1_r2 | Opus 5 | none (session ended mid-run, no forecast written) | n/a | leaked | 3 | yes | no | yes | 0 | 11 | n/a | 81 | 31 | incomplete |
| opus5_L1_r3 | Opus 5 | ridge regression on log load with a calendar analog and temperature proxy, chosen on historical backtests | 5.35 | leaked (selection stage; headline features clean) | 3 | yes | yes | yes | 2 | 8 | n/a | 92 | 66 | complete |
| opus5_L2_r2 | Opus 5 | LightGBM (calendar features), chosen on 2019 backtests | 4.14 | leaked | 3 | yes | no | no | 1 | 1 | yes | 50 | 42 | complete |
| opus5_L2_r3 | Opus 5 | harmonic regression, chosen over LightGBM and a weekly naive | 4.80 | test_selected (5) | 3 | yes | no | no | 2 | 2 | yes | 79 | 46 | complete |
| opus5_L3_r2 | Opus 5 | LightGBM, winner of six by test MAPE as the prompt orders | *5.47* | leaked (supplement only; headline clean) | 6 | yes | yes | yes | 8 | 8 | yes | 122 | 90 | complete |
| solos_L2_r1 | GPT-5.6-Sol + note | ExtraTrees regressor, chosen over three model classes on historical validation | 3.46 | test_selected (24) | 5 | yes | no | no | 1 | 2 | yes | 149 | 41 | complete |
| solos_L2_r2 | GPT-5.6-Sol + note | LightGBM (calendar and lag features), tuned on historical validation | 4.03 | leaked (selection stage; headline features clean) | 1 | yes | no | no | 1 | 2 | yes | 132 | 75 | complete |
| solos_L2_r3 | GPT-5.6-Sol + note | LightGBM (calendar and lag features), chosen over five alternatives | 3.24 | leaked (selection stage; headline features clean) | 6 | yes | no | no | 1 | 2 | yes | 197 | 131 | complete |
| gpt55os_L3_r1 | GPT-5.5 + note | none (session ended mid-run, no forecast written) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 8 | yes | 38 | 16 | incomplete |
| opus47_L3_r1 | Opus 4.7 | LightGBM, winner of six by test MAPE as the prompt orders | *3.24* | leaked (headline) | 6 | yes | yes | yes | 8 | 9 | yes | 111 | 326 | incomplete |
| opus47_L3_r2 | Opus 4.7 | LightGBM, winner of six by test MAPE as the prompt orders | *3.52* | leaked (headline) | 6 | yes | yes | yes | 8 | 8 | yes | 112 | 78 | complete |
| astraos_L3_r3 | GPT-6-Astra + note | LightGBM, winner of six by test MAPE as the prompt orders | 3.91 | test_selected (6) | 6 | yes | yes | yes | 8 | 14 | yes | 147 | 62 | complete |
| solos_L3_r1 | GPT-5.6-Sol + note | LightGBM, winner of six by test MAPE as the prompt orders | *5.31* | test_selected (6) | 6 | yes | yes | yes | 8 | 8 | yes | 153 | 56 | complete |
| solos_L3_r2 | GPT-5.6-Sol + note | LightGBM, winner of six by test MAPE as the prompt orders | *5.11* | test_selected (6) | 6 | yes | yes | yes | 8 | 10 | yes | 284 | 56 | complete |
| opus5_L3_r3 | Opus 5 | LightGBM, winner of six by test MAPE as the prompt orders | *4.99* | test_selected (6) | 6 | yes | yes | yes | 8 | 8 | yes | 139 | 153 | complete |
| opus48_L3_r3 | Opus 4.8 | none (session ended mid-run, no results written) | n/a | n/a (no forecast) | 0 | no | no | no | 0 | 8 | yes | 59 | 27 | incomplete |
| solos_L3_r3 | GPT-5.6-Sol + note | LightGBM, winner of six by test MAPE as the prompt orders | *5.28* | test_selected (6) | 6 | yes | yes | yes | 8 | 10 | yes | 0 | 180 | incomplete |
| astraos_L3_r2 | GPT-6-Astra + note | LightGBM, winner of six by test MAPE as the prompt orders | 5.12 | test_selected (6) | 6 | yes | yes | yes | 8 | 13 | yes | 143 | 58 | complete |
