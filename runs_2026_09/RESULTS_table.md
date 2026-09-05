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
| fable51_L3_r3 | Fable 5.1 | LightGBM (calendar + holiday + lag and rolling features, recursive 168-step), winner of six | 5.53 | test_selected (6) | 6 | yes | yes | yes | 7 | 9 | yes | 108 | 129 | complete |
