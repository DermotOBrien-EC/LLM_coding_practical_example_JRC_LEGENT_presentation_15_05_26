Verified the dataset: **50,400 consecutive hourly observations**, all finite and positive. The splits contain **41,616 training**, **2,208 validation**, and **168 test** observations. The specified packages are present in `../../.venv`.

One methodological decision needs confirmation before the implementation plan: **should all models issue the entire test-week forecast at midnight on January 1, without observing any actual loads during that week?**

That interpretation makes the comparison consistent with the neural models’ 168-hour horizon. LightGBM would then compute later lag-24 and rolling features recursively from predictions, never from held-out actuals. Using actual test-week loads in those features would give it information the other models do not receive.