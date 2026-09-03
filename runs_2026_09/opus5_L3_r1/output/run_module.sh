#!/bin/bash
# Run one model module and cache its ForecastResult under artifacts/.
set -u
mod="$1"
cd /Users/doob/dev/energy_forecast_ws/dac166/project/runs/dac166/code
../../../.venv/bin/python -W ignore "$mod.py" > "../logs/$mod.log" 2>&1
echo "DONE_$mod exit=$?" >> ../logs/status.log
