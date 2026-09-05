The supplied CSV contains hourly German load from 2015 through September 2020. I’ll treat this as a historical forecast, using only observations before the forecast starts and keeping January 2020 actuals out of model fitting.

Should “first week” mean **January 1–7 in UTC**, matching the CSV, or **German local time (CET)**? Both contain 168 hours, but their boundaries differ by one hour.