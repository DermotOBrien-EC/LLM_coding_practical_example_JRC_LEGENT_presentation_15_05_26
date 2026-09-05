The supplied CSV labels its timestamps as UTC. Should the 168-hour forecast cover January 1–7 in **UTC**, matching the file, or **German local time (CET)**? These windows differ by one hour.

I’ll train only on data available before the forecast starts and keep the target week’s actual values separate for evaluation.