The dataset timestamps are UTC. Should the 168-hour window use:

1. **UTC:** 2020-01-01 00:00 through 2020-01-07 23:00 UTC  
2. **German local time:** the same calendar dates in `Europe/Berlin`, corresponding to 2019-12-31 23:00 through 2020-01-07 22:00 UTC

Either is easy to implement, but the selected boundary changes all forecast and accuracy rows.