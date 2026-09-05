| run | model | status | stop | turns | tokens in/out (k) | cost $ | wall min | py | png | img+ | docs | csv | metrics.json | AGENTS.md read (turn / first impl) | outside refs | denials | tool errs | bad lines | 5h util max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fable51_L1_r1 | claude-fable-5-1 | completed | success | 15 | 1010/12.5 | 2.33 | 6 | 1 | 1 | 0 | 0 | 1 | True | n/a (no AGENTS.md) (a1:t19 / a1:t7) | 0 | 0 | 0 | 0 | 0.09 |
| fable51_L1_r2 | claude-fable-5-1 | completed | success | 17 | 834/10.7 | 2.17 | 16 | 1 | 1 | 0 | 0 | 1 | False | n/a (no AGENTS.md) (a1:t26 / a1:t24) | 0 | 0 | 0 | 0 | 0.26 |
| fable51_L1_r3 | claude-fable-5-1 | completed | success | 21 | 1044/14.2 | 2.52 | 30 | 3 | 1 | 0 | 0 | 1 | False | n/a (no AGENTS.md) (a1:t5 / a1:t5) | 0 | 0 | 0 | 0 | 0.32 |
| fable51_L2_r1 | claude-fable-5-1 | completed | success | 33 | 2464/29.6 | 4.13 | 30 | 1 | 1 | 0 | 2 | 1 | False | yes (a1:t4 / a1:t17) | 3 | 0 | 1 | 0 | 0.32 |
| fable51_L2_r2 | claude-fable-5-1 | completed | success | 27 | 2217/20.4 | 3.53 | 44 | 2 | 1 | 0 | 0 | 1 | False | yes (a1:t4 / a1:t12) | 1 | 0 | 3 | 0 | 0.33 |
| fable51_L2_r3 | claude-fable-5-1 | completed | success | 42 | 2098/29.9 | 4.08 | 24 | 1 | 1 | 0 | 0 | 1 | False | yes (a1:t4 / a1:t4) | 13 | 0 | 4 | 0 | 0.31 |
| fable51_L3_r1 | claude-fable-5-1 | running | success | 3 | 476/1.1 | 7.28 | 0 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t4 / a1:t8) | 7 | 0 | 2 | 0 | 0.63 |
| opus5_L1_r1 | claude-opus-5 | completed | success | 63 | 6651/60.5 | 6.23 | 93 | 8 | 1 | 0 | 1 | 2 | False | n/a (no AGENTS.md) (a1:t10 / a1:t5) | 0 | 0 | 6 | 0 | 0.39 |
| opus5_L2_r1 | claude-opus-5 | completed | success | 13 | 2177/6.5 | 8.31 | 81 | 2 | 2 | 0 | 0 | 1 | False | yes (a1:t3 / a1:t5) | 1 | 0 | 1 | 0 | 0.38 |
| opus5_L3_r1 | claude-opus-5 | completed (after resume) resumed | success | 63 | 14019/35.1 | 18.74 | 37 | 8 | 8 | 0 | 2 | 1 | True | yes (a1:t5 / a1:t12) | 26 | 0 | 6 | 0 | 0.61 |
