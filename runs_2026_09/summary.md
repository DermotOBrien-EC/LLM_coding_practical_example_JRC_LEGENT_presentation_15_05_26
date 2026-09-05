| run | model | status | stop | turns | tokens in/out (k) | cost $ | wall min | py | png | img+ | docs | csv | metrics.json | AGENTS.md read (turn / first impl) | outside refs | denials | tool errs | bad lines | 5h util max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| astra_L1_r1 | gpt-6-astra | completed | success | 4 | 218/0.4 | 0.38 | 1 | 0 | 0 | 0 | 0 | 0 | False | n/a (no AGENTS.md) | 0 | 0 | 0 | 0 | 0.00 |
| astra_L1_r2 | gpt-6-astra | completed | success | 4 | 218/0.6 | 0.38 | 1 | 0 | 0 | 0 | 0 | 0 | False | n/a (no AGENTS.md) | 0 | 0 | 0 | 0 | 0.00 |
| astra_L1_r3 | gpt-6-astra | completed | success | 4 | 218/0.4 | 0.37 | 1 | 0 | 0 | 0 | 0 | 0 | False | n/a (no AGENTS.md) | 0 | 0 | 0 | 0 | 0.00 |
| astra_L2_r1 | gpt-6-astra | completed | success | 5 | 273/0.5 | 0.41 | 1 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t2 / a1:t4) | 0 | 0 | 1 | 0 | 0.00 |
| astra_L2_r2 | gpt-6-astra | completed | success | 5 | 273/0.4 | 0.40 | 1 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t2 / a1:t4) | 0 | 0 | 1 | 0 | 0.00 |
| astra_L2_r3 | gpt-6-astra | completed | success | 4 | 218/0.4 | 0.37 | 1 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t2 / a1:t4) | 0 | 0 | 1 | 0 | 0.00 |
| astra_L3_r1 | gpt-6-astra | completed | success | 1 | 57/0.4 | 0.32 | 0 | 0 | 0 | 0 | 0 | 0 | False | no | 0 | 0 | 0 | 0 | 0.00 |
| astra_L3_r2 | gpt-6-astra | completed | success | 1 | 57/0.5 | 0.32 | 0 | 0 | 0 | 0 | 0 | 0 | False | no | 0 | 0 | 0 | 0 | 0.00 |
| astra_L3_r3 | gpt-6-astra | completed | success | 4 | 235/1.0 | 0.44 | 1 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t3 / a1:t5) | 0 | 0 | 0 | 0 | 0.00 |
| fable51_L1_r1 | claude-fable-5-1 | completed | success | 15 | 1010/12.5 | 2.33 | 6 | 1 | 1 | 0 | 0 | 1 | True | n/a (no AGENTS.md) (a1:t19 / a1:t7) | 0 | 0 | 0 | 0 | 0.09 |
| fable51_L1_r2 | claude-fable-5-1 | completed | success | 17 | 834/10.7 | 2.17 | 16 | 1 | 1 | 0 | 0 | 1 | False | n/a (no AGENTS.md) (a1:t26 / a1:t24) | 0 | 0 | 0 | 0 | 0.26 |
| fable51_L1_r3 | claude-fable-5-1 | completed | success | 21 | 1044/14.2 | 2.52 | 30 | 3 | 1 | 0 | 0 | 1 | False | n/a (no AGENTS.md) (a1:t5 / a1:t5) | 0 | 0 | 0 | 0 | 0.32 |
| fable51_L2_r1 | claude-fable-5-1 | completed | success | 33 | 2464/29.6 | 4.13 | 30 | 1 | 1 | 0 | 2 | 1 | False | yes (a1:t4 / a1:t17) | 3 | 0 | 1 | 0 | 0.32 |
| fable51_L2_r2 | claude-fable-5-1 | completed | success | 27 | 2217/20.4 | 3.53 | 44 | 2 | 1 | 0 | 0 | 1 | False | yes (a1:t4 / a1:t12) | 1 | 0 | 3 | 0 | 0.33 |
| fable51_L2_r3 | claude-fable-5-1 | completed | success | 42 | 2098/29.9 | 4.08 | 24 | 1 | 1 | 0 | 0 | 1 | False | yes (a1:t4 / a1:t4) | 13 | 0 | 4 | 0 | 0.31 |
| fable51_L3_r1 | claude-fable-5-1 | completed | success | 110 | 18469/120.0 | 16.16 | 97 | 8 | 8 | 0 | 2 | 7 | True | yes (a1:t4 / a1:t8) | 7 | 0 | 2 | 0 | 0.68 |
| fable51_L3_r2 | claude-fable-5-1 | completed | success | 134 | 24525/138.8 | 19.82 | 126 | 8 | 8 | 0 | 1 | 1 | True | yes (a1:t6 / a1:t12) | 6 | 0 | 2 | 0 | 0.46 |
| fable51_L3_r3 | claude-fable-5-1 | completed | success | 108 | 15454/107.1 | 14.14 | 129 | 9 | 7 | 0 | 1 | 2 | True | yes (a1:t7 / a1:t17) | 0 | 0 | 1 | 0 | 0.51 |
| gpt55_L1_r1 | gpt-5.5 | completed | success | 10 | 556/5.5 | 0.72 | 2 | 0 | 0 | 0 | 0 | 1 | False | n/a (no AGENTS.md) | 0 | 0 | 1 | 0 | 0.00 |
| gpt55_L1_r2 | gpt-5.5 | completed | success | 13 | 744/10.0 | 0.94 | 4 | 1 | 0 | 0 | 0 | 1 | False | n/a (no AGENTS.md) (a1:t23 / a1:t4) | 0 | 1 | 2 | 0 | 0.00 |
| gpt55_L1_r3 | gpt-5.5 | completed | success | 22 | 799/10.5 | 1.03 | 5 | 0 | 0 | 0 | 0 | 1 | False | n/a (no AGENTS.md) (a1:t25 / a1:t9) | 0 | 2 | 4 | 0 | 0.00 |
| gpt55_L2_r1 | gpt-5.5 | completed | success | 6 | 223/1.5 | 0.35 | 1 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t6 / None) | 0 | 0 | 0 | 0 | 0.00 |
| gpt55_L2_r2 | gpt-5.5 | completed | success | 15 | 512/4.2 | 0.65 | 2 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t9 / a1:t12) | 0 | 0 | 1 | 0 | 0.00 |
| gpt55_L2_r3 | gpt-5.5 | completed | success | 8 | 282/2.8 | 0.42 | 1 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t5 / a1:t8) | 0 | 0 | 1 | 0 | 0.00 |
| gpt55_L3_r1 | gpt-5.5 | completed | success | 14 | 662/6.8 | 0.77 | 4 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t15 / a1:t12) | 0 | 0 | 1 | 0 | 0.00 |
| gpt55_L3_r2 | gpt-5.5 | completed | success | 94 | 7810/50.9 | 5.96 | 61 | 8 | 8 | 0 | 1 | 1 | True | yes (a1:t15 / a1:t11) | 0 | 0 | 7 | 0 | 0.00 |
| gpt55_L3_r3 | gpt-5.5 | completed | success | 8 | 236/2.7 | 0.42 | 1 | 0 | 0 | 0 | 0 | 0 | False | no | 0 | 0 | 0 | 0 | 0.00 |
| opus5_L1_r1 | claude-opus-5 | completed | success | 63 | 6651/60.5 | 6.23 | 93 | 8 | 1 | 0 | 1 | 2 | False | n/a (no AGENTS.md) (a1:t10 / a1:t5) | 0 | 0 | 6 | 0 | 0.39 |
| opus5_L2_r1 | claude-opus-5 | completed | success | 89 | 10379/63.2 | 8.31 | 81 | 2 | 2 | 0 | 0 | 1 | False | yes (a1:t3 / a1:t5) | 1 | 0 | 1 | 0 | 0.38 |
| opus5_L3_r1 | claude-opus-5 | completed (after resume) resumed | success | 137 | 23149/111.6 | 18.74 | 108 | 8 | 8 | 0 | 2 | 1 | True | yes (a1:t5 / a1:t12) | 26 | 0 | 6 | 0 | 0.61 |
| sol_L1_r1 | gpt-5.6-sol | completed | success | 30 | 1789/12.9 | 1.58 | 9 | 1 | 0 | 0 | 0 | 1 | False | n/a (no AGENTS.md) (a1:t35 / a1:t6) | 0 | 1 | 4 | 0 | 0.00 |
| sol_L1_r2 | gpt-5.6-sol | completed | success | 43 | 3439/37.2 | 6.53 | 22 | 0 | 0 | 0 | 0 | 1 | False | n/a (no AGENTS.md) (a1:t63 / a1:t5) | 1 | 1 | 4 | 0 | 0.00 |
| sol_L1_r3 | gpt-5.6-sol | completed | success | 37 | 2094/18.8 | 1.91 | 12 | 0 | 0 | 0 | 0 | 1 | False | n/a (no AGENTS.md) (a1:t28 / a1:t12) | 1 | 1 | 5 | 0 | 0.00 |
| sol_L2_r1 | gpt-5.6-sol | completed | success | 8 | 282/1.4 | 0.44 | 1 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t6 / a1:t10) | 0 | 0 | 1 | 0 | 0.00 |
| sol_L2_r2 | gpt-5.6-sol | completed | success | 4 | 164/0.6 | 0.35 | 0 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t4 / None) | 0 | 0 | 0 | 0 | 0.00 |
| sol_L2_r3 | gpt-5.6-sol | completed | success | 12 | 396/3.4 | 0.49 | 2 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t4 / a1:t10) | 0 | 0 | 1 | 0 | 0.00 |
| sol_L3_r1 | gpt-5.6-sol | completed | success | 35 | 1223/9.1 | 1.27 | 6 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t20 / a1:t18) | 0 | 0 | 2 | 0 | 0.00 |
| sol_L3_r2 | gpt-5.6-sol | completed | success | 43 | 2312/14.9 | 1.89 | 11 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t20 / a1:t58) | 0 | 0 | 2 | 0 | 0.00 |
| sol_L3_r3 | gpt-5.6-sol | completed | success | 33 | 1172/11.7 | 1.32 | 6 | 0 | 0 | 0 | 0 | 0 | False | yes (a1:t18 / a1:t14) | 0 | 0 | 0 | 0 | 0.00 |
