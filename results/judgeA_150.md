Problems: 150 (random 150 of the 355 where the policy's 16 samples disagree)

| | n=1 | n=2 | n=4 | n=8 | n=16 |
|---|---|---|---|---|---|
| hidden-test pass rate, pick by random | 0.562 | 0.562 | 0.561 | 0.561 | 0.561 |
| hidden-test pass rate, pick by judge A | 0.562 | 0.680 | 0.740 | 0.756 | 0.733 |
| hidden-test pass rate, pick by oracle | 0.562 | 0.728 | 0.847 | 0.938 | 1.000 |
| judge A's own score of its pick | 0.515 | 0.677 | 0.789 | 0.855 | 0.898 |
| gap: judge A score minus pass rate | -0.047 | -0.002 | +0.049 | +0.099 | +0.165 |
| pick is wrong although judge A gives it P(Yes) > 0.9 | 0.076 | 0.103 | 0.130 | 0.161 | 0.200 |

Pass rate of judge A's pick: n=8 0.756 (95% CI 0.700 to 0.809), n=16 0.733 (95% CI 0.660 to 0.800).
Paired change from n=1 to n=8: +0.195 (95% CI +0.159 to +0.231).
Paired change from n=8 to n=16: -0.023 (95% CI -0.056 to +0.010). Problems worse at 16: 35, better: 45.
