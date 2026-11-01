## Hidden-test pass rate of the selected sample (mixed problems)

| selector | n=1 | n=2 | n=4 | n=8 | n=16 |
|---|---|---|---|---|---|
| random | 0.588 | 0.586 | 0.589 | 0.588 | 0.588 |
| judge A (Qwen2.5-7B) | 0.588 | 0.711 | 0.776 | 0.790 | 0.760 |
| judge B (Llama-3.1-8B) | 0.588 | 0.688 | 0.739 | 0.757 | 0.770 |
| mean of A and B | 0.588 | 0.706 | 0.774 | 0.801 | 0.810 |
| oracle (hidden tests) | 0.588 | 0.757 | 0.873 | 0.950 | 1.000 |

## Judge A under growing selection pressure

| n | judge A score of pick | hidden-test pass of pick | gap | judge B score of pick | A/B disagree | A>0.9 but wrong |
|---|---|---|---|---|---|---|
| 1 | 0.530 | 0.588 | -0.058 | 0.698 | 0.309 | 0.073 |
| 2 | 0.696 | 0.711 | -0.015 | 0.777 | 0.249 | 0.098 |
| 4 | 0.805 | 0.776 | +0.029 | 0.820 | 0.185 | 0.118 |
| 8 | 0.868 | 0.790 | +0.078 | 0.845 | 0.138 | 0.146 |
| 16 | 0.907 | 0.760 | +0.147 | 0.865 | 0.100 | 0.190 |
