# Fairness audit (hold-out, threshold 0.15)

| Group | Value | n | Base churn | Selection rate | Precision | Recall | FPR |
|---|---|---|---|---|---|---|---|
| Geography | Australia | 166 | 0.181 | 0.349 | 0.379 | 0.733 | 0.265 |
| Geography | Japan | 420 | 0.181 | 0.35 | 0.401 | 0.776 | 0.256 |
| Geography | USA | 214 | 0.294 | 0.556 | 0.462 | 0.873 | 0.424 |
| Gender | Female | 417 | 0.18 | 0.372 | 0.394 | 0.813 | 0.275 |
| Gender | Male | 383 | 0.245 | 0.441 | 0.444 | 0.798 | 0.325 |
| AgeBand | <=30 | 158 | 0.051 | 0.12 | 0.211 | 0.5 | 0.1 |
| AgeBand | 31-45 | 469 | 0.171 | 0.386 | 0.298 | 0.675 | 0.326 |
| AgeBand | 46-60 | 135 | 0.541 | 0.785 | 0.67 | 0.973 | 0.565 |
| AgeBand | 60+ | 38 | 0.211 | 0.474 | 0.389 | 0.875 | 0.367 |

Largest gaps within each attribute:

- **AgeBand**: selection-rate gap 0.665, recall gap 0.473, FPR gap 0.465
- **Gender**: selection-rate gap 0.069, recall gap 0.015, FPR gap 0.050
- **Geography**: selection-rate gap 0.207, recall gap 0.140, FPR gap 0.168

Caveat: hold-out is ~800 customers, so small groups are noisy. Differences in selection rate can reflect
real differences in base churn rather than bias; compare recall and FPR too.
