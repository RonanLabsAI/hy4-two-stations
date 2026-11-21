| per EP rank (GiB) | B: EP2 GB300+Grace | A: B + 6000 warm | single: TP1 full-copy | single: TP1 exclusive warm |
|---|---|---|---|---|
| --cpu-offload-gb | 166.5 | same | 531.0 | n/a (needs exclusive loader) |
| MoE layers in HBM / Grace | 40 / 37 (1-37) | same | 18 / 59 | same |
| HBM: non-expert (est.) + hot experts + KV (est.) + work (est.) | 236.1 of 238.2 (fits) | same | 235.2 of 238.2 (fits) | same |
| 6000: warm experts (mxfp4) + reserve (est.) | - | 4283 experts, 95.6 of 95.6 (fits) | 4283 experts, 95.6 of 95.6 | same |
| Grace: pinned cold experts + host overhead (est.) | 196.5 of 439.0 (fits) | 196.5 (unchanged: full copy) | 561.1 of 439.0 (DOES NOT FIT) | 410.5 of 439.0 (fits, margin 28.5) |
| warm share of Grace-resident experts | - | 90.4% | - | - |
Expert: 36.01 MiB block-FP8 on the GB300 / Grace, 19.12 MiB mxfp4 on the 6000. Experts total 693.2 GiB, non-expert 39.1 GiB (checkpoint total minus experts). util 0.95, KV 20.0, work 14.0, 6000 reserve 15.6, host overhead 30.0 GiB are estimates.

