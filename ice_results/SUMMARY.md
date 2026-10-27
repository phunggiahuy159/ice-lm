# Kaggle results (decision3 jobs)

Precision fp32, 500 windows/split, GPU Tesla T4

## Main
| run | code | copyable | copy>512 | lk_wrong | follow | prose | acc | cr g1 | cr g2 | cr g5 | cr ovr_diff | cr copy_same | syn fit | syn held | line em | line em (repeat/reuse/novel) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t200_B | 1.730 | 1.061 | 1.132 | 3.171 | 0.239 | 3.384 | 0.661 | 0.090 | 0.077 | 0.100 | 0.093 | 0.140 | 0.140 | 0.137 | 0.118 | 0.41/0.04/0.07 |
| t200_G | 1.899 | 1.374 | 1.680 | 3.215 | 0.208 | 3.444 | 0.624 | 0.077 | 0.093 | 0.087 | 0.060 | 0.080 | 0.100 | 0.103 | 0.094 | 0.26/0.04/0.09 |
| t200_G_VR | 1.690 | 0.989 | 0.933 | 3.211 | 0.340 | 3.324 | 0.665 | 0.317 | 0.000 | 0.000 | 0.007 | 0.310 | 0.017 | 0.063 | 0.133 | 0.51/0.03/0.08 |
| fit_GVRbb_vr | 1.683 | 0.973 | 0.898 | 3.207 | 0.329 | 3.324 | 0.666 | 0.423 | 0.137 | 0.120 | 0.007 | 0.350 | 0.113 | 0.237 | - | - |
| fit_Gbb_v2 | 1.709 | 0.987 | 0.909 | 3.271 | 0.314 | 3.376 | 0.664 | 0.970 | 0.667 | 0.770 | 0.050 | 0.217 | 0.520 | 0.617 | 0.157 | 0.64/0.03/0.08 |
| fit_GVRbb_v2 | 1.680 | 0.967 | 0.895 | 3.208 | 0.319 | 3.324 | 0.667 | 0.990 | 0.813 | 0.827 | 0.013 | 0.323 | 0.547 | 0.660 | - | - |
| fit_GVRbb_v2_s8 | 1.681 | 0.967 | 0.895 | 3.209 | 0.318 | 3.324 | 0.667 | 0.987 | 0.870 | 0.843 | 0.013 | 0.297 | 0.557 | 0.673 | - | - |
| fit_Bbb_v2 | 1.665 | 0.944 | 0.863 | 3.191 | 0.294 | 3.356 | 0.672 | 0.987 | 0.700 | 0.817 | 0.087 | 0.230 | 0.537 | 0.537 | - | - |
| fit_GVRbb_v2syn | 1.680 | 0.967 | 0.895 | 3.208 | 0.319 | 3.324 | 0.667 | 0.990 | 0.830 | 0.843 | 0.017 | 0.280 | 0.580 | 0.653 | - | - |

## Checks (decision3 §3)
| check | value | prediction | |
|---|---|---|---|
| 1-3: t200_B code, T4 - H100 | -0.0001 | within ±0.006 | PASS |
| 1-3: t200_B prose, T4 - H100 | -0.0001 | within ±0.006 | PASS |
| 1-3: t200_G code, T4 - H100 | -0.0000 | within ±0.006 | PASS |
| 1-3: t200_G prose, T4 - H100 | -0.0001 | within ±0.006 | PASS |
| 1-3: t200_G_VR code, T4 - H100 | -0.0001 | within ±0.006 | PASS |
| 1-3: t200_G_VR prose, T4 - H100 | -0.0001 | within ±0.006 | PASS |
| 1-3: (t200_G_VR - B) T4 vs H100 | +0.0000 | within ±0.003 | PASS |
| 1-3: (t200_G - B) T4 vs H100 | +0.0000 | within ±0.003 | PASS |
| 4-6: line EM G_VR - B (all, 95% CI) | +0.015 [+0.002, +0.028] | >= +0.01, CI excludes 0 | PASS |
| 4-6: line EM G_VR - B (repeat, 95% CI) | +0.105 [+0.044, +0.168] | >= +0.05, CI excludes 0 | PASS |
| 4-6: line EM G_VR - B (novel, 95% CI) | +0.008 [-0.013, +0.031] | within ±0.02 | PASS |
| 7: gate, refit old VR scorer - G_VR (T4) | -0.0064 | within ±0.005 | FAIL |
| 8: run 3, V2 on frozen G - B (code) | -0.0205 | -0.025 to +0.005 | PASS |
| 8: run 3 verdict | | | beats |
| 9: run 2, V2 on G_VR bb - G_VR (code) | -0.0093 | -0.003 to -0.010 | PASS |
| 9: run 2, prose | -0.0006 | within ±0.003 | PASS |
| 9: run 2, follow wrong lookup | +0.3188 | <= 0.32 | PASS |
| 9: run 2, conflict_g2 vs G_VR | +0.8133 | >= +0.3 | PASS |
| 9: run 2, conflict_g5 vs G_VR | +0.8267 | >= +0.3 | PASS |
| 10: run 2 seed 8 - seed 7 (code) | +0.0002 | within ±0.003 | PASS |
| 11: run 4, V2 on frozen B - B (code) | -0.0646 | -0.005 to -0.02 | FAIL |
| 12: recall, B repeat of real text (mean) | +0.5535 | 0.55-0.65 | PASS |
| 12: recall, G_VR repeat of real text (mean) | +0.9583 | >= 0.94 | PASS |
| 13: run 5 (syn), conflict_g2 - B | +0.7533 | >= +0.5 | PASS |
| 13: run 5 (syn), conflict_g5 - B | +0.7433 | >= +0.5 | PASS |
| 13: run 5 (syn), override_diff - B | -0.0767 | >= +0.3 | FAIL |
| 13: run 5, code cost vs run 2 | -0.0000 | <= +0.005 | PASS |

## Recall check (token accuracy)
| task | B | G | G_VR |
|---|---|---|---|
| repeat/full@32 | 0.056 | 0.019 | 0.987 |
| repeat/frequent@32 | 0.419 | 0.194 | 0.984 |
| repeat/natural@32 | 0.599 | 0.490 | 0.928 |
| repeat/full@128 | 0.052 | 0.002 | 0.999 |
| repeat/frequent@128 | 0.580 | 0.032 | 0.996 |
| repeat/natural@128 | 0.626 | 0.467 | 0.971 |
| repeat/full@512 | 0.040 | 0.000 | 1.000 |
| repeat/frequent@512 | 0.607 | 0.001 | 0.999 |
| repeat/natural@512 | 0.574 | 0.411 | 0.972 |
| repeat/full@2000 | 0.000 | 0.000 | 1.000 |
| repeat/frequent@2000 | 0.004 | 0.000 | 1.000 |
| repeat/natural@2000 | 0.414 | 0.381 | 0.963 |
| mqar/full@8 | 0.015 | 0.010 | 0.810 |
| mqar/frequent@8 | 0.068 | 0.060 | 0.527 |
| mqar/full@32 | 0.010 | 0.001 | 0.899 |
| mqar/frequent@32 | 0.115 | 0.026 | 0.630 |
| mqar/full@128 | 0.011 | 0.000 | 0.927 |
| mqar/frequent@128 | 0.164 | 0.005 | 0.650 |
| mqar/full@496 | 0.005 | 0.000 | 0.935 |
| mqar/frequent@496 | 0.075 | 0.001 | 0.654 |

## Jobs
- eval_t200_G_VR: done in 23 min
- eval_t200_B: done in 22 min
- eval_t200_G: done in 20 min
- line_t200_B: done in 12 min
- line_t200_G_VR: done in 14 min
- line_t200_G: done in 12 min
- eval_fit_GVRbb_vr: done in 21 min
- fit_fit_Gbb_v2: done in 52 min
- fit_fit_GVRbb_v2: done in 52 min
- eval_fit_Gbb_v2: done in 23 min
- eval_fit_GVRbb_v2: done in 22 min
- line_fit_Gbb_v2: done in 15 min
- fit_fit_GVRbb_v2_s8: done in 52 min
- fit_fit_Bbb_v2: done in 52 min
- eval_fit_GVRbb_v2_s8: done in 22 min
- eval_fit_Bbb_v2: done in 22 min
- recall_check: done in 8 min
- fit_fit_GVRbb_v2syn: done in 52 min
- eval_fit_GVRbb_v2syn: done in 22 min
