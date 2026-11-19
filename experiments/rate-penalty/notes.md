# rate-penalty: notes

## 2026-11-18: the control's prefix curve, and the arms queued

**The control's curve** (`ratecurve.py`, `data/out/sonar/rate_penalty/ratecurve_base/`), on the
tail's 32,765 rows over the 1023 free coordinates:

| prefix | nominal bits | realized bits | FVU, float gains | FVU, pinned |
|---|---|---|---|---|
| 1 | 380 | 378.1 | 0.6045 | 0.6373 |
| 2 | 760 | 757.4 | 0.4278 | 0.4553 |
| 3 | 1140 | 1136.3 | 0.3106 | 0.3360 |
| 4 | 1520 | 1514.3 | 0.2278 | 0.2511 |
| 5 | 1900 | 1891.5 | 0.1681 | 0.1894 |

- It reproduces the ladder's readings of the same run: `coordfvu.py`'s 0.16814, and
  `quantrate.py`'s pinned 0.18933 at 1,890.8 bits.
- The seed-43 replica agrees to 0.0011 or better at every prefix (0.1679 float and 0.1890
  pinned at prefix 5).
- At most one of a layer's 2,432 entries goes unused on the tail.
- **A layer's bits are worth less the later it comes:** 4.7, 3.1, 2.2 and 1.6 ×10⁻⁴ FVU per
  realized bit (float) for layers 2 to 5. The last value sets the penalty's scale (README.md).

**Queued** as unit `rate-penalty`: `arms.sh`, with `--s-hm=-2e-8`, then `-8e-8`, then `-5e-9`.
A CPU smoke run of 60 steps at `-2e-8` logged `s_Hm` −2e-08 with a finite loss.

## 2026-11-18: the arms beat the control's head-drop hull, by 1–5% of the rate

**Short version.** Trained on its rate, the hard stack reaches a given FVU with fewer realized
bits than the control does by dropping heads. The saving is 21 bits of 1,717 at 1.2× par
(`-2e-8`) and 47–57 bits of about 1,000 at 4.7× par (`-8e-8`). The `-5e-9` arm reads the same as
the control. The gain is outside the seed spread but small, and the SAEs stay far ahead at the
same rates.

**Training.** All three arms ran 50,000 steps with no restart (the unit's Result was
success, NRestarts 0). Over each run's last 1,000 rows:

| run | training MSE | `KL_m` |
|---|---|---|
| control (`s_Hm` 0) | 2.626e-6 | 0.55 |
| `-5e-9` | 2.630e-6 | 0.65 |
| `-2e-8` | 2.691e-6 | 3.03 |
| `-8e-8` | 3.187e-6 | 11.08 |

**The baseline is the control's head-drop hull** (`headdrop.py`, `data/out/sonar/rate_penalty/headdrop/`).
- For each layer, the control's prefix is scored with its last layer's heads dropped four at a
  time, cheapest per bit first.
- The baseline is the lower convex hull of all these points. Any chord between two of them is
  reachable by time-sharing.
- The control's layer-5 curve is nearly straight. With 36 of 76 heads dropped it reads 0.1932 at
  1,712.9 bits, only 0.0047 below the chord between prefixes 4 and 5. So the hull is a little
  harder to beat than straight interpolation, not much.
- With no heads dropped, every layer reproduces `ratecurve.py`'s float FVU.

**The arms against the hull** (float code; bits are the float code's realized bits; pinned
FVU from `ratecurve/curve.csv`). The last column is the bits the hull needs to reach the arm's
float FVU.

| arm | prefix | realized bits | FVU float | FVU pinned | hull | Δ | hull's bits |
|---|---|---|---|---|---|---|---|
| `-2e-8` | 1 | 375.9 | 0.6051 | 0.6370 | 0.6056 | −0.0005 | 376.8 |
| | 2 | 752.1 | 0.4290 | 0.4568 | 0.4295 | −0.0005 | 753.5 |
| | 3 | 1124.2 | 0.3127 | 0.3385 | 0.3133 | −0.0006 | 1126.4 |
| | 4 | 1462.8 | 0.2353 | 0.2590 | 0.2369 | −0.0017 | 1471.8 |
| | 5 | 1695.6 | 0.1926 | 0.2147 | 0.1960 | −0.0034 | 1716.9 |
| `-8e-8` | 1 | 337.8 | 0.6216 | 0.6516 | 0.6277 | −0.0061 | 347.9 |
| | 2 | 644.0 | 0.4605 | 0.4866 | 0.4709 | −0.0104 | 669.4 |
| | 3 | 901.5 | 0.3622 | 0.3870 | 0.3771 | −0.0149 | 948.1 |
| | 4 | 1028.5 | 0.3232 | 0.3467 | 0.3384 | −0.0152 | 1085.2 |
| | 5 | 1028.5 | 0.3240 | 0.3474 | 0.3384 | −0.0144 | 1082.1 |

- **`-5e-9` reads the control's curve:** within 0.0002 of the hull and 1.3 bits of its rate at
  every prefix.
- **Seed spread:** the seed-43 replica agrees with the control to 0.0011 or better at every
  prefix (above), so the margins of 0.0017 and up are outside the seed spread, while −0.0005 is
  not.
- **The pinned code shows the same pattern.** `-8e-8`'s pinned prefix 4, at 0.3467 and 1,030.2
  bits, sits 0.0225 below the straight line between the control's pinned prefixes 2 and 3.

**How the rate is shed: unevenly used and unused entries within each head, not dropped heads**
(`headbits.csv`). No arm has a head under 1 bit outside a switched-off layer.

| run | layer | bits per head, median (10th–90th pct) | entries used of 32, median (10th–90th pct) |
|---|---|---|---|
| control | 3, 4, 5 | 4.99, 4.98, 4.96 | 32 at every percentile |
| `-2e-8` | 3 | 4.90 (4.81–4.96) | 32 (31–32) |
| | 4 | 4.47 (4.26–4.65) | 29 (26–31) |
| | 5 | 3.12 (2.56–3.43) | 11 (7–14) |
| `-8e-8` | 3 | 3.42 (2.98–3.77) | 13 (10–16) |
| | 4 | 1.67 (1.26–2.13) | 6 (4–7) |
| | 5 | 0.00 | 1 |

- **The cut falls on the last layers first**, as the deep-supervision weighting predicts
  (README.md). `-2e-8` loses 2–7 bits in each of layers 1–3, 39 in layer 4 and 144 in layer 5.
- **`-8e-8` switches layer 5 off:** every head stays on one entry. Its prefix 5 (0.3240) is
  slightly worse than its prefix 4 (0.3232), so at that weight the layer is wasted.

**Against the SAEs** (`quantrate_recipe_ladder4`, every number they send priced).
- Near 1,030 bits: the k160 SAE reads 0.2066 at 1,036 bits, against `-8e-8`'s 0.3232 (float)
  and 0.3467 (pinned) at 1,029. The penalty closes 0.015 of a 0.116 gap.
- Near 1,700 bits: the SAE reads 0.1873 at 1,684, against `-2e-8`'s 0.1926 (float) and 0.2147
  (pinned) at 1,696.
- The float code leaves the gains unpriced, which favours the hard stack in this comparison.
- The hard code's shortfall at these rates is not entropy it fails to code.

**Not tested.**
- One seed per arm.
- `quantrate.py` was not run on the arms. Pricing the gains costs the control 7–36 bits, and
  the arms' gain means are within 0.02 of the control's.
- A control trained to the arm's rate as a fixed-rate code. For example, layer 5 with k = 16
  would match the 11 entries a `-2e-8` head uses, and could match the arm without the penalty.
