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
