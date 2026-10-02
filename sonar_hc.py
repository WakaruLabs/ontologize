# Loss ablation: resid_nc_hm with ONLY the head-similarity penalty.
#
# resid_nc and resid_nc_hm differ by TWO loss terms at once, so neither can
# be credited with the divergence they show. Their first phases (both ending
# at step 372600, one full 24-epoch pass over the 3,974,400-row mC4 cache):
#
#   resid_nc      s_hcossim = 0      s_Hm = 0        (n_stats 8)
#   resid_nc_hm   s_hcossim = 1e-5   s_Hm = 1e-6     (n_stats 9)
#
# Measured at that matched step, the two are identical through layer 2 and
# diverge at 3-4: the OOD FVU ratio (act_i Discord vs mC4 tail) saturates in
# resid_nc (12.0, 11.6, 10.8) but keeps climbing in resid_nc_hm (11.7, 17.8,
# 27.5). This run supplies the missing cell:
#
#   resid_nc_hc   s_hcossim = 1e-5   s_Hm = 0        (n_stats 9)
#
# resid_nc vs this isolates s_hcossim; this vs resid_nc_hm isolates s_Hm.
#
# s_hcossim is 1e-5 to match resid_nc_hm's FIRST phase -- sonar.py now reads
# 1e-6, which is what resid_nc_hm was RESUMED with at step 372600 for the
# remaining 2.68M steps. The comparison point is 372600, so 1e-5 is correct
# here; do not "fix" it to match the current sonar.py.
#
# s_Hm = 0.0 makes Hyperparams.s_loss report hmean_loss=False, which
# stop_gradients the KL_m stat rather than dropping it: n_stats stays 9 and
# loss.csv keeps its KL_m column, so the batch mean-entropy of this run is
# still observable, just not optimized.
#
# Everything else (seed 42, shuffle, lr, batch, noise and temperature
# schedules, model spec) is inherited from sonar.py unchanged by importing
# it and overriding module globals -- main() resolves these at call time.

import sonar

sonar.s_hcossim = 1e-5
sonar.s_Hm = 0.0
sonar.out = sonar.path / "out/sonar/multilingual/resid_nc_hc"

if __name__ == "__main__":
    print(f"resid_nc_hc: s_hcossim={sonar.s_hcossim} s_Hm={sonar.s_Hm} "
          f"-> {sonar.out}", flush=True)
    sonar.main()
