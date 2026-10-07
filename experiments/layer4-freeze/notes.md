# layer4-freeze: notes

## 2026-10-06: audit of results computed with the old `freeze_diag.py` probe

`freeze_diag.py`'s probe predated the constant coordinate, the gain-shape
split and the router: it classified each layer's raw next-layer input and
added `combine(hfwd(P))` without the layer gain. It now calls
`autointerp.onto_probe`, which applies all of them as the forward does
(pinned to the forward by `tests/test_autointerp.py`). Which recorded
results the old probe could have changed:

**`resid_nc` and `resid_nc_hm`: unaffected.** Both predate `constinput`
and `resid_gain` (no layer-0 constant, no gain-shape, no router), and on
both the old probe is bit-identical to `onto_probe` (max |difference| 0.0
over 1024 tail rows at the final checkpoint, and over 4096 at
`resid_nc` step 10000). A rerun of `resid_nc` reproduces the saved
`diag_nc_summary.csv` frozen counts (0, 5, 18, 4, 0 at step 10000); its
means differ from the saved ones by ~1e-4, the same between old and new
probe on this machine, so that is run-to-run numerics, not the probe.

**The four retrains (`retrain_variants.py`; Figure `fig-variants`):
unaffected.** Their checkpoints no longer exist, but their saved
summaries match `resid_nc_hm`'s per (step, layer) almost exactly: frozen
counts agree on 100% of 185 rows for the control and 98%, 100%, 93% for
`drop_ramp`, `sdk_floor`, `slow_anneal` (median |top-share difference|
0.010-0.026). Only a model with `resid_nc_hm`'s configuration -- no
gain-shape -- can track it that closely, and on that configuration the
old probe is exact.

**The HSIC-bottleneck arms (`diag_hsic`, `diag_both`; the appendix's
"HSIC bottleneck in place of the auxiliary losses" and Figure `fig-hsic`):
not verifiable.** `train_hsic.py` built its model through
`hyper.ontologizer(...)` without passing `resid_gain`, so the arms took
the dataclass default at training time, which became True on 2026-09-10.
Their run directories are gone and nothing recorded which side of that
date they trained on. They do predate `constinput` (the old probe would
have failed on a layer-0 classifier one coordinate wider). If they had
gain-shape on, the old probe measured them wrongly past layer 0, and the
error has the reported shape. On `bl_ctl_full`, a gain-shape model with no
frozen heads under the correct probe, the old probe's logic (raw residual
plus constant into the classifier, no gain) collapses argmax usage in the
deep layers (4.41, 4.38, 2.63, 0.92, 0.75 bits by layer, against 4.41,
4.62, 4.27, 4.11, 3.84) and flags 10 and 11 heads in layers 3 and 4 at a
modal-entry share >= 0.95 (`freeze_diag`'s share threshold), against 0
and 0: as the residual shrinks, the constant coordinate dominates the
classifier's input and every sample gets the same entry. The HSIC arms
froze late and deep (layer 3 32/32 from step 80k; 77 of 160 heads by the
end), which a real HSIC collapse would also produce -- a constant head has
zero dependence on the others. Settling it needs either the arms'
`log.jsonl`/spec from wherever they ran, or a retrain of one arm to ~80k
steps diagnosed with the fixed script. Their reconstruction numbers
(training MSE from `loss.csv`) do not depend on the probe.
