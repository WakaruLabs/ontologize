"""Neural network layers for dictionary learning and ontofeature decomposition.

This subpackage provides custom Flax Linen modules implementing sparse multihead
dictionary learning, multilinear and bilinear projections, classification-weighted
concept reconstruction, and causal intervention primitives.

Modules:
    sparse: Base module `Sparse` with conditional gradient gating and statistics.
    linear: `Linear` layer with `fwd`/`rev` methods for multi-layer ghost gradients.
    nlinear: `NLinear`, `Bilinear`, `NLinearBlock`, `BilinearBlock` interaction layers.
    dictblock: `DictBlock` multihead dictionary lookup and intervention module.
    dictenc: `DictEnc` single-layer dictionary encoder combining classifier and dictionary.
"""
