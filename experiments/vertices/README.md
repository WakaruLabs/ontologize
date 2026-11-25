# vertices

Where a head's vertices (its atoms) sit: at the centroids of its cells, at
extremes of the data, or outside it.

Jade proposed training with sparsemax for most of a run, switching to
argmax for a few epochs and back to sparsemax at the end. The aim is for the
simplex's vertices to become in-distribution extrema, rather than centroids
or points outside the data. That goal is the fixed point of archetypal
analysis (Cutler & Breiman 1994). There, rows are convex mixtures of
archetypes, and archetypes are convex mixtures of rows, which puts the
archetypes on the boundary of the data's hull. The archetypal SAE (Fel et
al. 2025) keeps the second constraint and drops the first.

Before any arm is trained, this directory measures where the atoms of
existing models sit.

## `place.py`

For each head, with every other head, layer and gain held as the model has
them, the head's target on row n is the final residual plus the head's own
write, `t_n = r_n + g_n w_n`, in the whitened frame. Per unit of gain it is
`z_n = t_n / g_n`, and rows weigh `g_n^2`. A row belongs to the cell of the
entry its head's argmax selects.

Atoms are read about their head's mass-weighted mean atom, and targets about
the head's one-atom least-squares write. The reason is the mean-atom gauge.
A vector shared by every atom of a head adds the gain times it to every row,
and any head can carry it. So a residual component proportional to the gain
moves each head's own least-squares atoms by all of it, while a joint refit
of the layer would remove it only once. It is reported as `common`, and not
counted as placement.

Per entry with at least `--min-rows` rows, the script reports:

| measure | what it reads |
|---|---|
| `off` | the centred distance from the atom to its cell's least-squares atom `mu = sum g t / sum g^2`, over the cell's spread; `off_raw` is the same without centring |
| `common` | per head, the distance between its mean atom and its one-atom write, over its RMS spread |
| `beyond` | along the ray from the head's centre in the atom's centred direction, the weighted share of the cell's rows that project past the atom. About 1/2 for an atom at a symmetric cell's centroid, near 0 at the cell's edge, and 0 past every row |
| `reach` | the atom's projection over the cell's 95th-percentile projection |
| `atom_over_mu` | the atom's centred length over its centroid's |
| `price_atom`, `price_extreme` | the squared error a hard decode adds when the entry writes the atom, or its trimmed extreme, instead of `mu` (`mass |a - mu|^2`), as FVU. Per-head prices do not add across heads, for the reason above |

The **trimmed extreme** of an entry is the weighted mean of its cell's rows
that project between the 80th and 95th percentiles along that ray.

There are two tests of "in distribution", in the whitened frame:
- **`forcing`.** Each row's distance to its nearest held-out row, as a ratio
  to the real rows' median distance. It is computed for:
  - the rows themselves and the model's reconstructions;
  - reconstructions with one random head per row forced, frozen, to another
    entry's atom or to that entry's trimmed extreme;
  - for a soft model, one head forced to its own argmax vertex, and the
    end-to-end argmax decode.

  The norms of these rows about the data mean are reported too.
- **`ood`.** The archetypal SAE's score: one minus an atom's best cosine to a
  data row. Here it is each atom's offset from its head's usage-weighted mean
  atom, against the centred rows of the layer's input. Beside it are random
  directions with those rows' covariance and, at layer 0, the decoder
  directions of an `sae.py` run (`--sae`).

The script checks that gain × atom decomposes the forward exactly, and
refuses models with a router or fibers, where it does not. It runs on CPU:

```bash
JAX_PLATFORMS=cpu uv run python experiments/vertices/place.py \
    --ckpt data/out/sonar/multilingual/ste_h76_init01 \
    --sae data/out/sonar/sae_conv/m5120_k32 --out data/out/sonar/vertices
```

Each run writes `<run>_<step>.json` under `--out`. The temperature defaults
to the run's final one, read from its `log.jsonl`. Results are in
`notes.md`.

## The one-head seed test from a k-means start (`kmstart.py`, `train_kmstart.py`, `kmstart_seeds.sh`)

The numbers to beat, both on `mc4_sent16` (`experiments/seed-stability`):
- one trained head of 32 entries reproduces across seeds at partition NMI
  0.418;
- spherical k-means at the same 32 cells reproduces at 0.732.

The archetypal SAE starts its atoms at fixed k-means centroids, so its
reported stability is confounded with its start. This test separates the
two. It trains the stock one-head recipe (`width_seeds.sh`, flag for flag)
from spherical k-means starts.

**The start (`kmstart.py fit`).** The start is fitted where the head can
represent it exactly.
- The classifier reads the input's shape `u = x/|x|` with a constant 1
  appended, and its logits are the product of two linear maps of that.
- With one factor on the constant and the other on a centroid, the logits
  are `lam c_j . (w (u - ubar))`. Their argmax is spherical k-means' nearest
  centre in the frame `w (u - ubar)`: the shape, centred on its training
  mean and weighted by the loss's `sqrt(MSE weights)`.
- The decoder moves by the least change that decodes each entry to its
  cell's least-squares target, `sum |x| x / sum |x|^2`. The dictionary keeps
  its initial atoms.
- `lam` scales the start's median top-two logit gap to the trained stock
  head's, so the straight-through surrogate starts as soft as a trained
  head's.
- `fit` also reports the fits' own seed agreement in this frame, and the
  stock heads' agreement with them.

**Training (`train_kmstart.py`).** This is `train_eps.py` with
`--km-start`, which installs the start in `init` and refuses to train if
the model does not then select the start's cells on probe rows, or does
not decode its atoms to their targets.

**The queue (`kmstart_seeds.sh`).** It fits the starts and waits for an
idle GPU. It then trains:
- `km42_s42` and `km43_s43`, each seed from its own fit;
- `km42_s43`, seed 43 from seed 42's fit (the shared start).

Finally `kmstart.py score` scores each run against its start, and every
pair of runs, beside rows-shuffled nulls. It runs as a systemd unit:

```bash
systemd-run --user --unit kmstart-seeds -p Restart=on-failure -p RestartSec=120 \
    --working-directory=$PWD \
    -p StandardOutput=append:$PWD/experiments/logs/kmstart_seeds.log \
    -p StandardError=append:$PWD/experiments/logs/kmstart_seeds.log \
    /usr/bin/bash experiments/vertices/kmstart_seeds.sh
```

## Steering with an extreme, read in text (`steer.py`, `steer.sh`)

Does a head forced to its cell's trimmed extreme steer better than one
forced to its vertex, or to its cell's centroid? `place.py`'s nearest-row
test tells the three apart only in the softmax model, so `steer.sh` runs
the comparison on `resid_nc`.

- **The force.** At each layer, each of `--queries` held-out rows has one
  head, drawn at random, forced to another entry with at least
  `--min-rows` rows. The query rows are the first of the rows that place
  the cells. The force is frozen, as in `place.py`. In place of its own
  write, the head writes the entry's vertex (its decoded atom), its
  cell's trimmed extreme, or its cell's centroid. `place_layer` returns
  the last two in atom coordinates.
- **Matched lengths.** The vertex's step is several times the extreme's,
  so forcing the vertex mixes direction with length. Each target's step
  (the target less the head's own write) is therefore also taken at
  `--lengths` times the extreme's step. A random direction with the
  rows' covariance is taken at the same lengths, as the control.
- **The read.** Each row is decoded through M2M100
  (`textfid.SonarDecoder`). The decoder rescales every row to SONAR's
  norm, so it sees only direction. Each decode is re-encoded by SONAR
  under the English tag at unit norm, and the model reads the
  re-encoding.

| measure | what it reads |
|---|---|
| `p`, `hit` | the forced entry's probability at the forced head on the re-encoding, and whether it is that head's argmax |
| `p_given` | the same on the row the decoder was given, at unit norm: what the force put in before the round trip |
| `effect` | `p` less the unforced reconstruction's, row by row |
| `collateral` | 1 − chrF between the reconstruction's decode and this one |
| `keep`, `keep0` | the re-encoding's cosine to its source row, and to the next query row |
| `fidelity` | the re-encoding's cosine to the row the decoder was given |
| `retain` | per layer, the share of rows whose own decode's re-encoding keeps the row's own entry at the forced head: the round trip's ceiling |

`summary.json` gives each condition's means with standard errors. It also
gives paired differences: the three targets at their own lengths, and,
at each matched length, each direction against the extreme's and against
the random one.

`steer.sh` waits for the unit named by `AFTER` (default `kmstart-seeds`)
and an idle GPU. It then runs JAX on the CPU and M2M100 on the GPU, as
a systemd unit:

```bash
systemd-run --user --unit vertices-steer --working-directory=$PWD \
    -p StandardOutput=append:$PWD/experiments/logs/vertices_steer_queue.log \
    -p StandardError=append:$PWD/experiments/logs/vertices_steer_queue.log \
    /usr/bin/bash experiments/vertices/steer.sh
```
