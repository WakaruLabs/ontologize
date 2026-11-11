# planted-hierarchy: notes

## 2026-11-11: the stack separates the two factors and orders them by variance; the hierarchy shows only as joint coding

**Runs.** `grid.sh`: the tree and its shuffle at child scale 1 and 3, for
the two-layer stack (`l=2, h=1`), the stack without deep supervision, and
one layer of two heads, at k = 16; the stack and the flat layer also at
k = 11. Five seeds each, straight-through, 4,000 steps, scored on 8,192
held-out samples (`out/<arm>_k<k>_<condition>_cs<scale>/summary.json`).

Which factor has more variance depends on the scale. At scale 1 the
parent does: the best reconstruction from the parent alone has FVU 0.26
on the tree, from the child alone 0.70. At scale 3 the child does: 0.56
against 0.31 (shuffled: 0.69 against 0.38). The noise floor is 0.14 at
scale 1 and 0.06 at scale 3.

**The stack, k = 16.** Mean over seeds of each head's share of the
parent, and of what the child adds beyond the parent (`child_new`; on
shuffled data this is the child's share). Permuted rows read 0.01 for
the parent and 0.02 for `child_new` on the tree, 0.09 shuffled. On the
tree a head that carried the child exactly would read 0.24 on the
parent, since a child names its parent; layer 1's 0.23 at scale 1 is
that and no more.

| condition | layer 0: parent | layer 0: `child_new` | layer 1: parent | layer 1: `child_new` | parent's head in layer 0 | prefix FVU, layer 0 / 1 |
|---|---|---|---|---|---|---|
| tree, scale 1 | 1.00 | 0.36 | 0.23 | 0.90 | 5/5 | 0.231 / 0.175 |
| shuffled, scale 1 | 1.00 | 0.05 | 0.01 | 0.84 | 5/5 | 0.295 / 0.182 |
| tree, scale 3 | 0.91 | 0.88 | 0.52 | 0.91 | 5/5 | 0.271 / 0.087 |
| shuffled, scale 3 | 0.65 | 0.93 | 0.74 | 0.34 | 1/5 | 0.299 / 0.117 |

**One layer of two heads, k = 16** (head 0 / head 1):

| condition | parent | `child_new` | parent's head is head 0 | FVU |
|---|---|---|---|---|
| tree, scale 1 | 0.94 / 0.91 | 0.66 / 0.70 | 3/5 | 0.156 |
| shuffled, scale 1 | 0.67 / 0.71 | 0.56 / 0.39 | 2/5 | 0.181 |
| tree, scale 3 | 0.86 / 0.90 | 0.91 / 0.90 | 3/5 | 0.083 |
| shuffled, scale 3 | 0.70 / 0.71 | 0.87 / 0.81 | 4/5 | 0.128 |

**Readings.**
- **Depth separates the factors and a flat layer does not.** On shuffled
  data at scale 1 the stack is the planted code: layer 0 carries all of
  the parent and none of the child, layer 1 the reverse (1.00 and 0.05;
  0.01 and 0.84), in every seed. Two heads in one layer each carry about
  two thirds of the parent and half of the child, and which head leans
  to the parent is a coin toss (2 and 3 seeds of 5 at scale 1).
- **The order is by variance.** With independent factors, layer 0 takes
  the parent when the parent has more variance (5/5 at scale 1) and the
  child when the child has (4/5 at scale 3, where layer 0 carries 0.93
  of the child and layer 1 0.74 of the parent). Nothing about which
  factor is the "parent" survives the shuffle; what layer 0 takes is
  whatever reduces its own error most.
- **The hierarchy shows as joint coding, not as order.** On the tree at
  scale 1, layer 0 carries the parent and also 0.36 of what the child
  adds, against 0.05 when shuffled: a child's cell lies inside its
  parent's, so a head with entries to spare can split a parent's cell by
  child and lose nothing of the parent. At scale 3 on the tree, layer 0
  carries both nearly in full (0.91 and 0.88). So where the tree changes
  anything, it puts more of the child into layer 0, the opposite of
  parents first and children after.
- **Spare entries are what allow it.** At k = 11, the parent's level
  count, layer 0 on the tree at scale 1 carries 0.94 of the parent and
  0.10 of `child_new`, and layer 1 carries 0.90 of it, with 0.48 of the
  parent. At scale 3 and k = 11 neither layer is a
  factor (parent 0.65 / 0.63; `child_new` 0.84 / 0.74) and the stack is
  far from the floor (0.217 against 0.063).
- **Deep supervision is not what orders the layers.** Without it the
  parent is still in layer 0 in every seed at scale 1 (parent 0.97 and
  1.00; shuffled, layer 1 carries 0.01 of it), and the final FVU is the
  same or slightly lower (0.169 and 0.174 against 0.175 and 0.182). What
  changes is the layer-0 prefix, which nothing trains (0.33 and 0.31
  against 0.23 and 0.30), and on the tree layer 1 then carries more of
  the parent (0.64 against 0.23).
- **Separation costs reconstruction when the factors are dependent.** On
  the tree the flat layer is closer to the floor than the stack (0.156
  against 0.175 at scale 1; 0.083 against 0.087 at scale 3): two heads
  that both see the input can share a joint code of the 20 planted
  cells. Shuffled at scale 1 the two are level (0.181, 0.182), and at
  scale 3 the stack is ahead (0.117 against 0.128).

**For the project.** Residual depth gives two heads an order that
reproduces across seeds, by error reduction, and it takes two factors
apart that one layer leaves mixed. That is a second route to an ordered
code beside `experiments/slow-order`'s penalty, and it needs no term at
all. It does not find a hierarchy: where residual layers are glossed as
imposing one, "ordering by variance explained" is what this supports.

**Caveats.** One head a layer, two factors, unit coefficients, one tree
shape, 4,000 steps, five seeds. Unit coefficients keep the data inside
a head's model class; the benchmark's real-valued magnitudes are not
planted. The plan's SAE arms (plain and nested-prefix, for absorption)
were not built. `parent_layer` and `child_layer` name the head with the
largest share, which is uninformative where both layers carry a factor
(the tree at scale 3); read the shares.

```bash
experiments/planted-hierarchy/grid.sh
```
