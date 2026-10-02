"""What share of embedding variance does each head's own partition explain?

eta^2 for a head = between-group over total variance, grouping rows by its
argmax entry. Compare against eta^2 for the label partition -- but correct
for group count first, since eta^2 rises mechanically with it: a random
partition into g groups of n rows scores about (g-1)/(n-1), so divide
through by that. On `ste_h76`:

    best head L0h54   235x its baseline      script     64x
    best lang head     91x                   language   51x
    layer-0 median     17x
    median head         5x
    layer-4 median      2x   (near-random; `lastlayer.py` ablates it to
                              no effect, which is the same finding twice)

So language is NOT too minor a factor for a head to hold -- it is a
stronger partition than 90% of the heads. What stops a head aligning with
it is boundary placement, not salience: `embedgeom.py` measures the
within/between-language cosine gap at 0.30 sd, so the cuts an MSE
objective wants pass through the language clusters rather than around
them.
"""
import os, sys
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/keira/flock/ontologize/experiments/ste-arm")
sys.path.insert(0, "/home/keira/flock/ontologize")
from headlang import codes
from ontologize.data.langs import MC4_TO_SONAR

CACHE = "/home/keira/flock/ontologize/data/sonar_embeddings/mc4_4M.npy"
N = 65536

def eta2(Xc, g, total):
    n = int(g.max()) + 1
    cnt = np.bincount(g, minlength=n)
    M = np.zeros((n, Xc.shape[1]))
    np.add.at(M, g, Xc)
    M /= np.maximum(cnt[:, None], 1)
    p = cnt / len(g)
    return float((p[:, None] * M ** 2).sum()) / total

X = np.asarray(np.load(CACHE, mmap_mode="r")[-N:], dtype=np.float32)
raw = np.load(CACHE.replace(".npy", ".langs.npy"))[-N:]
names, y = np.unique(raw, return_inverse=True)
Xc = (X - X.mean(0)).astype(np.float64)
total = float((Xc ** 2).sum(1).mean())

A, model, step = codes("data/out/sonar/multilingual/ste_h76", 0, 0.00015, X, 256)
h, l = model.h, model.l
E = np.array([eta2(Xc, A[:, j].astype(int), total) for j in range(A.shape[1])])
print(f"ste_h76 step {step}: {l}x{h} heads, {N} rows\n")
print(f"language partition (86 groups): eta^2 = {eta2(Xc, y, total):.4f}")
print(f"head partitions (<=32 groups):  eta^2 median {np.median(E):.4f}  "
      f"max {E.max():.4f}  min {E.min():.4f}")
for i in range(l):
    lay = E[i*h:(i+1)*h]
    print(f"  layer {i}: median {np.median(lay):.4f}  max {lay.max():.4f}")
best = int(E.argmax())
print(f"\nhighest-variance head: L{best//h} h{best%h} eta^2 {E[best]:.4f}")
print(f"the best LANGUAGE head (L0 h50) explains eta^2 {E[50]:.4f}")
print(f"\nratio of the median head's eta^2 to language's: "
      f"{np.median(E)/eta2(Xc, y, total):.2f}x")
