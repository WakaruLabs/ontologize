"""Does ConcatDictBlock tighten the non-negative correlation floor?

A head's k rows are non-negative unit vectors in R^n, n = d_head (the whole
of e_dec when heads sum, e_dec/h when they concatenate). Writing c for the
mean pairwise cosine, ||sum a||^2 = k + k(k-1)c, and ||v||_2 >= ||v||_1 /
sqrt(n) with ||sum a||_1 = sum ||a||_1 >= k, so

    c >= (k/n - 1) / (k - 1),

and non-negativity separately gives c >= 0. The binding floor is the max of
the two, so it only rises above zero once k > n.
"""
import numpy as np

K = 32  # entries per head in the live arms


def floor(k, n):
    return max(0.0, (k / n - 1) / (k - 1))


def construct(k, n):
    """Even disjoint supports: rows sharing a coordinate have cosine 1, the
    rest 0. Attains the bound when k > n, and 0 when k <= n."""
    A = np.zeros((k, n))
    for i in range(k):
        A[i, i % n] = 1.0
    return A


def meancos(A):
    k = len(A)
    U = A / np.linalg.norm(A, axis=1, keepdims=True)
    return (U @ U.T).sum() / (k * (k - 1)) - 1 / (k - 1)


print("floor is attained by even disjoint supports:")
print(f"{'k':>4} {'n':>6}  {'built':>8}  {'bound':>8}")
for k, n in ((32, 1536), (32, 32), (32, 20), (32, 11), (8, 4), (16, 8)):
    print(f"{k:>4} {n:>6}  {meancos(construct(k, n)):>8.4f}  {floor(k, n):>8.4f}")

print("\nper-head width in the live and reachable configs (k=32, h=76, d=768):")
print(f"{'config':>28} {'n = d_head':>11} {'floor':>8}  {'support/atom':>13}")
for name, n in (("summing, e_dec 1536", 1536), ("summing, e_dec 768", 768),
                ("concat --d-head 32", 32), ("concat --d-head 20", 20),
                ("concat --d-head 11 (minimum)", 11)):
    # coordinates per atom if the k atoms share n coordinates disjointly
    print(f"{name:>28} {n:>11} {floor(K, n):>8.4f}  {max(n // K, 0):>13}")
