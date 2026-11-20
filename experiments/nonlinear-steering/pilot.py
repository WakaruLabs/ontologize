"""Can GPT-2 small host Hewitt's non-linear steering constructions at the
`gpt2_l8` dictionaries' site?

Hewitt (2026, "Do Not Rely on the Linear Representation Hypothesis") builds
two counterexamples on Qwen3 by training steering vectors:
- one direction v with learned scales alpha_i, where alpha_i v steers task i
  alone (intensity is not scaling);
- two vectors v1 and v2 that each steer one task, whose sum steers a third
  (composition is not addition).

His objective is the NLL of task text under the steer plus 0.2 times the
KL that steer causes on every other task's text, summed over pairs. This
reimplements that objective from its description (his code carries no
licence) at `blocks.8.resid_post`, positions >= 1, for the four behaviours
`tasks.py` builds. The composition experiment is v1 -> french,
v2 -> recipes, v1 + v2 -> code. A linear control trains one independent
vector per task under the same objective.

Held-out efficacy is the drop in NLL on a task's test windows
(nats/token). Crosstalk is the KL from the unsteered model on the other
tasks' test windows. Both are scored at the predictions positions 1..T-1
make, the ones a steer at positions >= 1 can reach.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from jaxtyping import Float, Int

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import OUT, capture, device_for, load_gpt2, site_edit  # noqa: E402
from tasks import TASKS, build  # noqa: E402

COMPOSE = ("french", "recipes", "code")        # v1, v2, and what v1 + v2 steers
TEXTS = TASKS + ("neutral",)
Term = Tuple[str, str, str]                     # (condition, text, "nll" | "kl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch", type=int, default=8, help="windows per text per step")
    p.add_argument("--T", type=int, default=64, help="text tokens per window")
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--kl", type=float, default=0.2, help="weight on each KL term")
    p.add_argument("--alpha-init", type=float, nargs=4, default=[1.0, 3.0, 5.0, 7.0])
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT / "pilot")
    return p.parse_args()


def softplus_inv(y: float) -> float:
    """The x with softplus(x) = y, for y > 0."""
    return y + math.log(-math.expm1(-y))


class Steers(torch.nn.Module):
    """Every trained steering vector, zero-initialized: the shared direction
    `v` with scales `softplus(a_raw)`, the composition pair `v1`, `v2`, and
    the independent per-task vectors `U` (the linear control)."""

    def __init__(self, d: int, alpha_init: List[float]):
        super().__init__()
        self.v = torch.nn.Parameter(torch.zeros(d))
        self.a_raw = torch.nn.Parameter(torch.tensor([softplus_inv(a) for a in alpha_init]))
        self.v1 = torch.nn.Parameter(torch.zeros(d))
        self.v2 = torch.nn.Parameter(torch.zeros(d))
        self.U = torch.nn.Parameter(torch.zeros(len(TASKS), d))

    def alphas(self) -> Float[torch.Tensor, "4"]:
        return torch.nn.functional.softplus(self.a_raw)

    def conditions(self) -> Dict[str, Float[torch.Tensor, "d"]]:
        """Each named steer: `shared:<task>` is alpha_task v, `single:<task>`
        the control's vector, and `v1`, `v2`, `v1+v2` the composition."""
        a = self.alphas()
        out = {f"shared:{t}": a[i] * self.v for i, t in enumerate(TASKS)}
        out.update({f"single:{t}": self.U[i] for i, t in enumerate(TASKS)})
        out.update({"v1": self.v1, "v2": self.v2, "v1+v2": self.v1 + self.v2})
        return out


def objective_terms() -> List[Term]:
    """The training objective's (condition, text, kind) terms: each steer's
    NLL on its own task's text and its KL on every other task's text."""
    terms: List[Term] = []
    for t in TASKS:
        for u in TASKS:
            kind = "nll" if u == t else "kl"
            terms += [(f"shared:{t}", u, kind), (f"single:{t}", u, kind)]
    for cond, own in zip(("v1", "v2", "v1+v2"), COMPOSE):
        terms += [(cond, u, "nll" if u == own else "kl") for u in COMPOSE]
    return terms


def by_text(terms: List[Term]) -> Dict[str, List[Tuple[str, str]]]:
    """The terms grouped by the text they are scored on, so one batched
    forward pass per text covers every steer scored there."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    for cond, text, kind in terms:
        out.setdefault(text, []).append((cond, kind))
    return out


def steered_logprobs(model, tk: Int[torch.Tensor, "b t1"],
                     S: Float[torch.Tensor, "c d"]
                     ) -> Float[torch.Tensor, "c b t2 v"]:
    """Log-probabilities at positions 1..T-1 under each of `S`'s steers,
    added at positions >= 1; one pass over `c * b` sequences."""
    c, b = S.shape[0], tk.shape[0]
    rows = S.repeat_interleave(b, 0)                       # condition-major
    with site_edit(model, lambda h: h + rows[:, None, :].to(h.dtype)):
        logits = model(tk.repeat(c, 1), use_cache=False).logits[:, 1:-1]
    return torch.log_softmax(logits.float(), -1).view(c, b, *logits.shape[1:])


def base_logprobs(model, tk: Int[torch.Tensor, "b t1"]) -> Float[torch.Tensor, "b t2 v"]:
    with torch.no_grad():
        return torch.log_softmax(model(tk, use_cache=False).logits[:, 1:-1].float(), -1)


def nll_kl(lp: Float[torch.Tensor, "b t2 v"], ref: Float[torch.Tensor, "b t2 v"],
           tk: Int[torch.Tensor, "b t1"]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean NLL of the actual next tokens and mean KL(ref || lp)."""
    nll = -lp.gather(-1, tk[:, 2:, None])[..., 0].mean()
    kl = (ref.exp() * (ref - lp)).sum(-1).mean()
    return nll, kl


def train(model, steers: Steers, data: Dict[str, Tuple[np.ndarray, np.ndarray]],
          cfg: argparse.Namespace, dev: torch.device, log_path: Path) -> None:
    groups = by_text(objective_terms())
    opt = torch.optim.Adam(steers.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    clip_groups = ([steers.v, steers.a_raw], [steers.v1, steers.v2], [steers.U])
    t0 = time.time()
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        names = [f"{c}|{u}|{k}" for u, items in groups.items() for c, k in items]
        w.writerow(["step", "loss"] + [f"alpha_{t}" for t in TASKS] + names)
        for step in range(1, cfg.steps + 1):
            opt.zero_grad(set_to_none=True)
            total, vals = 0.0, {}
            for u, items in groups.items():
                train_w = data[u][0]
                tk = torch.as_tensor(train_w[rng.integers(len(train_w), size=cfg.batch)],
                                     device=dev)
                ref = base_logprobs(model, tk)
                conds = steers.conditions()
                lp = steered_logprobs(model, tk, torch.stack([conds[c] for c, _ in items]))
                loss = torch.zeros((), device=dev)
                for i, (c, kind) in enumerate(items):
                    nll, kl = nll_kl(lp[i], ref, tk)
                    term = nll if kind == "nll" else cfg.kl * kl
                    loss = loss + term
                    vals[f"{c}|{u}|{kind}"] = float(nll if kind == "nll" else kl)
                loss.backward()
                total += float(loss)
            for ps in clip_groups:
                torch.nn.utils.clip_grad_norm_(ps, 1.0)
            opt.step()
            if step % 50 == 0 or step == 1:
                a = steers.alphas().tolist()
                w.writerow([step, total] + a + [vals[n] for n in names])
                f.flush()
                print(f"  step {step} loss {total:.3f} alphas "
                      f"{[round(x, 2) for x in a]} |v| {float(steers.v.norm()):.1f} "
                      f"({time.time() - t0:.0f}s)", flush=True)


def evaluate(model, steers: Steers, data: Dict[str, Tuple[np.ndarray, np.ndarray]],
             batch: int, dev: torch.device) -> Dict[str, Dict[str, Dict[str, float]]]:
    """`{condition: {text: {"nll", "kl"}}}` on the test windows, `base`
    being the unsteered model."""
    with torch.no_grad():
        conds = {k: v.detach() for k, v in steers.conditions().items()}
    names = list(conds)
    S = torch.stack([conds[n] for n in names])
    out: Dict[str, Dict[str, Dict[str, float]]] = {n: {} for n in ["base"] + names}
    for u in TEXTS:
        test = data[u][1]
        sums = {n: [0.0, 0.0] for n in ["base"] + names}
        count = 0
        for i in range(0, len(test), batch):
            tk = torch.as_tensor(test[i:i + batch], device=dev)
            ref = base_logprobs(model, tk)
            sums["base"][0] += float(nll_kl(ref, ref, tk)[0]) * len(tk)
            with torch.no_grad():
                for j in range(0, len(names), 4):
                    lp = steered_logprobs(model, tk, S[j:j + 4])
                    for k, n in enumerate(names[j:j + 4]):
                        nll, kl = nll_kl(lp[k], ref, tk)
                        sums[n][0] += float(nll) * len(tk)
                        sums[n][1] += float(kl) * len(tk)
            count += len(tk)
        for n, (nll, kl) in sums.items():
            out[n][u] = {"nll": nll / count, "kl": kl / count}
    return out


def verdict(m: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, object]:
    """The pilot's criteria, fixed before it ran (notes.md): each steer
    reaches at least half its control's own-task efficacy, with mean KL on
    the other tasks' text at most a quarter of its own efficacy."""
    def eff(cond: str, task: str) -> float:
        return m["base"][task]["nll"] - m[cond][task]["nll"]

    def xtalk(cond: str, others: List[str]) -> float:
        return float(np.mean([m[cond][u]["kl"] for u in others]))

    rows = {}
    for t in TASKS:
        others = [u for u in TASKS if u != t]
        rows[f"shared:{t}"] = {"eff": eff(f"shared:{t}", t), "control": eff(f"single:{t}", t),
                               "xtalk": xtalk(f"shared:{t}", others)}
    a, b, c = COMPOSE
    rows["v1"] = {"eff": eff("v1", a), "control": eff(f"single:{a}", a), "xtalk": xtalk("v1", [b, c])}
    rows["v2"] = {"eff": eff("v2", b), "control": eff(f"single:{b}", b), "xtalk": xtalk("v2", [a, c])}
    rows["v1+v2"] = {"eff": eff("v1+v2", c), "control": eff(f"single:{c}", c),
                     "xtalk": xtalk("v1+v2", [a, b])}
    for r in rows.values():
        r["pass"] = bool(r["control"] > 0 and r["eff"] >= 0.5 * r["control"]
                         and r["xtalk"] <= 0.25 * max(r["eff"], 0.0))
    return {"rows": rows,
            "intensity_pass": all(rows[f"shared:{t}"]["pass"] for t in TASKS),
            "composition_pass": all(rows[k]["pass"] for k in ("v1", "v2", "v1+v2"))}


def main() -> None:
    cfg = parse_args()
    out = cfg.out / f"seed{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    dev = device_for(cfg.device)
    model = load_gpt2(dev)

    data = build(cfg.T, cfg.seed)
    sizes = {k: (len(a), len(b)) for k, (a, b) in data.items()}
    print("windows (train, test):", sizes, flush=True)

    steers = Steers(model.config.n_embd, cfg.alpha_init).to(dev)
    train(model, steers, data, cfg, dev, out / "loss.csv")

    m = evaluate(model, steers, data, cfg.eval_batch, dev)
    v = verdict(m)
    H = capture(model, torch.as_tensor(data["neutral"][1][:32], device=dev))
    site = float(H.norm(dim=-1).median())
    with torch.no_grad():
        conds = steers.conditions()
        size = {k: float(t.norm()) / site for k, t in conds.items()}
    np.savez(out / "vectors.npz", v=steers.v.detach().cpu().numpy(),
             alphas=steers.alphas().detach().cpu().numpy(),
             v1=steers.v1.detach().cpu().numpy(), v2=steers.v2.detach().cpu().numpy(),
             U=steers.U.detach().cpu().numpy(), tasks=np.array(TASKS))
    (out / "metrics.json").write_text(json.dumps(
        {"metrics": m, "verdict": v, "size_over_site_median": size,
         "site_median_norm": site, "windows": sizes, "args": {
             k: (str(x) if isinstance(x, Path) else x) for k, x in vars(cfg).items()}},
        indent=1))

    print(f"\nalphas {np.round(steers.alphas().tolist(), 3).tolist()}; "
          f"steer norm / site median {site:.1f}:",
          {k: round(x, 3) for k, x in size.items()})
    print("\nefficacy (NLL drop, nats/token), rows = steer, columns = text:")
    print(f"{'':>15}" + "".join(f"{u:>10}" for u in TEXTS))
    for n in [k for k in m if k != "base"]:
        print(f"{n:>15}" + "".join(
            f"{m['base'][u]['nll'] - m[n][u]['nll']:>10.3f}" for u in TEXTS))
    print("\ncriteria:", json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                         for kk, vv in r.items()}
                                     for k, r in v["rows"].items()}, indent=1))
    print("intensity pass:", v["intensity_pass"], "| composition pass:", v["composition_pass"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
