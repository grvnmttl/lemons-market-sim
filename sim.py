"""
lemons-market-sim
=================

An agent-based illustration of the equilibrium claims in Appendix A of
"Capability Advertisement as a Market for Lemons: A Trust Layer for
Heterogeneous Agent Networks".

IMPORTANT — what this is and is not
-----------------------------------
This simulation ILLUSTRATES the proofs; it does not VALIDATE them against the
real world. No language model is ever queried. Each provider's reliability is
a stipulated parameter, the correctness oracle is assumed, and every number
emitted is a consequence of the model's own assumptions. It is a visualization
of the dynamics that Theorems 1 and 2 establish analytically — useful for
intuition and figures, worthless as evidence about any real deployment.

Model (matches Appendix A)
--------------------------
Providers have a hidden type via an *investment* choice:
    invest = 1  ->  true reliability r_H   (the "high" type, costs kappa_i)
    invest = 0  ->  true reliability r_L   (the "low" type, free)
and an *advertisement* choice:
    claim 'hi' or 'lo'.

Trust price (provider benefit) is B(rho) = rho, so:
    B_H = r_H,  B_L = r_L,  overclaim gain g = B_H - B_L = r_H - r_L.

Two regimes:
  FAITH-BASED  (Theorem 1): advertising is cheap talk. A caller cannot verify,
     so a 'hi'-claimer is perceived at the *average true reliability of all
     hi-claimers* (rational-Bayesian pooling). Investing raises only your own
     reliability, a negligible nudge to the pool, so it is pure private cost ->
     no one invests (Corollary 1.1) and the hi-pool unravels toward r_L.

  TRUST-LAYER  (Theorem 2): claiming 'hi' requires a credential. A true H pays
     screening cost c_H; a true L must pay c to fake/sustain it (single-crossing,
     bundling screening + slashing + reputation forfeiture). A credential certifies
     the level, so a holder is perceived at r_H. Honest 'lo' is perceived at r_L.
     With c > g, faking is unprofitable (Theorem 2 / Corollary 2.1) and providers
     with cheap-enough investment (kappa_i <= g - c_H) invest and screen.

Dynamics: pairwise proportional imitation with a small mutation rate, run for T
rounds over S random seeds; we report mean +/- 1 std bands.

Outputs (written to ./output):
  timeseries.csv, threshold.csv, composition.csv, metrics.json
  fig1_collapse_vs_separation.png
  fig2_threshold.png
  fig3_chain_depth.png
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    r_L: float = 0.55          # true reliability of the low type
    r_H: float = 0.92          # true reliability of the high type
    c_H: float = 0.02          # screening cost for a genuine high type
    kappa_lo: float = 0.0      # min idiosyncratic investment cost
    kappa_hi: float = 0.60     # max idiosyncratic investment cost (Uniform draw)
    N: int = 2000              # number of providers
    T: int = 300               # rounds per run
    S: int = 24                # random seeds (for confidence bands)
    revision_rate: float = 0.30  # fraction revising per round
    mutation: float = 0.005      # exploration noise
    init_invest_frac: float = 0.50  # start from a "healthy" market (half invested)
    master_seed: int = 20260523

    @property
    def g(self) -> float:
        return self.r_H - self.r_L  # overclaim gain = B_H - B_L


def B(rho):
    """Trust price: strictly increasing benefit of being perceived at reliability rho."""
    return rho


# Actions encoded as integers 0..3: (invest, claim_hi)
A_LO_HONEST = 0   # invest=0, claim='lo'   -> true L, perceived L
A_LO_OVERCLAIM = 1  # invest=0, claim='hi' -> true L, perceived hi (overclaim)
A_HI_NOSCREEN = 2   # invest=1, claim='lo' -> true H, perceived L (dominated)
A_HI_HONEST = 3     # invest=1, claim='hi' -> true H, perceived hi (honest high)

INVEST = np.array([0, 0, 1, 1], dtype=bool)
CLAIM_HI = np.array([0, 1, 0, 1], dtype=bool)


# --------------------------------------------------------------------------- #
# Dynamics
# --------------------------------------------------------------------------- #
def best_respond(actions, kappa, rng, p: Params, regime: str, c: float = None):
    """Myopic best-response dynamics. Each revising agent picks the action that
    maximizes ITS OWN payoff (using its own investment cost kappa_i) given the
    current perceptions. This respects private costs — unlike action-imitation —
    and so reproduces the kappa-threshold of Corollary 2.1. Plus small mutation."""
    n = actions.shape[0]
    pm = np.empty((n, 4))  # payoff of each action for each agent

    if regime == "faith":
        invest = INVEST[actions]
        claim_hi = CLAIM_HI[actions]
        r = np.where(invest, p.r_H, p.r_L)
        perceived_hi = r[claim_hi].mean() if claim_hi.any() else p.r_L
        perceived_lo = r[~claim_hi].mean() if (~claim_hi).any() else p.r_L
        for a in range(4):
            perceived = perceived_hi if CLAIM_HI[a] else perceived_lo
            pm[:, a] = B(perceived) - kappa * INVEST[a]
    else:  # trust layer
        for a in range(4):
            if CLAIM_HI[a]:
                screen = p.c_H if INVEST[a] else c  # honest H pays c_H, faking L pays c
                pm[:, a] = B(p.r_H) - kappa * INVEST[a] - screen
            else:
                pm[:, a] = B(p.r_L) - kappa * INVEST[a]

    best = pm.argmax(axis=1)
    new = actions.copy()
    revisers = rng.random(n) < p.revision_rate
    new[revisers] = best[revisers]
    mut = rng.random(n) < p.mutation
    new[mut] = rng.integers(0, 4, size=mut.sum())
    return new


def market_reliability_faith(actions, p: Params):
    """Realized reliability obtained by a caller: it routes to the group with the
    higher *perceived* reliability; realized = that group's true mean reliability
    (which, in the pooling model, equals its perceived value)."""
    invest = INVEST[actions]
    claim_hi = CLAIM_HI[actions]
    r = np.where(invest, p.r_H, p.r_L)
    perceived_hi = r[claim_hi].mean() if claim_hi.any() else p.r_L
    perceived_lo = r[~claim_hi].mean() if (~claim_hi).any() else p.r_L
    return max(perceived_hi, perceived_lo)


def market_reliability_trust(actions, p: Params):
    """Caller routes to credential holders (perceived r_H); realized = the true
    mean reliability among hi-claimers. With c>g there are no fakers, so this is
    r_H; with c<g, fakers dilute it."""
    claim_hi = CLAIM_HI[actions]
    invest = INVEST[actions]
    r = np.where(invest, p.r_H, p.r_L)
    return r[claim_hi].mean() if claim_hi.any() else p.r_L


def run_once(regime: str, p: Params, seed: int, c: float = None):
    """One run. Returns per-round arrays of metrics."""
    rng = np.random.default_rng(seed)
    kappa = rng.uniform(p.kappa_lo, p.kappa_hi, size=p.N)

    # initialise: a "healthy" market — a fraction invested, claims match type
    invest0 = rng.random(p.N) < p.init_invest_frac
    actions = np.where(invest0, A_HI_HONEST, A_LO_HONEST).astype(int)

    rel = np.empty(p.T)
    overclaim = np.empty(p.T)
    invested = np.empty(p.T)

    for t in range(p.T):
        if regime == "faith":
            rel[t] = market_reliability_faith(actions, p)
        else:
            rel[t] = market_reliability_trust(actions, p)
        overclaim[t] = np.mean(actions == A_LO_OVERCLAIM)
        invested[t] = np.mean(INVEST[actions])
        actions = best_respond(actions, kappa, rng, p, regime, c=c)

    return dict(rel=rel, overclaim=overclaim, invested=invested)


def run_many(regime: str, p: Params, c: float = None):
    """Run S seeds, stack metrics -> shape (S, T)."""
    seeds = np.random.default_rng(p.master_seed).integers(0, 2**31 - 1, size=p.S)
    out = {k: [] for k in ("rel", "overclaim", "invested")}
    for s in seeds:
        r = run_once(regime, p, int(s), c=c)
        for k in out:
            out[k].append(r[k])
    return {k: np.vstack(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def band(ax, series, label, color):
    m = series.mean(axis=0)
    sd = series.std(axis=0)
    x = np.arange(series.shape[1])
    ax.plot(x, m, label=label, color=color, lw=2)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.18)


def experiment_1(p: Params, outdir: Path):
    """E1 / RQ1 & RQ2: collapse under faith vs separation under the layer (c=1.5g)."""
    faith = run_many("faith", p)
    trust = run_many("trust", p, c=1.5 * p.g)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    band(ax1, faith["rel"], "Faith-based", "#c1121f")
    band(ax1, trust["rel"], "Trust Layer (c = 1.5g)", "#2a9d8f")
    ax1.axhline(p.r_L, ls=":", c="grey", lw=1)
    ax1.axhline(p.r_H, ls=":", c="grey", lw=1)
    ax1.text(p.T * 0.55, p.r_L + 0.006, "r_L", color="grey")
    ax1.text(p.T * 0.55, p.r_H - 0.02, "r_H", color="grey")
    ax1.set_xlabel("round")
    ax1.set_ylabel("realized reliability of selected provider")
    ax1.set_title("(a) Market reliability over time")
    ax1.set_ylim(p.r_L - 0.05, p.r_H + 0.05)
    ax1.legend(loc="center right", fontsize=9)

    band(ax2, faith["invested"], "Faith-based", "#c1121f")
    band(ax2, trust["invested"], "Trust Layer (c = 1.5g)", "#2a9d8f")
    ax2.axhline((p.g - p.c_H) / (p.kappa_hi - p.kappa_lo), ls=":", c="grey", lw=1)
    ax2.text(p.T * 0.40, (p.g - p.c_H) / (p.kappa_hi - p.kappa_lo) + 0.02,
             "predicted invest share  P(kappa < g - c_H)", color="grey", fontsize=8)
    ax2.set_xlabel("round")
    ax2.set_ylabel("fraction investing in real reliability (high type)")
    ax2.set_title("(b) High-quality provider share over time")
    ax2.set_ylim(-0.03, 1.03)
    ax2.legend(loc="center right", fontsize=9)

    fig.suptitle("Faith-based advertising collapses to lemons (Thm 1); the Trust Layer separates (Thm 2)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "fig1_collapse_vs_separation.png", dpi=150)
    plt.close(fig)

    # write timeseries (means)
    with open(outdir / "timeseries.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round",
                    "faith_reliability_mean", "faith_overclaim_mean", "faith_invested_mean",
                    "trust_reliability_mean", "trust_overclaim_mean", "trust_invested_mean"])
        for t in range(p.T):
            w.writerow([t,
                        faith["rel"][:, t].mean(), faith["overclaim"][:, t].mean(),
                        faith["invested"][:, t].mean(),
                        trust["rel"][:, t].mean(), trust["overclaim"][:, t].mean(),
                        trust["invested"][:, t].mean()])

    return faith, trust


def experiment_2(p: Params, outdir: Path):
    """E2 / RQ2: sweep c/g and locate the separation threshold at c = g."""
    ratios = np.linspace(0.0, 2.0, 41)
    rel_mean, rel_sd, oc_mean, inv_mean = [], [], [], []
    for ratio in ratios:
        res = run_many("trust", p, c=ratio * p.g)
        tail = slice(int(p.T * 0.75), p.T)  # steady-state tail
        rel_mean.append(res["rel"][:, tail].mean())
        rel_sd.append(res["rel"][:, tail].mean(axis=1).std())
        oc_mean.append(res["overclaim"][:, tail].mean())
        inv_mean.append(res["invested"][:, tail].mean())
    ratios, rel_mean, rel_sd = np.array(ratios), np.array(rel_mean), np.array(rel_sd)
    oc_mean, inv_mean = np.array(oc_mean), np.array(inv_mean)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ratios, rel_mean, color="#264653", lw=2, label="market reliability")
    ax.fill_between(ratios, rel_mean - rel_sd, rel_mean + rel_sd, color="#264653", alpha=0.15)
    ax.plot(ratios, oc_mean, color="#c1121f", lw=2, ls="--", label="fraction overclaiming")
    ax.axvline(1.0, color="black", ls=":", lw=1.5)
    ax.text(1.03, 0.45, "c = g\n(separation threshold)", fontsize=9)
    ax.set_xlabel("screening cost ratio  c / g")
    ax.set_ylabel("steady-state value")
    ax.set_title("RQ2: separation appears exactly when c exceeds g")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="center left", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_threshold.png", dpi=150)
    plt.close(fig)

    with open(outdir / "threshold.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["c_over_g", "reliability_mean", "reliability_sd",
                    "overclaim_mean", "invested_mean"])
        for i in range(len(ratios)):
            w.writerow([ratios[i], rel_mean[i], rel_sd[i], oc_mean[i], inv_mean[i]])

    return ratios, rel_mean, oc_mean


def experiment_3(p: Params, faith, trust, outdir: Path):
    """E3 / RQ3: end-to-end reliability vs delegation-chain depth."""
    tail = slice(int(p.T * 0.75), p.T)
    r_faith = faith["rel"][:, tail].mean()
    r_trust = trust["rel"][:, tail].mean()
    depths = np.arange(1, 9)
    v = 0.5  # verification catch-rate per hop in the layer

    e2e_faith = r_faith ** depths
    e2e_trust = r_trust ** depths
    r_trust_verified = 1 - (1 - r_trust) * (1 - v)
    e2e_trust_verified = r_trust_verified ** depths

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(depths, e2e_faith, "o-", color="#c1121f", lw=2,
            label=f"Faith-based (per-hop {r_faith:.2f})")
    ax.plot(depths, e2e_trust, "s-", color="#e9c46a", lw=2,
            label=f"Trust Layer, no verification (per-hop {r_trust:.2f})")
    ax.plot(depths, e2e_trust_verified, "^-", color="#2a9d8f", lw=2,
            label=f"Trust Layer + verification v={v} (per-hop {r_trust_verified:.2f})")
    ax.set_xlabel("delegation-chain depth (hops)")
    ax.set_ylabel("end-to-end reliability")
    ax.set_title("RQ3: the layer's advantage compounds with chain depth")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_chain_depth.png", dpi=150)
    plt.close(fig)

    with open(outdir / "composition.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth", "faith", "trust_no_verify", "trust_verified"])
        for i, d in enumerate(depths):
            w.writerow([int(d), e2e_faith[i], e2e_trust[i], e2e_trust_verified[i]])

    return r_faith, r_trust, r_trust_verified


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = Params()
    outdir = Path(__file__).parent / "output"
    outdir.mkdir(exist_ok=True)

    print(f"Params: r_L={p.r_L}, r_H={p.r_H}, g={p.g:.3f}, c_H={p.c_H}, "
          f"N={p.N}, T={p.T}, seeds={p.S}")
    print("Running E1 (collapse vs separation) ...")
    faith, trust = experiment_1(p, outdir)
    print("Running E2 (threshold sweep c/g) ...")
    ratios, rel_mean, oc_mean = experiment_2(p, outdir)
    print("Running E3 (chain depth) ...")
    r_faith, r_trust, r_trust_v = experiment_3(p, faith, trust, outdir)

    tail = slice(int(p.T * 0.75), p.T)
    metrics = {
        "params": asdict(p),
        "overclaim_gain_g": p.g,
        "E1_steady_state": {
            "faith_reliability": float(faith["rel"][:, tail].mean()),
            "faith_overclaim_frac": float(faith["overclaim"][:, tail].mean()),
            "faith_invested_frac": float(faith["invested"][:, tail].mean()),
            "trust_reliability": float(trust["rel"][:, tail].mean()),
            "trust_overclaim_frac": float(trust["overclaim"][:, tail].mean()),
            "trust_invested_frac": float(trust["invested"][:, tail].mean()),
        },
        "E2_threshold": {
            "reliability_at_c_below_g": float(rel_mean[np.argmin(np.abs(ratios - 0.5))]),
            "reliability_at_c_eq_g": float(rel_mean[np.argmin(np.abs(ratios - 1.0))]),
            "reliability_at_c_above_g": float(rel_mean[np.argmin(np.abs(ratios - 1.5))]),
            "overclaim_at_c_below_g": float(oc_mean[np.argmin(np.abs(ratios - 0.5))]),
            "overclaim_at_c_above_g": float(oc_mean[np.argmin(np.abs(ratios - 1.5))]),
        },
        "E3_chain_depth": {
            "per_hop_faith": float(r_faith),
            "per_hop_trust": float(r_trust),
            "per_hop_trust_verified": float(r_trust_v),
            "e2e_depth5_faith": float(r_faith ** 5),
            "e2e_depth5_trust_verified": float(r_trust_v ** 5),
        },
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Steady-state summary ===")
    print(json.dumps(metrics["E1_steady_state"], indent=2))
    print("Threshold (reliability):",
          {k: round(v, 3) for k, v in metrics["E2_threshold"].items()})
    print("Chain depth:", {k: round(v, 3) for k, v in metrics["E3_chain_depth"].items()})
    print(f"\nWrote figures + data to {outdir}")


if __name__ == "__main__":
    main()
