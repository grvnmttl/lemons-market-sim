# lemons-market-sim

An agent-based **simulation** that illustrates the equilibrium results in Appendix A of the paper
*"Capability Advertisement as a Market for Lemons: A Trust Layer for Heterogeneous Agent Networks."*

## What this is — and is not

This simulation **illustrates** the proofs; it does **not validate** them against the real world.
No language model is ever queried. Each provider's reliability is a stipulated parameter, the
correctness oracle is assumed, and every number it emits is a consequence of the model's own
assumptions. It is a visualization of dynamics that Theorems 1 and 2 establish analytically —
useful for intuition and figures, worthless as evidence about any real deployment.

## Model (matches Appendix A)

Providers choose whether to **invest** in genuine reliability (becoming the high type `r_H` at an
idiosyncratic cost `kappa_i`, else the low type `r_L`) and what to **advertise**. The trust price is
`B(rho) = rho`, so the overclaim gain is `g = r_H - r_L`.

- **Faith-based regime (Theorem 1).** Advertising is cheap talk; a `hi`-claimer is perceived at the
  average true reliability of all `hi`-claimers. Investing is private cost with no signalling benefit,
  so the market unravels to `r_L` (Corollary 1.1).
- **Trust-Layer regime (Theorem 2).** Claiming `hi` needs a credential: a true `H` pays `c_H`, a true
  `L` pays `c` to fake/sustain it (single-crossing). With `c > g`, faking is unprofitable and
  providers with `kappa_i < g - c_H` invest (Corollary 2.1).

Dynamics: myopic best-response with small mutation, run over many rounds and 24 seeds (mean +/- 1 s.d.).

## Run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# or:  source .venv/bin/activate && pip install -r requirements.txt
python sim.py
```

Outputs are written to `./output/`:

- `fig1_collapse_vs_separation.png` — faith-based collapse vs. Trust-Layer separation over time (RQ1, RQ2)
- `fig2_threshold.png` — the separation threshold at `c = g` (RQ2)
- `fig3_chain_depth.png` — end-to-end reliability vs. delegation-chain depth (RQ3)
- `metrics.json`, `timeseries.csv`, `threshold.csv`, `composition.csv` — the underlying data

## Reproducibility

All randomness is seeded from `Params.master_seed` (default `20260523`). Re-running reproduces the
figures and data exactly. Parameters live in the `Params` dataclass at the top of `sim.py`.
