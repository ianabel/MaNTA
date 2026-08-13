# Stellarator transport, coupled to DESC and yancc

Neoclassical transport in a stellarator, with the transport coefficients coming
from a drift-kinetic solve (`yancc`) on an equilibrium from `desc`, and the
whole thing differentiated so the equilibrium shape can be optimised against a
transport objective.

## Status: unverified

These scripts need `desc`, `yancc` and `interpax`, none of which are in the
project's `requirements.txt` and none of which are installable in the
environment this repository is tested in. Their imports were repointed at the
`manta.jax` package layout by inspection and by a parse check; **they have not
been run since**.

Three specific things to expect:

* `run.conf` asks for a `StellaratorTransport` case from `stellarator.py`, which
  defines no `registerTransportSystems()`. The config has not run in that form.
  It was left as found rather than repaired on a guess, since nothing here can
  check the repair.
* `stellarator.py` and `stellarator2.py` need an `XLA_FFI` build
  (`make python XLA_FFI=on`), because they use `manta.jax.FFIRunner` to put the
  solver inside a JAX computation.
* There are two generations of the whole stack — `stellarator.py` /
  `objective.py` / `yancc_wrapper.py` and their `2` counterparts.
  `desc_optimize.py` and the notebook drive the second; the first is reachable
  only from `desc_optimize-vp.ipynb`. How they differ is not recorded anywhere,
  so both were kept.

## Layout

| File | What it is |
|---|---|
| `stellarator.py`, `stellarator2.py` | The transport case, built on `manta.jax.FFIRunner` and `manta.jax.State` |
| `yancc_wrapper.py`, `yancc_wrapper2.py` | Drift-kinetic solves through `yancc`, wrapped as differentiable functions |
| `objective.py`, `objective2.py` | The optimisation objective, as a DESC objective |
| `desc_optimize.py` | The driver: build an equilibrium, optimise it against the objective |
| `desc_optimize-vp.ipynb` | The same, interactively, with plots |
| `desc_optimize_bfgs.py` | Optimise with BFGS rather than DESC's default, with the dG/dt gate armed |
| `scan_eq.py` | Scan an equilibrium parameter, solving transport at each point |
| `scan_eq_ambipolar.py` | The same scan for the multi-channel ambipolar case |
| `stellarator_example.py` | One multi-channel solve, no scan — the smallest example of that case |
| `yancc_gpu_test.py` | A `yancc` GPU sharding benchmark; touches no transport case |
| `stellarator_state.py` | Named accessors for a multi-channel state. Orphaned — see below |

The notebook lives here rather than in a notebooks directory of its own because
it imports `stellarator2`, `objective2` and `yancc_wrapper2` — it has to sit
beside them.

## The six files taken from `origin/optimize-mode`

The bottom six rows came from `python/` on that branch, which forked at
`9af1105` (2026-07-27) — before `python/` became the `manta` package. Their
imports were repointed (`from State import State` → `from manta.jax import
State`, `Stellarator2`/`Objective2` → the lowercase modules here, `import
MaNTA` → `import manta as MaNTA`) and they were parse-checked. Like everything
else here, **they have not been run.**

They are *not* equally close to working, and the differences are not visible
from the imports:

| | against this directory's `stellarator2.py` |
|---|---|
| `yancc_gpu_test.py` | Closest. Uses `manta.jax.State` and `yancc_wrapper2` only — no transport case, so nothing below applies to it |
| `scan_eq.py` | Config matches `StellaratorParams.from_config`. Needed `use_chunking`, which `stellarator2.py` reads and the branch had no notion of; added as `False`, which is what it ran with |
| `desc_optimize_bfgs.py` | As above, and its `"optimizeMode": True` became `"ObjectiveDecreaseTolerance": 0.05` — see the comment at that line |
| `scan_eq_ambipolar.py`, `stellarator_example.py` | **Cannot run as they stand.** Their `st_config` is written against `stellarator_state.StellaratorParams` (`ParticleSourceCenter`, `HeatSourceCenter`, …, `evolveDensity: True`), not the `SourceCenter`/`SourceHeight`/`SourceWidth` that `stellarator2.py` reads, so construction fails on the first missing key |
| `stellarator_state.py` | Imported by nothing. It belongs to the branch's rewritten `Stellarator2.py` |

The last three go together: the branch rewrote `Stellarator2.py` to evolve
density, ion energy and electron energy with an ambipolar Er, and
`stellarator_state.py`, `scan_eq_ambipolar.py` and `stellarator_example.py` are
that rewrite's state module and its two drivers. **The rewrite itself was
deliberately not merged** — it is written against the pre-`SystemSpec` interface
main has removed, so taking it would undo that migration. `TODO` records what
taking it properly would involve.

`scan_eq_ambipolar.py` was left as found rather than repointed at
`stellarator2.py`'s parameter names, for the same reason `run.conf` was: the
config it asks for describes physics (a split particle and heat source, three
evolved channels) that `stellarator2.py` does not have, so translating the key
names would produce something that runs and does not mean what it says.

`optimizeMode` was translated rather than left, because unlike the above it is a
plain rename: main has the same dG/dt early-exit gate, keyed
`ObjectiveDecreaseTolerance` and with the branch's hardcoded `stoptol = 0.05`
exposed as the value.
