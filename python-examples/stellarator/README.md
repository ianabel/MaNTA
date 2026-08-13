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

The notebook lives here rather than in a notebooks directory of its own because
it imports `stellarator2`, `objective2` and `yancc_wrapper2` — it has to sit
beside them.
