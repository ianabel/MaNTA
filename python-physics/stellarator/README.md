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
  so both were kept. `stellarator_multichannel.py` is a third, and the one place
  that is written down: see below.

## Layout

| File | What it is |
|---|---|
| `stellarator.py`, `stellarator2.py` | The transport case, built on `manta.jax.FFIRunner` and `manta.jax.State` |
| `stellarator_multichannel.py` | The same, with density, ion energy and electron energy coupled and an ambipolar `Er` |
| `yancc_wrapper.py`, `yancc_wrapper2.py` | Drift-kinetic solves through `yancc`, wrapped as differentiable functions |
| `objective.py`, `objective2.py` | The optimisation objective, as a DESC objective |
| `desc_optimize.py` | The driver: build an equilibrium, optimise it against the objective |
| `desc_optimize-vp.ipynb` | The same, interactively, with plots |
| `desc_optimize_bfgs.py` | Optimise with BFGS rather than DESC's default |
| `scan_eq.py` | Scan an equilibrium parameter, solving transport at each point |
| `scan_eq_ambipolar.py` | The same scan for the multi-channel ambipolar case |
| `stellarator_example.py` | One multi-channel solve, no scan — the smallest example of that case |
| `yancc_gpu_test.py` | A `yancc` GPU sharding benchmark; touches no transport case |
| `stellarator_state.py` | Named accessors for a multi-channel state, and that case's parameters |

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

| | how close |
|---|---|
| `yancc_gpu_test.py` | Closest. Uses `manta.jax.State` and `yancc_wrapper2` only — no transport case, so nothing below applies to it |
| `scan_eq.py` | Against `stellarator2.py`. Needed `use_chunking`, which that file reads and the branch had no notion of; added as `False`, which is what it ran with |
| `desc_optimize_bfgs.py` | As above. Its `"optimizeMode": True` has no MaNTA equivalent — see below |
| `scan_eq_ambipolar.py`, `stellarator_example.py` | Against `stellarator_multichannel.py`, which is what their `st_config` describes. `scan_eq_ambipolar.py` has one loose end — see below |
| `stellarator_state.py` | `stellarator_multichannel.py`'s state accessors and parameters |

## `stellarator_multichannel.py`

The branch also rewrote `Stellarator2.py` itself, to evolve density, ion energy
and electron energy together with an ambipolar `Er`, and
`stellarator_state.py`, `scan_eq_ambipolar.py` and `stellarator_example.py` are
that rewrite's state module and its two drivers. It is here as
`stellarator_multichannel.py`, ported to `SystemSpec`.

It was produced by a **three-way merge** against the common ancestor rather than
by hand: `stellarator2.py` here and `Stellarator2.py` there are two descendants
of the same file, one carrying main's interface migration and the other the new
physics, and merging them is what keeps both. Five hunks conflicted — the two
import blocks, the constructor, and two docstrings.

The port's own change is the constructor. `nVars`, `nAux` and the boundary flags
used to be assigned in the body:

```python
MaNTA.TransportSystem.__init__(self)
self.isUpperDirichlet = True
self.isLowerDirichlet = False
...
if self.params.evolveDensity:
    self.nVars = 3
    self.nAux = 1
```

They are read-only now, derived from the `SystemSpec` passed to `__init__`, so
the shape has to be decided before the base class exists. `buildSpec(params)`
does that, and the case is `MaNTA.TransportSystem.__init__(self,
buildSpec(params))` — the same shape as `python-physics/mirror-plasma`. The spec
also names the variables, where the old interface only counted them; the names
are `Channel`'s, because `stellarator_state.StellaratorState` indexes the
solution by that enum.

**What was checked.** `buildSpec` needs only `manta`, so it was run: it
validates, and it reproduces the branch's shape exactly — `evolveDensity` on
gives 3 variables and 1 aux, off gives 1 and 0, every field Neumann below and
Dirichlet above, matching the flags above. The variable order was asserted equal
to `Channel`'s, and a stub subclass was constructed through the same
`__init__` path. Nothing else in the file was run, and cannot be.

**Why it is a separate module and not a new `stellarator2.py`.** Its parameters
are a different class — `stellarator_state.StellaratorParams` splits the single
`SourceCenter`/`SourceHeight`/`SourceWidth` into a particle and a heat source —
so a config written for one will not construct the other. `stellarator2.py` has
five consumers configured its way (`objective2.py`, `desc_optimize.py`,
`scan_eq.py`, `desc_optimize_bfgs.py`, the notebook) and none of them can be run
here to check a conversion. Promoting this file over `stellarator2.py` is a
rename plus those configs; it was not done on a guess.

**The loose end.** `objective2.py` does `from stellarator2 import
StellaratorTransport`, hardwired, so `scan_eq_ambipolar.py` — which calls
`make_objective` as well as constructing the case directly — gets the
single-channel case out of the objective. `stellarator_example.py` does not use
the objective and has no such split. The branch's own `Objective2.py` changes
(a `solveAdjoint` flag and inlining `StellaratorFun`) were **not** taken; `TODO`
records it.

`scan_eq_ambipolar.py` was left as found rather than repointed at
`stellarator2.py`'s parameter names, for the same reason `run.conf` was: the
config it asks for describes physics (a split particle and heat source, three
evolved channels) that `stellarator2.py` does not have, so translating the key
names would produce something that runs and does not mean what it says.

`optimizeMode` has no translation, because MaNTA has no early-exit gate. A sweep
pays for every steady solve it asks for. `TODO` records what a sound early exit
would need — a bound on the objective *between steady states*, not a derivative at
the initial condition.
