# nc-parallel: the nc toy problem on an N-wide machine

The nc toy problem from [`../nc-toy-model/`](../nc-toy-model/),

    -d/dx[ 2 x D u^n u' ] = H exp(-x^2 / W),   sigma(0) = 0,   u(1) = u_b,

written as a `manta.jax.VectorizedTransportSystem` whose flux runs on a pool of
N worker threads. It shows two things:

* **how a case uses MaNTA's evaluation plan to lay out its own work**, and
* **what `PhysicsParallelism` buys** when MaNTA is told how wide that machine is.

    python nc_parallel.py [N]        # N defaults to 64

Needs `manta[jax]`. Nothing in CI runs it, so the numbers below are what it
printed when it was written.

## The machine, and where the case changes the plan

MaNTA hands a batched case each batch of M points in one call, and before its
first call it hands over the *plan*: every batch it will make, with its entry
point, its cadence and its exact points
([`docs/physics_interface.rst`](../../docs/physics_interface.rst), "Evaluation
plans"). This case does not evaluate a batch as one call. It **redistributes**
it: M points become N shares of `ceil(M / N)` points, one vectorised JAX call per
worker, run side by side. A batch therefore takes `ceil(M / N)` point-times on
the longest share, which is exactly the cost `PhysicsParallelism = N` describes.

That redistribution is decided in two places, and the comments there carry the
detail:

* **`prepareEvaluation(plan)`** turns MaNTA's plan into the case's own. For every
  batched site it maps the batch size M to a share size, and compiles the one
  share shape each M needs, so that no compile lands inside the solve. MaNTA
  calls it again only when the plan changes -- a new mesh from MeshAdaptation, a
  new degree from the degree loop -- and the case rebuilds its layout from the
  new plan. That is also why it can declare `regrid = manta.Regrid.InPlace`:
  nothing else it holds depends on the mesh.
* **`_redistribute`**, called by `ComputePhysics` and `ComputePhysicsDerivatives`,
  carries it out: worker w takes points `[w * share, (w + 1) * share)`, the last
  share is padded to the compiled shape by repeating its final point, and the
  padding is dropped as the shares are reassembled in MaNTA's order. A batch the
  plan did not announce is refused, not run, because that would be a broken
  promise rather than something to accommodate.

The workers left with no points are idle slots. `PhysicsParallelism` exists to
fill them.

## What telling MaNTA the width buys

The same machine runs the problem twice: with `PhysicsParallelism = 1`, MaNTA told
nothing, and with `PhysicsParallelism = N`. Both start from 5 cells at k = 4 with
a Diffusive tau, `MeshAdaptation` and `DegreeTolerance = 1e-6`:

| N | MaNTA told | levels (cells, k) | rounds | occupied | relative L_inf |
|---|---|---|---|---|---|
| 64 | nothing | (5,4) (5,4) (5,7) (5,9) (5,10) | 193 | 46% | 1.69e-5 |
| 64 | N = 64 | (5,4) (10,4) (10,10) | 180 | 51% | 1.69e-5 |
| 16 | nothing | (5,4) (5,4) (5,7) (5,9) (5,10) | 391 | 90% | 1.69e-5 |
| 16 | N = 16 | (5,4) (5,4) (5,7) (5,10) | 365 | 91% | 1.69e-5 |

The first level of each row is the uniform sample, the second the graded mesh.
At 64, MaNTA warns that the configured 30 points fill 47% of a round, grades 10
cells instead of 5 for the same round, and then raises k straight to 10 where the
error rule asked for 6: two degree levels instead of four, 7% fewer rounds. At 16
the sample's 30 points already fill two rounds, so the mesh is not filled, but
the last raise is, which saves one level. At 8 nothing fills at all: 30 points
fill four rounds of 8 almost exactly, and so does every level after.

The final error is the same in every row, because every run ends at k = 10 with
the same wall cell, and that cell is what limits it here (see
[`docs/adaptivity.rst`](../../docs/adaptivity.rst), "Machines that evaluate N
points at once"). What filling buys is reaching that answer in fewer solves.

**The wall-clock column is not the point, and the script prints it only so that
nobody mistakes it for one.** This flux costs almost nothing, so the time goes on
Python dispatching tiny tasks to threads -- 64 workers take longer than 8 for
the same problem. Rounds measure a machine whose flux is what a run waits for,
which is the only kind on which `PhysicsParallelism` is worth setting.

**One trap, met while writing this.** At `SteadyStateTolerance = 1e-9` the
filled 10-cell graded mesh could not converge: its residual stalled at 1.02e-9,
just above the tolerance, until `MaxRejectedSteps` stopped it. The cold retry
then cost six times the rounds the whole unfilled run did (1385 against 231). A finer level has a
higher round-off floor on the residual, and a fill can choose a finer level than
any the configuration names, so set the tolerance above the floor of the finest
level a fill might reach -- here 1e-8.

## `nc_multigpu.py`: the same problem on JAX devices, with a batch cap

    python nc_multigpu.py
    XLA_FLAGS=--xla_force_host_platform_device_count=8 python nc_multigpu.py

Here the machine is every JAX device on the host. One round is one jitted call on
`WIDTH = devices x EVALS_PER_DEVICE` points (sharded across the devices once
`data_sharding` is set), and a batch of M points is padded to whole rounds and
run as `ceil(M / WIDTH)` of them through `manta.jax.vmap_batched`. Two keys
describe that machine, and the module docstring and the comments in `run()` say
why each is set the way it is:

* **`PhysicsParallelism = WIDTH`** is what a batch costs. It is advice: the run
  is right without it, and with it the controllers fill each level to the rounds
  it already pays for.
* **`MaxPhysicsBatch = MAX_ROUNDS x WIDTH`** is what a batch may be, because the
  case holds a whole batch on the devices at once. It is a limit, so it is set on
  both runs: the degree loop's ceiling becomes the highest degree within it, no
  fill passes it, and a configured level past it is refused before the case is
  asked for a point.

`prepareEvaluation` builds each batch's padded parameters from the plan's own
points, keyed by batch size: under `Superconvergent` the residual, the Jacobian
and the initial condition share the k+2 star nodes, and the Diffusive tau adds
the faces.
