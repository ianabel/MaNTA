"""The nc toy problem on a machine that evaluates the flux N points at a time.

    -d/dx[ 2 x D u^n u' ] = H exp(-x^2 / W),   sigma(0) = 0,   u(1) = u_b

has a wall layer at x = 1 about 5e-4 wide at n = 2.5, which MeshAdaptation grades
towards and the degree loop then resolves.

The machine here is N worker threads. MaNTA hands the case each batch of M points
in one call; the case redistributes it across the N workers as N vectorised calls
of ceil(M / N) points each, so a batch takes ceil(M / N) point-times on the
longest share -- the cost PhysicsParallelism = N describes. The redistribution is
the case's own plan, and it is made from MaNTA's: see prepareEvaluation.

The same machine runs the problem twice: once with MaNTA told nothing
(PhysicsParallelism = 1), once told N, which lets each adaptive level be filled
to the rounds it already costs.

    python nc_parallel.py [N]        # N defaults to 64
"""

import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from scipy.integrate import quad

import manta
from manta.jax import VectorizedTransportSystem

PARAMS = dict(n=2.5, H=0.1, W=0.2, D=0.01, u_b=0.2)

# The two batched entry points this case serves. Everything else MaNTA calls --
# InitialValue, the boundary values -- is pointwise and stays on the calling
# thread.
BATCHED = (manta.EvaluationEntry.ComputePhysics, manta.EvaluationEntry.ComputePhysicsDerivatives)


class Params(eqx.Module):
    n: Float
    H: Float
    W: Float
    D: Float
    u_b: Float


class ParallelNC(VectorizedTransportSystem):
    """The nc flux on an N-thread machine.

    `regrid = InPlace`: this case keeps nothing that depends on the mesh except
    the distribution below, which prepareEvaluation rebuilds from every new plan.
    That is exactly what the adaptive drivers need to move one instance from
    level to level, so it can say so.
    """

    regrid = manta.Regrid.InPlace

    def __init__(self, width):
        # Zero flux on the axis -- sigma(0) = 0 -- rather than u'(0) = 0, which the
        # steady state does not satisfy.
        super().__init__(manta.numbered_spec(1, lower=manta.Mixed(d=1.0)))
        self.params = Params(**PARAMS)
        self.width = width
        self.pool = ThreadPoolExecutor(max_workers=width)

        # batch size M -> share, the points each worker evaluates per call. Set
        # by prepareEvaluation from MaNTA's plan; see there.
        self.share = {}

        self.levels = []   # (cells, k) of each plan MaNTA handed over
        self.rounds = 0    # point-times the machine spent: the longest share, per batch
        self.points = 0    # points it was asked for

    # --- the physics, written once for one point; the parent vmaps it -----------

    def sigma(self, index, state, x, t, p):
        return 2 * x * p.D * state.Variable[index] ** p.n * state.Derivative[index]

    def source(self, index, state, x, t, p):
        return p.H * jnp.exp(-x * x / p.W)

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return PARAMS["u_b"]

    def InitialValue(self, index, x):
        return PARAMS["u_b"]

    # --- where this case changes the plan -----------------------------------------

    def prepareEvaluation(self, plan):
        """Turn MaNTA's plan into this machine's: one share size per batch size.

        MaNTA's plan lists every batch it will hand over: for each site, its entry
        point, how often it comes, and its exact points. MaNTA calls this before
        its first evaluation of a run, and again only when the plan changes -- a
        new mesh or degree from MeshAdaptation, the degree loop or a ladder rung.

        This case does not evaluate a batch the way MaNTA hands it over. It
        *redistributes* each one: M points become N shares of ceil(M / N) points,
        one vectorised call per worker. That is a plan of its own, and it is made
        here, from MaNTA's, for two reasons:

          * the share size is fixed by M alone, and every M this run will use is
            listed, so the distribution is decided once per plan rather than
            worked out again on each of the thousands of calls;
          * every worker's call has the share's shape, padded where M does not
            divide evenly, so each batch size needs one compiled shape -- and
            compiling it here, before MaNTA's first call, keeps the compile out of
            the solve.

        Anything not announced is refused in _redistribute: a batch outside the
        plan is a broken promise, and running it anyway would hide that.
        """
        self.levels.append((plan.grid.getNCells(), plan.k))
        self.share = {}
        for site in plan.sites:
            if site.entry in BATCHED:
                m = site.batchSize()
                share = -(-m // self.width)   # ceil(M / N)
                if share not in self.share.values():
                    self._compile(share)
                self.share[m] = share

    def _compile(self, share):
        """Trace and compile both entry points at one share's shape.

        Through the parent's ComputePhysics and ComputePhysicsDerivatives, with a
        batch in MaNTA's own dict layout, so that what is compiled is exactly the
        call a worker will make.
        """
        per_point = np.zeros((share, self.nVars))
        batch = {"Variable": per_point + PARAMS["u_b"], "Derivative": per_point,
                 "Flux": per_point, "Aux": np.zeros((share, self.nAux)),
                 "Geometry": np.zeros((share, 0)), "VariableDot": np.zeros((0, 0)),
                 "Scalars": np.zeros(self.nScalars)}
        x = np.full(share, 0.5)
        VectorizedTransportSystem.ComputePhysics(self, batch, x, 0.0)
        VectorizedTransportSystem.ComputePhysicsDerivatives(self, batch, x, 0.0)

    # --- where the redistribution is carried out -----------------------------------

    def ComputePhysics(self, states, positions, t):
        return self._redistribute(super().ComputePhysics, states, positions, t)

    def ComputePhysicsDerivatives(self, states, positions, t):
        return self._redistribute(super().ComputePhysicsDerivatives, states, positions, t)

    def _redistribute(self, evaluate, states, positions, t):
        """One batch from MaNTA, as `width` vectorised calls run side by side.

        Worker w takes points [w * share, (w + 1) * share). The last worker with
        anything to do is padded up to `share` by repeating its final point, so
        that its call has the compiled shape; the padding is dropped on the way
        back, and the shares are reassembled in their original order, which is
        the order MaNTA reads the results in. Workers with no points sit idle --
        those are the slots PhysicsParallelism exists to fill.
        """
        m = len(positions)
        if m not in self.share:
            raise RuntimeError(f"a batch of {m} points that the evaluation plan did not announce")
        share = self.share[m]

        rows = {key: np.asarray(v) for key, v in states.items()}
        per_point = {key: v.ndim == 2 and v.shape[0] == m for key, v in rows.items()}
        x = np.asarray(positions)

        def worker(start):
            take = np.minimum(np.arange(start, start + share), m - 1)   # pad by repetition
            part = {key: (v[take] if per_point[key] else v) for key, v in rows.items()}
            return evaluate(part, x[take], t), min(share, m - start)

        shares = list(self.pool.map(worker, range(0, m, share)))
        self.rounds += share
        self.points += m
        return _reassemble(shares)


def _reassemble(shares):
    """Each worker's result, padding dropped, joined back into MaNTA's one batch.

    A result is three slots -- fluxes, sources, aux -- of one entry per variable:
    an array of values from ComputePhysics, a dict of derivative arrays from
    ComputePhysicsDerivatives.
    """
    first = shares[0][0]
    out = []
    for slot in range(len(first)):
        entries = []
        for j in range(len(first[slot])):
            parts = [(result[slot][j], used) for result, used in shares]
            if isinstance(parts[0][0], dict):
                entries.append({
                    key: (np.concatenate([np.asarray(p[key])[:used] for p, used in parts])
                          if np.asarray(parts[0][0][key]).ndim == 2 else parts[0][0][key])
                    for key in parts[0][0]})
            else:
                entries.append(np.concatenate([np.atleast_1d(np.asarray(p))[:used]
                                               for p, used in parts]))
        out.append(entries)
    return out


def u_exact(x):
    """The steady state, from integrating the flux balance once."""
    n, H, W, D, u_b = (PARAMS[k] for k in ("n", "H", "W", "D", "u_b"))
    flux = lambda s: H * math.sqrt(math.pi * W) / 2 * math.erf(s / math.sqrt(W))  # noqa: E731
    f = lambda s: flux(s) / s if s > 0 else H  # noqa: E731
    integral = np.array([quad(f, xi, 1.0, epsabs=1e-14, epsrel=1e-13)[0] for xi in x])
    return (u_b ** (n + 1) + (n + 1) / (2 * D) * integral) ** (1 / (n + 1))


def run(width, told):
    case = ParallelNC(width)
    runner = manta.Runner(case)
    runner.configure({
        "OutputFilename": "nc_parallel",
        "WriteOutput": False,
        "PolynomialDegree": 4,
        "GridSize": 5,
        "LowerBoundary": 0.0,
        "UpperBoundary": 1.0,
        "tau": 1.0,
        "tauScaling": "Diffusive",
        "tauUpdate": "ContinuationStep",
        "MeshAdaptation": True,
        "DegreeTolerance": 1e-6,
        # The machine is `width` wide either way; this is whether MaNTA knows.
        "PhysicsParallelism": width if told else 1,
        "Relative_tolerance": 1e-6,
        "Absolute_tolerance": [1e-6],
        "initialTimestep": 1e-3,
        "MinStepSize": 1e-12,
        "SteadyStateSolver": "PseudoTransient",
        "SteadyStateTolerance": 1e-8,
        "delta_t": 1.0,
    })
    start = time.perf_counter()
    runner.run_ss()
    seconds = time.perf_counter() - start

    x = np.unique(np.concatenate([np.linspace(0, 1, 401)[1:-1] + 1 / 800,
                                  1 - np.geomspace(1e-6, 0.05, 200)]))
    x = x[(x > 0) & (x < 1)]
    u = np.asarray(runner.getSolution(0, list(x))).ravel()
    ue = u_exact(x)
    case.pool.shutdown()
    return dict(levels=case.levels, rounds=case.rounds,
                occupancy=case.points / (case.rounds * width),
                linf=float(np.max(np.abs(u - ue)) / np.max(ue)), seconds=seconds)


if __name__ == "__main__":
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    print(f"{width} workers; a batch of M points takes ceil(M/{width}) point-times.\n")
    print(f"{'MaNTA told':<11} {'levels (cells, k)':<34} {'rounds':>7} {'occupied':>9} "
          f"{'L_inf':>9} {'wall':>7}")
    for told in (False, True):
        r = run(width, told)
        levels = " ".join(f"({c},{k})" for c, k in r["levels"])
        print(f"{'N = ' + str(width) if told else 'nothing':<11} {levels:<34} {r['rounds']:7d} "
              f"{r['occupancy']:8.0%} {r['linf']:9.2e} {r['seconds']:6.1f}s")
