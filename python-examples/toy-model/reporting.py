"""A black box that reports its own error bar, and a case that uses it.

GX hands a transport solver a time-averaged flux. The average and its error bar
come from QUENDS (https://github.com/sandialabs/quends): it trims the initial
transient, takes the mean over the statistically steady segment, and reports
`mean_uncertainty`, the standard error of autotuned, independent block means.
If it cannot find a steady segment, it returns the mean of the last third of the
series with a 100% relative uncertainty. Every flux tube is its own GX run, so
all of this happens independently at every point on every call.

`QuendsFlux` stands in for that, on the toy model's flux. At each point of each
call:

* the run has `n` effective independent samples, log-uniform on
  [`n_min`, `n_max`]. The true standard error is `eps |sigma| sqrt(n_ref / n)`,
  with `n_ref` the geometric mean, so `eps` is the median relative error and
  `n_max / n_min` sets how much it varies;
* the reported error is QUENDS's estimate of it from `n` block means, so it
  carries a chi factor with `n - 1` degrees of freedom;
* with probability `p_irregular` there is no steady segment. The mean is then
  biased by a uniform relative draw on +-`irregular_bias`, and the reported
  error is 100% of it.

`FilteredCase` passes every value call through a scalar Kalman filter at each
point. The filter's estimate is carried from the last state to this one by the
flux's derivatives (trapezoidal in u and q). It is given process noise
`kappa` times the size of that predicted change, so that it trusts old samples
only while the state has stopped moving. It is then updated with the new sample.
The variance the filter assigns that sample is the arm under test:

* "raw"      -- no filter: MaNTA sees each sample as it comes, as today;
* "uniform"  -- one global level, `assumed_eps` times the estimate. This is the
                best that can be done without the error bar, and it is given
                the true median level;
* "weighted" -- the error bar the box reported, times `misreport`.

Two choices of derivative are independent of the arm. `jacobian` is the
Jacobian MaNTA's Newton iteration uses, as in `BlackBoxCase`: "analytic" (the
control) or differenced. `filter_jacobian` is what carries the filter between
states:

* "exact" -- the box's exact derivatives at both ends, which a real box does
             not have: the control;
* "build" -- the derivatives of MaNTA's last Jacobian build at those points,
             which cost nothing more. Before the first build there are none,
             and each sample is then taken as it stands.

**A difference the filter reuses must not share a sample with it.** By
default `BlackBoxCase` differences from the last value call's sample, the one
the filter has just absorbed. The derivative's error is then that sample's
noise divided by h, and the filter's prediction carries it back into the
estimate. That correlation drifts the solve to about 0.07 eps, where
independent errors in J of up to 30% cost the filter nothing. `fresh_base`
differences from a sample of its own, which costs one call per build and is
never shown to the filter.

`significance` rejects any difference whose standard error,
`sqrt(s_base^2 + s_pert^2) / h`, exceeds that fraction of its value. The point
then keeps the previous build's derivative, if there is one. `s` is the arm's
view of a sample's error: the reported error for "weighted", and `assumed_eps`
times the sample otherwise. A flagged run reports 100%, so only "weighted" can
see it.
"""

import numpy as np

from blackbox import BlackBoxCase, ToyFlux
from toy_model import A, B, C, D


class QuendsFlux(ToyFlux):
    def __init__(self, noise=1.0e-2, seed=0, n_min=4, n_max=64,
                 p_irregular=0.0, irregular_bias=0.5):
        super().__init__(noise=noise, seed=seed)
        self.n_min, self.n_max = n_min, n_max
        self.p_irregular = p_irregular
        self.irregular_bias = irregular_bias
        self.irregular = 0              # samples QUENDS would flag

    def sample(self, u, q, x):
        """(mean, source, reported standard error), each (nPoints, nVars)."""
        self.calls += 1
        self.points += u.shape[0]
        exact = self.flux(u, q)
        rng = self.rng
        shape = exact.shape
        n_ref = np.sqrt(self.n_min * self.n_max)
        n = np.exp(rng.uniform(np.log(self.n_min), np.log(self.n_max), shape))
        sem = self.noise * np.abs(exact) * np.sqrt(n_ref / n)
        mean = exact + sem * rng.standard_normal(shape)
        dof = np.maximum(np.rint(n) - 1.0, 1.0)
        reported = sem * np.sqrt(rng.chisquare(dof) / dof)

        bad = rng.random(shape) < self.p_irregular
        if bad.any():
            self.irregular += int(np.count_nonzero(bad))
            bias = rng.uniform(-self.irregular_bias, self.irregular_bias, shape)
            mean = np.where(bad, exact * (1.0 + bias), mean)
            reported = np.where(bad, np.abs(mean), reported)
        self.last_reported = reported
        return mean, self.source(x), reported

    def __call__(self, u, q, x):
        mean, source, _ = self.sample(u, q, x)
        return mean, source

    def flux(self, u, q):
        return A * u ** -1.5 * q

    def source(self, x):
        return np.reshape(D * np.exp(-(x - C) ** 2 / B), (-1, 1))


class _PointFilter:
    """The state of one batch shape's filters: one scalar filter per point."""

    def __init__(self, u, q, dsdu, dsdq, y, var):
        self.u, self.q, self.dsdu, self.dsdq = u, q, dsdu, dsdq
        self.est, self.var = y.copy(), var.copy()


class FilteredCase(BlackBoxCase):
    def __init__(self, box, mode="weighted", kappa=0.1, assumed_eps=None,
                 misreport=1.0, jacobian="analytic", filter_jacobian="exact",
                 fresh_base=False, significance=None, **kw):
        super().__init__(box, jacobian=jacobian, **kw)
        self.fresh_base = fresh_base
        self.significance = significance
        self.rejected_differences = 0
        if mode not in ("raw", "uniform", "weighted"):
            raise ValueError(f"unknown mode {mode!r}")
        if filter_jacobian not in ("exact", "build"):
            raise ValueError(f"unknown filter_jacobian {filter_jacobian!r}")
        self.filter_jacobian = filter_jacobian
        self.built = {}             # positions -> (dsigma/du, dsigma/dq) of the last build
        self.mode = mode
        self.kappa = kappa
        self.assumed_eps = box.noise if assumed_eps is None else assumed_eps
        self.misreport = misreport
        self.filters = {}

    def _variance(self, y, reported, prior):
        if self.mode == "weighted":
            return (self.misreport * reported) ** 2
        scale = np.abs(prior) if prior is not None else np.abs(y)
        return (self.assumed_eps * scale) ** 2

    def ComputePhysics(self, states, positions, t):
        u, q, x = self._unpack(states, positions)
        y, source, reported = self.box.sample(u, q, x)
        self.value_calls += 1
        self._last = (t, u, q, x, y, source)
        self._last_reported = reported
        if self.mode == "raw":
            return [list(y.T), list(source.T), []]

        key = x.tobytes()
        if self.filter_jacobian == "exact":
            dsdu, dsdq = (d[:, :, 0] for d in self.box.derivatives(u, q, x))
        else:
            dsdu, dsdq = self.built.get(key, (None, None))
        f = self.filters.get(key)
        if f is None or dsdu is None:
            f = _PointFilter(u, q, dsdu, dsdq, y, self._variance(y, reported, None))
            self.filters[key] = f
        else:
            dsdu0 = dsdu if f.dsdu is None else f.dsdu
            dsdq0 = dsdq if f.dsdq is None else f.dsdq
            change = 0.5 * ((dsdu0 + dsdu) * (u - f.u) + (dsdq0 + dsdq) * (q - f.q))
            f.est += change
            f.var += (self.kappa * change) ** 2
            r = self._variance(y, reported, f.est)
            total = f.var + r
            # Both vanish where the flux does -- q = 0 at the initial state --
            # and the sample is then as good as anything.
            gain = np.divide(f.var, total, out=np.ones_like(total), where=total > 0.0)
            f.est += gain * (y - f.est)
            f.var *= 1.0 - gain
            f.u, f.q, f.dsdu, f.dsdq = u, q, dsdu, dsdq
        return [list(f.est.T), list(source.T), []]

    def _error(self, y, reported):
        if self.mode == "weighted":
            return self.misreport * reported
        return self.assumed_eps * np.abs(y)

    def _difference(self, u, q, x, t):
        """As BlackBoxCase's, from a fresh base if asked and with the significance
        test applied. One channel; the source does not depend on u or q."""
        assert u.shape[1] == 1, "written for one channel"
        if self.fresh_base:
            y0, _, s0 = self.box.sample(u, q, x)
            self.fd_calls += 1
        else:
            calls = self.fd_calls
            y0, _ = self._base(u, q, x, t)
            s0 = self.box.last_reported if self.fd_calls > calls else self._last_reported
        s0 = self._error(y0, s0)
        previous = self.built.get(x.tobytes())

        def derivative(up, qp, h, slot):
            y, _, s = self.box.sample(up, qp, x)
            self.fd_calls += 1
            d = (y - y0) / h[:, None]
            if self.significance is not None:
                se = np.sqrt(s0 ** 2 + self._error(y, s) ** 2) / h[:, None]
                bad = se > self.significance * np.abs(d)
                if previous is not None:
                    d = np.where(bad, previous[slot], d)
                self.rejected_differences += int(np.count_nonzero(bad))
            return d

        n = u.shape[0]
        zero = np.zeros((n, 1, 1))
        h = self.dz * np.abs(u[:, 0])
        qp = q.copy(); qp[:, 0] += h
        dsdq = derivative(u, qp, h, 1)
        if self.jacobian == "full":
            h = self.du * np.abs(u[:, 0])
            up = u.copy(); up[:, 0] += h
            dsdu = derivative(up, q, h, 0)
        else:
            dsdu = np.zeros((n, 1))
        return dsdu[:, :, None], dsdq[:, :, None], zero, zero.copy()

    def ComputePhysicsDerivatives(self, states, positions, t):
        rows = super().ComputePhysicsDerivatives(states, positions, t)
        flux = rows[0][0]
        self.built[np.asarray(positions, dtype=float).tobytes()] = (
            np.array(flux["Variable"], dtype=float), np.array(flux["Derivative"], dtype=float))
        return rows
