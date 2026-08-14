"""Why this benchmark's error is what it is, and why the case is left alone.

    python diagnostics.py          # every section; several minutes
    python diagnostics.py 5 6      # just those sections

`benchmark.py` measures the shipped configuration. This measures the *reasons*
for what it reports, and is the evidence behind `ANALYSIS.md`. Nothing here is
part of the benchmark: it exists so that the claims in `README.md` can be
checked, and so that the several plausible-looking improvements to the case can
be seen to be improvements to MaNTA's *score* rather than to MaNTA.

  1  the shipped initial condition against the steady state it must reach
  2  four initial conditions: cost, tractability, and identical accuracy
  3  Shestakov's own start, and a piecewise-linear one
  4  the same ramp taken in w = n^(1/3), where it is bounded by construction
  5  where in x the error lives -- per-bin, at four resolutions
  6  the flux offset, and the collocation-node law that sets it
  7  refining the boundary cell alone
  8  control: the same flux law with a regular solution

Sections 6 and 8 need netCDF output; the rest do not.
"""

import sys

import numpy as np
from netCDF4 import Dataset

import manta

from shestakov_nonlinear import (ShestakovNonlinear, ExactSolution, S0,
                                 SOURCE_WIDTH, D0, LX, BOUNDARY_DENSITY)

d = SOURCE_WIDTH
N_B = BOUNDARY_DENSITY
GAMMA_WALL = S0 * d                       # the total source, hence Gamma(Lx)
C_PHYS = (S0 * d / D0) ** (1.0 / 3.0) / 3.0   # dw/de of the exact outer branch
SAMPLE = np.linspace(0.0, LX, 401)


# --------------------------------------------------------------- closed form
def exact_q(x, n_b=N_B):
    """d n_e/dx. n_e = w^3 with w piecewise as in the module docstring."""
    x = np.asarray(x, dtype=float)
    w0 = n_b ** (1.0 / 3.0)
    inner_w = w0 + (S0 / D0) ** (1.0 / 3.0) * (
        0.75 * (d ** (4.0 / 3.0) - x ** (4.0 / 3.0))
        + d ** (1.0 / 3.0) * (LX - d)) / 3.0
    w = np.where(x < d, inner_w, w0 + C_PHYS * (LX - x))
    wp = np.where(x < d,
                  -(S0 / D0) ** (1.0 / 3.0)
                  * np.power(np.maximum(x, 0.0), 1.0 / 3.0) / 3.0,
                  -C_PHYS)
    return 3.0 * w ** 2 * wp


def sigma_hat(u, q):
    return D0 * q ** 3 / u ** 2


def node_inset(h, k):
    """Distance from a cell edge to the nearest Chebyshev point of the 1st kind.

    The k+1 nodes are cos((2j+1)pi/2(k+1)); the outermost sits at
    (h/2)(1 - cos(pi/2(k+1))) ~ pi^2 h/16(k+1)^2 from the edge.
    """
    return 0.5 * h * (1.0 - np.cos(np.pi / (2.0 * (k + 1))))


# ------------------------------------------------------------- initial values
class Variant(ShestakovNonlinear):
    """The shipped case with a substituted initial condition.

    `q0` is optional: without it the derivative is central-differenced. That is
    fine for a smooth profile but not for one with a kink, where the difference
    straddles it and hands IDACalcIC a guess no polynomial would produce -- and
    the guess is load-bearing here, so a piecewise IC must supply its own.
    """

    def __init__(self, n_b, u0, q0=None, label="?"):
        super().__init__(n_b=n_b)
        self._u0 = u0
        self._q0 = q0
        self.label = label

    def InitialValue(self, index, x):
        return float(self._u0(float(x)))

    def InitialDerivative(self, index, x):
        x = float(x)
        if self._q0 is not None:
            return float(self._q0(x))
        h = 1.0e-6
        xp, xm = min(x + h, LX), max(x - h, 0.0)
        return float((self._u0(xp) - self._u0(xm)) / (xp - xm))


def ic_shipped(n_b):
    return lambda x: n_b + (1.0 - n_b) * (LX - x) ** 3 * (1.0 + 3.0 * x)


def ic_wall_matched(n_b):
    """The shipped shape with its amplitude taken from the physics instead."""
    amp = C_PHYS ** 3 / 4.0
    return lambda x: n_b + amp * (LX - x) ** 3 * (1.0 + 3.0 * x)


def ic_hot_and_matched(n_b):
    """n(0) = 1 and q(0) = 0 as the shipped one, plus the exact wall slope.

    w = w0 + c e + B e^2 + C e^3 with e = Lx - x, fixing w(x=0) = 1 (so n(0) = 1)
    and w'(x=0) = 0 (so q(0) = 0).
    """
    w0 = n_b ** (1.0 / 3.0)
    B, C = np.linalg.solve(np.array([[1.0, 1.0], [2.0, 3.0]]),
                           np.array([1.0 - w0 - C_PHYS, -C_PHYS]))

    def u0(x):
        e = LX - x
        return (w0 + C_PHYS * e + B * e ** 2 + C * e ** 3) ** 3
    return u0


def ic_exact(n_b):
    return lambda x: float(ExactSolution(x, n_b))


def ic_piecewise_linear(n_b, a, wall):
    """1 on [0,a), linear down to `wall` on (a,Lx]. Linear in n."""
    def u0(x):
        if x < a:
            return 1.0
        return wall + (1.0 - wall) * (LX - x) / (LX - a)

    def q0(x):
        return 0.0 if x < a else -(1.0 - wall) / (LX - a)
    return u0, q0


def ic_w_ramp(n_b, a, slope=None):
    """The same construction taken in w = n^(1/3): flat on [0,a), then linear.

    On a w-linear ramp sigma_hat = D0 q^3/u^2 = 27 D0 (dw/dx)^3 is *constant*.
    `slope=None` reaches w = 1 at x = a; a given slope fixes the ramp flux.
    """
    w0 = n_b ** (1.0 / 3.0)
    s = (1.0 - w0) / (LX - a) if slope is None else slope

    def u0(x):
        e = LX - min(max(x, 0.0), LX)
        return (w0 + s * min(e, LX - a)) ** 3
    return u0, -27.0 * D0 * s ** 3


# -------------------------------------------------------------------- solving
def configure(case, k=2, ncells=None, pts=None, suppress=True, write=False,
              npoints=1001, tag="diagnostics"):
    runner = manta.Runner(case)
    cfg = {
        "OutputFilename": tag,
        "Polynomial_degree": k,
        "Relative_tolerance": 1.0e-6,
        "Absolute_tolerance": 1.0e-3,
        "delta_t": 1.0e3,
        "t_final": 1.0e3,
        "SteadyStateTolerance": 1.0e-11,
        # Pinned so every table measures one algorithm; see README.md.
        "SteadyStateSolver": "TimeMarch",
        "SuppressAlgebraicError": suppress,
        "OutputPoints": npoints,
        "WriteOutput": write,
        "WriteDatFile": False,
    }
    if pts is None:
        cfg.update({"Grid_size": ncells, "Lower_boundary": 0.0,
                    "Upper_boundary": LX})
        nodes = ncells * (k + 1)
    else:
        # Grid_points supersedes the three above, but Grid_size is still required.
        cfg.update({"Grid_points": list(pts), "Grid_size": len(pts) - 1})
        nodes = (len(pts) - 1) * (k + 1)
    runner.configure(cfg)
    return runner, nodes


def solve(case, **kw):
    """Run to steady state; return (visits per node, relative L1 error in u)."""
    runner, nodes = configure(case, **kw)
    case.reset_counts()
    runner.run_ss()
    u = np.asarray(runner.getSolution(0, list(SAMPLE))).reshape(-1)
    ue = np.asarray(ExactSolution(SAMPLE, case.n_b))
    err = np.sum(np.abs(u - ue)) / np.sum(np.abs(ue))
    return (case.nFlux + case.nDeriv) // nodes, err


def solve_fields(case, tag="diagnostics", npoints=1001, **kw):
    """Run to steady state; return (x, u, q, sigma) from the netCDF output."""
    runner, _ = configure(case, write=True, npoints=npoints, tag=tag, **kw)
    runner.run_ss()
    with Dataset(f"{tag}.nc") as nc:          # close it: the next run reopens it
        g = nc.groups["n"]
        return (np.linspace(0.0, LX, npoints),
                np.array(g.variables["u"][-1, :]),
                np.array(g.variables["q"][-1, :]),
                np.array(g.variables["sigma"][-1, :]))


def attempt(fn, *a, **kw):
    """(result, None) or (None, one-line reason)."""
    try:
        return fn(*a, **kw), None
    except RuntimeError as e:
        return None, str(e).splitlines()[0]


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ------------------------------------------------------------------ section 1
def section1():
    rule("1. The shipped initial condition against the steady state")
    print("   n0 = n_b + (1-n_b)(Lx-x)^3 (1+3x),  n_b = %g" % N_B)
    print("   It meets the Dirichlet value, and q0(0) = 0 matches the Neumann")
    print("   value on the axis. The flux is another matter: the exact steady")
    print("   flux is just sigma_hat_e = -S0 min(x,d).")
    print()
    print(f"   {'x':>5} {'n0':>11} {'n_e':>11} {'ratio':>7} "
          f"{'sigma_hat0':>12} {'exact':>10} {'ratio':>8}")
    # The case's own hooks, so this table reports what the solver is handed.
    case = ShestakovNonlinear(n_b=N_B)
    for x in (0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0):
        a0 = case.InitialValue(0, x)
        q0 = case.InitialDerivative(0, x)
        ue, qe = float(ExactSolution(x, N_B)), float(exact_q(x, N_B))
        s0, se = sigma_hat(a0, q0), -S0 * min(x, d)
        print(f"   {x:5.2f} {a0:11.4e} {ue:11.4e} {a0/ue:7.2f} "
              f"{s0:12.4e} {se:10.4e} {s0/se if se else np.nan:8.1f}")

    print()
    print("   The cause is an amplitude, not a shape. Near the wall (e = Lx - x)")
    print(f"   the exact solution is (n_b^(1/3) + c e)^3 with c = {C_PHYS:.6f},")
    print(f"   so its amplitude is c^3 = {C_PHYS**3:.4e}; the shipped IC's is 4,")
    print(f"   a ratio of {4.0/C_PHYS**3:.0f}. sigma_hat is homogeneous of degree 1")
    print("   in that amplitude, so the flux carries the same factor, uniformly")
    print("   in e. At x = 0.9:")
    print(f"   {'n_b':>8} {'sigma_hat0':>13} {'exact':>10} {'ratio':>8}")
    for nb in (0.0, 1.0e-3, 1.0e-2, 1.0e-1):
        f, x = ic_shipped(nb), 0.9
        hh = 1.0e-6
        q0 = (f(x + hh) - f(x - hh)) / (2.0 * hh)
        print(f"   {nb:8.0e} {sigma_hat(f(x), q0):13.4e} {-S0*d:10.4e} "
              f"{sigma_hat(f(x), q0)/(-S0*d):8.1f}")

    print()
    print("   Implied initial dn/dt = S + d_x sigma_hat, by region:")
    fine = np.linspace(0.0, LX, 200001)
    # Vectorised copies of the case's InitialValue/InitialDerivative.
    u_f = N_B + (1.0 - N_B) * (LX - fine) ** 3 * (1.0 + 3.0 * fine)
    q_f = (1.0 - N_B) * (-3.0 * (LX - fine) ** 2 * (1.0 + 3.0 * fine)
                         + 3.0 * (LX - fine) ** 3)
    dt0 = (np.where(fine < d, S0, 0.0)
           + np.gradient(sigma_hat(u_f, q_f), fine))[1:-1]
    xi = fine[1:-1]
    for lo, hi in ((0.0, 0.1), (0.1, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0)):
        m = (xi >= lo) & (xi < hi)
        print(f"      x in [{lo:.2f},{hi:.2f}):  max |dn/dt| = "
              f"{np.max(np.abs(dt0[m])):11.4e}")


# ------------------------------------------------------------------ section 2
def section2():
    rule("2. Four initial conditions: cost, tractability, identical accuracy")
    print("   10 cells, k = 2. `visits` is physics evaluations per node, the")
    print("   quantity PERFORMANCE.md asks for, given as (flag off / flag on).")
    print()
    builders = [("A shipped", ic_shipped),
                ("B wall-matched amplitude", ic_wall_matched),
                ("E n(0)=1 + wall slope", ic_hot_and_matched),
                ("C the exact steady state", ic_exact)]
    print(f"   {'n_b':>8} {'initial condition':<26} {'off':>8} {'on':>8} "
          f"{'rel L1 error':>13}")
    for n_b in (1.0e-1, 1.0e-2, 1.0e-3, 0.0):
        print()
        for label, mk in builders:
            cells, errs = [], []
            for suppress in (False, True):
                r, why = attempt(solve, Variant(n_b, mk(n_b), label=label),
                                 ncells=10, suppress=suppress)
                cells.append(f"{r[0]:d}" if r else "fail")
                if r:
                    errs.append(r[1])
            e = f"{errs[-1]:13.3e}" if errs else " " * 13
            print(f"   {n_b:8.0e} {label:<26} {cells[0]:>8} {cells[1]:>8} {e}")
    print()
    print("   The error does not move. Accuracy is not the initial condition's")
    print("   business; cost and tractability are. B and E differ from A only in")
    print("   the wall amplitude, and that alone decides whether the run needs")
    print("   SuppressAlgebraicError.")


# ------------------------------------------------------------------ section 3
def section3():
    rule("3. Shestakov's own start, and a piecewise-linear one")
    print("   Shestakov uses n0 = 1. That is constant, so D = (n_x/n)^2 = 0 and")
    print("   d(sigma_hat)/dq = 0 everywhere: the trace system degenerates and")
    print("   IDACalcIC has nothing to solve. The obvious repair is to keep 1 on")
    print("   [0,a) and ramp down, which fixes the axis and breaks the wall --")
    print("   u ~ (Lx-x) with q ~ const gives sigma_hat ~ -(Lx-x)^-2, where the")
    print("   true solution stays finite by vanishing as (Lx-x)^3.")
    print()
    e1 = node_inset(LX / 10.0, 2)
    print(f"   What the solver sees at the innermost node of the last cell,")
    print(f"   {e1:.4e} from the wall at 10 cells, k = 2:")
    print(f"   {'a':>6} {'ramp to':>8} {'u there':>11} {'q':>10} "
          f"{'sigma_hat':>12} {'exact':>10}")
    for a in (0.1, 0.5, 0.8):
        for wall, wl in ((0.0, "0"), (1.0e-2, "n_b")):
            u = wall + (1.0 - wall) * e1 / (LX - a)
            q = -(1.0 - wall) / (LX - a)
            print(f"   {a:6.2f} {wl:>8} {u:11.4e} {q:10.4e} "
                  f"{sigma_hat(u, q):12.4e} {-S0*d:10.4e}")

    print()
    print("   Runs, 10 cells k = 2, (flag off / flag on):")
    print(f"   {'n_b':>8} {'a':>5} {'ramp to':>8} {'off':>8} {'on':>8} "
          f"{'rel L1 error':>13}")
    for n_b in (1.0e-1, 1.0e-2, 0.0):
        for a in (0.1, 0.5):
            for wall, wl in ((0.0, "0"), (n_b, "n_b")):
                if n_b == 0.0 and wl == "n_b":
                    continue          # the same thing
                u0, q0 = ic_piecewise_linear(n_b, a, wall)
                cells, errs = [], []
                for suppress in (False, True):
                    r, why = attempt(solve, Variant(n_b, u0, q0), ncells=10,
                                     suppress=suppress)
                    cells.append(f"{r[0]:d}" if r else "fail")
                    if r:
                        errs.append(r[1])
                e = f"{errs[-1]:13.3e}" if errs else " " * 13
                print(f"   {n_b:8.0e} {a:5.2f} {wl:>8} {cells[0]:>8} "
                      f"{cells[1]:>8} {e}")
    print()
    print("   The ramp carries *both* obstructions, and SUNLOGGER_INFO_FILENAME")
    print("   separates them. With the flag off every failure is IDA_ERR_FAIL")
    print("   (-3), the error test, at t = 0 with h = 1e-7 -- the same mode the")
    print("   shipped IC has, and the one the flag exists for. With the flag on")
    print("   what is left is IDA_CONV_FAIL (-4): the *corrector*, which no")
    print("   error-test setting can reach. At that wall node")
    print("   d(sigma_hat)/du = -2 sigma_hat/u ~ 5e5 and 3 q^2/u^2 ~ 1e4, so the")
    print("   linearisation Newton is handed is meaningless. Only n_b = 1e-1")
    print("   never reaches the second mode, which is why it is the one that")
    print("   runs. This is what the shipped IC buys by being 1080x too high at")
    print("   the wall: too much density there is survivable, too little is not.")


# ------------------------------------------------------------------ section 4
def section4():
    rule("4. The same ramp taken in w = n^(1/3)")
    print("   Shestakov substitutes w = n^(1/3) to linearise his own steady")
    print("   equation, and outside the source region the exact solution is")
    print(f"   exactly linear in w: w = n_b^(1/3) + c (Lx - x), c = {C_PHYS:.6f}.")
    print("   On any w-linear ramp sigma_hat = 27 D0 (dw/dx)^3 is constant, so")
    print("   the wall blow-up of section 3 cannot happen.")
    print()
    print("   W1 reaches w = 1 at x = a; W2 uses the physical slope -c, giving")
    print(f"   sigma_hat = {-S0*d:.4e} exactly. a = 0 is W2 with no flat region.")
    print()
    print(f"   {'n_b':>8} {'variant':>10} {'a':>5} {'n0(0)':>8} "
          f"{'sig ramp':>11} {'off':>8} {'on':>8} {'rel L1 error':>13}")
    for n_b in (1.0e-1, 1.0e-2, 1.0e-3, 0.0):
        print()
        specs = [("W1", 0.1, None), ("W1", 0.3, None),
                 ("W2", 0.1, C_PHYS), ("W2", 0.3, C_PHYS), ("W2", 0.0, C_PHYS)]
        for name, a, slope in specs:
            u0, sig = ic_w_ramp(n_b, a, slope)
            cells, errs = [], []
            for suppress in (False, True):
                r, why = attempt(solve, Variant(n_b, u0), ncells=10,
                                 suppress=suppress)
                cells.append(f"{r[0]:d}" if r else "fail")
                if r:
                    errs.append(r[1])
            e = f"{errs[-1]:13.3e}" if errs else " " * 13
            print(f"   {n_b:8.0e} {name:>10} {a:5.2f} {u0(0.0):8.4f} "
                  f"{sig:11.4e} {cells[0]:>8} {cells[1]:>8} {e}")
    print()
    print("   W2 with a = 0 is the cheapest start found, and the only family")
    print("   that runs with the flag *off* at every n_b > 0 -- including")
    print("   Shestakov's Section 2.2 value of 1e-3. W1 never pays: you cannot")
    print("   start at n = 1 and still reach the correct wall flux, because")
    print(f"   (1 - n_b^(1/3))/c = {(1-N_B**(1/3))/C_PHYS:.2f} > Lx. n(0) = 1 is")
    print("   simply 20x above the steady axis value and no gentle profile")
    print("   connects them over a unit interval.")


# ------------------------------------------------------------------ section 5
def section5():
    rule("5. Where in x the error lives")
    bins = np.linspace(0.0, LX, 11)
    per_bin = 400
    xs, owner = [], []
    for b in range(len(bins) - 1):
        e = (np.arange(per_bin) + 0.5) / per_bin
        xs.append(bins[b] + (bins[b + 1] - bins[b]) * e)
        owner.append(np.full(per_bin, b))
    X, OWNER = np.concatenate(xs), np.concatenate(owner)
    dx = (bins[1] - bins[0]) / per_bin
    UE = np.asarray(ExactSolution(X, N_B))
    norm = np.sum(np.abs(UE)) * dx

    print("   Ten fixed bins of width 0.1, common to every resolution, so the")
    print("   contributions and rates are comparable. Every bin edge is a cell")
    print("   boundary at all four resolutions, so the source kink at x = 0.1")
    print("   never falls inside a cell.")
    print()
    runs = [10, 20, 40, 80]
    got = {}
    for ncells in runs:
        runner, _ = configure(ShestakovNonlinear(n_b=N_B), ncells=ncells)
        runner.run_ss()
        got[ncells] = np.asarray(
            runner.getSolution(0, list(X))).reshape(-1) - UE

    print(f"   {'bin':<13}" + "".join(f"{c:>12}" for c in runs)
          + f"{'rate':>7}{'share':>8}")
    for b in range(len(bins) - 1):
        v = [np.sum(np.abs(got[c][OWNER == b])) * dx / norm for c in runs]
        tot = np.sum(np.abs(got[runs[-1]])) * dx / norm
        print(f"   [{bins[b]:.1f},{bins[b+1]:.1f})    "
              + "".join(f"{q:12.3e}" for q in v)
              + f"{np.log2(v[-2]/v[-1]):7.2f}{100*v[-1]/tot:7.1f}%")
    tot = [np.sum(np.abs(got[c])) * dx / norm for c in runs]
    print(f"   {'total':<13}" + "".join(f"{q:12.3e}" for q in tot))
    print(f"   {'rate':<13}" + " " * 12
          + "".join(f"{np.log2(tot[i]/tot[i+1]):12.2f}"
                   for i in range(len(tot) - 1)))
    print()
    print("   The error is not localised. Every bin converges at the same rate")
    print("   and the shares are constant to 0.1% over an 8x refinement, so this")
    print("   is one global mode rather than a local defect. Section 6 finds it.")


# ------------------------------------------------------------------ section 6
def section6():
    rule("6. The flux offset, and the collocation-node law behind it")
    print("   The exact stored sigma is +Gamma = S0 min(x,d): piecewise linear")
    print("   with its kink on a cell boundary, hence exactly representable in")
    print("   P_k for every k >= 1 here. Any error in it is the scheme's.")
    print()
    print(f"   {'cells':>6} {'k':>2} {'mean dsigma':>13} {'std over x>d':>13} "
          f"{'/Gamma_wall':>12} {'x(Gamma_h=0)/x1':>16} {'Gamma_h(0)/-S0x1':>17}")
    for ncells, k in [(10, 1), (10, 2), (10, 3), (10, 4), (10, 5),
                      (20, 2), (40, 2), (80, 2), (20, 4)]:
        x, u, q, sig = solve_fields(ShestakovNonlinear(n_b=N_B), ncells=ncells,
                                    k=k, npoints=20001, tag="diag_flux")
        dsig = sig - S0 * np.minimum(x, d)
        outer = x > d
        h = LX / ncells
        x1 = node_inset(h, k)
        near = x < 3.0 * h
        s = sig[near]
        idx = np.where(np.sign(s[:-1]) != np.sign(s[1:]))[0]
        if len(idx):
            i = idx[0]
            x0 = x[i] - s[i] * (x[i + 1] - x[i]) / (s[i + 1] - s[i])
        else:
            x0 = np.nan
        print(f"   {ncells:6d} {k:2d} {np.mean(dsig[outer]):13.4e} "
              f"{np.std(dsig[outer]):13.4e} "
              f"{np.mean(dsig[outer])/GAMMA_WALL:12.4e} {x0/x1:16.3f} "
              f"{sig[0]/(-S0*x1):17.3f}")
    print()
    print("   The error in sigma is a *constant* offset -- the standard")
    print("   deviation is at round-off for k >= 3 -- and it is exactly the flux")
    print("   the source deposits between the axis and the innermost collocation")
    print("   node. Gamma_h vanishes there rather than at x = 0. So the axis")
    print("   condition is effectively imposed a distance x1 inside the domain,")
    print("   and conservation spreads the resulting deficit over every cell.")
    print()
    print("   x1 = (h/2)(1 - cos(pi/2(k+1))) ~ pi^2 h/16(k+1)^2, which is where")
    print("   the observed O(h) and O((k+1)^-2) both come from. Fitted at")
    print(f"   n_b = {N_B:g}: relative L1 error in u ~ 1.8 h0/(k+1)^2, with h0 the")
    print("   width of the *boundary* cell (section 7). The coefficient rises as")
    print("   n_b falls; the sigma statement above does not depend on the fit.")
    print()
    print("   Two ingredients are needed, and section 8 separates them:")
    print("     (i)  sigma_hat = D q with D(0) = 0, so the Neumann condition")
    print("          q(0) = 0 says nothing about the flux -- any q gives zero;")
    print("     (ii) q_e ~ -C x^(1/3), whose derivative is unbounded, so no")
    print("          polynomial resolves it on the boundary cell. Measured, the")
    print("          max error in q there converges at 0.34, i.e. h^(1/3).")


# ------------------------------------------------------------------ section 7
def section7():
    rule("7. Refining the boundary cell alone")
    print("   Nine cells stay at h = 0.1; only the first is split, geometrically.")
    print()
    print(f"   {'first cell':>12} {'cells':>6} {'nodes':>6} "
          f"{'dsigma/Gamma':>13} {'visits':>7} {'rel L1 error':>13}")
    outer = list(np.linspace(d, LX, 10))
    for nsplit in (1, 2, 4, 8, 16):
        pts = ([0.0] + ([d * 0.5 ** j for j in range(nsplit - 1, 0, -1)]
                        if nsplit > 1 else []) + outer)
        x, u, q, sig = solve_fields(ShestakovNonlinear(n_b=N_B), pts=pts,
                                    tag="diag_graded")
        off = np.mean((sig - S0 * np.minimum(x, d))[x > d])
        r, why = attempt(solve, ShestakovNonlinear(n_b=N_B), pts=pts)
        print(f"   {pts[1]-pts[0]:12.3e} {len(pts)-1:6d} {(len(pts)-1)*3:6d} "
              f"{off/GAMMA_WALL:13.4e} {r[0] if r else 0:7d} "
              f"{r[1] if r else np.nan:13.3e}")
    print()
    print("   For comparison, uniform meshes:")
    for ncells in (10, 20, 40, 80):
        r, why = attempt(solve, ShestakovNonlinear(n_b=N_B), ncells=ncells)
        print(f"      uniform {ncells:3d} cells, {ncells*3:4d} nodes: "
              f"visits {r[0]:4d}, error {r[1]:.3e}" if r
              else f"      uniform {ncells:3d} cells: fail")
    print()
    print("   The offset tracks the first cell's width and nothing else, which")
    print("   is where the error is made. A graded mesh therefore buys two")
    print("   orders of magnitude at equal cost -- and is exactly the kind of")
    print("   per-problem tuning this benchmark exists *not* to do. See")
    print("   ANALYSIS.md.")


# ------------------------------------------------------------------ section 8
class SmoothControl(ShestakovNonlinear):
    """The same flux law and the same D(0) = 0, with a regular solution.

    S = 3 S0 d x^2 gives Gamma = S0 d x^3, so w_x = -c x and
        w = n_b^(1/3) + c (Lx^2 - x^2)/2,   n_e = w^3
    a degree-6 polynomial, with sigma_hat_e = -S0 d x^3 of degree 3. q_e is
    linear at the axis instead of x^(1/3), and D = (q/u)^2 ~ x^2 still vanishes
    there -- so this isolates ingredient (ii) from ingredient (i).
    """

    def Sources(self, index, state, x, t):
        return 3.0 * S0 * d * x * x

    @staticmethod
    def exact(x, n_b=N_B):
        x = np.asarray(x, dtype=float)
        return (n_b ** (1.0 / 3.0) + C_PHYS * (LX ** 2 - x ** 2) / 2.0) ** 3

    def InitialValue(self, index, x):
        return float(self.exact(float(x), self.n_b))

    def InitialDerivative(self, index, x):
        h, x = 1.0e-6, float(x)
        xp, xm = min(x + h, LX), max(x - h, 0.0)
        return float((self.exact(xp, self.n_b) - self.exact(xm, self.n_b))
                     / (xp - xm))


def section8():
    rule("8. Control: the same flux law with a regular solution")
    print("   S = 3 S0 d x^2 instead of a step, so q_e is linear at the axis")
    print("   rather than x^(1/3). D(0) = 0 exactly as before. If the order")
    print("   returns, the degeneracy is harmless on its own and the x^(4/3) in")
    print("   n_e is the whole of the O(h).")
    print()
    print(f"   {'cells':>6} {'k':>2} {'dsigma/Gamma':>14} "
          f"{'rel L1 error':>13} {'rate':>7}")
    prev = {}
    for ncells, k in [(10, 2), (20, 2), (40, 2), (10, 3), (20, 3), (10, 4),
                      (10, 6)]:
        x, u, q, sig = solve_fields(SmoothControl(n_b=N_B), ncells=ncells, k=k,
                                   tag="diag_smooth")
        ue = SmoothControl.exact(x, N_B)
        off = np.mean((sig - S0 * d * x ** 3)[x > 0.5])
        rel = np.mean(np.abs(u - ue)) / np.mean(np.abs(ue))
        rate = (f"{np.log2(prev[(ncells//2, k)]/rel):7.2f}"
                if (ncells // 2, k) in prev else "")
        prev[(ncells, k)] = rel
        print(f"   {ncells:6d} {k:2d} {off/GAMMA_WALL:14.4e} {rel:13.3e} "
              f"{rate:>7}")
    print()
    print("   Full order, and exact at k = 6 where n_e lies in P_k. So a")
    print("   degenerate diffusivity is not by itself a problem for MaNTA; a")
    print("   degenerate diffusivity *at a boundary where the solution has a")
    print("   fractional power* is.")


SECTIONS = {1: section1, 2: section2, 3: section3, 4: section4,
            5: section5, 6: section6, 7: section7, 8: section8}


def main(argv):
    wanted = [int(a) for a in argv[1:]] or sorted(SECTIONS)
    for n in wanted:
        SECTIONS[n]()


if __name__ == "__main__":
    main(sys.argv)
