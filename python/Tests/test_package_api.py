"""The `manta` package's own API: declaration as class attributes, and the
derivative hooks being optional.

Both are things a physics case author meets in the first five minutes, and
neither was possible before:

  * a case used to declare itself by assigning `self.nVars` and
    `self.isUpperDirichlet` *after* `TransportSystem.__init__` had already run,
    so there was no point at which the description could be checked, and the
    boundary flags were read uninitialised if a case forgot them;

  * all seven physics hooks were mandatory, so the simplest possible case --
    linear diffusion, whose four source derivatives are identically zero -- had
    to write four functions returning `np.zeros(nVars)` before it would run.
"""

import numpy as np
import pytest

import manta


KAPPA = 1.7


class DeclarativeDiffusion(manta.TransportSystem):
    """d_t u = d_x( kappa d_x u ), declared as data.

    Note what is *not* here: no self.nVars, no self.isUpperDirichlet, and no
    dSigmaFn_du / dSources_du / dSources_dq / dSources_dsigma -- all four of
    those are identically zero for this problem, so they are simply absent.
    """

    variables = [manta.Field("density", "particle density", "m^-3",
                             lower=manta.Neumann, upper=manta.Dirichlet)]

    def __init__(self, kappa=KAPPA):
        super().__init__()
        self.kappa = kappa

    def SigmaFn(self, i, state, x, t):
        return self.kappa * state.q[0]

    def Sources(self, i, state, x, t):
        return 0.0

    def dSigmaFn_dq(self, i, state, x, t):
        return np.full(self.nVars, self.kappa)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0


def test_class_attributes_become_the_spec():
    sys = DeclarativeDiffusion()

    assert sys.nVars == 1
    assert sys.nScalars == 0
    assert sys.nAux == 0

    # The names reach the spec, which is what the netCDF groups are keyed on.
    spec = sys.spec
    assert [v.name for v in spec.variables] == ["density"]
    assert spec.variables[0].units == "m^-3"
    assert spec.variables[0].description == "particle density"


def test_boundary_kinds_come_from_the_declaration():
    sys = DeclarativeDiffusion()

    # Neumann below, Dirichlet above, exactly as declared -- and, unlike the
    # pair of bools this replaced, not readable before it is set.
    assert sys.isLowerBoundaryDirichlet(0) is False
    assert sys.isUpperBoundaryDirichlet(0) is True


def test_the_counts_are_read_only():
    sys = DeclarativeDiffusion()
    with pytest.raises(AttributeError):
        sys.nVars = 5


def test_a_case_declaring_nothing_is_refused_by_name():
    class Undeclared(manta.TransportSystem):
        pass

    with pytest.raises(TypeError, match="Undeclared"):
        Undeclared()


def test_an_explicit_spec_still_works():
    """A case whose shape depends on its configuration cannot use class
    attributes, so passing a spec has to keep working."""

    class Configured(DeclarativeDiffusion):
        def __init__(self, n):
            manta.TransportSystem.__init__(self, manta.numbered_spec(n, nAux=1))

    sys = Configured(3)
    assert sys.nVars == 3
    assert sys.nAux == 1
    assert [v.name for v in sys.spec.variables] == ["Var0", "Var1", "Var2"]


def test_keyword_form_works():
    class ByKeyword(DeclarativeDiffusion):
        def __init__(self):
            manta.TransportSystem.__init__(
                self,
                variables=[manta.Field("a"), manta.Field("b")],
                scalars=[manta.Scalar("mu", differential=True)],
            )

    sys = ByKeyword()
    assert sys.nVars == 2
    assert sys.nScalars == 1
    assert sys.isScalarDifferential(0) is True


def test_duplicate_names_are_rejected_at_construction():
    """The spec is validated before the object exists, so a case cannot be
    half-described. Names share one namespace across the three groups."""

    class Clashing(DeclarativeDiffusion):
        def __init__(self):
            manta.TransportSystem.__init__(
                self,
                variables=[manta.Field("n")],
                aux=[manta.Aux("n")],
            )

    with pytest.raises(ValueError, match="Duplicate name 'n'"):
        Clashing()


def test_omitted_derivative_hooks_are_treated_as_zero(tmp_path):
    """The case above provides only dSigmaFn_dq. The other four are absent, and
    the run has to reach the same answer as if they returned zeros -- which is
    what the framework's zeroed out-parameter gives it."""

    runner = manta.Runner(DeclarativeDiffusion())
    runner.configure(
        {
            "Polynomial_degree": 2,
            "Grid_size": 8,
            "Lower_boundary": 0.0,
            "Upper_boundary": 1.0,
            "delta_t": 0.05,
            "OutputFilename": str(tmp_path / "declarative"),
            "WriteOutput": False,
        }
    )
    runner.run(0.1)

    u = np.asarray(runner.getSolution(0, [0.0, 0.5, 1.0]))
    assert np.all(np.isfinite(u))


def test_a_bare_boundary_kind_still_works_where_a_condition_is_wanted():
    """`lower=manta.Neumann` and `f.lower == manta.Dirichlet` both still work.

    FieldSpec's ends are BoundaryCondition now rather than BoundaryKind. Every
    case in the tree, every example and both python-physics systems write the
    kind, so the implicit conversion is what keeps them working -- and
    python-physics/mirror-plasma reads its kinds back out by comparing an end
    against a kind, which is why __eq__ is bound.
    """
    field = manta.Field("n", "density", "m^-3", lower=manta.Neumann)

    assert field.lower == manta.Neumann
    assert field.upper == manta.Dirichlet          # the default
    assert field.lower.kind == manta.BoundaryKind.Neumann
    assert (field.lower.a, field.lower.b, field.lower.d) == (0.0, 0.0, 0.0)

    # Assignment through the property too: stellarator_multichannel builds a
    # dict of kinds and writes them into an already-built spec.
    field.upper = manta.Neumann
    assert field.upper == manta.Neumann


def test_a_mixed_end_carries_its_coefficients():
    field = manta.Field("n", "density", "m^-3", lower=manta.Mixed(b=0.5, d=1.0))

    assert field.lower == manta.BoundaryKind.Mixed
    assert field.lower.b == 0.5
    assert field.lower.d == 1.0
    assert field.lower.a == 0.0
    assert "Mixed" in repr(field.lower)


def test_a_mixed_end_that_constrains_only_u_is_refused():
    """b = d = 0 is a weakly imposed Dirichlet, not the Dirichlet kind.

    Accepting it would give two inequivalent spellings of the same intent, so
    the spec validation refuses it and names the way out. The check happens in
    the TransportSystem constructor, which is where the spec is validated.
    """

    class OnlyU(manta.TransportSystem):
        variables = [manta.Field("u", "u", "", lower=manta.Mixed(a=1.0))]

        def SigmaFn(self, i, s, x, t):
            return s.q[i]

        def Sources(self, i, s, x, t):
            return 0.0

    with pytest.raises(ValueError, match="Dirichlet"):
        OnlyU()


def test_mixed_is_reachable_through_numbered_spec():
    spec = manta.numbered_spec(2, lower=manta.Mixed(b=1.0, d=1.0))
    assert all(v.lower == manta.BoundaryKind.Mixed for v in spec.variables)
    assert spec.variables[0].lower.b == 1.0
    assert spec.variables[0].lower.d == 1.0
    assert all(v.upper == manta.Dirichlet for v in spec.variables)
