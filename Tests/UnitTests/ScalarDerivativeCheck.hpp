#ifndef SCALARDERIVATIVECHECK_HPP
#define SCALARDERIVATIVECHECK_HPP

// Finite-difference a physics case's ScalarG and compare against its own
// ScalarGPrime.
//
// This is the check that separates a broken bordered elimination from a physics
// case that misreports its own derivative, and it works for any case: perturb
// one solution coefficient at a time and difference the scalar constraint. It is
// the first thing to run when a scalar system converges slowly, because a wrong
// scalar Jacobian costs Newton iterations rather than accuracy and so leaves a
// reference comparison perfectly green.
//
// Lives in a header because two suites need it: ScalarJacobianTests.cpp, where
// it was written, and MMSAuxScalarTests.cpp, which runs it on the manufactured
// scalar problems before trusting an order study built on them.

#include <boost/test/unit_test.hpp>

#include "PyIntegrator.hpp"
#include "SystemSolver.hpp"
#include "Types.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace scalarcheck
{

// The scalar rows IDACalcIC cannot move.
//
// IDA_YA_YDP_INIT solves for the algebraic *values* and the differential
// *derivatives*, and holds the differential values fixed. A row is reachable if
// it touches one of those unknowns: a d/dt of u or of a differential scalar, or
// the value of sigma, q, an aux variable or an algebraic scalar. A row reading
// only u and differential scalars is a constant to CalcIC, and unless it already
// holds at the initial state the run dies with IDA_LINESEARCH_FAIL (-13) before
// the first step -- a message about the line search, for a defect in the
// declaration. The usual way to write one is to declare a scalar differential
// whose G has no d/dt in it; that is what kept python-physics/mirror-plasma's
// voltage controller from ever starting.
inline std::vector<Index> scalarRowsCalcICCannotReach(TransportSystem &problem, DGSoln &y, DGSoln &dydt,
                                                      Grid const &grid, Index k, double t)
{
    const Index nVars = problem.getNumVars();
    const Index nScalars = problem.getNumScalars();
    const Index nAux = problem.getNumAux();
    const Index nCells = static_cast<Index>(grid.getNCells());

    Integrator::Cache integrator;
    GlobalStateMatrix dG(nScalars), dGdot(nScalars);
    for (Index s = 0; s < nScalars; ++s)
    {
        dG.add(nCells, k, nVars, nScalars, nAux);
        dGdot.add(nCells, k, nVars, nScalars, nAux);
    }
    problem.ScalarGPrime(dG, dGdot, y.evalOnNodes(), dydt.evalOnNodes(), y.getPoints(),
                         integrator.integrationWeights(y.getBasis(), grid),
                         integrator.phiBoundary(y.getBasis(), grid), t);

    std::vector<Index> unreachable;
    for (Index s = 0; s < nScalars; ++s)
    {
        GlobalState const &value = dG[s], &rate = dGdot[s];
        bool reaches = !rate.Variable().isZero(0.0) || !value.Derivative().isZero(0.0) ||
                       !value.Flux().isZero(0.0) || !value.Aux().isZero(0.0);
        for (Index m = 0; m < nScalars && !reaches; ++m)
            reaches = problem.isScalarDifferential(m) ? rate.Scalars()(m) != 0.0 : value.Scalars()(m) != 0.0;
        if (!reaches)
            unreachable.push_back(s);
    }
    return unreachable;
}

// The hooks report every scalar's derivative in one call, indexed by node rather
// than being asked once per (cell, basis function), so the reported value for
// cell i node l is column i*(k+1)+l -- the same flattening the solver reads back
// into the w vectors, which is exactly the indexing an off-by-one here would
// hide.
//
// Returns the largest disagreement found, and reports each one.
//
// It also requires every scalar row to be one IDACalcIC can move, apart from the
// rows named in `knownUnreachable` -- see scalarRowsCalcICCannotReach.
inline double checkScalarDerivative(TransportSystem &problem, DGSoln &y, DGSoln &dydt,
                                    Grid const &grid, Index k, double t, double tolerance,
                                    std::vector<Index> const &knownUnreachable = {})
{
    const Index nVars = problem.getNumVars();
    const Index nScalars = problem.getNumScalars();
    const Index nAux = problem.getNumAux();
    const Index nCells = static_cast<Index>(grid.getNCells());

    Integrator::Cache integrator;
    const Vector &weights = integrator.integrationWeights(y.getBasis(), grid);
    const Matrix &phiBoundary = integrator.phiBoundary(y.getBasis(), grid);

    double worst = 0.0;

    // What the case says its derivative is -- all scalars, all nodes, one call.
    GlobalStateMatrix reported(nScalars), reported_dt(nScalars);
    for (Index s = 0; s < nScalars; ++s)
    {
        reported.add(nCells, k, nVars, nScalars, nAux);
        reported_dt.add(nCells, k, nVars, nScalars, nAux);
    }
    problem.ScalarGPrime(reported, reported_dt, y.evalOnNodes(), dydt.evalOnNodes(),
                         y.getPoints(), weights, phiBoundary, t);

    // Re-sampled on every call: the coefficient being perturbed has to reach
    // the nodal values the constraint actually reads.
    auto G = [&](Index s)
    {
        return problem.ScalarG(s, y.evalOnNodes(), dydt.evalOnNodes(), y.getPoints(), weights,
                               phiBoundary, t);
    };

    auto compare = [&](const char *what, Index s, Index cell, Index l, Index var,
                       double expected, double *coefficient)
    {
        const double h = 1e-6 * std::max(1.0, std::abs(*coefficient));
        const double original = *coefficient;

        *coefficient = original + h;
        const double gp = G(s);
        *coefficient = original - h;
        const double gm = G(s);
        *coefficient = original;

        const double fd = (gp - gm) / (2.0 * h);
        const double err = std::abs(fd - expected);
        worst = std::max(worst, err);

        BOOST_TEST(err < tolerance,
                   "dG_" << s << "/d" << what << " (cell " << cell << ", node " << l
                         << ", var " << var << "): reported " << expected
                         << ", finite difference " << fd);
    };

    BOOST_TEST(scalarRowsCalcICCannotReach(problem, y, dydt, grid, k, t) == knownUnreachable,
               boost::test_tools::per_element());

    for (Index s = 0; s < nScalars; ++s)
    {
        for (Index cell = 0; cell < nCells; ++cell)
        {
            for (Index l = 0; l < k + 1; ++l)
            {
                const Index node = cell * (k + 1) + l;

                for (Index v = 0; v < nVars; ++v)
                {
                    compare("u", s, cell, l, v, reported[s].Variable()(v, node),
                            &y.u(v).getCoeff(cell).second(l));
                    compare("q", s, cell, l, v, reported[s].Derivative()(v, node),
                            &y.q(v).getCoeff(cell).second(l));
                    compare("sigma", s, cell, l, v, reported[s].Flux()(v, node),
                            &y.sigma(v).getCoeff(cell).second(l));
                }
                for (Index a = 0; a < nAux; ++a)
                    compare("aux", s, cell, l, a, reported[s].Aux()(a, node),
                            &y.Aux(a).getCoeff(cell).second(l));
            }
        }

        // The scalar-scalar block, and the dY'/dt half.
        for (Index m = 0; m < nScalars; ++m)
        {
            compare("mu", s, -1, -1, m, reported[s].Scalars()(m), &y.Scalar(m));
            compare("dmu/dt", s, -1, -1, m, reported_dt[s].Scalars()(m), &dydt.Scalar(m));
        }
    }

    return worst;
}

} // namespace scalarcheck

#endif // SCALARDERIVATIVECHECK_HPP
