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

namespace scalarcheck
{

// The hooks report every scalar's derivative in one call, indexed by node rather
// than being asked once per (cell, basis function), so the reported value for
// cell i node l is column i*(k+1)+l -- the same flattening the solver reads back
// into the w vectors, which is exactly the indexing an off-by-one here would
// hide.
//
// Returns the largest disagreement found, and reports each one.
inline double checkScalarDerivative(TransportSystem &problem, DGSoln &y, DGSoln &dydt,
                                    Grid const &grid, Index k, double t, double tolerance)
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
