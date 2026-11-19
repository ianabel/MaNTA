// Bounds on a case's variables: FieldSpec::bound.
//
// A bounded variable is one whose value the case cannot be evaluated at, or
// must not be, below some limit -- a density, an energy. Nothing in the
// discretisation keeps it there: HDG is free to undershoot, and a damped Newton
// step or an IDA corrector is free to land anywhere. What this file provides is
// the test every solve path asks before it evaluates the physics at a state
// and before it accepts one:
//
//   * the residual callbacks -- IDA's static_residual, and steadyResidual for
//     KINSOL and the NewtonStep loop -- refuse a breaching state as a
//     recoverable failure, without evaluating the physics there;
//   * IDA's corrector, KINSOL's iterate and the NewtonStep loop's step are each
//     checked before they are accepted (attachBoundedNonlinearSolver here,
//     SteadyState.cpp for the other two).
//
// So the physics is never evaluated at a state that breaches a bound, and no
// such state is ever accepted. A breach that cannot be cured -- the step cut
// to its floor, the rejections at their cap -- ends the solve with the
// violation named. A NonNegative variable additionally reaches the physics as
// max(u, 0): physicsNodesAt clamps, and the Jacobian zeroes the clamped
// points' derivatives to match.
//
// None of it runs for a case that bounds nothing: every entry point is guarded
// on an empty list, so such a run is bit for bit what it was.

#include <Eigen/Core>
#include <Eigen/Dense>

#include "SystemSolver.hpp"

#include <sundials/sundials_nonlinearsolver.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#include <ida/ida.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

void SystemSolver::clampForPhysics(PhysicsNodes &nodes) const
{
    if (clampedVars.empty())
        return;
    Matrix &u = nodes.states.Variable();
    for (Index v : clampedVars)
        for (Index p = 0; p < u.cols(); ++p)
            if (u(v, p) < 0.0)
            {
                u(v, p) = 0.0;
                nodes.clamped.emplace_back(v, p);
            }
}

void SystemSolver::zeroClampedDerivatives(PhysicsNodes const &nodes, GlobalStateMatrix &dSigma,
                                          GlobalStateMatrix &dSource, GlobalStateMatrix &dAux) const
{
    for (auto const &[v, p] : nodes.clamped)
    {
        for (Index i = 0; i < nVars; ++i)
        {
            dSigma[i].Variable()(v, p) = 0.0;
            dSource[i].Variable()(v, p) = 0.0;
        }
        for (Index a = 0; a < nAux; ++a)
            dAux[a].Variable()(v, p) = 0.0;
    }
}

void SystemSolver::resolveBounds()
{
    boundedVars.clear();
    boundLimit.assign(nVars, 0.0);
    nBoundRejections = 0;
    lastBoundBreach.reset();

    auto const &vars = problem->spec().variables;
    for (Index v = 0; v < nVars; ++v)
    {
        FieldSpec const &f = vars[v];
        if (f.bound == VariableBound::Uncontrolled)
            continue;
        boundedVars.push_back(v);
        const double tolerance = atol.size() == static_cast<size_t>(nVars) ? atol[v] : atol[0];
        const double eps = f.boundEpsilon.value_or(f.bound == VariableBound::Positive ? 0.0 : tolerance);
        boundLimit[v] = f.bound == VariableBound::Positive ? eps : -eps;
    }

    // The stabilisation's kappa is evaluated at the faces (TauKappa = "Face")
    // and its d tau / dy differenced at the nodes (tauUpdate = "Residual"),
    // and neither applies the clamp: the face state is the trace, which no
    // bound constrains, and the differencing perturbs the clamped value.
    if (!clampedVars.empty() && tauScaling != TauScaling::Constant)
        throw std::invalid_argument(
            "A NonNegative variable cannot be used with tauScaling = \"Diffusive\": the "
            "stabilisation evaluates kappa where the clamp to max(u, 0) does not reach. Use a "
            "constant tau, or declare the variable Positive.");
}

namespace
{
// Does `value` satisfy the bound whose limit is `limit`? Written so that a NaN
// fails it.
bool satisfies(VariableBound kind, double value, double limit)
{
    return kind == VariableBound::Positive ? value > limit : value >= limit;
}
} // namespace

std::optional<SystemSolver::BoundViolation> SystemSolver::boundViolation(DGSoln const &Y) const
{
    if (boundedVars.empty())
        return std::nullopt;

    const bool star = superconvergent;
    const Index nPer = star ? k + 2 : k + 1;
    const Vector refNodes = NodalBasis::getBasis(star ? k + 1 : k).getNodes();

    const bool wholeCell = boundCheck == BoundCheck::WholeCell;
    manta::PolynomialMinimum const *minimum = nullptr;
    if (wholeCell)
    {
        auto &slot = star ? starMinimum : basisMinimum;
        if (!slot)
            slot = std::make_unique<manta::PolynomialMinimum>(refNodes);
        minimum = slot.get();
    }

    std::optional<BoundViolation> worst;
    double worstAmount = -std::numeric_limits<double>::infinity();
    auto consider = [&](Index v, VariableBound kind, double value, Position x, bool between)
    {
        if (satisfies(kind, value, boundLimit[v]))
            return;
        // A NaN breach outranks any finite one.
        const double amount = std::isnan(value) ? std::numeric_limits<double>::infinity()
                                                : boundLimit[v] - value;
        if (!worst || amount > worstAmount)
        {
            worst = BoundViolation{v, x, value, boundLimit[v], kind, between};
            worstAmount = amount;
        }
    };

    auto const &vars = problem->spec().variables;
    Vector vals(nPer);
    for (Index v : boundedVars)
    {
        const VariableBound kind = vars[v].bound;
        for (Index i = 0; i < nCells; ++i)
        {
            Interval const &I = grid[i];
            if (star)
                vals = postprocessor->B11(i) * Y.q(v).getCoeff(i).second +
                       postprocessor->B12(i) * Y.u(v).getCoeff(i).second;
            else
                vals = Y.u(v).getCoeff(i).second;

            bool nodesClean = true;
            for (Index m = 0; m < nPer; ++m)
            {
                if (!satisfies(kind, vals(m), boundLimit[v]))
                    nodesClean = false;
                consider(v, kind, vals(m), I.fromRef(refNodes(m)), false);
            }
            if (minimum != nullptr)
            {
                const auto r = minimum->minimum(vals);
                consider(v, kind, r.value, I.fromRef(r.s), nodesClean);
            }
        }
    }
    return worst;
}

std::string SystemSolver::describe(BoundViolation const &b) const
{
    const std::string name = problem->getVariableName(b.var);
    const std::string rule = b.kind == VariableBound::Positive
                                 ? std::format("{} > {:g}", name, b.limit)
                                 : std::format("{} >= {:g}", name, b.limit);
    return std::format("{} = {:.6g} at x = {:.6g}{}, breaching the bound {} ({})", name, b.value, b.x,
                       b.betweenNodes ? " (between the nodes; BoundCheck = \"WholeCell\")" : "",
                       rule, b.kind == VariableBound::Positive ? "Positive" : "NonNegative");
}

bool SystemSolver::refuseBreach(DGSoln const &Y)
{
    if (boundedVars.empty())
        return false;
    auto v = boundViolation(Y);
    if (!v)
        return false;
    ++nBoundRejections;
    lastBoundBreach = v;
    return true;
}

bool SystemSolver::refuseBreach(N_Vector y)
{
    if (boundedVars.empty())
        return false;
    DGSoln Yh(nVars, grid, k, N_VGetArrayPointer(y), nScalars, nAux, nField);
    return refuseBreach(Yh);
}

// ---- IDA's nonlinear solver, bounded.
//
// A SUNNonlinearSolver of our own whose every operation forwards to a
// SUNNonlinSol_Newton, except that a converged solve is checked: IDA hands
// solve() its predictor y0 and takes y0 + ycor as the step's result, so that
// is the state checked, before IDA's error test and before acceptance. A
// breach is SUN_NLS_CONV_RECVR, which IDA answers as it answers any
// convergence failure: the step is cut by its eta_cf (a quarter) and retried,
// up to its MaxConvFails per step.
namespace
{
struct BoundedNLSContent
{
    SUNNonlinearSolver inner;
    SystemSolver *solver;
    N_Vector candidate;
};

BoundedNLSContent *content(SUNNonlinearSolver NLS) { return static_cast<BoundedNLSContent *>(NLS->content); }
SUNNonlinearSolver inner(SUNNonlinearSolver NLS) { return content(NLS)->inner; }

SUNNonlinearSolver_Type boundedGetType(SUNNonlinearSolver) { return SUNNONLINEARSOLVER_ROOTFIND; }
SUNErrCode boundedInitialize(SUNNonlinearSolver NLS) { return SUNNonlinSolInitialize(inner(NLS)); }
int boundedSetup(SUNNonlinearSolver NLS, N_Vector y, void *mem) { return SUNNonlinSolSetup(inner(NLS), y, mem); }
SUNErrCode boundedSetSysFn(SUNNonlinearSolver NLS, SUNNonlinSolSysFn f) { return SUNNonlinSolSetSysFn(inner(NLS), f); }
SUNErrCode boundedSetLSetupFn(SUNNonlinearSolver NLS, SUNNonlinSolLSetupFn f) { return SUNNonlinSolSetLSetupFn(inner(NLS), f); }
SUNErrCode boundedSetLSolveFn(SUNNonlinearSolver NLS, SUNNonlinSolLSolveFn f) { return SUNNonlinSolSetLSolveFn(inner(NLS), f); }
SUNErrCode boundedSetCTestFn(SUNNonlinearSolver NLS, SUNNonlinSolConvTestFn f, void *data)
{
    return SUNNonlinSolSetConvTestFn(inner(NLS), f, data);
}
SUNErrCode boundedSetMaxIters(SUNNonlinearSolver NLS, int n) { return SUNNonlinSolSetMaxIters(inner(NLS), n); }
SUNErrCode boundedGetNumIters(SUNNonlinearSolver NLS, long *n) { return SUNNonlinSolGetNumIters(inner(NLS), n); }
SUNErrCode boundedGetCurIter(SUNNonlinearSolver NLS, int *n) { return SUNNonlinSolGetCurIter(inner(NLS), n); }
SUNErrCode boundedGetNumConvFails(SUNNonlinearSolver NLS, long *n) { return SUNNonlinSolGetNumConvFails(inner(NLS), n); }

int boundedSolve(SUNNonlinearSolver NLS, N_Vector y0, N_Vector ycor, N_Vector w, sunrealtype tol,
                 sunbooleantype callLSetup, void *mem)
{
    const int retval = SUNNonlinSolSolve(inner(NLS), y0, ycor, w, tol, callLSetup, mem);
    if (retval != SUN_SUCCESS)
        return retval;
    BoundedNLSContent *c = content(NLS);
    N_VLinearSum(1.0, y0, 1.0, ycor, c->candidate);
    return c->solver->refuseBreach(c->candidate) ? SUN_NLS_CONV_RECVR : SUN_SUCCESS;
}

SUNErrCode boundedFree(SUNNonlinearSolver NLS)
{
    if (NLS == nullptr)
        return SUN_SUCCESS;
    if (NLS->content != nullptr)
    {
        BoundedNLSContent *c = content(NLS);
        SUNNonlinSolFree(c->inner);
        N_VDestroy(c->candidate);
        delete c;
        NLS->content = nullptr;
    }
    SUNNonlinSolFreeEmpty(NLS);
    return SUN_SUCCESS;
}
} // namespace

SUNNonlinearSolver SystemSolver::makeBoundedNonlinearSolver(N_Vector like)
{
    SUNNonlinearSolver NLS = SUNNonlinSolNewEmpty(ctx);
    if (NLS == nullptr)
        throw std::runtime_error("SUNNonlinSolNewEmpty failed");
    NLS->ops->gettype = boundedGetType;
    NLS->ops->initialize = boundedInitialize;
    NLS->ops->setup = boundedSetup;
    NLS->ops->solve = boundedSolve;
    NLS->ops->free = boundedFree;
    NLS->ops->setsysfn = boundedSetSysFn;
    NLS->ops->setlsetupfn = boundedSetLSetupFn;
    NLS->ops->setlsolvefn = boundedSetLSolveFn;
    NLS->ops->setctestfn = boundedSetCTestFn;
    NLS->ops->setmaxiters = boundedSetMaxIters;
    NLS->ops->getnumiters = boundedGetNumIters;
    NLS->ops->getcuriter = boundedGetCurIter;
    NLS->ops->getnumconvfails = boundedGetNumConvFails;
    NLS->content = new BoundedNLSContent{SUNNonlinSol_Newton(like, ctx), this, N_VClone(like)};
    if (content(NLS)->inner == nullptr || content(NLS)->candidate == nullptr)
    {
        boundedFree(NLS);
        throw std::runtime_error("Could not build the bounded nonlinear solver");
    }
    return NLS;
}

void SystemSolver::attachBoundedNonlinearSolver()
{
    if (boundedVars.empty())
        return;

    SUNNonlinearSolver NLS = makeBoundedNonlinearSolver(Y);
    idaBoundedNLS = NLS;
    if (IDASetNonlinearSolver(IDA_mem, NLS) != IDA_SUCCESS)
        throw std::runtime_error("IDASetNonlinearSolver failed");
}

void SystemSolver::checkSteadyStartWithinBounds()
{
    if (!boundsActive())
        return;
    DGSoln Yh(nVars, grid, k, N_VGetArrayPointer(Y), nScalars, nAux, nField);
    if (auto breach = boundViolation(Yh))
        throw std::invalid_argument(std::format(
            "The steady solve starts from a state that breaches a bound: {}. Every residual "
            "there would be refused.",
            describe(*breach)));
}

std::string SystemSolver::boundRejectionNote(long since) const
{
    if (nBoundRejections <= since || !lastBoundBreach)
        return {};
    return std::format(" {} states were refused for breaching a bound; the last: {}.",
                       nBoundRejections - since, describe(*lastBoundBreach));
}
