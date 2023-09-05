#ifndef AUTODIFF3VarCyl_HPP
#define AUTODIFF3VarCyl_HPP

#include "PhysicsCases.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

using autodiff::dual;
using autodiff::dual2nd;
using autodiff::VectorXdual;

typedef std::function<dual(VectorXdual u, VectorXdual q, dual x, double t)> sigmaFn;
typedef std::vector<sigmaFn> sigmaFnArray;

typedef std::function<dual(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)> SourceFn;
typedef std::vector<SourceFn> SourceFnArray;

typedef std::function<dual2nd(dual2nd x, dual2nd t)> Solution;
using TestSolVec = std::vector<Solution>;

class Autodiff3VarCyl : public TransportSystem
{
public:
    explicit Autodiff3VarCyl(toml::value const &config);
    //  Function for passing boundary conditions to the solver
    Value LowerBoundary(Index i, Time t) const override;
    Value UpperBoundary(Index i, Time t) const override;

    bool isLowerBoundaryDirichlet(Index i) const override;
    bool isUpperBoundaryDirichlet(Index i) const override;

    // The same for the flux and source functions -- the vectors have length nVars
    Value SigmaFn(Index i, const Values &u, const Values &q, Position x, Time t) override;
    Value Sources(Index i, const Values &u, const Values &q, const Values &sigma, Position x, Time t) override;

    // We need derivatives of the flux functions
    void dSigmaFn_du(Index i, Values &, const Values &u, const Values &q, Position x, Time t) override;
    void dSigmaFn_dq(Index i, Values &, const Values &u, const Values &q, Position x, Time t) override;

    // and for the sources
    void dSources_du(Index i, Values &, const Values &u, const Values &q, Position x, Time t) override;
    void dSources_dq(Index i, Values &, const Values &u, const Values &q, Position x, Time t) override;
    void dSources_dsigma(Index i, Values &, const Values &u, const Values &q, Position x, Time t) override;

    // and initial conditions for u & q
    Value InitialValue(Index i, Position x) const override;
    Value InitialDerivative(Index i, Position x) const override;
    double TestSource(Index i, Position x, Time t);

private:
    Values uL;
    bool isUpperDirichlet;

    Values uR;
    bool isLowerDirichlet;

    Position xL;
    Position xR;

    static sigmaFnArray sigmaVec;
    static SourceFnArray SourceVec;

    bool isTestProblem;

    static dual2nd TestDirichlet(dual2nd x, dual2nd t, double u_R, double u_L, double x_L, double x_R);

    // normalizations
    const Value n0 = 3e18;
    const Value T0 = 1.6e-19 * 1e2;
    const Value p0 = n0 * T0;
    const Value L = 1;

    REGISTER_PHYSICS_HEADER(Autodiff3VarCyl)
};
#endif