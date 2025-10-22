#ifndef AUTODIFFADJOINTPROBLEM_HPP
#define AUTODIFFADJOINTPROBLEM_HPP

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>

#include "AutodiffTransportSystem.hpp"
#include "Types.hpp"
#include "AdjointProblem.hpp"

class AutodiffAdjointProblem : public AdjointProblem
{
public:
    AutodiffAdjointProblem(AutodiffTransportSystem *TransportSystem) : PhysicsProblem(TransportSystem) {}

    virtual Value GFn(Index i, DGSoln &y) const override;
    virtual Value dGFndp(Index i, DGSoln &y) const override;

    virtual Value gFn(Index i, const State &s, Position x) const override;

    virtual void dgFn_du(Index i, Values &, const State &s, Position x) override;
    virtual void dgFn_dq(Index i, Values &, const State &s, Position x) override;
    virtual void dgFn_dsigma(Index i, Values &, const State &s, Position x) override;
    virtual void dgFn_dphi(Index i, Values &, const State &s, Position x) override;

    virtual void dSigmaFn_dp(Index i, Value &, const State &s, Position x) override;
    virtual void dSources_dp(Index i, Value &, const State &, Position x) override;

    void setG(std::function<Real(Position, Real, RealVector &, RealVector &, RealVector &, RealVector &)> gin) { g = gin; }

private:
    int np;
    std::function<Real(Position, Real, RealVector &, RealVector &, RealVector &, RealVector &)> g;
    using integrator = boost::math::quadrature::gauss_kronrod<double, 15>;
    constexpr static int max_depth = 2;
    AutodiffTransportSystem *PhysicsProblem;
};
#endif