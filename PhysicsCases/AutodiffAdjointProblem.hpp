#ifndef AUTODIFFADJOINTPROBLEM_HPP
#define AUTODIFFADJOINTPROBLEM_HPP

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>

#include "AutodiffTransportSystem.hpp"
#include "Types.hpp"
#include "AdjointProblem.hpp"

class AutodiffAdjointProblem : public AdjointProblem<Real, RealVector, RealVector, RealVector>
{
public:
    AutodiffAdjointProblem(std::shared_ptr<AutodiffTransportSystem> TransportSystem) : PhysicsProblem(TransportSystem) {}

    virtual Value GFn(Index i, DGSoln &y) const override;
    virtual Value dGFndp(Index i, DGSoln &y) const override;

    virtual void dgFn_du(Index i, Values &, const State &s, Position x) const override;
    virtual void dgFn_dq(Index i, Values &, const State &s, Position x) const override;
    virtual void dgFn_dsigma(Index i, Values &, const State &s, Position x) const override;

    virtual void dSigmaFn_dp(Index i, Value &, const State &s, Position x, Time t) const override;
    virtual void dSources_dp(Index i, Value &, const State &, Position x, Time t) const override;

private:
    using integrator = boost::math::quadrature::gauss_kronrod<double, 15>;
    constexpr static int max_depth = 2;
    std::shared_ptr<AutodiffTransportSystem> PhysicsProblem;
};
#endif