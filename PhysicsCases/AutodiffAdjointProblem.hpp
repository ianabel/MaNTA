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

    virtual void dSigmaFn_dp(Index i, Index pIndex, Value &, const State &s, Position x) override;
    virtual void dSources_dp(Index i, Index pIndex, Value &, const State &, Position x) override;

    virtual bool computeUpperBoundarySensitivity(Index i, Index pIndex) override;
    virtual bool computeLowerBoundarySensitivity(Index i, Index pIndex) override;

    void addUpperBoundarySensitivity(Index i, Index pIndex);
    void addLowerBoundarySensitivity(Index i, Index pIndex);

    void setG(std::function<Real(Position, Real, RealVector &, RealVector &, RealVector &, RealVector &)> gin) { g = gin; }
    void setNp(int n) { AdjointProblem::np = n; }

private:
    std::function<Real(Position, Real, RealVector &, RealVector &, RealVector &, RealVector &)> g;

    using integrator = boost::math::quadrature::gauss_kronrod<double, 15>;
    constexpr static int max_depth = 2;
    AutodiffTransportSystem *PhysicsProblem;

    std::map<std::tuple<Index, Index>, bool> upperBoundarySensitivities;
    std::map<std::tuple<Index, Index>, bool> lowerBoundarySensitivities;
};
#endif