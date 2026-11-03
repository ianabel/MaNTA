#ifndef AUTODIFFADJOINTPROBLEM_HPP
#define AUTODIFFADJOINTPROBLEM_HPP

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include "AutodiffTransportSystem.hpp"
#include "Types.hpp"
#include "AdjointProblem.hpp"

using ftype = std::function<Real(Position, RealVector &, RealVector &, RealVector &, RealVector &)>;
/// An objective that may also read the geometry slots a field model provides,
/// passed last; the vector is empty when no field model is attached.
using ftypeGeom = std::function<Real(Position, RealVector &, RealVector &, RealVector &, RealVector &, RealVector &)>;

class AutodiffAdjointProblem : public AdjointProblem
{
public:
    AutodiffAdjointProblem(AutodiffTransportSystem *TransportSystem) : PhysicsProblem(TransportSystem) { AdjointProblem::ng = 0; }

    // GFn is AdjointProblem's nodal default, the objective dg differentiates.
    virtual Value dGFndp(Index i, Index pIndex, DGSoln &y) const override;
    virtual Matrix dGFndp(Index i, DGSoln &y) const override;
    virtual Matrix dGFndp(Index i, DGSoln &y, Postprocessor const &pp) const override;
    virtual Matrix dGFndpAtNodes(Index i, GlobalState const &states, std::vector<Position> const &points,
                                 Vector const &weights, Index nPerCell, DGSoln &y,
                                 Postprocessor const *pp) const override;

    virtual Value gFn(Index i, const State &s, Position x) const override;

    virtual void dgFn_du(Index i, VectorRef, const State &s, Position x) override;
    virtual void dgFn_dq(Index i, VectorRef, const State &s, Position x) override;
    virtual void dgFn_dsigma(Index i, VectorRef, const State &s, Position x) override;
    virtual void dgFn_dphi(Index i, VectorRef, const State &s, Position x) override;
    virtual void dgFn_dgeom(Index i, VectorRef, const State &s, Position x) override;

    virtual void dSigmaFn_dp(Index i, Index pIndex, Value &, const State &s, Position x) override;
    virtual void dSources_dp(Index i, Index pIndex, Value &, const State &, Position x) override;
    virtual void dAux_dp(Index i, Index pIndex, Value &, const State &, Position x) override;

    virtual bool computeUpperBoundarySensitivity(Index i, Index pIndex) override;
    virtual bool computeLowerBoundarySensitivity(Index i, Index pIndex) override;

    void addUpperBoundarySensitivity(Index i, Index pIndex);
    void addLowerBoundarySensitivity(Index i, Index pIndex);

    void addG(ftype gin)
    {
        g.push_back([gin](Position x, RealVector &u, RealVector &q, RealVector &sigma, RealVector &phi, RealVector &)
                    { return gin(x, u, q, sigma, phi); });
        AdjointProblem::ng++;
    }
    /// An objective reading geometry as well; see ftypeGeom.
    void addGWithGeometry(ftypeGeom gin)
    {
        g.push_back(std::move(gin));
        AdjointProblem::ng++;
    }
    void setNp(int n) { AdjointProblem::np = n; }

private:
    std::vector<ftypeGeom> g;

    Value nodalParameterDerivative(Index i, Index pIndex, GlobalState const &states,
                                   std::vector<Position> const &points, Vector const &weights) const;
    Matrix nodalParameterDerivatives(Index i, GlobalState const &states,
                                     std::vector<Position> const &points, Vector const &weights) const;
    static Vector nodalWeights(DGSoln const &y);
    void gradientOf(Index i, int which, VectorRef grad, const State &s, Position x) const;
    AutodiffTransportSystem *PhysicsProblem;

    std::map<std::tuple<Index, Index>, bool> upperBoundarySensitivities;
    std::map<std::tuple<Index, Index>, bool> lowerBoundarySensitivities;
};

#endif