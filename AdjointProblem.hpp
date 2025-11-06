#ifndef ADJOINTPROBLEM_HPP
#define ADJOINTPROBLEM_HPP

#include "Types.hpp"
#include "DGSoln.hpp"

class AdjointProblem
{
public:
    virtual ~AdjointProblem() = default;

    virtual Value GFn(Index i, DGSoln &y) const = 0;
    virtual Value dGFndp(Index i, DGSoln &y) const = 0;

    // We're assuming Gfn = Int gFn dx for now
    virtual Value gFn(Index i, const State &s, Position x) const = 0;
    // For compute g_y
    virtual void dgFn_du(Index i, Values &, const State &s, Position x) = 0;
    virtual void dgFn_dq(Index i, Values &, const State &s, Position x) = 0;
    virtual void dgFn_dsigma(Index i, Values &, const State &s, Position x) = 0;
    virtual void dgFn_dphi(Index i, Values &, const State &s, Position x) = 0;
    // For computing F_p
    virtual void dSigmaFn_dp(Index i, Index pIndex, Value &, const State &s, Position x) = 0;
    virtual void dSources_dp(Index i, Index pIndex, Value &, const State &s, Position x) = 0;

    virtual bool computeUpperBoundarySensitivity(Index i, Index pIndex) { return false; };
    virtual bool computeLowerBoundarySensitivity(Index i, Index pIndex) { return false; };

    int getNp() const { return np; }
    int getNpBoundary() const { return np_boundary; }

protected:
    int np;
    int np_boundary = 0;
};
#endif