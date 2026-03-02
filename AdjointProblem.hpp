#ifndef ADJOINTPROBLEM_HPP
#define ADJOINTPROBLEM_HPP

#include "Types.hpp"
#include "DGSoln.hpp"

class AdjointProblem
{
public:
    virtual ~AdjointProblem() = default;

    virtual Value GFn(Index gIndex, DGSoln &y) const = 0;
    virtual Value dGFndp(Index gIndex, DGSoln &y) const = 0;

    // We're assuming Gfn = Int gFn dx for now
    virtual Value gFn(Index gIndex, const State &s, Position x) const = 0;
    virtual Value dgFndp(Index gIndex, const State &s, Position x) const
    {
        throw std::runtime_error("Virtual function dgFndp only for use within python class.");
    }
    // For compute g_y
    virtual void dgFn_du(Index gIndex, VectorRef, const State &s, Position x) = 0;
    virtual void dgFn_dq(Index gIndex, VectorRef, const State &s, Position x) = 0;
    virtual void dgFn_dsigma(Index gIndex, VectorRef, const State &s, Position x) = 0;
    virtual void dgFn_dphi(Index gIndex, VectorRef, const State &s, Position x) = 0;

    virtual void dg(Index gIndex, GlobalState &out, GlobalState const &states, std::vector<Position> const &abscissae)
    {
        for (size_t j = 0; j < states.size(); ++j)
        {
            dgFn_du(gIndex, out.Variable(j), states[j], abscissae[j]);
            dgFn_dq(gIndex, out.Derivative(j), states[j], abscissae[j]);
            dgFn_dsigma(gIndex, out.Flux(j), states[j], abscissae[j]);
            dgFn_dphi(gIndex, out.Aux(j), states[j], abscissae[j]);
        }
    }
    // For computing F_p
    virtual void dSigmaFn_dp(Index i, Index pIndex, Value &, const State &s, Position x) = 0;
    virtual void dSources_dp(Index i, Index pIndex, Value &, const State &s, Position x) = 0;

    virtual void dSigma(Index i, GlobalState &out, GlobalState const &states, std::vector<Position> const &abscissae)
    {
        for (size_t j = 0; j < states.size(); ++j)
        {
            for (Index pIndex = 0; pIndex < getNpInternal(); ++pIndex)
            {
                auto &vout = out.Variable(j)(pIndex); // we use the variable to represent p derivatives
                dSigmaFn_dp(i, pIndex, vout, states[j], abscissae[j]);
            }
        }
    }
    virtual void dSources(Index i, GlobalState &out, GlobalState const &states, std::vector<Position> const &abscissae)
    {
        for (size_t j = 0; j < states.size(); ++j)
        {
            for (Index pIndex = 0; pIndex < getNpInternal(); ++pIndex)
            {
                auto &vout = out.Variable(j)(pIndex); // we use the variable to represent p derivatives
                dSources_dp(i, pIndex, vout, states[j], abscissae[j]);
            }
        }
    }

    virtual void dAux_dp(Index i, Index pIndex, Value &, const State &s, Position x)
    {
        std::logic_error("nAux > 0 but no G derivative provided");
    };

    virtual std::string getName(Index pIndex) const { return "p" + std::to_string(pIndex); };

    virtual bool computeUpperBoundarySensitivity(Index i, Index pIndex) { return false; };
    virtual bool computeLowerBoundarySensitivity(Index i, Index pIndex) { return false; };

    int getNp() const { return np; }
    int getNpBoundary() const { return np_boundary; }

    int getNpInternal() const { return np - np_boundary; }

    // True if internal index ; false if boundary index
    inline bool isAdjointIndexInternal(int pIndex) const
    {
        return (pIndex < np - np_boundary);
    }

protected:
    int np;
    int np_boundary = 0;
};
#endif
