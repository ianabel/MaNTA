#ifndef ADJOINTPROBLEM_HPP
#define ADJOINTPROBLEM_HPP

#include "Types.hpp"
#include "DGSoln.hpp"

template <typename T, typename... Args>
class AdjointProblem
{
public:
    virtual ~AdjointProblem() = default;

    virtual Value GFn(Index i, DGSoln &y) const = 0;
    virtual Value dGFndp(Index i, DGSoln &y) const = 0;

    virtual void dgFn_du(Index i, Values &, const State &s, Position x) const = 0;
    virtual void dgFn_dq(Index i, Values &, const State &s, Position x) const = 0;
    virtual void dgFn_dsigma(Index i, Values &, const State &s, Position x) const = 0;
    // For computing F_p
    virtual void dSigmaFn_dp(Index i, Value &, const State &s, Position x, Time t) const = 0;
    virtual void dSources_dp(Index i, Value &, const State &s, Position x, Time t) const = 0;

    int getNp() const { return np; }
    void setNp(int n) const { np = n; }

    void setG(std::function<T(Position, T, Args &...args)> g) { gFn = g; }


protected:
    int np;
    std::function<T(Position, T, Args &...args)> gFn;
};
#endif