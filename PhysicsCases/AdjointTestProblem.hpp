#ifndef ADTESTPROBLEM
#define ADTESTPROBLEM

#include "AutodiffTransportSystem.hpp"
#include "AutodiffAdjointProblem.hpp"

class AdjointTestProblem : public AutodiffTransportSystem
{
public:
    AdjointTestProblem(toml::value const &config, Grid const &grid);

    virtual AdjointProblem *createAdjointProblem() override
    {
        AutodiffAdjointProblem *p = new AutodiffAdjointProblem(this);
        p->setG([this](Position x, Real p, RealVector &u, RealVector &q, RealVector &sigma, RealVector &phi)
                { return this->g(x, p, u, q, sigma, phi); });
        return p;
    }

    Real g(Position, Real, RealVector &, RealVector &, RealVector &, RealVector &);

private:
    Real Flux(Index, RealVector, RealVector, Real, Time) override;
    Real Source(Index, RealVector, RealVector, RealVector, RealVector, Real, Time) override;

    Real T_s, D, SourceWidth, SourceCentre;

    REGISTER_PHYSICS_HEADER(AdjointTestProblem)
};

#endif
