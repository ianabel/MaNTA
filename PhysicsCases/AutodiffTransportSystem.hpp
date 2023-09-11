#ifndef AUTODIFFTRANSPORTSYSTEM_HPP
#define AUTODIFFTRANSPORTSYSTEM_HPP

#include "PhysicsCases.hpp"
#include "AutodiffFlux.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

class AutodiffTransportSystem : public TransportSystem
{
public:
    explicit AutodiffTransportSystem(toml::value const &config);
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

    std::shared_ptr<FluxObject> fluxObject = nullptr;

    bool isTestProblem;

    static int InitialProfile;
    std::map<std::string, int> InitialProfiles = {{"Gaussian", 0}, {"Dirichlet", 1}};

    static Vector InitialHeights;

    static dual2nd TestDirichlet(Index i, dual2nd x, dual2nd t, double u_R, double u_L, double x_L, double x_R);

    static dual2nd InitialFunction(Index i, dual2nd x, dual2nd t, double u_R, double u_L, double x_L, double x_R);

    REGISTER_PHYSICS_HEADER(AutodiffTransportSystem)
};
#endif