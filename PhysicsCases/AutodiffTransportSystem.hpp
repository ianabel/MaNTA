#ifndef AUTODIFFTRANSPORTSYSTEM_HPP
#define AUTODIFFTRANSPORTSYSTEM_HPP

#include "PhysicsCases.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

using Real = autodiff::dual;
using RealVector = autodiff::VectorXdual;

class AutodiffTransportSystem : public TransportSystem
{
public:
    explicit AutodiffTransportSystem( toml::value const &config, Grid const& );

    //  Function for passing boundary conditions to the solver
    virtual Value LowerBoundary(Index i, Time t) const override { return InitialFunction(i, xL, 0.0, uR(i), uL(i), xL, xR).val.val; };
    virtual Value UpperBoundary(Index i, Time t) const override { return InitialFunction(i, xR, 0.0, uR(i), uL(i), xL, xR).val.val; };

    bool isLowerBoundaryDirichlet(Index i) const override { return isLowerDirichlet;};
    bool isUpperBoundaryDirichlet(Index i) const override { return isUpperDirichlet;};

	 // Implement the TransportSystem interface.
    Value SigmaFn(Index i, const State &, Position x, Time t) override;
    Value Sources(Index i, const State &, Position x, Time t) override;

    void dSigmaFn_du(Index i, Values &, const State &, Position x, Time t) override;
    void dSigmaFn_dq(Index i, Values &, const State &, Position x, Time t) override;

    void dSources_du(Index i, Values &, const State &, Position x, Time t) override;
    void dSources_dq(Index i, Values &, const State &, Position x, Time t) override;
    void dSources_dsigma(Index i, Values &, const State &, Position x, Time t) override;

    // and initial conditions for u & q
    Value InitialValue(Index i, Position x) const override;
    Value InitialDerivative(Index i, Position x) const override;

private:
	 // API to underlying flux model
	 virtual Real Flux( Index, RealVector, RealVector, Position, Time ) = 0;
	 virtual Real Source( Index, RealVector, RealVector, RealVector, Position, Time ) = 0;

    Values uL;
    bool isUpperDirichlet;

    Values uR;
    bool isLowerDirichlet;

    Position xL;
    Position xR;

    static std::vector<int> InitialProfile;
    std::map<std::string, int> InitialProfiles = {{"Gaussian", 0}, {"Dirichlet", 1}, {"Cosine", 2}, {"Uniform", 3}, {"Linear", 4}};

    static Vector InitialHeights;

    static autodiff::dual2nd DirichletIC(Index i, autodiff::dual2nd x, autodiff::dual2nd t, double u_R, double u_L, double x_L, double x_R);

    static autodiff::dual2nd InitialFunction(Index i, autodiff::dual2nd x, autodiff::dual2nd t, double u_R, double u_L, double x_L, double x_R);

};
#endif
