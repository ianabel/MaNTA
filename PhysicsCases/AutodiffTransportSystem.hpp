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
    explicit AutodiffTransportSystem( toml::value const &config, Grid const&, Index nVars, Index nScalars );

	 // Implement the TransportSystem interface.
    Value SigmaFn(Index i, const State &, Position x, Time t) override;
    Value Sources(Index i, const State &, Position x, Time t) override;

    void dSigmaFn_du(Index i, Values &, const State &, Position x, Time t) override;
    void dSigmaFn_dq(Index i, Values &, const State &, Position x, Time t) override;

    void dSources_du(Index i, Values &, const State &, Position x, Time t) override;
    void dSources_dq(Index i, Values &, const State &, Position x, Time t) override;
    void dSources_dsigma(Index i, Values &, const State &, Position x, Time t) override;

    // and initial conditions for u & q
    virtual Value InitialValue(Index i, Position x) const override;
    virtual Value InitialDerivative(Index i, Position x) const override;

protected:
	Position xR,xL;
private:
	 // API to underlying flux model
	virtual Real Flux( Index, RealVector, RealVector, Position, Time ) = 0;
	virtual Real Source( Index, RealVector, RealVector, RealVector, Position, Time ) = 0;

	static std::vector<int> InitialProfile;
	enum ProfileType
	{
		Gaussian,
		Cosine,
		CosineSquared,
		Uniform,
		Linear,
	};
	std::map<std::string, int> InitialProfiles = {{"Gaussian", Gaussian}, {"Cosine", Cosine}, {"CosineSquared",CosineSquared}, {"Uniform", Uniform}, {"Linear", Linear}};

	static Vector InitialHeights;


	static autodiff::dual2nd DirichletIC(Index i, autodiff::dual2nd x, autodiff::dual2nd t, double u_R, double u_L, double x_L, double x_R);

	static autodiff::dual2nd InitialFunction(Index i, autodiff::dual2nd x, autodiff::dual2nd t, double u_R, double u_L, double x_L, double x_R);

};
#endif
