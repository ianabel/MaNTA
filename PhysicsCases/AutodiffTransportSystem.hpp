#ifndef AUTODIFFTRANSPORTSYSTEM_HPP
#define AUTODIFFTRANSPORTSYSTEM_HPP

#include "PhysicsCases.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
using spline = boost::math::interpolators::cardinal_cubic_b_spline<double>;

using Real = autodiff::dual;
using Real2nd = autodiff::dual2nd;
using RealVector = autodiff::VectorXdual;

class AutodiffTransportSystem : public TransportSystem
{
public:
	explicit AutodiffTransportSystem(toml::value const &config, Grid const &, Index nVars, Index nScalars, Index nAux);

	// Implement the TransportSystem interface.
	Value SigmaFn(Index i, const State &, Position x, Time t) override;
	Value Sources(Index i, const State &, Position x, Time t) override;

	void dSigmaFn_du(Index i, Values &, const State &, Position x, Time t) override;
	void dSigmaFn_dq(Index i, Values &, const State &, Position x, Time t) override;

	void dSources_du(Index i, Values &, const State &, Position x, Time t) override;
	void dSources_dq(Index i, Values &, const State &, Position x, Time t) override;
	void dSources_dsigma(Index i, Values &, const State &, Position x, Time t) override;

	Value AuxG(Index, const State &, Position, Time) override;
	void AuxGPrime(Index, State &, const State &, Position, Time) override;
	void dSources_dPhi(Index, Values &, const State &, Position, Time) override;

	// and initial conditions for u & q
	virtual Value InitialValue(Index i, Position x) const override;
	virtual Value InitialDerivative(Index i, Position x) const override;

	virtual autodiff::dual2nd InitialFunction(Index i, autodiff::dual2nd x, autodiff::dual2nd t) const;

protected:
	Position xR, xL;

private:
	// API to underlying flux model
	virtual Real Flux(Index, RealVector, RealVector, Position, Time) = 0;
	virtual Real Source(Index, RealVector, RealVector, RealVector, RealVector, Position, Time) = 0;

	virtual Real Phi(Index, RealVector, RealVector, RealVector, RealVector, Position, Time)
	{
		if (nAux > 0)
			throw std::logic_error("nAux > 0 but no implementation of auxiliary variable provided");
		else
			return 0.0;
	};

	enum class ProfileType
	{
		Gaussian,
		Cosine,
		CosineSquared,
		Uniform,
		Linear,
	};
	std::vector<ProfileType> InitialProfile;

	std::map<std::string, ProfileType> InitialProfiles = {{"Gaussian", ProfileType::Gaussian}, {"Cosine", ProfileType::Cosine}, {"CosineSquared", ProfileType::CosineSquared}, {"Uniform", ProfileType::Uniform}, {"Linear", ProfileType::Linear}};

	Vector InitialHeights;

	autodiff::dual2nd DirichletIC(Index i, autodiff::dual2nd x, autodiff::dual2nd t, double u_R, double u_L, double x_L, double x_R) const;
};
#endif
