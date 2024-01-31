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
	explicit AutodiffTransportSystem(toml::value const &config, Grid const &, Index nVars, Index nScalars);

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
	Position xR, xL;

	using FluxWrapper = std::function<Real(Position, Time, std::vector<Value> *)>;
	using GradWrapper = std::function<Values(Position, Time, std::vector<Value> *)>;

private:
	// API to underlying flux model
	virtual Real Flux(Index, RealVector, RealVector, Position, Time, std::vector<Position> * = nullptr) = 0;
	virtual Real Source(Index, RealVector, RealVector, RealVector, Position, Time, std::vector<Position> * = nullptr) = 0;

	// Postprocessor for fluxes and sources, e.g. to do flux surface averages
	virtual Real Postprocessor(const FluxWrapper &f, Position x, Time t) = 0;
	virtual Values Postprocessor(const GradWrapper &f, Position x, Time t) = 0;

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

	autodiff::dual2nd InitialFunction(Index i, autodiff::dual2nd x, autodiff::dual2nd t, double u_R, double u_L, double x_L, double x_R) const;
};

// default postprocessor - just call the input function using the given arguments
// template <typename T, typename... Args>
// inline auto AutodiffTransportSystem::Postprocessor(const T &f, Args... args)
// {
// 	return f(args...);
// }
#endif
