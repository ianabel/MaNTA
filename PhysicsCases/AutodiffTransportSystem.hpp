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
	AutodiffTransportSystem() = default;
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

	virtual autodiff::dual2nd InitialFunction(Index i, autodiff::dual2nd x, autodiff::dual2nd t) const;

protected:
	Position xR, xL;
	bool loadInitialConditionsFromFile = false;
	std::string filename;
	void LoadDataToSpline(const std::string &file);
	bool useMMS = false;

	double growth_rate = 0.5;
	double growth = 1.0;

private:
	// API to underlying flux model
	virtual Real Flux(Index, RealVector, RealVector, Position, Time) { return 0; };
	virtual Real Source(Index, RealVector, RealVector, RealVector, Position, Time) { return 0; };

	virtual Real Flux(Index i, RealVector u, RealVector q, Real x, Time t) { return Flux(i, u, q, x.val, t); }
	virtual Real Source(Index i, RealVector u, RealVector q, RealVector sigma, Real x, Time t) { return Source(i, u, q, sigma, x.val, t); };

	// For loading initial conditions from a netCDF file
	netCDF::NcFile data_file;
	std::vector<std::unique_ptr<spline>> NcFileInitialValues;
	std::vector<std::unique_ptr<spline>> NcFileInitialDerivatives;

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

	virtual autodiff::dual2nd MMS_Solution(Index i, Real2nd x, Real2nd t);

	Value MMS_Source(Index, Position, Time);
};
#endif
