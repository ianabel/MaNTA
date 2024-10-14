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
using Real2ndVector = autodiff::VectorXdual2nd;

class AutodiffTransportSystem : public TransportSystem
{
public:
	AutodiffTransportSystem() = default;
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
	Value InitialValue(Index i, Position x) const override;
	Value InitialDerivative(Index i, Position x) const override;

	// Override base initial phi value, add t dependency for MMS solutions
	Value InitialAuxValue(Index i, Position x) const override
	{
		if (loadInitialConditionsFromFile && nAux > 0)
		{
			return (*NcFileInitialAuxValue[i])(x);
		}
		else
		{
			return InitialAuxValue(i, x, 0.0);
		}
	};

	virtual Real2nd InitialFunction(Index i, Real2nd x, Real2nd t) const;

protected:
	Position xR, xL;

	// For loading initial conditions from netCDF file, needs to be accessed by child classes
	bool loadInitialConditionsFromFile = false;
	std::string filename;
	void LoadDataToSpline(const std::string &file);

	// MMS Solution
	bool useMMS = false;
	double growth_rate = 0.5;
	double growth = 1.0;
	virtual Real2nd MMS_Solution(Index i, Real2nd x, Real2nd t);

	void initialiseDiagnostics(NetCDFIO &nc) override;
	void writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex) override;

private:
	// API to underlying flux models
	virtual Real Flux(Index i, RealVector u, RealVector q, Real x, Time t) = 0;
	virtual Real Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector phi, Real x, Time t) = 0;

	// Auxiliary variables are optional, so provide a default implementation
	virtual Real GFunc(Index, RealVector, RealVector, RealVector, RealVector, Position, Time)
	{
		if (nAux > 0)
			throw std::logic_error("nAux > 0 but no implementation of auxiliary variable function provided");
		else
			return 0.0;
	};

	virtual Value InitialAuxValue(Index, Position, Time) const
	{
		if (nAux > 0)
			throw std::logic_error("nAux > 0 but no implementation of auxiliary variable initial value provided");
		else
			return 0.0;
	};

	// For loading initial conditions from a netCDF file
	netCDF::NcFile data_file;
	std::vector<std::unique_ptr<spline>> NcFileInitialValues;
	std::vector<std::unique_ptr<spline>> NcFileInitialDerivatives;
	std::vector<std::unique_ptr<spline>> NcFileInitialAuxValue;

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

	Value MMS_Source(Index, Position, Time);
};
#endif
