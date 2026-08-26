#ifndef AUTODIFFTRANSPORTSYSTEM_HPP
#define AUTODIFFTRANSPORTSYSTEM_HPP

#include "PhysicsCases.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include "AdjointProblem.hpp"

#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
using spline = boost::math::interpolators::cardinal_cubic_b_spline<double>;

using Real = autodiff::dual;
using Real2nd = autodiff::dual2nd;
using RealVector = autodiff::VectorXdual;
using Real2ndVector = autodiff::VectorXdual2nd;

class AutodiffTransportSystem : public TransportSystem
{
public:
	explicit AutodiffTransportSystem(toml::value const &config, Grid const &, SystemSpec spec);

	/// Apply the `[AutodiffTransportSystem]` boundary keys to a spec.
	///
	/// `isUpperDirichlet` / `isLowerDirichlet` are single flags applied to every
	/// variable, which is what they have always meant. They are honoured only
	/// when actually present, so a case that declares per-variable boundary
	/// kinds in its own spec keeps them.
	static SystemSpec withBoundaryConfig(SystemSpec spec, toml::value const &config);

	// Implement the TransportSystem interface.
	Value SigmaFn(Index i, const State &, Position x, Time t) override;
	Value Sources(Index i, const State &, Position x, Time t) override;

	void dSigmaFn_du(Index i, VectorRef, const State &, Position x, Time t) override;
	void dSigmaFn_dq(Index i, VectorRef, const State &, Position x, Time t) override;

	void dSources_du(Index i, VectorRef, const State &, Position x, Time t) override;
	void dSources_dq(Index i, VectorRef, const State &, Position x, Time t) override;
	void dSources_dsigma(Index i, VectorRef, const State &, Position x, Time t) override;
	void dSources_dScalars(Index, VectorRef, const State &, Position, Time) override;
	void dSigmaFn_dp(Index i, Index pIndex, Value &, const State &s, Position x, Time t);
	void dSources_dp(Index i, Index pIndex, Value &, const State &, Position x, Time t);

	// Geometry derivatives -- see TransportSystem.hpp for the contract. Derived
	// by autodiff like every other derivative here: SigmaFn/Sources/AuxG all
	// evaluate through the geometry-aware Flux/Source/GFunc overloads below, so
	// a case that never reads geometry gets an identically zero gradient from
	// the same mechanism that gives one to a case that does.
	void dSigmaFn_dGeometry(Index i, VectorRef, const State &, Position x, Time t) override;
	void dSources_dGeometry(Index i, VectorRef, const State &, Position x, Time t) override;
	void dAuxG_dGeometry(Index i, VectorRef, const State &, Position x, Time t) override;

	Value AuxG(Index, const State &, Position, Time) override;
	void AuxGPrime(Index, State &, const State &, Position, Time) override;
	void dSources_dPhi(Index, VectorRef, const State &, Position, Time) override;

	// and initial conditions for u & q
	virtual Value InitialValue(Index i, Position x) const override;
	virtual Value InitialDerivative(Index i, Position x) const override;

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

	virtual void setPval(Index i, Real p)
	{
		if (static_cast<size_t>(i) < pvals.size())
			pvals[i].get() = p;
	}
	virtual Real getPval(Index i) const
	{
		if (static_cast<size_t>(i) < pvals.size())
			return pvals[i].get();
		else
			return 0.0;
	}

	// Set gradients of p values to 0
	void clearGradients()
	{
		for (auto pval : pvals)
			pval.get().grad = 0.0;
	}

	Position xR, xL;

protected:
	// For loading initial conditions from netCDF file, needs to be accessed by child classes
	bool loadInitialConditionsFromFile = false;
	std::string filename;
	void LoadDataToSpline(const std::string &file);

	// MMS Solution
	bool useMMS = false;
	double growth_rate = 0.5;
	double growth = 1.0;
	virtual Real2nd MMS_Solution(Index i, Real2nd x, Real2nd t);
	Value MMS_Source(Index, Position, Time);

	void initialiseDiagnostics(NetCDFIO &nc) override;
	void writeDiagnostics(DGSoln const &y, DGSoln const &dydt, Time t, NetCDFIO &nc, size_t tIndex) override;

	void addP(std::reference_wrapper<Real> p)
	{
		pvals.push_back(p);
	}
	std::vector<std::reference_wrapper<Real>> pvals;

private:
	// API to underlying flux models
	virtual Real Flux(Index i, RealVector u, RealVector q, Real x, Time t) = 0;

	/// The geometry-aware overload. Defaults to ignoring geometry and forwarding
	/// to the required overload above, so an existing case's Flux keeps
	/// compiling unchanged and contributes zero geometry coupling -- the same
	/// "an unread hook means no coupling" convention as TransportSystem's own
	/// default dSigmaFn_dGeometry. SigmaFn and every d.../dGeometry hook below
	/// evaluate through this overload rather than the one above, so a case that
	/// wants sigma_hat to depend on geometry overrides this one instead and gets
	/// the derivative for free by the same autodiff mechanism as du/dq.
	virtual Real Flux(Index i, RealVector u, RealVector q, RealVector /* geom */, Real x, Time t)
	{
		return Flux(i, u, q, x, t);
	}

	virtual Real Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector phi, Real x, Time t)
	{
		if (nScalars > 0)
		{
			throw std::logic_error("nScalars > 0 but no implementation of scalar sources provided");
		}
		else
			return 0.0;
	}
	virtual Real Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector phi, RealVector Scalars, Real x, Time t)
	{
		if (nScalars > 0)
		{
			throw std::logic_error("nScalars > 0 but no implementation of scalar sources provided");
		}
		else
			return Source(i, u, q, sigma, phi, x, t);
	}

	/// The geometry-aware overload, on the same footing as Flux's above.
	virtual Real Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector phi, RealVector Scalars, RealVector /* geom */, Real x, Time t)
	{
		return Source(i, u, q, sigma, phi, Scalars, x, t);
	}

	// Auxiliary variables are optional, so provide a default implementation
	virtual Real GFunc(Index, RealVector, RealVector, RealVector, RealVector, Position, Time)
	{
		if (nAux > 0)
			throw std::logic_error("nAux > 0 but no implementation of auxiliary variable function provided");
		else
			return 0.0;
	};

	/// The geometry-aware overload, on the same footing as Flux's above.
	virtual Real GFunc(Index i, RealVector u, RealVector q, RealVector sigma, RealVector phi, RealVector /* geom */, Position x, Time t)
	{
		return GFunc(i, u, q, sigma, phi, x, t);
	}

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
};

#define STRINGIFY(Var) #Var

#endif
