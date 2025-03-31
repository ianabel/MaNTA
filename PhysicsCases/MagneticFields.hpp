#ifndef MAGNETICFIELDS_HPP
#define MAGNETICFIELDS_HPP

#include <numbers>
#include <cmath>
#include <netcdf>
#include <string>
#include <vector>
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
#include <autodiff/forward/dual.hpp>
#include "../util/trapezoid.hpp"
using std::numbers::pi;
using Real = autodiff::dual;
using Real2nd = autodiff::dual2nd;

// Magnetic field
// All of these values are returned in SI units
// (we may later introduce a B0 & R0, but these are 1T & 1m for now)

// Base magnetic field class
// Functions depend on the flux surface volume and another coordinate
// This is usually the arc length "s", but it is sometimes easier to make it another coordinate, so we leave it open ended
// The base implementation should be of the "Real" type
class MagneticField
{
public:
	MagneticField() = default;
	virtual Real Psi_V(Real V) const = 0;
	virtual double Psi_V(double V) const { return Psi_V(static_cast<Real>(V)).val; }

	virtual Real B(Real V, Real) const = 0;
	virtual double B(double V, double z) const { return B(static_cast<Real>(V), static_cast<Real>(z)).val; }

	virtual Real R(Real Psi, Real) const = 0;
	virtual double R(double Psi, double s) const { return R(static_cast<Real>(Psi), static_cast<Real>(s)).val; }

	virtual Real R_V(Real V, Real s) const { return R(Psi_V(V), s); }
	virtual double R_V(double V, double s) const { return R_V(static_cast<Real>(V), static_cast<Real>(s)).val; }
	virtual Real2nd R_V(Real2nd V, Real2nd s) const { throw std::logic_error("If 2nd derivative behavior is required it must be explicitly implemented in your magnetic field class"); }

	virtual Real dRdV(Real V, Real) const = 0;
	virtual double dRdV(double V, double s) const { return dRdV(static_cast<Real>(V), static_cast<Real>(s)).val; };

	virtual Real VPrime(Real V) const = 0;
	virtual double VPrime(double V) const { return VPrime(static_cast<Real>(V)).val; }

	virtual Real MirrorRatio(Real V, Real) const = 0;
	virtual double MirrorRatio(double V, double s) const { return MirrorRatio(static_cast<Real>(V), static_cast<Real>(s)).val; }
	virtual Real2nd MirrorRatio(Real2nd V, Real2nd s) const { throw std::logic_error("If 2nd derivative behavior is required it must be explicitly implemented in your magnetic field class"); }

	virtual Real L_V(Real V) const = 0;
	virtual double L_V(double V) const { return L_V(static_cast<Real>(V)).val; }

	virtual Real Rmax(Real V) const { return R(Psi_V(V), 0.5 * (LeftEndpoint(Psi_V(V)) + RightEndpoint(Psi_V(V)))); }
	virtual Real Rmin(Real V) const { return R(Psi_V(V), RightEndpoint(Psi_V(V))); }

	/// @brief Computes the flux surface average of a function f(V,s).
	/// @tparam F is a generic function argument
	/// @param f is the function that is being flux surface averaged, must be a function of the flux surface volume and another coordinate
	/// @param V is the flux surface volume
	/// @return Flux surface average <f>
	template <typename F>
	Real FluxSurfaceAverage(const F &f, Real V) const
	{
		auto Integrand = [&](Real s)
		{ return f(s) / B_s(Psi_V(V), s); };
		Real I = 2 * M_PI / VPrime(V) * trapezoid(Integrand, LeftEndpoint(Psi_V(V)), RightEndpoint(Psi_V(V)), 1e-3);
		return I;
	}

private:
	virtual Real LeftEndpoint(Real Psi) const = 0;
	virtual Real RightEndpoint(Real Psi) const = 0;

	virtual Real B_s(Real Psi, Real s) const { return B(Psi, s); }
};

// Just a uniform magnetic field
// Leave these template functions for now so we don't break MirrorPlasma
class StraightMagneticField : public MagneticField
{
public:
	StraightMagneticField() = default;
	StraightMagneticField(double L_z, double B_z, double Rm) : L_z(L_z), B_z(B_z), Rm(Rm) {};

	StraightMagneticField(double L_z, double B_z, double Rm, double Vmin, double m) : L_z(L_z), B_z(B_z), Rm(Rm), Vmin(Vmin), m(m) {};

	// Override base class methods by calling template versions
	virtual Real B(Real V, Real) const override { return B_T(V); };
	virtual Real Psi_V(Real V) const override { return Psi_V_T(V); };

	Real VPrime(Real V) const override
	{
		return VPrime_T(V); // 2 * pi * L_z / B_T(V);
	}

	Real R(Real Psi, Real) const override { return sqrt(2 * Psi / B_z); };

	Real R_V(Real V, Real) const override { return R_V_T(V); }
	Real2nd R_V(Real2nd V, Real2nd) const override { return R_V_T(V); }

	Real dRdV(Real V, Real) const override { return dRdV_T(V); }

	Real MirrorRatio(Real V, Real) const override { return MirrorRatio_T(V); }
	Real2nd MirrorRatio(Real2nd V, Real2nd) const override { return MirrorRatio_T(V); }

	Real L_V(Real V) const override { return L_V_T(V); }

private:
	// template versions
	template <typename T>
	T B_T(T V) const
	{
		return B_z - m * (V - Vmin);
	}
	template <typename T>
	T Psi(T R) const
	{
		return R * R * Bz_R(R) / 2.0;
	}
	template <typename T>
	T Psi_V_T(T V) const
	{
		return B_T(V) * V / (2 * pi * L_V_T(V));
	}
	template <typename T>
	T VPrime_T(T V) const
	{
		return 2 * pi * L_V_T(V) / B_T(V);
	}

	template <typename T>
	T R_V_T(T V) const
	{
		return sqrt(V / (pi * L_z));
	}
	template <typename T>
	T dRdV_T(T V) const
	{
		return 1.0 / (2 * pi * L_V_T(V) * R_V_T(V));
	}
	template <typename T>
	T MirrorRatio_T(T V) const
	{
		return Rm * B_z / B_T(V);
	}
	template <typename T>
	T L_V_T(T) const
	{
		return L_z;
	}

private:
	virtual Real LeftEndpoint(Real Psi) const override { return 0.0; };
	virtual Real RightEndpoint(Real Psi) const override { return L_z; };
	double L_z = 0.6;
	double B_z = 0.3;
	double Rm = 10.0;

	double Vmin = 0.0;
	double m = 0.0;
};

class CylindricalMagneticField : public MagneticField
{
public:
	CylindricalMagneticField(const std::string &file);
	~CylindricalMagneticField() = default;

	virtual double B(double V, double z = 0.0) const override;
	virtual Real B(Real V, Real z = 0.0) const override
	{
		Real Bz = B(V.val);
		if (V.grad != 0)
			Bz.grad = B_spline->prime(V.val);
		return Bz;
	}

	double Psi_V(double V) const override;
	Real Psi_V(Real V) const override
	{
		Real Psi = Psi_V(V.val);
		if (V.grad != 0)
			Psi.grad = Psi_spline->prime(V.val);
		return Psi;
	}

	double VPrime(double V) const override;
	Real VPrime(Real V) const override
	{
		Real Vp = (*Vp_spline)(V.val);
		if (V.grad != 0)
			Vp.grad = Vp_spline->prime(V.val);
		return Vp;
	};

	double R_V(double V, double s = 0.0) const override;
	Real R_V(Real V, Real s = 0.0) const override
	{
		Real R = R_V(V.val);
		if (V.grad != 0.0)
			R.grad = dRdV(V.val);
		return R;
	};
	Real2nd R_V(Real2nd V, Real2nd s = 0.0) const override
	{
		Real2nd R = R_V(V.val);
		if (V.grad != 0.0)
			R.grad.val = dRdV(V.val.val);
		return R;
	};

	Real R(Real Psi, Real) const override { return sqrt(2 * Psi / B(Psi)); };

	double dRdV(double V, double s = 0.0) const override;
	Real dRdV(Real V, Real s = 0.0) const override
	{
		Real drdv = dRdV(V.val);
		if (V.grad != 0)
			drdv.grad = dRdV_spline->prime(V.val);
		return drdv;
	}

	double MirrorRatio(double V, double s = 0.0) const override;
	Real MirrorRatio(Real V, Real s = 0.0) const override
	{
		Real Rm = MirrorRatio(V.val);
		if (V.grad != 0)
			Rm.grad = Rm_spline->prime(V.val);
		return Rm;
	};
	Real2nd MirrorRatio(Real2nd V, Real2nd s = 0.0) const override
	{
		Real2nd Rm = MirrorRatio(V.val);
		if (V.grad != 0)
			Rm.grad.val = Rm_spline->prime(V.val.val);
		return Rm;
	};

	double L_V(double V) const override;
	Real L_V(Real V) const override
	{
		Real L = L_V(V.val);
		if (V.grad != 0)
			L.grad = L_spline->prime(V.val);
		return L;
	}

	void CheckBoundaries(double VL, double VR) const;

private:
	using spline = boost::math::interpolators::cardinal_cubic_b_spline<double>;

	virtual Real LeftEndpoint(Real V) const override { return 0.0; };
	virtual Real RightEndpoint(Real V) const override { return L_V(V); };
	// double R_root_solver(double Psi);

	std::string filename;
	std::vector<double> gridpoints;
	netCDF::NcFile data_file;
	unsigned int nPoints;
	netCDF::NcDim V_dim;
	std::vector<double> V;

	std::unique_ptr<spline> B_spline;
	std::unique_ptr<spline> Vp_spline;
	std::unique_ptr<spline> Psi_spline;
	std::unique_ptr<spline> Rm_spline;
	std::unique_ptr<spline> R_V_spline;
	std::unique_ptr<spline> dRdV_spline;
	std::unique_ptr<spline> L_spline;
};

// Function for creating an instance of a magnetic field
// Allow for constructors that take a different number of arguments
template <typename T, typename... Args>
static std::shared_ptr<MagneticField> createMagneticField(Args &&...args)
{
	return std::make_shared<T>(std::forward<Args>(args)...);
};

#endif // MAGNETICFIELDS_HPP
