#ifndef MAGNETICFIELDS_HPP
#define MAGNETICFIELDS_HPP

#include <numbers>
#include <cmath>
#include <netcdf>
#include <string>
#include <vector>
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <autodiff/forward/dual.hpp>
#include "../util/trapezoid.hpp"
using std::numbers::pi;
using Real = autodiff::dual;
// Magnetic field
// All of these values are returned in SI units
// (we may later introduce a B0 & R0, but these are 1T & 1m for now)

// Base magnetic field class
// Functions depend on the flux surface volume and another coordinate
// This is usually the arc length "s", but it is sometimes easier to make it another coordinate, so we leave it open ended
class MagneticField
{
public:
	MagneticField() = default;
	virtual Real Psi_V(Real V) const = 0;
	virtual Real B(Real Psi, Real) const = 0;
	virtual Real R(Real Psi, Real) const = 0;
	virtual Real R_V(Real V, Real s) const { return R(Psi_V(V), s); }
	virtual Real dRdV(Real V, Real) const = 0;
	virtual Real VPrime(Real V) const = 0;
	virtual Real MirrorRatio(Real V, Real) const = 0;
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

	StraightMagneticField(double L_z, double B_z, double Rm, double Vmin, double m) : L_z(L_z), B_z(B_z), Rm(Rm), Rmin(R_V(Vmin)), m(m) {};

	template <typename T>
	T Bz_R(T R) { return B_z - m * (R - Rmin); }

	template <typename T>
	T B(T V) { return Bz_R(R_V(V)); }

	template <typename T>
	T Psi(T R)
	{
		return R * R * Bz_R(R) / 2.0;
	}
	template <typename T>
	T Psi_V(T V)
	{
		return B(V) * V / (2 * pi * L_z);
	}
	template <typename T>
	T VPrime(T V)
	{
		return 2 * pi * L_z / B(V);
	}
	Real R(Real Psi) const
	{
		return sqrt(2 * Psi / B_z);
	}
	Real R(Real Psi, Real) const override { return R(Psi); };

	Real VPrime(Real V) const override
	{
		return 2 * pi * L_z / B_z;
	}

	template <typename T>
	T R_V(T V) const
	{
		return sqrt(V / (pi * L_z));
	}
	Real R_V(Real V, Real) const override { return R_V(V); }

	template <typename T>
	T dRdV(T V) const
	{
		return 1.0 / (2 * pi * L_z * R_V(V));
	}
	template <typename T>
	T MirrorRatio(T V) const
	{
		return Rm * B(V) / B_z;
	}
	Real MirrorRatio(Real V, Real) const override { return MirrorRatio(V); }
	Real dRdV(Real V, Real) const override { return dRdV(V); }

	template <typename T>
	T L_V(T)
	{
		return L_z;
	}

	void setRmin(double R) { Rmin = R; };
	void setm(double min) { m = min; };

	void setRmin(double R) { Rmin = R; };
	void setm(double min) { m = min; };

private:
	double L_z = 0.6;
	double B_z = 0.3;
	double Rm = 10.0;

	double Rmin = 0.0;
	double m = 0.0;
};

class CylindricalMagneticField
{
public:
	CylindricalMagneticField(const std::string &file);
	~CylindricalMagneticField() = default;

	double B(double V);
	autodiff::dual B(autodiff::dual V)
	{
		autodiff::dual Bz = B(V.val);
		if (V.grad != 0)
			Bz.grad = B_spline->prime(V.val);
		return Bz;
	}

	double Psi_V(double V);
	autodiff::dual Psi_V(autodiff::dual V)
	{
		autodiff::dual Psi = B(V.val);
		if (V.grad != 0)
			Psi.grad = Psi_spline->prime(V.val);
		return Psi;
	}

	double VPrime(double V);
	autodiff::dual VPrime(autodiff::dual V)
	{
		autodiff::dual Vp = VPrime(V.val);
		if (V.grad != 0)
			Vp.grad = Vp_spline->prime(V.val);
		return Vp;
	};

	double R_V(double V);
	autodiff::dual R_V(autodiff::dual V)
	{
		autodiff::dual R = R_V(V.val);
		if (V.grad != 0.0)
			R.grad = dRdV(V.val);
		return R;
	};
	autodiff::dual2nd R_V(autodiff::dual2nd V)
	{
		autodiff::dual2nd R = R_V(V.val);
		if (V.grad != 0.0)
			R.grad.val = dRdV(V.val.val);
		return R;
	};

	double dRdV(double V);
	autodiff::dual dRdV(autodiff::dual V)
	{
		autodiff::dual drdv = MirrorRatio(V.val);
		if (V.grad != 0)
			drdv.grad = dRdV_spline->prime(V.val);
		return drdv;
	}

	double MirrorRatio(double V);
	autodiff::dual MirrorRatio(autodiff::dual V)
	{
		autodiff::dual Rm = MirrorRatio(V.val);
		if (V.grad != 0)
			Rm.grad = Rm_spline->prime(V.val);
		return Rm;
	};

	void CheckBoundaries(double VL, double VR);

private:
	using spline = boost::math::barycentric_rational<double>;
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
