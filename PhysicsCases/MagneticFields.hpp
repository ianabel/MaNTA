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
using spline = boost::math::interpolators::cardinal_cubic_b_spline<double>;
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

	template <typename T>
	T Bz_R(T R) const { return B_z; }

	Real B(Real Psi, Real) const override { return Bz_R(R(Psi)); };

	Real Psi_V(Real V) const override
	{
		return B_z * V / (2 * pi * L_z);
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
		return 1.0 / (2 * pi * R_V(V));
	}
	Real dRdV(Real V, Real) const override { return dRdV(V); }

	template <typename T>
	T MirrorRatio(T) const
	{
		return Rm;
	}
	Real MirrorRatio(Real V, Real) const override { return MirrorRatio(V); }

private:
	const double L_z = 0.6;
	const double B_z = 0.3;
	const double Rm = 10.0;
	Real LeftEndpoint(Real) const override { return -L_z / 2.0; }
	Real RightEndpoint(Real) const override { return L_z / 2.0; }
};

// Define an analytic B_z(z) that sort of looks like a mirror field
class CurvedMagneticField : public MagneticField
{
public:
	CurvedMagneticField() = default;
	CurvedMagneticField(double L_z, double Bmid, double Rm) : L_z(L_z), Bmid(Bmid), Rm(Rm) {}
	Real Psi_V(Real V) const override;
	Real B(Real Psi, Real z) const override;
	Real R(Real Psi, Real z) const override;
	Real R_V(Real V, Real z) const override { return R(Psi_V(V), z); }
	Real dRdV(Real V, Real z) const override;
	Real VPrime(Real V) const override;
	Real MirrorRatio(Real V, Real z) const override;

private:
	const double L_z = 1.0;
	const double Bmid = 0.34;
	const double Rm = 10.0;

	const double A = (Rm - 1) / (1 + Rm);
	const double gamma = 2 * M_PI / L_z;
	const double B0 = Bmid * (1 + A);

	Real LeftEndpoint(Real Psi) const override;
	Real RightEndpoint(Real Psi) const override;

	Real B_s(Real Psi, Real s) const override { return B_z(s); }
	Real B_z(Real z) const;
	Real B_r(Real Psi, Real z) const;
};

// Function for creating an instance of a magnetic field
// Allow for constructors that take a different number of arguments
template <typename T, typename... Args>
static std::shared_ptr<MagneticField> createMagneticField(Args &&...args)
{
	return std::make_shared<T>(std::forward<Args>(args)...);
};

#endif // MAGNETICFIELDS_HPP
