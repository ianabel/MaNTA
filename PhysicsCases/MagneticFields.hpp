#ifndef MAGNETICFIELDS_HPP
#define MAGNETICFIELDS_HPP

#include <numbers>
#include <cmath>
#include <netcdf>
#include <string>
#include <vector>
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <autodiff/forward/dual.hpp>
using std::numbers::pi;
// Magnetic field
// All of these values are returned in SI units
// (we may later introduce a B0 & R0, but these are 1T & 1m for now)
class StraightMagneticField
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
	template <typename T>
	T R_V(T V)
	{
		return sqrt(V / (pi * L_z));
	}
	template <typename T>
	T dRdV(T V)
	{
		return 1.0 / (2 * pi * L_z * R_V(V));
	}
	template <typename T>
	T MirrorRatio(T V)
	{
		return Rm * B(V) / B_z;
	}
	template <typename T>
	T L_V(T)
	{
		return L_z;
	}

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

#endif // MAGNETICFIELDS_HPP
