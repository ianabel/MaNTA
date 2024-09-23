#ifndef MAGNETICFIELDS_HPP
#define MAGNETICFIELDS_HPP

#include <numbers>
#include <cmath>
#include <netcdf>
#include <string>
#include <vector>
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
#include <autodiff/forward/dual.hpp>
using std::numbers::pi;
using spline = boost::math::interpolators::cardinal_cubic_b_spline<double>;
// Magnetic field
// All of these values are returned in SI units
// (we may later introduce a B0 & R0, but these are 1T & 1m for now)
class StraightMagneticField
{
public:
	StraightMagneticField() = default;
	StraightMagneticField(double L_z, double B_z, double Rm) : L_z(L_z), B_z(B_z), Rm(Rm) {};
	template <typename T>
	T Bz_R(T R) { return B_z; }
	template <typename T>
	T V(T Psi)
	{
		return 2 * pi * Psi * L_z / B_z;
	}
	template <typename T>
	T Psi(T R)
	{
		return R * R * B_z / 2.0;
	}
	template <typename T>
	T Psi_V(T V)
	{
		return B_z * V / (2 * pi * L_z);
	}
	template <typename T>
	T VPrime(T V)
	{
		return 2 * pi * L_z / B_z;
	}
	template <typename T>
	T R(T Psi)
	{
		return sqrt(2 * Psi / B_z);
	}
	template <typename T>
	T R_V(T V)
	{
		return sqrt(V / (pi * L_z));
	}
	template <typename T>
	T dRdV(T V)
	{
		return 1.0 / (2 * pi * R_V(V));
	}
	template <typename T>
	double MirrorRatio(T)
	{
		return Rm;
	}

private:
	double L_z = 0.6;
	double B_z = 0.3;
	double Rm = 10.0;
};

class CylindricalMagneticField
{
public:
	CylindricalMagneticField(const std::string &file);
	~CylindricalMagneticField() = default;

	double Bz_R(double R);
	double Bz_R(autodiff::dual R) { return Bz_R(R.val); };

	double V(double Psi);
	double Psi(double R);
	double Psi_V(double V);
	double VPrime(double V);
	double VPrime(autodiff::dual V) { return VPrime(V.val); };

	double R(double Psi);
	double R_V(double V);
	autodiff::dual R_V(autodiff::dual V)
	{
		autodiff::dual R = R_V(V.val);
		if (V.grad != 0.0)
			R.grad += V.grad * dRdV(V.val);
		return R;
	};

	double dRdV(double V);
	autodiff::dual dRdV(autodiff::dual V);
	double MirrorRatio(double V);
	double MirrorRatio(autodiff::dual V) { return MirrorRatio(V.val); };
	void CheckBoundaries(double VL, double VR);

private:
	double L_z = 1.0;
	double h;

	double R_root_solver(double Psi);

	std::string filename;
	std::vector<double> gridpoints;
	netCDF::NcFile data_file;
	unsigned int nPoints;
	netCDF::NcDim R_dim;
	std::vector<double> R_var;
	std::vector<double> Bz_var;
	std::vector<double> Psi_var;
	std::vector<double> Rm_var;

	std::unique_ptr<spline> B_spline;
	std::unique_ptr<spline> Psi_spline;
	std::unique_ptr<spline> Rm_spline;
	std::unique_ptr<spline> R_Psi_spline;
};

#endif // MAGNETICFIELDS_HPP
