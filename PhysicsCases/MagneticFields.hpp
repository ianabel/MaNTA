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
	double Bz_R(double R) { return B_z; };
	double V(double Psi)
	{
		return 2 * pi * Psi * L_z / B_z;
	};
	double Psi(double R)
	{
		return R * R * B_z / 2.0;
	};
	double Psi_V(double V)
	{
		return B_z * V / (2 * pi * L_z);
	};
	double VPrime(double V)
	{
		return 2 * pi * L_z / B_z;
	};
	double R(double Psi)
	{
		return sqrt(2 * Psi / B_z);
	};
	double R_V(double V)
	{
		return sqrt(V / (pi * L_z));
	};
	double dRdV(double V)
	{
		return 1.0 / (2 * pi * R_V(V));
	}
	double MirrorRatio(double)
	{
		return 3.3;
	};
	autodiff::dual Bz_R(autodiff::dual R) { return B_z; };
	autodiff::dual V(autodiff::dual Psi)
	{
		return 2 * pi * Psi * L_z / B_z;
	};
	autodiff::dual Psi(autodiff::dual R)
	{
		return R * R * B_z / 2.0;
	};
	autodiff::dual Psi_V(autodiff::dual V)
	{
		return B_z * V / (2 * pi * L_z);
	};
	autodiff::dual VPrime(autodiff::dual V)
	{
		return 2 * pi * L_z / B_z;
	};
	autodiff::dual R(autodiff::dual Psi)
	{
		return sqrt(2 * Psi / B_z);
	};
	autodiff::dual R_V(autodiff::dual V)
	{
		return sqrt(V / (pi * L_z));
	};
	autodiff::dual dRdV(autodiff::dual V)
	{
		return 1.0 / (2 * pi * R_V(V));
	}
	double MirrorRatio(autodiff::dual)
	{
		return 3.3;
	};

private:
	double L_z = 1.0;
	double B_z = 1.0;
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
