#ifndef MAGNETICFIELDS_HPP
#define MAGNETICFIELDS_HPP

#include <numbers>
#include <cmath>
#include <netcdf>
#include <string>
#include <vector>
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
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
	double MirrorRatio(double)
	{
		return 3.0;
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
	double V(double Psi);
	double Psi(double R);
	double Psi_V(double V);
	double VPrime(double V);
	double R(double Psi);
	double R_V(double V);
	double dRdV(double V);
	double MirrorRatio(double V);
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
