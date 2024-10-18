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
using Real = autodiff::dual;
// Magnetic field
// All of these values are returned in SI units
// (we may later introduce a B0 & R0, but these are 1T & 1m for now)

class MagneticField
{
public:
	MagneticField() = default;
	virtual Real Psi(Real V) const = 0;
	virtual Real B(Real Psi, Real s) const = 0;
	virtual Real R(Real Psi, Real s) const = 0;
	Real R_V(Real V, Real s) const { return R(Psi(V), s); }
	virtual Real dRdV(Real V, Real s) const = 0;
	virtual Real VPrime(Real V) const = 0;
	virtual Real MirrorRatio(Real V, Real s) const = 0;

	template <typename F>
	Real FluxSurfaceAverage(const F &f, Real V) { return f(V, 0.0); };

private:
	virtual Real LeftEndpoint(Real Psi) const = 0;
	virtual Real RightEndpoint(Real Psi) const = 0;
};



class StraightMagneticField : public MagneticField
{
public:
	StraightMagneticField() = default;
	StraightMagneticField(double L_z, double B_z, double Rm) : L_z(L_z), B_z(B_z), Rm(Rm) {};

	template <typename T>
	T Bz_R(T R) const { return B_z; }

	Real B(Real Psi, Real) const override { return Bz_R(R(Psi)); };

	Real Psi(Real V) const override
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
	double L_z = 0.6;
	double B_z = 0.3;
	double Rm = 10.0;
	Real LeftEndpoint(Real) const override { return -L_z / 2.0; }
	Real RightEndpoint(Real) const override { return L_z / 2.0; }
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

// Function for creating an instance of a magnetic field
// Allow for constructors that take a different number of arguments
template <typename T, typename... Args>
static std::shared_ptr<MagneticField> createMagneticField(Args &&...args) { return std::make_shared<T>(std::forward<Args>(args)...); };

#endif // MAGNETICFIELDS_HPP
