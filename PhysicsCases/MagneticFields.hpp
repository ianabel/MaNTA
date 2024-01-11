#ifndef MAGNETICFIELDS_HPP
#define MAGNETICFIELDS_HPP

#include <numbers>
#include <cmath>
using std::numbers::pi;

// Magnetic field
class StraightMagneticField
{
	public:
		double Bz_R( double R ) { return B_z; };
		double V( double Psi ) { 
			return 2 * pi * Psi * L_z / B_z;
		};
		double Psi( double R ) {
			return R*R*B_z/2.0;
		};
		double Psi_V( double V ) {
			return B_z * V / ( 2 * pi * L_z );
		};
		double VPrime( double V ) {
			return 2*pi*L_z/B_z;
		};
		double R( double Psi ) {
			return sqrt( 2*Psi/B_z );
		};
		double R_V( double V ) {
			return sqrt( V / ( pi * L_z ) );
		};
	private:
		double L_z = 1.0;
		double B_z = 1.0;
};


#endif // MAGNETICFIELDS_HPP
