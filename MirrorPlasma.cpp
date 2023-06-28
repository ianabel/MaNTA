#include "MirrorPlasma.hpp"
#include "Constants.hpp"
#include "Species.hpp"

#include <boost/math/tools/roots.hpp>
#include <cmath>
#include <iostream>
#include <functional>

MirrorPlasma::MirrorPlasma()
{
	ionSpecies.type = Species::Type::Ion;
	ionSpecies.Charge = 1.0;
	ionSpecies.Mass = 2.5;
	ionSpecies.Name = "ion";

	mirrorRatio = 10.0;
	Zeff = 1.5;
	plasmaLength = 0.5;
	parallelFudgeFactor = 1.0;
	density = 1.6e19;
}

double MirrorPlasma::ParallelElectronPastukhovLossRate(double chi_e, double Te, double R) const
{
	// For consistency, the integral in Pastukhov's paper is 1.0, as the
	// entire theory is an expansion in M^2 >> 1
	double R_m = mirrorRatio;
	double tau_ee = ElectronCollisionTime(Te, R);
	double sigma = 1.0 + Zeff; // Include collisions with ions and impurities as well as self-collisions
	double lossRate = ( M_2_SQRTPI / tau_ee ) * sigma * electronDensity(R) * ( 1.0 / ::log( R_m * sigma ) ) * ( ::exp( - chi_e ) / chi_e );

	// To prevent false solutions, apply strong losses if the Mach number drops
	if ( chi_e < 1.0 ) {
		double baseLossRate = electronDensity(R) * ( SoundSpeed(Te) / plasmaLength );
		double smoothing = Transition( chi_e, 0.5, 1.0 );
		return smoothing*baseLossRate + ( 1-smoothing )*lossRate;
	}

	return lossRate*parallelFudgeFactor;
}

double MirrorPlasma::ParallelElectronParticleLoss(double Te, double Ti, double omega, double R) const
{
		if ( collisional ) {
		// Particle loss from the mirror throat
		// given by the density at the throat and the sounds transit time
		double mirrorThroatDensity = ionDensity(R) * ::exp( CentrifugalPotential(Te, Ti, omega, R) );
#ifdef DEBUG
		std::cout << "Electron parallel particle loss is " << SoundSpeed() * MirrorThroatDensity << "\t";
		std::cout << "Collisionless parallel losse would have been " << ParallelElectronPastukhovLossRate( -AmbipolarPhi(), Te, R );
#endif
		return SoundSpeed(Te) * mirrorThroatDensity;
	}
	double Chi_e = -AmbipolarPhi(Te, Ti, omega, R); // Ignore small electron mass correction
	return ParallelElectronPastukhovLossRate( Chi_e, Te, R );
}

double MirrorPlasma::ParallelElectronHeatLoss(double Te, double Ti, double omega, double R) const
{
		if ( collisional )
	{
		double kappa_parallel = 3.16 * electronDensity(R) * Te * ElectronCollisionTime(Te, R) / ( electronMass  );
		double L_parallel = plasmaLength;
#ifdef DEBUG
		std::cout << "Electron parallel heat flux is " << kappa_parallel / ( L_parallel * L_parallel ) << std::endl;
		std::cout << "Collisionless parallel heat flux would have been "
		          << ParallelElectronPastukhovLossRate( -AmbipolarPhi(), Te, R ) * (  1.0 - AmbipolarPhi() );
#endif
		return kappa_parallel * Te / ( L_parallel * L_parallel );
	}

	// Energy loss per particle is ~ e Phi + T_e
	// AmbipolarPhi = e Phi / T_e so loss is T_e * ( AmbipolarPhi + 1)
	double chi_e = -AmbipolarPhi(Te, Ti, omega, R); // Ignore small electron mass correction
	// Particle energy is roughly T_e + Chi_e (thermal + potential)
	return ParallelElectronPastukhovLossRate( chi_e, Te, R ) * ( Te ) * ( chi_e + 1.0 );

}

double MirrorPlasma::Chi_i(double phi, double Te, double Ti, double omega, double R) const
{
	double M = machNumber(R, Te, omega);
	double phi1 = ionSpecies.Charge * phi * ( Te/Ti );
	double phi2 = 0.5 * M * M * ( 1.0 - 1.0/mirrorRatio ) * ( Te / Ti );
	return ionSpecies.Charge * phi * ( Te/Ti ) + 0.5 * M * M * ( 1.0 - 1.0/mirrorRatio ) * ( Te / Ti );
}

double MirrorPlasma::Chi_i(double Te, double Ti, double omega, double R) const
{
	return Chi_i( AmbipolarPhi(Te, Ti, omega, R), Te, Ti, omega, R );
}

double MirrorPlasma::ParallelIonPastukhovLossRate(double chi_i, double Te, double Ti, double R) const
{
	// For consistency, the integral in Pastukhov's paper is 1.0, as the
	// entire theory is an expansion in M^2 >> 1
	double R_m = mirrorRatio;
	double tau_ii = IonCollisionTime(Ti, R);
	double Sigma = 1.0;
	double LossRate = ( M_2_SQRTPI / tau_ii ) * Sigma * ionDensity(R) * ( 1.0 / ::log( R_m * Sigma ) ) * ( ::exp( - chi_i ) / chi_i );


	// To prevent false solutions, apply strong losses if the Mach number drops
	if ( chi_i < 1.0 ) {
		// std::cerr << "Warning: Chi_i dropped below one" << std::endl;
		double BaseLossRate = ionDensity(R) * ( SoundSpeed(Te) / plasmaLength );
		double smoothing = Transition( chi_i, .5, 1.0 );
		return smoothing*BaseLossRate + ( 1-smoothing )*LossRate;
	}

	return LossRate*parallelFudgeFactor;
}

double MirrorPlasma::ParallelIonParticleLoss(double Te, double Ti, double omega, double R) const
{
	if ( collisional ) {
		// Particle loss at the sound speed from the mirror throat
		double MirrorThroatDensity = ionDensity(R) * ::exp( CentrifugalPotential(Te, Te, omega, R) );
		return SoundSpeed(Te) * MirrorThroatDensity;
	}

	// Electrostatic energy + centrifugal potential energy
	return ParallelIonPastukhovLossRate( Chi_i(Te, Ti, omega, R), Te, Ti, R );

}

double MirrorPlasma::ParallelIonHeatLoss(double Te, double Ti, double omega, double R) const
{
		if ( collisional ) {
		// Collisional parallel heat transport
		double ionMass = ionSpecies.Mass * protonMass;
		double kappa_parallel = 3.9 * ionDensity(R) * Ti * IonCollisionTime(Ti, R) / ( ionMass );
		double L_parallel = plasmaLength;
		return kappa_parallel * Ti / ( L_parallel * L_parallel );
	}

	// Energy loss per particle is ~ Chi_i + T_i
	double chi_i = Chi_i(Te, Ti, omega, R);
	double loss = ParallelIonPastukhovLossRate( chi_i, Te, Ti, R );
	double heatloss = loss* ( Ti ) * ( ::fabs( chi_i )  + 1.0 );
	return heatloss;
}

double MirrorPlasma::ElectronCollisionTime( double Te, double R) const
{
	double PiThreeHalves = ::pow( M_PI, 1.5 ); // pi^(3/2)
	double TeThreeHalves = ::pow( Te, 1.5 );
	double ZIon = ionSpecies.Charge;
	return 12 * ::sqrt( electronMass ) * PiThreeHalves * TeThreeHalves * vacuumPermittivity * vacuumPermittivity / ( ::sqrt(2) * ionDensity(R) * ::pow( ZIon, 2 ) * ::pow( e_charge, 4 ) * LogLambdaElectron(Te) );
 	//if(Te>0)return 3.44e11*(1.0/::pow(electronDensity(R),3.0/2.0))*(::pow(J_eV(Te),3.0/2.0))*(1.0/LogLambdaElectron(Te));
	//else return 3.44e11*(1.0/::pow(electronDensity(R),5.0/2.0))*(::pow(1.0,3.0/2.0))*(1.0/LogLambdaElectron(Te)); //if we have a negative temp just treat it as 1eV

}

double MirrorPlasma::delTe_ElectronCollisionTime(double Te, double R) const
{
	double PiThreeHalves = ::pow( M_PI, 1.5 ); // pi^(3/2)
	double TeOneHalf = ::pow( Te, 0.5 );
	double ZIon = ionSpecies.Charge;
	return (3.0/2.0)*12 * ::sqrt( electronMass ) * PiThreeHalves * TeOneHalf * vacuumPermittivity * vacuumPermittivity / ( ::sqrt(2) * ionDensity(R) * ::pow( ZIon, 2 ) * ::pow( e_charge, 4 ) * LogLambdaElectron(Te) );


	//if(Te>0)return (3.0/2.0)*3.44e11*(1.0/::pow(electronDensity(R),3.0/2.0))*(::pow(J_eV(Te),1.0/2.0))*(1.0/LogLambdaElectron(Te));
	//else return (3.0/2.0)*3.44e11*(1.0/::pow(electronDensity(R),3.0/2.0))*(::pow(1.0,1.0/2.0))*(1.0/LogLambdaElectron(R));
}

/*
double MirrorPlasma::IonCollisionTime(double Ti) const
{
	double piThreeHalves = ::pow( M_PI, 1.5 ); // pi^(3/2)
	double TiThreeHalves = ::pow( Ti * referenceTemperature, 1.5 );
	double ZIon = ionSpecies.Charge;
	return 12 * ::sqrt( ionSpecies.Mass * protonMass ) * piThreeHalves * TiThreeHalves * vacuumPermittivity * vacuumPermittivity / ( ::sqrt(2) * ionDensity(R) * referenceDensity * ::pow( ZIon * e_charge, 4 ) * LogLambdaIon(Ti) );

}
*/

double MirrorPlasma::IonCollisionTime(double Ti, double R) const
{
	double PiThreeHalves = ::pow( M_PI, 1.5 ); // pi^(3/2)
	double TiThreeHalves = ::pow( Ti, 1.5 );
	double ZIon = ionSpecies.Charge;
	double a_ = 12 * ::sqrt( ionSpecies.Mass * protonMass ) * PiThreeHalves * TiThreeHalves * vacuumPermittivity * vacuumPermittivity;
	double b_ = ( ::sqrt(2) * ionDensity(R) * ::pow( ZIon * e_charge, 4 ) * LogLambdaIon(Ti) );
	return a_ / b_;

	//if(Ti>0)return 3.44e11*(1.0/::pow(ionDensity(R),3.0/2.0))*(::pow(J_eV(Ti),3.0/2.0))*(1.0/LogLambdaIon(Ti))*(::sqrt(ionMass/electronMass));
	//else return 3.44e11*(1.0/::pow(ionDensity(R),5.0/2.0))*(::pow(1.0,3.0/2.0))*(1.0/LogLambdaIon(Ti))*(::sqrt(ionMass/electronMass)); //if we have a negative temp just treat it as 1eV

}

double MirrorPlasma::delTi_IonCollisionTime(double Ti, double R) const
{
	double PiThreeHalves = ::pow( M_PI, 1.5 ); // pi^(3/2)
	double TiOneHalf = ::pow( Ti, 1.5 );
	double ZIon = ionSpecies.Charge;
	return (3.0/2.0) * 12 * ::sqrt( ionSpecies.Mass * protonMass ) * PiThreeHalves * TiOneHalf * vacuumPermittivity * vacuumPermittivity / ( ::sqrt(2) * ionDensity(R) * ::pow( ZIon * e_charge, 4 ) * LogLambdaIon(Ti) );


	//if(Ti>0)return (3.0/2.0)*3.44e11*(1.0/::pow(ionDensity(R),3.0/2.0))*(::pow(J_eV(Ti),1.0/2.0))*(1.0/LogLambdaIon(R))*(::sqrt(ionMass/electronMass));
	//else return (3.0/2.0)*3.44e11*(1.0/::pow(ionDensity(R),5.0/2.0))*(::pow(1.0,1.0/2.0))*(1.0/LogLambdaIon(R))*(::sqrt(ionMass/electronMass));
	//return 4.5e9/lambda()*::sqrt(mi/(2*mp))*::pow(Pi,1/2)/::pow(n,5/2);
}

double MirrorPlasma::CollisionalTemperatureEquilibrationTime(double Te, double R) const
{
	return ElectronCollisionTime(Te, R)/( (3./ionSpecies.Mass)*(electronMass/protonMass) );
}

double MirrorPlasma::CentrifugalPotential(double Te, double Ti, double omega, double R) const
{
	double tau = Ti / Te;
	return -( 0.5/tau ) * ( 1.0 - 1.0 / mirrorRatio ) * machNumber(R, Te, omega) * machNumber(R, Te, omega) / ( ionSpecies.Charge / tau + 1 );

}

double MirrorPlasma::AmbipolarPhi(double Te, double Ti, double omega, double R) const
{
	double ambipolarPhi = CentrifugalPotential(Te, Ti, omega, R);

	if ( collisional )
		return ambipolarPhi;

	if ( useAmbipolarPhi ) {

		// Add correction.
		double sigma = 1.0 + Zeff;
		double R_m = mirrorRatio;
		double Correction = ::log( (  ElectronCollisionTime(Te, R) / IonCollisionTime(Ti, R) ) * ( ::log( R_m*sigma ) / ( sigma * ::log( R_m ) ) ) );
		ambipolarPhi += Correction/2.0;

		// This gives us a first-order guess for the Ambipolar potential. Now we solve j_|| = 0 to get the better answer.
		//
		// compute current density in units of e/s/m^2
		auto ParallelCurrent = [ & ]( double Phi ) {
			double Chi_e = -Phi; // Ignore small electron mass correction

			// If Alphas are included, they correspond to a (small) charge flow
			//??To Do: alpha heatig stuff
			if ( false ) // includeAlphaHeating )
			{
				/*
				double AlphaLossRate =  AlphaProductionRate() * PromptAlphaLossFraction();
				return 2.0*AlphaLossRate + ParallelIonPastukhovLossRate( Chi_i( Phi ) )*IonSpecies.Charge - ParallelElectronPastukhovLossRate( Chi_e );
				*/
			}
			else
			{
				return ParallelIonPastukhovLossRate( Chi_i( Phi, Te, Ti, omega, R ), Te, Ti, R )*ionSpecies.Charge - ParallelElectronPastukhovLossRate( Chi_e, Te, R );
			}
		};


		double thresholdCurrent = 1e-6 * electronDensity(R);

		boost::uintmax_t iters = 1000;
		boost::math::tools::eps_tolerance<double> tol( 25 );

		double guess;
		if ( storedPhi != 0 )
			guess = storedPhi;
		else
			guess = ambipolarPhi;


		if ( ::fabs( ParallelCurrent( guess ) ) <  thresholdCurrent )
		{
			return guess;
		}


		double eps = 0.1;
		double l_bracket = std::min( guess*( 1.0 - eps ), guess*( 1.0 + eps ) );
		double u_bracket = std::max( guess*( 1.0 - eps ), guess*( 1.0 + eps ) );
		if ( ParallelCurrent( l_bracket )*ParallelCurrent( u_bracket ) < 0 ) {

			auto [ Phi_l, Phi_u ] = boost::math::tools::toms748_solve( ParallelCurrent, l_bracket, u_bracket, tol, iters );
			ambipolarPhi = ( Phi_l + Phi_u )/2.0;

		} else {

			auto [ Phi_l, Phi_u ] = boost::math::tools::bracket_and_solve_root( ParallelCurrent, guess, 1.2, false, tol, iters );
			ambipolarPhi = ( Phi_l + Phi_u )/2.0;

			if ( ::fabs( Phi_u - Phi_l )/2.0 > ::fabs( 0.05*ambipolarPhi ) ) {
				std::cerr << "Unable to find root of j_|| = 0, using approximation" << std::endl;
				return CentrifugalPotential(Te, Ti, omega, R) + Correction/2.0;
			}
		}
	}

	return ambipolarPhi;
}

double MirrorPlasma::LogLambdaElectron(double Te) const
{
	// Convert to NRL Formulary Units
	/*
	double neNRL = electronDensity(R) * 1e14; // We use 10^20 / m^3 = 10^14 / cm^3
	double TeNRL = Te * 1000; // We normalize to 1 keV they use 1 ev
	// For sensible values of the coulomb logarithm, the lambda_ee value in the NRL formulary
	// can be simplified to be the same as the middle lambda_ei value.
	return 24.0 - 0.5*::log( neNRL ) + ::log( TeNRL );
	*/

	//??To Do: The actual implementation of this
	return 22.5;

}

double MirrorPlasma::LogLambdaIon(double Ti) const
{
	/*
	double TiNRL = Ti * 1000;
	double niNRL = ionDensity(R) * 1e14;
	double ZIon = ionSpecies.Charge;
	return 23.0 - 0.5 * ::log( niNRL ) - 1.5 * ::log( ZIon * ZIon / TiNRL );
	*/

	//??To Do: Actually implement this
	return 22.5;
}

double MirrorPlasma::delTi_ParallelIonParticleLoss(double Te, double Ti, double omega, double R)
{
	double chi_i = Chi_i(Te, Ti, omega, R);
	return ParallelIonParticleLoss(Te, Ti, omega, R)*(delTi_Chi_i(Te, Ti, omega, R) - (1/IonCollisionTime(Ti, R))*delTi_IonCollisionTime(Ti, R) - (1/chi_i)*delTi_Chi_i(Te, Ti, omega, R));
}

double MirrorPlasma::delTi_ParallelIonHeatLoss(double Te, double Ti, double omega, double R)
{
	double chi_i = Chi_i(Te, Ti, omega, R);
	return delTi_ParallelIonParticleLoss(Te, Ti, omega, R)*Ti*(chi_i + 1) + ParallelIonParticleLoss(Te, Ti, omega, R)*(chi_i + 1.0) + ParallelIonParticleLoss(Te, Ti, omega, R)*Ti*delTi_Chi_i(Te, Ti, omega, R);
}

double MirrorPlasma::delTi_Chi_i(double Te, double Ti, double omega, double R)
{
	double chi_i = Chi_i(Te, Ti, omega, R);
	return -chi_i/Ti;
}

double MirrorPlasma::delTe_ParallelIonParticleLoss(double Te, double Ti, double omega, double R)
{
	double chi_i = Chi_i(Te, Ti, omega, R);
	return ParallelIonParticleLoss(Te, Ti, omega, R)*(delTe_Chi_i(Te, Ti, omega, R) - (1/chi_i)*delTe_Chi_i(Te, Ti, omega, R));
}

double MirrorPlasma::delTe_ParallelIonHeatLoss(double Te, double Ti, double omega, double R)
{
	double chi_i = Chi_i(Te, Ti, omega, R);
	return delTe_ParallelIonParticleLoss(Te, Ti, omega, R)*Ti*(chi_i + 1) + ParallelIonParticleLoss(Te, Ti, omega, R)*Ti*delTe_Chi_i(Te, Ti, omega, R);
}

double MirrorPlasma::delTe_Chi_i(double Te, double Ti, double omega, double R)
{
	double chi_i = Chi_i(Te, Ti, omega, R);
	double R_m = mirrorRatio;
	return chi_i/Te + delTe_MachNumber(Ti,Te,omega,R)*machNumber(R, Te, omega)*(1-1/R_m)*Te/Ti;
}

double MirrorPlasma::delTe_ParallelElectronParticleLoss(double Te, double Ti, double omega, double R)
{
	return ParallelElectronParticleLoss(Te,Ti,omega,R)*(-1/ElectronCollisionTime(Te,R))*delTe_ElectronCollisionTime(Te, R);
}

double MirrorPlasma::delTe_ParallelElectronHeatLoss(double Te, double Ti, double omega, double R)
{
	double chi_e = -AmbipolarPhi(Te, Ti, omega, R); 
	return delTe_ParallelElectronParticleLoss(Te,Ti,omega,R)*(Te + chi_e) + ParallelElectronParticleLoss(Te, Ti, omega, R);
}

double MirrorPlasma::delTe_MachNumber(double Te, double Ti, double omega, double R)
{
	return -0.5*machNumber(R, Te, omega)/Te;
}

double MirrorPlasma::delOmega_ParallelIonParticleLoss(double Te, double Ti, double omega, double R)
{
	double chi_i = Chi_i(Te, Ti, omega, R);
	return ParallelIonParticleLoss(Te, Ti, omega, R)*(delOmega_Chi_i(Te, Ti, omega, R) - (1/chi_i)*delOmega_Chi_i(Te, Ti, omega, R));
}

double MirrorPlasma::delOmega_ParallelIonHeatLoss(double Te, double Ti, double omega, double R)
{
	double chi_i = Chi_i(Te, Ti, omega, R);
	return delOmega_ParallelIonParticleLoss(Te, Ti, omega, R)*Ti*(chi_i + 1) + ParallelIonParticleLoss(Te, Ti, omega, R)*Ti*delOmega_Chi_i(Te, Ti, omega, R);
}

double MirrorPlasma::delOmega_Chi_i(double Te, double Ti, double omega, double R)
{
	double R_m = mirrorRatio;
	return delOmega_MachNumber(Te, Ti, omega, R)*machNumber(R, Te, omega)*(1.0-1.0/R_m)*(Te/Ti);
}

double MirrorPlasma::delOmega_MachNumber(double Te, double Ti, double omega, double R)
{
	return machNumber(R, Te, omega)/omega;
}

// Transition Function that is smooth,
// equal to 1.0 for x < L and 0.0 for x > U
// and takes values in (0.0,1.0) for x in (L,U)
double MirrorPlasma::Transition(double x, double L, double U) const
{
	if ( L > U ) throw std::invalid_argument( "When calling Transition(x, L, U), L&U must satisfy L < U" );
	if ( x <= L ) return 1.0;
	if ( x >= U ) return 0.0;

	double y = ( x - L )/( U - L );
	double arg = 1.0 - 1.0 / ( 1.0 - y*y );
	return ::exp( arg );
}

double MirrorPlasma::machNumber(double R, double Te, double omega) const
{
	//std::cerr << omega*R/::sqrt(ionSpecies.Charge*Te/ionMass) << std::endl;
	double M = omega*R/::sqrt(ionSpecies.Charge*Te/ionMass);
	return M;
}

double MirrorPlasma::electronDensity(double R) const
{
	return density;
}

double MirrorPlasma::ionDensity(double R) const
{
	return density;
}
