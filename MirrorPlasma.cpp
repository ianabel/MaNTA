#include "MirrorPlasma.hpp"
#include "Constants.hpp"
#include <boost/math/tools/roots.hpp>
#include <cmath>
#include <iostream>
#include <functional>

double MirrorPlasma::ParallelElectronPastukhovLossRate(double chi_e, double Te) const
{
	// For consistency, the integral in Pastukhov's paper is 1.0, as the
	// entire theory is an expansion in M^2 >> 1
	double R = mirrorRatio;
	double tau_ee = ElectronCollisionTime(Te);
	double sigma = 1.0 + Zeff; // Include collisions with ions and impurities as well as self-collisions
	double lossRate = ( M_2_SQRTPI / tau_ee ) * sigma * electronDensity * referenceDensity * ( 1.0 / ::log( R * sigma ) ) * ( ::exp( - chi_e ) / chi_e );

	// To prevent false solutions, apply strong losses if the Mach number drops
	if ( chi_e < 1.0 ) {
		double baseLossRate = electronDensity * referenceDensity * ( SoundSpeed(Te) / plasmaLength );
		double smoothing = Transition( chi_e, 0.5, 1.0 );
		return smoothing*baseLossRate + ( 1-smoothing )*lossRate;
	}

	return lossRate*parallelFudgeFactor;
}

double MirrorPlasma::ParallelElectronParticleLoss(double Te, double Ti) const
{
		if ( collisional ) {
		// Particle loss from the mirror throat
		// given by the density at the throat and the sounds transit time
		double mirrorThroatDensity = ionDensity * referenceDensity * ::exp( CentrifugalPotential(Te, Ti) );
#ifdef DEBUG
		std::cout << "Electron parallel particle loss is " << SoundSpeed() * MirrorThroatDensity << "\t";
		std::cout << "Collisionless parallel losse would have been " << ParallelElectronPastukhovLossRate( -AmbipolarPhi() );
#endif
		return SoundSpeed(Te) * mirrorThroatDensity;
	}
	double Chi_e = -AmbipolarPhi(Te, Ti); // Ignore small electron mass correction
	return ParallelElectronPastukhovLossRate( Chi_e, Te );
}

double MirrorPlasma::ParallelElectronHeatLoss(double Te, double Ti) const
{
		if ( collisional )
	{
		double kappa_parallel = 3.16 * electronDensity * Te * referenceDensity * referenceTemperature * ElectronCollisionTime(Te) / ( electronMass  );
		double L_parallel = plasmaLength;
#ifdef DEBUG
		std::cout << "Electron parallel heat flux is " << kappa_parallel * ElectronTemperature * ReferenceTemperature / ( L_parallel * L_parallel ) << std::endl;
		std::cout << "Collisionless parallel heat flux would have been "
		          << ParallelElectronPastukhovLossRate( -AmbipolarPhi() ) * ( ElectronTemperature * ReferenceTemperature ) * (  1.0 - AmbipolarPhi() );
#endif
		return kappa_parallel * Te * referenceTemperature / ( L_parallel * L_parallel );
	}

	// Energy loss per particle is ~ e Phi + T_e
	// AmbipolarPhi = e Phi / T_e so loss is T_e * ( AmbipolarPhi + 1)
	double chi_e = -AmbipolarPhi(Te, Ti); // Ignore small electron mass correction
	// Particle energy is roughly T_e + Chi_e (thermal + potential)
	return ParallelElectronPastukhovLossRate( chi_e, Te ) * ( Te * referenceTemperature ) * ( chi_e + 1.0 );

}

double MirrorPlasma::Chi_i(double phi, double Te, double Ti) const
{
	return ionSpecies.Charge * phi * ( Te/Ti ) + 0.5 * machNumber * machNumber * ( 1.0 - 1.0/mirrorRatio ) * ( Te / Ti );
}

double MirrorPlasma::Chi_i(double Te, double Ti) const
{
	return Chi_i( AmbipolarPhi(Te, Ti), Te, Ti );
}

double MirrorPlasma::ParallelIonPastukhovLossRate(double chi_i, double Te, double Ti) const
{
	// For consistency, the integral in Pastukhov's paper is 1.0, as the
	// entire theory is an expansion in M^2 >> 1
	double R = mirrorRatio;
	double tau_ii = IonCollisionTime(Ti);
	double Sigma = 1.0;
	double LossRate = ( M_2_SQRTPI / tau_ii ) * Sigma * ionDensity * referenceDensity * ( 1.0 / ::log( R * Sigma ) ) * ( ::exp( - chi_i ) / chi_i );


	// To prevent false solutions, apply strong losses if the Mach number drops
	if ( chi_i < 1.0 ) {
		// std::cerr << "Warning: Chi_i dropped below one" << std::endl;
		double BaseLossRate = ionDensity * referenceDensity * ( SoundSpeed(Te) / plasmaLength );
		double smoothing = Transition( chi_i, .5, 1.0 );
		return smoothing*BaseLossRate + ( 1-smoothing )*LossRate;
	}

	return LossRate*parallelFudgeFactor;
}

double MirrorPlasma::ParallelIonParticleLoss(double Te, double Ti) const
{
	if ( collisional ) {
		// Particle loss at the sound speed from the mirror throat
		double MirrorThroatDensity = ionDensity * referenceDensity * ::exp( CentrifugalPotential(Te, Te) );
		return SoundSpeed(Te) * MirrorThroatDensity;
	}

	// Electrostatic energy + centrifugal potential energy
	return ParallelIonPastukhovLossRate( Chi_i(Te, Ti), Te, Ti );

}

double MirrorPlasma::ParallelIonHeatLoss(double Te, double Ti) const
{
		if ( collisional ) {
		// Collisional parallel heat transport
		double ionMass = ionSpecies.Mass * protonMass;
		double kappa_parallel = 3.9 * ionDensity * Ti * referenceDensity * referenceTemperature * IonCollisionTime(Ti) / ( ionMass );
		double L_parallel = plasmaLength;
		return kappa_parallel * Ti * referenceTemperature / ( L_parallel * L_parallel );
	}

	// Energy loss per particle is ~ Chi_i + T_i
	return ParallelIonPastukhovLossRate( Chi_i(Te, Ti), Te, Ti ) * ( Ti * referenceTemperature ) * ( ::fabs( Chi_i(Te, Ti) )  + 1.0 );
}

double MirrorPlasma::ElectronCollisionTime( double Te) const
{
	double piThreeHalves = ::pow( M_PI, 1.5 ); // pi^(3/2)
	double TeThreeHalves = ::pow( Te * referenceTemperature, 1.5 );
	double ZIon = ionSpecies.Charge;
	return 12 * ::sqrt( electronMass ) * piThreeHalves * TeThreeHalves * vacuumPermittivity * vacuumPermittivity / ( ::sqrt(2) * ionDensity * referenceDensity * ::pow( ZIon, 2 ) * ::pow( e_charge, 4 ) * LogLambdaElectron(Te) );

}

double MirrorPlasma::IonCollisionTime(double Ti) const
{
	double piThreeHalves = ::pow( M_PI, 1.5 ); // pi^(3/2)
	double TiThreeHalves = ::pow( Ti * referenceTemperature, 1.5 );
	double ZIon = ionSpecies.Charge;
	return 12 * ::sqrt( ionSpecies.Mass * protonMass ) * piThreeHalves * TiThreeHalves * vacuumPermittivity * vacuumPermittivity / ( ::sqrt(2) * ionDensity * referenceDensity * ::pow( ZIon * e_charge, 4 ) * LogLambdaIon(Ti) );

}

double MirrorPlasma::CollisionalTemperatureEquilibrationTime(double Te) const
{
	return ElectronCollisionTime(Te)/( (3./ionSpecies.Mass)*(electronMass/protonMass) );
}

double MirrorPlasma::CentrifugalPotential(double Te, double Ti) const
{
	double tau = Ti / Te;
	return -( 0.5/tau ) * ( 1.0 - 1.0 / mirrorRatio ) * machNumber * machNumber / ( ionSpecies.Charge / tau + 1 );

}

double MirrorPlasma::AmbipolarPhi(double Te, double Ti) const
{
	double ambipolarPhi = CentrifugalPotential(Te, Ti);

	if ( collisional )
		return ambipolarPhi;

	if ( useAmbipolarPhi ) {

		// Add correction.
		double sigma = 1.0 + Zeff;
		double R = mirrorRatio;
		double Correction = ::log( (  ElectronCollisionTime(Te) / IonCollisionTime(Ti) ) * ( ::log( R*sigma ) / ( sigma * ::log( R ) ) ) );
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
				return ParallelIonPastukhovLossRate( Chi_i( Phi, Te, Ti ), Te, Ti )*ionSpecies.Charge - ParallelElectronPastukhovLossRate( Chi_e, Te );
			}
		};


		double thresholdCurrent = 1e-6 * electronDensity * referenceDensity;

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
				return CentrifugalPotential(Te, Ti) + Correction/2.0;
			}
		}
	}

	return ambipolarPhi;
}

double MirrorPlasma::LogLambdaElectron(double Te) const
{
	// Convert to NRL Formulary Units
	double neNRL = electronDensity * 1e14; // We use 10^20 / m^3 = 10^14 / cm^3
	double TeNRL = Te * 1000; // We normalize to 1 keV they use 1 ev
	// For sensible values of the coulomb logarithm, the lambda_ee value in the NRL formulary
	// can be simplified to be the same as the middle lambda_ei value.
	return 24.0 - 0.5*::log( neNRL ) + ::log( TeNRL );

}

double MirrorPlasma::LogLambdaIon(double Ti) const
{
	double TiNRL = Ti * 1000;
	double niNRL = ionDensity * 1e14;
	double ZIon = ionSpecies.Charge;
	return 23.0 - 0.5 * ::log( niNRL ) - 1.5 * ::log( ZIon * ZIon / TiNRL );

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
