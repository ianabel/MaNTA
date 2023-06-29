#pragma once

#include "Species.hpp"

#include <boost/math/interpolators/makima.hpp>

#include <memory>
#include <cmath>
#include <map>
#include <optional>

class MirrorPlasma {
public:
	MirrorPlasma();
	//MirrorPlasma( MirrorPlasma const& ) = delete;
	//MirrorPlasma(const std::map<std::string, double>& parameterMap, std::string FuelName, bool reportThrust, std::optional<bool> IncludeAlphaHeating, std::optional<bool> ReportNuclearDiagnostics, bool ambiPolPhi, bool collisions, bool includeCXLosses, std::string asciiOut, std::string netCdfOut, std::string vTrace);

	double ParallelElectronPastukhovLossRate( double chi_e, double Te, double R ) const;
	double ParallelElectronParticleLoss(double Te, double Ti, double omega, double R) const;
	double ParallelElectronHeatLoss(double Te, double Ti, double omega, double R) const;

	double Chi_i( double phi, double Te, double Ti, double omega, double R) const;
	double Chi_i(double Te, double Ti, double omega, double R) const;

	double ParallelIonPastukhovLossRate( double phi, double Te, double Ti, double R) const;
	double ParallelIonParticleLoss(double Te, double Ti, double omega, double R) const;
	double ParallelIonHeatLoss(double Te, double Ti, double omega, double R) const;

	double SoundSpeed(double Te) const {
		// We *define* c_s^2 = Z_i T_e / m_i.
		double cs = ::sqrt( ionSpecies.Charge * Te*referenceTemperature / ( ionSpecies.Mass * protonMass ) );
		return cs;
	};

	double ElectronCollisionTime(double Te, double R) const;
	double delTe_ElectronCollisionTime(double Te, double R) const;
	double IonCollisionTime(double Ti, double R) const;
	double delTi_IonCollisionTime(double Ti, double R) const;
	double CollisionalTemperatureEquilibrationTime(double Te, double R) const;

	double CentrifugalPotential(double Te, double Ti, double omega, double R) const;
	double AmbipolarPhi(double Te, double Ti, double omega, double R) const;

	double LogLambdaElectron(double Te) const;
	double LogLambdaIon(double Ti) const;

	//T_i derivative functions
	double delTi_ParallelIonParticleLoss(double Te, double Ti, double omega, double R);
	double delTi_ParallelIonHeatLoss(double Te, double Ti, double omega, double R);
	double delTi_Chi_i(double Te, double Ti, double omega, double R);
	double delTi_ParallelElectronParticleLoss() {return 0.0;};
	double delTi_ParallelElectronHeatLoss() {return 0.0;};
	double delTi_MachNumber() {return 0.0;}

	//T_e derivative functions
	double delTe_ParallelIonParticleLoss(double Te, double Ti, double omega, double R);
	double delTe_ParallelIonHeatLoss(double Te, double Ti, double omega, double R);
	double delTe_Chi_i(double Te, double Ti, double omega, double R);
	double delTe_ParallelElectronParticleLoss(double Te, double Ti, double omega, double R);
	double delTe_ParallelElectronHeatLoss(double Te, double Ti, double omega, double R);
	double delTe_MachNumber(double Te, double Ti, double omega, double R);

	//omega derivative functions
	double delOmega_ParallelIonParticleLoss(double Te, double Ti, double omega, double R);
	double delOmega_ParallelIonHeatLoss(double Te, double Ti, double omega, double R);
	double delOmega_Chi_i(double Te, double Ti, double omega, double R);
	double delOmega_ParallelElectronParticleLoss(double Te, double Ti, double omega, double R) {return 0.0;}
	double delOmega_ParallelElectronHeatLoss(double Te, double Ti, double omega, double R) {return 0.0;}
	double delOmega_MachNumber(double Te, double Ti, double omega, double R);

	bool collisional = false;
	bool useAmbipolarPhi = false;
	bool includeAlphaHeating = false;

	double Transition( double x, double L, double U ) const;
	double machNumber(double R, double Te, double omega) const;
	double electronDensity(double R) const;
	double ionDensity(double R) const;

	Species_t ionSpecies;
	double mirrorRatio;
	double Zeff;
	double plasmaLength;
	double parallelFudgeFactor;
	double density;

	double storedPhi = 0.0;
};
