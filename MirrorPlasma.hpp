#pragma once

#include "Species.hpp"

#include <boost/math/interpolators/makima.hpp>

#include <memory>
#include <cmath>
#include <map>
#include <optional>

class MirrorPlasma {
public:
	MirrorPlasma() = default;
	//MirrorPlasma( MirrorPlasma const& ) = delete;
	//MirrorPlasma(const std::map<std::string, double>& parameterMap, std::string FuelName, bool reportThrust, std::optional<bool> IncludeAlphaHeating, std::optional<bool> ReportNuclearDiagnostics, bool ambiPolPhi, bool collisions, bool includeCXLosses, std::string asciiOut, std::string netCdfOut, std::string vTrace);

	double ParallelElectronPastukhovLossRate( double chi_e, double Te ) const;
	double ParallelElectronParticleLoss(double Te, double Ti) const;
	double ParallelElectronHeatLoss(double Te, double Ti) const;

	double Chi_i( double phi, double Te, double Ti) const;
	double Chi_i(double Te, double Ti) const;

	double ParallelIonPastukhovLossRate( double phi, double Te, double Ti) const;
	double ParallelIonParticleLoss(double Te, double Ti) const;
	double ParallelIonHeatLoss(double Te, double Ti) const;

	double SoundSpeed(double Te) const {
		// We *define* c_s^2 = Z_i T_e / m_i.
		double cs = ::sqrt( ionSpecies.Charge * Te*referenceTemperature / ( ionSpecies.Mass * protonMass ) );
		return cs;
	};

	double ElectronCollisionTime(double Te) const;
	double IonCollisionTime(double Ti) const;
	double CollisionalTemperatureEquilibrationTime(double Te) const;

	double CentrifugalPotential(double Te, double Ti) const;
	double AmbipolarPhi(double Te, double Ti) const;

	double LogLambdaElectron(double Te) const;
	double LogLambdaIon(double Ti) const;

	bool collisional;
	bool useAmbipolarPhi;
	bool includeAlphaHeating;

	double Transition( double x, double L, double U ) const;

	Species_t ionSpecies;
	double mirrorRatio;
	double electronDensity;
	double ionDensity;
	double Zeff;
	double plasmaLength;
	double parallelFudgeFactor;

	double machNumber;

	double storedPhi;
};
