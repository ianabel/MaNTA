#pragma once

constexpr double electronMass = 9.1094e-31;	//Electron Mass, kg
constexpr double ionMass = 1.6726e-27;	//Ion Mass ( = proton mass) kg
constexpr double protonMass = 1.6726e-27;	//Proton mass, kg
constexpr double e_charge = 1.60217663e-19; //Coulombs
constexpr double vacuumPermittivity = 8.8541878128e-12;

double eV_J(double T_ev);
double J_eV(double T_J);

// Reference temperature of 1 keV, expressed in Joules
constexpr double referenceTemperature = 1000 * e_charge;
constexpr double referenceDensity = 1e20;