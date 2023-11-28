#pragma once

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

using namespace autodiff;

constexpr double electronMass = 9.1094e-31; // Electron Mass, kg
constexpr double ionMass = 1.6726e-27;      // Ion Mass ( = proton mass) kg
constexpr double protonMass = 1.6726e-27;   // Proton mass, kg
constexpr double e_charge = 1.60217663e-19; // Coulombs
constexpr double vacuumPermittivity = 8.8541878128e-12;
constexpr double lambda = 15.0;

double eV_J(double T_ev);
double J_eV(double T_J);

// Reference temperature of 1 keV, expressed in Joules
constexpr double referenceTemperature = 1000 * e_charge;
constexpr double referenceDensity = 1e20;

constexpr double B_mid = 1.0; // Tesla
dual Om_i(dual B);
dual Om_e(dual B);

dual nu(dual n, dual Pe);
dual tau_e(dual n, dual Pe);
dual tau_i(dual n, dual Pi);
dual tau_hat(dual n, dual P);
dual Ce(dual n, dual Pi, dual Pe);

dual Ci(dual n, dual Pi, dual Pe);
dual RT(dual n, dual Pe);
dual RDT(dual n, dual Pe);
