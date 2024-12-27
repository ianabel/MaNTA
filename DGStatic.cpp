
// Translation unit for static objects related to DGApprox

#include "DGApprox.hpp"

std::map<unsigned int, LegendreBasis> LegendreBasis::singletons;
LegendreBasis::IntegratorType LegendreBasis::integrator;

std::map<unsigned int, ChebyshevBasis> ChebyshevBasis::singletons;
ChebyshevBasis::IntegratorType ChebyshevBasis::integrator;



