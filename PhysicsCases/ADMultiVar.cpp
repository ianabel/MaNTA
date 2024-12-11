#include "ADMultiVar.hpp"
#include "Constants.hpp"
#include <iostream>

REGISTER_PHYSICS_IMPL(ADMultiVar);

/* 
 * Autodiff implementation of the coupled test problem
 *
 * d_t v_0 - d_x v_0^a d_x v_0 = S_0
 * d_t v_1 - d_x v_1^b d_x v_1 = S_1
 *
 * which we solve in the form
 *
 * v_0 = (u+w) ; v_1 = (u-w)
 *
 * d_t u - d_x [ (u+w)^a ( d_x u + d_x w )/2 + (u-w)^b ( d_x u - d_x w )/2 ] = (S_0 + S_1)/2
 * d_t w - d_x [ (u+w)^a ( d_x u + d_x w )/2 - (u-w)^b ( d_x u - d_x w )/2 ] = (S_0 - S_1)/2
 *
 * v_0'(x=0) = 0
 * v_1'(x=0) = 0
 * v_0(x = 1) = 0.1
 * v_1(x = 1) = 0.1
 *
 * implies
 *
 * u' = w' = 0 at x=0
 *
 * and
 *
 * u = 0.1
 * w = 0.0
 *
 * at x = 1
 *
 * using gaussian sources S_0, S_1
 */

ADMultiVar::ADMultiVar(toml::value const &config, Grid const &grid)
	: AutodiffTransportSystem(config, grid, 2, 0, 0) // Configure a blank autodiff system with two variables and no scalars
{
	if (config.count("ADMultiVar") != 1)
	{
		throw std::invalid_argument("There should be a [ADMultiVar] section.");
	}

    a = 1.0;
    b = -1.5;
    A0 = 1.0;
    A1 = 1.0;
    x0 = 0.0;
    x1 = 0.3;
    w0 = 0.1;
    w1 = 0.1;
}

Real ADMultiVar::Flux(Index i, RealVector uV, RealVector qV, Real x, Time t)
{
  Real u = uV(0), w = uV(1);
  Real du = qV(0), dw = qV(1);
  switch( i )
  {
    case 0:
      return pow( u + w, a ) * ( du + dw )/2.0 + pow( u - w, b ) * ( du - dw )/2.0;
    case 1:
      return pow( u + w, a ) * ( du + dw )/2.0 - pow( u - w, b ) * ( du - dw )/2.0;
  }
  return 0;
}

Real ADMultiVar::Source(Index i, RealVector uV, RealVector qV, RealVector sigma, RealVector, Real x, Time t)
{
  Real u = uV(0), w = uV(1);
  Real du = qV(0), dw = qV(1);
  switch( i )
  {
    case 0:
      return pow( u + w, a ) * ( du + dw )/2.0 + pow( u - w, b ) * ( du - dw )/2.0;
    case 1:
      return pow( u + w, a ) * ( du + dw )/2.0 - pow( u - w, b ) * ( du - dw )/2.0;
  }
  return 0;
}

Real ADMultiVar::S0( Real x )
{
  Real y = (x - x0)/w0;
  return A0*exp( - y*y );
}


Real ADMultiVar::S1( Real x )
{
  Real y = (x - x1)/w1;
  return A1*exp( - y*y );
}

// Initially v_0=v_1=(1.1-x*x)
Value ADMultiVar::InitialValue( Index i, Position x ) const
{
  switch( i ) {
    case 0: // u = (1.1-x*x)
      return 1.1-x*x;
    case 1: // w = 0
      return 0;
  }
  throw std::logic_error("Index illegitimate");
}

Value ADMultiVar::InitialDerivative( Index i, Position x ) const
{
  switch( i ) {
    case 0: // u' = -2.0*x
      return -2.0*x;
    case 1: // w = 0
      return 0;
  }
  throw std::logic_error("Index illegitimate");
}
