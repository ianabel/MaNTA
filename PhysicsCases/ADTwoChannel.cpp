#include "ADTwoChannel.hpp"
#include "Constants.hpp"
#include <iostream>

REGISTER_PHYSICS_IMPL(ADTwoChannel);

/*
 * Semi-coupled test problem with two channels.
 Let u(0) = v / u(1) = w

 d_t v + d_x ( (kappa/w^1.5) d_x v ) = S_v(x)
 d_t w + d_x ( v * (kappa/w^1.5) * d_x w ) = v * S_w(x)
 */

ADTwoChannel::ADTwoChannel(toml::value const &config, Grid const &grid)
	: AutodiffTransportSystem(config, grid, 2, 0, 0) // Configure a blank autodiff system with three variables and no scalars
{
	isLowerDirichlet = true;
	isUpperDirichlet = true;
	uR = {0.1, 0.3};
	uL = {0.1, 0.3};
	double x_l = grid.lowerBoundary(), x_u = grid.upperBoundary();
	c = (x_u - x_l) / 2.0;
	A = M_PI / (x_u + x_l);
	H = {1., 3.};
	S_w = 3;
	kappa = 1;
};

Real ADTwoChannel::Flux(Index i, RealVector u, RealVector q, Position x, Time t)
{
	if (u(1) < 0 || u(0) < 0)
	{
		throw std::runtime_error("ABORT, negative value encountered");
	}
	switch (i)
	{
	case 0:
		return (kappa / pow(u(1), 1.5)) * q(0);
		break;
	case 1:
		return (kappa / pow(u(1), 1.5)) * u(0) * q(1);
		break;
	default:
		throw std::logic_error("i > nVars in ADTwoChannel::Flux");
	}
	return 0.0;
}

Real ADTwoChannel::Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector, Position x, Time t)
{
	switch (i)
	{
	default:
		throw std::logic_error("i > nVars in ADTwoChannel::Flux");
		break;
	case 0:
		return exp(-A * A * (x - c) * (x - c));
		break;
	case 1:
		Real sin_x = sin(2.0 * A * (x - c));
		return u(0) * S_w * (sin_x * sin_x);
		break;
	}
	return 0.0;
}

Value ADTwoChannel::InitialValue(Index i, Position x) const
{
	return H[i] * cos(A * (x - c)) + uL[i];
}

Value ADTwoChannel::InitialDerivative(Index i, Position x) const
{
	return -H[i] * A * sin(A * (x - c));
}
