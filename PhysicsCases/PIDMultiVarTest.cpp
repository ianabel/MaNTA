
#include "PIDMultiVarTest.hpp"
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <cmath>
#include <numbers>

/*
   Nonlinear Diffusion test case with a coupled scalar.

   du    d       du
   -- - -- Kappa -- = J S( x )
   dt   dx       dx

   with Kappa = kappa * u^a

   where J is chosen to enforce constant total mass M of u i.e.

    d  /1       dM
   --  |   u =  --  = 0
   dt  /-1      dt

   and

   S( x ) = A exp( -( x/ alpha )^2 ) ; with A^-1 = alpha * sqrt( pi ) * Erf[ 1/alpha ] so S has unit mass

   The explicit equation for J is

   J_exact = [ - Kappa du/dx ]_( x = 1 ) - [ - Kappa du/dx ]_( x = -1 )

   but we use a PID controller on top of that to keep M constant:

   E = M(t=0) - M
   J = gamma * E + gamma_d * dE/dt + gamma_I * Int_0^t ( E(t') dt' ) + J_exact

   to handle the integral term, we write

   J = gamma * E + gamma_d * dE/dt + gamma_I * I + J_exact
   dI/dt = E

   and treat I as a third scalar.

   For boundary conditions we take u = u0 at +/-1 and expect the solution to remain symmettric around 0 if it starts so.


   M is constant in a steady state, so Mathematica kindly works out that the general steady state is

   (((a + 1)/kappa) * ( J*S2(x) + u0^(a + 1)))^(1/(a + 1))

   where

   S2(x) = 1/2 A alpha ((-exp(-(1/alpha^2)) + exp(-(x^2/alpha^2))) alpha + Sqrt(Pi) (-Erf[1/alpha] + x Erf[x/alpha]))


*/

// Needed to register the class
REGISTER_PHYSICS_IMPL(PIDMultiVarTest);

PIDMultiVarTest::PIDMultiVarTest(toml::value const &config, Grid const&)
{
    // Always set nVars in a derived constructor
    nVars = 1;
    nScalars = 3;

    // Construst your problem from user-specified config
    // throw an exception if you can't. NEVER leave a part-constructed object around
    // here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

    if (config.count("DiffusionProblem") != 1)
        throw std::invalid_argument("There should be a [DiffusionProblem] section if you are using the PIDMultiVarTest physics model.");

    auto const &DiffConfig = config.at("DiffusionProblem");

    kappa = toml::find_or(DiffConfig, "Kappa", 1.0);
    alpha = toml::find_or(DiffConfig, "alpha", 0.2);
    beta = toml::find_or(DiffConfig, "beta", 1.0);
    gamma = toml::find_or(DiffConfig, "gamma", 1.0);
    gamma_d = toml::find_or(DiffConfig, "gamma_d", 0.0);
    gamma_I = toml::find_or(DiffConfig, "gamma_I", 0.0);
    u0 = toml::find_or(DiffConfig, "u0", 0.1);


    a = 1.0;
    M0 = 2*u0 + 4*beta/std::numbers::pi;
    std::cerr << "M0 : " << M0 << std::endl;
}

// Dirichlet Boundary Conditon
Value PIDMultiVarTest::LowerBoundary(Index, Time) const
{
    return u0;
}

Value PIDMultiVarTest::UpperBoundary(Index, Time) const
{
    return u0;
}

bool PIDMultiVarTest::isLowerBoundaryDirichlet(Index) const { return true; };
bool PIDMultiVarTest::isUpperBoundaryDirichlet(Index) const { return true; };

Value PIDMultiVarTest::SigmaFn(Index, const State &s, Position x, Time)
{
    return kappa * pow(s.Variable[ 0 ],a) * s.Derivative[0];
}

Value PIDMultiVarTest::ScaledSource( Position x ) const
{
    double Ainv = alpha * std::sqrt( std::numbers::pi ) * std::erf( 1.0/alpha );
    return exp( -( x/alpha )*( x/alpha ) )/Ainv;
}

Value PIDMultiVarTest::Sources(Index, const State &s, Position x, Time)
{
    double J = s.Scalars[ 1 ];

    return J * ScaledSource( x ) + 0.5*std::cos( std::numbers::pi * x );
}

void PIDMultiVarTest::dSigmaFn_dq(Index, Values &v, const State &s, Position, Time)
{
    v[0] = kappa * pow(s.Variable[ 0 ], a);
};

void PIDMultiVarTest::dSigmaFn_du(Index, Values &v, const State &s, Position, Time)
{
    if( a == 0 )
        v[ 0 ] = 0.0;
    v[0] = kappa * a * pow( s.Variable[ 0 ], a - 1 ) * s.Derivative[ 0 ];
};

void PIDMultiVarTest::dSources_du(Index, Values &v, const State &, Position, Time)
{
    v[0] = 0.0;
};

void PIDMultiVarTest::dSources_dq(Index, Values &v, const State &, Position, Time)
{
    v[0] = 0.0;
};

void PIDMultiVarTest::dSources_dsigma(Index, Values &v, const State &, Position, Time)
{
    v[0] = 0.0;
};

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value PIDMultiVarTest::InitialValue(Index, Position x) const
{
    return u0 + beta*std::cos( std::numbers::pi * x / 2.0 );
}

Value PIDMultiVarTest::InitialDerivative(Index, Position x) const
{
    return -( beta * std::numbers::pi / 2.0 )*std::sin( std::numbers::pi * x / 2.0 );
}

bool PIDMultiVarTest::isScalarDifferential( Index s ) 
{
    if( s == 0 || s == 2) 
        return true; // E & I are differential, as we depend on d{E,I}/dt expliticly
    else
        return false; // J is not differential
}

Value PIDMultiVarTest::ScalarGExtended( Index s, const DGSoln & y, const DGSoln & dydt, Time )
{
    double dEdt = dydt.Scalar(0);
    double dIdt = dydt.Scalar(2);
    double E = y.Scalar(0);
    double J = y.Scalar(1);
    double I = y.Scalar(2);
    if( s == 0 ) {
        // E = (M0 - M)
        // => G_0 = E - (M-M0)
        double M = boost::math::quadrature::gauss_kronrod<double, 31>::integrate( [ & ]( double x ){ return y.u( 0 )( x );}, -1, 1 );
        return E - (M0-M);
    } else if ( s == 1 ) {
        // J = gamma * E + gamma_d * dE/dt + gamma_I * I + [ sigma(x = +1) - sigma(x = -1) ]
        // => G_1 = J - gamma * E - gamma_d * dE/dt - gamma_I * I - [ sigma(x = +1) - sigma(x = -1) ]
        return J - gamma * E - gamma_d * dEdt - gamma_I * I - ( y.sigma( 0 )( 1 ) - y.sigma( 0 )( -1 ) );
    } else if ( s == 2 ) {
        // dI/dt = E <=> I = Int_0^{t} E
        return dIdt - E;
    } else {
        throw std::logic_error("scalar index > nScalars");
    }
}

void PIDMultiVarTest::ScalarGPrimeExtended( Index scalarIndex, State &s, State &out_dt, const DGSoln &y, const DGSoln & dydt, std::function<double( double )> P, Interval I, Time )
{
    s.zero();
    out_dt.zero();
    if ( scalarIndex == 0 ) {
        s.Flux[ 0 ] = 0.0; // d G_0 / d sigma
        s.Derivative[ 0 ] = 0.0; // d G_0 / d (u')
                                 // dG_0 / du = - dM/du (as functional derivative, taken as an inner product with P)
        double P_mass = boost::math::quadrature::gauss_kronrod<double, 31>::integrate( P, I.x_l, I.x_u );
        s.Variable[ 0 ] = -P_mass;
        s.Scalars[ 0 ] = 1.0; // dG_0/dE
        s.Scalars[ 1 ] = 0.0; // dG_0/dJ
    } else if ( scalarIndex == 1 ) {
        // dG_1 / d sigma = -[ delta(x-1) - delta(x + 1) ] ;
        // return as functional derivative acting on P
        s.Flux[ 0 ] = 0.0;
        if ( abs( I.x_u - 1 ) < 1e-9 )
            s.Flux[ 0 ] -= P( I.x_u );
        if ( abs( I.x_l + 1 ) < 1e-9 )
            s.Flux[ 0 ] += P( I.x_l );
        // dG_1/dE
        s.Scalars[ 0 ] = -gamma;
        // dG_1/dJ
        s.Scalars[ 1 ] = 1.0;
        out_dt.Scalars[ 0 ] = -gamma_d;
    } else if ( scalarIndex == 2 ) {
        // G_2 = I-dot - E
        // dG_2/dE = -1 
        s.Scalars[ 0 ] = -1.0;
        // dG_2/dIdot = 1
        out_dt.Scalars[ 2 ] = 1.0;
    } else {
        throw std::logic_error("scalar index > nScalars");
    }
}

void PIDMultiVarTest::dSources_dScalars( Index, Values &v, const State &, Position x, Time )
{
    v[ 0 ] = 0.0;
    v[ 1 ] = ScaledSource( x );
}

Value PIDMultiVarTest::InitialScalarValue( Index s ) const
{
    // Our job to make sure this is consistent!
    if( s == 0 ) // E
        return 0;
    else if (s == 1) // J
        return -kappa * ( pow(u0,a) * InitialDerivative( 0, 1 ) - pow( u0 , a )*InitialDerivative( 0, -1 ) );
    else if (s == 2) // I
        return 0.0;
    else
        throw std::logic_error("scalar index > nScalars");
}

Value PIDMultiVarTest::InitialScalarDerivative( Index s, const DGSoln& y, const DGSoln &dydt ) const
{
    // Our job to make sure this is consistent!
    if( s == 0 ) // dE/dt at t=0
    {
        double Mdot = boost::math::quadrature::gauss_kronrod<double, 31>::integrate( [ & ]( double x ){ return dydt.u( 0 )( x );}, -1, 1 );
        return Mdot;
    } else if ( s == 2 ) {
        double E = y.Scalar(0);
        return E; // dI/dt = E
    } else
        throw std::logic_error("Initial derivative called for algebraic (non-differential) scalar");
}

void PIDMultiVarTest::initialiseDiagnostics( NetCDFIO &nc )
{
    nc.AddTimeSeries( "Mass", "Integral of the solution over the domain", "", M0 );
}

void PIDMultiVarTest::writeDiagnostics( DGSoln const& y, double, NetCDFIO &nc, size_t tIndex )
{
    double mass = boost::math::quadrature::gauss_kronrod<double, 31>::integrate( [ & ]( double x ){ return y.u( 0 )( x );}, -1, 1 );
    nc.AppendToTimeSeries( "Mass", mass, tIndex );
}


