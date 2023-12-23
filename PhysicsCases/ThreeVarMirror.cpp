
#include "ThreeVarMirror.hpp"
#include "Constants.hpp"
#include <iostream>
#include <boost/math/tools/roots.hpp>

REGISTER_PHYSICS_IMPL(ThreeVarMirror);

enum
{
    None = 0,
    Gaussian = 1,
};

ThreeVarMirror::ThreeVarMirror( toml::value const &config, Grid const& grid )
	: AutodiffTransportSystem( config, grid, 3, 0 ) // nVars = 3, nScalars = 0
{
    if (config.count("3VarMirror") != 1)
        throw std::invalid_argument("There should be a [3VarMirror] section if you are using the 3VarMirror physics model.");

    auto const &DiffConfig = config.at("3VarMirror");

    std::string profile = toml::find_or(DiffConfig, "SourceType", "None");
    ParticleSource = ParticleSources[profile];

    sourceStrength = toml::find_or(DiffConfig, "SourceStrength", 0.0);
    sourceCenter = toml::find_or(DiffConfig, "SourceCenter", 0.25);
    sourceWidth = toml::find_or(DiffConfig, "SourceWidth", 0.01);

    Rmin = toml::find_or(DiffConfig, "Rmin", 0.1);
    Rmax = toml::find_or(DiffConfig, "Rmax", 1.0);

    M0 = toml::find_or(DiffConfig, "M0", 6.72);

    useConstantOmega = toml::find_or(DiffConfig, "useConstantOmega", false);
    includeParallelLosses = toml::find_or(DiffConfig, "includeParallelLosses", false);
    omegaOffset = toml::find_or(DiffConfig, "omegaOffset", 0.0);

    // reference values
    n0 = toml::find_or(DiffConfig, "n0", 3e19);
    T0 = e_charge * toml::find_or(DiffConfig, "T0", 1e3);
    Bmid = toml::find_or(DiffConfig, "Bmid", 1.0);
    L = toml::find_or(DiffConfig, "L", 1.0);
    double Rm = toml::find_or(DiffConfig, "Rm", 3.3);
    Bmax = Bmid.val * Rm;
    p0 = n0 * T0;

    Gamma0 = p0 / (electronMass * Om_e(Bmid) * Om_e(Bmid) * tau_e(n0, p0));
    V0 = Gamma0 / n0;

    taue0 = tau_e(n0, p0);
    taui0 = tau_i(n0, p0);

};

enum Channel : Index {
	Density = 0,
	ElectronEnergy = 1,
	IonEnergy = 2,
};

Real ThreeVarMirror::Flux( Index i, RealVector u, RealVector q, Position x, Time t )
{
	Channel c = static_cast<Channel>(i);
	switch(c) {
		case Density:
			return Gamma_hat( u, q, x, t );
			break;
		case ElectronEnergy:
			return qe_hat( u, q, x, t );
			break;
		case IonEnergy:
			return qi_hat( u, q, x, t );
			break;
		default:
			throw std::runtime_error("Request for flux for undefined variable!");
	}
}

Real ThreeVarMirror::Source( Index i, RealVector u, RealVector q, RealVector sigma, Position x, Time t )
{
	Channel c = static_cast<Channel>(i);
	switch(c) {
		case Density:
			return Sn_hat( u, q, sigma, x, t );
			break;
		case ElectronEnergy:
			return Spe_hat( u, q, sigma, x, t );
			break;
		case IonEnergy:
			return Spi_hat( u, q, sigma, x, t );
			break;
		default:
			throw std::runtime_error("Request for source for undefined variable!");
	}
}

Real ThreeVarMirror::Gamma_hat(RealVector u, RealVector q, Real x, double t)
{

    // maybe add a factor of sqrt x if x = r^2/2
    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    Real G = coef * u(1) / (tau_hat(u(0), u(1)) * lambda_hat(u(0), u(1), n0, p0)) * ((-q(1) / 2. + q(2)) / u(1) + 3. / 2. * q(0) / u(0));

    if ( !std::isfinite( G.val ) )
		 throw std::logic_error( "Particle flux returned Inf or NaN" );
    else
        return G;
};

Real ThreeVarMirror::qi_hat(RealVector u, RealVector q, Real x, double t)
{

    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    Real dT = q(2) / u(2) - q(0) / u(0);

    // Real G = Gamma_hat(u, q, x, t);`
    Real qri = ::sqrt(ionMass / (2. * electronMass)) * 1.0 / (tau_hat(u(0), u(2)) * lambda_hat(u(0), u(1), n0, p0)) * 2. * u(2) * u(2) / u(0) * dT;
    Real Q = (2. / 3.) * coef * qri;
    if ((Q != Q))
    {
        return 0.0;
    }
    else
    {
        return Q;
    }
    // return 0.0;
};
Real ThreeVarMirror::qe_hat(RealVector u, RealVector q, Real x, double t)
{

    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    // Real G = Gamma_hat(u, q, x, t);
    Real qre = 1.0 / (tau_hat(u(0), u(1)) * lambda_hat(u(0), u(1), n0, p0)) * (4.66 * u(1) * u(1) / u(0) * (q(1) / u(1) - q(0) / u(0)) - (3. / 2.) * u(1) / u(0) * (q(2) + q(1)));

    Real Q = (2. / 3.) * coef * qre;
    if (Q != Q)
    {
        return 0.0;
    }
    else

        return Q;
};

Real ThreeVarMirror::Sn_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
    Real S = 0.0;
    Real Spast = 0.0;
    Real Sfus = 0.0;
    Real n = u(0) * n0;
    Real T = u(2) / u(0) * T0;
    Real TeV = T / (e_charge);

    Real R = 1e-6 * 3.68e-12 * pow(TeV / 1000, -2. / 3.) * exp(-19.94 * pow(TeV / 1000, -1. / 3.));
    Sfus = 0.25 * L / (n0 * V0) * R * n * n;
    if (includeParallelLosses)
    {
        Real Rm = Bmax / B(x.val, t);
        Real Xe = Chi_e(u, q, x, t); // phi0(u, q, x, t) * u(0) / u(1) * (1 - 1 / Rm);
                                     //  Real Xi = Chi_e(u, q, x, t);
        Real c1 = L / (taue0 * V0);
        Real l1 = PastukhovLoss(u(0), u(1), Xe, Rm);
        // Real c2 = L / (taui0 * V0);
        // Real l2 = PastukhovLoss(u(0), u(2), Xi, Rm);
        Spast = (c1 * l1);
    }
    switch (ParticleSource)
    {
    case None:
        break;
    case Gaussian:
        S = sourceStrength * exp(-1 / sourceWidth * (x - sourceCenter) * (x - sourceCenter));
        break;
    default:
        break;
    }
    return S - Sfus + Spast;
};

Real ThreeVarMirror::Spi_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
    Real Ppot = 0;
    Real Pvis = 0;
    Real Ppast = 0.0;

    if (includeParallelLosses)
    {

        Real Rm = Bmax / B(x.val, t);
        Real Xi = Chi_i(u, q, x, t);
        Real Spast = L / (taui0 * V0) * PastukhovLoss(u(0), u(2), Xi, Rm);
        Ppast = u(2) / u(0) * (1 + Xi) * Spast;
    }

    if (useConstantOmega)
    {
        double Rval = R(x.val, t);
        double Vpval = Vprime(Rval);
        double coef = Vpval * Vpval * Rval * Rval * Rval * Rval;

        Real dV = domegadV(x, t);
        Real ghi = ::pow(ionMass / electronMass, 1. / 2.) * 1.0 / (::sqrt(2) * tau_hat(u(0), u(2)) * lambda_hat(u(0), u(1), n0, p0)) * 3. / 10. * u(2);
        Pvis = ghi * dV * dV * coef;

        Real G = sigma(0); //-Gamma_hat(u, q, x, t);
        // sigma(0);
        //  Gamma_hat(u, q, x, t); // sigma(0); // / (coef);
        Ppot = -G * dphi0dV(u, q, x, t) + 0.5 * pow(omega(Rval, t), 2) / M_PI * G;
    }
    Real Pcol = Ci(u(0), u(2), u(1)) * L / (V0 * taue0);
    Real S = 2. / 3. * (Ppot + Pcol + Pvis + Ppast);

    if (S != S)
    {
        return 0.0;
    }

    else
    {
        return S; //+ 10 * Sn_hat(u, q, sigma, x, t); //+ ::pow(ionMass / electronMass, 1. / 2.) * u(2) / u(0) * Sn_hat(u, q, sigma, x, t);
    }
    // return 0.0;
}

Real ThreeVarMirror::Spe_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
    // double Rval = R(x.val, t);
    // double Vpval = Vprime(Rval);
    // double coef = Vpval * Rval;
    // Real G = Gamma_hat(u, q, x, t); // (coef);
    // Real V = G / u(0);              //* L / (p0);

    // Real S = -2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
    ///*V * q(1)*/ -2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
    // Real Pcol = 2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
    Real Pfus = 0.0;
    Real Pbrem = 0.0;
    Real Ppast = 0.0;
    Real Rm = Bmax / B(x.val, t);
    if (includeParallelLosses)
    {
        Real Xe = Chi_e(u, q, x, t); // phi0(u, q, x, t) * u(0) / u(1) * (1 - 1 / Rm);
        Real Spast = L / (taue0 * V0) * PastukhovLoss(u(0), u(1), Xe, Rm);
        Ppast = u(1) / u(0) * (1 + Xe) * Spast;
    }

    //
    Real n = u(0) * n0;
    Real T = u(1) / u(0) * T0;
    Real TeV = T / (e_charge);
    Pbrem = 2 * -1e6 * 1.69e-32 * (n * n) * 1e-12 * sqrt(TeV); //(-5.34e3 * pow(n / 1e20, 2) * pow(TkeV, 0.5)) * L / (p0 * V0);
    Real TikeV = u(2) / u(0) / 1000 * T0 / e_charge;
    if (TikeV > 25)
        TikeV = 25;

    Real R = 3.68e-12 * pow(TikeV, -2. / 3.) * exp(-19.94 * pow(TikeV, -1. / 3.));
    // 1e-6 * n0 * n0 * R * u(0) * u(0);
    Pfus = 0.25 * sqrt(1 - 1 / Rm) * 1e6 * 5.6e-13 * n * n * 1e-12 * R; // n *n * 5.6e-13

    Real Pcol = Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
    Real S = 2. / 3. * (Pcol + Ppast + L / (p0 * V0) * (Pfus + Pbrem));

    if (S != S)
    {
        return 0.0;
    }
    else
    {
        return S; //+ u(1) / u(0) * Sn_hat(u, q, sigma, x, t);
    }
    // return 0.0;
};

Real ThreeVarMirror::omega(Real R, double t)
{
    double u_L = omegaOffset.val;
    double u_R = (omegaOffset.val * Rmin / Rmax);
    Real a = (asinh(u_L) - asinh(u_R)) / (Rmin - Rmax);
    Real b = (asinh(u_L) - Rmin / Rmax * asinh(u_R)) / (a * (Rmin / Rmax - 1));

    Real shape = 20.0;
    Real C = 0.5 * (Rmin + Rmax);
    Real c = (M_PI / 2 - 3 * M_PI / 2) / (Rmin - Rmax);
    Real d = (M_PI / 2 - Rmin / Rmax * (3 * M_PI / 2)) / (c * (Rmin / Rmax - 1));
    // Real coef = (omegaOffset - M0 / C) * 1 / cos(c * (C - d));
    Real coef = M0 / C;
    if (omegaOffset == 0.0)
    {
        return omegaOffset - cos(c * (R - d)) * coef * exp(-shape * (R - C) * (R - C));
    }
    else
    {
        return sinh(a * (R - b)) - cos(c * (R - d)) * coef * exp(-shape * (R - C) * (R - C));
    }
}

double ThreeVarMirror::domegadV(Real x, double t)
{
    Real Rval = R(x.val, t);
    double Bval = B(x.val, t) / Bmid.val;
    double Vpval = Vprime(Rval.val);
	 auto omegaFn = [ this, t ]( Real x ) -> Real { return this->omega(x,t); };
    double domegadR = derivative( omegaFn, wrt(Rval), at(Rval));
    return domegadR / (Vpval * Rval.val * Bval);
}

Real ThreeVarMirror::phi0(RealVector u, RealVector q, Real x, double t)
{
    Real Rval = R(x.val, t);
    Real Rm = Bmax / Bmid.val;
    Real Romega = Rval * omega(Rval, t);
    Real tau = u(2) / u(0);
    Real phi0 = 1 / (1 + tau) * (1 / Rm - 1) * Romega * Romega / 2; // pow(omega(Rval, t), 2) * Rval * Rval / (2 * u(2)) * 1 / (1 / u(2) + 1 / u(1));

    return phi0;
}
Real ThreeVarMirror::dphi0dV(RealVector u, RealVector q, Real x, double t)
{
    Real Rval = R(x.val, t);
	 auto phi0fn = [ this ]( RealVector u, RealVector q, Real x, double t ) -> Real {
		 return this->phi0( u, q, x, t );
	 };
    Real dphi0dV = derivative(phi0fn, wrt(Rval), at(u, q, x, t)) / (2 * M_PI * Rval);
    auto dphi0du = gradient(phi0fn, wrt(u), at(u, q, x, t));
    auto qi = q.begin();
    for (auto &dphi0i : dphi0du)
    {
        dphi0dV += *qi * dphi0i;
        ++qi;
    }

    //  Real dphi0dV = (q.val * dphi0du).sum();
    return dphi0dV;
}

Real ThreeVarMirror::Chi_e(RealVector u, RealVector q, Real x, double t)
{
    Real Rval = R(x.val, t);
    Real Rm = Bmax / B(x.val, t);
    Real tau = u(2) / u(1);
    Real M2 = Rval * Rval * pow(omega(Rval, t), 2) * u(0) / u(1);

    return 0.5 * (1 - 1 / Rm) * M2 * 1 / (tau + 1);
}

Real ThreeVarMirror::Chi_i(RealVector u, RealVector q, Real x, double t)
{
    Real Rval = R(x.val, t);
    Real Rm = Bmax / B(x.val, t);
    Real tau = u(2) / u(1);
    Real M2 = Rval * Rval * pow(omega(Rval, t), 2) * u(0) / u(2);
    return 0.5 * tau / (1 + tau) * (1 - 1 / Rm) * M2;
}

double ThreeVarMirror::psi(double R)
{
    return M_PI * R * R * Bmid.val;
}

double ThreeVarMirror::V(double R)
{
    return M_PI * R * R * L;
}

// V' == dV/dPsi or dV/dR ?
double ThreeVarMirror::Vprime(double R)
{
    return L/Bmid.val;
}

double ThreeVarMirror::B(double x, double t)
{
    return Bmid.val; //* (1 + m * (Rval - Rmin)); //* exp(-0.5 * Rval * Rval); // / R(x, t);
}

double ThreeVarMirror::R(double x, double t)
{
    using boost::math::tools::bracket_and_solve_root;
    using boost::math::tools::eps_tolerance;
    double guess = 0.5;                                     // 
    // double min = Rmin;                                      // Minimum possible value is half our guess.
    // double max = Rmax;                                      // Maximum possible value is twice our guess.
    const int digits = std::numeric_limits<double>::digits; // Maximum possible binary digits accuracy for type T.
    int get_digits = static_cast<int>(digits * 0.6);        // Accuracy doubles with each step, so stop when we have
                                                            // just over half the digits correct.
    double factor = 2;
    bool is_rising = true;

    auto getPair = [this](double x, double R) { return this->V(R) - x; }; // change to V(R(psi))
    auto func = std::bind_front(getPair, x);

    eps_tolerance<double> tol(get_digits);

    const boost::uintmax_t maxit = 20;
    boost::uintmax_t it = maxit;
    std::pair<double, double> r = bracket_and_solve_root(func, guess, factor, is_rising, tol, it);
    return r.first + (r.second - r.first) / 2;
};

