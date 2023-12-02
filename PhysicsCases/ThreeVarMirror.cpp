#include "ThreeVarMirror.hpp"
#include "Constants.hpp"
#include <iostream>
#include <boost/math/tools/roots.hpp>

REGISTER_FLUX_IMPL(ThreeVarMirror);

enum
{
    None = 0,
    Gaussian = 1,
};

template <typename T>
int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

ThreeVarMirror::ThreeVarMirror(toml::value const &config, Index nVars)
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

    sigma.insert(std::pair<Index, sigmaptr>(0, &Gamma_hat));
    sigma.insert(std::pair<Index, sigmaptr>(1, &qe_hat));
    sigma.insert(std::pair<Index, sigmaptr>(2, &qi_hat));

    source.insert(std::pair<Index, sourceptr>(0, &Sn_hat));
    source.insert(std::pair<Index, sourceptr>(1, &Spe_hat));
    source.insert(std::pair<Index, sourceptr>(2, &Spi_hat));
};

int ThreeVarMirror::ParticleSource;
double ThreeVarMirror::sourceStrength;
dual ThreeVarMirror::sourceCenter;
dual ThreeVarMirror::sourceWidth;

// reference values
dual ThreeVarMirror::n0;
dual ThreeVarMirror::T0;
dual ThreeVarMirror::Bmid;
Value ThreeVarMirror::L;
double ThreeVarMirror::Bmax;

dual ThreeVarMirror::p0;
dual ThreeVarMirror::Gamma0;
dual ThreeVarMirror::V0;
dual ThreeVarMirror::taue0;
dual ThreeVarMirror::taui0;

double ThreeVarMirror::Rmin;
double ThreeVarMirror::Rmax;
dual ThreeVarMirror::M0;
bool ThreeVarMirror::useConstantOmega;
bool ThreeVarMirror::includeParallelLosses;
dual ThreeVarMirror::omegaOffset;

dual ThreeVarMirror::Gamma_hat(VectorXdual u, VectorXdual q, dual x, double t)
{

    // maybe add a factor of sqrt x if x = r^2/2
    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    dual G = coef * u(1) / (tau_hat(u(0), u(1)) * lambda_hat(u(0), u(1), n0, p0)) * ((-q(1) / 2. + q(2)) / u(1) + 3. / 2. * q(0) / u(0));

    if (G != G)
    {
        return 0.0;
    }

    else
        return G;
};
dual ThreeVarMirror::qi_hat(VectorXdual u, VectorXdual q, dual x, double t)
{

    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    dual dT = q(2) / u(2) - q(0) / u(0);

    // dual G = Gamma_hat(u, q, x, t);`
    dual qri = ::sqrt(ionMass / (2. * electronMass)) * 1.0 / (tau_hat(u(0), u(2)) * lambda_hat(u(0), u(1), n0, p0)) * 2. * u(2) * u(2) / u(0) * dT;
    dual Q = (2. / 3.) * coef * qri;
    if ((Q != Q))
    {
        //  std::cout << Q << std::endl;
        return 0.0;
    }
    else
    {
        // std::cout << sgn(q(2).val) << std::endl;
        return Q;
    }
    // return 0.0;
};
dual ThreeVarMirror::qe_hat(VectorXdual u, VectorXdual q, dual x, double t)
{

    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    // dual G = Gamma_hat(u, q, x, t);
    dual qre = 1.0 / (tau_hat(u(0), u(1)) * lambda_hat(u(0), u(1), n0, p0)) * (4.66 * u(1) * u(1) / u(0) * (q(1) / u(1) - q(0) / u(0)) - (3. / 2.) * u(1) / u(0) * (q(2) + q(1)));

    dual Q = (2. / 3.) * coef * qre;
    if (Q != Q)
    {
        return 0.0;
    }
    else

        return Q;
};

dual ThreeVarMirror::Sn_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual S = 0.0;
    dual Spast = 0.0;
    dual Sfus = 0.0;
    dual n = u(0) * n0;
    dual T = u(2) / u(0) * T0;
    dual TeV = T / (e_charge);

    dual R = 1e-6 * 3.68e-12 * pow(TeV / 1000, -2. / 3.) * exp(-19.94 * pow(TeV / 1000, -1. / 3.));
    Sfus = 0.25 * L / (n0 * V0) * R * n * n;
    if (includeParallelLosses)
    {
        dual Rm = Bmax / B(x.val, t);
        dual Xe = Chi_e(u, q, x, t); // phi0(u, q, x, t) * u(0) / u(1) * (1 - 1 / Rm);
                                     // dual Xi = Chi_e(u, q, x, t);
        dual c1 = L / (taue0 * V0);
        dual l1 = PastukhovLoss(u(0), u(1), Xe, Rm);
        // dual c2 = L / (taui0 * V0);
        // dual l2 = PastukhovLoss(u(0), u(2), Xe, Rm);
        Spast = c1 * l1; //+ c2 * l2;
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

dual ThreeVarMirror::Spi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual Ppot = 0;
    dual Pvis = 0;
    dual Ppast = 0.0;

    if (includeParallelLosses)
    {

        dual Rm = Bmax / B(x.val, t);
        dual Xi = Chi_i(u, q, x, t);
        dual Spast = L / (taui0 * V0) * 0.5 * PastukhovLoss(u(0), u(2), Xi, Rm);
        Ppast = u(2) / u(0) * (1 + Xi) * Spast;
    }

    if (useConstantOmega)
    {
        double Rval = R(x.val, t);
        double Vpval = Vprime(Rval);
        double coef = Vpval * Vpval * Rval * Rval * Rval * Rval;

        dual dV = domegadV(x, t);
        dual ghi = ::pow(ionMass / electronMass, 1. / 2.) * 1.0 / (::sqrt(2) * tau_hat(u(0), u(2)) * lambda_hat(u(0), u(1), n0, p0)) * 3. / 10. * u(2);
        Pvis = ghi * dV * dV * coef;

        dual G = -Gamma_hat(u, q, x, t); // sigma(0); //-Gamma_hat(u, q, x, t);
        // sigma(0);
        //  Gamma_hat(u, q, x, t); // sigma(0); // / (coef);
        Ppot = -G * dphi0dV(u, q, x, t) + 0.5 * pow(omega(Rval, t), 2) / M_PI * G;
    }
    dual Pcol = Ci(u(0), u(2), u(1)) * L / (V0 * taue0);
    dual S = 2. / 3. * (Ppot + Pcol + Pvis + Ppast);

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

dual ThreeVarMirror::Spe_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    // double Rval = R(x.val, t);
    // double Vpval = Vprime(Rval);
    // double coef = Vpval * Rval;
    // dual G = Gamma_hat(u, q, x, t); // (coef);
    // dual V = G / u(0);              //* L / (p0);

    // dual S = -2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
    ///*V * q(1)*/ -2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
    // dual Pcol = 2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
    dual Pfus = 0.0;
    dual Pbrem = 0.0;
    dual Ppast = 0.0;
    dual Rm = Bmax / B(x.val, t);
    if (includeParallelLosses)
    {
        dual Xe = Chi_e(u, q, x, t); // phi0(u, q, x, t) * u(0) / u(1) * (1 - 1 / Rm);
        dual Spast = L / (taue0 * V0) * PastukhovLoss(u(0), u(1), Xe, Rm);
        Ppast = u(1) / u(0) * (1 + Xe) * Spast;
    }

    //
    dual n = u(0) * n0;
    dual T = u(1) / u(0) * T0;
    dual TeV = T / (e_charge);
    Pbrem = -2 * 1e6 * 1.69e-32 * (n * n) * 1e-12 * sqrt(TeV); //(-5.34e3 * pow(n / 1e20, 2) * pow(TkeV, 0.5)) * L / (p0 * V0);
    dual TikeV = u(2) / u(0) / 1000 * T0 / e_charge;
    if (TikeV > 25)
        TikeV = 25;

    dual R = 3.68e-12 * pow(TikeV, -2. / 3.) * exp(-19.94 * pow(TikeV, -1. / 3.));
    // 1e-6 * n0 * n0 * R * u(0) * u(0);
    Pfus = 0.25 * sqrt(1 - 1 / Rm) * 1e6 * 5.6e-13 * n * n * 1e-12 * R; // n *n * 5.6e-13

    dual Pcol = Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
    dual S = 2. / 3. * (Pcol + Ppast + L / (p0 * V0) * (Pfus + Pbrem));

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
dual ThreeVarMirror::omega(dual R, double t)
{
    double u_L = omegaOffset.val;
    double u_R = (omegaOffset.val * Rmin / Rmax);
    dual a = (asinh(u_L) - asinh(u_R)) / (Rmin - Rmax);
    dual b = (asinh(u_L) - Rmin / Rmax * asinh(u_R)) / (a * (Rmin / Rmax - 1));

    dual shape = 10.0;
    dual C = 0.5 * (Rmin + Rmax);
    dual c = (M_PI / 2 - 3 * M_PI / 2) / (Rmin - Rmax);
    dual d = (M_PI / 2 - Rmin / Rmax * (3 * M_PI / 2)) / (c * (Rmin / Rmax - 1));
    // dual coef = (omegaOffset - M0 / C) * 1 / cos(c * (C - d));
    dual coef = M0 / C;
    if (omegaOffset == 0.0)
    {
        return omegaOffset - cos(c * (R - d)) * coef * exp(-shape * (R - C) * (R - C));
    }
    else
    {
        return sinh(a * (R - b)) - cos(c * (R - d)) * coef; //* exp(-shape * (R - C) * (R - C));
    }
    //
}

double ThreeVarMirror::domegadV(dual x, double t)
{
    dual Rval = R(x.val, t);
    double Bval = B(x.val, t) / Bmid.val;
    double Vpval = Vprime(Rval.val);
    double domegadR = derivative(omega, wrt(Rval), at(Rval, t));
    return domegadR / (Vpval * Rval.val * Bval);
}

dual ThreeVarMirror::phi0(VectorXdual u, VectorXdual q, dual x, double t)
{
    dual Rval = R(x.val, t);
    dual Rm = Bmax / Bmid.val;
    dual Romega = Rval * omega(Rval, t);
    dual tau = u(2) / u(0);
    dual phi0 = 1 / (1 + tau) * (1 / Rm - 1) * Romega * Romega / 2; // pow(omega(Rval, t), 2) * Rval * Rval / (2 * u(2)) * 1 / (1 / u(2) + 1 / u(1));

    return phi0;
}
dual ThreeVarMirror::dphi0dV(VectorXdual u, VectorXdual q, dual x, double t)
{
    dual Rval = R(x.val, t);
    dual dphi0dV = derivative(phi0, wrt(Rval), at(u, q, x, t)) / (2 * M_PI * Rval);
    auto dphi0du = gradient(phi0, wrt(u), at(u, q, x, t));
    auto qi = q.begin();
    for (auto &dphi0i : dphi0du)
    {
        dphi0dV += *qi * dphi0i;
        ++qi;
    }

    //  dual dphi0dV = (q.val * dphi0du).sum();
    return dphi0dV;
}

dual ThreeVarMirror::Chi_e(VectorXdual u, VectorXdual q, dual x, double t)
{
    dual Rval = R(x.val, t);
    dual Rm = Bmax / B(x.val, t);
    dual tau = u(2) / u(1);
    dual M2 = Rval * Rval * pow(omega(Rval, t), 2) * u(0) / u(1);

    return 0.5 * (1 - 1 / Rm) * M2 * 1 / (tau + 1);
}
dual ThreeVarMirror::Chi_i(VectorXdual u, VectorXdual q, dual x, double t)
{

    dual Rval = R(x.val, t);
    dual Rm = Bmax / B(x.val, t);
    dual tau = u(2) / u(1);
    dual M2 = Rval * Rval * pow(omega(Rval, t), 2) * u(0) / u(1);

    return 0.5 * tau / (1 + tau) * (1 - 1 / Rm) * M2;
}

double ThreeVarMirror::psi(double R)
{
    return double();
}
double ThreeVarMirror::V(double R)
{
    return M_PI * R * R * L;
}
double ThreeVarMirror::Vprime(double R)
{
    double m = -0.9 / (Rmax - Rmin);
    return 2 * M_PI; /// (1 + m * (R - Rmin)); /// exp(-0.5 * R * R);
}
double ThreeVarMirror::B(double x, double t)
{
    double m = -0.9 / (Rmax - Rmin);
    double Rval = R(x, t);
    return Bmid.val; //* (1 + m * (Rval - Rmin)); //* exp(-0.5 * Rval * Rval); // / R(x, t);
}

double ThreeVarMirror::R(double x, double t)
{
    using boost::math::tools::bracket_and_solve_root;
    using boost::math::tools::eps_tolerance;
    double guess = 0.5;                                     // Rough guess is to divide the exponent by three.
    double min = Rmin;                                      // Minimum possible value is half our guess.
    double max = Rmax;                                      // Maximum possible value is twice our guess.
    const int digits = std::numeric_limits<double>::digits; // Maximum possible binary digits accuracy for type T.
    int get_digits = static_cast<int>(digits * 0.6);        // Accuracy doubles with each step, so stop when we have
                                                            // just over half the digits correct.
    double factor = 2;
    bool is_rising = true;
    auto getPair = [](double x, double R)
    { return V(R) - x; }; // change to V(R(psi))

    auto func = std::bind_front(getPair, x);
    eps_tolerance<double> tol(get_digits);

    const boost::uintmax_t maxit = 20;
    boost::uintmax_t it = maxit;
    std::pair<double, double> r = bracket_and_solve_root(func, guess, factor, is_rising, tol, it);
    return r.first + (r.second - r.first) / 2;
};

// double ThreeVarMirror::R(double x, double t)
// {
//     using boost::math::tools::newton_raphson_iterate;
//     double guess = 0.5;                                     // Rough guess is to divide the exponent by three.
//     double min = Rmin;                                      // Minimum possible value is half our guess.
//     double max = Rmax;                                      // Maximum possible value is twice our guess.
//     const int digits = std::numeric_limits<double>::digits; // Maximum possible binary digits accuracy for type T.
//     int get_digits = static_cast<int>(digits * 0.6);        // Accuracy doubles with each step, so stop when we have
//                                                             // just over half the digits correct.

//     auto getPair = [](double x, double R)
//     { return std::pair<double, double>(V(R) - x, 2*M_PI*L/); };

//     auto func = std::bind_front(getPair, x);

//     const boost::uintmax_t maxit = 25;
//     boost::uintmax_t it = maxit;
//     double R = newton_raphson_iterate(func, guess, min, max, get_digits, it);
//     return R;
// };
