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

    // reference values
    n0 = toml::find_or(DiffConfig, "n0", 3e19);
    T0 = e_charge * toml::find_or(DiffConfig, "T0", 1e3);
    Bmid = toml::find_or(DiffConfig, "Bmid", 1.0);
    L = toml::find_or(DiffConfig, "L", 1.0);
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

dual ThreeVarMirror::p0;
dual ThreeVarMirror::Gamma0;
dual ThreeVarMirror::V0;
dual ThreeVarMirror::taue0;
dual ThreeVarMirror::taui0;

double ThreeVarMirror::Rmin;
double ThreeVarMirror::Rmax;

dual ThreeVarMirror::Gamma_hat(VectorXdual u, VectorXdual q, dual x, double t)
{

    // maybe add a factor of sqrt x if x = r^2/2
    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    dual G = coef * u(1) / tau_hat(u(0), u(1)) * ((-q(1) / 2. + q(2)) / u(1) + 3. / 2. * q(0) / u(0));

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
    dual qri = ::sqrt(ionMass / (2. * electronMass)) * 1.0 / tau_hat(u(0), u(2)) * 2. * u(2) * u(2) / u(0) * dT;
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
    dual qre = 1.0 / tau_hat(u(0), u(1)) * (4.66 * u(1) * u(1) / u(0) * (q(1) / u(1) - q(0) / u(0)) - (3. / 2.) * u(1) / u(0) * (q(2) + q(1)));

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
    switch (ParticleSource)
    {
    case None:
        break;
    case Gaussian:
        S = -sourceStrength * exp(-1 / sourceWidth * (x - sourceCenter) * (x - sourceCenter));
        break;
    default:
        break;
    }
    return S;
};

// look at ion and electron sources again -- they should be opposite
dual ThreeVarMirror::Spi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    // double Rval = R(x.val, t);
    // double Vpval = Vprime(Rval);
    // double coef = Vpval * Rval;
    // dual G = Gamma_hat(u, q, x, t); // / (coef);
    // dual V = G / u(0);              //* L / (p0);
    dual S = 0.0; //-2. / 3. * Ci(u(0), u(2), u(1)) * L / (V0 * taue0);

    if (S != S)
    {
        return 0.0;
    }
    else
    {
        return S + (1.1 + u(2) / u(0)) * Sn_hat(u, q, sigma, x, t);
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

    dual S = 0.0;
    //-2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);

    if (S != S)
    {
        return 0.0;
    }
    else
    {
        return S + u(1) / u(0) * Sn_hat(u, q, sigma, x, t);
    }
    // return 0.0;
};

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
    return 2 * M_PI; /// ((1 - 0.5 * R));
}
double ThreeVarMirror::B(double x, double t)
{
    return Bmid.val; //* (1 - 0.5 * R(x, t)); // / R(x, t);
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
