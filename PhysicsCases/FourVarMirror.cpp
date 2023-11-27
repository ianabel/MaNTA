#include "FourVarMirror.hpp"
#include "Constants.hpp"
#include <iostream>
#include <boost/math/tools/roots.hpp>

REGISTER_FLUX_IMPL(FourVarMirror);

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

FourVarMirror::FourVarMirror(toml::value const &config, Index nVars)
{
    if (config.count("4VarMirror") != 1)
        throw std::invalid_argument("There should be a [4VarMirror] section if you are using the 4VarMirror physics model.");

    auto const &DiffConfig = config.at("4VarMirror");

    std::string profile = toml::find_or(DiffConfig, "SourceType", "None");
    ParticleSource = ParticleSources[profile];

    sourceStrength = toml::find_or(DiffConfig, "sourceStrength", 0.0);
    sourceCenter = toml::find_or(DiffConfig, "SourceCenter", 0.25);
    sourceWidth = toml::find_or(DiffConfig, "SourceWidth", 0.01);

    Rmin = toml::find_or(DiffConfig, "Rmin", 0.1);
    Rmax = toml::find_or(DiffConfig, "Rmax", 1.0);

    // reference values
    n0 = toml::find_or(DiffConfig, "n0", 3e19);
    T0 = e_charge * toml::find_or(DiffConfig, "T0", 1e3);
    Bmid = toml::find_or(DiffConfig, "Bmid", 1.0);
    //   E0 = toml::find_or(DiffConfig, "E0", 1e5);
    L = toml::find_or(DiffConfig, "L", 1.0);
    J0 = toml::find_or(DiffConfig, "J0", 0.01);

    p0 = n0 * T0;

    Gamma0 = (p0 / L) / (electronMass * Om_e(Bmid) * Om_e(Bmid) * tau_e(n0, p0));
    V0 = Gamma0 / n0;
    omega0 = 1 / L * sqrt(T0 / ionMass);
    taue0 = tau_e(n0, p0);
    taui0 = tau_i(n0, p0);
    h0 = ionMass * n0 * L * L * omega0;

    sigma.insert(std::pair<Index, sigmaptr>(0, &Gamma_hat));
    sigma.insert(std::pair<Index, sigmaptr>(1, &qe_hat));
    sigma.insert(std::pair<Index, sigmaptr>(2, &qi_hat));
    sigma.insert(std::pair<Index, sigmaptr>(3, &hi_hat));

    source.insert(std::pair<Index, sourceptr>(0, &Sn_hat));
    source.insert(std::pair<Index, sourceptr>(1, &Spe_hat));
    source.insert(std::pair<Index, sourceptr>(2, &Spi_hat));
    source.insert(std::pair<Index, sourceptr>(3, &Shi_hat));
};

int FourVarMirror::ParticleSource;
double FourVarMirror::sourceStrength;
dual FourVarMirror::sourceCenter;
dual FourVarMirror::sourceWidth;

// reference values
dual FourVarMirror::n0;
dual FourVarMirror::T0;
dual FourVarMirror::Bmid;
dual FourVarMirror::J0;
Value FourVarMirror::L;

dual FourVarMirror::omega0;
dual FourVarMirror::E0;
dual FourVarMirror::p0;
dual FourVarMirror::Gamma0;
dual FourVarMirror::V0;
dual FourVarMirror::taue0;
dual FourVarMirror::taui0;
dual FourVarMirror::h0;

double FourVarMirror::Rmin;
double FourVarMirror::Rmax;

dual FourVarMirror::Gamma_hat(VectorXdual u, VectorXdual q, dual x, double t)
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

dual FourVarMirror::hi_hat(VectorXdual u, VectorXdual q, dual x, double t)
{
    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    dual G = Gamma_hat(u, q, x, t);
    dual ghi = ::pow(ionMass / electronMass, 1. / 2.) * 1.0 / (::sqrt(2) * tau_hat(u(0), u(2))) * 3. / 10. * u(3) * u(2) / u(1) * (q(3) / u(3) - q(0) / u(0) - 1 / (M_PI * Rval * Rval));
    dual H = u(3) * G / u(0) + coef * ghi;
    if (H != H)
    {
        return 0.0;
    }
    else
    {
        return H;
    }
};
dual FourVarMirror::qi_hat(VectorXdual u, VectorXdual q, dual x, double t)
{

    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    // dual G = Gamma_hat(u, q, x, t);
    dual qri = ::sqrt(ionMass / (2 * electronMass)) * 1.0 / tau_hat(u(0), u(2)) * 2. * u(2) * u(2) / u(0) * (q(2) / u(2) - q(0) / u(0));
    dual Q = (2. / 3.) * (coef * qri); // + 5. / 2. * u(2) / u(0) * G);
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
dual FourVarMirror::qe_hat(VectorXdual u, VectorXdual q, dual x, double t)
{

    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    // dual G = Gamma_hat(u, q, x, t);
    dual qre = 1.0 / tau_hat(u(0), u(1)) * (4.66 * u(1) * u(1) / u(0) * (q(1) / u(1) - q(0) / u(0)) - (3. / 2.) * u(1) / u(0) * (q(2) + q(1)));

    dual Q = (2. / 3.) * (coef * qre); // + 5. / 2. * u(1) / u(0) * G + coef * qre);
    if (Q != Q)
    {
        return 0.0;
    }
    else

        return Q;
};

dual FourVarMirror::Sn_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual S = 0.0;
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
    return S;
};

dual FourVarMirror::Shi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    double Rval = R(x.val, t);
    dual coef = L / (h0 * V0);
    dual S = tanh(10 * t) * (J0 / Rval) * coef * B(x.val, t) * Rval * Rval;
    return S; // + u(3) / (u(0) * Rval * Rval) * Sn_hat(u, q, sigma, x, t);
};

// look at ion and electron sources again -- they should be opposite
dual FourVarMirror::Spi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double Bval = B(x.val, t);
    double coef = Rval * Rval * Vpval;

    dual dV = coef * u(3) / u(0) * (q(3) / u(3) - q(0) / u(0) - 1 / (M_PI * Rval * Rval));
    dual ghi = ::pow(ionMass / electronMass, 1. / 2.) * 1.0 / (::sqrt(2) * tau_hat(u(0), u(2))) * 3. / 10. * u(2);
    dual Pvis = ghi * dV * dV;

    dual G = -Gamma_hat(u, q, x, t); // / (coef);
    dual Ppot = -G * dphi0dV(u, q, x, t) + u(3) * u(3) / (pow(Rval, 4) * u(0) * u(0) * M_PI) * G;
    // u(3) * u(3) / (Rval * u(0) * u(0)) * G
    dual Pcol = Ci(u(0), u(2), u(1)) * L / (V0 * taue0);

    // dual Ppot = -0.5 * u(3) * u(3) / (Rval * Rval * u(0) * u(0)) * Sn_hat(u, q, sigma, x, t);
    //  dual S = -2. / 3. * Ci(u(0), u(2), u(1)) * L / (V0 * taue0) + 2. / 3. * Svis + Ppot;
    ///*V * q(2)*/ -2. / 3. * Ci(u(0), u(2), u(1)) * L / (V0 * taue0); //+ 2. / 3. * Svis + Ppot;
    // dual S = 2. / 3. * Ci(u(0), u(2), u(1)) * L / (V0 * taue0);
    dual S = 2. / 3. * (Ppot + Pcol + Pvis);

    if (S != S)
    {
        return 0.0;
    }
    else
    {
        return S; //- tanh(100 * t) * 1000 * Sn_hat(u, q, sigma, x, t);
    }
    // return 0.0;
}

dual FourVarMirror::Spe_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    // double Rval = R(x.val, t);
    // double Vpval = Vprime(Rval);
    // double coef = Vpval * Rval;
    // dual G = Gamma_hat(u, q, x, t); // (coef);
    // dual V = G / u(0);              //* L / (p0);

    // dual S = -2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
    ///*V * q(1)*/ -2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
    dual Pcol = 2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
    dual S = Pcol;

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

dual FourVarMirror::phi0(VectorXdual u, VectorXdual q, dual x, double t)
{
    double Rval = R(x.val, t);
    dual phi0 = u(3) * u(3) / (u(2) * u(0) * u(0) * Rval * Rval) * 1 / (1 / u(2) + 1 / u(1));
    return phi0;
}

dual FourVarMirror::dphi0dV(VectorXdual u, VectorXdual q, dual x, double t)
{
    dual Rval = R(x.val, t);
    dual dphi0dV = -phi0(u, q, x, t) / (M_PI * Rval * Rval);
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

double FourVarMirror::psi(double R)
{
    return double();
}
double FourVarMirror::V(double R)
{
    return M_PI * R * R * L;
}
double FourVarMirror::Vprime(double R)
{
    return 2 * M_PI / ((1 - 1.1 * R));
}
double FourVarMirror::B(double x, double t)
{
    return Bmid.val * (1 - 1.1 * R(x, t)); // / R(x, t);
}

double FourVarMirror::R(double x, double t)
{
    // return sqrt(x / (M_PI * L));
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
    { return V(R) - x; }; // change to V(psi(R))

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
