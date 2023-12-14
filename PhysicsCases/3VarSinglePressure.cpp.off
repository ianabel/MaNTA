#include "3VarSinglePressure.hpp"
#include "Constants.hpp"
#include <iostream>
#include <boost/math/tools/roots.hpp>

REGISTER_FLUX_IMPL(ThreeVarSinglePressure);

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

ThreeVarSinglePressure ::ThreeVarSinglePressure(toml::value const &config, Index nVars)
{
    if (config.count("3VarSinglePressure") != 1)
        throw std::invalid_argument("There should be a [3VarSinglePressure] section if you are using the 3VarSinglePressure physics model.");

    auto const &DiffConfig = config.at("3VarSinglePressure");

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
    sigma.insert(std::pair<Index, sigmaptr>(1, &q_hat));
    sigma.insert(std::pair<Index, sigmaptr>(2, &hi_hat));

    source.insert(std::pair<Index, sourceptr>(0, &Sn_hat));
    source.insert(std::pair<Index, sourceptr>(1, &Sp_hat));
    source.insert(std::pair<Index, sourceptr>(2, &Shi_hat));
};

int ThreeVarSinglePressure ::ParticleSource;
double ThreeVarSinglePressure ::sourceStrength;
dual ThreeVarSinglePressure ::sourceCenter;
dual ThreeVarSinglePressure ::sourceWidth;

// reference values
dual ThreeVarSinglePressure ::n0;
dual ThreeVarSinglePressure ::T0;
dual ThreeVarSinglePressure ::Bmid;
dual ThreeVarSinglePressure ::J0;
Value ThreeVarSinglePressure ::L;

dual ThreeVarSinglePressure ::omega0;
dual ThreeVarSinglePressure ::E0;
dual ThreeVarSinglePressure ::p0;
dual ThreeVarSinglePressure ::Gamma0;
dual ThreeVarSinglePressure ::V0;
dual ThreeVarSinglePressure ::taue0;
dual ThreeVarSinglePressure ::taui0;
dual ThreeVarSinglePressure ::h0;

double ThreeVarSinglePressure ::Rmin;
double ThreeVarSinglePressure ::Rmax;

dual ThreeVarSinglePressure ::Gamma_hat(VectorXdual u, VectorXdual q, dual x, double t)
{

    // maybe add a factor of sqrt x if x = r^2/2
    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    dual G = coef * u(1) / tau_hat(u(0), u(1)) * ((q(1) / 2.) / u(1) + 3. / 2. * q(0) / u(0));

    if (G != G)
    {
        return 0.0;
    }

    else
        return G;
};

dual ThreeVarSinglePressure ::hi_hat(VectorXdual u, VectorXdual q, dual x, double t)
{
    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    dual G = Gamma_hat(u, q, x, t);
    dual ghi = ::pow(ionMass / electronMass, 1. / 2.) * 1.0 / (::sqrt(2) * tau_hat(u(0), u(1))) * 3. / 10. * u(2) * u(2) / u(1) * (q(2) / u(2) - q(0) / u(0) - 1 / (M_PI * Rval * Rval));
    dual H = u(2) * G / u(0) + coef * ghi;
    if (H != H)
    {
        return 0.0;
    }
    else
    {
        return H;
    }
};
dual ThreeVarSinglePressure ::q_hat(VectorXdual u, VectorXdual q, dual x, double t)
{

    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    // dual G = Gamma_hat(u, q, x, t);
    dual qri = ::sqrt(ionMass / (2 * electronMass)) * 1.0 / tau_hat(u(0), u(1)) * 2. * u(1) * u(1) / u(0) * (q(1) / u(1) - q(0) / u(0));
    dual Q = (1. / 3.) * (coef * qri); // + 5. / 2. * u(2) / u(0) * G);
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
// dual ThreeVarSinglePressure ::qe_hat(VectorXdual u, VectorXdual q, dual x, double t)
// {

//     double Rval = R(x.val, t);
//     double Vpval = Vprime(Rval);
//     double coef = Rval * Rval * Vpval * Vpval;
//     // dual G = Gamma_hat(u, q, x, t);
//     dual qre = 1.0 / tau_hat(u(0), u(1)) * (4.66 * u(1) * u(1) / u(0) * (q(1) / u(1) - q(0) / u(0)) - (3. / 2.) * u(1) / u(0) * (q(2) + q(1)));

//     dual Q = (2. / 3.) * (coef * qre); // + 5. / 2. * u(1) / u(0) * G + coef * qre);
//     if (Q != Q)
//     {
//         return 0.0;
//     }
//     else

//         return Q;
// };

dual ThreeVarSinglePressure ::Sn_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
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

dual ThreeVarSinglePressure ::Shi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    double Rval = R(x.val, t);
    dual coef = L / (h0 * V0);
    dual S = tanh(3 * t) * (J0 / Rval) * coef * B(x.val, t) * Rval * Rval;
    return S; // + u(3) / (u(0) * Rval * Rval) * Sn_hat(u, q, sigma, x, t);
};

// look at ion and electron sources again -- they should be opposite
dual ThreeVarSinglePressure ::Sp_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double Bval = B(x.val, t);
    double coef = Rval * Rval * Vpval * Vpval;

    dual dV = u(2) / u(0) * (q(2) / u(2) - q(0) / u(0) - 1 / (M_PI * Rval * Rval));
    dual ghi = coef * ::pow(ionMass / electronMass, 1. / 2.) * 1.0 / (::sqrt(2) * tau_hat(u(0), u(1))) * 3. / 10. * u(2) * u(1) / u(1) * (q(2) / u(2) - q(0) / u(0) - 1 / (M_PI * Rval * Rval));
    dual Pvis = ghi * coef / Vpval * dV;

    dual G = -Gamma_hat(u, q, x, t); // / (coef);
    dual Ppot = -G * Vpval * dphi0dV(u, q, x, t) + u(2) * u(2) / (Rval * Rval * Rval * u(0) * u(0) * Bval) * G;
    // dual Ppot = 0;
    // Ci(u(0), u(2), u(1)) * L / (V0 * taue0);
    // // dual Ppot = -0.5 * u(3) * u(3) / (Rval * Rval * u(0) * u(0)) * Sn_hat(u, q, sigma, x, t);
    // //  dual S = -2. / 3. * Ci(u(0), u(2), u(1)) * L / (V0 * taue0) + 2. / 3. * Svis + Ppot;
    // ///*V * q(2)*/ -2. / 3. * Ci(u(0), u(2), u(1)) * L / (V0 * taue0); //+ 2. / 3. * Svis + Ppot;
    // // dual S = 2. / 3. * Ci(u(0), u(2), u(1)) * L / (V0 * taue0);
    dual S = 1. / 3. * (Ppot + Pvis);

    if (S != S)
    {
        return 0.0;
    }
    else
    {
        return S + u(1) / u(0) * Sn_hat(u, q, sigma, x, t);
    }
    // return 0.0;
}

// dual ThreeVarSinglePressure ::Spe_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
// {
//     // double Rval = R(x.val, t);
//     // double Vpval = Vprime(Rval);
//     // double coef = Vpval * Rval;
//     // dual G = Gamma_hat(u, q, x, t); // (coef);
//     // dual V = G / u(0);              //* L / (p0);

//     // dual S = -2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
//     ///*V * q(1)*/ -2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
//     dual Pcol = 2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
//     dual S = Pcol;

//     if (S != S)
//     {
//         return 0.0;
//     }
//     else
//     {
//         return S + u(1) / u(0) * Sn_hat(u, q, sigma, x, t);
//     }
//     // return 0.0;
// };

dual ThreeVarSinglePressure ::phi0(VectorXdual u, VectorXdual q, dual x, double t)
{
    double Rval = R(x.val, t);
    dual phi0 = u(2) * u(2) / (u(1) * u(0) * Rval * Rval) * 1 / (2 / u(1));
    return phi0;
}

dual ThreeVarSinglePressure ::dphi0dV(VectorXdual u, VectorXdual q, dual x, double t)
{
    dual dphi0dV = 0;
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

double ThreeVarSinglePressure ::psi(double R)
{
    return double();
}
double ThreeVarSinglePressure ::V(double R)
{
    return M_PI * R * R * L;
}
double ThreeVarSinglePressure ::Vprime(double R)
{
    return 2 * M_PI / ((1 - 0.9 * R));
}
double ThreeVarSinglePressure ::B(double x, double t)
{
    return Bmid.val * (1 - 0.9 * R(x, t)); // / R(x, t);
}

double ThreeVarSinglePressure ::R(double x, double t)
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
