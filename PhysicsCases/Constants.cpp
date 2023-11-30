#include "Constants.hpp"

double eV_J(double T_ev) { return T_ev * 1.60218e-19; }
double J_eV(double T_J) { return T_J / 1.60218e-19; }

dual lambda(dual n, dual Pe)
{
    dual Te = Pe / n * 1 / e_charge;
    return 24 - log(sqrt(n * 1e-6) / Te); // 18.4 - 1.15 * log10(n) + 2.3 * log10(1 / e_charge * Pe / n);
}

dual lambda_hat(dual nhat, dual Pehat, dual n0, dual Pe0)
{
    return lambda(n0, Pe0) / lambda(nhat * n0, Pehat * Pe0);
}

dual nu(dual n, dual Pe)
{
    return 3.44e-11 * pow(n, 5.0 / 2.0) * lambda(n, Pe) / pow(Pe / e_charge, 3 / 2);
}

dual tau_e(dual n, dual Pe)
{
    if (Pe > 0)
        return 3.44e11 * (1.0 / pow(n, 5.0 / 2.0)) * (pow(Pe / e_charge, 3.0 / 2.0)) * 1 / lambda(n, Pe);
    else
        return (1.0 / n);
}

dual tau_i(dual n, dual Pi)
{
    if (Pi > 0)

        return ::sqrt(2.) * tau_e(n, Pi) * (::sqrt(ionMass / electronMass));
    //  return ::sqrt(2) * (1.0 / pow(n, 5.0 / 2.0)) * (pow(Pi, 3.0 / 2.0));

    else
    {

        return ::sqrt(2) * 1.0 / (n); // if we have a negative temp just treat it as 1eV
    }
}

dual Om_i(dual B)
{
    return e_charge * B / ionMass;
}

dual Om_e(dual B)
{
    return e_charge * B / electronMass;
}

dual tau_hat(dual n, dual P)
{
    if (P > 0)
        return (1.0 / pow(n, 5.0 / 2.0)) * (pow(P, 3.0 / 2.0));
    else
        return 1.0 / n;
}

dual Ce(dual n, dual Pi, dual Pe)
{
    dual c = 3 * electronMass / ionMass;
    return c * (Pi - Pe) / tau_hat(n, Pe);
}

dual Ci(dual n, dual Pi, dual Pe)
{
    return -Ce(n, Pi, Pe);
}

dual RT(dual n, dual Pe)
{
    return dual();
}

dual RDT(dual n, dual Pe)
{
    dual T = Pe / n;
    return 1e-6 * 3.68e-12 * pow(T, -2. / 3.) * exp(-19.94 * pow(T, -1. / 3.));
}

dual PastukhovLoss(dual n, dual P, dual Xs, dual Rm)
{
    double Sigma = 2;
    return -2 * n * Sigma / sqrt(M_PI) * 1 / tau_hat(n, P) * 1 / log(Sigma * Rm) * exp(-Xs) / Xs;
}