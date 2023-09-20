#include "Constants.hpp"

double eV_J(double T_ev) { return T_ev * 1.60218e-19; }
double J_eV(double T_J) { return T_J / 1.60218e-19; }

dual nu(dual n, dual Pe)
{
    return 3.44e-11 * pow(n, 5.0 / 2.0) * lambda / pow(Pe / e_charge, 3 / 2);
}

dual tau_e(dual n, dual Pe)
{
    if (Pe > 0)
        return 3.44e11 * (1.0 / pow(n, 5.0 / 2.0)) * (pow(Pe / e_charge, 3.0 / 2.0)) * 1 / lambda;
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
