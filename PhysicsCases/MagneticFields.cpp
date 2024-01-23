#include "MagneticFields.hpp"
#include <boost/math/tools/roots.hpp>

CylindricalMagneticField::CylindricalMagneticField(const std::string &file)
{
    filename = file;
    data_file.open(filename, netCDF::NcFile::FileMode::read);

    R_dim = data_file.getDim("R");
    nPoints = R_dim.getSize();

    double *temp = new double[nPoints];

    // copy nc data into vectors
    data_file.getVar("R").getVar(temp);
    R_var.insert(R_var.end(), &temp[0], &temp[nPoints]);

    data_file.getVar("Bz").getVar(temp);
    Bz_var.insert(Bz_var.end(), &temp[0], &temp[nPoints]);

    data_file.getVar("Psi").getVar(temp);
    Psi_var.insert(Psi_var.end(), &temp[0], &temp[nPoints]);

    data_file.getVar("Rm").getVar(temp);
    Rm_var.insert(Rm_var.end(), &temp[0], &temp[nPoints]);

    data_file.close();
    delete[] temp;

    // create spline interperlant objects

    h = R_var[1] - R_var[0];

    B_spline = std::make_unique<spline>(Bz_var.begin(), Bz_var.end(), R_var[0], h);
    Psi_spline = std::make_unique<spline>(Psi_var.begin(), Psi_var.end(), R_var[0], h);
    Rm_spline = std::make_unique<spline>(Rm_var.begin(), Rm_var.end(), R_var[0], h);
}

double CylindricalMagneticField::Bz_R(double R)
{
    return (*B_spline)(R);
}

double CylindricalMagneticField::V(double Psi)
{
    double R_Psi = R(Psi);
    return L_z * pi * R_Psi * R_Psi;
}

double CylindricalMagneticField::Psi(double R)
{
    return (*Psi_spline)(R);
}

double CylindricalMagneticField::Psi_V(double V)
{
    return Psi(R_V(V));
}

double CylindricalMagneticField::VPrime(double V)
{
    return 2 * pi * L_z / Bz_R(R_V(V));
}

double CylindricalMagneticField::R(double Psi)
{
    // return sqrt(x / (M_PI * L));
    using boost::math::tools::bracket_and_solve_root;
    using boost::math::tools::eps_tolerance;
    double guess = R_var[nPoints / 2]; // Rough guess is to divide the exponent by three.
    // double min = Rmin;                                      // Minimum possible value is half our guess.
    // double max = Rmax;                                      // Maximum possible value is twice our guess.
    const int digits = std::numeric_limits<double>::digits; // Maximum possible binary digits accuracy for type T.
    int get_digits = static_cast<int>(digits * 0.6);        // Accuracy doubles with each step, so stop when we have
                                                            // just over half the digits correct.
    double factor = 2;
    bool is_rising = true;
    auto getPair = [this](double x, double R)
    { return this->Psi(R) - x; };

    auto func = std::bind_front(getPair, Psi);
    eps_tolerance<double> tol(get_digits);

    const boost::uintmax_t maxit = 20;
    boost::uintmax_t it = maxit;
    std::pair<double, double> r = bracket_and_solve_root(func, guess, factor, is_rising, tol, it);
    return r.first + (r.second - r.first) / 2;
}

double CylindricalMagneticField::R_V(double V)
{
    return sqrt(V / (pi * L_z));
}

// dRdV = dPsidR*dVdPsi
double CylindricalMagneticField::dRdV(double V)
{
    return 1 / Psi_spline->prime(R_V(V)) * 1 / VPrime(V);
}

double CylindricalMagneticField::MirrorRatio(double V)
{
    return (*Rm_spline)(R_V(V));
}
