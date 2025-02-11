#include "MagneticFields.hpp"
#include <boost/math/tools/roots.hpp>
#include <iostream>
#include <filesystem>
#include <string>

CylindricalMagneticField::CylindricalMagneticField(const std::string &file)
{
    filename = file;
    try
    {
        data_file.open(filename, netCDF::NcFile::FileMode::read);
    }
    catch (...)
    {
        std::string msg = "Failed to open netCDF file at: " + std::string(std::filesystem::absolute(std::filesystem::path(filename)));
        throw std::runtime_error(msg);
    }

    V_dim = data_file.getDim("V");
    nPoints = V_dim.getSize();

    V.resize(nPoints);
    data_file.getVar("V").getVar(V.data());

    std::vector<double> temp(nPoints);

    data_file.getVar("Bz").getVar(temp.data());
    B_spline = std::make_unique<spline>(temp.data(), V.data(), nPoints);

    data_file.getVar("VPrime").getVar(temp.data());
    Vp_spline = std::make_unique<spline>(temp.data(), V.data(), nPoints);

    data_file.getVar("Psi").getVar(temp.data());
    Psi_spline = std::make_unique<spline>(temp.data(), V.data(), nPoints);

    data_file.getVar("Rm").getVar(temp.data());
    Rm_spline = std::make_unique<spline>(temp.data(), V.data(), nPoints);

    data_file.getVar("R").getVar(temp.data());
    R_V_spline = std::make_unique<spline>(temp.data(), V.data(), nPoints);

    data_file.getVar("dRdV").getVar(temp.data());
    dRdV_spline = std::make_unique<spline>(temp.data(), V.data(), nPoints);

    data_file.getVar("L").getVar(temp.data());
    L_spline = std::make_unique<spline>(temp.data(), V.data(), nPoints);
}

double CylindricalMagneticField::B(double V)
{
    return (*B_spline)(V);
}

double CylindricalMagneticField::Psi_V(double V)
{
    return (*Psi_spline)(V);
}

double CylindricalMagneticField::VPrime(double V)
{
    return (*Vp_spline)(V);
}

double CylindricalMagneticField::R_V(double V)
{
    return (*R_V_spline)(V);
}

double CylindricalMagneticField::dRdV(double V)
{
    return (*dRdV_spline)(V);
}

double CylindricalMagneticField::MirrorRatio(double V) const
{
    return (*Rm_spline)(V);
}

void CylindricalMagneticField::CheckBoundaries(double VL, double VR)
{
    if ((VL < V.front()) || (VR > V.back()))
        throw std::runtime_error("Magnetic field file must include entire domain");
}

// double CylindricalMagneticField::R_root_solver(double Psi)
// {
//     // return sqrt(x / (M_PI * L));
//     using boost::math::tools::bisect;
//     using boost::math::tools::eps_tolerance;
//     const int digits = std::numeric_limits<double>::digits; // Maximum possible binary digits accuracy for type T.
//     int get_digits = static_cast<int>(digits * 0.6);        // Accuracy doubles with each step, so stop when we have
//                                                             // just over half the digits correct.
//     auto getPair = [this](double x, double R)
//     { return this->Psi(R) - x; };

//     auto func = std::bind_front(getPair, Psi);
//     eps_tolerance<double> tol(get_digits);

//     const boost::uintmax_t maxit = 20;
//     boost::uintmax_t it = maxit;
//     std::pair<double, double> r = bisect(func, *R_var.begin(), R_var.back(), tol, it);
//     return r.first + (r.second - r.first) / 2;
// }
