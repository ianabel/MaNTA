#include "MagneticFields.hpp"
#include <boost/math/tools/roots.hpp>

CurvedMagneticField::CurvedMagneticField(const std::string &file)
{
    // filename = file;
    filename = "/home/eatocco/projects/MaNTA/PhysicsCases/Bfield.nc";
    data_file.open(filename, netCDF::NcFile::FileMode::read);

    R_dim = data_file.getDim("R");
    nPoints = R_dim.getSize();

    R_var.reserve(nPoints);
    Bz_var.reserve(nPoints);
    Psi_var.reserve(nPoints);
    Rm_var.reserve(nPoints);

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

    std::vector<double> R_Psi;
    R_Psi.reserve(nPoints);

    R_Psi.push_back(*(R_var.begin()));

    for (auto &Psi : std::vector<double>(Psi_var.begin() + 1, Psi_var.end() - 1))
    {
        R_Psi.push_back(R_root_solver(Psi));
    }

    R_Psi.push_back(R_var.back());

    R_Psi_spline = std::make_unique<spline>(R_Psi.begin(), R_Psi.end(), R_var[0], h);
}

double CurvedMagneticField::Bz_R(double R)
{
    return (*B_spline)(R);
}

double CurvedMagneticField::V(double Psi)
{
    double R_Psi = R(Psi);
    return L_z * pi * R_Psi * R_Psi;
}

double CurvedMagneticField::Psi(double R)
{
    return (*Psi_spline)(R);
}

double CurvedMagneticField::Psi_V(double V)
{
    return Psi(R_V(V));
}

double CurvedMagneticField::VPrime(double V)
{
    return 2 * pi * L_z / Bz_R(R_V(V));
}

double CurvedMagneticField::R(double Psi)
{
    return (*R_Psi_spline)(Psi);
}

double CurvedMagneticField::R_V(double V)
{
    return sqrt(V / (pi * L_z));
}

// dRdV = dPsidR*dVdPsi
double CurvedMagneticField::dRdV(double V)
{
    return 1 / Psi_spline->prime(R_V(V)) * 1 / VPrime(V);
}

double CurvedMagneticField::MirrorRatio(double V)
{
    return (*Rm_spline)(R_V(V));
}

double CurvedMagneticField::R_root_solver(double Psi)
{
    // return sqrt(x / (M_PI * L));
    using boost::math::tools::bisect;
    using boost::math::tools::eps_tolerance;
    const int digits = std::numeric_limits<double>::digits; // Maximum possible binary digits accuracy for type T.
    int get_digits = static_cast<int>(digits * 0.6);        // Accuracy doubles with each step, so stop when we have
                                                            // just over half the digits correct.
    auto getPair = [this](double x, double R)
    { return this->Psi(R) - x; };

    auto func = std::bind_front(getPair, Psi);
    eps_tolerance<double> tol(get_digits);

    const boost::uintmax_t maxit = 20;
    boost::uintmax_t it = maxit;
    std::pair<double, double> r = bisect(func, *R_var.begin(), R_var.back(), tol, it);
    return r.first + (r.second - r.first) / 2;
}
