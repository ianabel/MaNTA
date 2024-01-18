#include "MagneticFields.hpp"

CylindricalMagneticField::CylindricalMagneticField(const std::string &file)
{
    filename = file;
    data_file.open(file, netCDF::NcFile::FileMode::read);

    R_dim = data_file.getDim("R");
    nPoints = R_dim.getSize();
    double tmp[nPoints];

    // copy nc data into vectors
    data_file.getVar("R").getVar(tmp);
    R_var.insert(R_var.end(), tmp[0], tmp[nPoints]);

    data_file.getVar("Bz").getVar(tmp);
    Bz_var.insert(Bz_var.end(), tmp[0], tmp[nPoints]);

    data_file.getVar("Psi").getVar(tmp);
    Psi_var.insert(Psi_var.end(), tmp[0], tmp[nPoints]);

    data_file.getVar("Rm").getVar(tmp);
    Rm_var.insert(Rm_var.end(), tmp[0], tmp[nPoints]);

    h = R_var[1] - R_var[0];
    delete tmp;
}

CylindricalMagneticField::~CylindricalMagneticField()
{
    data_file.close();
}

double CylindricalMagneticField::Bz_R(double R)
{
    spline B_spline(Bz_var.begin(), Bz_var.end(), R_var[0], h);
    return B_spline(R);
}

double CylindricalMagneticField::V(double Psi)
{
    return 0.0;
}

double CylindricalMagneticField::Psi(double R)
{
    spline Psi_spline(Psi_var.begin(), Psi_var.end(), R_var[0], h);
    return Psi_spline(R);
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
    return 0.0;
}

double CylindricalMagneticField::R_V(double V)
{
    return sqrt(V / (pi * L_z));
}

double CylindricalMagneticField::MirrorRatio(double V)
{
    spline Rm_spline(Rm_var.begin(), Rm_var.end(), R_var[0], h);
    return Rm_spline(R_V(V));
}
