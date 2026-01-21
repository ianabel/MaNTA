class CylindricalMagneticField
{
public:
    CylindricalMagneticField(const std::string &file);
    ~CylindricalMagneticField() = default;

    double Bz_R(double R);
    double Bz_R(autodiff::dual R) { return Bz_R(R.val); };

    double V(double Psi);
    double Psi(double R);
    double Psi_V(double V);
    double VPrime(double V);
    double VPrime(autodiff::dual V) { return VPrime(V.val); };

    double R(double Psi);
    double R_V(double V);
    autodiff::dual R_V(autodiff::dual V)
    {
        autodiff::dual R = R_V(V.val);
        if (V.grad != 0.0)
            R.grad += V.grad * dRdV(V.val);
        return R;
    };

    double dRdV(double V);
    autodiff::dual dRdV(autodiff::dual V);
    double MirrorRatio(double V);
    double MirrorRatio(autodiff::dual V) { return MirrorRatio(V.val); };
    void CheckBoundaries(double VL, double VR);

private:
    double L_z = 1.0;
    double h;

    double R_root_solver(double Psi);

    std::string filename;
    std::vector<double> gridpoints;
    netCDF::NcFile data_file;
    unsigned int nPoints;
    netCDF::NcDim R_dim;
    std::vector<double> R_var;
    std::vector<double> Bz_var;
    std::vector<double> Psi_var;
    std::vector<double> Rm_var;

    std::unique_ptr<spline> B_spline;
    std::unique_ptr<spline> Psi_spline;
    std::unique_ptr<spline> Rm_spline;
    std::unique_ptr<spline> R_Psi_spline;
};