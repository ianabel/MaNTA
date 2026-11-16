// The equilibrium-model registry mirrors the field-model one and PhysicsCases::map:
// a duplicate name throws rather than the first registration quietly winning,
// and an unknown name throws with the list of what *is* registered.
//
// Then the registered native model, ThetaPinchPressureBalance, on its own: its
// table, and its discretisation of the balance against the closed form, slot by
// slot -- which is where its one subtlety lives, the recovered pressure gradient.
//
// The map is never reset, so every test here uses a throwaway name.
#include <boost/test/unit_test.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../../EquilibriumModel.hpp"
#include "../../PhysicsCases/ThetaPinchPressureBalance.hpp"
#include "../../gridStructures.hpp"

#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/tools/roots.hpp>
#include <toml.hpp>

#include <cmath>
#include <format>
#include <string>
#include <vector>

namespace
{
class RegistryProbeEquilibrium : public EquilibriumModel
{
public:
    RegistryProbeEquilibrium(toml::value const &, Grid const &)
        : EquilibriumModel({.geometry = {{"g", "probe", "1"}}, .label = "x"})
    {
    }
    std::unique_ptr<Equilibrium> Solve(GlobalState const &, std::vector<Position> const &, Vector const &, Time,
                                       Equilibrium const *) override
    {
        return std::make_unique<Equilibrium>();
    }
    void Geometry(MatrixRef g, Equilibrium const &, std::vector<Position> const &) override { g.setZero(); }
};

using Balance = ThetaPinchPressureBalance;
constexpr double kPi = M_PI;

/// The closed form for p(x) = T (1 + cos(pi x) / 2): C from the wall, then the
/// slots, as ThetaPinchMMSTests' PinchExact derives them.
struct ExactBalance
{
    static constexpr double T = 0.45;
    static double p(double x) { return T * (1.0 + 0.5 * std::cos(kPi * x)); }
    static double px(double x) { return -0.5 * T * kPi * std::sin(kPi * x); }

    double C = 0.0;
    ExactBalance()
    {
        using GK = boost::math::quadrature::gauss_kronrod<double, 61>;
        auto wall = [](double c)
        { return GK::integrate([&](double x) { return kPi / std::sqrt(c - p(x)); }, 0.0, 1.0, 5, 1e-15) - kPi; };
        boost::math::tools::eps_tolerance<double> tol(52);
        std::uintmax_t it = 200;
        const auto r = boost::math::tools::toms748_solve(wall, p(0.0) * (1.0 + 1e-12), 10.0, tol, it);
        C = 0.5 * (r.first + r.second);
    }
    double Vp(double x) const { return kPi / std::sqrt(C - p(x)); }
    Vector slots(double x) const
    {
        using GK = boost::math::quadrature::gauss_kronrod<double, 61>;
        const double A = x > 0.0 ? GK::integrate([&](double s) { return Vp(s); }, 0.0, x, 5, 1e-15) : 0.0;
        Vector g(4);
        g(Balance::Vprime) = Vp(x);
        g(Balance::dVprime_dx) = 0.5 * kPi * px(x) * std::pow(C - p(x), -1.5);
        g(Balance::gradx2) = 4.0 * kPi * A / (Vp(x) * Vp(x));
        g(Balance::Bfield) = kPi / Vp(x);
        return g;
    }
};

/// Chebyshev nodes of the first kind in every cell, as the transport's basis
/// places them, and a one-variable, one-aux state carrying p at them.
struct Nodes
{
    std::vector<Position> x;
    GlobalState states;
    Nodes(Grid const &grid, Index k) : states(static_cast<Index>(grid.getNCells()), k, 1, 0, 1)
    {
        for (Index i = 0; i < static_cast<Index>(grid.getNCells()); ++i)
            for (Index j = 0; j <= k; ++j)
            {
                const double s = -std::cos(kPi * (2.0 * j + 1.0) / (2.0 * (k + 1)));
                x.push_back(grid[i].x_l + 0.5 * (s + 1.0) * (grid[i].x_u - grid[i].x_l));
            }
        for (size_t j = 0; j < x.size(); ++j)
            states.Aux()(0, static_cast<Eigen::Index>(j)) = ExactBalance::p(x[j]);
    }
};
} // namespace

BOOST_AUTO_TEST_SUITE(equilibrium_registry_tests)

BOOST_AUTO_TEST_CASE(a_registered_model_can_be_instantiated_by_name)
{
    EquilibriumModels::RegisterEquilibriumModel("RegistryProbeEquilibriumA",
                                                createEquilibriumModel<RegistryProbeEquilibrium>);
    Grid grid(0.0, 1.0, 4);
    toml::value config;
    auto model = EquilibriumModels::InstantiateEquilibriumModel("RegistryProbeEquilibriumA", config, grid);
    BOOST_REQUIRE(model != nullptr);
    BOOST_CHECK_EQUAL(model->nGeometry(), 1);
}

BOOST_AUTO_TEST_CASE(a_duplicate_name_throws)
{
    EquilibriumModels::RegisterEquilibriumModel("RegistryProbeEquilibriumB",
                                                createEquilibriumModel<RegistryProbeEquilibrium>);
    // std::invalid_argument, as the other two registries throw.
    BOOST_CHECK_THROW(EquilibriumModels::RegisterEquilibriumModel("RegistryProbeEquilibriumB",
                                                                  createEquilibriumModel<RegistryProbeEquilibrium>),
                      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(an_unknown_name_throws_and_names_what_is_registered)
{
    EquilibriumModels::RegisterEquilibriumModel("RegistryProbeEquilibriumC",
                                                createEquilibriumModel<RegistryProbeEquilibrium>);
    Grid grid(0.0, 1.0, 4);
    toml::value config;
    try
    {
        EquilibriumModels::InstantiateEquilibriumModel("NoSuchEquilibriumModel", config, grid);
        BOOST_FAIL("expected InstantiateEquilibriumModel to throw");
    }
    catch (std::invalid_argument const &e)
    {
        std::string const msg = e.what();
        BOOST_CHECK(msg.find("NoSuchEquilibriumModel") != std::string::npos);
        BOOST_CHECK(msg.find("RegistryProbeEquilibriumC") != std::string::npos);
        // The registered native model is listed too: it is linked in.
        BOOST_CHECK(msg.find("ThetaPinchPressureBalance") != std::string::npos);
    }
}

BOOST_AUTO_TEST_CASE(the_native_theta_pinch_is_registered_and_reads_its_own_table)
{
    using namespace toml::literals::toml_literals;
    Grid grid(0.0, 1.0, 5);
    const toml::value config = u8R"(
        [ThetaPinchPressureBalance]
        PressureAux = 0
        WallRadius = 2.0
        DifferencedResponse = true
    )"_toml;
    auto model = EquilibriumModels::InstantiateEquilibriumModel("ThetaPinchPressureBalance", config, grid);
    BOOST_TEST(model->nGeometry() == 4);
    BOOST_TEST((model->getSpec().response == EquilibriumResponse::Differenced));
    BOOST_TEST(model->getSpec().name == "ThetaPinch");
    BOOST_TEST((EquilibriumModels::InstantiateEquilibriumModel("ThetaPinchPressureBalance", toml::value{}, grid)
                    ->getSpec()
                    .response == EquilibriumResponse::Model));

    BOOST_CHECK_THROW(Balance(Balance::Parameters{}, Grid(0.0, 2.0, 4)), std::invalid_argument);
    BOOST_CHECK_THROW(Balance(Balance::Parameters{.wallRadius = -1.0}, grid), std::invalid_argument);
    const toml::value misspelt = u8R"(
        [ThetaPinchPressureBalance]
        Radius = 2.0
    )"_toml;
    BOOST_CHECK_THROW(EquilibriumModels::InstantiateEquilibriumModel("ThetaPinchPressureBalance", misspelt, grid),
                      std::invalid_argument);
}

// The balance on an exact pressure, against the closed form. C satisfies the
// wall to round-off at every mesh, and every slot converges at k + 1 or better:
// V', <|grad x|^2> and B at the interpolant's k + 1, dV'/dx at the recovered
// gradient's. That last is what the recovery is for. Read off p_h itself,
// dV'/dx goes at k; fitted to p_h's values over a patch, it goes at k only at
// k = 1, but the error p_h repeats in every cell leaks into the fit's slope
// there; fitted to cell means, which carry that error only as a smooth function,
// it goes at k + 1 at every k (k + 2 at k = 3, its fit being degree k + 1 on
// five cells). Measured, max-norm orders 32 -> 64 -> 128 cells:
//
//     k = 1   V' 2.00, 2.00   dV'/dx 2.00, 2.00
//     k = 2   V' 4.00, 4.00   dV'/dx 2.98, 2.99
//     k = 3   V' 4.00, 4.00   dV'/dx 5.00, 5.00
BOOST_AUTO_TEST_CASE(the_native_balance_converges_slot_by_slot)
{
    const ExactBalance exact;
    for (Index k : {1, 2, 3})
    {
        std::vector<double> errs[5];
        const std::vector<Index> cells = {32, 64, 128};
        for (Index n : cells)
        {
            Grid grid(0.0, 1.0, n);
            Nodes nodes(grid, k);
            Balance model(Balance::Parameters{.wallRadius = 1.0, .vacuumField = 1.0}, grid);
            const auto e = model.Solve(nodes.states, nodes.x, Vector::Ones(nodes.x.size()), 0.0, nullptr);
            Matrix g = Matrix::Zero(4, static_cast<Eigen::Index>(nodes.x.size()));
            model.Geometry(g, *e, nodes.x);
            double worst[4] = {0, 0, 0, 0};
            for (size_t j = 0; j < nodes.x.size(); ++j)
            {
                const Vector ge = exact.slots(nodes.x[j]);
                for (int r = 0; r < 4; ++r)
                    worst[r] = std::max(worst[r], std::abs(g(r, static_cast<Eigen::Index>(j)) - ge(r)));
            }
            for (int r = 0; r < 4; ++r)
                errs[r].push_back(worst[r]);
            errs[4].push_back(std::abs(Balance::totalPressure(*e) - exact.C));

            // The wall: the model's own C satisfies its own constraint.
            BOOST_TEST(Balance::totalPressure(*e) > ExactBalance::p(0.0));
        }
        for (size_t i = 1; i < cells.size(); ++i)
        {
            auto order = [&](int r) { return std::log2(errs[r][i - 1] / errs[r][i]); };
            BOOST_TEST_MESSAGE(std::format("k = {}, n = {}: orders V' {:.2f} dV'/dx {:.2f} gradx2 {:.2f} B {:.2f} "
                                           "C {:.2f}",
                                           k, cells[i], order(0), order(1), order(2), order(3), order(4)));
            for (int r = 0; r < 5; ++r)
                BOOST_TEST(order(r) > k + 1 - 0.1,
                           "k = " << k << ", n = " << cells[i] << ", slot " << r << ": order " << order(r));
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
