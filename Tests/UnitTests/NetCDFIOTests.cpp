// Round-trip tests for NetCDFIO.
//
// Everything MaNTA writes goes through this class, and most of it is templated
// on a callable sampled over the output grid -- AddVariable, AppendToVariable
// and the two AppendToGroup overloads. Those templates were entirely
// uncovered, which also badly skews the coverage figure (gcov counts a
// templated line once per instantiation, so this 128-line header was reported
// as 551 lines; see Tests/README.md).
//
// StoreGridInfo gets particular attention: reading it back and rebuilding a
// Grid is exactly the restart contract that MaNTA.cpp and PyRunner depend on.

#include <boost/test/unit_test.hpp>

#include "NetCDFIO.hpp"
#include "gridStructures.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(netcdfio_tests, *boost::unit_test::tolerance(1e-12))

namespace
{
// Writes into a uniquely named file and removes it afterwards, so the tests do
// not collide and leave nothing behind.
struct TempNc
{
    std::filesystem::path path;

    explicit TempNc(std::string const &stem)
        : path(std::filesystem::temp_directory_path() /
               ("manta_netcdfio_" + stem + ".nc"))
    {
        std::filesystem::remove(path);
    }
    ~TempNc() { std::filesystem::remove(path); }

    std::string str() const { return path.string(); }
};

std::vector<double> uniformPoints(double a, double b, size_t n)
{
    std::vector<double> pts(n);
    for (size_t i = 0; i < n; ++i)
        pts[i] = a + (b - a) * static_cast<double>(i) / static_cast<double>(n - 1);
    return pts;
}
} // namespace

BOOST_AUTO_TEST_CASE(open_and_close_creates_a_readable_file)
{
    TempNc tmp("open_close");
    {
        NetCDFIO nc;
        nc.Open(tmp.str());
        nc.Close();
    }
    BOOST_TEST(std::filesystem::exists(tmp.path));

    // The time dimension and variable are created by Open().
    netCDF::NcFile f(tmp.str(), netCDF::NcFile::FileMode::read);
    BOOST_TEST(!f.getDim("t").isNull());
    BOOST_TEST(!f.getVar("t").isNull());

    double t0 = -1.0;
    f.getVar("t").getVar({0}, {1}, &t0);
    BOOST_TEST(t0 == 0.0);
}

BOOST_AUTO_TEST_CASE(destructor_closes_an_open_file)
{
    TempNc tmp("dtor");
    {
        NetCDFIO nc;
        nc.Open(tmp.str());
        // No explicit Close(): the destructor must flush and close.
    }
    netCDF::NcFile f(tmp.str(), netCDF::NcFile::FileMode::read);
    BOOST_TEST(!f.getVar("t").isNull());
}

BOOST_AUTO_TEST_CASE(scalar_and_text_variables_round_trip)
{
    TempNc tmp("scalars");
    {
        NetCDFIO nc;
        nc.Open(tmp.str());
        nc.AddScalarVariable("kappa", "diffusivity", "m^2/s", 1.25);
        nc.AddScalarVariable("noUnits", "dimensionless", "", -3.5);
        nc.AddTextVariable("model", "physics model", "", "LinearDiffusion");
        nc.Close();
    }

    netCDF::NcFile f(tmp.str(), netCDF::NcFile::FileMode::read);

    double kappa = 0.0, noUnits = 0.0;
    f.getVar("kappa").getVar(&kappa);
    f.getVar("noUnits").getVar(&noUnits);
    BOOST_TEST(kappa == 1.25);
    BOOST_TEST(noUnits == -3.5);

    // The units attribute is written only when non-empty. getAtt() throws
    // rather than returning a null attribute when the name is absent, so probe
    // the attribute map instead.
    auto kappaAtts = f.getVar("kappa").getAtts();
    BOOST_TEST(kappaAtts.count("units") == 1u);
    std::string units;
    kappaAtts.at("units").getValues(units);
    BOOST_TEST(units == "m^2/s");

    BOOST_TEST(f.getVar("noUnits").getAtts().count("units") == 0u);
}

BOOST_AUTO_TEST_CASE(groups_scalars_and_time_series_round_trip)
{
    TempNc tmp("groups");
    const size_t nT = 4;
    {
        NetCDFIO nc;
        nc.Open(tmp.str());
        nc.SetOutputGrid(uniformPoints(0.0, 1.0, 9));

        nc.AddGroup("Diagnostics", "derived quantities");
        nc.AddScalarVariable("Diagnostics", "Voltage", "terminal voltage", "V", 12.0);
        nc.AddTimeSeries("Energy", "total energy", "J", 100.0);
        nc.AddTimeSeries("Diagnostics", "Current", "total current", "A", 5.0);

        for (size_t i = 1; i < nT; ++i)
        {
            size_t idx = nc.AddTimeSlice(0.5 * i);
            BOOST_TEST(idx == i);
            nc.AppendToTimeSeries("Energy", 100.0 + i, idx);
            nc.AppendToTimeSeries("Diagnostics", "Current", 5.0 + i, idx);
        }
        nc.Close();
    }

    netCDF::NcFile f(tmp.str(), netCDF::NcFile::FileMode::read);

    double voltage = 0.0;
    f.getGroup("Diagnostics").getVar("Voltage").getVar(&voltage);
    BOOST_TEST(voltage == 12.0);

    std::vector<double> energy(nT), current(nT), times(nT);
    f.getVar("Energy").getVar({0}, {nT}, energy.data());
    f.getGroup("Diagnostics").getVar("Current").getVar({0}, {nT}, current.data());
    f.getVar("t").getVar({0}, {nT}, times.data());

    for (size_t i = 0; i < nT; ++i)
    {
        BOOST_TEST(energy[i] == 100.0 + i);
        BOOST_TEST(current[i] == 5.0 + i);
        BOOST_TEST(times[i] == 0.5 * i);
    }
}

BOOST_AUTO_TEST_CASE(create_group_returns_a_usable_handle)
{
    TempNc tmp("creategroup");
    {
        NetCDFIO nc;
        nc.Open(tmp.str());
        netCDF::NcGroup g = nc.CreateGroup("Extra", "a group made directly");
        BOOST_TEST(!g.isNull());
        nc.AddScalarVariable("Extra", "value", "", "", 7.0);
        nc.Close();
    }

    netCDF::NcFile f(tmp.str(), netCDF::NcFile::FileMode::read);
    double v = 0.0;
    f.getGroup("Extra").getVar("value").getVar(&v);
    BOOST_TEST(v == 7.0);
}

// ----------------------------------------------- the templated samplers --

BOOST_AUTO_TEST_CASE(add_variable_samples_the_callable_on_the_output_grid)
{
    TempNc tmp("addvar");
    const auto pts = uniformPoints(0.0, 1.0, 11);
    auto f0 = [](double x) { return std::sin(x) + 2.0; };

    {
        NetCDFIO nc;
        nc.Open(tmp.str());
        nc.SetOutputGrid(pts);
        nc.AddVariable("u", "solution", "K", f0);
        nc.AddGroup("Vars", "grouped variables");
        nc.AddVariable("Vars", "v", "grouped solution", "", f0);
        nc.Close();
    }

    netCDF::NcFile f(tmp.str(), netCDF::NcFile::FileMode::read);

    // The space grid itself must have been stored by SetOutputGrid.
    std::vector<double> x(pts.size());
    f.getVar("x").getVar({0}, {pts.size()}, x.data());
    for (size_t i = 0; i < pts.size(); ++i)
        BOOST_TEST(x[i] == pts[i]);

    std::vector<double> u(pts.size()), v(pts.size());
    f.getVar("u").getVar({0, 0}, {1, pts.size()}, u.data());
    f.getGroup("Vars").getVar("v").getVar({0, 0}, {1, pts.size()}, v.data());

    for (size_t i = 0; i < pts.size(); ++i)
    {
        BOOST_TEST(u[i] == f0(pts[i]));
        BOOST_TEST(v[i] == f0(pts[i]));
    }
}

BOOST_AUTO_TEST_CASE(append_to_variable_writes_successive_timeslices)
{
    TempNc tmp("appendvar");
    const auto pts = uniformPoints(-1.0, 1.0, 7);
    const size_t nT = 3;

    {
        NetCDFIO nc;
        nc.Open(tmp.str());
        nc.SetOutputGrid(pts);
        nc.AddVariable("u", "solution", "", [](double x) { return x; });

        for (size_t i = 1; i < nT; ++i)
        {
            size_t idx = nc.AddTimeSlice(static_cast<double>(i));
            const double scale = 1.0 + i;
            nc.AppendToVariable("u", [scale](double x) { return scale * x; }, idx);
        }
        nc.Close();
    }

    netCDF::NcFile f(tmp.str(), netCDF::NcFile::FileMode::read);
    for (size_t i = 0; i < nT; ++i)
    {
        std::vector<double> u(pts.size());
        f.getVar("u").getVar({i, 0}, {1, pts.size()}, u.data());
        const double scale = (i == 0) ? 1.0 : 1.0 + i;
        for (size_t j = 0; j < pts.size(); ++j)
            BOOST_TEST(u[j] == scale * pts[j]);
    }
}

BOOST_AUTO_TEST_CASE(append_to_group_handles_both_overloads)
{
    TempNc tmp("appendgroup");
    const auto pts = uniformPoints(0.0, 2.0, 5);

    auto fa = [](double x) { return x * x; };
    auto fb = [](double x) { return 1.0 - x; };

    {
        NetCDFIO nc;
        nc.Open(tmp.str());
        nc.SetOutputGrid(pts);
        nc.AddGroup("G", "group");
        nc.AddVariable("G", "a", "", "", fa);
        nc.AddVariable("G", "b", "", "", fb);

        size_t idx = nc.AddTimeSlice(1.0);

        // Single-variable overload...
        nc.AppendToGroup("G", idx, "a", fa);
        // ...and the initializer-list overload.
        using Fn = std::function<double(double)>;
        Fn gb = fb;
        nc.AppendToGroup<Fn>("G", idx, {{"b", gb}});

        nc.Close();
    }

    netCDF::NcFile f(tmp.str(), netCDF::NcFile::FileMode::read);
    std::vector<double> a(pts.size()), b(pts.size());
    f.getGroup("G").getVar("a").getVar({1, 0}, {1, pts.size()}, a.data());
    f.getGroup("G").getVar("b").getVar({1, 0}, {1, pts.size()}, b.data());

    for (size_t i = 0; i < pts.size(); ++i)
    {
        BOOST_TEST(a[i] == fa(pts[i]));
        BOOST_TEST(b[i] == fb(pts[i]));
    }
}

// ------------------------------------------------- the restart contract --

BOOST_AUTO_TEST_CASE(store_grid_info_round_trips_to_an_identical_grid)
{
    // This is what restart depends on: MaNTA.cpp and PyRunner rebuild the Grid
    // from the stored CellBoundaries and must get the same grid back.
    struct Case
    {
        const char *name;
        Grid grid;
    };
    const std::vector<Case> cases{
        {"uniform", Grid(0.0, 1.0, 8)},
        {"shifted", Grid(-2.5, 3.5, 5)},
        {"clustered", Grid(0.0, 1.0, 9, true, 0.2, 0.2)},
    };

    for (auto const &[name, grid] : cases)
    {
        BOOST_TEST_CONTEXT("grid = " << name)
        {
        TempNc tmp("gridinfo");
        const unsigned int k = 4;
        {
            NetCDFIO nc;
            nc.Open(tmp.str());
            nc.StoreGridInfo(grid, k);
            nc.Close();
        }

        netCDF::NcFile f(tmp.str(), netCDF::NcFile::FileMode::read);
        netCDF::NcGroup g = f.getGroup("Grid");
        BOOST_TEST(!g.isNull());

        const size_t nPoints = g.getDim("Index").getSize();
        BOOST_TEST(nPoints == grid.getNCells() + 1);

        std::vector<double> boundaries(nPoints);
        g.getVar("CellBoundaries").getVar({0}, {nPoints}, boundaries.data());

        unsigned int order = 0;
        g.getVar("PolyOrder").getVar(&order);
        BOOST_TEST(order == k);

        Grid rebuilt(boundaries);
        bool identical = (rebuilt == grid);
        BOOST_TEST(identical);
        if (!identical)
            for (Grid::Index c = 0; c < grid.getNCells(); ++c)
                BOOST_TEST_MESSAGE("  cell " << c << ": orig [" << grid[c].x_l << ", "
                                             << grid[c].x_u << "] rebuilt ["
                                             << rebuilt[c].x_l << ", " << rebuilt[c].x_u
                                             << "]");
        }
    }
}

BOOST_AUTO_TEST_CASE(store_grid_info_writes_a_contiguous_index)
{
    TempNc tmp("gridindex");
    Grid grid(0.0, 1.0, 6);
    {
        NetCDFIO nc;
        nc.Open(tmp.str());
        nc.StoreGridInfo(grid, 2);
        nc.Close();
    }

    netCDF::NcFile f(tmp.str(), netCDF::NcFile::FileMode::read);
    netCDF::NcGroup g = f.getGroup("Grid");
    const size_t n = g.getDim("Index").getSize();

    std::vector<int> idx(n);
    g.getVar("Index").getVar({0}, {n}, idx.data());
    for (size_t i = 0; i < n; ++i)
        BOOST_TEST(idx[i] == static_cast<int>(i));
}

BOOST_AUTO_TEST_SUITE_END()
