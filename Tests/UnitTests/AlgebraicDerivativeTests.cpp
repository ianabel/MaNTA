// The algebraic time derivatives: q', sigma' and phi' at t0, obtained by
// differentiating the constraints that define them.
//
// At t0 IDA leaves those blocks of dydt identically zero -- IDA_YA_YDP_INIT
// computes algebraic *values* and differential *derivatives*, so there is no y'
// for them to fetch -- which at_t0_only_the_differential_part_of_dydt_exists in
// SolverLifecycleTests.cpp pins. This file is about the vector that fills them
// in, and about the solve that produces it.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "SystemSolver.hpp"
#include "TestDiffusion.hpp"
#include "Types.hpp"

#include <nvector/nvector_serial.h>

#include <string>
#include <toml.hpp>
#include <vector>

using namespace toml::literals::toml_literals;

namespace
{
const toml::value alg_config = u8R"(
    [DiffusionProblem]
    Kappa = 1.0
    Centre = 0.0
)"_toml;

constexpr Index k = 2, nCells = 4;

void configure(SystemSolver &sys, std::string const &stem)
{
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile(stem);
    sys.setOutputCadence(0.05);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-12);
    sys.setTolerances({1e-8}, 1e-6);
}

// The Frobenius-ish norm of one field across every cell, for "is this block
// populated at all" questions.
Value blockNorm(DGSoln const &soln, Index nCellsIn, char which)
{
    Value total = 0.0;
    for (Index i = 0; i < nCellsIn; ++i)
    {
        switch (which)
        {
        case 'u':
            total += soln.u(0).getCoeff(i).second.norm();
            break;
        case 'q':
            total += soln.q(0).getCoeff(i).second.norm();
            break;
        case 's':
            total += soln.sigma(0).getCoeff(i).second.norm();
            break;
        }
    }
    return total;
}
} // namespace

BOOST_AUTO_TEST_SUITE(algebraic_derivative_tests)

BOOST_AUTO_TEST_CASE(dydtComplete_starts_as_a_copy_of_idas_derivative)
{
    // Separate storage, seeded from IDA's. The separation is the point: writing
    // the algebraic blocks into IDA's own dYdt would change the state it takes
    // its first step from, and the symptom would be a step-size failure
    // somewhere later rather than anything pointing back here.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(alg_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "algderiv_storage");

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    // The u block is IDA's, and is not zero -- u is differential.
    BOOST_TEST(blockNorm(sys.dydtComplete, nCells, 'u') > 1e-8,
               "dydtComplete's u block is empty, so it was never seeded");

    // And it is genuinely distinct storage.
    BOOST_TEST(static_cast<const void *>(sys.dydtCompleteMem) !=
                   static_cast<const void *>(N_VGetArrayPointer(sys.dYdt)),
               "dydtComplete aliases IDA's dYdt, so writing to it would change the run");

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
}

BOOST_AUTO_TEST_SUITE_END()
