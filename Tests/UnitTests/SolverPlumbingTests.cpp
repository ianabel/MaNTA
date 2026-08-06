// Tests for the small pieces of solver plumbing: the SystemSolver setters,
// ErrorChecker, and the two SUNDIALS shims.
//
// None of this is glamorous, but the setters are the only validation standing
// between a bad config and a nonsense run, and SunMatrixWrapper exists purely
// to convince IDA it has a matrix-based solver -- if its ops stop behaving,
// IDA's behaviour changes silently.

#include <boost/test/unit_test.hpp>

#include "ErrorChecker.hpp"
#include "SunLinSolWrapper.hpp"
#include "SunMatrixWrapper.hpp"
#include "SystemSolver.hpp"
#include "TestDiffusion.hpp"
#include "Types.hpp"

#include <sundials/sundials_context.h>

#include <stdexcept>
#include <toml.hpp>

using namespace toml::literals::toml_literals;

namespace
{
const toml::value plumbing_config = u8R"(
    [DiffusionProblem]
    Kappa = 1.0
    Centre = 0.0
)"_toml;
} // namespace

BOOST_AUTO_TEST_SUITE(solver_plumbing_tests)

// ------------------------------------------------------------- setters --

BOOST_AUTO_TEST_CASE(setters_reject_invalid_values)
{
    Grid grid(0.0, 1.0, 4);
    TestDiffusion problem(plumbing_config);
    SystemSolver sys(grid, 2, &problem);

    // Output cadence may be zero but not negative.
    BOOST_CHECK_NO_THROW(sys.setOutputCadence(0.0));
    BOOST_CHECK_NO_THROW(sys.setOutputCadence(0.25));
    BOOST_CHECK_THROW(sys.setOutputCadence(-1e-12), std::logic_error);

    // Steady-state tolerance must be strictly positive.
    BOOST_CHECK_NO_THROW(sys.setSteadyStateTolerance(1e-3));
    BOOST_CHECK_THROW(sys.setSteadyStateTolerance(0.0), std::logic_error);
    BOOST_CHECK_THROW(sys.setSteadyStateTolerance(-1.0), std::logic_error);

    // Number of output points must be strictly positive.
    BOOST_CHECK_NO_THROW(sys.setNOutput(1));
    BOOST_CHECK_THROW(sys.setNOutput(0), std::logic_error);
    BOOST_CHECK_THROW(sys.setNOutput(-5), std::logic_error);

    // Minimum step size must be strictly positive.
    BOOST_CHECK_NO_THROW(sys.setMinStepSize(1e-9));
    BOOST_CHECK_THROW(sys.setMinStepSize(0.0), std::logic_error);
    BOOST_CHECK_THROW(sys.setMinStepSize(-1e-9), std::logic_error);

    // Relative tolerance must be strictly positive; the absolute tolerance
    // vector is not validated.
    BOOST_CHECK_NO_THROW(sys.setTolerances({1e-4}, 1e-4));
    BOOST_CHECK_THROW(sys.setTolerances({1e-4}, 0.0), std::logic_error);
    BOOST_CHECK_THROW(sys.setTolerances({1e-4}, -1e-4), std::logic_error);
}

BOOST_AUTO_TEST_CASE(setters_that_do_not_validate_still_round_trip)
{
    Grid grid(0.0, 1.0, 4);
    TestDiffusion problem(plumbing_config);
    SystemSolver sys(grid, 2, &problem);

    sys.setOutputCadence(0.375);
    BOOST_TEST(sys.getdt() == 0.375);

    BOOST_CHECK_NO_THROW(sys.setInitialTimestep(1e-6));
    BOOST_CHECK_NO_THROW(sys.setInitialTime(-2.0));
    BOOST_CHECK_NO_THROW(sys.setTau(0.5));
    BOOST_CHECK_NO_THROW(sys.setAlpha(1.5));
    BOOST_CHECK_NO_THROW(sys.setJacTime(0.25));
    BOOST_CHECK_NO_THROW(sys.setTime(0.25));
    BOOST_CHECK_NO_THROW(sys.setZeroFlux(true));
    BOOST_CHECK_NO_THROW(sys.setSolveAdjoint(false));
    BOOST_CHECK_NO_THROW(sys.setInputFile("somewhere.conf"));

    sys.setTesting(true);
    BOOST_TEST(sys.isTesting());
    sys.setTesting(false);
    BOOST_TEST(!sys.isTesting());
}

// -------------------------------------------------------- ErrorChecker --

BOOST_AUTO_TEST_CASE(check_retval_flags_null_pointers_for_opt_0_and_2)
{
    int ok = 0;

    // opt 0: SUNDIALS allocator returning NULL is an error.
    BOOST_TEST(ErrorChecker::check_retval(nullptr, "alloc", 0) == 1);
    BOOST_TEST(ErrorChecker::check_retval(&ok, "alloc", 0) == 0);

    // opt 2: same check, different message.
    BOOST_TEST(ErrorChecker::check_retval(nullptr, "alloc", 2) == 1);
    BOOST_TEST(ErrorChecker::check_retval(&ok, "alloc", 2) == 0);
}

BOOST_AUTO_TEST_CASE(check_retval_flags_negative_flags_for_opt_1)
{
    int negative = -1, zero = 0, positive = 7;

    BOOST_TEST(ErrorChecker::check_retval(&negative, "call", 1) == 1);
    BOOST_TEST(ErrorChecker::check_retval(&zero, "call", 1) == 0);
    BOOST_TEST(ErrorChecker::check_retval(&positive, "call", 1) == 0);
}

BOOST_AUTO_TEST_CASE(check_retval_passes_anything_for_an_unknown_opt)
{
    // Only 0, 1 and 2 are defined; anything else falls through to success.
    BOOST_TEST(ErrorChecker::check_retval(nullptr, "call", 3) == 0);
}

// --------------------------------------------------- the SUNDIALS shims --

BOOST_AUTO_TEST_CASE(sun_matrix_wrapper_is_a_well_behaved_empty_matrix)
{
    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    SUNMatrix mat = SunMatrixNew(ctx);
    BOOST_TEST_REQUIRE(mat != nullptr);

    // It must claim to be a custom matrix so IDA does not try to interpret its
    // (nonexistent) contents.
    BOOST_TEST(MatGetID(mat) == SUNMATRIX_CUSTOM);
    BOOST_TEST(SUNMatGetID(mat) == SUNMATRIX_CUSTOM);

    // Zeroing an empty matrix succeeds and does nothing.
    BOOST_TEST(MatZero(mat) == 0);
    BOOST_TEST(SUNMatZero(mat) == 0);

    BOOST_CHECK_NO_THROW(MatDestroy(mat));
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_CASE(sun_lin_sol_wrapper_reports_the_right_type_and_id)
{
    Grid grid(0.0, 1.0, 4);
    TestDiffusion problem(plumbing_config);
    SystemSolver sys(grid, 2, &problem);
    sys.setTau(0.5);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    SUNLinearSolver LS = SunLinSolWrapper::SunLinSol(&sys, nullptr, ctx);
    BOOST_TEST_REQUIRE(LS != nullptr);

    // IDA branches on these. DIRECT (not iterative) is what makes IDA treat
    // the solve as exact and skip its own iteration; CUSTOM stops it assuming
    // a known internal layout. Changing either alters IDA's strategy silently.
    BOOST_TEST(SUNLinSolGetType(LS) == SUNLINEARSOLVER_DIRECT);
    BOOST_TEST(SUNLinSolGetID(LS) == SUNLINEARSOLVER_CUSTOM);

    // Initialise is a no-op that must report success; setup forwards to
    // SystemSolver and must too.
    BOOST_TEST(SUNLinSolInitialize(LS) == SUN_SUCCESS);

    SUNMatrix mat = SunMatrixNew(ctx);
    BOOST_TEST(SUNLinSolSetup(LS, mat) == SUN_SUCCESS);

    SUNLinSolFree(LS);
    MatDestroy(mat);
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_SUITE_END()
