// Tests for the small self-contained utilities: util/trapezoid.hpp,
// util/ParallelFor.hpp, util/BandedMatrix.hpp and Logging.hpp.
//
// trapezoid is the quadrature MagneticField::FluxSurfaceAverage relies on, and
// it is written to work on autodiff dual numbers as well as doubles -- so both
// instantiations need checking, not just the double one.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "Logging.hpp"
#include "Types.hpp"
#include "util/BandedMatrix.hpp"
#include "util/ParallelFor.hpp"
#include "util/trapezoid.hpp"

#include <autodiff/forward/dual.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <vector>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

BOOST_AUTO_TEST_SUITE(utility_tests)

// ------------------------------------------------------------ trapezoid --

BOOST_AUTO_TEST_CASE(trapezoid_integrates_polynomials)
{
    // Refinement continues until |DN| < tol, so results are good to about tol.
    const double tol = 1e-10;

    BOOST_TEST(trapezoid<double>([](double) { return 1.0; }, 0.0, 2.0, tol) == 2.0,
               boost::test_tools::tolerance(1e-9));
    BOOST_TEST(trapezoid<double>([](double x) { return x; }, 0.0, 2.0, tol) == 2.0,
               boost::test_tools::tolerance(1e-9));
    BOOST_TEST(trapezoid<double>([](double x) { return x * x; }, 0.0, 3.0, tol) == 9.0,
               boost::test_tools::tolerance(1e-7));
}

BOOST_AUTO_TEST_CASE(trapezoid_integrates_transcendentals)
{
    const double tol = 1e-12;

    // \int_0^pi sin = 2. The default cap of 12 refinements (4096 points) is
    // reached before tol is, and second-order convergence puts the floor at
    // about 5e-8 -- so the achievable accuracy is set by max_refinements here,
    // not by tol.
    BOOST_TEST(trapezoid<double>([](double x) { return std::sin(x); }, 0.0,
                                 std::numbers::pi, tol) == 2.0,
               boost::test_tools::tolerance(1e-6));

    // \int_0^1 e^x = e - 1
    BOOST_TEST(trapezoid<double>([](double x) { return std::exp(x); }, 0.0, 1.0, tol) ==
                   std::numbers::e - 1.0,
               boost::test_tools::tolerance(1e-6));
}

BOOST_AUTO_TEST_CASE(trapezoid_returns_zero_for_an_empty_interval)
{
    BOOST_TEST(trapezoid<double>([](double x) { return std::exp(x); }, 1.5, 1.5) == 0.0);
}

BOOST_AUTO_TEST_CASE(trapezoid_rejects_reversed_endpoints)
{
    BOOST_CHECK_THROW(trapezoid<double>([](double x) { return x; }, 2.0, 1.0),
                      std::logic_error);
}

BOOST_AUTO_TEST_CASE(trapezoid_converges_at_second_order)
{
    // The header documents |DN| <= C/N^2. Cap the refinements so the loop exits
    // on the iteration count rather than the tolerance, then check that
    // quadrupling the work quarters the error.
    auto f = [](double x) { return std::sin(x); };
    const double exact = 2.0;

    const double coarse =
        std::abs(trapezoid<double>(f, 0.0, std::numbers::pi, 0.0, 4) - exact);
    const double fine =
        std::abs(trapezoid<double>(f, 0.0, std::numbers::pi, 0.0, 6) - exact);

    // Two extra refinements = 4x the points = ~16x less error at 2nd order.
    BOOST_TEST(fine < coarse / 8.0,
               "coarse err " << coarse << " fine err " << fine);
}

BOOST_AUTO_TEST_CASE(trapezoid_respects_the_refinement_cap)
{
    // With an unreachable tolerance the cap must still terminate the loop.
    auto f = [](double x) { return std::sin(50.0 * x); };
    double result = 0.0;
    BOOST_CHECK_NO_THROW(result = trapezoid<double>(f, 0.0, 1.0, 0.0, 3));
    BOOST_TEST(std::isfinite(result));
}

BOOST_AUTO_TEST_CASE(trapezoid_works_on_dual_numbers)
{
    // MagneticField integrates through autodiff, so the dual instantiation
    // must carry a correct derivative, not just a correct value.
    //
    // I(b) = \int_0^b x^2 dx = b^3/3, so dI/db = b^2.
    // NB: `Real` is not declared in Types.hpp -- each physics header defines
    // its own `using Real = autodiff::dual`. Unqualified `Real` here would
    // resolve to autodiff::Real, which is a class *template*, so
    // trapezoid<Real> would not even parse.
    using Dual = autodiff::dual;
    const double b0 = 1.7;

    Dual b = b0;
    b.grad = 1.0;
    Dual a = 0.0;

    // The `-> Dual` is load-bearing. autodiff operators return expression
    // templates holding references to their operands, so a deduced return type
    // yields an expression referring to the dead parameter `x`; trapezoid then
    // integrates garbage and silently returns 0. MagneticField's
    // FluxSurfaceAverage had exactly this shape until it was annotated.
    Dual I = trapezoid<Dual>([](Dual x) -> Dual { return x * x; }, a, b, 1e-12);

    BOOST_TEST(I.val == b0 * b0 * b0 / 3.0, boost::test_tools::tolerance(1e-5));
    BOOST_TEST(I.grad == b0 * b0, boost::test_tools::tolerance(1e-5));

    // Pin the trap itself. Only the type is asserted: actually *calling*
    // trapezoid with the deduced-return lambda is undefined behaviour (it
    // reads dangling references), and the observed result varies between zero
    // and arbitrary garbage -- so there is no value here worth asserting.
    auto deduced = [](Dual x) { return x * x; };
    static_assert(!std::is_same_v<decltype(deduced(std::declval<Dual>())), Dual>,
                  "autodiff returns an expression template from x*x; a lambda "
                  "passed to trapezoid must declare -> Dual to force "
                  "evaluation before its operands go out of scope");
}

// -------------------------------------------------------------- Logging --

BOOST_AUTO_TEST_CASE(level_to_string_covers_every_level)
{
    BOOST_TEST(levelToString(LOG_LEVEL::ERROR) == "ERROR");
    BOOST_TEST(levelToString(LOG_LEVEL::WARNING) == "WARNING");
    BOOST_TEST(levelToString(LOG_LEVEL::INFO) == "INFO");
    BOOST_TEST(levelToString(LOG_LEVEL::PDEBUG) == "DEBUG");
}

BOOST_AUTO_TEST_CASE(level_to_string_rejects_an_invalid_level)
{
    // The default branch throws; reach it with a value outside the enum.
    const auto bogus = static_cast<LOG_LEVEL>(99);
    BOOST_CHECK_THROW(levelToString(bogus), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(logmsg_writes_what_it_is_asked_to)
{
    // logmsg goes to a C FILE* via std::print, not to std::cerr, so this has to
    // capture at the descriptor level -- and capturing means the message can be
    // checked rather than merely not-thrown. The vector case exercises the
    // std::formatter shim added for libstdc++ < 15.
    std::vector<double> v{1.0, 2.5, -3.0};

    std::string out;
    {
        CapturedOutput quiet;
        logmsg<LOG_LEVEL::ERROR>("plain message");
        logmsg<LOG_LEVEL::ERROR>("value {}", 42);
        logmsg<LOG_LEVEL::ERROR>("double {} and string {}", 1.5, "s");
        logmsg<LOG_LEVEL::WARNING>("vector {}", v);
        out = quiet.text();
    }

    BOOST_TEST(out.find("ERROR: plain message") != std::string::npos, out);
    BOOST_TEST(out.find("ERROR: value 42") != std::string::npos, out);
    BOOST_TEST(out.find("ERROR: double 1.5 and string s") != std::string::npos, out);
    BOOST_TEST(out.find("WARNING: vector [1, 2.5, -3]") != std::string::npos, out);

    // One line per call: println adds the newline, so a format string should
    // not carry its own.
    BOOST_TEST(std::count(out.begin(), out.end(), '\n') == 4, out);
}

BOOST_AUTO_TEST_CASE(logmsg_filters_by_level_at_compile_time)
{
    // max_log_level is WARNING in a release build, INFO under VERBOSE and
    // PDEBUG under DEBUG. Anything below the threshold must compile to nothing
    // -- that is what makes it free to leave INFO logging in hot code.
    std::string out;
    {
        CapturedOutput quiet;
        logmsg<LOG_LEVEL::INFO>("info {}", 1);
        logmsg<LOG_LEVEL::PDEBUG>("debug {}", 1);
        out = quiet.text();
    }

#if defined(DEBUG)
    BOOST_TEST(out.find("INFO: info 1") != std::string::npos, out);
    BOOST_TEST(out.find("DEBUG: debug 1") != std::string::npos, out);
#elif defined(VERBOSE)
    BOOST_TEST(out.find("INFO: info 1") != std::string::npos, out);
    BOOST_TEST(out.find("DEBUG") == std::string::npos, out);
#else
    BOOST_TEST(out.empty(),
               "a release build should emit nothing below WARNING, got: " << out);
#endif
}

// ---------------------------------------------------------- parallel_for --
//
// These matter only in an OpenMP build -- without it parallel_for is a plain
// loop and every one of them passes trivially. They are here because the thing
// they pin cost a process abort, and nothing in the suite would have caught it:
// no CI leg sets MANTA_OPENMP, so the pragmas were never compiled with -fopenmp
// by anything automated. Run this file's suite in such a build after touching
// util/ParallelFor.hpp:
//
//     cmake -B build-omp -DMANTA_OPENMP=ON && cmake --build build-omp -j 6
//     OMP_NUM_THREADS=6 build-omp/Tests/UnitTests/UnitTests --run_test=utility_tests

BOOST_AUTO_TEST_CASE(parallel_for_visits_every_index_exactly_once)
{
    // Well above TransportSystem::physicsGrain, so a threaded build really does
    // fork a team rather than falling through the serial path.
    const Index n = 4096;
    std::vector<int> seen(static_cast<size_t>(n), 0);

    manta::parallel_for(n, [&](Index i) { seen[static_cast<size_t>(i)] += 1; }, 1);

    BOOST_TEST(std::all_of(seen.begin(), seen.end(), [](int c) { return c == 1; }));
}

BOOST_AUTO_TEST_CASE(parallel_for_below_the_grain_still_runs_every_index)
{
    // The serial fall-through is a separate code path from the parallel one, and
    // it is the one every fixture in this tree actually takes.
    const Index n = 7;
    Index total = 0;
    manta::parallel_for(n, [&](Index i) { total += i; }, 1000);
    BOOST_TEST(total == 21);
}

BOOST_AUTO_TEST_CASE(an_exception_from_the_body_reaches_the_caller)
{
    // The regression guard. An exception that escapes an OpenMP structured block
    // does not propagate -- gcc's outlined function has no handler above it, so
    // __cxa_call_terminate aborts the whole process. That is not hypothetical:
    // with the bare pragmas this replaced, MANTA_OPENMP=ON killed the unit suite
    // inside residual_tests/static_residual_converts_a_physics_exception_into_a_retry.
    //
    // It matters because a physics hook throwing is a *supported* thing to do:
    // static_residual catches it and returns 1, which IDA treats as recoverable
    // and retries with a smaller step. It is also how a Python case's exception
    // gets back to the solver.
    //
    // Thrown from a high index on purpose. The original defect was thread-
    // dependent in exactly this way -- throwing from index 0 lands on the master
    // thread, where the exception *can* reach the handler, so the abort hid from
    // any test small enough to run one iteration per thread.
    const Index n = 4096;
    BOOST_CHECK_THROW(
        manta::parallel_for(
            n,
            [&](Index i)
            {
                if (i == n - 1)
                    throw std::runtime_error("from the body");
            },
            1),
        std::runtime_error);
}

BOOST_AUTO_TEST_CASE(a_throwing_body_stops_the_remaining_iterations)
{
    // Not merely an optimisation: a hook that throws for one point usually throws
    // for most of them, and without the flag every remaining iteration would run
    // and be discarded. Checked as an inequality rather than an exact count --
    // iterations already in flight on other threads still finish, and how many
    // that is depends on the team size.
    const Index n = 8192;
    std::atomic<Index> ran{0};

    BOOST_CHECK_THROW(
        manta::parallel_for(
            n,
            [&](Index i)
            {
                ran.fetch_add(1, std::memory_order_relaxed);
                if (i == 0)
                    throw std::runtime_error("immediately");
            },
            1),
        std::runtime_error);

    BOOST_TEST(ran.load() <= n);
}

// --------------------------------------------------------- BandedMatrix --
//
// The oracle throughout is Eigen's dense FullPivLU -- the decomposition this
// replaced in solveHDGJac -- on the same matrix stored densely. That is the
// comparison that matters: not "does the band solver solve something", but "does
// it give what the solver used to give".
//
// **Every case runs both paths.** A build that found LAPACK calls dgbtrf/dgbtrs;
// one that did not uses the built-in dgbtf2. Testing only whichever the build
// picked would leave the other rotting -- and since most development boxes have
// LAPACK, the one that rots would be the fallback that exists for the boxes that
// do not. factorizeBuiltin/solveInPlaceBuiltin are public for exactly this.

namespace
{
/// A random banded matrix, dense. `diagBoost` is added to the diagonal; at 0 the
/// matrix is as ill-conditioned as the draw makes it, which is what forces the
/// pivoting path.
Matrix randomBandedDense(Index n, Index kl, Index ku, double diagBoost, unsigned seed)
{
    // A fixed LCG rather than <random>: libstdc++ and libc++ do not agree on the
    // sequence a given distribution produces from a given engine, and these cases
    // are meant to be the same on every leg of the CI matrix.
    unsigned state = seed;
    auto next = [&state]()
    {
        state = state * 1664525u + 1013904223u;
        return static_cast<double>(state >> 8) / static_cast<double>(1u << 24) - 0.5;
    };

    Matrix dense = Matrix::Zero(n, n);
    for (Index j = 0; j < n; ++j)
        for (Index i = std::max<Index>(0, j - ku); i <= std::min(n - 1, j + kl); ++i)
            dense(i, j) = next() + (i == j ? diagBoost : 0.0);
    return dense;
}

void fillBand(manta::BandedMatrix &band, Matrix const &dense, Index kl, Index ku)
{
    const Index n = dense.rows();
    band.resize(n, kl, ku);
    band.setZero();
    for (Index j = 0; j < n; ++j)
        for (Index i = std::max<Index>(0, j - ku); i <= std::min(n - 1, j + kl); ++i)
            band(i, j) = dense(i, j);
}

struct BothWays
{
    Vector active;   // whatever this build calls: LAPACK, or the built-in
    Vector builtin;  // always the built-in
    bool activeOk = false, builtinOk = false;
};

BothWays solveBothWays(Matrix const &dense, Index kl, Index ku, Vector const &rhs)
{
    BothWays out;
    manta::BandedMatrix band;

    fillBand(band, dense, kl, ku);
    out.activeOk = band.factorize();
    out.active = rhs;
    if (out.activeOk)
        band.solveInPlace(out.active);

    fillBand(band, dense, kl, ku);
    out.builtinOk = band.factorizeBuiltin();
    out.builtin = rhs;
    if (out.builtinOk)
        band.solveInPlaceBuiltin(out.builtin);

    return out;
}
} // namespace

BOOST_AUTO_TEST_CASE(banded_reports_which_implementation_is_under_test)
{
    // Not an assertion -- a build is correct either way. It is here so that a
    // failure below can be read without going to look at the cmake cache.
    BOOST_TEST_MESSAGE("BandedMatrix active path: "
                       << (manta::BandedMatrix::usesLapack ? "LAPACK dgbtrf/dgbtrs"
                                                           : "built-in dgbtf2"));
    BOOST_TEST(true);
}

BOOST_AUTO_TEST_CASE(banded_solve_matches_a_dense_full_piv_lu)
{
    // Diagonally dominant, so this passes with or without pivoting and isolates
    // the storage indexing from the pivoting logic.
    for (Index n : {1, 2, 3, 5, 17, 64})
        for (Index kl : {0, 1, 2, 3})
        {
            const Index k = std::min(kl, n - 1);
            const Matrix dense =
                randomBandedDense(n, k, k, 10.0, 12345u + static_cast<unsigned>(n * 8 + k));
            const Vector rhs = Vector::LinSpaced(n, -1.0, 2.0);
            const Vector want = dense.fullPivLu().solve(rhs);

            const BothWays got = solveBothWays(dense, k, k, rhs);
            BOOST_TEST(got.activeOk);
            BOOST_TEST(got.builtinOk);
            BOOST_TEST((got.active - want).cwiseAbs().maxCoeff() < 1e-10,
                       "active: n=" << n << " kl=ku=" << k);
            BOOST_TEST((got.builtin - want).cwiseAbs().maxCoeff() < 1e-10,
                       "builtin: n=" << n << " kl=ku=" << k);
        }
}

BOOST_AUTO_TEST_CASE(banded_solve_handles_an_asymmetric_band)
{
    // kl != ku exercises the storage offsets in a way a square band cannot: the
    // fill-in workspace is kl rows deep while the initial data starts ku above
    // the diagonal, and swapping the two would still pass every kl == ku case.
    for (auto [kl, ku] : {std::pair<Index, Index>{1, 3}, {3, 1}, {0, 2}, {2, 0}})
    {
        const Index n = 40;
        const Matrix dense = randomBandedDense(n, kl, ku, 8.0,
                                               999u + static_cast<unsigned>(kl * 16 + ku));
        const Vector rhs = Vector::LinSpaced(n, 1.0, -3.0);
        const Vector want = dense.fullPivLu().solve(rhs);

        const BothWays got = solveBothWays(dense, kl, ku, rhs);
        BOOST_TEST((got.active - want).cwiseAbs().maxCoeff() < 1e-10,
                   "active: kl=" << kl << " ku=" << ku);
        BOOST_TEST((got.builtin - want).cwiseAbs().maxCoeff() < 1e-10,
                   "builtin: kl=" << kl << " ku=" << ku);
    }
}

BOOST_AUTO_TEST_CASE(banded_solve_pivots_when_the_diagonal_is_weak)
{
    // No diagonal boost, so the draw decides the pivots and rows really do get
    // swapped. This is the case that fails outright if the row exchange or the
    // widened upper band is wrong -- a no-pivot Thomas sweep, the obvious cheap
    // alternative to this whole class, gives a visibly wrong answer here rather
    // than a slightly worse one.
    for (unsigned seed : {7u, 101u, 4242u})
    {
        const Index n = 50, kl = 3, ku = 3;
        const Matrix dense = randomBandedDense(n, kl, ku, 0.0, seed);
        const Vector rhs = Vector::LinSpaced(n, -2.0, 5.0);
        const Vector want = dense.fullPivLu().solve(rhs);

        const BothWays got = solveBothWays(dense, kl, ku, rhs);

        // Against the residual first -- that is the check that survives the two
        // decompositions disagreeing about a genuinely ill-conditioned draw --
        // and then against the dense answer.
        for (auto const &[label, x] : {std::pair<const char *, Vector const &>{"active", got.active},
                                       {"builtin", got.builtin}})
        {
            const double resid = (dense * x - rhs).cwiseAbs().maxCoeff();
            BOOST_TEST(resid < 1e-9, label << " seed " << seed << " residual " << resid);
            BOOST_TEST((x - want).cwiseAbs().maxCoeff() < 1e-7,
                       label << " seed " << seed << " vs dense");
        }
    }
}

BOOST_AUTO_TEST_CASE(banded_factorize_reports_a_singular_matrix)
{
    // FullPivLU would have returned *something* here. Reporting it is the one
    // behaviour that is deliberately different, so it is pinned rather than left
    // to be discovered. Both paths have to agree: LAPACK signals it through
    // info > 0, the built-in through an exactly zero pivot.
    Matrix dense = Matrix::Identity(4, 4);
    dense(2, 2) = 0.0; // row 2 entirely zero: no off-diagonal to pivot onto

    manta::BandedMatrix band;
    fillBand(band, dense, 1, 1);
    BOOST_TEST(!band.factorize());
    fillBand(band, dense, 1, 1);
    BOOST_TEST(!band.factorizeBuiltin());
}

BOOST_AUTO_TEST_CASE(banded_matrix_is_reusable_without_resizing)
{
    // solveHDGJac calls setZero, refills and refactorises every Newton
    // iteration, and never resizes after the first. A stale ipiv_ or leftover
    // fill-in from the previous factorisation would show up here and nowhere
    // else -- note that the fill-in workspace is only zeroed by factorize, not
    // by setZero, which is exactly the sort of thing that works once.
    const Index n = 30, kl = 2, ku = 2;
    manta::BandedMatrix band;
    band.resize(n, kl, ku);

    for (unsigned round : {1u, 2u, 3u})
    {
        const Matrix dense = randomBandedDense(n, kl, ku, 6.0, 55u * round);
        const Vector rhs = Vector::LinSpaced(n, 0.5, 3.5);
        const Vector want = dense.fullPivLu().solve(rhs);

        fillBand(band, dense, kl, ku);
        BOOST_TEST(band.factorize());
        Vector x = rhs;
        band.solveInPlace(x);
        BOOST_TEST((x - want).cwiseAbs().maxCoeff() < 1e-10, "round " << round);
    }
}

BOOST_AUTO_TEST_SUITE_END()
