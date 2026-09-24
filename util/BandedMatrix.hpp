#ifndef MANTA_UTIL_BANDEDMATRIX_HPP
#define MANTA_UTIL_BANDEDMATRIX_HPP

// A banded matrix and its LU factorisation with partial pivoting.
//
// This exists for one matrix: the HDG trace operator K_global, which
// static condensation leaves behind and which solveHDGJac factorises on every
// Newton iteration. That matrix is block-tridiagonal -- cell i couples only the
// two faces it owns -- and it was being handed to a *dense* Eigen::FullPivLU,
// which is O(n^3) in a quantity the method exists to make O(n). Measured before
// the change: 91% of a 400-cell k=3 run and 73% of a 200-cell k=4 one. TODO has
// the full breakdown.
//
// **LAPACK does the work when there is one.** `dgbtrf` and `dgbtrs` are called
// directly on the array below, with no copy and no repacking, because the
// storage layout *is* theirs: `A(i, j)` at `ab(kl + ku + i - j, j)` in an array
// of `2*kl + ku + 1` rows, of which the top `kl` are workspace for the fill-in
// that row swaps create above the diagonal. That is the whole reason the layout
// was chosen. `MANTA_HAVE_LAPACK` is defined by cmake/MantaDependencies.cmake
// when one is found; the same vendor as the BLAS, never a free choice, for the
// dlopen reason that file documents at length.
//
// The built-in factorisation below is a *fallback*, not a preference -- Debian
// splits liblapack-dev from libblas-dev, and a container with only the latter
// should still build. It is dgbtf2, the unblocked reference algorithm, written
// to the same storage and the same 1-based-absolute `ipiv` convention so the two
// are interchangeable, and so a build that has LAPACK can still test it:
// `factorizeBuiltin`/`solveInPlaceBuiltin` are always compiled and
// `utility_tests` exercises both paths whatever the build found. Without that
// the fallback would rot on every developer box that has LAPACK, which is most
// of them.
//
// **Partial pivoting is not optional here, and it is what makes this a
// replacement rather than a gamble.** The obvious cheap thing for a
// block-tridiagonal system is a block Thomas sweep with no pivoting at all, and
// that is only stable for a diagonally dominant or definite matrix. K_global is
// neither in general: it carries whatever the physics case's boundary
// coefficients and source Jacobians produce, and the operator it replaces --
// FullPivLU -- is the most conservative choice Eigen offers. Dropping straight
// from full pivoting to none would be a silent change in the class of problem
// the solver can survive. Partial pivoting costs `kl` extra superdiagonals of
// storage and nothing measurable in time.
//
// What it does NOT do, and what the dense FullPivLU did: reveal rank. Nothing
// asked it to -- neither call site ever queried rank or invertibility -- but a
// singular K_global now returns false from factorize() rather than quietly
// producing whatever FullPivLU produces for a rank-deficient system.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "../Types.hpp"

#ifdef MANTA_HAVE_LAPACK
// Declared here rather than taken from a header. Reference LAPACK ships no C
// prototypes at all (lapacke is a separate package, and MKL's is a third
// spelling), so the Fortran symbols are declared directly -- which is what Eigen
// itself does for its LAPACK backend. The trailing underscore is the gfortran
// and MKL convention on every platform this builds on.
//
// `int` is LP64, which is what Debian's liblapack and MKL's libmkl_rt both are.
// An ILP64 LAPACK would need these to be 64-bit; the factorize() guard below
// turns that into a thrown error rather than silent stack corruption.
//
// dgbtrs takes a Fortran CHARACTER, whose hidden length argument is omitted here
// as it is in every C caller of LAPACK: lsame reads the first byte and never the
// length.
extern "C"
{
    void dgbtrf_(const int *m, const int *n, const int *kl, const int *ku,
                 double *ab, const int *ldab, int *ipiv, int *info);
    void dgbtrs_(const char *trans, const int *n, const int *kl, const int *ku,
                 const int *nrhs, const double *ab, const int *ldab,
                 const int *ipiv, double *b, const int *ldb, int *info);
}
#endif

namespace manta
{

class BandedMatrix
{
  public:
    /// Size for an n x n matrix with `kl` subdiagonals and `ku` superdiagonals.
    /// Keeps any allocation it already has, so a solver that resizes to the same
    /// shape every step allocates once.
    void resize(Index n, Index kl, Index ku)
    {
        assert(n >= 0 && kl >= 0 && ku >= 0);
        n_ = n;
        kl_ = kl;
        ku_ = ku;
        // The top kl rows are fill-in workspace; see the header comment.
        ab_.resize(2 * kl + ku + 1, n);
        ipiv_.resize(static_cast<size_t>(n));
    }

    void setZero() { ab_.setZero(); }

    Index rows() const { return n_; }
    Index subDiagonals() const { return kl_; }
    Index superDiagonals() const { return ku_; }

    /// A(i, j). Both call sites accumulate, so this is the only accessor they
    /// need; it is deliberately not const-overloaded, because reading outside
    /// the band is a question the band form cannot answer.
    double &operator()(Index i, Index j)
    {
        assert(i >= 0 && i < n_ && j >= 0 && j < n_);
        assert(i - j <= kl_ && j - i <= ku_ && "index outside the declared band");
        return ab_(kl_ + ku_ + i - j, j);
    }

    /// True when this build calls LAPACK rather than the built-in factorisation.
    /// Public so a test can say which path it just exercised.
    static constexpr bool usesLapack =
#ifdef MANTA_HAVE_LAPACK
        true;
#else
        false;
#endif

    /// LU with partial pivoting, in place. False if the matrix is singular, in
    /// which case the contents are junk and solveInPlace must not be called.
    bool factorize()
    {
#ifdef MANTA_HAVE_LAPACK
        if (n_ > static_cast<Index>(std::numeric_limits<int>::max()))
            throw std::runtime_error(
                "BandedMatrix: this LAPACK is LP64 and the system is larger than "
                "an int can index");

        const int m = static_cast<int>(n_), n = static_cast<int>(n_);
        const int kl = static_cast<int>(kl_), ku = static_cast<int>(ku_);
        const int ldab = static_cast<int>(ab_.rows());
        int info = 0;
        dgbtrf_(&m, &n, &kl, &ku, ab_.data(), &ldab, ipiv_.data(), &info);
        // info < 0 is a bad argument -- a bug here, not in the matrix -- and
        // info > 0 is an exactly zero pivot, which is what the bool reports.
        if (info < 0)
            throw std::runtime_error("BandedMatrix: dgbtrf rejected argument " +
                                     std::to_string(-info));
        return info == 0;
#else
        return factorizeBuiltin();
#endif
    }

    /// Solve A x = b for x, overwriting b. factorize() must have returned true.
    void solveInPlace(Eigen::Ref<Vector> b) const
    {
        assert(b.size() == n_);
#ifdef MANTA_HAVE_LAPACK
        const char trans = 'N';
        const int n = static_cast<int>(n_);
        const int kl = static_cast<int>(kl_), ku = static_cast<int>(ku_);
        const int nrhs = 1, ldab = static_cast<int>(ab_.rows()), ldb = n;
        int info = 0;
        // b is an Eigen::Ref over a contiguous vector, so data() is a valid
        // Fortran column. Ref<Vector> without an InnerStride cannot be strided,
        // so this cannot silently be handed a gappy view.
        dgbtrs_(&trans, &n, &kl, &ku, &nrhs, ab_.data(), &ldab, ipiv_.data(),
                b.data(), &ldb, &info);
        if (info < 0)
            throw std::runtime_error("BandedMatrix: dgbtrs rejected argument " +
                                     std::to_string(-info));
#else
        solveInPlaceBuiltin(b);
#endif
    }

    /// The built-in factorisation, always compiled so that it can be tested on a
    /// build that would otherwise never reach it.
    ///
    /// This is dgbtf2 -- the unblocked reference algorithm -- and not the blocked
    /// dgbtrf, because the bands here are narrow (2*nVars - 1, so 1 for a
    /// single-variable case) and blocking buys nothing at that width.
    bool factorizeBuiltin()
    {
        const Index kv = kl_ + ku_;

        // The fill-in rows have to start at zero: the rank-1 update below writes
        // into them through the general indexing, and reads them back on a later
        // column.
        if (kl_ > 0)
            ab_.topRows(kl_).setZero();

        // The highest column any fill-in has reached. Tracked rather than
        // assumed to be j + kv: a column whose pivot needed no swap does not
        // widen the band, and updating only as far as necessary is what keeps
        // this O(n * kl * ku).
        Index ju = 0;

        for (Index j = 0; j < n_; ++j)
        {
            const Index km = std::min(kl_, n_ - 1 - j);

            Index jp = 0;
            double best = std::abs(ab_(kv, j));
            for (Index i = 1; i <= km; ++i)
            {
                const double v = std::abs(ab_(kv + i, j));
                if (v > best)
                {
                    best = v;
                    jp = i;
                }
            }
            // 1-based and absolute, which is LAPACK's convention -- so the two
            // factorisations produce the same ipiv and either solve can read
            // the other's output.
            ipiv_[static_cast<size_t>(j)] = static_cast<int>(j + jp + 1);

            if (ab_(kv + jp, j) == 0.0)
                return false;

            ju = std::max(ju, std::min(j + ku_ + jp, n_ - 1));

            // Swap rows j and j+jp, which in band storage means walking both
            // rows across the columns they share -- one column to the right is
            // one row up.
            if (jp != 0)
                for (Index c = j; c <= ju; ++c)
                    std::swap(ab_(kv + jp + j - c, c), ab_(kv + j - c, c));

            if (km > 0)
            {
                const double piv = ab_(kv, j);
                for (Index i = 1; i <= km; ++i)
                    ab_(kv + i, j) /= piv;

                for (Index c = j + 1; c <= ju; ++c)
                {
                    const double u = ab_(kv + j - c, c);
                    if (u != 0.0)
                        for (Index i = 1; i <= km; ++i)
                            ab_(kv + j + i - c, c) -= ab_(kv + i, j) * u;
                }
            }
        }
        return true;
    }

    /// The built-in triangular solves; see factorizeBuiltin.
    void solveInPlaceBuiltin(Eigen::Ref<Vector> b) const
    {
        assert(b.size() == n_);
        const Index kv = kl_ + ku_;

        // L y = P b. L is unit lower triangular with kl subdiagonals.
        for (Index j = 0; j < n_; ++j)
        {
            const Index km = std::min(kl_, n_ - 1 - j);
            const Index p = static_cast<Index>(ipiv_[static_cast<size_t>(j)]) - 1;
            if (p != j)
                std::swap(b(p), b(j));
            for (Index i = 1; i <= km; ++i)
                b(j + i) -= ab_(kv + i, j) * b(j);
        }

        // U x = y. U has up to kl + ku superdiagonals -- pivoting widened it
        // from ku, which is what the extra storage rows are for.
        for (Index j = n_ - 1; j >= 0; --j)
        {
            b(j) /= ab_(kv, j);
            const Index lo = std::max<Index>(0, j - kv);
            for (Index i = lo; i < j; ++i)
                b(i) -= ab_(kv + i - j, j) * b(j);
        }
    }

  private:
    Index n_ = 0, kl_ = 0, ku_ = 0;
    Matrix ab_;            // (2*kl + ku + 1) x n, column major -- LAPACK's layout
    std::vector<int> ipiv_; // 1-based absolute, LAPACK's convention
};

} // namespace manta

#endif // MANTA_UTIL_BANDEDMATRIX_HPP
