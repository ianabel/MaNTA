#ifndef MANTA_UTIL_PARALLELFOR_HPP
#define MANTA_UTIL_PARALLELFOR_HPP

// One cell-independent loop, run in parallel when the build has OpenMP.
//
// Every `#pragma omp` in MaNTA lives here. That is not tidiness for its own
// sake -- the pragmas used to be written out at each site, and three properties
// they all need were present at none of them.
//
//  * **Exceptions.** A physics hook may throw: `static_residual` catches it,
//    prints, and returns 1, which IDA treats as recoverable and retries with a
//    smaller step. That is the documented contract, it has a test, and it is how
//    a Python case's exception reaches the solver. An exception that escapes an
//    OpenMP structured block does *not* propagate -- gcc's outlined function has
//    no handler above it, so `__cxa_call_terminate` aborts the process. Measured
//    on this tree: with the bare pragmas, `MANTA_OPENMP=ON` aborted the unit
//    suite inside
//    `residual_tests/static_residual_converts_a_physics_exception_into_a_retry`,
//    on a worker thread, with this stack:
//
//        __cxa_call_terminate
//        __cxa_throw
//        ThrowingDiffusion::SigmaFn(...)
//        TransportSystem::SigmaFn(...) [clone ._omp_fn.0]
//        libgomp.so.1
//
//    It is worse than a plain crash, because it is thread-dependent: run that
//    test alone and the throwing iteration lands on the master thread, where the
//    exception *can* reach the handler, and it passes. So this catches per
//    iteration, keeps the first one, and rethrows on the calling thread once the
//    region is over.
//
//  * **A trip-count floor.** Forking a team for a handful of iterations costs
//    more than it saves, and MaNTA's own fixtures are 3-10 cells. `grain` is the
//    smallest `n` worth a parallel region for *this* loop's body, so a caller
//    whose body is a 20x20 factorisation and one whose body is a single
//    pointwise hook can say so separately.
//
//  * **The index type.** OpenMP's canonical loop form is happiest with a signed
//    integer, and the sites differed -- `size_t`, `unsigned int`, `Index`. The
//    body is handed an `Index` (which is `Eigen::Index`, i.e. signed) whatever
//    the caller counted in.
//
// Without OpenMP the pragma is not compiled at all, rather than being emitted
// and ignored, so the build does not need `-Wno-unknown-pragmas` and a genuinely
// mistyped pragma anywhere else in the tree is still reported.
//
// `body` must be safe to run concurrently for distinct `i`: read what it likes,
// write only to storage indexed by `i`. That is the HDG cell property for a loop
// over cells -- but note it stops holding the moment a loop touches `lambda`,
// `K_global` or `F`, which live on cell *faces* and so are shared between
// neighbours. Those loops are not candidates and are marked as such at the site.
//
// Nested calls are safe but pointless: OpenMP leaves nested parallelism off by
// default, so an inner region runs with a team of one.

#include <atomic>
#include <exception>
#include <utility>

#include "../Types.hpp"

namespace manta
{

/// Run `body(i)` for i in [0, n), in parallel where the build allows it.
///
/// `grain` is the smallest `n` for which a parallel region is worth entering.
template <typename Body>
void parallel_for(Index n, Body &&body, Index grain = 32)
{
#ifdef _OPENMP
    if (n >= grain)
    {
        // The flag is what stops the rest of the iteration space running after a
        // throw; the exception_ptr itself is written under a critical section
        // because copying one is not atomic. Reading `failed` relaxed is enough:
        // it is a hint, and a thread that misses the update merely does work
        // whose result is discarded.
        std::atomic<bool> failed{false};
        std::exception_ptr firstError;

#pragma omp parallel for schedule(static)
        for (Index i = 0; i < n; ++i)
        {
            if (failed.load(std::memory_order_relaxed))
                continue;
            try
            {
                body(i);
            }
            catch (...)
            {
#pragma omp critical(manta_parallel_for_error)
                {
                    if (!firstError)
                        firstError = std::current_exception();
                }
                failed.store(true, std::memory_order_relaxed);
            }
        }

        if (firstError)
            std::rethrow_exception(firstError);
        return;
    }
#else
    (void)grain;
#endif

    for (Index i = 0; i < n; ++i)
        body(i);
}

} // namespace manta

#endif // MANTA_UTIL_PARALLELFOR_HPP
