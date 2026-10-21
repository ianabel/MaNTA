#ifndef FFI_HPP
#define FFI_HPP

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

// The floating-point width of every FFI buffer, chosen at configure time by
// MANTA_FFI_FP64. ffi_fp_t is the matching C++ type, for the copies that stage
// through a host temporary; Python.cpp reports its width to manta.jax.
#ifdef MANTA_FFI_FP64
#define fp_dtype ffi::F64
using ffi_fp_t = double;
#else
#define fp_dtype ffi::F32
using ffi_fp_t = float;
#endif
#define i_dtype ffi::S32

#ifdef CUDA
#include "cuda_runtime_api.h"
#define fp_dtype_cuda fp_dtype
#define i_dtype_cuda ffi::S32
#endif
#include "PyRunner.hpp"

namespace ffi = xla::ffi;
namespace py = pybind11;

/*
        Bindings for XLA and JAX, enables native MaNTA calls within JAX
   transforms

        Basically emululate base functions but take in a pointer to a PyRunner
   object
*/

template <typename T, typename Buffer>
inline void copyToBuffer(Buffer &&lhs, const T &rhs) {
  // check if T is an Eigen type
  if constexpr (std::is_base_of<Eigen::MatrixBase<typename std::decay<T>::type>,
                                typename std::decay<T>::type>::value) {
    auto *lhs_data = lhs->typed_data();
    // is T a Vector?
    if constexpr (Eigen::MatrixBase<
                      typename std::decay<T>::type>::IsVectorAtCompileTime) {
      for (Index i = 0; i < rhs.size(); i++)
        lhs_data[i] = rhs(i);
    } else {
      // otherwise assume it's a Matrix
#ifdef DEBUG
      auto const lhs_dim = lhs->dimensions();

      assert(lhs_dim.front() == rhs.rows());
      assert(lhs_dim.back() == rhs.cols());
#endif
      for (Index i = 0; i < rhs.rows(); i++)
        for (Index j = 0; j < rhs.cols(); j++) {
          auto const idx =
              i * rhs.cols() +
              j; // formula for indexing into 2D
                 // buffer:
                 // https://github.com/openxla/xla/blob/main/xla/tests/custom_call_test.cc#L1577
          lhs_data[idx] = rhs(i, j);
        }
    }
  }

  else {
    // otherwise assume we can just set them equal
    lhs->typed_data() = *rhs;
  }
}
// can use either 64 or 32 bit math, based on jax config
static ffi::Error run_ffi_impl(PyRunner *runner, ffi::BufferR0<fp_dtype> args) {
  py::gil_scoped_acquire gil;
  double tFinal = static_cast<double>(*args.typed_data());
  try {
    runner->run(tFinal);
  } catch (const std::exception &e) {
    return ffi::Error::Internal(e.what());
  }
  return ffi::Error::Success();
};

static ffi::Error run_ffi_ss_impl(PyRunner *runner) {
  py::gil_scoped_acquire gil;

  try {
    runner->run_ss();
  } catch (const std::exception &e) {
    return ffi::Error::Internal(e.what());
  }
  return ffi::Error::Success();
};
// A steady solve taken one slice at a time.
//
// The outcome is *returned*, not signalled as an error. OutOfSteps means the
// MaxContinuationSteps budget was spent and nothing is wrong -- the state and
// the pseudo-time step are both good and the next slice resumes from them --
// which is a distinction the driver makes on every iteration, and one
// ffi::Error would flatten into a single runtime error indistinguishable from a
// dead solve. A genuine failure still throws, and becomes that error.
static inline ffi::Error steady_slice(PyRunner *runner, bool resume,
                                      bool estimate,
                                      ffi::Result<ffi::BufferR0<i_dtype>> out) {
  try {
    const auto outcome = resume ? runner->continue_steady(estimate)
                                : runner->start_steady(estimate);
    *out->typed_data() = static_cast<int32_t>(outcome);
  } catch (const std::exception &e) {
    return ffi::Error::Internal(e.what());
  }
  return ffi::Error::Success();
};

static ffi::Error start_steady_ffi_impl(PyRunner *runner,
                                        ffi::Buffer<i_dtype> estimate,
                                        ffi::Result<ffi::BufferR0<i_dtype>> out) {
  py::gil_scoped_acquire gil;
  return steady_slice(runner, false, *estimate.typed_data() != 0, out);
};

static ffi::Error continue_steady_ffi_impl(
    PyRunner *runner, ffi::Buffer<i_dtype> estimate,
    ffi::Result<ffi::BufferR0<i_dtype>> out) {
  py::gil_scoped_acquire gil;
  return steady_slice(runner, true, *estimate.typed_data() != 0, out);
};

// Ends the solve: the output files, the restart file, the adjoint solve, then
// teardown. Separate from abandon_steady, which frees the same objects and
// writes nothing.
static ffi::Error finish_steady_ffi_impl(PyRunner *runner) {
  py::gil_scoped_acquire gil;
  try {
    runner->finish_steady();
  } catch (const std::exception &e) {
    return ffi::Error::Internal(e.what());
  }
  return ffi::Error::Success();
};

static ffi::Error abandon_steady_ffi_impl(PyRunner *runner) {
  py::gil_scoped_acquire gil;
  try {
    runner->abandon_steady();
  } catch (const std::exception &e) {
    return ffi::Error::Internal(e.what());
  }
  return ffi::Error::Success();
};

static ffi::Error get_g_val(PyRunner *runner,
                            ffi::Result<ffi::BufferR1<fp_dtype>> Gout) {
  py::gil_scoped_acquire gil;

  copyToBuffer(Gout, runner->G());
  return ffi::Error::Success();
};
// G_p as the single (rows, np) matrix FFIRunner declares. PyRunner hands the
// internal and boundary columns back separately; joining them here keeps the
// FFI output one buffer whatever np_boundary is, where a separate boundary
// output would need every caller to know to request it.
inline Matrix adjointGradientMatrix(py::dict const &G_p) {
  Matrix internal = G_p["G_p"].cast<Matrix>();
  if (!G_p.contains("G_p_boundary"))
    return internal;
  Matrix boundary = G_p["G_p_boundary"].cast<Matrix>();
  Matrix full(internal.rows(), internal.cols() + boundary.cols());
  full << internal, boundary;
  return full;
}

// The outcome of the solve whose gradients these are, as start_steady or
// continue_steady returned it -- or after finish_steady, which is where the
// adjoint solve runs. It is an operand rather than something to look up so that
// the read *depends* on the solve: inside a compiled program (a lax.scan, as
// DESC's jac_chunk_size uses) XLA orders by data dependence alone, and with no
// operand this call could run before the solve, or be hoisted out of the loop,
// and read a G_p nothing had filled. Absent, it is taken as Converged, which is
// what an eager caller that has already checked the outcome means.
static ffi::Error
checkAdjointOutcome(std::optional<SystemSolver::SteadyOutcome> outcome) {
  using SO = SystemSolver::SteadyOutcome;
  const SO o = outcome.value_or(SO::Converged);
  if (o == SO::Converged)
    return ffi::Error::Success();
  return ffi::Error::InvalidArgument(
      "get_adjoint_gradients: the steady solve did not converge (SteadyOutcome " +
      std::to_string(static_cast<int>(o)) +
      "), so there are no adjoint gradients to read");
}

static ffi::Error get_adjoint_gradients_ffi_impl(
    PyRunner *runner, std::optional<ffi::BufferR0<i_dtype>> outcome,
    ffi::Result<ffi::BufferR1<fp_dtype>> Gout,
    ffi::Result<ffi::BufferR2<fp_dtype>> G_p_out) {
  py::gil_scoped_acquire gil;
  std::optional<SystemSolver::SteadyOutcome> o;
  if (outcome)
    o = static_cast<SystemSolver::SteadyOutcome>(*outcome->typed_data());
  if (auto err = checkAdjointOutcome(o); err.failure())
    return err;
  try {
    py::tuple result = runner->getAdjointGradients();
    auto G = result[0].cast<Vector>();
    copyToBuffer(Gout, G);
    copyToBuffer(G_p_out, adjointGradientMatrix(result[1]));
  } catch (const std::exception &e) {
    return ffi::Error::Internal(e.what());
  }

  return ffi::Error::Success();
};

static ffi::Error
get_solution_ffi_impl(PyRunner *runner, ffi::Buffer<i_dtype> var,
                      std::optional<ffi::BufferR1<fp_dtype>> points,
                      ffi::Result<ffi::BufferR1<fp_dtype>> out) {
  py::gil_scoped_acquire gil;
  auto var_index = *var.typed_data();
  if (points) {
    int num_points = points.value().element_count();
    std::vector<double> points_vec(points.value().typed_data(),
                                   points.value().typed_data() + num_points);
    Vector result = runner->getSolution(var_index, points_vec);

    copyToBuffer(out, result);
    return ffi::Error::Success();
  } else {
    Vector result = runner->getSolution(var_index, std::nullopt);
    copyToBuffer(out, result);
    return ffi::Error::Success();
  }
};

#ifdef CUDA

template <typename T, typename Buffer>
static void copyToBufferCUDA(cudaStream_t stream, Buffer &&lhs, const T &rhs) {
  // check if T is an Eigen Matrix
  if constexpr (std::is_base_of<Eigen::MatrixBase<typename std::decay<T>::type>,
                                typename std::decay<T>::type>::value) {
    auto *lhs_data = lhs->typed_data();
    // is T a Vector?
    if constexpr (Eigen::MatrixBase<
                      typename std::decay<T>::type>::IsVectorAtCompileTime) {
      for (Index i = 0; i < rhs.size(); i++) {
        ffi_fp_t tmp = static_cast<ffi_fp_t>(rhs(i));
        cudaMemcpyAsync(&lhs->typed_data()[i], &tmp, sizeof(ffi_fp_t),
                        cudaMemcpyHostToDevice, stream);
      }
    } else {
      // otherwise assume it's a Matrix
      auto const lhs_dim = lhs->dimensions();

      assert(lhs_dim.front() == rhs.rows());
      assert(lhs_dim.back() == rhs.cols());
      for (Index i = 0; i < rhs.rows(); i++)
        for (Index j = 0; j < rhs.cols(); j++) {
          ffi_fp_t tmp = static_cast<ffi_fp_t>(rhs(i, j));
          auto const idx =
              i * lhs_dim.back() +
              j; // formula for indexing into 2D
                 // buffer:
                 // https://github.com/openxla/xla/blob/main/xla/tests/custom_call_test.cc#L1577
          cudaMemcpyAsync(&lhs->typed_data()[idx], &tmp, sizeof(ffi_fp_t),
                          cudaMemcpyHostToDevice, stream);
        }
    }
  }
}

static ffi::Error get_adjoint_gradients_ffi_impl_cuda(
    cudaStream_t stream, PyRunner *runner,
    std::optional<ffi::BufferR0<i_dtype_cuda>> outcome,
    ffi::Result<ffi::BufferR1<fp_dtype_cuda>> Gout,
    ffi::Result<ffi::BufferR2<fp_dtype_cuda>> G_p_out) {
  py::gil_scoped_acquire gil;
  std::optional<SystemSolver::SteadyOutcome> o;
  if (outcome) {
    int32_t value;
    cudaMemcpyAsync(&value, outcome->typed_data(), sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    o = static_cast<SystemSolver::SteadyOutcome>(value);
  }
  if (auto err = checkAdjointOutcome(o); err.failure())
    return err;
  try {
    py::tuple result = runner->getAdjointGradients();
    auto G = result[0].cast<Vector>();
    copyToBufferCUDA(stream, Gout, G);
    copyToBufferCUDA(stream, G_p_out, adjointGradientMatrix(result[1]));
  } catch (const std::exception &e) {
    return ffi::Error::Internal(e.what());
  }
  cudaStreamSynchronize(stream);
  return ffi::Error::Success();
};

static ffi::Error
get_solution_ffi_impl_cuda(cudaStream_t stream, PyRunner *runner,
                           ffi::Buffer<i_dtype_cuda> var,
                           std::optional<ffi::BufferR1<fp_dtype_cuda>> points,
                           ffi::Result<ffi::BufferR1<fp_dtype_cuda>> out) {
  // auto runner = static_cast<PyRunner *>(obj);
  py::gil_scoped_acquire gil;
  int var_index;
  cudaMemcpyAsync(&var_index, var.typed_data(), sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  if (points) {
    int num_points = points.value().element_count();
    std::vector<ffi_fp_t> points_vec(num_points);
    cudaMemcpyAsync(points_vec.data(), points.value().typed_data(),
                    num_points * sizeof(ffi_fp_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::vector<double> points_double(points_vec.begin(), points_vec.end());
    Vector result = runner->getSolution(var_index, points_double);
    copyToBufferCUDA(stream, out, result);
    cudaStreamSynchronize(stream);
    return ffi::Error::Success();
  } else {
    Vector result = runner->getSolution(var_index, std::nullopt);

    copyToBufferCUDA(stream, out, result);
    cudaStreamSynchronize(stream);
    return ffi::Error::Success();
  }
};
#endif

XLA_FFI_DEFINE_HANDLER_SYMBOL(run_ffi_ops, run_ffi_impl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Pointer<PyRunner>>("obj")
                                  .Arg<ffi::BufferR0<fp_dtype>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    run_ss_ffi_ops, run_ffi_ss_impl,
    ffi::Ffi::Bind().Attr<ffi::Pointer<PyRunner>>("obj"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(get_adjoint_gradients_ffi_ops,
                              get_adjoint_gradients_ffi_impl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Pointer<PyRunner>>("obj")
                                  .OptionalArg<ffi::BufferR0<i_dtype>>()
                                  .Ret<ffi::BufferR1<fp_dtype>>()
                                  .Ret<ffi::BufferR2<fp_dtype>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(start_steady_ffi_ops, start_steady_ffi_impl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Pointer<PyRunner>>("obj")
                                  .Arg<ffi::Buffer<i_dtype>>()
                                  .Ret<ffi::BufferR0<i_dtype>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(continue_steady_ffi_ops, continue_steady_ffi_impl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Pointer<PyRunner>>("obj")
                                  .Arg<ffi::Buffer<i_dtype>>()
                                  .Ret<ffi::BufferR0<i_dtype>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    finish_steady_ffi_ops, finish_steady_ffi_impl,
    ffi::Ffi::Bind().Attr<ffi::Pointer<PyRunner>>("obj"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    abandon_steady_ffi_ops, abandon_steady_ffi_impl,
    ffi::Ffi::Bind().Attr<ffi::Pointer<PyRunner>>("obj"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(get_g_val_ffi_ops, get_g_val,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Pointer<PyRunner>>("obj")
                                  .Ret<ffi::BufferR1<fp_dtype>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(get_solution_ffi_ops, get_solution_ffi_impl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Pointer<PyRunner>>("obj")
                                  .Arg<ffi::Buffer<i_dtype>>()
                                  .OptionalArg<ffi::BufferR1<fp_dtype>>()
                                  .Ret<ffi::BufferR1<fp_dtype>>());

#ifdef CUDA
XLA_FFI_DEFINE_HANDLER_SYMBOL(get_adjoint_gradients_ffi_ops_cuda,
                              get_adjoint_gradients_ffi_impl_cuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Pointer<PyRunner>>("obj")
                                  .OptionalArg<ffi::BufferR0<i_dtype_cuda>>()
                                  .Ret<ffi::BufferR1<fp_dtype_cuda>>()
                                  .Ret<ffi::BufferR2<fp_dtype_cuda>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(get_solution_ffi_ops_cuda,
                              get_solution_ffi_impl_cuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Pointer<PyRunner>>("obj")
                                  .Arg<ffi::Buffer<i_dtype_cuda>>()
                                  .OptionalArg<ffi::BufferR1<fp_dtype_cuda>>()
                                  .Ret<ffi::BufferR1<fp_dtype_cuda>>());
#endif
#endif // FFI_HPP
