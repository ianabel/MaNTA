#ifndef FFI_HPP
#define FFI_HPP

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#ifdef CUDA
#include "cuda_runtime_api.h"
#define fp_dtype_cuda ffi::F32
#define i_dtype_cuda ffi::S32
#endif
#define fp_dtype ffi::F32
#define i_dtype ffi::S32
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
              i * rhs.rows() +
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

  runner->run(tFinal);
  return ffi::Error::Success();
};

static ffi::Error run_ffi_ss_impl(PyRunner *runner) {
  py::gil_scoped_acquire gil;

  runner->run_ss();
  return ffi::Error::Success();
};
static ffi::Error get_g_val(PyRunner *runner,
                            ffi::Result<ffi::BufferR1<fp_dtype>> Gout) {
  py::gil_scoped_acquire gil;

  copyToBuffer(Gout, runner->G());
  return ffi::Error::Success();
};
static ffi::Error get_adjoint_gradients_ffi_impl(
    PyRunner *runner, ffi::Result<ffi::BufferR1<fp_dtype>> Gout,
    ffi::Result<ffi::BufferR2<fp_dtype>> G_p_out,
    std::optional<ffi::Result<ffi::BufferR1<fp_dtype>>> G_p_boundary_out) {
  py::gil_scoped_acquire gil;
  py::tuple result = runner->getAdjointGradients();
  auto G = result[0].cast<Vector>();
  py::dict G_p = result[1];
  auto G_p_internal = G_p["G_p"].cast<Matrix>();
  copyToBuffer(Gout, G);
  copyToBuffer(G_p_out, G_p_internal);
  if (G_p.contains("G_p_boundary")) {
    auto G_p_boundary = G_p["G_p_boundary"].cast<Vector>();
    copyToBuffer(G_p_boundary_out.value(), G_p_boundary);
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
      for (Index i = 0; i < rhs.size(); i++)
        float tmp = static_cast<float>(rhs(i));
      cudaMemcpyAsync(&lhs->typed_data()[i], &tmp, sizeof(float),
                      cudaMemcpyHostToDevice, stream);
    } else {
      // otherwise assume it's a Matrix
      auto const lhs_dim = lhs->dimensions();

      assert(lhs_dim.front() == rhs.rows());
      assert(lhs_dim.back() == rhs.cols());
      for (Index i = 0; i < rhs.rows(); i++)
        for (Index j = 0; j < rhs.cols(); j++) {
          float tmp = static_cast<float>(rhs(i, j));
          auto const idx =
              i * out_dim.back() +
              j; // formula for indexing into 2D
                 // buffer:
                 // https://github.com/openxla/xla/blob/main/xla/tests/custom_call_test.cc#L1577
          cudaMemcpyAsync(&lhs->typed_data()[idx], &tmp, sizeof(float),
                          cudaMemcpyHostToDevice, stream);
        }
    }
  }

  static ffi::Error get_adjoint_gradients_ffi_impl_cuda(
      cudaStream_t stream, PyRunner * runner,
      ffi::Result<ffi::BufferR1<fp_dtype_cuda>> Gout,
      ffi::Result<ffi::BufferR2<fp_dtype_cuda>> G_p_out,
      std::optional<ffi::Result<ffi::BufferR1<fp_dtype_cuda>>>
          G_p_boundary_out) {
    py::gil_scoped_acquire gil;
    py::tuple result = runner->getAdjointGradients();
    auto G = result[0].cast<Vector>();
    copyToBufferCUDA(stream, Gout, G) py::dict G_p = result[1];
    auto G_p_internal = G_p["G_p"].cast<Matrix>();
    copyToBufferCUDA(stream, G_p_out, G_p_internal);
    if (G_p.contains("G_p_boundary")) {
      auto G_p_boundary = G_p["G_p_boundary"].cast<Vector>();
      copyToBufferCUDA(stream, G_p_boundary_out.value(), G_p_boundary);
    }
    cudaStreamSynchronize(stream);
    return ffi::Error::Success();
  };

  static ffi::Error get_solution_ffi_impl_cuda(
      cudaStream_t stream, PyRunner * runner, ffi::Buffer<i_dtype_cuda> var,
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
      std::vector<float> points_vec(num_points);
      cudaMemcpyAsync(points_vec.data(), points.value().typed_data(),
                      num_points * sizeof(float), cudaMemcpyDeviceToHost,
                      stream);
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
                                    .Ret<ffi::BufferR1<fp_dtype>>()
                                    .Ret<ffi::BufferR2<fp_dtype>>()
                                    .OptionalRet<ffi::BufferR1<fp_dtype>>());

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
  XLA_FFI_DEFINE_HANDLER_SYMBOL(
      get_adjoint_gradients_ffi_ops_cuda, get_adjoint_gradients_ffi_impl_cuda,
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<cudaStream_t>>()
          .Attr<ffi::Pointer<PyRunner>>("obj")
          .Ret<ffi::BufferR1<fp_dtype_cuda>>()
          .Ret<ffi::BufferR2<fp_dtype_cuda>>()
          .OptionalRet<ffi::BufferR1<fp_dtype_cuda>>());

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
