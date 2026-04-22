#ifndef FFI_HPP
#define FFI_HPP

#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <type_traits>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#ifdef CUDA
#include "cuda_runtime_api.h"
// cuda only supports 32 bit floats
#define fp_dtype_cuda ffi::F32
#define i_dtype_cuda ffi::S32
#endif
#define fp_dtype ffi::F64
#define i_dtype ffi::S64
#include "PyRunner.hpp"

namespace ffi = xla::ffi;
namespace py = pybind11;

/*
    Bindings for XLA and JAX, enables native jit compilation

    Basically emululate base functions but take in a void pointer to a PyRunner object
*/

// can use either 64 or 32 bit math, based on jax config

static ffi::Error run_ffi_impl(void *ctx, ffi::AnyBuffer args)
{

    auto runner = static_cast<PyRunner *>(ctx);
    py::gil_scoped_acquire gil;
    double tFinal = *args.typed_data<double>();
    runner->run(tFinal);
    return ffi::Error::Success();
};

static ffi::Error run_ffi_ss_impl(void *ctx)
{

    auto runner = static_cast<PyRunner *>(ctx);
    py::gil_scoped_acquire gil;
    runner->run_ss();
    return ffi::Error::Success();
};

static ffi::Error run_adjoint_ffi_impl(void *ctx, ffi::Result<ffi::BufferR1<fp_dtype>> Gout, ffi::Result<ffi::BufferR2<fp_dtype>> G_p_out, std::optional<ffi::Result<ffi::BufferR1<fp_dtype>>> G_p_boundary_out)
{

    auto runner = static_cast<PyRunner *>(ctx);
    py::gil_scoped_acquire gil;
    py::tuple result = runner->runAdjointSolve();
    auto G = result[0].cast<Vector>();
    py::dict G_p = result[1];
    auto G_p_internal = G_p["G_p"].cast<Matrix>();

    auto *G_out_data = Gout->typed_data();
    for (Index i = 0; i < G.size(); i++)
        G_out_data[i] = G(i);

    auto *G_p_out_data = G_p_out->typed_data();
    auto const out_dim = G_p_out->dimensions();

    // Make sure retval is correct shape
    assert(out_dim.front() == G_p_internal.rows());
    assert(out_dim.back() == G_p_internal.cols());

    for (Index i = 0; i < G_p_internal.rows(); i++)
        for (Index j = 0; j < G_p_internal.cols(); j++)
        {
            auto const idx = i * out_dim.back() + j; // formula for indexing into 2D buffer: https://github.com/openxla/xla/blob/main/xla/tests/custom_call_test.cc#L1577
            G_p_out_data[idx] = G_p_internal(i, j);
        }

    if (G_p.contains("G_p_boundary"))
    {
        auto G_p_boundary = G_p["G_p_boundary"].cast<Vector>();
        auto *G_p_boundary_out_data = G_p_boundary_out.value()->typed_data();
        for (Index i = 0; i < G_p_boundary.size(); i++)
            G_p_boundary_out_data[i] = G_p_boundary(i);
    }

    return ffi::Error::Success();
};

static ffi::Error get_solution_ffi_impl(void *ctx, ffi::Buffer<i_dtype> var, std::optional<ffi::BufferR1<fp_dtype>> points, ffi::Result<ffi::BufferR1<fp_dtype>> out)
{

    auto runner = static_cast<PyRunner *>(ctx);
    py::gil_scoped_acquire gil;
    auto var_index = *var.typed_data();
    if (points)
    {
        int num_points = points.value().element_count();
        std::vector<double> points_vec(points.value().typed_data(), points.value().typed_data() + num_points);
        Vector result = runner->getSolution(var_index, points_vec);
        auto *out_ptr = out->typed_data();

        for (Index i = 0; i < result.size(); i++)
            out_ptr[i] = result(i);

        return ffi::Error::Success();
    }
    else
    {
        Vector result = runner->getSolution(var_index, std::nullopt);
        auto *out_ptr = out->typed_data();
        for (Index i = 0; i < result.size(); i++)
            out_ptr[i] = result(i);
        return ffi::Error::Success();
    }
};

#ifdef CUDA
static ffi::Error run_ffi_impl_cuda(cudaStream_t stream, void *ctx, ffi::AnyBuffer args)
{

    auto runner = static_cast<PyRunner *>(ctx);
    py::gil_scoped_acquire gil; // needed to prevent segfault
    float tFinal;
    cudaMemcpyAsync(&tFinal, args.typed_data<float>(), sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    runner->run(tFinal);
    return ffi::Error::Success();
};

static ffi::Error run_ffi_ss_impl_cuda(cudaStream_t stream, void *ctx, ffi::Result<ffi::BufferR0<i_dtype_cuda>> is_err)
{
    py::gil_scoped_acquire gil;
    auto runner = static_cast<PyRunner *>(ctx);

    runner->run_ss();
    int err = 0;
    cudaMemcpyAsync(is_err->typed_data(), &err, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return ffi::Error::Success();
};

static ffi::Error run_adjoint_ffi_impl_cuda(cudaStream_t stream, void *ctx, ffi::Result<ffi::BufferR1<fp_dtype_cuda>> Gout, ffi::Result<ffi::BufferR2<fp_dtype_cuda>> G_p_out, std::optional<ffi::Result<ffi::BufferR1<fp_dtype_cuda>>> G_p_boundary_out)
{

    auto runner = static_cast<PyRunner *>(ctx);
    py::gil_scoped_acquire gil;
    py::tuple result = runner->runAdjointSolve();
    auto G = result[0].cast<Vector>();
    py::dict G_p = result[1];
    auto G_p_internal = G_p["G_p"].cast<Matrix>();
    for (Index i = 0; i < G.size(); i++)
    {
        float tmp = static_cast<float>(G(i));
        cudaMemcpyAsync(&Gout->typed_data()[i], &tmp, sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    // auto *G_p_out_data = G_p_out->typed_data();
    auto const out_dim = G_p_out->dimensions();
    std::cerr << out_dim.front() << std::endl;
    std::cerr << G_p_internal.rows() << std::endl;

    std::cerr << out_dim.back() << std::endl;
    std::cerr << G_p_internal.cols() << std::endl;
    // Make sure retval is correct shape
    assert(out_dim.front() == G_p_internal.rows());
    assert(out_dim.back() == G_p_internal.cols());

    for (Index i = 0; i < G_p_internal.rows(); i++)
    {
        for (Index j = 0; j < G_p_internal.cols(); j++)
        {
            float tmp = static_cast<float>(G_p_internal(i, j));
            auto const idx = i * out_dim.back() + j; // formula for indexing into 2D buffer: https://github.com/openxla/xla/blob/main/xla/tests/custom_call_test.cc#L1577
            cudaMemcpyAsync(&G_p_out->typed_data()[idx], &tmp, sizeof(float), cudaMemcpyHostToDevice, stream);
        }
    }

    if (G_p.contains("G_p_boundary"))
    {
        auto G_p_boundary = G_p["G_p_boundary"].cast<Vector>();
        for (Index i = 0; i < G_p_boundary.size(); i++)
        {
            float tmp = static_cast<float>(G_p_boundary(i));
            cudaMemcpyAsync(&G_p_boundary_out.value()->typed_data()[i], &tmp, sizeof(float), cudaMemcpyHostToDevice, stream);
        }
    }
    cudaStreamSynchronize(stream);
    return ffi::Error::Success();
};

static ffi::Error get_solution_ffi_impl_cuda(cudaStream_t stream, void *ctx, ffi::Buffer<i_dtype_cuda> var, std::optional<ffi::BufferR1<fp_dtype_cuda>> points, ffi::Result<ffi::BufferR1<fp_dtype_cuda>> out)
{

    auto runner = static_cast<PyRunner *>(ctx);
    py::gil_scoped_acquire gil;
    int var_index;
    cudaMemcpyAsync(&var_index, var.typed_data(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (points)
    {
        int num_points = points.value().element_count();
        std::vector<float> points_vec(num_points);
        cudaMemcpyAsync(points_vec.data(), points.value().typed_data(), num_points * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        std::vector<double> points_double(points_vec.begin(), points_vec.end());
        Vector result = runner->getSolution(var_index, points_double);
        for (Index i = 0; i < result.size(); i++)
        {
            float tmp = static_cast<float>(result(i));
            cudaMemcpyAsync(&out->typed_data()[i], &tmp, sizeof(float), cudaMemcpyHostToDevice, stream);
        }
        cudaStreamSynchronize(stream);
        return ffi::Error::Success();
    }
    else
    {
        Vector result = runner->getSolution(var_index, std::nullopt);

        for (Index i = 0; i < result.size(); i++)
        {
            float tmp = static_cast<float>(result(i));
            cudaMemcpyAsync(&out->typed_data()[i], &tmp, sizeof(float), cudaMemcpyHostToDevice, stream);
        }
        cudaStreamSynchronize(stream);
        return ffi::Error::Success();
    }
};
#endif

XLA_FFI_DEFINE_HANDLER_SYMBOL(run_ffi_ops, run_ffi_impl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Pointer<void>>("obj")
                                  .Arg<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(run_ss_ffi_ops, run_ffi_ss_impl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Pointer<void>>("obj"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(run_adjoint_solve_ffi_ops, run_adjoint_ffi_impl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Pointer<void>>("obj")
                                  .Ret<ffi::BufferR1<fp_dtype>>()
                                  .Ret<ffi::BufferR2<fp_dtype>>()
                                  .OptionalRet<ffi::BufferR1<fp_dtype>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(get_solution_ffi_ops, get_solution_ffi_impl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Pointer<void>>("obj")
                                  .Arg<ffi::Buffer<i_dtype>>()
                                  .OptionalArg<ffi::BufferR1<fp_dtype>>()
                                  .Ret<ffi::BufferR1<fp_dtype>>());

#ifdef CUDA
XLA_FFI_DEFINE_HANDLER_SYMBOL(run_ffi_ops_cuda, run_ffi_impl_cuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Pointer<void>>("obj")
                                  .Arg<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(run_ss_ffi_ops_cuda, run_ffi_ss_impl_cuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Pointer<void>>("obj")
                                  .Ret<ffi::BufferR0<i_dtype_cuda>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(run_adjoint_solve_ffi_ops_cuda, run_adjoint_ffi_impl_cuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Pointer<void>>("obj")
                                  .Ret<ffi::BufferR1<fp_dtype_cuda>>()
                                  .Ret<ffi::BufferR2<fp_dtype_cuda>>()
                                  .OptionalRet<ffi::BufferR1<fp_dtype_cuda>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(get_solution_ffi_ops_cuda, get_solution_ffi_impl_cuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Pointer<void>>("obj")
                                  .Arg<ffi::Buffer<i_dtype_cuda>>()
                                  .OptionalArg<ffi::BufferR1<fp_dtype_cuda>>()
                                  .Ret<ffi::BufferR1<fp_dtype_cuda>>());
#endif
#endif // FFI_HPP