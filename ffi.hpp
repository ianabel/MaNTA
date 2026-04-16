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

#include "PyRunner.hpp"

namespace ffi = xla::ffi;
namespace py = pybind11;

// can use either 64 or 32 bit math, based on jax config
#define fp_dtype ffi::F64
#define i_dtype ffi::S64

static ffi::Error run_ffi_impl(void *ctx, ffi::AnyBuffer args)
{
    auto runner = static_cast<PyRunner *>(ctx);
    py::gil_scoped_acquire gil; // needed to prevent segfault
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
    assert(out_dim.front() == G_p_internal.rows());
    assert(out_dim.back() == G_p_internal.cols());
    for (Index i = 0; i < G_p_internal.rows(); i++)
        for (Index j = 0; j < G_p_internal.cols(); j++)
        {
            auto const idx = i * out_dim.back() + j;
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
    int var_index = *var.typed_data();
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

#endif // FFI_HPP