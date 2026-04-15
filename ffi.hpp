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

// enum class FFI_OPS
// {
//     Run,
//     RunSS,
//     RunAdjoint,
//     GetSolution
// };

std::pair<int64_t, int64_t> GetDims(const ffi::AnyBuffer buffer)
{
    const ffi::AnyBuffer::Dimensions dims = buffer.dimensions();
    if (dims.size() == 0)
    {
        return std::make_pair(0, 0);
    }
    return std::make_pair(buffer.element_count(), dims.back());
}

// static constexpr std::map<FFI_OPS, &InvokeFn> ffi_ops = {
//     {FFI_OPS::Run, &InvokeFn<void, double>},
//     {FFI_OPS::RunSS, &InvokeFn<void>},
//     {FFI_OPS::RunAdjoint, &InvokeFn<py::tuple>},
//     {FFI_OPS::GetSolution, &InvokeFn<Vector, Index, std::optional<std::vector<Position>>>}};

// static ffi::Error run_ffi_impl(void *ctx, ffi::Buffer<ffi::DataType::F64> tFinal)
// {
//     auto *runner = static_cast<PyRunner *>(ctx);
//     InvokeFn(runner, &PyRunner::run, *static_cast<double *>(tFinal.typed_data()));
//     return ffi::Error::Success();
// }

// static ffi::Error run_ffi_ss_impl(void *ctx)
// {
//     auto *runner = static_cast<PyRunner *>(ctx);
//     InvokeFn(runner, &PyRunner::run_ss);
//     return ffi::Error::Success();
// }

static ffi::Error run_ffi_impl(void *ctx, ffi::AnyBuffer args)
{
    auto runner = static_cast<PyRunner *>(ctx);
    double tFinal = *args.typed_data<double>();
    runner->run(tFinal);
    return ffi::Error::Success();
};

static ffi::Error run_ffi_ss_impl(void *ctx)
{
    auto runner = static_cast<PyRunner *>(ctx);
    runner->run_ss();
    return ffi::Error::Success();
};

static ffi::Error run_adjoint_ffi_impl(void *ctx, ffi::Result<ffi::BufferR1<ffi::F64>> Gout, ffi::Result<ffi::BufferR2<ffi::F64>> G_p_out, ffi::Result<ffi::BufferR1<ffi::F64>> G_p_boundary_out)
{
    auto runner = static_cast<PyRunner *>(ctx);
    py::tuple result = runner->runAdjointSolve();
    auto G = result[0].cast<Vector>();
    py::dict G_p = result[1];
    auto G_p_internal = G_p["G_p"].cast<Matrix>();

    double *G_out_data = Gout->typed_data();
    G_out_data = G.data();
    double *G_p_out_data = G_p_out->typed_data();
    G_p_out_data = G_p_internal.data();

    if (G_p.contains("G_p_boundary"))
    {
        auto G_p_boundary = G_p["G_p_boundary"].cast<Vector>();
        double *G_p_boundary_out_data = G_p_boundary_out->typed_data();
        G_p_boundary_out_data = G_p_boundary.data();
    }

    return ffi::Error::Success();
};

static ffi::Error get_solution_ffi_impl(void *ctx, ffi::Buffer<ffi::S32> var, ffi::BufferR1<ffi::F64> points, ffi::Result<ffi::BufferR1<ffi::F64>> out)
{
    auto runner = static_cast<PyRunner *>(ctx);
    int num_points = points.element_count();
    int var_index = *var.typed_data();
    if (num_points > 0)
    {
        std::vector<double> points_vec(points.typed_data(), points.typed_data() + sizeof(double) * num_points);
        Vector result = runner->getSolution(var_index, points_vec);
        double *out_ptr = out->typed_data();
        out_ptr = result.data();
        return ffi::Error::Success();
    }
    else
    {
        Vector result = runner->getSolution(var_index, std::nullopt);
        double *out_ptr = out->typed_data();
        out_ptr = result.data();
        return ffi::Error::Success();
    }
    return ffi::Error::Success();
};

XLA_FFI_DEFINE_HANDLER_SYMBOL(run_ffi_ops, run_ffi_impl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Pointer<void>>("ctx")
                                  .Arg<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(run_ss_ffi_ops, run_ffi_ss_impl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Pointer<void>>("ctx"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(run_adjoint_solve_ffi_ops, run_adjoint_ffi_impl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Pointer<void>>("ctx")
                                  .Ret<ffi::BufferR1<ffi::F64>>()
                                  .Ret<ffi::BufferR2<ffi::F64>>()
                                  .OptionalRet<ffi::BufferR1<ffi::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(get_solution_ffi_ops, get_solution_ffi_impl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Pointer<void>>("ctx")
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .OptionalArg<ffi::BufferR1<ffi::F64>>()
                                  .Ret<ffi::BufferR1<ffi::F64>>());

#endif // FFI_HPP

// static ffi::Error runner_ffi_ops_impl(void *ctx, ffi::Buffer<ffi::S32> op, ffi::AnyBuffer args, ffi::Result<ffi::AnyBuffer> rets)
// {
//     auto runner = static_cast<PyRunner *>(ctx);
//     FFI_OPS op_ = static_cast<FFI_OPS>(*op.typed_data());

//     switch (op_)
//     {
//     case FFI_OPS::Run:
//     {
//         double tFinal = *args.typed_data<double>();
//         runner->run(tFinal);
//         break;
//     }
//     case FFI_OPS::RunSS:
//     {
//         runner->run_ss();
//         break;
//     }
//     default:
//         break;
//         // case FFI_OPS::RunAdjoint:
//         // {
//         //     InvokeFn(runner, &PyRunner::runAdjointSolve);
//         //     break;
//         // }
//         // case FFI_OPS::GetSolution:
//         // {
//         //     auto [var, points] = *args.typed_data<std::tuple<Index, std::optional<std::vector<Position>>>>();
//         //     Vector result = InvokeFn(runner, &PyRunner::getSolution, var, points);
//         //     rets.typed_data<std::vector<double>>() = result.data();
//         //     break;
//         // }
//     };
//     // auto *runner = static_cast<PyRunner *>(ctx);
//     // // Handle the operation based on the buffer content
//     return ffi::Error::Success();
// }