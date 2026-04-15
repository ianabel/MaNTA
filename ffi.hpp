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

ffi::TypeId PyRunner::id = {};

template <typename T, typename... Args>
static constexpr T InvokeFn(PyRunner *runner, T (PyRunner::*fn)(Args...), Args... args)
{
    return (runner->*fn)(args...);
}

static ffi::ErrorOr<std::unique_ptr<PyRunner>> PyRunnerInstantiate()
{
    return std::make_unique<PyRunner>();
}

static ffi::Error RunnerExecute(PyRunner *runner)
{
    // do nothing
    return ffi::Error::Success();
}

static ffi::Error run_ffi_impl(PyRunner *runner, ffi::Buffer<ffi::DataType::F64> tFinal)
{
    InvokeFn(runner, &PyRunner::run, *static_cast<double *>(tFinal.typed_data()));
    return ffi::Error::Success();
}

static ffi::Error run_ffi_ss_impl(PyRunner *runner)
{
    InvokeFn(runner, &PyRunner::run_ss);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(py_runner_instantiate, PyRunnerInstantiate, ffi::Ffi::BindInstantiate());
XLA_FFI_DEFINE_HANDLER(kRunnerExecute, RunnerExecute, ffi::Ffi::BindExecute().Ctx<ffi::State<PyRunner>>());
XLA_FFI_DEFINE_HANDLER(run_ffi, run_ffi_impl, ffi::Ffi::Bind().Ctx<ffi::State<PyRunner>>().Arg<ffi::Buffer<ffi::DataType::F64>>());
XLA_FFI_DEFINE_HANDLER(run_ffi_ss, run_ffi_ss_impl, ffi::Ffi::Bind().Ctx<ffi::State<PyRunner>>());

#endif // FFI_HPP