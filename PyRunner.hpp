#ifndef PYRUNNER_HPP
#define PYRUNNER_HPP

#include <iostream>
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <string>
#include <variant>

#include "SystemSolver.hpp"

#include "PhysicsCases.hpp"

namespace ffi = xla::ffi;
namespace py = pybind11;

// Generic parameter
template <typename T>
struct Parameter
{
    bool required;
    T _default = T();
};

using ParameterType = std::variant<Parameter<double>,
                                   Parameter<std::string>,
                                   Parameter<int>,
                                   Parameter<unsigned int>,
                                   Parameter<bool>,
                                   Parameter<std::vector<double>>>;

using map_t = std::map<std::string, ParameterType>;

class PyRunner
{
public:
    /*
        Creates runner object for running MaNTA from Python given a config dictionary
        Takes constructed transport system as input
    */
    PyRunner() = default;
    explicit PyRunner(std::shared_ptr<TransportSystem> problem) : pProblem(problem) {};
    ~PyRunner();

    // Configure solver from Python
    void configure(const py::dict &);

    // Runs solver to time tFinal
    void run(double tFinal);

    // Runs solver to steady state
    void run_ss(void);

    void setTransportSystem(std::shared_ptr<TransportSystem> problem)
    {
        pProblem = problem;
    };

    void setAdjointProblem(std::shared_ptr<AdjointProblem> ap)
    {
        adjoint = ap;
        system->setAdjointProblem(ap.get());
    };
    // Run adjoint solver and return tuple (G, G_p)
    py::tuple runAdjointSolve(void);

    Vector getSolution(Index var, std::optional<std::vector<Position>> const &points);

public:
    static ffi::TypeId id;

private:
    // Shared ownership of TransportSystem so user can update in Python without recreating object
    std::shared_ptr<TransportSystem> pProblem = nullptr;
    std::shared_ptr<AdjointProblem> adjoint = nullptr;

    // Ownership of objects handled by C++
    std::unique_ptr<SystemSolver> system;
    std::unique_ptr<Grid> grid;

    // Runner function that runs solver to tf
    std::function<void(double)> runner;

    bool configured = false;
    double steady_state_tolerance;

private: // solver data
    /*
        This class controls the lifetime of the solver data so that we can request more timesteps without restarting the integration
    */
    SUNLinearSolver LS = NULL; // linear solver memory structure
    SUNMatrix sunMat = NULL;   //
    void *IDA_mem = NULL;      // IDA memory structure
    int retval;

    N_Vector Y = NULL;           // vector for storing solution
    N_Vector dYdt = NULL;        // vector for storing time derivative of solution
    N_Vector constraints = NULL; // vector for storing constraints
    N_Vector _id = NULL;         // vector for storing id (which elements are algebraic or differentiable)
    N_Vector res = NULL;         // vector for storing residual
    N_Vector absTolVec = NULL;   // vector for storing absolute tolerances
    sunrealtype tout, tret;
};

#endif