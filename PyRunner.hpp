#include <iostream>
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <string>
#include <variant>

#include "SystemSolver.hpp"
#include "PhysicsCases.hpp"

namespace py = pybind11;

// Generic parameter
template <typename T>
struct Parameter
{
    bool required;
    T _default = T();
};

using ParameterType = std::variant<Parameter<double>, Parameter<std::string>, Parameter<int>, Parameter<unsigned int>, Parameter<bool>, Parameter<std::vector<double>>>;

// Parameters required by "configure" function to be passed to SystemSolver
static std::map<std::string, ParameterType> params = {{"restart", Parameter<bool>{.required = false, ._default = false}},
                                                      {"RestartFile", Parameter<std::string>{.required = false, ._default = ""}},
                                                      {"High_Grid_Boundary", Parameter<bool>{.required = false, ._default = false}},
                                                      {"Lower_Boundary_Fraction", Parameter<double>{.required = false, ._default = 0.2}},
                                                      {"Upper_Boundary_Fraction", Parameter<double>{.required = false, ._default = 0.2}},
                                                      {"Polynomial_degree", Parameter<unsigned int>{.required = true}},
                                                      {"Grid_size", Parameter<int>{.required = true}},
                                                      {"tau", Parameter<double>{.required = false, ._default = 1.0}},
                                                      {"delta_t", Parameter<double>{.required = true}},
                                                      {"tZero", Parameter<double>{.required = false, ._default = 0.0}},
                                                      {"tFinal", Parameter<double>{.required = true}},
                                                      {"Relative_tolerance", Parameter<double>{.required = false, ._default = 1e-3}},
                                                      {"Absolute_tolerance", Parameter<std::vector<double>>{.required = false, ._default = {1e-2}}},
                                                      {"MinStepSize", Parameter<double>{.required = false, ._default = 1e-7}},
                                                      {"OutputPoints", Parameter<int>{.required = false, ._default = 301}},
                                                      {"solveAdjoint", Parameter<bool>{.required = false, ._default = false}},
                                                      {"OutputFilename", Parameter<std::string>{.required = true}},
                                                      {"SteadyStateTolerance", Parameter<double>{.required = false}}};

class PyRunner
{
public:
    /*
        Creates runner object for running MaNTA from Python given a config dictionary
        Takes constructed transport system as input
    */
    PyRunner(std::shared_ptr<TransportSystem> problem) : pProblem(problem) {};

    void configure(const py::dict &);
    int run(void);

private:
    const std::shared_ptr<TransportSystem> pProblem;
    std::unique_ptr<AdjointProblem> adjoint;
    std::unique_ptr<SystemSolver> system;
    std::unique_ptr<Grid> grid;
    double tFinal;
};