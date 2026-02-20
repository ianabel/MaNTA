#ifndef PYRUNNER_HPP
#define PYRUNNER_HPP

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
using map_t = std::map<std::string, ParameterType>;
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
    // Shared ownership of TransportSystem so user can update in Python without recreating object
    const std::shared_ptr<TransportSystem> pProblem;

    // Ownership of objects handled by C++
    std::unique_ptr<AdjointProblem> adjoint;
    std::unique_ptr<SystemSolver> system;
    std::unique_ptr<Grid> grid;
    double tFinal;
};

#endif