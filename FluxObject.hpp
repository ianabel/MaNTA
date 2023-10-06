#ifndef FLUXOBJECT_HPP
#define FLUXOBJECT_HPP

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include "Types.hpp"

using namespace autodiff;

struct FluxObject
{
public:
    virtual ~FluxObject() {}

    typedef dual (*sigmaptr)(VectorXdual u, VectorXdual q, dual x, double t);
    typedef dual (*sourceptr)(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);

    typedef std::map<Index, sigmaptr> sigmaMap;
    typedef std::map<Index, sourceptr> sourceMap;

    sigmaMap sigma;
    sourceMap source;

protected:
    Index nVars;
};

#endif