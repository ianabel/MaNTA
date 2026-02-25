#ifndef STATE_HPP
#define STATE_HPP

#include "Types.hpp"

class State
{
public:
    State() = default;
    explicit State(Index nv, Index ns = 0, Index naux = 0)
    {
        Variable.resize(nv);
        Derivative.resize(nv);
        Flux.resize(nv);
        Scalars.resize(ns);
        Aux.resize(naux);
    }

    void clone(const State &other)
    {
        Variable.resize(other.Variable.size());
        Derivative.resize(other.Derivative.size());
        Flux.resize(other.Flux.size());
        Scalars.resize(other.Scalars.size());
        Aux.resize(other.Aux.size());
    }

    void zero()
    {
        Variable.setZero();
        Derivative.setZero();
        Flux.setZero();
        Scalars.setZero();
        Aux.setZero();
    }

    Vector Variable, Derivative, Flux, Aux;
    Vector Scalars;
};

class GlobalState
{
public:
    GlobalState() = default;

    explicit GlobalState(Index nCells, Index k, Index nv, Index ns = 0, Index naux = 0) : nCells(nCells), k(k), nVars(nv), nScalars(ns), nAux(naux)
    {
        _Variable.resize(nVars, nCells * (k + 1));
        _Derivative.resize(nVars, nCells * (k + 1));
        _Flux.resize(nVars, nCells * (k + 1));
        _Aux.resize(nAux, nCells * (k + 1));
        _Scalars.resize(nScalars);
    }

    void setWithState(Index i, const State &s)
    {
        _Variable.col(i) = s.Variable;
        _Derivative.col(i) = s.Derivative;
        _Flux.col(i) = s.Flux;
        _Aux.col(i) = s.Aux;
        _Scalars = s.Scalars;
    }

    // Return state at point i
    State operator[](Index i) const
    {
        State out(nVars, nScalars, nAux);

        out.Variable = _Variable.col(i);
        out.Derivative = _Derivative.col(i);
        out.Flux = _Flux.col(i);
        out.Aux = _Aux.col(i);
        out.Scalars = _Scalars;

        return out;
    }

    // Need to do it like this because jax likes to transpose things
    GlobalState &operator=(const GlobalState &other)
    {
        Index rowCount = other.Variable().rows();
        Index colCount = other.Variable().cols();
        Index sz = nCells * (k + 1);
        if (rowCount == nVars && sz == colCount)
        {
            _Variable = other.Variable();
            _Derivative = other.Derivative();
            _Flux = other.Flux();
            _Aux = other.Aux();
            _Scalars = other.Scalars();
        }
        else if (rowCount == sz && colCount == nVars)
        {
            _Variable = other.Variable().transpose();
            _Derivative = other.Derivative().transpose();
            _Flux = other.Flux().transpose();
            _Aux = other.Aux().transpose();
            _Scalars = other.Scalars().transpose();
        }
        else
        {
            throw std::runtime_error("Global states must be the same size when copying.");
        }
        return *this;
    }

    /*
        Variable
    */
    // Accessor methods for translating between python and C++
    Matrix &Variable()
    {
        return _Variable;
    }
    const Matrix &Variable() const
    {
        return _Variable;
    }
    // Accessor methods for getting elements at a point or in a cell
    VectorRef Variable(Index i)
    {
        return _Variable.col(i);
    }
    // Grabs data on a whole cell for Jacobian computation, **implicitly assumes we're doing interpolation
    Eigen::Ref<Matrix> cellwiseVariable(Index cell)
    {
        return _Variable(Eigen::all, Eigen::seq(cell * (k + 1), (cell + 1) * (k + 1) - 1));
    }

    /*
        Derivative
    */
    Matrix &Derivative()
    {
        return _Derivative;
    }
    const Matrix &Derivative() const
    {
        return _Derivative;
    }
    VectorRef Derivative(Index i)
    {
        return _Derivative.col(i);
    }
    Eigen::Ref<Matrix> cellwiseDerivative(Index cell)
    {
        return _Derivative(Eigen::all, Eigen::seq(cell * (k + 1), (cell + 1) * (k + 1) - 1));
    }

    /*
        Flux
    */
    Matrix &Flux()
    {
        return _Flux;
    }
    const Matrix &Flux() const
    {
        return _Flux;
    }
    VectorRef Flux(Index i)
    {
        return _Flux.col(i);
    }
    Eigen::Ref<Matrix> cellwiseFlux(Index cell)
    {
        return _Flux(Eigen::all, Eigen::seq(cell * (k + 1), (cell + 1) * (k + 1) - 1));
    }

    /*
        Aux
    */
    Matrix &Aux()
    {
        return _Aux;
    }
    const Matrix &Aux() const
    {
        return _Aux;
    }
    VectorRef Aux(Index i)
    {
        return _Aux.col(i);
    }
    Eigen::Ref<Matrix> cellwiseAux(Index cell)
    {
        return _Aux(Eigen::all, Eigen::seq(cell * (k + 1), (cell + 1) * (k + 1) - 1));
    }

    /*
        Scalars
    */
    Vector &Scalars()
    {
        return _Scalars;
    }
    const Vector &Scalars() const
    {
        return _Scalars;
    }

    size_t size() const { return static_cast<size_t>(nCells * (k + 1)); }

private:
    // We hold global state data in matrices that are (nVars x nPoints)
    Matrix _Variable, _Derivative, _Flux, _Aux;

    // Scalars are global so this is just a vector
    Vector _Scalars;

    // Hold sizes internally for checking & preallocating memory
    Index nCells, k, nVars, nScalars, nAux;
};

// Wrapper class to make Jacobian computation cleaner
class GlobalStateMatrix
{
public:
    GlobalStateMatrix(Index nVars) : nVars(nVars) { data.reserve(nVars); };

    void add(Index nCells, Index k, Index nVars, Index nScalars, Index nAux)
    {
        data.emplace_back(nCells, k, nVars, nScalars, nAux);
    }

    void add(GlobalState &g_in)
    {
        data.push_back(g_in);
    }
    /*
        Returns vector of Matrix for per-cell operations, index like Variable[Var1](Var2, i),
        where i is within-cell index
    */
    std::vector<Matrix> Variable(Index cell)
    {
        std::vector<Matrix> out;

        for (Index var = 0; var < nVars; var++)
        {
            out.emplace_back(data[var].cellwiseVariable(cell));
        }
        return out;
    }

    std::vector<Matrix> Derivative(Index cell)
    {
        std::vector<Matrix> out;

        for (Index var = 0; var < nVars; var++)
        {
            out.emplace_back(data[var].cellwiseDerivative(cell));
        }
        return out;
    }

    std::vector<Matrix> Flux(Index cell)
    {
        std::vector<Matrix> out;

        for (Index var = 0; var < nVars; var++)
        {
            out.emplace_back(data[var].cellwiseFlux(cell));
        }
        return out;
    }

    GlobalState &operator[](Index var)
    {
        return data[var];
    }

private:
    std::vector<GlobalState> data;

    Index nVars;
};

#endif