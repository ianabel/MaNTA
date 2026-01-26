#ifndef DGAPPROX_HPP
#define DGAPPROX_HPP

#include "gridStructures.hpp"

#include <map>
#include <memory>
#include <algorithm>
#include <boost/math/quadrature/gauss.hpp>
#include <cmath>
#include <vector>
#include <iostream>

#include <Eigen/LU>

template<class BasisType> class DGApproxImpl
{
    public:
        using Position = double;

        DGApproxImpl() = delete;
        DGApproxImpl(const DGApproxImpl<BasisType> &other) = default;

        ~DGApproxImpl() = default;

        DGApproxImpl(Grid const &_grid, BasisType const& basis )
            : grid(_grid), Basis( basis )
        {
            k = Basis.Order();
            coeffs.reserve(grid.getNCells());
        };

        DGApproxImpl(Grid const &_grid, BasisType const& basis, double *block_data, size_t stride) 
            : grid(_grid), Basis( basis )
        {
            k = Basis.Order();
            Grid::Index nCells = grid.getNCells();
            coeffs.clear();
            coeffs.reserve(nCells);
            auto const &cells = grid.getCells();
            for (Grid::Index i = 0; i < nCells; ++i)
            {
                VectorWrapper v(block_data + i * stride, k + 1);
                coeffs.emplace_back(cells[i], v);
            }
        }

        void Map(double *block_data, size_t stride)
        {
            Grid::Index nCells = grid.getNCells();
            coeffs.clear();
            coeffs.reserve(nCells);
            if (stride < k + 1)
                throw std::invalid_argument("stride too short, memory corrption guaranteed.");
            auto const &cells = grid.getCells();
            for (Grid::Index i = 0; i < nCells; ++i)
            {
                VectorWrapper v(block_data + i * stride, k + 1);
                coeffs.emplace_back(cells[i], v);
            }
        }

        // Do a copy from other's memory into ours
        void copy(DGApproxImpl<BasisType> const &other)
        {
            if (grid != other.grid)
                throw std::invalid_argument("To use copy, construct from the same grid.");
            if (k != other.k)
                throw std::invalid_argument("Cannot change order of polynomial approximation via copy().");

            coeffs = other.coeffs;
        }

        DGApproxImpl<BasisType> &operator=(std::function<double(double)> const &f)
        {
            // check for data ownership
            for (auto pair : coeffs)
            {
                Interval const &I = pair.first;
                // Project onto polynomials
                // (u_fn,phi_j) = Sum_i u_i (phi_i,phi_j) = M^T . u_vec (M is the Mass Matrix)
                // => u_vec = M^T^-1( (u_fn,phi_j) )
                pair.second = Basis.MassMatrix(I).transpose().inverse() * Basis.ProjectOntoBasis( I, f );
            }
            return *this;
        }

        size_t getDoF() const { return (k + 1) * grid.getNCells(); };

        DGApproxImpl<BasisType> &operator+=(DGApproxImpl<BasisType> const &other)
        {
            if (grid != other.grid)
                throw std::invalid_argument("Cannot add two DGApprox's on different grids");
            for (unsigned int i = 0; i < coeffs.size(); ++i)
            {
                // std::assert( coeffs[ i ].first == other.coeffs[ i ].first );
                coeffs[i].second += other.coeffs[i].second;
            }
            return *this;
        }

        double operator()(Position x) const
        {
            constexpr double eps = 1e-15;
            if (x < grid.lowerBoundary() && ::fabs(x - grid.lowerBoundary()) < eps)
                x = grid.lowerBoundary();
            else if (x > grid.upperBoundary() && ::fabs(x - grid.upperBoundary()) < eps)
                x = grid.upperBoundary();

            for (auto const &I : coeffs)
            {
                if (I.first.contains(x))
                    return Basis.Evaluate(I.first, I.second, x);
            }
            std::cerr << "x = " << x << ", grid bounds = [" << grid.lowerBoundary() << ", " << grid.upperBoundary() << "]" << std::endl;
            throw std::logic_error("Evaluation outside of grid");
        };

        double operator()(Position x, Interval const &I) const
        {
            if (!I.contains(x))
                throw std::invalid_argument("Evaluate(x, I) requires x to be in the interval I");
            auto it = std::find_if(coeffs.begin(), coeffs.end(), [I](std::pair<Interval, VectorWrapper> p)
                    { return (p.first == I); });
            if (it == coeffs.end())
                throw std::logic_error("Interval I not part of the grid");
            else
                return Basis.Evaluate(it->first, it->second, x);
        };

        

        void zeroCoeffs()
        {
            for (auto pair : coeffs)
                pair.second = Vector::Zero(pair.second.size());
        }

        void printCoeffs()
        {
            for (auto const &x : coeffs)
            {
                std::cerr << x.second << std::endl;
            }
            std::cerr << std::endl;
        }

        double maxCoeff()
        {
            double coeff = 0.0;
            for (auto pair : coeffs)
            {
                if (::abs(coeff) < ::abs(pair.second.maxCoeff()))
                    coeff = pair.second.maxCoeff();
            }
            return coeff;
        }

        using Coeff_t = std::vector<std::pair<Interval, VectorWrapper>>;

        static const BasisType::IntegratorType &Integrator() { return BasisType::integrator; };

        unsigned int getOrder() { return k; };
        Coeff_t const &getCoeffs() { return coeffs; };
        std::pair<Interval, VectorWrapper> &getCoeff(Index i) { return coeffs[i]; };
        std::pair<Interval, VectorWrapper> const &getCoeff(Index i) const { return coeffs[i]; };

    private:
        const Grid &grid;
        unsigned int k;
        Coeff_t coeffs;
        const BasisType& Basis;

        friend class DGSolnImpl<BasisType>;
};

#endif
