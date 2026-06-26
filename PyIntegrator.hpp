#ifndef PYINTEGRATOR_HPP
#define PYINTEGRATOR_HPP

#include "DGSoln.hpp"
#include "PyGrid.hpp"
#include "Types.hpp"

class PyIntegrator {
public:
  explicit PyIntegrator(const Grid &grid, const BasisType &basis)
      : m_grid(grid), m_basis(basis) {}
  ~PyIntegrator() = default;

  Value operator()(const Values &f) const {

    Value out = 0.0;
    for (size_t i = 0; i < m_grid.getNCells(); i++) {
      out += integrateOnCell(f, i);
    }
    return out;
  };

  Value integrateOnCell(const Values &f, Index i) const {
    Value out = 0.0;
    const Interval &I = m_grid[i];

    // https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas
    // integrate interpolation to get weights
    // compute integral as sum g * weights
    const auto k = m_basis.Order();

    const auto ind = Eigen::seq(i * (k + 1), (i + 1) * (k + 1) - 1);

    const auto weights = m_basis.getIntegrationWeights(I);
    const auto &f_cellwise = f(ind);
    out += f_cellwise.dot(weights);
  }

  Value Phi(Index i, Index j, Position x) {
    const Interval &I = m_grid[i];
    return m_basis.Evaluate(I, j, x);
  }

private:
  Grid m_grid;
  BasisType m_basis;
};

#endif // PYINTEGRATOR_HPP
