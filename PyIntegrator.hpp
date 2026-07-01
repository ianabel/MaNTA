#ifndef PYINTEGRATOR_HPP
#define PYINTEGRATOR_HPP

#include "DGSoln.hpp"
#include "PyGrid.hpp"
#include "Types.hpp"

namespace Integrator {

// Save static values to avoid recomputation
static std::map<Interval, Vector> integrationWeights;
static std::map<Interval, std::vector<Vector>> phiCell;
static const BasisType *m_basis =
    nullptr; // This can be static because NodalBasis has singletons
static const Vector &getIntegrationWeights(Interval const &I) {
  if (integrationWeights.contains(I))
    return integrationWeights.at(I);
  else {
    integrationWeights.insert({I, m_basis->getIntegrationWeights(I)});
    return integrationWeights[I];
  }
}

static const std::vector<Values> &getPhiCell(Interval const &I) {
  if (phiCell.contains(I))
    return phiCell.at(I);
  else {
    const auto k = m_basis->Order();
    std::vector<Values> phis(k + 1);
    auto nodes = m_basis->getNodes();
    for (auto i = 0; i < k + 1; i++) {
      Position x = I.fromRef(nodes[i]);
      phis[i].resize(k + 1);
      for (auto j = 0; j < k + 1; j++)
        phis[i][j] = m_basis->Evaluate(I, j, x);
    }
    phiCell.insert({I, phis});
    return phiCell.at(I);
  }
}

class PyIntegrator {
public:
  explicit PyIntegrator(const Grid &grid, const BasisType &basis)
      : m_grid(grid) {
    if (!m_basis)
      m_basis = &basis;
  }
  ~PyIntegrator() = default;

  Value operator()(const Values &f) const {

    const auto k = m_basis->Order();
    Value out = 0.0;
    for (size_t i = 0; i < m_grid.getNCells(); i++) {

      const auto ind = Eigen::seq(i * (k + 1), (i + 1) * (k + 1) - 1);
      out += integrateOnCell(f(ind), i);
    }
    return out;
  };

  Value integrateOnCell(Values &&f, Index i) const {
    const Interval &I = m_grid[i];

    // https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas
    // integrate interpolation to get weights
    // compute integral as sum g * weights

    const auto weights = getIntegrationWeights(I);
    return (f.transpose() * weights).value();
  }

  Values cellProducts(const Values &f) const {
    Values out(f.size());
    const auto k = m_basis->Order();
    for (size_t i = 0; i < m_grid.getNCells(); i++) {
      const auto &phis = getPhiCell(m_grid[i]);
      const auto ind = Eigen::seq(i * (k + 1), (i + 1) * (k + 1) - 1);
      for (Index j = 0; j < k + 1; j++) {
        out(i * (k + 1) + j) = integrateOnCell(f(ind).cwiseProduct(phis[j]), i);
      }
    }
    return out;
  }

  Values Phi(Index i, Position x) {
    const Interval &I = m_grid[i];
    const auto k = m_basis->Order();
    Values out(k + 1);
    for (size_t j = 0; j < k + 1; j++)
      out(j) = m_basis->Evaluate(I, j, x);
    return out;
  }

private:
  Grid m_grid;
};
} // namespace Integrator
#endif // PYINTEGRATOR_HPP
