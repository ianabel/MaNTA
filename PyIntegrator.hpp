#ifndef PYINTEGRATOR_HPP
#define PYINTEGRATOR_HPP

#include "DGSoln.hpp"
#include "PyGrid.hpp"
#include "Types.hpp"

namespace Integrator {

static const BasisType *m_basis =
    nullptr; // This can be static because NodalBasis has singletons
// Save static values to avoid recomputation
static std::map<Interval, Vector> integrationWeights;
static std::map<Interval, Matrix> phiCell;
static Vector globalIntegrationWeights;
static Matrix globalCellWeights;
static Matrix phiBoundary;

static const Vector &getIntegrationWeights(Interval const &I) {
  if (integrationWeights.contains(I))
    return integrationWeights.at(I);
  else {
    integrationWeights.insert({I, m_basis->getIntegrationWeights(I)});
    return integrationWeights[I];
  }
}

static const Vector &getIntegrationWeights(const BasisType &basis,
                                           const Grid &grid) {
  if (!m_basis)
    m_basis = &basis;
  if (globalIntegrationWeights.size() == 0) {
    auto const k = m_basis->Order();
    globalIntegrationWeights.resize(grid.getNCells() * (k + 1));

    for (Index i = 0; i < grid.getNCells(); i++) {
      const auto ind = Eigen::seq(i * (k + 1), (i + 1) * (k + 1) - 1);
      globalIntegrationWeights(ind) = getIntegrationWeights(grid[i]);
    }
  }
  return globalIntegrationWeights;
}

static const Matrix &getPhiCell(Interval const &I) {
  if (phiCell.contains(I))
    return phiCell.at(I);
  else {
    const auto k = m_basis->Order();
    Matrix phis(k + 1, k + 1);
    auto nodes = m_basis->getNodes();
    for (auto i = 0; i < k + 1; i++) {
      Position x = I.fromRef(nodes[i]);
      for (auto j = 0; j < k + 1; j++)
        phis(i, j) = m_basis->Evaluate(I, j, x);
    }
    phiCell.insert({I, phis});
    return phiCell.at(I);
  }
}

static const Matrix &getPhiCell(const BasisType &basis, const Grid &grid) {
  if (!m_basis)
    m_basis = &basis;
  if (globalCellWeights.size() == 0) {
    auto const k = m_basis->Order();
    globalCellWeights.resize(k + 1, grid.getNCells() * (k + 1));
    for (Index i = 0; i < grid.getNCells(); i++) {
      Vector temp(grid.getNCells() * (k + 1));
      auto const &phiCell = getPhiCell(grid[i]);
      for (Index j = 0; j < k + 1; j++) {
        const auto ind = Eigen::seq(i * (k + 1), (i + 1) * (k + 1) - 1);
        globalCellWeights(j, ind) = phiCell(Eigen::all, j);
      }
    }
  }
  return globalCellWeights;
}

static const Matrix &getPhiBoundary(const BasisType &basis, const Grid &grid) {

  if (!m_basis)
    m_basis = &basis;

  if (phiBoundary.size() == 0) {
    const Interval &I_l = grid[0];
    const Interval &I_u = grid[grid.getNCells() - 1];

    const auto k = m_basis->Order();
    phiBoundary.resize(k + 1, 2);

    for (Index i = 0; i < k + 1; i++) {
      phiBoundary(i, 0) = m_basis->Evaluate(I_l, i, I_l.x_l);
      phiBoundary(i, 1) = m_basis->Evaluate(I_u, i, I_u.x_u);
    }
  }
  return phiBoundary;
}
}; // namespace Integrator
#endif // PYINTEGRATOR_HPP
