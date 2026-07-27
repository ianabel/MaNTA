#ifndef PYINTEGRATOR_HPP
#define PYINTEGRATOR_HPP

#include "DGSoln.hpp"
#include "PyGrid.hpp"
#include "Types.hpp"

namespace Integrator {

inline const BasisType *m_basis = nullptr; // Pointer to singleton
// Save static values to avoid recomputation
inline std::map<Interval, Vector> integrationWeights;
inline Vector globalIntegrationWeights;
inline Matrix phiBoundary;

inline const Vector &getIntegrationWeights(Interval const &I) {
  if (integrationWeights.contains(I))
    return integrationWeights.at(I);
  else {
    integrationWeights.insert({I, m_basis->getIntegrationWeights(I)});
    return integrationWeights[I];
  }
}

inline const Vector &getIntegrationWeights(const BasisType &basis,
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

inline const Matrix &getPhiBoundary(const BasisType &basis, const Grid &grid) {

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
