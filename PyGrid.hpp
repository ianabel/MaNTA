#ifndef PYGRID_HPP
#define PYGRID_HPP

// Helper functions for obtaining points fluxes and sources are evaluated at for
// Python output

#include "Basis.hpp"
#include "gridStructures.hpp"

using BasisType = NodalBasis;

// For passing cell boundaries in
constexpr Vector getNodes(const std::vector<double> &cellBoundaries,
                          unsigned int k) {
  Grid grid(cellBoundaries);
  Vector points((k + 1) * cellBoundaries.size());
  auto nodes = BasisType::getBasis(k).getNodes();
  for (size_t i = 0; i < grid.getNCells(); ++i) {
    auto const &cell = grid[i];

    for (auto j = 0; j < nodes.size(); ++j)
      points(i * (k + 1) + j) = cell.fromRef(nodes(j));
  }
  return points;
}

// For using MaNTA's grid structure
constexpr Vector getNodes(Position x_l, Position x_u, Index nCells,
                          unsigned int k) {
  Grid grid(x_l, x_u, nCells);
  Vector points(nCells * (k + 1));
  auto nodes = BasisType::getBasis(k).getNodes();
  for (size_t i = 0; i < grid.getNCells(); ++i) {
    auto const &cell = grid[i];

    for (auto j = 0; j < nodes.size(); ++j)
      points(i * (k + 1) + j) = cell.fromRef(nodes(j));
  }
  return points;
}

#endif // PYGRID_HPP
