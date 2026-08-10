#ifndef PYGRID_HPP
#define PYGRID_HPP

// Helper functions for obtaining points fluxes and sources are evaluated at for
// Python output

#include "Basis.hpp"
#include "gridStructures.hpp"

using BasisType = NodalBasis;

// `inline`, not `constexpr`. Both of these return a Vector -- a dynamically
// allocating Eigen type, so not a literal type -- and neither can ever be
// evaluated in a constant expression. C++23 permits declaring such a function
// constexpr as long as nobody tries (P2448R2), which is why gcc and clang 19+
// accept it, but clang 18 does not implement P2448R2 and rejects the declaration
// outright. The keyword bought nothing and cost a compiler.
//
// `inline` rather than nothing at all, because constexpr was implying inline and
// these are non-template definitions in a header. As it happens only one
// translation unit per link reaches them today -- Python.cpp in the module,
// PyIntegratorTests.cpp in the unit tests, and no solver TU at all -- so dropping
// the keyword outright would still have linked. It would break as soon as a second
// includer appeared, which is not a trap worth leaving behind.

// For passing cell boundaries in
inline Vector getNodes(const std::vector<double> &cellBoundaries,
                       unsigned int k) {
  Grid grid(cellBoundaries);
  Vector points((k + 1) * (cellBoundaries.size() - 1));
  auto nodes = BasisType::getBasis(k).getNodes();
  for (size_t i = 0; i < grid.getNCells(); ++i) {
    auto const &cell = grid[i];

    for (auto j = 0; j < nodes.size(); ++j)
      points(i * (k + 1) + j) = cell.fromRef(nodes(j));
  }
  return points;
}

// For using MaNTA's grid structure
inline Vector getNodes(Position x_l, Position x_u, Index nCells,
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
