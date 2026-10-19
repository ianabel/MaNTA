#ifndef PYGRID_HPP
#define PYGRID_HPP

// Helper functions for obtaining points fluxes and sources are evaluated at for
// Python output

#include "Postprocessing.hpp"
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

// The k+1 basis nodes of every cell, cell-major.
inline Vector basisNodesOn(Grid const &grid, unsigned int k) {
  Vector points((k + 1) * grid.getNCells());
  auto nodes = BasisType::getBasis(k).getNodes();
  for (size_t i = 0; i < grid.getNCells(); ++i) {
    auto const &cell = grid[i];

    for (auto j = 0; j < nodes.size(); ++j)
      points(i * (k + 1) + j) = cell.fromRef(nodes(j));
  }
  return points;
}

// For passing cell boundaries in
inline Vector getNodes(const std::vector<double> &cellBoundaries,
                       unsigned int k) {
  return basisNodesOn(Grid(cellBoundaries), k);
}

// For using MaNTA's grid structure
inline Vector getNodes(Position x_l, Position x_u, Index nCells,
                       unsigned int k) {
  return basisNodesOn(Grid(x_l, x_u, nCells), k);
}

// Where the physics is evaluated: the k+1 basis nodes of each cell, or its k+2
// star nodes with the superconvergent scheme, cell-major. A spatial adjoint
// parameter takes one value at each of these points, in this order, and G_p
// has one row per point.
inline Vector physicsNodesOn(Grid const &grid, unsigned int k,
                             bool superconvergent) {
  if (!superconvergent)
    return basisNodesOn(grid, k);
  const std::vector<Position> star = Postprocessor::starPointsOn(grid, k);
  return Eigen::Map<const Vector>(star.data(), star.size());
}

inline Vector getPhysicsNodes(const std::vector<double> &cellBoundaries,
                              unsigned int k, bool superconvergent) {
  return physicsNodesOn(Grid(cellBoundaries), k, superconvergent);
}

inline Vector getPhysicsNodes(Position x_l, Position x_u, Index nCells,
                              unsigned int k, bool superconvergent) {
  return physicsNodesOn(Grid(x_l, x_u, nCells), k, superconvergent);
}

#endif // PYGRID_HPP
