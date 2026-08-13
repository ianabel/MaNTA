#include "../../DGSoln.hpp"
#include "../../gridStructures.hpp"
#include "Eigen/src/Core/Matrix.h"
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>

constexpr static std::array<Index, 3> kvals{4, 5, 6};
constexpr static std::array<Index, 4> nCellsvals{7, 10, 11, 18};

constexpr static auto u0 = 1.2;
constexpr static auto M0 = 2 * u0 + 4 / M_PI;
// Chebyshev nodes on [0, 1]
const Grid *makeNonUniform(Index nCells) {

  std::vector<double> cell_boundaries(nCells + 1);

  for (unsigned int i = 0; i < nCells + 1; i++)
    cell_boundaries[nCells - i] = std::cos(i * M_PI / (nCells));

  return new Grid(cell_boundaries);
}

BOOST_AUTO_TEST_SUITE(integrator_tests, *boost::unit_test::tolerance(1e-6))

double testIntegral(Index nCells, Index k, const Grid &grid) {
  auto nPoints = nCells * (k + 1);
  Vector integrationWeights(nPoints);
  Vector fval(nPoints);

  const NodalBasis basis = NodalBasis::getBasis(k);
  const auto test_function = [](double x) { return u0 + std::cos(M_PI_2 * x); };

  for (size_t i = 0; i < grid.getNCells(); i++) {

    const auto ind = Eigen::seq(i * (k + 1), (i + 1) * (k + 1) - 1);
    const auto cellWeights = basis.getIntegrationWeights(grid[i]);
    integrationWeights(ind) = cellWeights;

    auto nodes = basis.getNodes();
    for (Index j = 0; j < k + 1; j++) {
      fval(i * (k + 1) + j) = test_function(grid[i].fromRef(nodes(j)));
    }
  }

  return fval.dot(integrationWeights);
}

BOOST_AUTO_TEST_CASE(uniform_grid_test) {
  for (const auto &k : kvals) {
    for (const auto &nCells : nCellsvals) {

      const Grid uniform_grid(-1.0, 1.0, nCells);
      auto val = testIntegral(nCells, k, uniform_grid);
      BOOST_TEST(val == M0);
    }
  }
}

BOOST_AUTO_TEST_CASE(nonuniform_grid_test) {

  for (const auto &k : kvals) {
    for (const auto &nCells : nCellsvals) {
      const Grid nonuniform_grid = *makeNonUniform(nCells);

      auto val = testIntegral(nCells, k, nonuniform_grid);
      BOOST_TEST(val == M0);
    }
  }
}

inline const void getPhiCell(MatrixRef phiCell, NodalBasis *basis,
                             Interval const &I) {

  const auto k = basis->Order();
  Matrix phis(k + 1, k + 1);
  auto nodes = basis->getNodes();
  for (Index i = 0; i < k + 1; i++) {
    Position x = I.fromRef(nodes[i]);
    for (Index j = 0; j < k + 1; j++)
      phiCell(i, j) = basis->Evaluate(I, j, x);
  }
}

Vector nodalJacTest(std::function<double(double)> const &f,
                    std::function<double(double)> const &dfdu, Index nCells,
                    Index k, const Grid &grid) {

  Matrix globalCellWeights(k + 1, nCells * (k + 1));
  NodalBasis basis = NodalBasis::getBasis(k);
  Vector cellProducts(nCells * (k + 1));
  cellProducts.setZero();
  for (Index i = 0; i < nCells; i++) {
    integrator_tests::getPhiCell(globalCellWeights.middleCols(i * (k + 1), k + 1),
                                 &basis, grid[i]);
    const auto cellWeights = basis.getIntegrationWeights(grid[i]);

    auto nodes = basis.getNodes();
    for (Index j = 0; j < k + 1; j++) {

      double u = f(grid[i].fromRef(nodes(j)));

      cellProducts(j + i * (k + 1)) = dfdu(u) * cellWeights[j];
    }
  }
  return cellProducts;
}

Vector jacTest(std::function<double(double)> const &f,
               std::function<double(double)> const &dfdu, Index nCells, Index k,
               const Grid &grid) {

  Matrix globalCellWeights(k + 1, nCells * (k + 1));
  NodalBasis basis = NodalBasis::getBasis(k);
  Vector cellProducts(nCells * (k + 1));
  Vector points(nCells * (k + 1));

  boost::math::quadrature::gauss_kronrod<double, 15> integrator;

  for (Index i = 0; i < nCells; i++) {
    const auto &I = grid[i];
    for (Index j = 0; j < k + 1; j++) {

      auto cellf = [&](double x) {
        return dfdu(f(x)) * basis.Evaluate(I, j, x);
      };

      cellProducts(j + i * (k + 1)) = integrator.integrate(cellf, I.x_l, I.x_u);
    }
  }
  return cellProducts;
}

BOOST_AUTO_TEST_CASE(cell_product_test, *boost::unit_test::tolerance(5e-3)) {

  const auto test_function = [](double x) {
    return M_PI_2 * std::cos(M_PI_2 * x);
  };

  // const auto test_G = [](double u) { return u * u; };
  const auto test_dgdu = [](double u) { return 2 * u; };

  for (const auto &k : kvals) {
    for (const auto &nCells : nCellsvals) {
      const Grid nonuniform_grid = *makeNonUniform(nCells);

      const Grid uniform_grid(-1.0, 1.0, nCells);

      auto nodal_uniform =
          nodalJacTest(test_function, test_dgdu, nCells, k, uniform_grid);

      auto nodal_nonuniform =
          nodalJacTest(test_function, test_dgdu, nCells, k, nonuniform_grid);

      auto test_uniform =
          jacTest(test_function, test_dgdu, nCells, k, uniform_grid);

      auto test_nonuniform =
          jacTest(test_function, test_dgdu, nCells, k, nonuniform_grid);

      BOOST_TEST((nodal_uniform - test_uniform).norm() == 0.0);
      BOOST_TEST((nodal_nonuniform - test_nonuniform).norm() == 0.0);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
