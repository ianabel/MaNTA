
#include <boost/test/unit_test.hpp>

#include "Types.hpp"
#include <toml.hpp>
#include "SystemSolver.hpp"
#include "TestDiffusion.hpp"

#include <nvector/nvector_serial.h>			/* access to serial N_Vector            */
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>		/* defs of realtype, sunindextype  */

#include "PhysicsCases/MatrixDiffusion.hpp"

using namespace toml::literals::toml_literals;

// raw string literal (`R"(...)"` is useful for this purpose)
const toml::value config_snippet = u8R"(
    [DiffusionProblem]
	 Kappa = 1.0
	 Centre = 0.0
)"_toml;

BOOST_TEST_DONT_PRINT_LOG_VALUE(Grid);
BOOST_TEST_DONT_PRINT_LOG_VALUE(Matrix);

BOOST_AUTO_TEST_SUITE(system_solver_test_suite)

BOOST_AUTO_TEST_CASE(systemsolver_init_tests)
{
	Grid testGrid(0.0, 1.0, 4);
	Index k = 1; // Start piecewise linear
	SystemSolver *system = nullptr;
	double tau = 0.5;

	TestDiffusion problem(config_snippet);
	BOOST_TEST(problem.Centre == 0.0);

	BOOST_CHECK_NO_THROW(system = new SystemSolver(testGrid, k, &problem));

	system->setTau(tau);

	system->resetCoeffs();

	BOOST_TEST(system->k == k);
	BOOST_TEST(system->grid == testGrid);
	BOOST_TEST(system->nVars == 1);

	BOOST_CHECK_NO_THROW(system->initialiseMatrices());

    DGSoln::basis_type const & basis = system->y.getBasis();

    // Should be the mass matrix (checking the mass matrix is correct is for the Chebyshev basis tests)
	BOOST_TEST((system->A_cellwise[0] - basis.MassMatrix( testGrid[0] ) ).norm() < 1e-9);
	BOOST_TEST((system->A_cellwise[1] - basis.MassMatrix( testGrid[1] ) ).norm() < 1e-9);
	BOOST_TEST((system->A_cellwise[2] - basis.MassMatrix( testGrid[2] ) ).norm() < 1e-9);
	BOOST_TEST((system->A_cellwise[3] - basis.MassMatrix( testGrid[3] ) ).norm() < 1e-9);

	Matrix ref(k + 1, k + 1);
	// Derivative matrix
	ref << 0.0, 13.85640646055103,
		0.0, 0.0;
	BOOST_TEST((system->B_cellwise[0] - basis.DerivativeMatrix( testGrid[0] ) ).norm() < 1e-9);
	BOOST_TEST((system->B_cellwise[1] - basis.DerivativeMatrix( testGrid[1] ) ).norm() < 1e-9);
	BOOST_TEST((system->B_cellwise[2] - basis.DerivativeMatrix( testGrid[2] ) ).norm() < 1e-9);
	BOOST_TEST((system->B_cellwise[3] - basis.DerivativeMatrix( testGrid[3] ) ).norm() < 1e-9);

	for (unsigned int i = 0; i < 4; i++)
    {
      ref.setZero();
      Interval const &I(testGrid[i]);
      for (Eigen::Index ii = 0; ii < k + 1; ii++)
      {
        for (Eigen::Index j = 0; j < k + 1; j++)
        {
          ref(ii, j) +=
            system->tau(I.x_l) * basis.Evaluate(I, j, I.x_l) * basis.Evaluate(I, ii, I.x_l) +
            system->tau(I.x_u) * basis.Evaluate(I, j, I.x_u) * basis.Evaluate(I, ii, I.x_u);
        }
      }
      BOOST_TEST((system->D_cellwise[i] - ref).norm() < 1e-9);

      ref.setZero();
      for(Eigen::Index j = 0; j < k + 1; j++)
      {
        ref(0,j) = -basis.Evaluate(I, j, I.x_l);
        ref(1,j) =  basis.Evaluate(I, j, I.x_u);
        if (i==0) {
          ref(0,j) = 0;
        }

        if (i==3) {
          ref(1,j) = 0;
        }
      }

      BOOST_TEST((system->C_cellwise[i] - ref).norm() < 1e-9);

      ref.setZero();
      for(Eigen::Index j = 0; j < k + 1; j++)
      {
        ref(j,0) = basis.Evaluate(I, j, I.x_l) * (-system->tau(I.x_l));
        ref(j,1) = basis.Evaluate(I, j, I.x_u) * (-system->tau(I.x_u));
        if (i==0) {
          ref(j,0) = 0;
        }

        if (i==3) {
          ref(j,1) = 0;
        }
      }
      BOOST_TEST((system->E_cellwise[i] - ref).norm() < 1e-9);
    }


	// Should check k > 1 here

	// Initial conditions
	SUNContext ctx;
	SUNContext_Create(SUN_COMM_NULL, &ctx);

	N_Vector y0, y0_dot;

	y0 = N_VNew_Serial(3 * 4 * (2) + 1 * (4 + 1), ctx);
	y0_dot = N_VClone(y0);
	system->setInitialConditions(y0, y0_dot);
	// Check y0 & y0dot

	DGSoln yMap(system->nVars, testGrid, k, N_VGetArrayPointer(y0));
	Vector lambdaRef(5);
	// Values of exp( -25x^2 ) at 0/0.25/0.5/0.75/1.0
	lambdaRef << 1.0, ::cos(M_PI / 8), ::cos(M_PI / 4), ::cos(3 * M_PI / 8), 0.0;
	BOOST_TEST((yMap.lambda(0) - lambdaRef).norm() < 0.025);
}

const toml::value config_snippet_2 = u8R"(
[DiffusionProblem]
nVars = 2
InitialHeights = [ 1.0, 1.0 ]

Kappa = [1.0,0.0,
        0.0,1.0]
)"_toml;

BOOST_AUTO_TEST_CASE(systemsolver_multichannel_init_tests)
{
	Grid testGrid(0.0, 1.0, 4);
	Index k = 1; // Start piecewise linear
	SystemSolver *system = nullptr;
	double tau = 0.5;

	MatrixDiffusion problem(config_snippet_2, testGrid);

	BOOST_CHECK_NO_THROW(system = new SystemSolver(testGrid, k, &problem));

	system->setTau(tau);
	system->resetCoeffs();

	BOOST_TEST(system->k == k);
	BOOST_TEST(system->grid == testGrid);
	BOOST_TEST(system->nVars == 2);

	BOOST_CHECK_NO_THROW(system->initialiseMatrices());

	Index N = 2 * (k + 1);
	BOOST_TEST((system->A_cellwise[0] - Matrix::Identity(N, N)).norm() < 1e-9);
	BOOST_TEST((system->A_cellwise[1] - Matrix::Identity(N, N)).norm() < 1e-9);
	BOOST_TEST((system->A_cellwise[2] - Matrix::Identity(N, N)).norm() < 1e-9);
	BOOST_TEST((system->A_cellwise[3] - Matrix::Identity(N, N)).norm() < 1e-9);

	Matrix ref(N, N);
	// Derivative matrix
	ref << 0.0, 13.85640646055103, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 13.85640646055103,
		0.0, 0.0, 0.0, 0.0;
	BOOST_TEST((system->B_cellwise[0] - ref).norm() < 1e-9);
	BOOST_TEST((system->B_cellwise[1] - ref).norm() < 1e-9);
	BOOST_TEST((system->B_cellwise[2] - ref).norm() < 1e-9);
	BOOST_TEST((system->B_cellwise[3] - ref).norm() < 1e-9);

	ref << 4.0, 0.0, 0.0, 0.0,
		0.0, 12.0, 0.0, 0.0,
		0.0, 0.0, 4.0, 0.0,
		0.0, 0.0, 0.0, 12.0;

	BOOST_TEST((system->D_cellwise[0] - ref).norm() < 1e-9);
	BOOST_TEST((system->D_cellwise[1] - ref).norm() < 1e-9);
	BOOST_TEST((system->D_cellwise[2] - ref).norm() < 1e-9);
	BOOST_TEST((system->D_cellwise[3] - ref).norm() < 1e-9);

	double TwoRootThree = 2.0 * ::sqrt(3.0);

	ref << 0.0, 0.0, 0.0, 0.0,
		2.0, TwoRootThree, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 2.0, TwoRootThree;

	BOOST_TEST((system->C_cellwise[0] - ref).norm() < 1e-9);

	ref << -2.0, TwoRootThree, 0.0, 0.0,
		2.0, TwoRootThree, 0.0, 0.0,
		0.0, 0.0, -2.0, TwoRootThree,
		0.0, 0.0, 2.0, TwoRootThree;

	BOOST_TEST((system->C_cellwise[1] - ref).norm() < 1e-9);
	BOOST_TEST((system->C_cellwise[2] - ref).norm() < 1e-9);

	ref << -2.0, TwoRootThree, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, -2.0, TwoRootThree,
		0.0, 0.0, 0.0, 0.0;

	BOOST_TEST((system->C_cellwise[3] - ref).norm() < 1e-9);

	double RootThree = ::sqrt(3.0);
	ref << 0.0, -1.0, 0.0, 0.0,
		0.0, -RootThree, 0.0, 0.0,
		0.0, 0.0, 0.0, -1.0,
		0.0, 0.0, 0.0, -RootThree;
	BOOST_TEST((system->E_cellwise[0] - ref).norm() < 1e-9);

	ref << -1.0, -1.0, 0.0, 0.0,
		RootThree, -RootThree, 0.0, 0.0,
		0.0, 0.0, -1.0, -1.0,
		0.0, 0.0, RootThree, -RootThree;

	BOOST_TEST((system->E_cellwise[1] - ref).norm() < 1e-9);
	BOOST_TEST((system->E_cellwise[2] - ref).norm() < 1e-9);
}

BOOST_AUTO_TEST_CASE(systemsolver_matrix_tests)
{
	Grid testGrid(0.0, 1.0, 4);
	Index k = 1; // Start piecewise linear

	SystemSolver *system = nullptr;
	double tau = 0.5;

	TestDiffusion problem(config_snippet);

	BOOST_CHECK_NO_THROW(system = new SystemSolver(testGrid, k, &problem));

	system->setTau(tau);
	system->resetCoeffs();

	BOOST_CHECK_NO_THROW(system->initialiseMatrices());

	SUNContext ctx;
	SUNContext_Create(SUN_COMM_NULL, &ctx);

	N_Vector y0, y0_dot;

	y0 = N_VNew_Serial(3 * 4 * (2) + 1 * (4 + 1), ctx);
	y0_dot = N_VClone(y0);
	system->setInitialConditions(y0, y0_dot);

	Matrix NLMat(k + 1, k + 1);

	for (Index i = 0; i < 4; ++i)
	{
		system->NLqMat(NLMat, system->y, testGrid[i]);
		BOOST_TEST((NLMat - Matrix::Identity(k + 1, k + 1)).norm() < 1e-9);
		system->NLuMat(NLMat, system->y, testGrid[i]);
		BOOST_TEST((NLMat - Matrix::Zero(k + 1, k + 1)).norm() < 1e-9);
	}

	delete system;
	MatrixDiffusion problem2(config_snippet_2, testGrid);

	BOOST_CHECK_NO_THROW(system = new SystemSolver(testGrid, k, &problem2));

	system->setTau(tau);
	BOOST_CHECK_NO_THROW(system->initialiseMatrices());

	N_VDestroy(y0);
	N_VDestroy(y0_dot);

	y0 = N_VNew_Serial(2 * (3 * 4 * (2) + 1 * (4 + 1)), ctx);
	y0_dot = N_VClone(y0);
	system->setInitialConditions(y0, y0_dot);

	NLMat.resize(2 * (k + 1), 2 * (k + 1));

	for (Index i = 0; i < 4; ++i)
	{
		system->NLqMat(NLMat, system->y, testGrid[i]);
		BOOST_TEST((NLMat - Matrix::Identity(2 * (k + 1), 2 * (k + 1))).norm() < 1e-9);
		system->NLuMat(NLMat, system->y, testGrid[i]);
		BOOST_TEST((NLMat - Matrix::Zero(2 * (k + 1), 2 * (k + 1))).norm() < 1e-9);
	}
}

BOOST_AUTO_TEST_SUITE_END()
