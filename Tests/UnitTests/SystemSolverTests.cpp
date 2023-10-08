
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
)"_toml;

BOOST_TEST_DONT_PRINT_LOG_VALUE(Grid);
BOOST_TEST_DONT_PRINT_LOG_VALUE(Matrix);

BOOST_AUTO_TEST_SUITE(system_solver_test_suite)

BOOST_AUTO_TEST_CASE(systemsolver_init_tests)
{
	Grid testGrid(0.0, 1.0, 4);
	Index k = 1; // Start piecewise linear
	SystemSolver *system = nullptr;
	double dt = 0.1;
	double tau = 0.5;

	TestDiffusion problem(config_snippet);

	BOOST_CHECK_NO_THROW(system = new SystemSolver(testGrid, k, dt, tau, &problem));

	system->resetCoeffs();

	BOOST_TEST(system->k == k);
	BOOST_TEST(system->grid == testGrid);
	BOOST_TEST(system->nVars == 1);

	BOOST_TEST((system->A_cellwise[0] - Matrix::Identity(k + 1, k + 1)).norm() < 1e-9);
	BOOST_TEST((system->A_cellwise[1] - Matrix::Identity(k + 1, k + 1)).norm() < 1e-9);
	BOOST_TEST((system->A_cellwise[2] - Matrix::Identity(k + 1, k + 1)).norm() < 1e-9);
	BOOST_TEST((system->A_cellwise[3] - Matrix::Identity(k + 1, k + 1)).norm() < 1e-9);

	Matrix ref(k + 1, k + 1);
	// Derivative matrix
	ref << 0.0, 13.85640646055103,
		0.0, 0.0;
	BOOST_TEST((system->B_cellwise[0] - ref).norm() < 1e-9);
	BOOST_TEST((system->B_cellwise[1] - ref).norm() < 1e-9);
	BOOST_TEST((system->B_cellwise[2] - ref).norm() < 1e-9);
	BOOST_TEST((system->B_cellwise[3] - ref).norm() < 1e-9);

	ref << 4.0, 0.0,
		0.0, 12.0;

	BOOST_TEST((system->D_cellwise[0] - ref).norm() < 1e-9);
	BOOST_TEST((system->D_cellwise[1] - ref).norm() < 1e-9);
	BOOST_TEST((system->D_cellwise[2] - ref).norm() < 1e-9);
	BOOST_TEST((system->D_cellwise[3] - ref).norm() < 1e-9);

	double TwoRootThree = 2.0 * ::sqrt(3.0);

	ref << 0.0, 0.0,
		2.0, TwoRootThree;
	BOOST_TEST((system->C_cellwise[0] - ref).norm() < 1e-9);

	ref << -2.0, TwoRootThree,
		2.0, TwoRootThree;
	BOOST_TEST((system->C_cellwise[1] - ref).norm() < 1e-9);
	BOOST_TEST((system->C_cellwise[2] - ref).norm() < 1e-9);

	ref << -2.0, TwoRootThree,
		0.0, 0.0;
	BOOST_TEST((system->C_cellwise[3] - ref).norm() < 1e-9);

	double RootThree = ::sqrt(3.0);
	ref << 0.0, -1.0,
		0.0, -RootThree;
	BOOST_TEST((system->E_cellwise[0] - ref).norm() < 1e-9);

	ref << -1.0, -1.0,
		RootThree, -RootThree;
	BOOST_TEST((system->E_cellwise[1] - ref).norm() < 1e-9);
	BOOST_TEST((system->E_cellwise[2] - ref).norm() < 1e-9);

	ref << -1.0, 0.0,
		RootThree, 0.0;
	BOOST_TEST((system->E_cellwise[3] - ref).norm() < 1e-9);

	// Should check k > 1 here

	// Initial conditions
	SUNContext ctx;
	SUNContext_Create(nullptr, &ctx);

	N_Vector y0, y0_dot;

	y0 = N_VNew_Serial(3 * 4 * (2) + 1 * (4 + 1), ctx);
	y0_dot = N_VClone(y0);
	system->setInitialConditions(y0, y0_dot);
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
	double dt = 0.1;

	MatrixDiffusion problem(config_snippet_2);

	BOOST_CHECK_NO_THROW(system = new SystemSolver(testGrid, k, dt, tau, &problem));

	system->resetCoeffs();

	BOOST_TEST(system->k == k);
	BOOST_TEST(system->grid == testGrid);
	BOOST_TEST(system->nVars == 2);

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
	double dt = 0.1;
	double tau = 0.5;

	TestDiffusion problem(config_snippet);

	BOOST_CHECK_NO_THROW(system = new SystemSolver(testGrid, k, dt, tau, &problem));

	system->resetCoeffs();

	SUNContext ctx;
	SUNContext_Create(nullptr, &ctx);

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
	MatrixDiffusion problem2(config_snippet_2);

	BOOST_CHECK_NO_THROW(system = new SystemSolver(testGrid, k, dt, tau, &problem2));

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
