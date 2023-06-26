#include "SystemSolver.hpp"
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */
#include <Eigen/Core>
#include <Eigen/Dense>
#include <toml.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include "gridStructures.hpp"
#include "DiffusionObj.hpp"
#include "SourceObj.hpp"
#include "InitialConditionLibrary.hpp"

void makePlasmaCase(std::string const& plasmaCase, std::shared_ptr<Plasma>& plasmaPtr);

SystemSolver::SystemSolver(Grid const& Grid, unsigned int polyNum, unsigned int N_cells, unsigned int N_Variables, double Dt, Fn const& rhs, Fn const& Tau, Fn const& c)
	: grid(Grid), k(polyNum), nCells(N_cells), nVar(N_Variables), dt(Dt), RHS(rhs), tau(Tau), c_fn(c), sig(grid,k), q(grid,k), u(grid,k), dudt(grid,k), dqdt(grid,k), dsigdt(grid,k), BCs (nullptr)
{
	seta_fns();
}

SystemSolver::SystemSolver(std::string const& inputFile)
{
	const auto configFile = toml::parse( inputFile );
	const auto config = toml::find<toml::value>( configFile, "configuration" );

	//Solver parameters
	double lBound, uBound;

	auto polyDegree = toml::find(config, "Polynomial_degree");
	if( !config.count("Polynomial_degree") == 1 ) throw std::invalid_argument( "Polynomial_degree unspecified or specified more than once" );
	else if( !polyDegree.is_integer() ) throw std::invalid_argument( "Polynomial_degree must be specified as an integer" );
	else k = polyDegree.as_integer();

	if ( config.count( "High_Grid_Boundary" ) != 1 ) highGridBoundary = false;
	else
	{
		std::string denseEdges = config.at( "High_Grid_Boundary" ).as_string();
		if (denseEdges == "true") highGridBoundary = true;
		else if(denseEdges == "false") highGridBoundary = false;
		else throw std::invalid_argument( "high_Grid_Boundary specified incorrrectly" );
	}

	auto numberOfCells = toml::find(config, "Grid_size");
	if( !config.count("Grid_size") == 1 ) throw std::invalid_argument( "Grid_size unspecified or specified more than once" );
	if( !numberOfCells.is_integer() ) throw std::invalid_argument( "Grid_size must be specified as an integer" );
	else nCells = numberOfCells.as_integer();

	if(nCells<4 && highGridBoundary)
		throw std::invalid_argument( "Grid size must exceed 4 cells in order to implemet dense boundaries" );
	if(highGridBoundary) nCells += 8;

	auto numberOfVariables = toml::find(config, "Number_of_channels");
	if( !config.count("Number_of_channels") == 1 ) throw std::invalid_argument( "Number_of_channels unspecified or specified more than once" );
	if( !numberOfVariables.is_integer() ) throw std::invalid_argument( "Number_of_channels must be specified as an integer" );
	else nVar = numberOfVariables.as_integer();

	//Initial Conditions
	if ( config.count( "Initial_condition" ) != 1 )
		throw std::invalid_argument( "[error] Initial_condition must be specified once in the [configuration] block" );
	std::string initCondition = config.at( "Initial_condition" ).as_string();

	if( config.count( "Plasma_case" ) == 0 )
	{
		//??TO DO: Obseletize this code and replace with plasma cases??
		// Non-linear inputs
		// Diffusion
		if ( config.count( "Diffusion_case" ) != 1 )
			throw std::invalid_argument( "[error] Diffusion_case must be specified once in the [configuration] block" );
		std::string diffusionCase = config.at( "Diffusion_case" ).as_string();
		auto diffobj = std::make_shared< DiffusionObj >(k, nVar, diffusionCase);
		// Reaction
		if ( config.count( "Reaction_case" ) != 1 )
			throw std::invalid_argument( "[error] Reaction_case must be specified once in the [configuration] block" );
		std::string reactionCase = config.at( "Reaction_case" ).as_string();
		auto sourceobj = std::make_shared< SourceObj >(k, nVar, reactionCase);

		initConditionLibrary.set(initCondition, diffusionCase);

		setDiffobj(diffobj);
		setSourceobj(sourceobj);
	}
	else if( config.count( "Plasma_case" ) == 1 )
	{
		std::string plasmaCase = config.at( "Plasma_case" ).as_string();
		makePlasmaCase(plasmaCase, plasma);
		plasma->constructPlasma();

		setDiffobj(plasma->diffObj);
		setSourceobj(plasma->sourceObj);

		initConditionLibrary.set(initCondition);
	}
	seta_fns();

	//---------Boundary Conditions---------------
	auto lowerBoundary = toml::find(config, "Lower_boundary");
	if( !config.count("Lower_boundary") == 1 ) throw std::invalid_argument( "Lower_boundary unspecified or specified more than once" );
	else if( lowerBoundary.is_integer() ) lBound = static_cast<double>(lowerBoundary.as_floating());
	else if( lowerBoundary.is_floating() ) lBound = static_cast<double>(lowerBoundary.as_floating());
	else throw std::invalid_argument( "Lower_boundary specified incorrrectly" );

	auto upperBoundary = toml::find(config, "Upper_boundary");
	if( !config.count("Upper_boundary") == 1 ) throw std::invalid_argument( "Upper_boundary unspecified or specified more than once" );
	else if( upperBoundary.is_integer() ) uBound = static_cast<double>(upperBoundary.as_floating());
	else if( upperBoundary.is_floating() ) uBound = static_cast<double>(upperBoundary.as_floating());
	else throw std::invalid_argument( "Upper_boundary specified incorrrectly" );

	grid = Grid(lBound, uBound, nCells, highGridBoundary);

	sig.k = k; q.k = k; u.k = k;
	dudt.k = k; dqdt.k = k; dsigdt.k = k;

	std::function<double( double, double, int )> g_D_ = [ = ]( double x, double t, int var ) {
		if ( x == lBound ) {
			//if(t>0.5) return 0.5;
			//else return 1.0-t;
			if( var == 2) return 10000.0;
			else return 10.0;
		} else if ( x == uBound ) {
			// u(1.0) == a
			//if(t<0.99999) return 1.0 - t;
			//else return 0.00001;
			if( var == 2) return 10000.0;
			else return 10.0;
		}
		throw std::logic_error( "Boundary condition function being eval'd not on boundary ?!" );
	};

	std::function<double( double, double, int )> g_N_ = [ = ]( double x, double t, int var ) {
		if ( x == lBound ) {
			// ( q + c u ) . n  a @ x = 0.0
			return 0.0;
		} else if ( x == uBound ) {
			// ( q + c u ) . n  b @ x = 1.0
			return 0.0;
		}
		throw std::logic_error( "Boundary condition function being eval'd not on boundary ?!" );
	};

	auto DirichletBCs = std::make_shared<BoundaryConditions>();
	DirichletBCs->LowerBound = lBound;
	DirichletBCs->UpperBound = uBound;

	if ( config.count( "LB_Type" ) != 1 )
		throw std::invalid_argument( "[error] LB_Type must be specified once in the [configuration] block" );
	std::string LBoundary_Condition = config.at( "LB_Type" ).as_string();
	if (LBoundary_Condition == "Dirichlet") DirichletBCs->isLBoundDirichlet = true;
	else if(LBoundary_Condition == "VonNeumann") DirichletBCs->isLBoundDirichlet = false;
	else throw std::invalid_argument( "LB_type specified incorrrectly" );

	if ( config.count( "UB_Type" ) != 1 )
		throw std::invalid_argument( "[error] UB_Type must be specified once in the [configuration] block" );
	std::string UBoundary_Condition = config.at( "UB_Type" ).as_string();
	if (UBoundary_Condition == "Dirichlet") DirichletBCs->isUBoundDirichlet = true;
	else if(UBoundary_Condition == "VonNeumann") DirichletBCs->isUBoundDirichlet = false;
	else throw std::invalid_argument( "UB_type specified incorrrectly" );

	DirichletBCs->g_D = g_D_;
	DirichletBCs->g_N = g_N_;
	setBoundaryConditions(DirichletBCs);

	//-------------End of Boundary conditions------------------------------------
}

void SystemSolver::setInitialConditions( N_Vector& Y , N_Vector& dYdt )
{
	setInitialConditions(initConditionLibrary.getuInitial(), initConditionLibrary.getqInitial(), initConditionLibrary.getSigInitial(), Y, dYdt);
}

void SystemSolver::setInitialConditions( std::function< double ( double, int )> u_0, std::function< double ( double, int )> gradu_0, std::function< double ( double, int )> sigma_0, N_Vector& Y , N_Vector& dYdt ) 
{
	if ( !initialised )
		initialiseMatrices();

	if(!lambda.has_value())
	{
		double memBlock[nVar*(nCells+1)];
		lambda = VectorWrapper(memBlock, nVar*(nCells+1));
		dlamdt = VectorWrapper(memBlock, nVar*(nCells+1));
	}
	mapDGtoSundials( sig, q, u, lambda.value(), N_VGetArrayPointer( Y ));
	mapDGtoSundials( dsigdt, dqdt, dudt, dlamdt.value(), N_VGetArrayPointer( dYdt ));

	resetCoeffs();

	u = u_0;
	q = gradu_0;

	//??TO DO: remove sigma from initial condition library all together
	if(plasma){ sig = [ = ]( double R, int var ){return -1.0*plasma->getVariable(var).kappaFunc(R,q,u);}; }
	else sig = sigma_0;

	double dx = ::abs(BCs->UpperBound - BCs->LowerBound)/nCells;
	for(int var = 0; var < nVar; var++)
	{
		for ( unsigned int i=0; i < nCells; i++ )
		{
			Interval I = grid.gridCells[ i ];
			lambda.value()[var*(nCells+1) + i] += u.Basis.Evaluate(I, u.coeffs[var][i].second, BCs->LowerBound + i*(dx))/2;
			lambda.value()[var*(nCells+1) + i+1] += u.Basis.Evaluate(I, u.coeffs[var][i].second, BCs->LowerBound + (i+1)*(dx))/2;

			dlamdt.value()[var*(nCells+1) + i] += dudt.Basis.Evaluate(I, dudt.coeffs[var][i].second, BCs->LowerBound + i*(dx))/2;
			dlamdt.value()[var*(nCells+1) + i+1] += dudt.Basis.Evaluate(I, dudt .coeffs[var][i].second, BCs->LowerBound + (i+1)*(dx))/2;
		}
		lambda.value()[var*(nCells+1)] = BCs->g_D(grid.lowerBound, static_cast<double>(0.0),var);
		lambda.value()[var*(nCells+1) + nCells] = BCs->g_D(grid.upperBound, static_cast<double>(0.0),var);
	}
	
	/*
	Eigen::VectorXd CsGuL_global(nVar*(nCells+1));
	CsGuL_global.setZero();
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Eigen::VectorXd LVarCell(2);
		Eigen::VectorXd CsGuLVarCell(2);
		CsGuLVarCell.setZero();
		LVarCell.setZero();
		for(int var = 0; var < nVar; var++)
		{
			LVarCell = L_global.block<2,1>(var*(nCells+1) + i,0);
			CsGuLVarCell = LVarCell - C_cellwise[i].block(var*2,var*(k+1),2,k+1)*sig.coeffs[ var ][ i ].second - G_cellwise[i].block(var*2,var*(k+1),2,k+1)*u.coeffs[ var ][ i ].second;
			CsGuL_global.block(var*(nCells + 1) + i, 0, 2, 1) += CsGuLVarCell;
		}
	}
	lambda.value() = H_global.solve(CsGuL_global);
	*/

	for(int var = 0; var < nVar; var++)
	{
		//Solver For dudt with dudt = X^-1( -B*Sig - D*U - E*Lam + F )
		Eigen::Vector2d lamCell;
		for ( unsigned int i=0; i < nCells; i++ )
		{
			Interval I = grid.gridCells[ i ];

			std::function< double (double) > sourceFunc = [ = ]( double x ) { return getSourceObj()->getSourceFunc(var)( x, sig, q, u); };
			//Evaluate Source Function
			Eigen::VectorXd S_cellwise(k+1);
			S_cellwise.setZero();
			for ( Eigen::Index j = 0; j < k+1; j++ )
				S_cellwise( j ) = u.CellProduct( I, sourceFunc, u.Basis.phi( I, j ) );
			
			auto cTInv = Eigen::FullPivLU< Eigen::MatrixXd >(C_cellwise[i].transpose());
			lamCell[0] = lambda.value()[var*(nCells+1) + i]; lamCell[1] = lambda.value()[var*(nCells+1) + i+1];
			//dudt.coeffs[ var ][ i ].second.setZero();
			dudt.coeffs[ var ][ i ].second = XMats[i].block(var*(k+1), var*(k+1), k+1, k+1).inverse()*(-B_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*sig.coeffs[ var ][ i ].second - D_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*u.coeffs[ var ][ i ].second - E_cellwise[i].block(var*(k+1), var*2, k+1, 2)*lamCell + RF_cellwise[ i ].block( nVar*(k + 1) + var*(k+1), 0, k + 1, 1 ) - S_cellwise);
			if(var==2) std::cerr << -B_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*sig.coeffs[ var ][ i ].second - D_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*u.coeffs[ var ][ i ].second - E_cellwise[i].block(var*(k+1), var*2, k+1, 2)*lamCell + RF_cellwise[ i ].block( nVar*(k + 1) + var*(k+1), 0, k + 1, 1 ) - S_cellwise << std::endl << (XMats[i].block(var*(k+1), var*(k+1), k+1, k+1)*dudt.coeffs[ var ][ i ].second) << std::endl << std::endl;
		}
	}
}

void SystemSolver::seta_fns()
{
	if(plasma)
	{
		for(int var = 0; var < nVar; var++)
		{
			a_fn.push_back(plasma->getVariable(var).a_fn);
		}
	}
	//TO DO: this is to be retired
	else
	{
		for(int var = 0; var < nVar; var++)
		{
			a_fn.push_back([ = ]( double R ){ return 1.0;});
		}
	}
}

void SystemSolver::initialiseMatrices()
{
	// These are temporary working space
	// Matrices we need per cell
	Eigen::MatrixXd A( nVar*(k + 1), nVar*(k + 1) );
	Eigen::MatrixXd B( nVar*(k + 1), nVar*(k + 1) );
	Eigen::MatrixXd D( nVar*(k + 1), nVar*(k + 1) );
	// Two endpoints per cell
	Eigen::MatrixXd C( 2*nVar, nVar*(k + 1) );
	Eigen::MatrixXd E( nVar*(k + 1), 2*nVar );

	// Tomporary variable matrices that will be printed into the larger cell matrices as blocks
	Eigen::MatrixXd Avar( k + 1, k + 1 );
	Eigen::MatrixXd Bvar( k + 1, k + 1 );
	Eigen::MatrixXd Dvar( k + 1, k + 1 );
	Eigen::MatrixXd Cvar( 2, k + 1 );
	Eigen::MatrixXd Evar( k + 1, 2 );

	Eigen::MatrixXd HGlobalMat( nVar*(nCells+1), nVar*(nCells+1) );
	HGlobalMat.setZero();
	K_global.resize( nVar*(nCells + 1), nVar*(nCells + 1) );
	K_global.setZero();
	L_global.resize( nVar*(nCells + 1) );
	L_global.setZero();

	clearCellwiseVecs();
	for ( unsigned int i = 0; i < nCells; i++ )
	{
		A.setZero();
		B.setZero();
		C.setZero();
		D.setZero();
		E.setZero();
		Interval const& I( grid.gridCells[ i ] );
		for(int var=0; var<nVar; var++)
		{
			Avar.setZero();
			Bvar.setZero();
			Dvar.setZero();
			// A_ij = ( phi_j, phi_i )
			u.MassMatrix( I, Avar);
			// B_ij = ( phi_i, phi_j' )
			u.DerivativeMatrix( I, Bvar );
			// D_ij = -(c phi_j, phi_i') + < w, tau u > 
			u.DerivativeMatrix( I, Dvar, c_fn );
			// As DerivativeMatrix gives the weighted product (c phi_i, phi_j')
			// we flip the sign and the indices on D.
			Dvar *= -1.0;
			Dvar.transposeInPlace();

			// Now do all the boundary terms
			for ( Eigen::Index i=0; i<k+1;i++ )
			{
				for ( Eigen::Index j=0; j<k+1;j++ )
				{
					Dvar( i, j ) += 
						tau( I.x_l )*u.Basis.phi( I, j )( I.x_l )*u.Basis.phi( I, i )( I.x_l ) +
						tau( I.x_u )*u.Basis.phi( I, j )( I.x_u )*u.Basis.phi( I, i )( I.x_u )	;
				}
			}

			A.block(var*(k+1),var*(k+1),k+1,k+1) = Avar;
			D.block(var*(k+1),var*(k+1),k+1,k+1) = Dvar;
			B.block(var*(k+1),var*(k+1),k+1,k+1) = Bvar;
		}

		A_cellwise.emplace_back(A);
		B_cellwise.emplace_back(B);
		D_cellwise.emplace_back(D);

		Eigen::MatrixXd M( 3*nVar*(k + 1), 3*nVar*(k + 1) );
		M.setZero();
		//row1
		M.block( 0, 0, nVar*(k+1), nVar*(k+1) ).setZero();
		M.block( 0, nVar*(k+1), nVar*(k+1), nVar*(k+1) ) = -A;
		M.block( 0, 2*nVar*(k+1), nVar*(k+1), nVar*(k+1) ) = -B.transpose();

		//row2
		M.block( nVar*(k+1), 0, nVar*(k+1), nVar*(k+1) ) = B;
		M.block( nVar*(k+1), nVar*(k+1), nVar*(k+1), nVar*(k+1) ).setZero();
		M.block( nVar*(k+1), 2*nVar*(k+1), nVar*(k+1), nVar*(k+1) ) = D;			//X added at Jac step

		//row3
		M.block( 2*nVar*(k+1), 0, nVar*(k+1), nVar*(k+1) ) = A;
		M.block( 2*nVar*(k+1), nVar*(k+1), nVar*(k+1), nVar*(k+1) ).setZero();		//NLq added at Jac step
		M.block( 2*nVar*(k+1), 2*nVar*(k+1), nVar*(k+1), nVar*(k+1) ).setZero();	//NLu added at Jac step

		// 2*( k + 1) is normally only medium-sized (10 - 20)
		// so just do a full LU factorization to solve 
		// ?Now this is nVar*3*(k+1) so maybe this should be changed?
		MBlocks.emplace_back( M );

		Eigen::MatrixXd CE_vec( 3*nVar*(k + 1), 2*nVar );
		CE_vec.setZero();
		for(int var=0; var<nVar; var++)
		{
			Cvar.setZero();
			Evar.setZero();
			for ( Eigen::Index i=0; i<k+1;i++ )
			{
				// C_ij = < psi_i, phi_j * n_x > , where psi_i are edge degrees of
				// freedom and n_x is the unit normal in the x direction
				// for a line, edge degrees of freedom are just 1 at each end

				Cvar( 0, i ) = -u.Basis.phi( I, i )( I.x_l );
				Cvar( 1, i ) = u.Basis.phi( I, i )( I.x_u ); 

				// E_ij = < phi_i, (c n_x - tau ) lambda >
				Evar( i, 0 ) = u.Basis.phi( I, i )( I.x_l ) * ( -c_fn( I.x_l ) - tau( I.x_l ) );
				Evar( i, 1 ) = u.Basis.phi( I, i )( I.x_u ) * (  c_fn( I.x_u ) - tau( I.x_u ) );

				if ( I.x_l == BCs->LowerBound && BCs->isLBoundDirichlet )
				{
					Cvar( 0, i ) = 0;
					Evar( i, 0 ) = 0;
				}

				if ( I.x_u == BCs->UpperBound && BCs->isUBoundDirichlet )
				{
					Cvar( 1, i ) = 0;
					Evar( i, 1 ) = 0;
				}
			}

			// Construct per-cell Matrix solutions
			// ( 0    A    B^T )    [ C^T ]
			// ( B    0     D )     [  E  ]
			// ( A   NLu   NLq )^-1 [  0  ]
			// These are the homogeneous solution, that depend on lambda
			C.block(var*2,var*(k+1),2,k+1) = Cvar;
			E.block(var*(k+1),var*2,k+1,2) = Evar;
		}
		
		CE_vec.block( 0, 0, nVar*(k + 1), nVar*2 ) = C.transpose();
		CE_vec.block( nVar*(k+1), 0, nVar*(k + 1), nVar*2 ) = E;
		CE_vec.block( 2*nVar*(k+1), 0, nVar*(k + 1), nVar*2 ).setZero();
		CEBlocks.emplace_back(CE_vec);

		C_cellwise.emplace_back(C);
		E_cellwise.emplace_back(E);

		// To store the RHS
		RF_cellwise.emplace_back(Eigen::VectorXd(nVar*2*(k+1)));

		// R is composed of parts of the values of 
		// u on the total domain boundary
		RF_cellwise[ i ].setZero();
		// Take components of f
		for ( Eigen::Index j = 0; j < nVar*(k+1); j++ )
			RF_cellwise[ i ]( nVar*(k + 1) + j ) = u.CellProduct( I, RHS, u.Basis.phi( I, j%(k+1) ) );

		if ( I.x_l == BCs->LowerBound  && BCs->isLBoundDirichlet )
		{
			for(int var = 0; var < nVar; var++)
			{
				for ( Eigen::Index j = 0; j < k+1; j++ )
				{
					// < g_D , v . n > ~= g_D( x_0 ) * phi_j( x_0 ) * ( n_x = -1 ) 
					RF_cellwise[ i ]( j + var*(k+1) ) += -u.Basis.phi( I, j )( I.x_l ) * ( -1 ) * BCs->g_D( I.x_l, 0.0, var );
					// - < ( c.n - tau ) g_D, w >
					RF_cellwise[ i ]( nVar*(k + 1) + j + var*(k+1) ) -= u.Basis.phi( I, j )( I.x_l ) * ( -c_fn( I.x_l ) - tau( I.x_l ) ) * BCs->g_D( I.x_l, 0.0, var );
				}
			}
		}

		if ( I.x_u == BCs->UpperBound && BCs->isUBoundDirichlet )
		{
			for(int var = 0; var < nVar; var++)
			{
				for ( Eigen::Index j = 0; j < k+1; j++ )
				{
					// < g_D , v . n > ~= g_D( x_1 ) * phi_j( x_1 ) * ( n_x = +1 ) 
					RF_cellwise[ i ]( j + var*(k+1) ) += -u.Basis.phi( I, j )( I.x_u ) * ( +1 ) * BCs->g_D( I.x_u, 0.0, var );
					RF_cellwise[ i ]( nVar*(k + 1) + j + var*(k+1) ) -= u.Basis.phi( I, j )( I.x_u ) * (  c_fn( I.x_u ) - tau( I.x_u ) ) * BCs->g_D( I.x_u, 0.0, var );
				}
			}
		}

		// Per-cell contributions to the global matrices K and F.
		// First fill G
		Eigen::MatrixXd G( 2*nVar, nVar*(k + 1) );
		G.setZero();
		for(int var = 0; var < nVar; var++)
		{
			Eigen::MatrixXd Gvar( 2, k + 1 );
			for ( Eigen::Index i = 0; i < k+1; i++ )
			{
				Gvar( 0, i ) = tau( I.x_l )*u.Basis.phi( I, i )( I.x_l );
				if ( I.x_l == BCs->LowerBound && BCs->isLBoundDirichlet )
					Gvar( 0, i ) = 0.0;
				Gvar( 1, i ) = tau( I.x_u )*u.Basis.phi( I, i )( I.x_u );
				if ( I.x_u == BCs->UpperBound && BCs->isUBoundDirichlet )
					Gvar( 1, i ) = 0.0;
			}
			G.block(2*var,(k+1)*var,2, (k + 1)) = Gvar;
		}

		//[ C 0 G ]
		CG_cellwise.emplace_back(Eigen::MatrixXd(2*nVar, 3*nVar*(k+1) ));
		CG_cellwise[i].setZero();
		CG_cellwise[ i ].block( 0, 0, 2*nVar, nVar*(k + 1) ) = C;
		CG_cellwise[ i ].block( 0, nVar*(k + 1), 2*nVar, nVar*(k + 1) ).setZero();
		CG_cellwise[ i ].block( 0, 2*nVar*(k + 1), 2*nVar, nVar*(k + 1) ) = G;
		G_cellwise.emplace_back(G);

		// Now fill H
		Eigen::MatrixXd H( 2*nVar, 2*nVar );
		H.setZero();
		for(int var = 0; var < nVar; var++)
		{
			Eigen::MatrixXd Hvar( 2, 2 );
			Hvar.setZero();
			Hvar( 0, 0 ) = -c_fn( I.x_l ) - tau( I.x_l );
			Hvar( 1, 0 ) = 0.0;
			Hvar( 0, 1 ) = 0.0;
			Hvar( 1, 1 ) = c_fn( I.x_u ) - tau( I.x_u );

			if ( I.x_l == BCs->LowerBound && BCs->isLBoundDirichlet )
					Hvar( 0, 0 ) = Hvar( 1, 0 ) = Hvar( 0, 1 ) = 0.0;

			if ( I.x_u == BCs->UpperBound && BCs->isUBoundDirichlet )
					Hvar( 1, 1 ) = Hvar( 1, 0 ) = Hvar( 0, 1 ) = 0.0;

			H.block(2*var,2*var,2,2) = Hvar;
			HGlobalMat.block(var*(nCells+1) + i,var*(nCells+1) + i, 2, 2) += Hvar;
		}

		H_cellwise.emplace_back(H);

		// Finally fill L
		for(int var = 0; var < nVar; var++)
		{
			if ( I.x_l == BCs->LowerBound && /* is b.d. Neumann at lower boundary */ !BCs->isLBoundDirichlet )
				L_global( var*(nCells+1) + i )     += BCs->g_N( BCs->LowerBound, 0.0, var );
			if ( I.x_u == BCs->UpperBound && /* is b.d. Neumann at upper boundary */ !BCs->isUBoundDirichlet )
				L_global( var*(nCells+1) + i + 1 ) += BCs->g_N( BCs->UpperBound, 0.0, var );
		}

		Eigen::MatrixXd X(nVar*(k+1), nVar*(k+1));
		X.setZero();
		for(int var = 0; var < nVar; var++)
		{
			Eigen::MatrixXd Xvar( k + 1, k + 1 );
			u.MassMatrix( I, Xvar, a_fn[var]);
			X.block(var*(k+1), var*(k+1), k+1, k+1) = Xvar;
		}
		XMats.emplace_back(X);
	}
	H_global = static_cast<Eigen::FullPivLU< Eigen::MatrixXd >>(HGlobalMat);
	H_global_mat = HGlobalMat;
	initialised = true;
}

void SystemSolver::clearCellwiseVecs()
{
	XMats.clear();
	MBlocks.clear();
	CG_cellwise.clear();
	RF_cellwise.clear();
	A_cellwise.clear();
	B_cellwise.clear();
	E_cellwise.clear();
	C_cellwise.clear();
	G_cellwise.clear();
	H_cellwise.clear();
}

void SystemSolver::mapDGtoSundials(DGApprox& sigma, DGApprox& q, DGApprox& u, Eigen::Map<Eigen::VectorXd>& lam, realtype* Y)
{
	std::vector< std::pair< Interval, Eigen::Map<Eigen::VectorXd >>> cellSigCoeffs, cellQCoeffs, cellUCoeffs;

	for(int var = 0; var < nVar; var++)
	{
		for(int i=0; i<nCells; i++)
		{
			cellSigCoeffs.emplace_back( grid.gridCells[ i ], VectorWrapper( Y +                var*(k+1) + i*3*nVar*(k+1), k+1 ));
			cellQCoeffs.emplace_back( grid.gridCells[ i ],   VectorWrapper( Y + nVar*(k+1)   + var*(k+1) + i*3*nVar*(k+1), k+1 ));
			cellUCoeffs.emplace_back( grid.gridCells[ i ],   VectorWrapper( Y + 2*nVar*(k+1) + var*(k+1) + i*3*nVar*(k+1), k+1 ));
		}
		sigma.coeffs.push_back(cellSigCoeffs);
		q.coeffs.push_back(cellQCoeffs);
		u.coeffs.push_back(cellUCoeffs);
		cellSigCoeffs.clear();
		cellQCoeffs.clear();
		cellUCoeffs.clear();

		new (&lam) VectorWrapper( Y + nVar*(nCells)*(3*k+3), nVar*(nCells+1) );
	}
}

void SystemSolver::mapDGtoSundials(DGApprox& u, realtype* Y)
{
	std::vector< std::pair< Interval, Eigen::Map<Eigen::VectorXd >>> cellUCoeffs;

	for(int var = 0; var < nVar; var++)
	{
		for(int i=0; i<nCells; i++)
		{
			cellUCoeffs.emplace_back( grid.gridCells[ i ], VectorWrapper( Y + 2*nVar*(k+1) + var*(k+1) + i*3*nVar*(k+1), k+1 ));  
		}
		u.coeffs.push_back(cellUCoeffs);
		cellUCoeffs.clear();
	}
}

void SystemSolver::mapDGtoSundials(std::vector< Eigen::Map<Eigen::VectorXd > >& SQU_cell, Eigen::Map<Eigen::VectorXd>& lam, realtype* const& Y)
{
	SQU_cell.clear();
	for(int i=0; i<nCells; i++)
	{
		SQU_cell.emplace_back( VectorWrapper( Y + i*3*nVar*(k+1), nVar*3*(k+1) ) );
	}

		new (&lam) VectorWrapper( Y + nVar*(nCells)*(3*k+3), nVar*(nCells+1) );
}

void SystemSolver::updateBoundaryConditions(double t)
{
	L_global.setZero();
	for ( unsigned int i = 0; i < nCells; i++ )
	{
		Interval const& I( grid.gridCells[ i ] );
		RF_cellwise[ i ].setZero();

		//??To Do: this should be a time dependent RHS function. Fairly easy implementation
		for ( Eigen::Index j = 0; j < nVar*(k+1); j++ )
			RF_cellwise[ i ]( nVar*(k + 1) + j ) = u.CellProduct( I, RHS, u.Basis.phi( I, j%(k+1) ) );

		if ( I.x_l == BCs->LowerBound  && BCs->isLBoundDirichlet )
		{
			for(int var = 0; var < nVar; var++)
			{
				for ( Eigen::Index j = 0; j < k+1; j++ )
				{
					// < g_D , v . n > ~= g_D( x_0 ) * phi_j( x_0 ) * ( n_x = -1 ) 
					RF_cellwise[ i ]( j + var*(k+1) ) += -u.Basis.phi( I, j )( I.x_l ) * ( -1 ) * BCs->g_D( I.x_l, t, var );
					// - < ( c.n - tau ) g_D, w >
					RF_cellwise[ i ]( nVar*(k + 1) + j + var*(k+1) ) -= u.Basis.phi( I, j )( I.x_l ) * ( -c_fn( I.x_l ) - tau( I.x_l ) ) * BCs->g_D( I.x_l, t, var );
				}
			}
		}


		if ( I.x_u == BCs->UpperBound && BCs->isUBoundDirichlet )
		{
			for(int var = 0; var < nVar; var++)
			{
				for ( Eigen::Index j = 0; j < k+1; j++ )
				{
					// < g_D , v . n > ~= g_D( x_1 ) * phi_j( x_1 ) * ( n_x = +1 ) 
					RF_cellwise[ i ]( j+var*(k+1)) += -u.Basis.phi( I, j )( I.x_u ) * ( +1 ) * BCs->g_D( I.x_u, t, var );
					RF_cellwise[ i ]( nVar*(k + 1) + j + var*(k+1) ) -= u.Basis.phi( I, j )( I.x_u ) * (  c_fn( I.x_u ) - tau( I.x_u ) ) * BCs->g_D( I.x_u, t, var );
				}
			}
		}


		for(int var = 0; var < nVar; var++)
		{
			if ( I.x_l == BCs->LowerBound && /* is b.d. Neumann at lower boundary */ !BCs->isLBoundDirichlet )
				L_global( var*(nCells+1) + i )     += BCs->g_N( BCs->LowerBound, t, var );
			if ( I.x_u == BCs->UpperBound && /* is b.d. Neumann at upper boundary */ !BCs->isUBoundDirichlet )
				L_global( var*(nCells+1) + i + 1 ) += BCs->g_N( BCs->UpperBound, t, var );
		}
	}
}

Vector SystemSolver::resEval(std::vector<Vector> resTerms)
{
	Vector maxTerms, res;
	maxTerms.resize(resTerms[0].size());
	res.resize(resTerms[0].size());
	maxTerms.setZero();
	res.setZero();
	for(auto term : resTerms)
	{
		res += term;
		for(int i = 0; i< term.size(); i++)
		{
			if( std::abs(term[i]) > maxTerms[i]) maxTerms[i] = std::abs(term[i]);
		}
	}
	for(int i = 0; i< resTerms[0].size(); i++)
	{
		if(res[i] < maxTerms[i]*10e-10) res[i] = 0.0;
	}
	return res;
}

void SystemSolver::resetCoeffs()
{
	sig.zeroCoeffs();
	q.zeroCoeffs();
	u.zeroCoeffs();
	dsigdt.zeroCoeffs();
	dqdt.zeroCoeffs();
	dudt.zeroCoeffs();
	lambda.value().setZero();
	dlamdt.value().setZero();
}

void SystemSolver::updateMForJacSolve(std::vector< Eigen::FullPivLU< Eigen::MatrixXd > >& MXsolvers, double const alpha, DGApprox& delSig, DGApprox& delQ, DGApprox& delU)
{
	MXsolvers.clear();
	DGApprox newU(grid, k), newQ(grid, k), newSig(grid, k);
	double qMemBlock[nVar*nCells*(k+1)], uMemBlock[nVar*nCells*(k+1)], sigMemBlock[nVar*nCells*(k+1)]; //??need to assign memory block as DGAs don't own memory
	newQ.setCoeffsToArrayMem(qMemBlock, nVar, nCells, grid);
	newU.setCoeffsToArrayMem(uMemBlock, nVar, nCells, grid);
	newSig.setCoeffsToArrayMem(sigMemBlock, nVar, nCells, grid);

	//We want to base our matrix off the current guess
	newQ.sum(q, delQ);
	newU.sum(u, delU);
	newSig.sum(sig, delSig);


	Eigen::MatrixXd X( nVar*(k + 1), nVar*(k + 1) );
	Eigen::MatrixXd NLq(nVar*(k+1), nVar*(k+1));
	Eigen::MatrixXd NLu(nVar*(k+1), nVar*(k+1));
	Eigen::MatrixXd Ssig(nVar*(k+1), nVar*(k+1));
	Eigen::MatrixXd Sq(nVar*(k+1), nVar*(k+1));
	Eigen::MatrixXd Su(nVar*(k+1), nVar*(k+1));
	for ( unsigned int i = 0; i < nCells; i++ )
	{
		X.setZero();
		NLq.setZero();
		NLu.setZero();
		Ssig.setZero();
		Sq.setZero();
		Su.setZero();

		Interval const& I( grid.gridCells[ i ] );
		Eigen::MatrixXd MX(3*nVar*(k+1),3*nVar*(k+1));
		MX.setZero();
		MX = MBlocks[i];
		//X matrix
		for(int var = 0; var < nVar; var++)
		{
		std::function<double( double )> alphaF = [ = ]( double x ){ return alpha*a_fn[var](x);};
			Eigen::MatrixXd Xsubmat( (k + 1), (k + 1) );
			Xsubmat.setZero();
			u.MassMatrix( I, Xsubmat, alphaF);
			X.block(var*(k+1), var*(k+1), k+1, k+1) = Xsubmat;
		}
		MX.block( nVar*(k+1), 2*nVar*(k+1), nVar*(k+1), nVar*(k+1) ) += X;

		//NLq Matrix
		diffObj->NLqMat( NLq, newQ, newU, I);
		MX.block( 2*nVar*(k+1), nVar*(k+1), nVar*(k+1), nVar*(k+1)) = NLq;

		//NLu Matrix
		diffObj->NLuMat( NLu, newQ, newU, I);
		MX.block( 2*nVar*(k+1), 2*nVar*(k+1), nVar*(k+1), nVar*(k+1)) = NLu;

		//S_sig Matrix
		sourceObj->setdFdsigMat( Ssig, newSig, newQ, newU, I);
		MX.block( nVar*(k+1), nVar*(k+1), nVar*(k+1), nVar*(k+1) ) = Ssig;

		//S_q Matrix
		sourceObj->setdFdqMat( Sq, newSig, newQ, newU, I);
		MX.block( nVar*(k+1), nVar*(k+1), nVar*(k+1), nVar*(k+1) ) = Sq;

		//S_u Matrix
		sourceObj->setdFduMat( Su, newSig, newQ, newU, I);
		MX.block( nVar*(k+1), 2*nVar*(k+1), nVar*(k+1), nVar*(k+1) ) += Su;

		//if(i==0) std::cerr << MX << std::endl << std::endl;
		//if(i==0)std::cerr << MX.inverse() << std::endl << std::endl;

		MXsolvers.emplace_back(MX);
	}
}

void SystemSolver::solveJacEq(N_Vector& g, N_Vector& delY)
{
	DGApprox delSig(grid,k), delQ(grid,k), delU(grid,k);
	double memBlock[nVar*(nCells+1)];
	Eigen::Map<Eigen::VectorXd> delLambda(memBlock, nVar*(nCells+1)), g4(memBlock, nVar*(nCells+1));
	std::vector< Eigen::Map<Eigen::VectorXd > > g1g2g3_cellwise;
	K_global.setZero(); 

	VectorWrapper delYVec( N_VGetArrayPointer( delY ), N_VGetLength( delY ) );
	mapDGtoSundials(delSig, delQ, delU, delLambda, N_VGetArrayPointer( delY ));
	delYVec.setZero();

	//assemble temp cellwise M blocks
	std::vector< Eigen::FullPivLU< Eigen::MatrixXd > > factorisedM{};
	updateMForJacSolve(factorisedM, alpha, delSig, delQ, delU);

	// Assemble RHS g into cellwise form and solve for SQU blocks
	mapDGtoSundials(g1g2g3_cellwise, g4, N_VGetArrayPointer( g ));

	std::vector< Eigen::VectorXd > SQU_f( nCells );
	std::vector< Eigen::MatrixXd > SQU_0( nCells );
	for ( unsigned int i = 0; i < nCells; i++ )
	{  
		Interval const& I( grid.gridCells[ i ] );

		//SQU_f
		Eigen::VectorXd g1g2g3 = g1g2g3_cellwise[ i ];

		SQU_f[ i ] = factorisedM[ i ].solve( g1g2g3 );


		//SQU_0
		Eigen::MatrixXd CE = CEBlocks[ i ];
		SQU_0[ i ] = factorisedM[ i ].solve( CE );
		//std::cerr << SQU_0[i] << std::endl << std::endl;
		//std::cerr << CE << std::endl << std::endl;


		Eigen::MatrixXd K_cell(nVar*2,nVar*2);
		K_cell.setZero();
		K_cell = H_cellwise[i] - CG_cellwise[ i ] * SQU_0[i];

		//std::cerr << SQU_f[i] << std::endl << std::endl;
		//K
		for(int var = 0; var < nVar; var++)
		{
			K_global.block( var*(nCells + 1) + i, var*(nCells + 1) + i, 2, 2 ) += K_cell.block(var*2,var*2,2,2);
		}
	}

	// Construct the RHS of K Lambda = F
	Eigen::VectorXd F( nVar*(nCells + 1) );
	F.setZero();
	F = g4;
	for ( unsigned int i=0; i < nCells; i++ )
	{
		for(int var = 0; var < nVar; var++)
		{
			F.block<2,1>( var*(nCells + 1) + i, 0 ) -= (CG_cellwise[ i ] * SQU_f[ i ]).block(var*2,0,2,1);
		}
	}

	Eigen::FullPivLU< Eigen::MatrixXd > lu( K_global );
	delLambda = lu.solve( F );
	//std::cerr << delLambda << std::endl << std::endl;

	// Now find del sigma, del q and del u to eventually find del Y
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval const& I = grid.gridCells[ i ];
		Eigen::VectorXd delSQU( 3*nVar*( k + 1 ) );
		Eigen::VectorXd delLambdaCell(2*nVar);
		for(int var = 0; var < nVar; var++)
		{
			delLambdaCell.block<2,1>(2*var,0) = delLambda.block<2,1>(var*(nCells + 1) + i,0);
		}
		delSQU = SQU_f[ i ] - SQU_0[ i ] * delLambdaCell;
		for(int var = 0; var < nVar; var++)
		{
			delSig.coeffs[ var ][ i ].second = delSQU.block( var*(k+1), 0, k + 1, 1 );
			delQ.coeffs[ var ][ i ].second =   delSQU.block( nVar*(k + 1) + var*(k+1), 0, k + 1, 1 );
			delU.coeffs[ var ][ i ].second =   delSQU.block( 2*nVar*(k + 1) + var*(k+1), 0, k + 1, 1 );
		}
	}
	std::ofstream dyfile;
	dyfile.open("dyfile.txt");
	print(dyfile, 0.0, 200, 0, delY);
	//delSig.printCoeffs(0);
	//delU.printCoeffs(0);
	//delQ.printCoeffs(0);
	//std::cerr << delLambda << std::endl << std::endl;
}

int residual(realtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data)
{
	auto system = static_cast<UserData*>(user_data)->system;
	auto k = system->k;
	auto grid(system->grid);
	auto nCells = system->nCells;
	auto c_fn = system->getcfn();
	auto nVar = system->nVar;

	system->updateBoundaryConditions(tres);

	DGApprox tempSig(grid, k), tempU(grid, k), tempQ(grid, k), tempdudt(grid, k);
	DGApprox tempdSigdt(grid, k), tempdQdt(grid, k);
	DGApprox res1(grid, k), res2(grid, k), res3(grid, k) ;
	DGApprox tempKappa(grid, k);
	double memBlock[nVar*(nCells+1)];
	Eigen::Map<Eigen::VectorXd> tempLambda(memBlock, nVar*(nCells+1)), res4(memBlock, nVar*(nCells+1)), lam(memBlock, nVar*(nCells+1)), tempdLamdt(memBlock, nVar*(nCells+1));

	system->mapDGtoSundials(tempSig, tempQ, tempU, tempLambda, N_VGetArrayPointer( Y ));
	//system->mapDGtoSundials(tempdSigdt, tempdQdt, tempdudt, tempdLamdt, N_VGetArrayPointer( Y ));
	system->mapDGtoSundials(tempdudt, N_VGetArrayPointer( dydt )); 
	system->mapDGtoSundials(res1, res2, res3, res4, N_VGetArrayPointer( resval )); 
	VectorWrapper resVec( N_VGetArrayPointer( resval ), N_VGetLength( resval ) );
	resVec.setZero();

	//Solve for Lambda with Lam = (H^T)^-1*[ -C*Sig - G*U + L ] 
	Eigen::VectorXd CsGuL_global(nVar*(nCells+1));
	CsGuL_global.setZero();
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Eigen::VectorXd LVarCell(2);
		Eigen::VectorXd CsGuLVarCell(2);
		CsGuLVarCell.setZero();
		LVarCell.setZero();
		for(int var = 0; var < nVar; var++)
		{
			LVarCell = system->L_global.block<2,1>(var*(nCells+1) + i,0);
			CsGuLVarCell = LVarCell - system->C_cellwise[i].block(var*2,var*(k+1),2,k+1)*tempSig.coeffs[ var ][ i ].second - system->G_cellwise[i].block(var*2,var*(k+1),2,k+1)*tempU.coeffs[ var ][ i ].second;
			CsGuL_global.block(var*(nCells + 1) + i, 0, 2, 1) += CsGuLVarCell;
		}
	}
	lam = system->H_global.solve( CsGuL_global );
	for(int var = 0; var < nVar; var++)
	{
		if(system->BCs->isLBoundDirichlet) lam[var*(nCells+1)] = system->BCs->g_D(system->grid.lowerBound, static_cast<double>(tres), var);
		if(system->BCs->isLBoundDirichlet) tempLambda[var*(nCells+1)] = system->BCs->g_D(system->grid.lowerBound, static_cast<double>(tres), var);
		if(system->BCs->isUBoundDirichlet) lam[nCells+var*(nCells+1)] = system->BCs->g_D(system->grid.upperBound, static_cast<double>(tres), var);
		if(system->BCs->isUBoundDirichlet) tempLambda[nCells+var*(nCells+1)] = system->BCs->g_D(system->grid.upperBound, static_cast<double>(tres), var);
	}
	res4 = -tempLambda + lam;

	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Eigen::VectorXd lamCell(2*nVar);

		for(int var = 0; var < nVar; var++)
		{
			lamCell[2*var] = tempLambda[var*(nCells+1) + i]; lamCell[2*var + 1] = tempLambda[var*(nCells+1) + i+1];
		}

		//length = nVar*(k+1)
		for(int var = 0; var < nVar; var++)
		{
			std::function< double (double) > kappaFunc = [ = ]( double x ) { return system->getDiffObj()->getKappaFunc(var)( x, tempQ, tempU); };
			std::function< double (double) > sourceFunc = [ = ]( double x ) { return system->getSourceObj()->getSourceFunc(var)( x, tempSig, tempQ, tempU); };

			//Evaluate Diffusion Function
			Eigen::VectorXd kappa_cellwise(k+1);
			kappa_cellwise.setZero();
			for ( Eigen::Index j = 0; j < k+1; j++ )
				kappa_cellwise( j ) = tempU.CellProduct( I, kappaFunc, tempU.Basis.phi( I, j ) );

			//Evaluate Source Function
			Eigen::VectorXd S_cellwise(k+1);
			S_cellwise.setZero();
			for ( Eigen::Index j = 0; j < k+1; j++ )
				S_cellwise( j ) = tempU.CellProduct( I, sourceFunc, tempU.Basis.phi( I, j%(k+1) ) );

			//res1.coeffs[ var ][ i ].second = -system->A_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempQ.coeffs[var][i].second - system->B_cellwise[i].transpose().block(var*(k+1), var*(k+1), k+1, k+1)*tempU.coeffs[ var ][ i ].second + system->C_cellwise[i].transpose().block(var*(k+1), var*2, k+1, 2)*lamCell.block<2,1>(var*2,0) - system->RF_cellwise[ i ].block( var*(k+1), 0, k + 1, 1 );
			//res2.coeffs[ var ][ i ].second = system->B_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempSig.coeffs[ var ][ i ].second + system->D_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempU.coeffs[ var ][ i ].second + system->E_cellwise[i].block(var*(k+1), var*2, k+1, 2)*lamCell.block<2,1>(var*2,0) - system->RF_cellwise[ i ].block( nVar*(k + 1) + var*(k+1), 0, k + 1, 1 ) + S_cellwise + system->XMats[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempdudt.coeffs[ var ][ i ].second;
			//res3.coeffs[ var ][ i ].second = tempSig.coeffs[ var ][ i ].second + kappa_cellwise;
			res1.coeffs[ var ][ i ].second = system->resEval({(-1.0)*system->A_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempQ.coeffs[var][i].second, (-1.0)*system->B_cellwise[i].transpose().block(var*(k+1), var*(k+1), k+1, k+1)*tempU.coeffs[ var ][ i ].second, system->C_cellwise[i].transpose().block(var*(k+1), var*2, k+1, 2)*lamCell.block<2,1>(var*2,0), - system->RF_cellwise[ i ].block( var*(k+1), 0, k + 1, 1 )});
			res2.coeffs[ var ][ i ].second = system->resEval({system->B_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempSig.coeffs[ var ][ i ].second, system->D_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempU.coeffs[ var ][ i ].second, system->E_cellwise[i].block(var*(k+1), var*2, k+1, 2)*lamCell.block<2,1>(var*2,0), - system->RF_cellwise[ i ].block( nVar*(k + 1) + var*(k+1), 0, k + 1, 1 ), S_cellwise, system->XMats[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempdudt.coeffs[ var ][ i ].second});
			res3.coeffs[ var ][ i ].second = system->resEval({tempSig.coeffs[ var ][ i ].second, kappa_cellwise});
			if(var==2) std::cerr << (system->B_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempSig.coeffs[ var ][ i ].second)[0] << "	" << (system->D_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempU.coeffs[ var ][ i ].second)[0] << "	" << (system->E_cellwise[i].block(var*(k+1), var*2, k+1, 2)*lamCell.block<2,1>(var*2,0))[0] << "	"  <<system->RF_cellwise[ i ].block( nVar*(k + 1) + var*(k+1), 0, 1, 1 ) << "	" << S_cellwise[0] << "	" << (system->XMats[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempdudt.coeffs[ var ][ i ].second)[0] << std::endl << std::endl;
		}
	}

	VectorWrapper Vec( N_VGetArrayPointer( resval ), N_VGetLength( resval ) );
	VectorWrapper yVec( N_VGetArrayPointer( Y ), N_VGetLength( Y ) );
	VectorWrapper ypVec( N_VGetArrayPointer( dydt ), N_VGetLength( dydt ) );

	res2.printCoeffs(0);
	res2.printCoeffs(1);
	res2.printCoeffs(2);

	//std::cerr << res4 << std::endl << std::endl;

	Eigen::Index maxloc, minloc;
 	std::cerr << Vec.norm() << "	" << "	" << Vec.maxCoeff(&maxloc) << "	" << Vec.minCoeff(&minloc) << "	" << maxloc << "	" << minloc << "	" << tres << std::endl << std::endl;
	system->total_steps++;

	std::ofstream Yfile;
	Yfile.open("yfile.txt");

	if(system->total_steps%100 == 99)
	{
		std::ofstream resfile0;
		std::ofstream resfile1;
		std::ofstream resfile4;
		resfile0.open("res0.txt", std::ofstream::app);
		resfile1.open("res1.txt", std::ofstream::app);
		resfile4.open("res4.txt", std::ofstream::app);
		system->print(resfile0, tres, 200, 0, resval);
		if(nVar > 1) system->print(resfile1, tres, 200, 1, resval);
		resfile4 << res4 << std::endl << std::endl;
	}

	if(system->isTesting())
	{	
		VectorWrapper residualVec( N_VGetArrayPointer( resval ), N_VGetLength( resval ) );
		system->resNorm = residualVec.norm();
	}

	std::ofstream file;
	file.open("time.txt", std::ofstream::app);
	file << tres << std::endl << std::endl;

	return 0;
}

void SystemSolver::print( std::ostream& out, double t, int nOut, int var )
{
	out << "# t = " << t << std::endl;
	for ( int i=0; i<nOut; ++i )
	{
		double x = BCs->LowerBound + ( BCs->UpperBound - BCs->LowerBound ) * ( static_cast<double>( i )/( nOut-1 ) );
		out << x << "\t" << EvalCoeffs( u.Basis, u.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, q.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, sig.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, dudt.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, dqdt.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, dsigdt.coeffs, x, var ) << std::endl;
	}
	out << std::endl;
}

void SystemSolver::print( std::ostream& out, double t, int nOut, int var, N_Vector& tempY )
{
	DGApprox Sig(grid,k), Q(grid,k), U(grid,k);
	double memBlock[nVar*(nCells+1)];
	Eigen::Map<Eigen::VectorXd> Lambda(memBlock, nVar*(nCells+1));

	mapDGtoSundials(Sig, Q, U, Lambda, N_VGetArrayPointer( tempY ));

	out << "# t = " << t << std::endl;
	for ( int i=0; i<nOut; ++i )
	{
		double x = BCs->LowerBound + ( BCs->UpperBound - BCs->LowerBound ) * ( static_cast<double>( i )/( nOut-1 ) );
		out << x << "\t" << EvalCoeffs( U.Basis, U.coeffs, x, var ) << "\t" << EvalCoeffs( U.Basis, Q.coeffs, x, var ) << "\t" << EvalCoeffs( U.Basis, Sig.coeffs, x, var ) << std::endl;
	}
	out << std::endl;
}

void SystemSolver::print( std::ostream& out, double t, int nOut, int var, N_Vector& tempY, N_Vector& tempRes )
{
	DGApprox Sig(grid,k), Q(grid,k), U(grid,k);
	DGApprox resSig(grid,k), resQ(grid,k), resU(grid,k);
	double memBlock[nVar*(nCells+1)];
	Eigen::Map<Eigen::VectorXd> Lambda(memBlock, nVar*(nCells+1));
	Eigen::Map<Eigen::VectorXd> resLambda(memBlock, nVar*(nCells+1));

	mapDGtoSundials(Sig, Q, U, Lambda, N_VGetArrayPointer( tempY ));
	mapDGtoSundials(resSig, resQ, resU, Lambda, N_VGetArrayPointer( tempRes ));

	out << "# t = " << t << std::endl;
	for ( int i=0; i<nOut; ++i )
	{
		double x = BCs->LowerBound + ( BCs->UpperBound - BCs->LowerBound ) * ( static_cast<double>( i )/( nOut-1 ) );
		out << x << "\t" << EvalCoeffs( U.Basis, U.coeffs, x, var ) << "\t" << EvalCoeffs( U.Basis, Q.coeffs, x, var ) << "\t" << EvalCoeffs( U.Basis, Sig.coeffs, x, var )  << "\t" << EvalCoeffs( U.Basis, resU.coeffs, x, var ) << "\t" << EvalCoeffs( U.Basis, resQ.coeffs, x, var ) << "\t" << EvalCoeffs( U.Basis, resSig.coeffs, x, var ) << std::endl;
	}
	out << std::endl;
}

double SystemSolver::EvalCoeffs( LegendreBasis & B, Coeff_t cs, double x, int var )
{
	for ( auto const & pair : cs[var] )
	{
		if ( pair.first.contains( x ) )
			return B.Evaluate( pair.first, pair.second, x );
	}
	return std::nan( "" );
}

std::shared_ptr<DiffusionObj> SystemSolver::getDiffObj()
{
	if(diffObj) return diffObj;
	else return nullptr;
}

std::shared_ptr<SourceObj> SystemSolver::getSourceObj()
{
	if(sourceObj) return sourceObj;
	else return nullptr;
}

