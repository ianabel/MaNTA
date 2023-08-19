
#include <boost/test/unit_test.hpp>

#include "../../gridStructures.hpp"
#include "../../DGSoln.hpp"
#include <cmath>
#include <vector>

// Defines the entry point and a simple set of functional tests for the MirrorPlasma class

// This defines the name of the test suite and causes
// the default behaviour of the BOOST_TEST macros to compare
// within a part in 10^9 rather than exact comparison.

BOOST_AUTO_TEST_SUITE( dg_approx_tests, * boost::unit_test::tolerance( 1e-9 ) )

BOOST_AUTO_TEST_CASE( grid_test )
{
	Grid *pGrid = nullptr;
	BOOST_CHECK_NO_THROW( pGrid = new Grid( 0.0, 1.0, 5 ) );

	BOOST_TEST( pGrid->getNCells() == 5 );
	for ( auto i=0; i<5; ++i )
	{
		Interval const& I = ( *pGrid )[ i ];
		BOOST_TEST( I.x_l == 0.2*i );
		BOOST_TEST( I.x_u == 0.2*( i + 1 ) );
	}

	Grid *pGrid2 = nullptr;
	BOOST_CHECK_NO_THROW( pGrid2 = new Grid( 1.0, 0.0, 5 ) );

	bool equality = ( *pGrid == *pGrid2 );
	BOOST_TEST( equality );

	delete pGrid2;
	BOOST_CHECK_NO_THROW( pGrid2 = new Grid( 0.0, 1.0, 10 ) );

	bool inequality = ( *pGrid != *pGrid2 );
	BOOST_TEST( inequality );

}

BOOST_AUTO_TEST_CASE( legendre_basis_test )
{
	Interval I1( 0.0, 1.0 ),I2( 0.5, 0.55 ),I3( -0.2,-0.13 );
	std::vector<Interval> test_intervals{ I1, I2, I3 };
	std::vector<unsigned int> test_indices{ 0, 1, 2, 4, 8 };
	std::vector<double> test_pt{ 0.1, 0.2, 0.5, 0.9 };

	for ( auto const& I : test_intervals ) {
		for ( auto i : test_indices ) {
			auto phi_fn = LegendreBasis::phi( I, i );
			auto phiPrime_fn = LegendreBasis::phiPrime( I, i );

			double sgn = ( i%2 == 0 ? 1.0 : -1.0 );
			double uVal = ::sqrt( ( 2* i + 1 )/( I.h() ) );
			double lVal = sgn * uVal;

			BOOST_TEST( LegendreBasis::Evaluate( I, i, I.x_l ) == lVal );
			BOOST_TEST( phi_fn( I.x_l ) == lVal );
			BOOST_TEST( LegendreBasis::Prime( I, i, I.x_l ) == sgn * i*( i + 1.0 )/2.0 );
			BOOST_TEST( phiPrime_fn( I.x_l ) == sgn * i*( i + 1.0 )/2.0 );

			for ( auto x : test_pt ) {
				double x_pt = x * I.h() + I.x_l;
				double y = 2*x - 1.0;
				BOOST_TEST( LegendreBasis::Evaluate( I, i, x_pt ) == ::sqrt( ( 2* i + 1 )/( I.h() ) ) * std::legendre( i, y ) );
				BOOST_TEST( phi_fn( x_pt ) == ::sqrt( ( 2* i + 1 )/( I.h() ) ) * std::legendre( i, y ) );
				double primeVal = i == 0 ? 0 : ::sqrt( ( 2* i + 1 )/( I.h() ) ) * ( 2*i/I.h() ) *( 1.0/( y*y-1.0 ) )*( y*std::legendre( i, y ) - std::legendre( i-1,y ) );
				BOOST_TEST( LegendreBasis::Prime( I, i, x_pt ) == primeVal );
				BOOST_TEST( phiPrime_fn( x_pt ) == primeVal );
			}

			BOOST_TEST( LegendreBasis::Evaluate( I, i, I.x_u ) == uVal );
			BOOST_TEST( phi_fn( I.x_u ) == uVal );
			BOOST_TEST( LegendreBasis::Prime( I, i, I.x_u ) == i*( i + 1.0 )/2.0 );
			BOOST_TEST( phiPrime_fn( I.x_u ) == i*( i + 1.0 )/2.0 );
		}
	}

}

BOOST_AUTO_TEST_CASE( dg_approx_construction )
{
	Grid testGrid( 0.0, 1.0, 4 );
	double a = 2.0;
	DGApprox constructedLinear( testGrid, 1, [=]( double x ){ return a*x; } );
	BOOST_TEST( constructedLinear.getCoeffs().size() == testGrid.getNCells() );
	BOOST_TEST( constructedLinear.getDoF() == testGrid.getNCells() * 2 );
	BOOST_TEST( constructedLinear( 0.1 ) == a*0.1 );
	BOOST_TEST( constructedLinear( 0.7 ) == a*0.7 );
	BOOST_TEST( constructedLinear( 0.1, testGrid[ 0 ] ) == a * 0.1 );
	BOOST_TEST( constructedLinear( 0.7, testGrid[ 2 ] ) == a * 0.7 );

	DGApprox linear( testGrid, 1 );

	auto const& cref = linear.getCoeffs();
	BOOST_TEST( cref.size() == 0 );
	BOOST_TEST( cref.capacity() == testGrid.getNCells() );
	BOOST_TEST( linear.getDoF() == 8 );

	// Map memory
	double* mem = new double[ linear.getDoF() ];
	VectorWrapper v( mem, linear.getDoF() );
	linear.Map( mem, 2 );

	// Pick something that is exact
	linear = [=]( double x ){ return a*x; };

	// v( 2*i ) == Average over Cell i / Sqrt( Length of Cell i )
	//          == (a/2)*[ x_u^2 - x_l^2 ] / Sqrt( I.h() )

	BOOST_TEST( v( 0 ) == ( a/2.0 ) * ( 0.25*0.25 ) / ::sqrt( 0.25 ) );
	BOOST_TEST( v( 2 ) == ( a/2.0 ) * ( 0.5*0.5 - 0.25*0.25 ) / ::sqrt( 0.25 ) );
	BOOST_TEST( v( 4 ) == ( a/2.0 ) * ( 0.75*0.75 - 0.5*0.5 ) / ::sqrt( 0.25 ) );
	BOOST_TEST( v( 6 ) == ( a/2.0 ) * ( 1*1 - 0.75*0.75 ) / ::sqrt( 0.25 ) );

	BOOST_TEST( linear( 0.1 ) == a*0.1 );
	BOOST_TEST( linear( 0.7 ) == a*0.7 );

	DGApprox constructedData( testGrid, 1, mem, 2 );

	BOOST_TEST( constructedData( 0.1 ) == a*0.1 );
	BOOST_TEST( constructedData( 0.7 ) == a*0.7 );
	BOOST_TEST( constructedData.getDoF() == 8 );

	// += operator testing.
	constructedData = [ = ]( double x ){ return ( a/2.0 )*x; };

	// check we overwrote teh data we thought we were using
	BOOST_TEST( v( 0 ) == ( a/4.0 ) * ( 0.25*0.25 ) / ::sqrt( 0.25 ) );
	BOOST_TEST( v( 2 ) == ( a/4.0 ) * ( 0.5*0.5 - 0.25*0.25 ) / ::sqrt( 0.25 ) );
	BOOST_TEST( v( 4 ) == ( a/4.0 ) * ( 0.75*0.75 - 0.5*0.5 ) / ::sqrt( 0.25 ) );
	BOOST_TEST( v( 6 ) == ( a/4.0 ) * ( 1*1 - 0.75*0.75 ) / ::sqrt( 0.25 ) );

	BOOST_CHECK_NO_THROW( constructedLinear += constructedData );

	BOOST_TEST( constructedLinear.getDoF() == testGrid.getNCells() * 2 );
	BOOST_TEST( constructedLinear( 0.1 ) == 1.5*a*0.1 );
	BOOST_TEST( constructedLinear( 0.7 ) == 1.5*a*0.7 );

	delete mem;
	// Interleaved data test
	mem = new double[ linear.getDoF() * 2 ];
	linear.Map( mem, 4 );
	constructedData.Map( mem + 2, 4 );
	linear = [=]( double x ){ return a*x; };
	constructedData = [ = ]( double x ){ return ( a/2.0 )*x; };

	BOOST_TEST( linear.getDoF() == 8 );
	BOOST_TEST( linear( 0.1 ) == a*0.1 );
	BOOST_TEST( linear( 0.7 ) == a*0.7 );

	BOOST_TEST( constructedData( 0.1 ) == ( a/2.0 )*0.1 );
	BOOST_TEST( constructedData( 0.7 ) == ( a/2.0 )*0.7 );
	BOOST_TEST( constructedData.getDoF() == 8 );

	new ( &v ) VectorWrapper( mem, linear.getDoF()*2 );
	// Data for 'linear'
	BOOST_TEST( v( 0 ) == ( a/2.0 ) * ( 0.25*0.25 ) / ::sqrt( 0.25 ) );
	BOOST_TEST( v( 4 ) == ( a/2.0 ) * ( 0.5*0.5 - 0.25*0.25 ) / ::sqrt( 0.25 ) );
	BOOST_TEST( v( 8 ) == ( a/2.0 ) * ( 0.75*0.75 - 0.5*0.5 ) / ::sqrt( 0.25 ) );
	BOOST_TEST( v( 12 ) == ( a/2.0 ) * ( 1*1 - 0.75*0.75 ) / ::sqrt( 0.25 ) );

	// Data for 'constructedData'
	BOOST_TEST( v( 2 ) == ( a/4.0 ) * ( 0.25*0.25 ) / ::sqrt( 0.25 ) );
	BOOST_TEST( v( 6 ) == ( a/4.0 ) * ( 0.5*0.5 - 0.25*0.25 ) / ::sqrt( 0.25 ) );
	BOOST_TEST( v( 10 ) == ( a/4.0 ) * ( 0.75*0.75 - 0.5*0.5 ) / ::sqrt( 0.25 ) );
	BOOST_TEST( v( 14 ) == ( a/4.0 ) * ( 1*1 - 0.75*0.75 ) / ::sqrt( 0.25 ) );

	delete mem;

}

BOOST_AUTO_TEST_CASE( dg_approx_static )
{
	Grid testGrid( 0.0, 1.0, 4 );
	auto foo = []( double x ){ return x; };
	auto bar = []( double x ){ return ::sin( x ); };

	BOOST_TEST( DGApprox::CellProduct( testGrid[ 0 ], foo, bar ) == ::sin( 0.25 ) - 0.25*::cos( 0.25 ) );
	BOOST_TEST( DGApprox::EdgeProduct( testGrid[ 0 ], foo, bar ) == 0.25*::sin( 0.25 ) );

	Eigen::MatrixXd tmp( 5, 5 );
	tmp.setZero();
	DGApprox::MassMatrix( testGrid[ 0 ], tmp );
	BOOST_TEST( static_cast<bool>( tmp == Eigen::MatrixXd::Identity( 5, 5 ) ) );

	DGApprox::MassMatrix( testGrid[ 0 ], tmp, []( double ){ return 1.0; } );
	BOOST_TEST( ( tmp - Eigen::MatrixXd::Identity( 5, 5 ) ).norm() < 1e-9 );


	DGApprox::MassMatrix( testGrid[ 0 ], tmp, []( double x ){ return x; } );
	Eigen::MatrixXd ref( 5,5 ); 
	// Courtesy of Mathematica
	ref << 0.125,      0.07216878, 0.0,        0.0,        0.0,
	       0.07216878, 0.125,      0.0645497,  0.0,        0.0,
	       0.0,        0.0645497,  0.125,      0.0633866,  0.0,
	       0.0,        0.0,        0.0633866,  0.125,      0.0629941,
	       0.0,        0.0,        0.0,        0.0629941,  0.125;

	BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );
	
	DGApprox::MassMatrix( testGrid[ 0 ], tmp, []( double x, int ){ return x; }, 0 );
	BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );

	DGApprox test( testGrid, 4 );
	BOOST_TEST( static_cast<bool>( test.MassMatrix( testGrid[ 0 ] ) == Eigen::MatrixXd::Identity( 5, 5 ) ) );
	
	tmp = test.MassMatrix( testGrid[ 0 ], []( double x ) { return x; } );
	BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );

	tmp.resize( 5, 5 );
	DGApprox::DerivativeMatrix( testGrid[ 0 ], tmp );
	ref << 0.0, 13.8564, 0.0,     21.166,  0.0,
	       0.0, 0.0,     30.9839, 0.0,     41.5692,
	       0.0, 0.0,     0.0,     47.3286, 0.0,
	       0.0, 0.0,     0.0,     0.0,     63.498,
	       0.0, 0.0,     0.0,     0.0,     0.0;
	BOOST_TEST( ( tmp - ref ).norm() < 1e-4 ); // Entries are only good to 0.0001 anyway

	DGApprox::DerivativeMatrix( testGrid[ 0 ], tmp, [](  double ){ return 1.0; } );
	BOOST_TEST( ( tmp - ref ).norm() < 1e-4 );

	DGApprox::DerivativeMatrix( testGrid[ 0 ], tmp, []( double x ){ return x; } );
	ref << 0.0, 1.73205, 2.23607, 2.64575, 3.0,
	       0.0, 1.0,     3.87298, 4.58258, 5.19615,
	       0.0, 0.0,     2.0,     5.91608, 6.7082,
	       0.0, 0.0,     0.0,     3.0,     7.93725,
	       0.0, 0.0,     0.0,     0.0,     4.0;
	BOOST_TEST( ( tmp - ref ).norm() < 1e-5 );

}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( dg_soln_tests, * boost::unit_test::tolerance( 1e-6 ) )

BOOST_AUTO_TEST_CASE( dg_soln_construction )
{
	Grid testGrid( 0.0, 1.0, 5 );
	// Use higher-order elements to allow for accurate representation of q / sigma
	Index k = 5;

	DGSoln single_var( 1, testGrid, k );

	BOOST_TEST( single_var.getDoF() == testGrid.getNCells() * ( 3*( k+1 ) + 1 ) + 1 );

	double *mem = new double[ single_var.getDoF() ];

	single_var.Map( mem );

	single_var.AssignU( []( Index, double x ){ return x; } );

	BOOST_TEST( single_var.u( 0 )( 0.1 ) == 0.1 );
	BOOST_TEST( single_var.u( 0 )( 0.7 ) == 0.7 );

	single_var.AssignQ( []( Index, double x ){ return ::cos( x ); } );

	BOOST_TEST( single_var.q( 0 )( 0.1 ) == ::cos( 0.1 ) );
	BOOST_TEST( single_var.q( 0 )( 0.7 ) == ::cos( 0.7 ) );

	single_var.EvaluateLambda();

	BOOST_TEST( single_var.lambda( 0 )( 0 ) == 0.0 );
	BOOST_TEST( single_var.lambda( 0 )( 1 ) == 0.2 );
	BOOST_TEST( single_var.lambda( 0 )( 2 ) == 0.4 );
	BOOST_TEST( single_var.lambda( 0 )( 3 ) == 0.6 );
	BOOST_TEST( single_var.lambda( 0 )( 4 ) == 0.8 );
	BOOST_TEST( single_var.lambda( 0 )( 5 ) == 1.0 );

	single_var.AssignSigma( []( Index, const Values& uV, const Values& qV, Position x, Time ) {
		return uV[ 0 ] * ( 1.0 - qV[ 0 ]*qV[ 0 ] );
	} );

	BOOST_TEST( single_var.sigma( 0 )( 0.1 ) == 0.1 * ( ::sin( 0.1 )*::sin( 0.1 ) ) );
	BOOST_TEST( single_var.sigma( 0 )( 0.5 ) == 0.5 * ( ::sin( 0.5 )*::sin( 0.5 ) ) );
	BOOST_TEST( single_var.sigma( 0 )( 0.7 ) == 0.7 * ( ::sin( 0.7 )*::sin( 0.7 ) ) );

	double *mem2 = new double[ single_var.getDoF() ];
	DGSoln other_var( 1, testGrid, k, mem2 );

	other_var.AssignU( []( Index, double x ){ return x; } );
	other_var.AssignQ( []( Index, double ){ return 1.0; } );
	other_var.EvaluateLambda();

	other_var.AssignSigma( []( Index, const Values& uV, const Values& qV, Position x, Time ) {
		return 1.0 - uV[ 0 ] * uV[ 0 ];
	} );

	single_var += other_var;

	BOOST_TEST( single_var.u( 0 )( 0.1 ) == 0.2 );
	BOOST_TEST( single_var.u( 0 )( 0.7 ) == 1.4 );
	BOOST_TEST( single_var.q( 0 )( 0.1 ) == 1.0 + ::cos( 0.1 ) );
	BOOST_TEST( single_var.q( 0 )( 0.7 ) == 1.0 + ::cos( 0.7 ) );

}


BOOST_AUTO_TEST_SUITE_END()


