
#include <boost/test/unit_test.hpp>
#include <boost/math/special_functions/legendre.hpp>

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

    std::vector<double> test_ref{ -1.0, -0.8, -0.5, -0.25, 0, 0.25, 0.5, 0.8, 1.0 };

    for ( auto i : test_indices ) {
        for( auto x : test_ref ) {

            BOOST_TEST( LegendreBasis::Evaluate( i, x ) == sqrt( 0.5 + i )*boost::math::legendre_p( i, x ) );
            BOOST_TEST( LegendreBasis::Prime( i, x ) == sqrt( 0.5 + i )*boost::math::legendre_p_prime( i, x ) );

        }
    }

    for ( auto const& I : test_intervals ) {
        for ( auto i : test_indices ) {

            double sgn = ( i%2 == 0 ? 1.0 : -1.0 );
            double uVal = ::sqrt( ( 2* i + 1 )/( I.h() ) );
            double lVal = sgn * uVal;

            BOOST_TEST( LegendreBasis::Evaluate( I, i, I.x_l ) == lVal );
            BOOST_TEST( LegendreBasis::Prime( I, i, I.x_l ) == boost::math::legendre_p_prime( i, -1 ) * ::sqrt( (2*i+1)/I.h() ) * (2.0/I.h()) );

            for ( auto x : test_pt ) {
                double x_pt = x * I.h() + I.x_l;
                double y = I.toRef( x_pt );
                BOOST_TEST( LegendreBasis::Evaluate( I, i, x_pt ) == ::sqrt( ( 2* i + 1 )/( I.h() ) ) * std::legendre( i, y ) );
                double primeVal = boost::math::legendre_p_prime( i, y ) * ( ::sqrt( (2*i + 1)/I.h() ) ) * (2.0/I.h());
                BOOST_TEST( LegendreBasis::Prime( I, i, x_pt ) == primeVal );
            }

            BOOST_TEST( LegendreBasis::Evaluate( I, i, I.x_u ) == uVal );
            BOOST_TEST( LegendreBasis::Prime( I, i, I.x_u ) == boost::math::legendre_p_prime( i, 1 ) * ::sqrt( (2*i+1)/I.h() ) * (2.0/I.h()) );
        }
    }

}

BOOST_AUTO_TEST_CASE( cheb_basis_test )
{
    Interval I1( 0.0, 1.0 ),I2( 0.5, 0.55 ),I3( -0.2,-0.13 );
    std::vector<Interval> test_intervals{ I1, I2, I3 };
    std::vector<unsigned int> test_indices{ 0, 1, 2, 4, 5, 7, 8 };
    std::vector<double> test_pt{ 0.1, 0.2, 0.5, 0.9 };

    for ( auto const& I : test_intervals ) {
        for ( auto i : test_indices ) {

            double sgn = ( i%2 == 0 ? 1.0 : -1.0 );

            BOOST_TEST( ChebyshevBasis::Evaluate( I, i, I.x_l ) == sgn );
            BOOST_TEST( ChebyshevBasis::Prime( I, i, I.x_l ) == (-sgn) * i*i * (2.0/I.h()) );
            BOOST_TEST( ChebyshevBasis::Evaluate( I, i, (I.x_l + I.x_u)/2.0 ) == std::cos( pi * i / 2.0 ) );
            BOOST_TEST( ChebyshevBasis::Evaluate( I, i, I.x_u ) == 1.0 );
            BOOST_TEST( ChebyshevBasis::Prime( I, i, I.x_u ) == i*i* (2.0/I.h()) );
        }
    }

    for ( auto i : test_indices ) {
      for ( auto x : test_pt ) {
        BOOST_TEST( ChebyshevBasis::Tn( i, I1.toRef(x) ) == std::cos( i * std::acos( I1.toRef(x) ) ) );
        BOOST_TEST( ChebyshevBasis::Un( i, I1.toRef(x) ) == std::sin( (i+1) * std::acos( I1.toRef(x) ) )/std::sin( std::acos( I1.toRef(x) ) ) );
      }
    }


}

BOOST_AUTO_TEST_CASE( nodal_basis_test )
{
    NodalBasis basis = NodalBasis::getBasis( 6 );
    std::vector<double> nodes{ -1, -0.83022389627856692986, -0.46884879347071421380, 0, 0.46884879347071421380, 0.83022389627856692986, 1.0 };

    Eigen::MatrixXd PrimeVals( 7, 7 );
    PrimeVals << -10.5,14.2016,-5.66899,3.2,-2.04996,1.31737,-0.5,
                 -2.44293,0.0,3.45583,-1.59861,0.96134,-0.602247,0.226612,
                  0.625257,-2.2158,0.,2.2667,-1.06644,0.616391,-0.226099,
                 -0.3125,0.907544,-2.00697,0.0,2.00697,-0.907544,0.3125,
                  0.226099,-0.616391,1.06644,-2.2667,0.0,2.2158,-0.625257,
                 -0.226612,0.602247,-0.96134,1.59861,-3.45583,0,2.44293,
                  0.5,-1.31737,2.04996,-3.2,5.66899,-14.2016,10.5;

    for( int i=0; i < 7; i++ ) {
        for( int j=0; j < 7; j++ ) {
            double delta = ( i == j ) ? 1.0 : 0.0;
            BOOST_TEST( basis.Evaluate( i, nodes[ j ] ) == delta );
            BOOST_TEST( basis.Prime( i, nodes[ j ] ) == PrimeVals( j, i ), boost::test_tools::tolerance( 1e-5 ) );
        }
    }


}

BOOST_AUTO_TEST_CASE( dg_approx_construction )
{
    Grid testGrid( 0.0, 1.0, 4 );
    double a = 2.0;
    LegendreBasis Basis = LegendreBasis::getBasis( 1 );
    DGApproxImpl<LegendreBasis> linear( testGrid, Basis );

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

    double *extraMem = new double[ linear.getDoF() ];
    DGApproxImpl<LegendreBasis> constructedLinear( testGrid, Basis, extraMem, 2 );

    // Check copy
    BOOST_CHECK_NO_THROW( constructedLinear.copy( linear ) );
    BOOST_TEST( constructedLinear.getCoeffs().size() == testGrid.getNCells() );
    BOOST_TEST( constructedLinear.getDoF() == testGrid.getNCells() * 2 );
    BOOST_TEST( constructedLinear( 0.1 ) == a*0.1 );
    BOOST_TEST( constructedLinear( 0.7 ) == a*0.7 );
    BOOST_TEST( constructedLinear( 0.1, testGrid[ 0 ] ) == a * 0.1 );
    BOOST_TEST( constructedLinear( 0.7, testGrid[ 2 ] ) == a * 0.7 );


    // Another window on mem
    DGApproxImpl<LegendreBasis> constructedData( testGrid, Basis, mem, 2 );

    BOOST_TEST( constructedData( 0.1 ) == a*0.1 );
    BOOST_TEST( constructedData( 0.7 ) == a*0.7 );
    BOOST_TEST( constructedData.getDoF() == 8 );

    // += operator testing.
    constructedData = [ = ]( double x ){ return ( a/2.0 )*x; };

    // check we overwrote the data we thought we were using
    BOOST_TEST( v( 0 ) == ( a/4.0 ) * ( 0.25*0.25 ) / ::sqrt( 0.25 ) );
    BOOST_TEST( v( 2 ) == ( a/4.0 ) * ( 0.5*0.5 - 0.25*0.25 ) / ::sqrt( 0.25 ) );
    BOOST_TEST( v( 4 ) == ( a/4.0 ) * ( 0.75*0.75 - 0.5*0.5 ) / ::sqrt( 0.25 ) );
    BOOST_TEST( v( 6 ) == ( a/4.0 ) * ( 1*1 - 0.75*0.75 ) / ::sqrt( 0.25 ) );

    BOOST_CHECK_NO_THROW( constructedLinear += constructedData );

    BOOST_TEST( constructedLinear.getDoF() == testGrid.getNCells() * 2 );
    BOOST_TEST( constructedLinear( 0.1 ) == 1.5*a*0.1 );
    BOOST_TEST( constructedLinear( 0.7 ) == 1.5*a*0.7 );

    delete[] mem;
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

    delete[] mem;

}

BOOST_AUTO_TEST_CASE( dg_approx_construction_nodal )
{
    Grid testGrid( 0.0, 1.0, 4 );
    double a = 2.0;
    NodalBasis Basis = NodalBasis::getBasis( 1 );
    DGApproxImpl<NodalBasis> linear( testGrid, Basis );

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

    // Linear LGL Nodes on [-1,1] are { -1, 1 }
    constexpr double node = 1.0;

    // v( 2*i ) == Value at x_l

    for( Index i = 0; i < 4; ++i )
    {
      BOOST_TEST( v( 2*i ) == a * ( testGrid[i].x_l ) );
      BOOST_TEST( v( 2*i + 1 ) == a * ( testGrid[i].x_u ) );
    }

    BOOST_TEST( linear( 0.1 ) == a*0.1 );
    BOOST_TEST( linear( 0.7 ) == a*0.7 );

    double *extraMem = new double[ linear.getDoF() ];
    DGApproxImpl<NodalBasis> constructedLinear( testGrid, Basis, extraMem, 2 );

    // Check copy
    BOOST_CHECK_NO_THROW( constructedLinear.copy( linear ) );
    BOOST_TEST( constructedLinear.getCoeffs().size() == testGrid.getNCells() );
    BOOST_TEST( constructedLinear.getDoF() == testGrid.getNCells() * 2 );
    BOOST_TEST( constructedLinear( 0.1 ) == a*0.1 );
    BOOST_TEST( constructedLinear( 0.7 ) == a*0.7 );
    BOOST_TEST( constructedLinear( 0.1, testGrid[ 0 ] ) == a * 0.1 );
    BOOST_TEST( constructedLinear( 0.7, testGrid[ 2 ] ) == a * 0.7 );


    // Another window on mem
    DGApproxImpl<NodalBasis> constructedData( testGrid, Basis, mem, 2 );

    BOOST_TEST( constructedData( 0.1 ) == a*0.1 );
    BOOST_TEST( constructedData( 0.7 ) == a*0.7 );
    BOOST_TEST( constructedData.getDoF() == 8 );

    // += operator testing.
    constructedData = [ = ]( double x ){ return ( a/2.0 )*x; };

    // check we overwrote the data we thought we were using
    for( Index i = 0; i < 4; ++i )
    {
      BOOST_TEST( v( 2*i ) == a * testGrid[i].x_l / 2.0 );
      BOOST_TEST( v( 2*i + 1 ) == a * testGrid[i].x_u / 2.0 );
    }

    BOOST_CHECK_NO_THROW( constructedLinear += constructedData );

    BOOST_TEST( constructedLinear.getDoF() == testGrid.getNCells() * 2 );
    BOOST_TEST( constructedLinear( 0.1 ) == 1.5*a*0.1 );
    BOOST_TEST( constructedLinear( 0.7 ) == 1.5*a*0.7 );

    delete[] extraMem;
    delete[] mem;

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
    for( Index i = 0; i < 4; ++i )
    {
      BOOST_TEST( v( 4*i ) == a * ( testGrid[i].x_u * ( 1.0 - node ) / 2.0 + testGrid[i].x_l * ( 1.0 + node ) / 2.0 ) );
      BOOST_TEST( v( 4*i + 1 ) == a * ( testGrid[i].x_u * ( 1.0 + node ) / 2.0 + testGrid[i].x_l * ( 1.0 - node ) / 2.0 ) );
      BOOST_TEST( v( 4*i + 2 ) == (a/2.0) * ( testGrid[i].x_u * ( 1.0 - node ) / 2.0 + testGrid[i].x_l * ( 1.0 + node ) / 2.0 ) );
      BOOST_TEST( v( 4*i + 3 ) == (a/2.0) * ( testGrid[i].x_u * ( 1.0 + node ) / 2.0 + testGrid[i].x_l * ( 1.0 - node ) / 2.0 ) );
    }

    delete[] mem;
}

BOOST_AUTO_TEST_CASE( dg_approx_construction_cheb )
{
    Grid testGrid( 0.0, 1.0, 4 );
    double a = 2.0;
    ChebyshevBasis Basis = ChebyshevBasis::getBasis( 1 );
    DGApproxImpl<ChebyshevBasis> linear( testGrid, Basis );

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

    // v( 2*i ) == Value at cell centre
    //          == a * ( x_u + x_l ) / 2.0
    // v( 2*i + 1 ) == Slope in ref units = a * cell size / 2

    BOOST_TEST( v( 0 ) == ( a * 0.125 ) );
    BOOST_TEST( v( 2 ) == ( a * 0.375 ) );
    BOOST_TEST( v( 4 ) == ( a * 0.625 ) );
    BOOST_TEST( v( 6 ) == ( a * 0.875 ) );
    BOOST_TEST( v( 1 ) == ( a * 0.125 ) );
    BOOST_TEST( v( 3 ) == ( a * 0.125 ) );
    BOOST_TEST( v( 5 ) == ( a * 0.125 ) );
    BOOST_TEST( v( 7 ) == ( a * 0.125 ) );

    BOOST_TEST( linear( 0.1 ) == a*0.1 );
    BOOST_TEST( linear( 0.7 ) == a*0.7 );

    double *extraMem = new double[ linear.getDoF() ];
    DGApproxImpl<ChebyshevBasis> constructedLinear( testGrid, Basis, extraMem, 2 );

    // Check copy
    BOOST_CHECK_NO_THROW( constructedLinear.copy( linear ) );
    BOOST_TEST( constructedLinear.getCoeffs().size() == testGrid.getNCells() );
    BOOST_TEST( constructedLinear.getDoF() == testGrid.getNCells() * 2 );
    BOOST_TEST( constructedLinear( 0.1 ) == a*0.1 );
    BOOST_TEST( constructedLinear( 0.7 ) == a*0.7 );
    BOOST_TEST( constructedLinear( 0.1, testGrid[ 0 ] ) == a * 0.1 );
    BOOST_TEST( constructedLinear( 0.7, testGrid[ 2 ] ) == a * 0.7 );


    // Another window on mem
    DGApproxImpl<ChebyshevBasis> constructedData( testGrid, Basis, mem, 2 );

    BOOST_TEST( constructedData( 0.1 ) == a*0.1 );
    BOOST_TEST( constructedData( 0.7 ) == a*0.7 );
    BOOST_TEST( constructedData.getDoF() == 8 );

    // += operator testing.
    constructedData = [ = ]( double x ){ return ( a/2.0 )*x; };

    // check we overwrote the data we thought we were using
    BOOST_TEST( v( 0 ) == ( a * 0.125 / 2.0 ) );
    BOOST_TEST( v( 2 ) == ( a * 0.375 / 2.0 ) );
    BOOST_TEST( v( 4 ) == ( a * 0.625 / 2.0 ) );
    BOOST_TEST( v( 6 ) == ( a * 0.875 / 2.0 ) );
    BOOST_TEST( v( 1 ) == ( a * 0.125 / 2.0 ) );
    BOOST_TEST( v( 3 ) == ( a * 0.125 / 2.0 ) );
    BOOST_TEST( v( 5 ) == ( a * 0.125 / 2.0 ) );
    BOOST_TEST( v( 7 ) == ( a * 0.125 / 2.0 ) );

    BOOST_CHECK_NO_THROW( constructedLinear += constructedData );

    BOOST_TEST( constructedLinear.getDoF() == testGrid.getNCells() * 2 );
    BOOST_TEST( constructedLinear( 0.1 ) == 1.5*a*0.1 );
    BOOST_TEST( constructedLinear( 0.7 ) == 1.5*a*0.7 );

    delete[] extraMem;
    delete[] mem;

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
    BOOST_TEST( v( 0 ) == ( a * 0.125 ) );
    BOOST_TEST( v( 4 ) == ( a * 0.375 ) );
    BOOST_TEST( v( 8 ) == ( a * 0.625 ) );
    BOOST_TEST( v( 12 ) == ( a * 0.875 ) );
    BOOST_TEST( v( 1 ) == ( a * 0.125 ) );
    BOOST_TEST( v( 5 ) == ( a * 0.125 ) );
    BOOST_TEST( v( 9 ) == ( a * 0.125 ) );
    BOOST_TEST( v( 13 ) == ( a * 0.125 ) );

    // Data for 'constructedData'
    BOOST_TEST( v( 2 ) == ( a * 0.125 / 2.0 ) );
    BOOST_TEST( v( 6 ) == ( a * 0.375 / 2.0 ) );
    BOOST_TEST( v( 10 ) == ( a * 0.625 / 2.0 ) );
    BOOST_TEST( v( 14 ) == ( a * 0.875 / 2.0 ) );
    BOOST_TEST( v( 3 ) == ( a * 0.125 / 2.0 ) );
    BOOST_TEST( v( 7 ) == ( a * 0.125 / 2.0 ) );
    BOOST_TEST( v( 11 ) == ( a * 0.125 / 2.0 ) );
    BOOST_TEST( v( 15 ) == ( a * 0.125 / 2.0 ) );

    delete[] mem;
}

BOOST_AUTO_TEST_CASE( dg_approx_static )
{
    Grid testGrid( 0.0, 1.0, 4 );
    auto foo = []( double x ){ return x; };
    auto bar = []( double x ){ return ::sin( x ); };


    Eigen::MatrixXd tmp( 5, 5 );
    tmp.setZero();
    LegendreBasis basis = LegendreBasis::getBasis( 4 );

    BOOST_TEST( basis.CellProduct( testGrid[ 0 ], foo, bar ) == ::sin( 0.25 ) - 0.25*::cos( 0.25 ) );
    BOOST_TEST( basis.EdgeProduct( testGrid[ 0 ], foo, bar ) == 0.25*::sin( 0.25 ) );

    basis.MassMatrix( testGrid[ 0 ], tmp );
    BOOST_TEST( ( tmp - Eigen::MatrixXd::Identity( 5, 5 ) ).norm() < 1e-7 );

    basis.MassMatrix( testGrid[ 0 ], tmp, []( double ){ return 1.0; } );
    BOOST_TEST( ( tmp - Eigen::MatrixXd::Identity( 5, 5 ) ).norm() < 1e-9 );


    basis.MassMatrix( testGrid[ 0 ], tmp, []( double x ){ return x; } );
    Eigen::MatrixXd ref( 5,5 ); 
    // Courtesy of Mathematica
    ref << 0.125,      0.07216878, 0.0,        0.0,        0.0,
        0.07216878, 0.125,      0.0645497,  0.0,        0.0,
        0.0,        0.0645497,  0.125,      0.0633866,  0.0,
        0.0,        0.0,        0.0633866,  0.125,      0.0629941,
        0.0,        0.0,        0.0,        0.0629941,  0.125;

    BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );

    basis.MassMatrix( testGrid[ 0 ], tmp, []( double x, int ){ return x; }, 0 );
    BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );

    BOOST_TEST( static_cast<bool>( basis.MassMatrix( testGrid[ 0 ] ) == Eigen::MatrixXd::Identity( 5, 5 ) ) );

    tmp = basis.MassMatrix( testGrid[ 0 ], []( double x ) { return x; } );
    BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );

    tmp.resize( 5, 5 );
    basis.DerivativeMatrix( testGrid[ 0 ], tmp );
    ref << 0.0, 13.8564, 0.0,     21.166,  0.0,
        0.0, 0.0,     30.9839, 0.0,     41.5692,
        0.0, 0.0,     0.0,     47.3286, 0.0,
        0.0, 0.0,     0.0,     0.0,     63.498,
        0.0, 0.0,     0.0,     0.0,     0.0;
    BOOST_TEST( ( tmp - ref ).norm() < 1e-4 ); // Entries are only good to 0.0001 anyway

    basis.DerivativeMatrix( testGrid[ 0 ], tmp, [](  double ){ return 1.0; } );
    BOOST_TEST( ( tmp - ref ).norm() < 1e-4 );

    basis.DerivativeMatrix( testGrid[ 0 ], tmp, []( double x ){ return x; } );
    ref << 0.0, 1.73205, 2.23607, 2.64575, 3.0,
           0.0, 1.0,     3.87298, 4.58258, 5.19615,
           0.0, 0.0,     2.0,     5.91608, 6.7082,
           0.0, 0.0,     0.0,     3.0,     7.93725,
           0.0, 0.0,     0.0,     0.0,     4.0;
    BOOST_TEST( ( tmp - ref ).norm() < 1e-5 );

}

BOOST_AUTO_TEST_CASE( dg_approx_static_cheb )
{
    Grid testGrid( 0.0, 1.0, 4 );
    auto foo = []( double x ){ return x; };
    auto bar = []( double x ){ return ::sin( x ); };


    Eigen::MatrixXd tmp( 5, 5 );
    tmp.setZero();
    ChebyshevBasis basis = ChebyshevBasis::getBasis( 4 );

    BOOST_TEST( basis.CellProduct( testGrid[ 0 ], foo, bar ) == ::sin( 0.25 ) - 0.25*::cos( 0.25 ) );
    BOOST_TEST( basis.EdgeProduct( testGrid[ 0 ], foo, bar ) == 0.25*::sin( 0.25 ) );

    Eigen::MatrixXd ref( 5,5 );

    // Chebyshev Mass Matrix should be Int_[a,b] T_n T_m
    basis.MassMatrix( testGrid[ 0 ], tmp );
    ref << 2.0,        0.0,        -2.0/3.0,        0.0,        -2.0/15.0,
           0.0,        2.0/3.0,         0.0,   -2.0/5.0,        0.0,
           -2.0/3.0,   0.0,       14.0/15.0,        0.0,        -38.0/105.0,
           0.0,        -2.0/5.0,        0.0,  34.0/35.0,        0.0,
           -2.0/15.0,  0.0,     -38.0/105.0,        0.0,        62.0/63.0;

    ref *= 1.0/8.0;


    BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );

    basis.MassMatrix( testGrid[ 0 ], tmp, []( double ){ return 1.0; } );
    BOOST_TEST( ( tmp - ref ).norm() < 1e-9 );

    basis.MassMatrix( testGrid[ 0 ], tmp, []( double x ){ return x; } );
    // Courtesy of Mathematica
    // {,,,,{-(1/480),-(13/3360),-(19/3360),7/1440,31/2016}}
    ref << 1.0/32.0,1.0/96.0,-(1.0/96.0),-(1.0/160.0),-(1.0/480.0),
           1.0/96.0,1.0/96.0,1.0/480.0,-(1.0/160.0),-(13.0/3360.0),
           -(1.0/96.0),1.0/480.0,7.0/480.0,1.0/224.0,-(19.0/3360.0),
           -(1.0/160.0),-(1.0/160.0),1.0/224.0,17.0/1120.0,7.0/1440.0,
           -(1.0/480.0),-(13.0/3360.0),-(19.0/3360.0),7.0/1440.0,31.0/2016.0;

    BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );

    basis.MassMatrix( testGrid[ 0 ], tmp, []( double x, int ){ return x; }, 0 );
    BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );


    tmp = basis.MassMatrix( testGrid[ 0 ], []( double x ) { return x; } );
    BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );

    tmp.resize( 5, 5 );
    basis.DerivativeMatrix( testGrid[ 0 ], tmp );
    // Courtesy of Mathematica 
    ref <<  0, 2.0, 0, 2.0, 0,
            0, 0, 8.0/3.0, 0, 32.0/15.0,
            0, -(2.0/3.0), 0, 18.0/5.0, 0,
            0, 0, -(8.0/5.0), 0, 32.0/7.0,
            0, -(2.0/15.0), 0, -(18.0/7.0),0;

    BOOST_TEST( ( tmp - ref ).norm() < 1e-4 ); // Entries are only good to 0.0001 anyway

    basis.DerivativeMatrix( testGrid[ 0 ], tmp, [](  double ){ return 1.0; } );
    BOOST_TEST( ( tmp - ref ).norm() < 1e-4 );

    basis.DerivativeMatrix( testGrid[ 0 ], tmp, []( double x ){ return x; } );
    /*
     * (0	1/4	1/3	1/4	4/15
        0	1/12	1/3	7/20	4/15
        0	-(1/12)	1/15	9/20	44/105
        0	-(1/20)	-(1/5)	9/140	4/7
        0	-(1/60)	-(13/105)	-(9/28)	4/63

) */
    ref <<  0.0,   1.0/4.0,    1.0/3.0,   1.0/4.0,   4.0/15.0,
            0.0,  1.0/12.0,    1.0/3.0,  7.0/20.0,   4.0/15.0,
            0.0, -1.0/12.0,   1.0/15.0,  9.0/20.0, 44.0/105.0,
            0.0, -1.0/20.0,   -1.0/5.0, 9.0/140.0,    4.0/7.0,
            0.0, -1.0/60.0,-13.0/105.0, -9.0/28.0,   4.0/63.0;
    BOOST_TEST( ( tmp - ref ).norm() < 1e-5 );

}

BOOST_AUTO_TEST_CASE( dg_approx_static_nodal )
{
    Grid testGrid( 0.0, 1.0, 4 );
    auto foo = []( double x ){ return x; };
    auto bar = []( double x ){ return ::sin( x ); };


    Eigen::MatrixXd tmp( 5, 5 );
    tmp.setZero();
    NodalBasis basis = NodalBasis::getBasis( 4 );

    BOOST_TEST( basis.CellProduct( testGrid[ 0 ], foo, bar ) == ::sin( 0.25 ) - 0.25*::cos( 0.25 ) );
    BOOST_TEST( basis.EdgeProduct( testGrid[ 0 ], foo, bar ) == 0.25*::sin( 0.25 ) );

    Eigen::MatrixXd ref( 5,5 );

    // Nodal Mass Matrix should be 
    basis.MassMatrix( testGrid[ 0 ], tmp );
    ref << 0.0888889,  0.0259259,  -0.0296296,   0.0259259, -0.0111111,
           0.0259259,  0.483951,    0.0691358,  -0.0604938,  0.0259259,
          -0.0296296,  0.0691358,   0.632099,    0.0691358, -0.0296296,
           0.0259259, -0.0604938,   0.0691358,   0.483951,   0.0259259,
          -0.0111111,  0.0259259,  -0.0296296,   0.0259259,  0.0888889;


    ref *= 1.0/8.0;


    BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );

    basis.MassMatrix( testGrid[ 0 ], tmp, []( double ){ return 1.0; } );
    BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );

    basis.MassMatrix( testGrid[ 0 ], tmp, []( double x ){ return x; } );
    ref << 0.000173611, -0.000265195, 0.,           0.000265195, -0.000173611,
          -0.000265195,  0.0032302,   0.000373059, -0.000945216,  0.00054499,
           0.,           0.000373059, 0.00987654,   0.00178743,  -0.000925926,
           0.000265195, -0.000945216, 0.00178743,   0.0118933,    0.00107538,
          -0.000173611,  0.00054499, -0.000925926,  0.00107538,   0.00260417;


    BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );

    basis.MassMatrix( testGrid[ 0 ], tmp, []( double x, int ){ return x; }, 0 );
    BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );


    tmp = basis.MassMatrix( testGrid[ 0 ], []( double x ) { return x; } );
    BOOST_TEST( ( tmp - ref ).norm() < 1e-7 );

    tmp.resize( 5, 5 );
    basis.DerivativeMatrix( testGrid[ 0 ], tmp );
    // Courtesy of Mathematica 
    ref << -0.5,      0.67565,  -0.266667,  0.141016, -0.05,
           -0.67565,  0.,        0.95046,  -0.415826,  0.141016,
           0.266667, -0.95046,   0.0,       0.95046,  -0.266667,
           -0.141016, 0.415826, -0.95046,   0.,        0.67565,
           0.05,     -0.141016,  0.266667, -0.67565,   0.5;

    BOOST_TEST( ( tmp - ref ).norm() < 1e-4 ); // Entries are only good to 0.0001 anyway

    basis.DerivativeMatrix( testGrid[ 0 ], tmp, [](  double ){ return 1.0; } );
    BOOST_TEST( ( tmp - ref ).norm() < 1e-4 );

    /*
    basis.DerivativeMatrix( testGrid[ 0 ], tmp, []( double x ){ return x; } );
    
    ref <<  0.0,   1.0/4.0,    1.0/3.0,   1.0/4.0,   4.0/15.0,
            0.0,  1.0/12.0,    1.0/3.0,  7.0/20.0,   4.0/15.0,
            0.0, -1.0/12.0,   1.0/15.0,  9.0/20.0, 44.0/105.0,
            0.0, -1.0/20.0,   -1.0/5.0, 9.0/140.0,    4.0/7.0,
            0.0, -1.0/60.0,-13.0/105.0, -9.0/28.0,   4.0/63.0;
    BOOST_TEST( ( tmp - ref ).norm() < 1e-5 );
    */

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

    single_var.AssignSigma( []( Index, const State& s, Position x, Time ) {
            double u = s.Variable[0], q = s.Derivative[0];
            return u * ( 1.0 - q*q );
            } );

    BOOST_TEST( single_var.sigma( 0 )( 0.1 ) == 0.1 * ( ::sin( 0.1 )*::sin( 0.1 ) ) );
    BOOST_TEST( single_var.sigma( 0 )( 0.5 ) == 0.5 * ( ::sin( 0.5 )*::sin( 0.5 ) ) );
    BOOST_TEST( single_var.sigma( 0 )( 0.7 ) == 0.7 * ( ::sin( 0.7 )*::sin( 0.7 ) ) );

    double *mem2 = new double[ single_var.getDoF() ];
    DGSoln other_var( 1, testGrid, k, mem2 );

    other_var.AssignU( []( Index, double x ){ return x; } );
    other_var.AssignQ( []( Index, double ){ return 1.0; } );
    other_var.EvaluateLambda();

    other_var.AssignSigma( []( Index, const State& s, Position x, Time ) {
            double u = s.Variable[0];
            return 1.0 - u * u;
            } );

    single_var += other_var;

    BOOST_TEST( single_var.u( 0 )( 0.1 ) == 0.2 );
    BOOST_TEST( single_var.u( 0 )( 0.7 ) == 1.4 );
    BOOST_TEST( single_var.q( 0 )( 0.1 ) == 1.0 + ::cos( 0.1 ) );
    BOOST_TEST( single_var.q( 0 )( 0.7 ) == 1.0 + ::cos( 0.7 ) );

    BOOST_TEST( other_var.u( 0 )( 0.42 ) == 0.42 );
    BOOST_TEST( other_var.q( 0 )( 0.42 ) == 1.0 );

    BOOST_CHECK_NO_THROW( other_var.copy( single_var ) );

    BOOST_TEST( other_var.u( 0 )( 0.1 ) == 0.2 );
    BOOST_TEST( other_var.u( 0 )( 0.7 ) == 1.4 );
    BOOST_TEST( other_var.q( 0 )( 0.1 ) == 1.0 + ::cos( 0.1 ) );
    BOOST_TEST( other_var.q( 0 )( 0.7 ) == 1.0 + ::cos( 0.7 ) );

}


BOOST_AUTO_TEST_SUITE_END()


