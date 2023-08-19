#pragma once
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */

#include <map>
#include <memory>
#include <algorithm>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <cmath>
#include <vector>
#include <iostream>

#include "Types.hpp"

typedef std::function<double( double )> Fn;

class Interval
{
public:
	Interval( double a, double b ) 
	{
		x_l = ( a > b ) ? b : a;
		x_u = ( a > b ) ? a : b;
	};
	Interval( Interval const &I )
	{
		x_l = I.x_l;
		x_u = I.x_u;
	};

	friend bool operator<( Interval const& I,  Interval const& J )
	{
		return I.x_l < J.x_l;
	}

	friend bool operator==( Interval const& I, Interval const& J )
	{
		return ( I.x_l == J.x_l ) && ( I.x_u == J.x_u );
	}

	double x_l,x_u;
	bool contains( double x ) const { return ( x_l <= x ) && ( x <= x_u );};
	double h() const { return ( x_u - x_l );};
};

class Grid
{
public:
	using Index = size_t;
	using Position = double;
	Grid() = default;
	Grid(Position lBound, Position uBound, Index nCells)
		: upperBound(uBound), lowerBound(lBound)
	{
		// Users eh?
		if ( upperBound < lowerBound ) 
			std::swap( upperBound, lowerBound );

		if ( upperBound - lowerBound < 1e-14 )
			throw std::invalid_argument( "uBound and lBound too close for representation by double" );

		if ( nCells == 0 )
			throw std::invalid_argument( "Strictly positive number of cells required to construct grid." );

		Position cellLength = (upperBound - lowerBound)/static_cast<double>(nCells);
		for ( int i = 0; i < nCells - 1; i++)
			gridCells.emplace_back(lowerBound + i*cellLength, lowerBound + (i+1)*cellLength);
		gridCells.emplace_back(lowerBound + (nCells-1)*cellLength, upperBound);
		
		if ( gridCells.size() != nCells )
			throw std::runtime_error( "Unable to construct grid." );
	}

	Grid(Position lBound, Position uBound, Index nCells, bool highGridBoundary)
		: upperBound(uBound), lowerBound(lBound)
	{
		if(!highGridBoundary)
		{
			Position cellLength = abs(uBound-lBound)/static_cast<double>(nCells);
			for ( int i = 0; i < nCells - 1; i++)
				gridCells.emplace_back(lBound + i*cellLength, lBound + (i+1)*cellLength);
			gridCells.emplace_back(lBound + (nCells-1)*cellLength, uBound);

			if ( gridCells.size() != nCells )
				throw std::runtime_error( "Unable to construct grid." );
		}
		else
		{
			double sCellLength = abs(uBound-lBound)/static_cast<double>(nCells-8)/4.0;
			double mCellLength = abs(uBound-lBound)/static_cast<double>(nCells-8)/2.0;
			double lCellLength = abs(uBound-lBound)/static_cast<double>(nCells-8);
			for ( unsigned int i = 0; i < 4; i++)
				gridCells.emplace_back(lBound + i*sCellLength, lBound + (i+1)*sCellLength);
			for ( unsigned int i = 0; i < 2; i++)
				gridCells.emplace_back(lBound + (i+2)*mCellLength, lBound + (i+3)*mCellLength);
			for ( int i = 0; i < nCells-12; i++)
				gridCells.emplace_back(lBound + (i+2)*lCellLength, lBound + (i+3)*lCellLength);
			for ( unsigned int i = 0; i < 2; i++)
				gridCells.emplace_back(lBound + (i+2*(nCells-8)-4)*mCellLength, lBound + (i+2*(nCells-8)-3)*mCellLength);
			for ( unsigned int i = 0; i < 3; i++)
				gridCells.emplace_back(lBound + (i+4*(nCells-8)-4)*sCellLength, lBound + (i+4*(nCells-8)-3)*sCellLength);
			gridCells.emplace_back(lBound + (4*(nCells-8)-1)*sCellLength, uBound);
		}
	}

	Grid(const Grid& grid) = default;

	Index getNCells() const { return gridCells.size(); };

	double lowerBoundary() const { return lowerBound; };
	double upperBoundary() const { return upperBound; };

	std::vector<Interval> const& getCells() const { return gridCells; };

	Interval& operator[]( Index i ) { return gridCells[ i ]; };
	Interval const& operator[]( Index i ) const { return gridCells[ i ]; };

	friend bool operator==( const Grid & a, const Grid & b )
	{
		return ( ( a.upperBound == b.upperBound ) && ( a.lowerBound == b.lowerBound ) && ( a.gridCells == b.gridCells ) );
	};
	friend bool operator!=( const Grid& a, const Grid & b ) 
	{
		return !( a == b );
	};
private:
	std::vector<Interval> gridCells;
	double upperBound, lowerBound;

};



class LegendreBasis
{
	public:
		LegendreBasis() {};
		~LegendreBasis() {};

		static double Evaluate( Interval const & I, Index i, double x )
		{
			return ::sqrt( ( 2* i + 1 )/( I.h() ) ) * std::legendre( i, 2*( x - I.x_l )/I.h() - 1.0 );
		};

		static double Prime(  Interval const & I, Index i, double x )
		{
			if ( i == 0 )
				return 0.0;

			double y = 2*( x - I.x_l )/I.h() - 1.0;

			if ( y == 1.0 )
				return i*( i + 1.0 )/2.0;
			if ( y == -1.0 )
				return ( i % 2 == 0 ? i*( i + 1.0 )/2.0 : - i*( i + 1.0 )/2.0 );

			return ::sqrt( ( 2* i + 1 )/( I.h() ) ) * ( 2*i/I.h() ) *( 1.0/( y*y-1.0 ) )*( y*std::legendre( i, y ) - std::legendre( i-1,y ) );
		};

		static double Evaluate( Interval const & I, Eigen::VectorXd const& vCoeffs, double x )
		{
			double result = 0.0;
			for ( Index i=0; i<vCoeffs.size(); ++i )
				result += vCoeffs( i ) * Evaluate( I, i, x );
			return result;
		};

		static std::function<double( double )> phi( Interval const& I, Index i )
		{
			return [=]( double x ){ 
				return ::sqrt( ( 2* i + 1 )/( I.h() ) ) * std::legendre( i, 2*( x - I.x_l )/I.h() - 1.0 );
			};
		}

		static std::function<double( double )> phiPrime( Interval const& I, Index i )
		{
			if ( i == 0 )
				return []( double ){ return 0.0; };

			return [=]( double x ){
				double y = 2*( x - I.x_l )/I.h() - 1.0;

				if ( y == 1.0 )
					return i*( i + 1.0 )/2.0;
				if ( y == -1.0 )
					return ( i % 2 == 0 ? i*( i + 1.0 )/2.0 : - i*( i + 1.0 )/2.0 );

				return ::sqrt( ( 2* i + 1 )/( I.h() ) ) * ( 2*i/I.h() ) *( 1.0/( y*y-1.0 ) )*( y*std::legendre( i, y ) - std::legendre( i-1,y ) );
			};
		}

};

class DGApprox 
{
	public:
		using Position = double;

		DGApprox() = delete;
		DGApprox( const DGApprox &other ) = default; // Allow copy-construction (the copy will reference the same data as the original, unless the original owned its data in which case a deep copy is done)
		DGApprox( DGApprox && ) = default;

		~DGApprox() = default;

		DGApprox( Grid const& _grid, unsigned int Order )
			: grid( _grid ),k( Order ), coeffs(), ownsData( false )
		{
			coeffs.reserve( grid.getNCells() );
		};

		DGApprox( Grid const& _grid, unsigned int Order, std::function<double( double )> const& F ) : grid( _grid )
		{
			k = Order;
			ownsData = true;

			ValueData.resize( ( k + 1 ) * grid.getNCells() );

			auto & cells = grid.getCells();
			Grid::Index nCells = grid.getNCells();
			for ( Grid::Index i = 0; i < nCells; ++i )
			{
				auto const & I = cells[ i ];
				VectorWrapper v( ValueData.data() + i * ( k + 1 ), k + 1 );
				// Interpolate onto k legendre polynomials
				for ( Index j=0; j<= k; j++ )
				{
					v( j ) = CellProduct( I, F, Basis.phi( I, j ) );
				}
				coeffs.emplace_back( I, v );
			}
		};

		DGApprox( Grid const& _grid, unsigned int Order, double* block_data, size_t stride ) : grid( _grid )
		{
			k = Order;
			Grid::Index nCells = grid.getNCells();
			coeffs.clear();
			coeffs.reserve( nCells );
			auto const& cells = grid.getCells();
			for ( Grid::Index i = 0; i < nCells; ++i )
			{
				VectorWrapper v( block_data + i*stride, k + 1 );
				coeffs.emplace_back( cells[ i ], v );
			}
		}

		void Map( double* block_data, size_t stride )
		{
			Grid::Index nCells = grid.getNCells();
			coeffs.clear();
			coeffs.reserve( nCells );
			if ( stride < k + 1 )
				throw std::invalid_argument( "stride too short, memory corrption guaranteed." );
			auto const& cells = grid.getCells();
			for ( Grid::Index i = 0; i < nCells; ++i )
			{
				VectorWrapper v( block_data + i*stride, k + 1 );
				coeffs.emplace_back( cells[ i ], v );
			}
		}

		DGApprox& operator=( std::function<double( double )> const & f )
		{
			// check for data ownership
			for ( auto pair : coeffs )
			{
				Interval const& I = pair.first;
				pair.second.setZero();
				// assert( pair.second.size == k + 1);
				// Interpolate onto k legendre polynomials
				for ( Index i=0; i<= k; i++ )
				{
					pair.second( i ) = CellProduct( I, f, Basis.phi( I, i ) );
				}
			}
			return *this;
		}

		size_t getDoF() const { return ( k + 1 ) * grid.getNCells(); };

		/*
		void sum( DGApprox& A, DGApprox& B)
		{
			for ( unsigned int i = 0; i < coeffs.size() ; i++ )
			{
				coeffs[i].second = A.coeffs[i].second + B.coeffs[i].second;
			}
		}
		*/

		DGApprox & operator+=( DGApprox const& other )
		{
			if ( grid != other.grid )
				throw std::invalid_argument( "Cannot add two DGApprox's on different grids" );
			for ( unsigned int i=0; i < coeffs.size(); ++i )
			{
				// std::assert( coeffs[ i ].first == other.coeffs[ i ].first );
				coeffs[ i ].second += other.coeffs[ i ].second;
			}
			return *this;
		}

		double operator()( Position x ) {
			for ( auto const & I : coeffs )
			{
				if (  I.first.contains( x ) )
					return Basis.Evaluate( I.first, I.second, x );
			}
			throw std::logic_error( "Evaluation outside of grid" );
		};

		double operator()( Position x, Interval const& I ) {
			if ( !I.contains( x ) ) 
				throw std::invalid_argument( "Evaluate(x, I) requires x to be in the interval I" );
			auto it = std::find_if ( coeffs.begin(), coeffs.end(), [I]( std::pair<Interval,VectorWrapper> p ) { return ( p.first == I ); } );
			if ( it == coeffs.end() )
				throw std::logic_error( "Interval I not part of the grid" );
			else
				return Basis.Evaluate( it->first, it->second, x );

		};

		static double CellProduct( Interval const& I, std::function< double( double )> f, std::function< double( double )> g )
		{
			auto u = [ & ]( double x ){ return f( x )*g( x );};
			return integrator.integrate( u, I.x_l, I.x_u );
		};

		static double EdgeProduct( Interval const& I, std::function< double( double )> f, std::function< double( double )> g )
		{
			return f( I.x_l )*g( I.x_l ) + f( I.x_u )*g( I.x_u );
		};

		static void MassMatrix( Interval const& I, Eigen::MatrixXd &u ) {
			// The unweighted mass matrix is the identity.
			u.setIdentity();
		};

		static void MassMatrix( Interval const& I, Eigen::MatrixXd &u, std::function< double( double )> const& w ) {
			for ( Index i = 0 ; i < u.rows(); i++ )
				for ( Index j = 0 ; j < u.cols(); j++ )
				{
					auto F = [ & ]( double x ) { return w( x ) * Basis.Evaluate( I, i, x ) * Basis.Evaluate( I, j, x ); };
					u( i, j ) = integrator.integrate( F, I.x_l, I.x_u );
				}
		};

		static void MassMatrix( Interval const& I, Eigen::MatrixXd &u, std::function< double( double, int )> const& w, int var ) {
			for ( Index i = 0 ; i < u.rows(); i++ )
				for ( Index j = 0 ; j < u.cols(); j++ )
				{
					auto F = [ & ]( double x ) { return w( x, var ) * Basis.Evaluate( I, i, x ) * Basis.Evaluate( I, j, x ); };
					u( i, j ) = integrator.integrate( F, I.x_l, I.x_u );
				}
		};

		Eigen::MatrixXd MassMatrix( Interval const& I )
		{
			return Eigen::MatrixXd::Identity( k + 1, k + 1 );
		}

		Eigen::MatrixXd MassMatrix( Interval const& I, std::function<double( double )> const&w )
		{
			Eigen::MatrixXd u ( k + 1, k + 1 );
			for ( Index i = 0 ; i < u.rows(); i++ )
				for ( Index j = 0 ; j < u.cols(); j++ )
				{
					auto F = [ & ]( double x ) { return w( x ) * Basis.Evaluate( I, i, x ) * Basis.Evaluate( I, j, x ); };
					u( i, j ) = integrator.integrate( F, I.x_l, I.x_u );
				}
			return u;
		}

		static void DerivativeMatrix( Interval const& I, Eigen::MatrixXd &D ) {
			for ( Index i = 0 ; i < D.rows(); i++ )
				for ( Index j = 0 ; j < D.cols(); j++ )
				{
					auto F = [ & ]( double x ) { return Basis.Evaluate( I, i, x ) * Basis.Prime( I, j, x ); };
					D( i, j ) = integrator.integrate( F, I.x_l, I.x_u );
				}
		}

		static void DerivativeMatrix( Interval const& I, Eigen::MatrixXd &D, std::function<double ( double )> const& w ) {
			for ( Index i = 0 ; i < D.rows(); i++ )
				for ( Index j = 0 ; j < D.cols(); j++ )
				{	
					auto F = [ & ]( double x ) { return w( x )*Basis.Evaluate( I, i, x ) * Basis.Prime( I, j, x ); };
					D( i, j ) = integrator.integrate( F, I.x_l, I.x_u );
				}
		}

		void zeroCoeffs() {
			for ( auto pair : coeffs )
				pair.second = Eigen::VectorXd::Zero( pair.second.size() );
		}

		/*
		//This is only to be used for temporary DGAs, for longer lived DGAs please create a sundials vector
		//DGAs don't own their own memory. Usually they will just be assigned memory blocks within sundials vectors.
		//In the case that a DGA is created without any sundials vector ever being made we need to assign a memory block
		//This just sets the arrayPtr (which you should make sure is the right size to hold all your coefficients) as the holding place for the data
		//Note: if the assigned arrayptr memory block goes out of scope you will have a memory leak and you'll get undefined behaviour
		void setCoeffsToArrayMem(double arrayPtr[], const unsigned int nVar, const int nCells, const Grid& grid)
		{
			std::vector< std::pair< Interval, Eigen::Map<Eigen::VectorXd >>> cellCoeffs;
			for ( unsigned int var = 0; var < nVar; var++)
			{
				for ( int i=0; i<nCells; i++)
				{
					cellCoeffs.emplace_back( grid.gridCells[ i ], VectorWrapper( arrayPtr + var*(k+1) + i*nVar*(k+1), k+1 ));
				}
				coeffs.push_back(cellCoeffs);
				cellCoeffs.clear();
			}
		}
		*/

		void printCoeffs()
		{
			for ( auto const& x : coeffs )
			{
				std::cerr << x.second << std::endl;
			}
			std::cerr << std::endl;
		}

		double maxCoeff()
		{
			double coeff = 0.0;
			for ( auto pair : coeffs )
			{
				if( ::abs(coeff) < ::abs( pair.second.maxCoeff() ) )
					coeff = pair.second.maxCoeff();
			}
			return coeff;
		}

		using IntegratorType = boost::math::quadrature::gauss<double, 30>;
		using Coeff_t = std::vector< std::pair< Interval, VectorWrapper > >;

		static const IntegratorType& Integrator() { return integrator; };

		unsigned int getOrder() { return k;};
		Coeff_t const& getCoeffs() { return coeffs; };
	private:
		const Grid& grid;
		unsigned int k;
		Coeff_t coeffs;
		std::vector<double> ValueData;
		bool ownsData = false;
		static LegendreBasis Basis;
		static IntegratorType integrator;

		friend class DGSoln;

};

/*
class BoundaryConditions {
public:
	double LowerBound;
	double UpperBound;
	bool isLBoundDirichlet;
	bool isUBoundDirichlet;
	std::function<double( double, double, int )> g_D,g_N;
};
*/
