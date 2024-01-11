#ifndef DGAPPROX_HPP
#define DGAPPROX_HPP

#include "gridStructures.hpp"
#include "DGBasis.hpp"

#include <map>
#include <memory>
#include <algorithm>
#include <boost/math/quadrature/gauss.hpp>
#include <cmath>
#include <vector>
#include <iostream>


template<typename BasisType> class DGApprox
{
	public:
		using Position = double;

		DGApprox() = delete;
		DGApprox( const DGApprox &other ) = default; // Allow copy-construction (the copy will reference the same data as the original, unless the original owned its data in which case a deep copy is done)
		DGApprox( DGApprox && ) = default;

		~DGApprox() = default;

		DGApprox( Grid const& _grid, unsigned int Order )
			: grid( _grid ),k( Order ), coeffs(), Basis( Order )
		{
			coeffs.reserve( grid.getNCells() );
		};

		/*
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
		*/

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

		// Do a copy from other's memory into ours
		void copy( DGApprox const& other )
		{
			if ( grid != other.grid )
				throw std::invalid_argument( "To use copy, construct from the same grid." );
			if ( k != other.k )
				throw std::invalid_argument( "Cannot change order of polynomial approximation via copy()." );
			
			coeffs = other.coeffs;

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

		double operator()( Position x ) const {
			constexpr double eps = 1e-15;
			if ( x < grid.lowerBoundary() && ::fabs( x - grid.lowerBoundary() ) < eps )
				x = grid.lowerBoundary();
			else if (  x > grid.upperBoundary() && ::fabs( x - grid.upperBoundary() ) < eps )
				x = grid.upperBoundary();

			for ( auto const & I : coeffs )
			{
				if (  I.first.contains( x ) )
					return Basis.Evaluate( I.first, I.second, x );
			}
			throw std::logic_error( "Evaluation outside of grid" );
		};

		double operator()( Position x, Interval const& I ) const {
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

		static void MassMatrix( Interval const& I, MatrixRef u ) {
			// The unweighted mass matrix is the identity.
			u.setIdentity();
		};

		static void MassMatrix( Interval const& I, MatrixRef u, std::function< double( double )> const& w ) {
			for ( Index i = 0 ; i < u.rows(); i++ )
				for ( Index j = 0 ; j < u.cols(); j++ )
				{
					auto F = [ & ]( double x ) { return w( x ) * Basis.Evaluate( I, i, x ) * Basis.Evaluate( I, j, x ); };
					u( i, j ) = integrator.integrate( F, I.x_l, I.x_u );
				}
		};

		static void MassMatrix( Interval const& I, MatrixRef u, std::function< double( double, int )> const& w, int var ) {
			for ( Index i = 0 ; i < u.rows(); i++ )
				for ( Index j = 0 ; j < u.cols(); j++ )
				{
					auto F = [ & ]( double x ) { return w( x, var ) * Basis.Evaluate( I, i, x ) * Basis.Evaluate( I, j, x ); };
					u( i, j ) = integrator.integrate( F, I.x_l, I.x_u );
				}
		};

		Matrix MassMatrix( Interval const& I )
		{
			return Matrix::Identity( k + 1, k + 1 );
		}

		Matrix MassMatrix( Interval const& I, std::function<double( double )> const&w )
		{
			Matrix u ( k + 1, k + 1 );
			for ( Index i = 0 ; i < u.rows(); i++ )
				for ( Index j = 0 ; j < u.cols(); j++ )
				{
					auto F = [ & ]( double x ) { return w( x ) * Basis.Evaluate( I, i, x ) * Basis.Evaluate( I, j, x ); };
					u( i, j ) = integrator.integrate( F, I.x_l, I.x_u );
				}
			return u;
		}

		static void DerivativeMatrix( Interval const& I, MatrixRef D ) {
			for ( Index i = 0 ; i < D.rows(); i++ )
				for ( Index j = 0 ; j < D.cols(); j++ )
				{
					auto F = [ & ]( double x ) { return Basis.Evaluate( I, i, x ) * Basis.Prime( I, j, x ); };
					D( i, j ) = integrator.integrate( F, I.x_l, I.x_u );
				}
		}

		static void DerivativeMatrix( Interval const& I, MatrixRef D, std::function<double ( double )> const& w ) {
			for ( Index i = 0 ; i < D.rows(); i++ )
				for ( Index j = 0 ; j < D.cols(); j++ )
				{	
					auto F = [ & ]( double x ) { return w( x )*Basis.Evaluate( I, i, x ) * Basis.Prime( I, j, x ); };
					D( i, j ) = integrator.integrate( F, I.x_l, I.x_u );
				}
		}

		void zeroCoeffs() {
			for ( auto pair : coeffs )
				pair.second = Vector::Zero( pair.second.size() );
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
		std::pair< Interval, VectorWrapper > & getCoeff( Index i ) { return coeffs[ i ]; };
		std::pair< Interval, VectorWrapper > const& getCoeff( Index i ) const { return coeffs[ i ]; };
	private:
		const Grid& grid;
		unsigned int k;
		Coeff_t coeffs;
		static IntegratorType integrator;
		BasisType Basis;

		friend class DGSolnImpl<BasisType>;

};

#endif
