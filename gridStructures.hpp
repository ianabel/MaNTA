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

typedef std::function<double( double )> Fn;
using Matrix = Eigen::Matrix<realtype,Eigen::Dynamic,Eigen::Dynamic>;
using MatrixWrapper = Eigen::Map<Matrix>;
using Vector = Eigen::Matrix<realtype,Eigen::Dynamic,1>;
using VectorWrapper = Eigen::Map<Vector>;
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

	double x_l,x_u;
	bool contains( double x ) const { return ( x_l <= x ) && ( x <= x_u );};
	double h() const { return ( x_u - x_l );};
};

typedef std::vector<std::vector< std::pair< Interval, Eigen::Map<Eigen::VectorXd >>>> Coeff_t;
class Grid
{
public:
	Grid(double lBound, double uBound, int nCells)
		: upperBound(uBound), lowerBound(lBound)
	{
		double cellLength = abs(uBound-lBound)/static_cast<double>(nCells);
		for(int i = 0; i < nCells - 1; i++)
			gridCells.emplace_back(lBound + i*cellLength, lBound + (i+1)*cellLength);
		gridCells.emplace_back(lBound + (nCells-1)*cellLength, uBound);
	}

	Grid(const Grid& grid) = default; 

	std::vector<Interval> gridCells;
	double upperBound, lowerBound;
};

class LegendreBasis
{
	public:
		LegendreBasis() {};
		~LegendreBasis() {};

		double Evaluate( Interval const & I, Eigen::Index i, double x )
		{
			return ::sqrt( ( 2* i + 1 )/( I.h() ) ) * std::legendre( i, 2*( x - I.x_l )/I.h() - 1.0 );
		};

		double Prime(  Interval const & I, Eigen::Index i, double x )
		{
			if ( i == 0 )
				return 0.0;
			double y = 2*( x - I.x_l )/I.h() - 1.0;
			return ::sqrt( ( 2* i + 1 )/( I.h() ) ) * ( 2*i/I.h() ) *( 1.0/( y*y-1.0 ) )*( y*std::legendre( i, y ) - std::legendre( i-1,y ) );
		};

		double Evaluate( Interval const & I, Eigen::VectorXd const& vCoeffs, double x )
		{
			double result = 0.0;
			for ( Eigen::Index i=0; i<vCoeffs.size(); ++i )
				result += vCoeffs( i ) * Evaluate( I, i, x );
			return result;
		};

		std::function<double( double )> phi( Interval const& I, Eigen::Index i )
		{
			return [=]( double x ){ 
				return ::sqrt( ( 2* i + 1 )/( I.h() ) ) * std::legendre( i, 2*( x - I.x_l )/I.h() - 1.0 );
			};
		}

		std::function<double( double )> phiPrime( Interval const& I, Eigen::Index i )
		{
			if ( i == 0 )
				return []( double ){ return 0.0; };

			return [=]( double x ){
				double y = 2*( x - I.x_l )/I.h() - 1.0;
				return ::sqrt( ( 2* i + 1 )/( I.h() ) ) * ( 2*i/I.h() ) *( 1.0/( y*y-1.0 ) )*( y*std::legendre( i, y ) - std::legendre( i-1,y ) );
			};
		}

};

class DGApprox 
{
	public:
		typedef std::vector<Interval> Mesh;
		~DGApprox() {};
		DGApprox( Grid const& grid, unsigned int Order )
		{
			k = Order;
		};

		DGApprox( unsigned int Order )
		{
			k = Order;
		};

		DGApprox( Grid const& grid, unsigned int Order, unsigned int nVariables, std::function<double( double )> const& F )
		{
			k = Order;
			for(int var=0; var<nVariables; var++)
			{
				std::vector< std::pair< Interval, Eigen::Map<Eigen::VectorXd >>> varCoeffs;
				std::vector<double> vec(k+1, 0.0);
				Eigen::Map<Eigen::VectorXd> v(&vec[0], k+1);
				{
					for ( auto const& I : grid.gridCells )
					{
						v.setZero();
						// Interpolate onto k legendre polynomials
						for ( Eigen::Index i=0; i<= k; i++ )
						{
							v( i ) = CellProduct( I, F, Basis.phi( I, i ) );
						}
						varCoeffs.emplace_back( I, v);
					}
				}
				coeffs.emplace_back(varCoeffs);
			}
		};

		DGApprox& operator=( std::function<double( double )> const & f )
		{
			Eigen::VectorXd v( k + 1 );
			for(int var=0; var<coeffs.size(); var++)
			{
				for ( auto pair : coeffs[var] )
				{
					Interval const& I = pair.first;
					v.setZero();
					// Interpolate onto k legendre polynomials
					for ( Eigen::Index i=0; i<= k; i++ )
					{
						v( i ) = CellProduct( I, f, Basis.phi( I, i ) );
					}
					pair.second = v;
				}
			}
			return *this;
		}

		void sum( DGApprox& A, DGApprox& B)
		{
			Eigen::VectorXd v( k + 1 );
			for(int var=0; var<coeffs.size(); var++)
			{
				for ( int i = 0; i < coeffs[var].size() ; i++ )
				{
					coeffs[var][i].second = A.coeffs[var][i].second + B.coeffs[var][i].second;
				}
			}
		}

		void setToFunc(std::function<double( double )> const & f, int var)
		{
			//Equivalent to operator= above but implemented for 1 variable
			Eigen::VectorXd v( k + 1 );
			for ( auto pair : coeffs[var] )
			{
				Interval const& I = pair.first;
				v.setZero();
				// Interpolate onto k legendre polynomials
				for ( Eigen::Index i=0; i<= k; i++ )
				{
					v( i ) = CellProduct( I, f, Basis.phi( I, i ) );
				}
				pair.second = v;
			}
		}
	
		double operator()( double x, int var ) {
			if(var>coeffs.size()) std::logic_error( "Out of bounds" );
			for ( auto const & I : coeffs[var] )
			{
				if (  I.first.contains( x ) )
					return Basis.Evaluate( I.first, I.second, x );
			}
			throw std::logic_error( "Out of bounds" );
		};
		
		double CellProduct( Interval const& I, std::function< double( double )> f, std::function< double( double )> g )
		{
			auto u = [ & ]( double x ){ return f( x )*g( x );};
			return integrator.integrate( u, I.x_l, I.x_u );
		};

		double EdgeProduct( Interval const& I, std::function< double( double )> f, std::function< double( double )> g )
		{
			return f( I.x_l )*g( I.x_l ) + f( I.x_u )*g( I.x_u );
		};

		void MassMatrix( Interval const& I, Eigen::MatrixXd &u ) {
			// The unweighted mass matrix is the identity.
			u.setIdentity();
		};


		void MassMatrix( Interval const& I, Eigen::MatrixXd &u, std::function< double( double )> const& w ) {
			for ( Eigen::Index i = 0 ; i < u.rows(); i++ )
				for ( Eigen::Index j = 0 ; j < u.cols(); j++ )
				{
					auto F = [ & ]( double x ) { return w( x ) * Basis.Evaluate( I, i, x ) * Basis.Evaluate( I, j, x ); };
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
			for ( Eigen::Index i = 0 ; i < u.rows(); i++ )
				for ( Eigen::Index j = 0 ; j < u.cols(); j++ )
				{
					auto F = [ & ]( double x ) { return w( x ) * Basis.Evaluate( I, i, x ) * Basis.Evaluate( I, j, x ); };
					u( i, j ) = integrator.integrate( F, I.x_l, I.x_u );
				}
			return u;
		}

		void DerivativeMatrix( Interval const& I, Eigen::MatrixXd &D ) {
			for ( Eigen::Index i = 0 ; i < D.rows(); i++ )
				for ( Eigen::Index j = 0 ; j < D.cols(); j++ )
				{
					auto F = [ & ]( double x ) { return Basis.Evaluate( I, i, x ) * Basis.Prime( I, j, x ); };
					D( i, j ) = integrator.integrate( F, I.x_l, I.x_u );
				}
		}

		void DerivativeMatrix( Interval const& I, Eigen::MatrixXd &D, std::function<double ( double )> const& w ) {
			for ( Eigen::Index i = 0 ; i < D.rows(); i++ )
				for ( Eigen::Index j = 0 ; j < D.cols(); j++ )
				{	
					auto F = [ & ]( double x ) { return w( x )*Basis.Evaluate( I, i, x ) * Basis.Prime( I, j, x ); };
					D( i, j ) = integrator.integrate( F, I.x_l, I.x_u );
				}
		}

		void zeroCoeffs() {
			for(int var=0; var<coeffs.size(); var++)
			{
				for ( auto pair : coeffs[var] )
				{
					for(int i = 0; i<pair.second.size(); i++)
						pair.second[i] = 0.0;
				}
			}
		}

		// To be only used as an R value. Do not write to this vector as it produces a copy of the values
		// Also avoid using it if possible as copying is slow
		Eigen::VectorXd multiVarCoeffs( int i )
		{
			Eigen::VectorXd coeffsLong(coeffs.size()*(k+1));
			for(int var = 0; var<coeffs.size(); var++)
			{
				coeffsLong.block(var*(k+1),0,k+1,1) = coeffs[var][i].second;
			}
			return coeffsLong;
		}

		void setCoeffsToArrayMem(double arrayPtr[], const int nVar, const int nCells, const Grid& grid)
		{
			std::vector< std::pair< Interval, Eigen::Map<Eigen::VectorXd >>> cellCoeffs;
			for(int var = 0; var < nVar; var++)
			{
				for(int i=0; i<nCells; i++)
				{
					cellCoeffs.emplace_back( grid.gridCells[ i ], VectorWrapper( arrayPtr + var*(k+1) + i*nVar*(k+1), k+1 ));
				}
				coeffs.push_back(cellCoeffs);
				cellCoeffs.clear();
			}
		}

		void printCoeffs(int var)
		{
			for(int i = 0; i < coeffs[var].size(); i++)
			{
				std::cerr << coeffs[var][i].second << std::endl;
			}
			std::cerr << std::endl;
		}

		unsigned int k;
		std::vector< std::vector< std::pair< Interval, Eigen::Map<Eigen::VectorXd >>>> coeffs; // [var][interval]
		LegendreBasis Basis;
	private:
		boost::math::quadrature::gauss<double, 30> integrator;

};

class BoundaryConditions {
public:
	double LowerBound;
	double UpperBound;
	bool isLBoundDirichlet;
	bool isUBoundDirichlet;
	std::function<double( double, double )> g_D,g_N;
};
