#ifndef DGSOLN_HPP
#define DGSOLN_HPP

#include "Types.hpp"

class DGSoln {

	DGSoln( Index n_var, Grid const& _grid, Index Order ) : nVars( n_var ), grid( _grid ),k( Order ),lambda( nullptr )
	{

	};

	DGSoln( Index n_var, Grid const& _grid, Index Order, double[] memory ) : nVars( n_var ), grid( _grid ),k( Order ),lambda( nullptr )
	{
		Map( memory );
	};

	virtual ~DGSoln() = default;

	Index getNumVars() const { return nVars; };

	size_t getMemSize() const { 
		// 3 = u + q + sigma
		// nCells + 1 for lambda because we store values at both ends
		return grid.getNCells() * nVars * ( k + 1 ) * 3 + 
		        ( grid.getNCells() + 1 ) * nVars;
	};
	
	void Map( double[] Y ) {
		sigma.clear(); sigma.reserve( nVar );
		q.clear();     q.reserve( nVar );
		u.clear();     u.reserve( nVar );
		for(int var = 0; var < nVar; var++)
		{
			sigma[ var ].emplace_back( grid, k, ( Y +                var*(k+1) ), 3*nVar*( k+1 ) );
			q[ var ]    .emplace_back( grid, k, ( Y +   nVar*(k+1) + var*(k+1) ), 3*nVar*( k+1 ) );
			u[ var ]    .emplace_back( grid, k, ( Y + 2*nVar*(k+1) + var*(k+1) ), 3*nVar*( k+1 ) );

			new (&lambda[ var ]) VectorWrapper( Y + nVar*(nCells)*(3*k+3) + var*( nCells + 1 ), (nCells+1) );
		}	
	};

	// Accessors, both const & non-const
	DGApprox &      u( Index i )       { return u[ i ]; };
	DGApprox const& u( Index i ) const { return u[ i ]; };

	DGApprox &      q( Index i )       { return q[ i ]; };
	DGApprox const& q( Index i ) const { return q[ i ]; };

	DGApprox &      sigma( Index i )       { return sigma[ i ]; };
	DGApprox const& sigma( Index i ) const { return sigma[ i ]; };

	Eigen::VectorXd &      lambda( Index i )       { return lambda[ i ]; };
	Eigen::VectorXd const& lambda( Index i ) const { return lambda[ i ]; };

	DGSoln& operator+=( DGSoln const& other )
	{
		if ( nVars != other.getNumVars() )
			throw std::invalid_argument( "Cannot add two DGSoln's with different numbers of variables" );
		for ( Index i=0; i < nVars; ++i )
		{
			u[ i ] += other.u[ i ];
			q[ i ] += other.q[ i ];
			sigma[ i ] += other.sigma[ i ];
			lambda[ i ] += other.lambda[ i ];
		}
	}

	void AssignU( std::function< double( Index, double ) > u_fn ) {
		for ( Index i = 0 ; i < nVars; ++i ) {
			u[ i ] = std::bind( u_fn, i, std::placeholder::_1 );
		}
	};

	void AssignQ( std::function< double( Index, double ) > q_fn ) {
		for ( Index i = 0 ; i < nVars; ++i ) {
			q[ i ] = std::bind( q_fn, i, std::placeholder::_1 );
		}
	};

	// Sets lambda = average of u either side of the boundary
	void EvaluateLambda() {
		Index nCells = grid.getNCells();
		for ( Index var = 0; var < nVars; ++iVar ) {
			for ( Index i = 0; i < nCells; ++i ) {
				Interval const& I = grid[ i ];
				lambda[ var ][ i ] += LegendreBasis::Evaluate( I, u[ var ].coeffs[ i ].second, I.x_l ) / 2.0;
				lambda[ var ][ i + 1 ] += LegendreBasis::Evaluate( I, u[ var ].coeffs[ i ].second, I.x_u ) / 2.0;

			}
			if ( problem->isLowerBoundaryDirichlet( var ) ) {
				lambda[ var ][ 0 ] = problem->LowerBoundary( var, 0 );
			} else {
				lambda[ var ][ 0 ] = LegendreBasis::Evaluate( I, u[ var ].coeffs[ 0 ].second, grid.lowerBoudary() );
			}

			if ( problem->isUpperBoundaryDirichlet( var ) ) {
				lambda[ var ][ nCells ] = problem->UpperBoundary( var, 0 );
			} else {
				lambda[ var ][ nCells ] = LegendreBasis::Evaluate( I, u[ var ].coeffs[ nCells - 1 ].second, grid.upperBoundary() );
			}
		}
	};

	void AssignSigma( std::function< Value( Index, const Values &, const Values &, Position, Time )> sigmaFn ) {

	}

	void setZero() {
		for ( Index i = 0; i < nVars; ++i ) {
			u[ i ].zeroCoeffs();
			q[ i ].zeroCoeffs();
			sigma[ i ].zeroCoeffs();
			lambda[ i ].setZero();
		}
	}

	private:
		const Index nVars;
		const Grid& grid;
		const Index k;
		std::vector< DGApprox > u,q,sigma;
		std::vector< VectorWrapper > lambda;
};
#endif // DGSOLN_HPP
