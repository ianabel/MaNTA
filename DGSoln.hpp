
class DGSoln {

	DGSoln( Index n_var, Grid const& _grid, Index Order ) : nVars( n_var ), grid( _grid ),k( Order )
	{

	};

	DGSoln( Index n_var, Grid const& _grid, Index Order, double[] memory ) : nVars( n_var ), grid( _grid ),k( Order )
	{

	};

	virtual ~DGSoln() = default;

	Index getNumVars() const { return nVars; };

	size_t getMemSize() const { 
		// 3 = u + q + sigma
		return grid.getNCells() * nVars * ( k + 1 ) * 3;
	};
	
	void Map( double[] memory ) {

	}

	DGApprox & u( Index i ) { return u[ i ]; };
	DGApprox const& u( Index i ) const { return u[ i ]; };
	DGApprox & q( Index i ) { return q[ i ]; };
	DGApprox const& q( Index i ) const { return q[ i ]; };
	DGApprox & sigma( Index i ) { return sigma[ i ]; };
	DGApprox const& sigma( Index i ) const { return sigma[ i ]; };

	DGSoln& operator+=( DGSoln const& other ) 
	{
		if ( nVars != other.getNumVars() )
			throw std::invalid_argument( "Cannot add two DGSoln's with different numbers of variables" );
		for ( Index i=0; i < nVars; ++i )
		{
			u[ i ] += other.u[ i ];
			q[ i ] += other.q[ i ];
			sigma[ i ] += other.sigma[ i ];
		}
	}



	private:
		const Index nVars;
		const Grid& grid;
		const Index k;
		std::vector<DGApprox> u,q,sigma;
}
