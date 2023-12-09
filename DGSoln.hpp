#ifndef DGSOLN_HPP
#define DGSOLN_HPP

#include "Types.hpp"
#include <functional>
#include <cassert>

#include "gridStructures.hpp"


class State {
	public:
		State() = default;
		explicit State( Index nv, Index ns = 0 ) {
			Variable.resize( nv, 0.0 );
			Derivative.resize( nv, 0.0 );
			Flux.resize( nv, 0.0 );
			Scalars.resize( ns, 0.0 );
		}

		std::vector<double> Variable,Derivative,Flux;
		std::vector<double> Scalars;
};

class DGSoln
{
public:
	DGSoln(Index n_var, Grid const &_grid, Index Order, Index Scalars = 0) : nVars(n_var), grid(_grid), k(Order), nScalars( Scalars ), mu_( nullptr, 0 ) {};

	DGSoln(Index n_var, Grid const &_grid, Index Order, double *memory, Index Scalars = 0) : nVars(n_var), grid(_grid), k(Order), nScalars( Scalars ), mu_( nullptr, 0 ) { Map(memory); };

	virtual ~DGSoln() = default;

	Index getNumVars() const { return nVars; };

	size_t getDoF() const
	{
		// 3 = u + q + sigma
		// nCells + 1 for lambda because we store values at both ends
		// and we are carrying nScalar scalar variables
		return grid.getNCells() * nVars * (k + 1) * 3 +
		         (grid.getNCells() + 1) * nVars + nScalars;
	};

	void Map(double *Y)
	{
		u_.clear();
		u_.reserve(nVars);
		q_.clear();
		q_.reserve(nVars);
		sigma_.clear();
		sigma_.reserve(nVars);
		lambda_.clear();
		lambda_.reserve(nVars);
		auto nCells = grid.getNCells();
		for (int var = 0; var < nVars; var++)
		{
			sigma_.emplace_back(grid, k, (Y + var * (k + 1)), 3 * nVars * (k + 1));
			q_.emplace_back(grid, k, (Y + nVars * (k + 1) + var * (k + 1)), 3 * nVars * (k + 1));
			u_.emplace_back(grid, k, (Y + 2 * nVars * (k + 1) + var * (k + 1)), 3 * nVars * (k + 1));

			lambda_.emplace_back(Y + nVars * (nCells) * (3 * k + 3) + var * (nCells + 1), (nCells + 1));
		}
		new ( &mu_ ) VectorWrapper( Y + nVars * nCells * ( 3 * k + 3 ) + nVars * ( nCells + 1 ), nScalars );
	};

	// Accessors, both const & non-const
	DGApprox &u(Index i) { return u_[i]; };
	DGApprox const &u(Index i) const { return u_[i]; };

	DGApprox &q(Index i) { return q_[i]; };
	DGApprox const &q(Index i) const { return q_[i]; };

	DGApprox &sigma(Index i) { return sigma_[i]; };
	DGApprox const &sigma(Index i) const { return sigma_[i]; };

	VectorWrapper &lambda(Index i) { return lambda_[i]; };
	VectorWrapper const &lambda(Index i) const { return lambda_[i]; };

	double Scalar( Index j ) const { return mu_[ j ]; };
	double& Scalar( Index j ) { return mu_[ j ]; };

	VectorWrapper const & Scalars() const { return mu_; };

	State eval( double x ) const {
		State out;
		for ( Index i = 0; i < nVars; ++i ) {
			out.Variable.emplace_back( u_[ i ]( x ) );
			out.Derivative.emplace_back( q_[ i ]( x ) );
			out.Flux.emplace_back( sigma_[ i ]( x ) );
		}
		for ( Index i = 0; i < nScalars; ++i ) {
			out.Scalars.emplace_back( mu_[ i ] );
		}
		return out;
	}

	// Deep copy of the data in other to the memory we are
	// wrapping
	void copy(DGSoln const &other)
	{
		if (nVars != other.nVars)
			throw std::invalid_argument("Cannot add two DGSoln's with different numbers of variables");
		if (grid != other.grid)
			throw std::invalid_argument("Cannot add two DGSoln's with different grids");
		for (Index i = 0; i < nVars; ++i)
		{
			u_[i].copy(other.u_[i]);
			q_[i].copy(other.q_[i]);

			sigma_[i].copy(other.sigma_[i]);

			lambda_[i] = other.lambda_[i];
		}
	}

	DGSoln &operator+=(DGSoln const &other)
	{
		if (nVars != other.nVars)
			throw std::invalid_argument("Cannot add two DGSoln's with different numbers of variables");
		if (grid != other.grid)
			throw std::invalid_argument("Cannot add two DGSoln's with different grids");
		for (Index i = 0; i < nVars; ++i)
		{
			u_[i] += other.u_[i];
			q_[i] += other.q_[i];

			sigma_[i] += other.sigma_[i];

			lambda_[i] += other.lambda_[i];
		}
		return *this;
	}

	void AssignU(std::function<double(Index, double)> u_fn)
	{
		for (Index i = 0; i < nVars; ++i)
		{
			u_[i] = std::bind(u_fn, i, std::placeholders::_1);
		}
	};

	void AssignQ(std::function<double(Index, double)> q_fn)
	{
		for (Index i = 0; i < nVars; ++i)
		{
			q_[i] = std::bind(q_fn, i, std::placeholders::_1);
		}
	};

	// Sets lambda = average of u either side of the boundary
	void EvaluateLambda()
	{
		Index nCells = grid.getNCells();
		for (Index var = 0; var < nVars; ++var)
		{
			for (Index i = 0; i < nCells; ++i)
			{
				Interval const &I = grid[i];
				lambda_[var](i) += LegendreBasis::Evaluate(I, u_[var].coeffs[i].second, I.x_l) / 2.0;
				lambda_[var](i + 1) += LegendreBasis::Evaluate(I, u_[var].coeffs[i].second, I.x_u) / 2.0;
			}
			// Just set boundaries to the trace value of u. BCs are someone else's job
			lambda_[var](0) = LegendreBasis::Evaluate(grid[0], u_[var].coeffs[0].second, grid.lowerBoundary());
			lambda_[var](nCells) = LegendreBasis::Evaluate(grid[nCells - 1], u_[var].coeffs[nCells - 1].second, grid.upperBoundary());
		}
	};

	void AssignSigma(std::function<Value(Index, const State &, Position, Time)> sigmaFn)
	{
		auto const &x_vals = DGApprox::Integrator().abscissa();
		auto const &x_wgts = DGApprox::Integrator().weights();
		const size_t n_abscissa = x_vals.size();

		for (Index var = 0; var < nVars; ++var)
		{
			for (auto &coeffPair : sigma_[var].coeffs)
			{
				Interval const &I = coeffPair.first;
				coeffPair.second.setZero();
				for (size_t i = 0; i < n_abscissa; ++i)
				{
					// Pull the loop over the gaussian integration points
					// outside so we can evaluate u, q, sigmaFn once and store the values

					// Abscissa only stores positive points, so we have to double up manually

					State v1( nVars, nScalars ),v2( nVars, nScalars );

					double y_plus = I.x_l + (1 + x_vals[i]) * I.h() / 2.0;
					double y_minus = I.x_l + (1 - x_vals[i]) * I.h() / 2.0;
					double wgt = x_wgts[i] * (I.h() / 2.0);
					for (Index j = 0; j < nVars; ++j)
					{
						v1.Variable[j]   = u_[j](y_plus, I);
						v1.Derivative[j] = q_[j](y_plus, I);
						v2.Variable[j] = u_[j](y_minus, I);
						v2.Derivative[j] = q_[j](y_minus, I);
					}

					double sigma_plus = sigmaFn(var, v1, y_plus, 0.0);
					double sigma_minus = sigmaFn(var, v2, y_minus, 0.0);

					for (Index j = 0; j < k + 1; ++j)
					{
						coeffPair.second[j] += wgt * sigma_plus * LegendreBasis::Evaluate(I, j, y_plus);
						coeffPair.second[j] += wgt * sigma_minus * LegendreBasis::Evaluate(I, j, y_minus);
					}
				}
			}
		}
	}

	void zeroCoeffs()
	{
		Index nInitialised = u_.size();
		assert(nInitialised == 0 || nInitialised == nVars);
		if (nInitialised == 0)
			return;
		for (Index i = 0; i < nVars; ++i)
		{
			u_[i].zeroCoeffs();
			q_[i].zeroCoeffs();
			sigma_[i].zeroCoeffs();
			lambda_[i].setZero();
		}
		mu_.setZero();
	}

private:
	const Index nVars;
	const Grid &grid;
	const Index k;
	const Index nScalars;
	std::vector<DGApprox> u_;
	std::vector<DGApprox> q_;
	std::vector<DGApprox> sigma_;
	std::vector<VectorWrapper> lambda_;
	VectorWrapper mu_;
};



#endif // DGSOLN_HPP
