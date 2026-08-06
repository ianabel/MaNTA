#pragma once
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>		/* defs of realtype, sunindextype  */

#include <map>
#include <memory>
#include <algorithm>
#include <boost/math/quadrature/gauss.hpp>
#include <cmath>
#include <vector>
#include <iostream>

#include "Types.hpp"

typedef std::function<double(double)> Fn;

#include <numbers>
using std::numbers::pi;

class LegendreBasis;
class ChebyshevBasis;
class NodalBasis;

template<class> class DGSolnImpl;

class Interval
{
public:
	Interval(double a, double b)
	{
		x_l = (a > b) ? b : a;
		x_u = (a > b) ? a : b;
	};
	Interval(Interval const &I)
	{
		x_l = I.x_l;
		x_u = I.x_u;
	};

	friend bool operator<(Interval const &I, Interval const &J)
	{
		return I.x_l < J.x_l;
	}

	friend bool operator>(Interval const &I, Interval const &J)
	{
		return I.x_u > J.x_u;
	}

	friend bool operator==(Interval const &I, Interval const &J)
	{
		return (I.x_l == J.x_l) && (I.x_u == J.x_u);
	}

	double x_l, x_u;
	bool inline contains(double x) const { return (x_l <= x) && (x <= x_u); };
	double inline h() const { return (x_u - x_l); };
    double inline toRef(double x) const { return 2 * (x - x_l) / (x_u - x_l) - 1.0; };
    double inline fromRef(double x) const { return (x+1.0)*(x_u-x_l)/2.0 + x_l; };
};

class Grid
{
public:
	using Index = size_t;
	using Position = double;
	Grid() = default;
	Grid(Position lBound, Position uBound, Index nCells, bool highGridBoundary = false, double lowerBoundaryFraction = 0.2, double upperBoundaryFraction = 0.2)
		: upperBound(uBound), lowerBound(lBound)
	{
		// Users eh?
		if (upperBound < lowerBound)
			std::swap(upperBound, lowerBound);

		if (upperBound - lowerBound < 1e-14)
			throw std::invalid_argument("uBound and lBound too close for representation by double");

		if (nCells == 0)
			throw std::invalid_argument("Strictly positive number of cells required to construct grid.");

		if (!highGridBoundary)
		{
			Position cellLength = abs(upperBound - lowerBound) / static_cast<double>(nCells);
			for (Index i = 0; i < nCells - 1; i++)
				gridCells.emplace_back(lowerBound + i * cellLength, lowerBound + (i + 1) * cellLength);
			gridCells.emplace_back(lowerBound + (nCells - 1) * cellLength, upperBound);

			if (gridCells.size() != nCells)
				throw std::runtime_error("Unable to construct grid.");
		}
		else
		{
			// [ 20 % ] [ 60 % ] [ 20 % ] with 1/3rd cells in each
			double lBoundaryFraction = lowerBoundaryFraction;
			double uBoundaryFraction = upperBoundaryFraction;
			double lBoundaryWidth = (upperBound - lowerBound) * (lBoundaryFraction);
			double uBoundaryWidth = (upperBound - lowerBound) * (uBoundaryFraction);
			double lBoundaryLayer = lowerBound + lBoundaryWidth;
			double uBoundaryLayer = upperBound - uBoundaryWidth;

			unsigned int BoundaryCells = nCells / 3;
			unsigned int BulkCells = nCells - 2 * BoundaryCells;

			double bulkCellLength = (uBoundaryLayer - lBoundaryLayer) / static_cast<double>(BulkCells);

			// Build the full list of cell boundaries first, then form cells
			// from consecutive entries. Constructing each cell from its own
			// independently-computed endpoints (as this used to) left the two
			// sides of a shared face differing in the last bits -- the bulk
			// accumulated lBoundaryLayer + i*bulkCellLength while the upper
			// layer started from uBoundaryLayer computed directly, so the grid
			// was not exactly contiguous. That is invisible to a tolerance
			// comparison but makes Grid::operator== false, which breaks the
			// restart round trip (StoreGridInfo writes one boundary per face,
			// so the rebuilt grid silently closes the gap and compares
			// unequal to the grid it came from).
			std::vector<Position> boundaries;
			boundaries.reserve(nCells + 1);
			boundaries.push_back(lowerBound);

			// Chebyshev locations for edge nodes
			for (Index i = 0; i < BoundaryCells; i++)
			{
				double cellRight = (i == BoundaryCells - 1)
									   ? lBoundaryLayer
									   : lBoundaryLayer - lBoundaryWidth * cos((pi * (i + 1)) / (2.0 * BoundaryCells - 1.0));
				boundaries.push_back(cellRight);
			}
			for (Index i = 0; i < BulkCells; i++)
			{
				double cellRight = (i == BulkCells - 1)
									   ? uBoundaryLayer
									   : lBoundaryLayer + (i + 1) * bulkCellLength;
				boundaries.push_back(cellRight);
			}
			for (Index i = 0; i < BoundaryCells; i++)
			{
				double cellRight = (i == BoundaryCells - 1)
									   ? upperBound
									   : uBoundaryLayer + uBoundaryWidth * cos(pi * (BoundaryCells - i - 1) / (2.0 * BoundaryCells - 1.0));
				boundaries.push_back(cellRight);
			}

			for (Index i = 0; i + 1 < boundaries.size(); ++i)
				gridCells.emplace_back(boundaries[i], boundaries[i + 1]);
		}
		if (gridCells.size() != nCells)
			throw std::runtime_error("Unable to construct grid.");
	}

	Grid(const std::vector<Position> &points)
	{
		// points.size() - 1 is computed in size_t, so fewer than two points
		// underflows to a huge cell count rather than failing cleanly (and
		// points.front() on an empty vector is undefined behaviour).
		if (points.size() < 2)
			throw std::invalid_argument("At least two cell boundaries are required to construct a grid.");

		auto nCells = points.size() - 1;
		gridCells.reserve(nCells);
		lowerBound = points.front();
		upperBound = points.back();
		for (Index i = 0; i < nCells; ++i)
		{
			gridCells.emplace_back(points[i], points[i + 1]);
		}
	}

	Grid(const Grid &grid) = default;

	Index getNCells() const { return gridCells.size(); };

	double lowerBoundary() const { return lowerBound; };
	double upperBoundary() const { return upperBound; };

	std::vector<Interval> const &getCells() const { return gridCells; };

	Interval &operator[](Index i) { return gridCells[i]; };
	Interval const &operator[](Index i) const { return gridCells[i]; };

	friend bool operator==(const Grid &a, const Grid &b)
	{
		return ((a.upperBound == b.upperBound) && (a.lowerBound == b.lowerBound) && (a.gridCells == b.gridCells));
	};
	friend bool operator!=(const Grid &a, const Grid &b)
	{
		return !(a == b);
	};

private:
	std::vector<Interval> gridCells;
	double upperBound, lowerBound;
};


#include "Basis.hpp"

