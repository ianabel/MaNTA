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
using BasisType = ChebyshevBasis;

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

			// Chebyshev Locations for edge nodes
			for (Index i = 0; i < BoundaryCells; i++)
			{
				double cellLeft = lBoundaryLayer - lBoundaryWidth * cos((pi * i) / (2.0 * BoundaryCells - 1.0));
				double cellRight = lBoundaryLayer - lBoundaryWidth * cos((pi * (i + 1)) / (2.0 * BoundaryCells - 1.0));
				if (i == BoundaryCells - 1)
					cellRight = lBoundaryLayer;
				gridCells.emplace_back(cellLeft, cellRight);
			}
			for (Index i = 0; i < BulkCells; i++)
				gridCells.emplace_back(lBoundaryLayer + i * bulkCellLength, lBoundaryLayer + (i + 1) * bulkCellLength);
			for (Index i = 0; i < BoundaryCells; i++)
			{
				double cellLeft = uBoundaryLayer + uBoundaryWidth * cos(pi * (BoundaryCells - i) / (2.0 * BoundaryCells - 1.0));
				double cellRight = uBoundaryLayer + uBoundaryWidth * cos(pi * (BoundaryCells - i - 1) / (2.0 * BoundaryCells - 1.0));
				if (i == 0)
					cellLeft = uBoundaryLayer;

				gridCells.emplace_back(cellLeft, cellRight);
			}
		}
		if (gridCells.size() != nCells)
			throw std::runtime_error("Unable to construct grid.");
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

