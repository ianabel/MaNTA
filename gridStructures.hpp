#pragma once
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>		/* defs of realtype, sunindextype  */

#include <map>
#include <memory>
#include <algorithm>
#include <boost/math/quadrature/gauss.hpp>
#include <cmath>
#include <format>
#include <stdexcept>
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
	Grid(Position lBound, Position uBound, Index nCells)
		: upperBound(uBound), lowerBound(lBound)
	{
		// Users eh?
		if (upperBound < lowerBound)
			std::swap(upperBound, lowerBound);

		if (upperBound - lowerBound < 1e-14)
			throw std::invalid_argument("uBound and lBound too close for representation by double");

		if (nCells == 0)
			throw std::invalid_argument("Strictly positive number of cells required to construct grid.");

		// Uniform, and only uniform. This constructor used to take a
		// `highGridBoundary` flag and two fractions and build a cosine-spaced
		// boundary layer at each end; that is gone, replaced by the geometric
		// grading in gradedMeshPoints below, which is measured rather than assumed
		// (MESH-REFINEMENT.md §8-9) and which reaches the same shapes through the
		// Grid(std::vector<Position>) constructor, where the validation lives.
		//
		// Deliberately a signature change rather than a silent reinterpretation of
		// the flag: this header is installed, so an out-of-tree caller passing the
		// old arguments should fail to compile rather than quietly get a different
		// mesh.
		Position cellLength = std::abs(upperBound - lowerBound) / static_cast<double>(nCells);
		gridCells.reserve(nCells);
		for (Index i = 0; i < nCells - 1; i++)
			gridCells.emplace_back(lowerBound + i * cellLength, lowerBound + (i + 1) * cellLength);
		gridCells.emplace_back(lowerBound + (nCells - 1) * cellLength, upperBound);

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

		// Everything below this line is about what Interval will *not* tell
		// you. Interval(a, b) silently swaps when a > b, so a descending or
		// out-of-order list yields a grid of plausible-looking cells that
		// overlap, with lowerBound/upperBound taken from the ends of the list
		// as given; and a repeated point yields a cell of zero width, whose
		// MassMatrix is (h/2) * RefMass -- identically zero -- and whose
		// toRef() divides by h. Neither reports anything here. The first
		// surfaces as a wrong answer, the second as a singular per-cell
		// FullPivLU somewhere inside the solve.
		//
		// Finiteness is checked first because a NaN compares false against
		// everything, so it would otherwise be reported as "not increasing" --
		// true, but pointing at the wrong problem.
		for (Index i = 0; i < points.size(); ++i)
			if (!std::isfinite(points[i]))
				throw std::invalid_argument(std::format("Cell boundary {} is not a finite number ({}).", i, points[i]));

		for (Index i = 0; i + 1 < points.size(); ++i)
			if (!(points[i] < points[i + 1]))
				throw std::invalid_argument(std::format("Cell boundaries must be strictly increasing, but boundary {} ({}) does not precede boundary {} ({}).", i, points[i], i + 1, points[i + 1]));

		// The same rule the (lBound, uBound, nCells) constructor applies to its
		// domain, so that the two agree on what a grid is: a config supplying
		// Grid_points should not be able to build what Grid_size would reject.
		// Individual cells are held only to strict increase -- an absolute
		// width is the wrong test for one cell of an arbitrary domain, and a
		// narrow-but-positive cell is merely expensive, not degenerate.
		if (points.back() - points.front() < 1e-14)
			throw std::invalid_argument("Cell boundaries span too small a domain for representation by double.");

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

// Which end (or ends) a graded mesh refines into.
enum class GradedEnd
{
	Lower,
	Upper,
	Both,
};

// Cell boundaries for a mesh graded geometrically into one or both ends.
//
// `gradedCells` cells cover the layer against each graded end -- of width
// `lowerFraction * span` at the bottom and `upperFraction * span` at the top --
// each `1/ratio` times the width of the one before it; the remainder are uniform
// over what is left. With `GradedEnd::Both` that is `nCells - 2*gradedCells`
// cells in the middle, and only the fraction belonging to a graded end is read.
//
// Hand it to Grid(std::vector<Position>), which is where the validation lives.
// Returning points rather than a Grid is deliberate: it is a pure function of
// six numbers, so it can be checked without constructing anything, and the
// geometry is the part worth checking.
//
// **The cell touching the graded end is the whole point of the construction**,
// and its width is exactly `fraction * (uBound - lBound) * ratio^(gradedCells-1)`.
// MESH-REFINEMENT.md §8 measures the error on Shestakov's problem as
// `0.0487 * h0` in that width and in nothing else -- not in the cell count -- so
// that expression is the knob, and §9 measures 14900x from turning it at a fixed
// cell count. Note it is *not* a pure geometric progression: this closing cell
// runs all the way to the end, so it is 1/(1-ratio) times wider than continuing
// the progression would give, and the first width ratio is (1-ratio)/ratio where
// every later one is 1/ratio. That is what makes h0 the clean expression above.
inline std::vector<Grid::Position> gradedMeshPoints(
	Grid::Position lBound, Grid::Position uBound, Grid::Index nCells,
	Grid::Index gradedCells, double lowerFraction, double upperFraction,
	double ratio, GradedEnd end)
{
	const bool gradeLower = end != GradedEnd::Upper;
	const bool gradeUpper = end != GradedEnd::Lower;
	const Grid::Index layers = (end == GradedEnd::Both) ? 2 : 1;

	if (!(ratio > 0.0) || !(ratio < 1.0))
		throw std::invalid_argument(std::format(
			"A graded mesh needs a ratio strictly between 0 and 1; got {}. At 1 "
			"every boundary lands on the same point, and above it the cells grow "
			"towards the end being refined rather than shrinking.", ratio));

	// Only the fractions actually read are checked, so a config that leaves the
	// other at something silly while grading one end is not refused for it.
	if (gradeLower && (!(lowerFraction > 0.0) || !(lowerFraction < 1.0)))
		throw std::invalid_argument(std::format(
			"A graded mesh needs a lower layer fraction strictly between 0 and 1; "
			"got {}.", lowerFraction));

	if (gradeUpper && (!(upperFraction > 0.0) || !(upperFraction < 1.0)))
		throw std::invalid_argument(std::format(
			"A graded mesh needs an upper layer fraction strictly between 0 and 1; "
			"got {}.", upperFraction));

	if (end == GradedEnd::Both && lowerFraction + upperFraction >= 1.0)
		throw std::invalid_argument(std::format(
			"Grading both ends needs the two layers to leave something between "
			"them, but the fractions sum to {} >= 1.",
			lowerFraction + upperFraction));

	if (gradedCells < 2)
		throw std::invalid_argument(std::format(
			"A graded mesh needs at least 2 cells in the graded layer; got {}. "
			"With one there is no ratio between neighbours and nothing is graded.",
			gradedCells));

	if (layers * gradedCells >= nCells)
		throw std::invalid_argument(std::format(
			"A graded mesh needs at least one cell outside the graded layer{}, so "
			"{} x {} graded cells does not fit in a grid of {}.",
			layers == 2 ? "s" : "", layers, gradedCells, nCells));

	const double span = uBound - lBound;

	// Built from the axis outwards in one pass, so the cells come from
	// consecutive entries of a single list and are contiguous by construction --
	// which is what Grid::operator== and the restart round trip need. Boundaries
	// are absolute positions; only the two endpoints are pinned exactly.
	std::vector<Grid::Position> points;
	points.reserve(nCells + 1);
	points.push_back(lBound);

	if (gradeLower)
	{
		const double layer = lowerFraction * span;
		for (Grid::Index j = gradedCells; j-- > 0;)
			points.push_back(lBound + layer * std::pow(ratio, static_cast<double>(j)));
	}

	// The uniform stretch between whatever the graded layers left.
	const double bulkStart = points.back();
	const double bulkEnd = gradeUpper ? uBound - upperFraction * span : uBound;
	const Grid::Index bulkCells = nCells - layers * gradedCells;
	const double bulkLength = (bulkEnd - bulkStart) / static_cast<double>(bulkCells);
	for (Grid::Index i = 1; i <= bulkCells; ++i)
		points.push_back(i == bulkCells ? bulkEnd : bulkStart + i * bulkLength);

	if (gradeUpper)
	{
		// The mirror of the lower layer: widths in the same geometric sequence,
		// laid out shrinking towards uBound.
		//
		// One caveat, and it is arithmetic rather than this code: the narrowest
		// cell's *width* here is a difference of two numbers near uBound, so it
		// carries an absolute error of eps(uBound) and a relative one of
		// eps(uBound)/h0 -- 3.4e-11 for h0 = 6.6e-6 on [0, 1]. No construction
		// avoids it. It bites only at very hard gradings, where the boundary
		// eventually coincides with uBound and Grid rejects the zero-width cell.
		const double layer = upperFraction * span;
		for (Grid::Index j = 1; j < gradedCells; ++j)
			points.push_back(uBound - layer * std::pow(ratio, static_cast<double>(j)));
		points.push_back(uBound);
	}

	// Exact, because Grid::operator== compares them and the restart round trip
	// rebuilds a grid from the boundaries it wrote.
	points.front() = lBound;
	points.back() = uBound;
	return points;
}


#include "Basis.hpp"

