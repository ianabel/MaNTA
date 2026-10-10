#include "MeshAdaptation.hpp"

#include "AdjointProblem.hpp"
#include "DegreeAdaptation.hpp"
#include "Logging.hpp"
#include "ParallelFill.hpp"
#include "PhysicsInstance.hpp"
#include "SmoothnessSensor.hpp"
#include "SystemSolver.hpp"
#include "TransportSystem.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <functional>
#include <string>
#include <tuple>
#include <limits>
#include <print>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// median / rate, both already capped at the measurable ceiling. A rate of zero
// means the spectrum has no decay at all, which is as rough as the sensor can
// report and so should always fire, and is not reachable by division.
double roughness(double median, double rate)
{
    if (rate <= 0.0)
        return std::numeric_limits<double>::infinity();
    return median / rate;
}

const char *name(GradingVerdict v)
{
    switch (v)
    {
    case GradingVerdict::GradeLower: return "Lower";
    case GradingVerdict::GradeUpper: return "Upper";
    case GradingVerdict::Uniform:    return "none";
    }
    return "none";
}
} // namespace

GradingDecision gradingDecision(std::vector<CellSmoothness> const &cells,
                                std::vector<double> const &widths,
                                unsigned int k, double threshold,
                                double neighbourMargin)
{
    if (!(threshold > 1.0))
        throw std::invalid_argument(
            "The grading threshold is a factor by which an end must be rougher "
            "than the interior, so it has to exceed 1; at or below it every mesh "
            "is graded, including one whose ends are the smoothest cells it has.");

    if (!(neighbourMargin > 0.0))
        throw std::invalid_argument(
            "The neighbour margin must be positive: at zero, any graded end whose "
            "neighbour reads a hair rougher than it would be graded again.");

    if (cells.size() < 3)
        throw std::invalid_argument(
            "Deciding whether to grade needs at least 3 cells: the two ends are "
            "compared against the interior, and with fewer there is no interior.");

    if (widths.size() != cells.size())
        throw std::invalid_argument(std::format(
            "gradingDecision needs one width per cell: {} cells, {} widths.",
            cells.size(), widths.size()));

    GradingDecision d;
    d.lowerRate = cells.front().decayRate;
    d.upperRate = cells.back().decayRate;
    d.rateCeiling = measurableDecayRate(k);

    // Every rate is compared at most at the ceiling, infinite ones included. An
    // infinite rate and one just under the ceiling are the same measurement -- a
    // top mode at round-off -- and treating the first as larger than any number
    // made an end whose top mode sat a hair above the floor "infinitely rougher"
    // than an interior whose top modes sat a hair below it. That graded Jardin,
    // whose steady state the space holds outright: an interior median of
    // infinity against a wall cell at about 30, at a ceiling of 31.5, for a cost
    // of 1.27x and an answer already at 3.9e-16. Capped, the ratio is 1.05.
    auto capped = [&](double rate) { return std::min(rate, d.rateCeiling); };

    const size_t n = cells.size();
    std::vector<double> interior;
    interior.reserve(n - 2);
    for (size_t i = 1; i + 1 < n; ++i)
        interior.push_back(capped(cells[i].decayRate));
    d.interiorMedian = median(interior);

    // Peers are interior cells within a factor 2 of the end's width. As many as
    // three are asked for, or every interior cell when there are fewer, so that
    // a uniform mesh -- where every interior cell is a peer -- reads exactly as
    // it always has, down to a 3-cell mesh's single interior cell.
    const size_t enough = std::min<size_t>(3, n - 2);
    auto readEnd = [&](size_t end, size_t inward) -> EndDecision
    {
        EndDecision e;
        const double w = widths[end];
        const double rate = capped(cells[end].decayRate);

        std::vector<double> peers;
        for (size_t i = 1; i + 1 < n; ++i)
            if (widths[i] <= 2.0 * w && widths[i] >= 0.5 * w)
                peers.push_back(capped(cells[i].decayRate));

        if (peers.size() >= enough)
        {
            e.reference = EndReference::Peers;
            e.peers = peers.size();
            e.referenceRate = median(std::move(peers));
            e.ratio = roughness(e.referenceRate, rate);
            e.reading = e.ratio >= threshold ? EndReading::Rough : EndReading::Smooth;
        }
        else if (widths[inward] >= 1.5 * w)
        {
            // Rough only if the end's own rate is also low -- below half the
            // measurable ceiling. Without that, a smooth end whose top mode sits
            // just above the round-off floor (section 13's Jardin mechanism) reads
            // as much as 1.33x rougher than its neighbour, against 1.36 for the
            // mildest singular end measured, and no margin separates them. With it
            // every smooth graded end measured reads at most 1.00x, and singular
            // ones sit at or below 0.18 of the ceiling. MESH-REFINEMENT.md section 14.
            e.reference = EndReference::Neighbour;
            e.referenceRate = capped(cells[inward].decayRate);
            e.ratio = roughness(e.referenceRate, rate);
            e.reading = e.ratio > 1.0 + neighbourMargin && rate < 0.5 * d.rateCeiling
                            ? EndReading::Rough
                            : EndReading::Smooth;
        }
        else
        {
            e.reference = EndReference::None;
            e.reading = EndReading::Undecidable;
        }
        return e;
    };

    d.lowerEnd = readEnd(0, 1);
    d.upperEnd = readEnd(n - 1, n - 2);
    d.lowerRatio = d.lowerEnd.ratio;
    d.upperRatio = d.upperEnd.ratio;

    // The single end a uniform start grades: the rougher of the rough ends, each
    // measured against its own bar so a Peers and a Neighbour reading compare
    // fairly. On a uniform mesh both bars are `threshold`, so this is the larger
    // ratio, with a tie going to the lower end -- arbitrary, and only reachable
    // when both ends are equally rough, at which point grading one end is the
    // wrong tool anyway.
    auto margin = [&](EndDecision const &e)
    {
        if (e.reading != EndReading::Rough)
            return -std::numeric_limits<double>::infinity();
        return e.ratio / (e.reference == EndReference::Peers ? threshold : 1.0 + neighbourMargin);
    };
    const double lower = margin(d.lowerEnd), upper = margin(d.upperEnd);
    if (lower > -std::numeric_limits<double>::infinity() ||
        upper > -std::numeric_limits<double>::infinity())
        d.verdict = (lower >= upper) ? GradingVerdict::GradeLower : GradingVerdict::GradeUpper;

    return d;
}

GradingDecision gradingDecision(std::vector<CellSmoothness> const &cells,
                                unsigned int k, double threshold)
{
    return gradingDecision(cells, std::vector<double>(cells.size(), 1.0), k, threshold);
}

GradingDecision gradingDecision(DGSoln const &Y, Index var, double threshold,
                                double neighbourMargin)
{
    Grid const &grid = Y.getGrid();
    std::vector<double> widths(grid.getNCells());
    for (Grid::Index i = 0; i < grid.getNCells(); ++i)
        widths[i] = grid[i].h();
    return gradingDecision(cellSmoothness(Y, var), widths, Y.getBasis().Order(), threshold,
                           neighbourMargin);
}

std::vector<Grid::Position> gradedMeshFor(GradingDecision const &decision,
                                          Grid const &uniform,
                                          Grid::Index gradedCells,
                                          double lowerFraction,
                                          double upperFraction,
                                          double ratio)
{
    if (decision.verdict == GradingVerdict::Uniform)
        throw std::logic_error(
            "gradedMeshFor was asked for a mesh from a decision that said not to "
            "grade.");

    const Grid::Index nCells = uniform.getNCells();

    // As many cells in the layer as the budget allows, unless told otherwise --
    // see the header for why this differs from GradingCells = 0 on the manual
    // path.
    const Grid::Index inLayer = (gradedCells == 0) ? nCells - 1 : gradedCells;

    return gradedMeshPoints(uniform.lowerBoundary(), uniform.upperBoundary(),
                            nCells, inLayer, lowerFraction, upperFraction, ratio,
                            decision.verdict == GradingVerdict::GradeLower
                                ? GradedEnd::Lower
                                : GradedEnd::Upper);
}

double gradingLayerFraction(GradingDecision const &decision,
                            Grid const &sampling,
                            SolverConfig const &config)
{
    if (decision.verdict == GradingVerdict::Uniform)
        throw std::logic_error(
            "gradingLayerFraction was asked for a layer from a decision that said "
            "not to grade.");

    const bool lower = decision.verdict == GradingVerdict::GradeLower;
    if (lower ? config.LowerBoundaryFractionGiven : config.UpperBoundaryFractionGiven)
        return lower ? config.LowerBoundaryFraction : config.UpperBoundaryFraction;

    const double span = sampling.upperBoundary() - sampling.lowerBoundary();
    const Grid::Index end = lower ? 0 : sampling.getNCells() - 1;
    return sampling[end].h() / span;
}

namespace
{
// A solver for one solve at a fixed degree on a given mesh, configured but not
// yet run, around whichever instance of the case may be evaluated there. Split
// out so that the sampling solve, the retry loop and the fallback share it, and
// so the "never two solvers alive" discipline lives in one place: the caller
// destroys the last solver before asking for the next, which is also what lets
// physics replace the case under it.
std::unique_ptr<SystemSolver> configuredSolver(SolverConfig const &config,
                                               PhysicsInstance &physics,
                                               Grid const &grid, unsigned int k)
{
    auto system = physics.solverFor(grid, k, [&](SystemSolver &s)
                                    { applySolverConfig(config, s); });

    // The same check runAdaptiveDegree makes, for the same reason: what selects
    // the steady path is TerminateOnSteadyState, not the SteadyStateSolver key,
    // and a configuration that time-marches would take each stage of this
    // sequence from the previous stage's final state and integrate the interval
    // again -- a wrong answer rather than a scope violation.
    if (!system->solvesForSteadyState())
        throw std::invalid_argument(
            "MeshAdaptation needs a steady solve, but this configuration "
            "time-marches: SteadyStateTolerance is absent, so steady-state "
            "termination is never armed. Set SteadyStateTolerance or "
            "SteadyStateSolve, or call run_ss().");

    return system;
}
} // namespace

namespace
{
bool isUniform(Grid const &grid)
{
    const double span = grid.upperBoundary() - grid.lowerBoundary();
    const double h = span / static_cast<double>(grid.getNCells());
    for (Grid::Index i = 0; i < grid.getNCells(); ++i)
        if (std::abs(grid[i].h() - h) > 1e-12 * span)
            return false;
    return true;
}

const char *name(EndReading r)
{
    switch (r)
    {
    case EndReading::Rough:       return "rough";
    case EndReading::Smooth:      return "smooth";
    case EndReading::Undecidable: return "undecidable";
    }
    return "undecidable";
}

const char *name(EndReference r)
{
    switch (r)
    {
    case EndReference::Peers:     return "vs peers";
    case EndReference::Neighbour: return "vs neighbour";
    case EndReference::None:      return "no reference";
    }
    return "no reference";
}

std::string describe(EndDecision const &e)
{
    if (e.reference == EndReference::None)
        return "undecidable (no cell of its width, and not narrower than its neighbour)";
    return std::format("{} ({} {:.2f}x)", name(e.reading), name(e.reference), e.ratio);
}

// The narrowest the wall cell may be made, as a fraction of the span. Section 9
// puts the integrator's ceiling and the bend in the h0 law near 1e-6, and section
// 14 puts the sensor's there too: at 1e-6 a smooth wall-layer axis reads a rate of
// 2.1-2.6 from discretisation error in u_h, as small a signal as Shestakov's
// singularity gives at that width, while at 1e-5 every end read correctly. A
// regrade the sensor could not judge on the next run is not worth making.
//
// It holds the regrades of a graded or explicit start, which are the passes a
// later run repeats. A uniform start's single redistribution is not held to it:
// sections 8-9 measured that one, at 6.6e-6 on 10 cells, and it solves.
constexpr double narrowestWallCell = 1.0e-5;
} // namespace

AdaptiveMeshResult runAdaptiveMesh(SolverConfig const &config,
                                   PhysicsInstance &physics,
                                   Grid const &start,
                                   unsigned int k0,
                                   double tFinal)
{
    // Refused rather than warned about, because at k = 2 the decision is
    // *inverted* rather than merely uncertain -- it grades a smooth problem harder
    // than a singular one. A driver that proceeded would confidently do the wrong
    // thing. See the header for the mechanism.
    if (k0 < 3)
        throw std::invalid_argument(std::format(
            "MeshAdaptation needs PolynomialDegree >= 3, but it is {}. The "
            "grading decision is read from the decay of the modal coefficients, "
            "and at k = 2 the fit has one genuinely decaying mode plus a first "
            "mode that a solution flat at a boundary -- a zero-flux axis, say -- "
            "suppresses. Measured on three problems, the verdict at k = 2 is not "
            "merely noisy but reversed. 4 or more is better still.", k0));

    // Grading moves the mesh and the degree loop moves k, so a case that can
    // follow neither is refused now rather than after the sampling solve.
    physics.requireAdaptable("MeshAdaptation");

    // What the mesh is decides what may be done to it:
    //
    //  * Uniform -- redistribute at a fixed cell count, grading one end as hard as
    //    the budget allows. Sections 8-10, unchanged.
    //  * Recipe -- a GradedGridBoundary mesh -- and Explicit -- GridPoints, or a
    //    restart file's mesh. Every face stays, because nothing says which of
    //    them sit on a feature the sensor cannot see, so each rough end cell is
    //    split into a layer and the count grows. They differ only in how many
    //    cells the split makes. The starting mesh is the fallback, since it is
    //    the one known to solve.
    enum class Start { Uniform, Recipe, Explicit };
    const Start kind = isUniform(start)                                  ? Start::Uniform
                       : config.GradedGridBoundary && !meshIsExplicit(config) ? Start::Recipe
                                                                              : Start::Explicit;

    const Index var = 0;
    const double span = start.upperBoundary() - start.lowerBoundary();

    std::println("Mesh adaptation: sampling at k = {} on {} {} cells", k0, start.getNCells(),
                 kind == Start::Uniform ? "uniform" : kind == Start::Recipe ? "graded" : "given");

    AdaptiveMeshResult result;
    result.grid = std::make_unique<Grid>(start);

    // --- p: the sampling solve, at a degree the decision can be trusted at -----
    auto sample = configuredSolver(config, physics, *result.grid, k0);
    sample->runSolver(tFinal);

    // --- h: decide, and regrade ------------------------------------------------
    result.decision = gradingDecision(sample->solution(), var,
                                      config.MeshAdaptationThreshold,
                                      config.MeshAdaptationNeighbourMargin);
    auto const &d = result.decision;

    std::println("  decay rate: lower end {:.3g}, interior median {:.3g}, upper end "
                 "{:.3g}; measurable up to {:.3g}",
                 d.lowerRate, d.interiorMedian, d.upperRate, d.rateCeiling);
    if (kind == Start::Uniform)
        std::println("  roughness vs interior: lower {:.2f}x, upper {:.2f}x, threshold "
                     "{:.2f}x -> grade {}",
                     d.lowerRatio, d.upperRatio, config.MeshAdaptationThreshold,
                     name(d.verdict));
    else
        std::println("  lower end {}; upper end {}; thresholds {:.2f}x vs peers, "
                     "{:.2f}x vs neighbour",
                     describe(d.lowerEnd), describe(d.upperEnd),
                     config.MeshAdaptationThreshold, 1.0 + config.MeshAdaptationNeighbourMargin);

    // A candidate mesh at a given grading ratio, and a line saying what it is.
    struct Candidate
    {
        std::vector<Grid::Position> points;
        std::string detail;
    };
    std::function<Candidate(double)> meshAt;
    std::function<double(double)> soften;
    double ratio = config.GradingRatio;
    const char *fallbackName = "uniform";

    // The pieces of the sample a warm start needs, taken before it is destroyed.
    std::vector<double> sampleState, sampleDerivative;
    std::unique_ptr<Grid> sampleGrid;
    std::unique_ptr<SolverConfig> warmConfig;
    auto keepSample = [&]()
    {
        // Destroyed before the next is built. Integrator's weight cache is a
        // process-wide global keyed on (order, grid) and residual() revalidates it
        // on every evaluation, so two live solvers on different meshes would clear
        // and rebuild that map once per residual instead of once per level.
        //
        // What the graded solve starts from is taken out of the sample first: its
        // state, the mesh that state lives on, and the pseudo-transient step it
        // finished at. Started cold, the graded solve repeats the whole climb from
        // the initial condition; warm, it is two continuation steps -- 3835
        // physics evaluations against 475 on the wall-layer case of
        // MESH-REFINEMENT.md section 12, for the same answer to 1e-10. The state
        // crosses through the restart path, which L2-projects the element
        // polynomials onto the new mesh and rebuilds the trace there (section 4's
        // transfer, not the spline that section 9 measured failing).
        sampleState = sample->stateVector();
        sampleDerivative = sample->derivativeVector();
        sampleGrid = std::make_unique<Grid>(*result.grid);
        warmConfig = std::make_unique<SolverConfig>(carriedStepConfig(config, *sample));
    };

    if (kind == Start::Uniform && d.verdict != GradingVerdict::Uniform)
    {
        keepSample();
        Grid const &uniform = start;

        // As many cells as cost no more rounds of physics than the sample's,
        // planned by the sample's own configuration; see layerFor below for
        // where they go.
        const Grid::Index sampleCells = uniform.getNCells();
        const Grid::Index gradedCount = filledCellCount(
            [&](Grid const &g, unsigned int kk) { return sample->evaluationPlanFor(g, kk); },
            uniform, k0, std::numeric_limits<Grid::Index>::max(), config.PhysicsParallelism,
            config.MaxPhysicsBatch);
        const Grid filledUniform(uniform.lowerBoundary(), uniform.upperBoundary(), gradedCount);

        // The layer the filled mesh grades at a given ratio. Its boundaries sit
        // at layer * ratio^j from the wall, so the wall cell is layer *
        // ratio^(m-1), its neighbour layer * ratio^(m-2) * (1 - ratio), and the
        // wall cell is the narrowest only while ratio <= 1/2. Extra cells
        // therefore go into the layer at the ratio that keeps the wall cell the
        // sample's count would have given it, ratio^((m-1)/(m'-1)), for as long
        // as that stays at or below 1/2; the rest go to the bulk. At a fixed ratio
        // each one would shrink the wall cell by the ratio again, which below the
        // layer's own width is lost accuracy -- 4, 5 and 6 cells at 0.3 measured
        // 5.5e-3, 8.9e-4 and 1.3e-3 on the n = 2.5 wall layer of
        // MESH-REFINEMENT.md section 12 -- and past 1/2 they would pack cells
        // narrower than the wall cell beside it. A GradingCells holds the layer
        // where it is, and every extra cell goes to the bulk.
        struct Layer
        {
            Grid::Index cells;
            double ratio;
        };
        // By value: meshAt below outlives this block, and calls it.
        const auto layerFor = [=, &config](double r) -> Layer
        {
            const auto m = config.GradingCells == 0 ? sampleCells - 1
                                                    : static_cast<Grid::Index>(config.GradingCells);
            if (config.GradingCells != 0 || gradedCount == sampleCells || m < 2 || !(r < 0.5))
                return {m, r};
            const auto most = static_cast<Grid::Index>(
                1.0 + std::floor(static_cast<double>(m - 1) * std::log(r) / std::log(0.5)));
            const Grid::Index cells = std::min(gradedCount - 1, std::max(m, most));
            return {cells, std::pow(r, static_cast<double>(m - 1) / static_cast<double>(cells - 1))};
        };
        if (gradedCount > sampleCells)
            std::println("  filled to {} cells: no more rounds of {} than {}", gradedCount,
                         config.PhysicsParallelism, sampleCells);

        const double layer = gradingLayerFraction(d, uniform, config);
        std::println("  layer: {:.4g} of the domain{}", layer,
                     (d.verdict == GradingVerdict::GradeLower ? config.LowerBoundaryFractionGiven
                                                              : config.UpperBoundaryFractionGiven)
                         ? "" : ", the sampling mesh's end cell");


        meshAt = [=, &d](double r) -> Candidate
        {
            const Layer layerShape = layerFor(r);
            return {gradedMeshFor(d, filledUniform, layerShape.cells, layer, layer, layerShape.ratio),
                    std::format("ratio {:.4g}, {} cells in the layer", layerShape.ratio,
                                layerShape.cells)};
        };
        // Towards 1, halving the distance each time, so the sequence is monotone
        // and terminates at the uniform mesh rather than oscillating.
        soften = [](double r) { return std::sqrt(r); };
    }
    else
    {
        // A graded or explicit start: every face it has is kept, because nothing
        // says which of them sit on a feature the sensor cannot see. Shestakov's
        // source edge at x = 0.1 is the measured case: regrading a 0.2 layer by
        // squaring its ratio moved the faces around it, and the answer came out
        // 15-19x worse than adapting the degree alone. So each rough end cell is
        // split into a geometric layer of its own, and the count grows by
        // `cells - 1` per split end.
        //
        // `cells` is the configured layer's for a GradedGridBoundary mesh, which
        // takes a graded end's wall cell down by ratio^(cells-1) -- exactly the
        // wall cell squaring the ratio gave, without moving anything else -- and
        // gives an ungraded end the wall cell a configured layer over its end
        // cell would have. GradingCells, or 4, for an explicit mesh.
        const Grid::Index N = start.getNCells();
        SolverConfig recipe = config;
        recipe.GridSize = static_cast<int>(N);
        const Grid::Index cells =
            kind == Start::Recipe          ? static_cast<Grid::Index>(gradingCellsFor(recipe))
            : config.GradingCells != 0     ? static_cast<Grid::Index>(config.GradingCells)
                                           : 4;

        if (d.lowerEnd.reading == EndReading::Undecidable || d.upperEnd.reading == EndReading::Undecidable)
            std::println("  an end with no cell of its width and no wider neighbour cannot be "
                         "judged, and is left as it is");
        if (kind == Start::Recipe)
        {
            if (config.GradingEnd != "Upper" && d.lowerEnd.reading == EndReading::Smooth)
                std::println("  the graded lower end reads smooth; its grading is kept as configured");
            if (config.GradingEnd != "Lower" && d.upperEnd.reading == EndReading::Smooth)
                std::println("  the graded upper end reads smooth; its grading is kept as configured");
        }

        // How each rough end is split. The wall cell must stay at or above the
        // floor, so a narrow end cell needs a larger ratio -- but past 1/2 the
        // wall cell stops being the narrowest of the split (its neighbour is
        // H r^(m-2) (1 - r)), and the cells beside it drop below the floor
        // instead. So: the most cells, up to `cells`, whose floor-respecting
        // ratio is at most 1/2, at the configured ratio or that one if larger;
        // and if not even two cells manage it -- an end cell under twice the
        // floor -- the end is left as it is. That last case is what a second pass
        // from a deeply graded mesh reaches.
        struct Split
        {
            GradedEnd end;
            Grid::Index cells = 0;
            double ratio = 0.0;
        };
        std::vector<Split> splits;
        for (auto [reading, H, end, endName] :
             {std::tuple{d.lowerEnd.reading, start[0].h(), GradedEnd::Lower, "lower"},
              std::tuple{d.upperEnd.reading, start[N - 1].h(), GradedEnd::Upper, "upper"}})
        {
            if (reading != EndReading::Rough)
                continue;
            Split sp{end};
            for (Grid::Index m = cells; m >= 2 && sp.cells == 0; --m)
            {
                const double needed = std::pow(narrowestWallCell * span / H,
                                               1.0 / static_cast<double>(m - 1));
                if (needed <= 0.5)
                    sp = {end, m, std::max(config.GradingRatio, needed)};
            }
            if (sp.cells == 0)
                std::println("  the {} end reads rough, but its cell, {:.2e} of the domain, is "
                             "under twice the {:g} floor; left as it is", endName, H / span,
                             narrowestWallCell);
            else
            {
                if (sp.ratio > config.GradingRatio || sp.cells < cells)
                    std::println("  {} end: {} cells at ratio {:.4g}, held to keep its wall cell "
                                 "above {:g} of the domain", endName, sp.cells, sp.ratio,
                                 narrowestWallCell);
                splits.push_back(sp);
            }
        }

        if (!splits.empty())
        {
            keepSample();
            fallbackName = kind == Start::Recipe ? "configured graded" : "given";
            std::println("  splitting the {} end cell{}; every other face is kept",
                         splits.size() == 2 ? "lower and upper"
                         : splits[0].end == GradedEnd::Lower ? "lower" : "upper",
                         splits.size() == 2 ? "s" : "");

            // `scale` softens every split's ratio towards 1/2 -- r^scale, so 1 is
            // the planned split and 0 would be 1 -- capped at 1/2, past which the
            // layer inverts. Once every ratio is at the cap the mesh stops
            // changing, and the attempt loop stops with it.
            ratio = 1.0;
            meshAt = [&start, splits](double scale) -> Candidate
            {
                std::vector<Grid::Position> points;
                std::string detail;
                Grid current = start;
                for (Split const &sp : splits)
                {
                    const double r = std::min(0.5, std::pow(sp.ratio, scale));
                    points = subdividedEndPoints(current, sp.end, sp.cells, r);
                    current = Grid(points);
                    detail += std::format("{}{} end {} cells at ratio {:.4g}",
                                          detail.empty() ? "" : ", ",
                                          sp.end == GradedEnd::Lower ? "lower" : "upper",
                                          sp.cells, r);
                }
                return {points, detail};
            };
            soften = [](double scale) { return scale / 2.0; };
        }
    }

    if (meshAt)
    {
        sample.reset();

        std::vector<Grid::Position> previous;
        for (unsigned int attempt = 1; attempt <= config.MeshAdaptationAttempts;
             ++attempt)
        {
            Candidate candidate = meshAt(ratio);
            // Softening that no longer changes the mesh would only re-solve a mesh
            // already known to fail.
            if (candidate.points == previous)
                break;
            previous = candidate.points;
            auto graded = std::make_unique<Grid>(candidate.points);

            double narrowest = span;
            for (Grid::Index i = 0; i < graded->getNCells(); ++i)
                narrowest = std::min(narrowest, (*graded)[i].h());

            std::println("  attempt {}: {}, narrowest cell {:.3e} of the domain",
                         attempt, candidate.detail, narrowest / span);

            // Built -- around a rebuilt case, if RebuildPhysicsOnRegrid calls for one --
            // outside the attempt's own failure handling below. A case that cannot
            // follow the mesh has not failed to solve on it, and softening the
            // grading would only bury that. The warm start goes on whichever
            // instance the solver was built around, so after it.
            std::unique_ptr<SystemSolver> trial =
                configuredSolver(*warmConfig, physics, *graded, k0);
            physics.problem().setRestartValues(sampleState, sampleDerivative, *sampleGrid, k0);

            try
            {
                try
                {
                    trial->runSolver(tFinal);
                    physics.problem().clearRestart();
                }
                catch (std::invalid_argument const &)
                {
                    physics.problem().clearRestart();
                    throw;
                }
                catch (std::exception const &e)
                {
                    // The warm start is a cost saving, so its failure is not
                    // evidence against the mesh: retry this mesh cold, exactly as
                    // it used to be solved, before softening anything. The failed
                    // solver goes first, so two are never alive at once.
                    physics.problem().clearRestart();
                    logmsg<LOG_LEVEL::WARNING>(
                        "Graded mesh attempt {} failed from the sample's state ({}). "
                        "Retrying it from the initial condition.", attempt, e.what());
                    std::println("  attempt {}: warm start failed; retrying cold", attempt);
                    trial.reset();
                    trial = configuredSolver(config, physics, *graded, k0);
                    trial->runSolver(tFinal);
                }
                result.grid = std::move(graded);
                result.solver = std::move(trial);
                result.gradingAttempts = attempt;
                break;
            }
            catch (std::invalid_argument const &)
            {
                // A configuration error, not a solver failure. Softening the mesh
                // would not fix it and retrying would bury it.
                throw;
            }
            catch (std::exception const &e)
            {
                // The rejected-step path. Section 9 measured this as the real
                // ceiling on grading -- IDA's corrector failing at
                // |h| = MinStepSize once the narrowest cell is around 1e-6 of the
                // span -- and it is not monotone in anything, so a softer mesh is
                // worth trying rather than assuming the whole idea has failed.
                logmsg<LOG_LEVEL::WARNING>(
                    "Graded mesh attempt {} failed to solve ({}). Softening the "
                    "grading ratio and retrying.", attempt, e.what());
                std::println("  attempt {} failed; softening the ratio", attempt);
                ratio = soften(ratio);
            }
        }

        if (result.solver == nullptr)
        {
            logmsg<LOG_LEVEL::WARNING>(
                "Every graded mesh attempted failed to solve, so the run continues "
                "on the {} mesh. The decision to grade stands -- the sensor "
                "said this problem wants it -- and what failed is the time "
                "integrator, so MinStepSize is the first thing to lower.", fallbackName);
            std::println("  all {} attempts failed; continuing on the {} mesh",
                         config.MeshAdaptationAttempts, fallbackName);
            result.grid = std::make_unique<Grid>(start);
            result.gradingAttempts = config.MeshAdaptationAttempts;

            // The degree loop's first level, solved here so that it is this
            // driver, which has just moved the case off the starting mesh, that
            // brings it back.
            result.solver = configuredSolver(config, physics, *result.grid, k0);
            result.solver->runSolver(tFinal);
        }
        else if (kind != Start::Uniform)
        {
            // The result is a list of boundaries whatever the start was, so say
            // what it is in the form a configuration takes, for a later run.
            std::string list;
            for (Grid::Index i = 0; i < result.grid->getNCells(); ++i)
                list += std::format("{:.17g}, ", (*result.grid)[i].x_l);
            list += std::format("{:.17g}", result.grid->upperBoundary());
            std::println("  GridPoints = [{}]", list);
        }
    }
    else
    {
        result.solver = std::move(sample);
    }

    // --- p: the degree loop, on whichever mesh won -----------------------------
    //
    // Handed the mesh rather than the config, so it never consults the grading
    // keys and cannot rebuild a different one. And handed the solve already made
    // on that mesh at k0 -- the sample, or the graded trial that converged -- as
    // its first level, because that is exactly the solve the loop would open with.
    // It used to be discarded here and repeated from cold: one solve in three on
    // every graded run, and one in two on every uniform one, for an identical
    // answer. When every graded attempt failed, that is the starting mesh's solve
    // made just above.
    const bool moved = meshAt && result.solver != nullptr &&
                       !(*result.grid == start);
    std::println("Mesh adaptation: adapting the degree on the {} mesh",
                 moved ? "regraded" : kind == Start::Uniform ? "uniform" : "starting");
    result.solver = runAdaptiveDegree(config, physics, *result.grid, k0,
                                      tFinal, std::move(result.solver));
    return result;
}
