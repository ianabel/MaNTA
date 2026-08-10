#ifndef POSTPROCESSING_HPP
#define POSTPROCESSING_HPP

#include "Types.hpp"

// Eigen/Core and Eigen/Dense before the project headers, matching
// SystemSolver.hpp. The build defines EIGEN_USE_BLAS, which swaps in BLAS-backed
// product specialisations, so a translation unit that reaches Eigen only through
// Basis.hpp's <Eigen/LU> sees a different set of definitions from one that
// includes <Eigen/Dense> -- an ODR difference across TUs that LTO resolves by
// picking one, with memory corruption as the symptom.
#include <Eigen/Core>
#include <Eigen/Dense>

#include "DGSoln.hpp"
#include "gridStructures.hpp"

#include <vector>

/*
 * Element-local superconvergent postprocessing.
 *
 * Chen, Cockburn, Singler & Zhang, "Superconvergent Interpolatory HDG Methods
 * for Reaction Diffusion Equations I: An HDGk Method", J Sci Comput 81 (2019)
 * 2188-2212, equations (6) and (7).
 *
 * On each cell K we reconstruct u* in P_{k+1} from the degree-k pair (u_h, q_h)
 * by solving the local Neumann problem
 *
 *     ( d_x u*, d_x z )_K + ( eta, z )_K = ( q_h, d_x z )_K   for all z in P_{k+1}
 *     ( u*, w )_K                        = ( u_h, w )_K       for all w in P_0
 *
 * where eta in P_0 is a Lagrange multiplier for the pure-Neumann singularity of
 * the first equation. Note the sign of the right-hand side: the paper carries
 * q = -grad u, MaNTA carries q = d_x u.
 *
 * In coefficients this is the saddle-point system (paper I, p. 2192)
 *
 *     [ A1  b1^T ] [ gamma ]   [ A2   0  ] [ alpha_q ]
 *     [ b1   0   ] [ eta   ] = [  0  b2  ] [ beta_u  ]
 *
 * whose elimination gives, per cell,
 *
 *     gamma = B11 alpha_q + B12 beta_u
 *
 * with gamma the coefficient vector of u* in the degree-(k+1) nodal basis. B11
 * and B12 depend only on the cell and k, so they are assembled once here and
 * reused for every residual evaluation -- that is what keeps the interpolatory
 * approach's "assemble the matrices once" advantage intact.
 *
 * Two further per-cell matrices support the superconvergent residual:
 *
 *     V  = [ phi_j(x_m)      ]   (k+2)x(k+1)  evaluate a P_k field at the
 *                                             k+2 nodes of the star basis
 *     A9 = [ ( chi_m, phi_i )_K ] (k+1)x(k+2) project a P_{k+1} interpolant
 *                                             onto the P_k test space
 *
 * A9 is the paper's matrix of the same name; it replaces the mass matrix that
 * Basis::InterpolateOntoBasis applies when the interpolation is into P_k.
 *
 * Because the star basis is nodal at its own k+2 points, u*(x_m) = gamma_m --
 * no evaluation is needed to get u* where the physics is evaluated.
 */
class Postprocessor
{
public:
    Postprocessor(Grid const &grid, unsigned int k, Index nVars, Index nScalars = 0,
                  Index nAux = 0);

    Postprocessor(const Postprocessor &) = delete;
    Postprocessor &operator=(const Postprocessor &) = delete;

    // Number of degrees of freedom per cell in the star (degree k+1) space.
    Index starDoF() const { return k + 2; }

    // The k+2 nodes of the star basis in every cell, in the same
    // cell-major order as DGSoln::getPoints().
    std::vector<Position> const &starPoints() const { return starPoints_; }

    // Reconstruct u* for every variable from Y. Must be called before uStar()
    // or evalOnStarNodes().
    void computeUStar(DGSoln const &Y);

    // u* as a degree-(k+1) DGApprox, so it can be evaluated at any x and handed
    // straight to NetCDFIO::AddVariable / AppendToGroup.
    DGSoln::DGApprox const &uStar(Index var) const { return uStar_[var]; }

    // The state to hand to TransportSystem::ComputePhysics with the
    // superconvergent scheme: u is u* at the star nodes, while q, sigma and the
    // auxiliary variables are the solver's own degree-k fields evaluated there.
    // Uses the u* from the last computeUStar() call.
    GlobalState evalOnStarNodes(DGSoln const &Y) const;

    // Per-cell operators. See the class comment for shapes.
    Matrix const &B11(Index cell) const { return B11_[cell]; }
    Matrix const &B12(Index cell) const { return B12_[cell]; }
    Matrix const &V(Index cell) const { return V_[cell]; }
    Matrix const &A9(Index cell) const { return A9_[cell]; }

    // ( chi_m, 1 )_K -- the star basis's integration weights, so that b . g is the
    // integral of g's P_{k+1} interpolant over the cell. Used by the adjoint for
    // both G and dG/dy, where the weight is 1 rather than a test function.
    Vector const &starWeights(Index cell) const { return b1_[cell]; }

    NodalBasis const &getStarBasis() const { return starBasis; }

private:
    Grid const &grid;
    unsigned int k;
    Index nVars, nScalars, nAux;

    // Held by value, as DGSolnImpl does: DGApproxImpl keeps a reference to the
    // basis it was constructed with, so a temporary from NodalBasis::getBasis
    // would dangle.
    const NodalBasis basis;     // degree k, the solution's own basis
    const NodalBasis starBasis; // degree k+1

    std::vector<Matrix> B11_, B12_, V_, A9_;
    std::vector<Vector> b1_;

    std::vector<Position> starPoints_;

    // Backing store for uStar_, laid out per cell as [ var0 | var1 | ... ]
    // with (k+2) coefficients each, so the stride is nVars*(k+2).
    std::vector<double> starMem_;
    std::vector<DGSoln::DGApprox> uStar_;
};

#endif // POSTPROCESSING_HPP
