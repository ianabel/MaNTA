/*
 * routines for Jacobi Polynomials
 * Calculate Gauss Quadratures for Jacobi Polynomials
 */

#include <vector>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using Eigen::Index;

class Jacobi {
  private:
    double alpha, beta;
    double a( Index n ) {
      double t = 2.0 * n + alpha + beta;
      return t*(t-1.0) / (2.0 * n * (t - n) );
    };
    double b( Index n ) {
      double t = 2.0 * n + alpha + beta;
      return (alpha - beta)*(t-1.0)*(t-2.0*n) / (2.0*n * (t - n) * ( t - 2.0*n ) );
    };
    double c( Index n ) {
      double t = 2.0 * n + alpha + beta;
      return ( n - 1.0 + alpha ) * ( n - 1.0 + beta ) * t / ((t - n) * ( t - 2.0 ) * n);
    };
    double alpha_n( Index n ) {
      double t = 2.0 * n + alpha + beta;
      return (beta - alpha)/t;
    };
    double beta_n( Index n ) {
      double t = 2.0 * n + alpha + beta;
      // c(n+1)/(a(n)a(n+1)) 
      // c(n+1) = (n + alpha) * ( n + beta ) * ( t + 2 ) / ( n + 1 + alpha + beta ) * ( t ) * (n + 1)
      // a(n)*a(n+1) = t*(t-1)*(t+2)*(t+1) / (4.0 * n * (n + 1) * (t-n) * ( n + 1 + alpha + beta )
      // beta_n^2 = (n + alpha) * ( n + beta ) * 4 * n * (t - n) / (t-1)*t*t*(t+1)
      return std::sqrt( (n + alpha) * ( n + beta ) * 4 * n * (t - n) / ((t-1)*t*t*(t+1)) );
    }

    double mu0() {
      return 2.0;
    };
  public:
    Jacobi( double alpha_in, double beta_in ) : alpha(alpha_in),beta(beta_in) {
    };
    virtual ~Jacobi() {};

    std::pair< Eigen::VectorXd, Eigen::VectorXd > GaussQuadrature( unsigned int N ) {
      Eigen::MatrixXd T(N,N);
      for( Index i = 1; i <= N; i++ ) {
            T(i-1,i-1) = alpha_n(i);
            if( i < N ) {
              T( i - 1 , i ) = beta_n(i);
              T( i , i - 1 ) = beta_n(i);
            }
      }
      Eigen::EigenSolver<Eigen::MatrixXd> eigs(T);
      Eigen::VectorXd abscissae(N);
      Eigen::VectorXd weights(N);
      Eigen::VectorXcd lambdas = eigs.eigenvalues();
      auto evs = eigs.eigenvectors();
      for( Index i = 0; i < N; ++i ) {
        abscissae[i] = lambdas[i].real();
        double q1j = ((evs.col(i))(0)).real();
        weights[i] = mu0() * q1j * q1j;
      }
      return std::make_pair( abscissae, weights );
    };
};


