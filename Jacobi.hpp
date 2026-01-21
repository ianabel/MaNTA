#ifndef JACOBI_HPP
#define JACOBI_HPP

/*
 * routines for Jacobi Polynomials
 * Calculate Gauss Quadratures for Jacobi Polynomials
 */

#include <vector>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <boost/math/special_functions/jacobi.hpp>

using Eigen::Index;

class Jacobi {
  private:
    double alpha, beta;
    
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

      if ( N == 0 ) {
          Eigen::VectorXd empty( 0 );
          return std::make_pair( empty, empty );
      }

      Eigen::MatrixXd T(N,N);

      T.setZero();

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

    double operator()(unsigned int n, double x) {
        return boost::math::jacobi<double>( n, alpha, beta, x );
    };

    static double operator()(unsigned int n, double alpha, double beta, double x) {
        return boost::math::jacobi<double>( n, alpha, beta, x );
    };

    double prime(unsigned int n, double x) {
        return boost::math::jacobi_prime<double>( n, alpha, beta, x );
    };

    static double prime(unsigned int n, double alpha, double beta, double x) {
        return boost::math::jacobi_prime<double>( n, alpha, beta, x );
    };
};

#endif
