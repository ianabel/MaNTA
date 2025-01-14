#ifndef BASIS_HPP
#define BASIS_HPP

#include <algorithm>
#include <utility>

#include "Types.hpp"
#include "Jacobi.hpp"
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/binomial.hpp>

class Interval;

class LegendreBasis
{
    private:
        unsigned int k;
        LegendreBasis( unsigned int  Order ) : k(Order) {};
        static std::map<unsigned int,LegendreBasis> singletons;
    public:
        ~LegendreBasis(){};

        unsigned int Order() const { return k; };

        static LegendreBasis getBasis( unsigned int k ) {
            if( singletons.contains( k ) )
                return singletons.at( k );
            else
            {
                singletons.insert( { k, LegendreBasis( k ) } );
                return singletons.at(k);
            }
        }

        static double Evaluate(Interval const &I, Index i, double x)
        {
            return ::sqrt((2 * i + 1) / (I.h())) * std::legendre(i, I.toRef(x) );
        };

        // Eval on ref interval
        static double Evaluate(Index i, double x)
        {
            return ::sqrt((2 * i + 1) / (2.0)) * std::legendre(i, x );
        };

        static double Prime(Index i, double x)
        {
            if (i == 0)
                return 0.0;

            if (x == 1.0)
                return ( ( i + 1 ) / 2.0 ) * ::sqrt( 0.5 + i ) * ( i );
            if (x == -1.0)
                return ::pow( -1, i-1 ) * ( ( i + 1 ) / 2.0 ) * ::sqrt( 0.5 + i ) * ( i );

            return ::sqrt(0.5 + i ) * ((1.0 + i) / (1.0-x*x)) * ( x * std::legendre(i, x) - std::legendre(i + 1, x));
        };


        static double Prime(Interval const &I, Index i, double x)
        {
            if (i == 0)
                return 0.0;

            double y = I.toRef(x);

            return ::sqrt( 2.0 / I.h() ) * (2.0 / I.h() ) * Prime( i, y );
        };

        static double Evaluate(Interval const &I, const VectorRef &vCoeffs, double x)
        {
            double result = 0.0;
            for (Index i = 0; i < vCoeffs.size(); ++i)
                result += vCoeffs(i) * Evaluate(I, i, x);
            return result;
        };

        static const std::array<double,15>& abscissae() { return integrator.abscissa(); };
        static const std::array<double,15>& weights() { return integrator.weights(); };
        using IntegratorType = boost::math::quadrature::gauss<double, 30>;
        static IntegratorType integrator;

        double CellProduct(Interval const &I, std::function<double(double)> f, std::function<double(double)> g) const
        {
            auto u = [&](double x)
            { return f(x) * g(x); };
            return integrator.integrate(u, I.x_l, I.x_u);
        };

        Vector ProjectOntoBasis( Interval const& I, std::function<double(double)> f ) const
        {
            Vector out( k + 1 );
            for( Index i = 0; i < k + 1; i++ )
            {
                auto u = [&,this](double x) { return f(x) * Evaluate(I, i, x); };
                out( i ) = integrator.integrate(u, I.x_l, I.x_u);
            }
            return out;
        };

        double EdgeProduct(Interval const &I, std::function<double(double)> f, std::function<double(double)> g) const
        {
            return f(I.x_l) * g(I.x_l) + f(I.x_u) * g(I.x_u);
        };

        void MassMatrix(Interval const &I, MatrixRef u) const
        {
            u.setIdentity( k + 1, k + 1 );
        };

        void MassMatrix(Interval const &I, MatrixRef u, std::function<double(double)> const &w) const
        {
            for (Index i = 0; i < u.rows(); i++)
                for (Index j = 0; j < u.cols(); j++)
                {
                    auto F = [&](double x)
                    { return w(x) * Evaluate(I, i, x) * Evaluate(I, j, x); };
                    u(i, j) = integrator.integrate(F, I.x_l, I.x_u);
                }
        };

        void MassMatrix(Interval const &I, MatrixRef u, std::function<double(double, int)> const &w, int var) const
        {
            for (Index i = 0; i < u.rows(); i++)
                for (Index j = 0; j < u.cols(); j++)
                {
                    auto F = [&](double x)
                    { return w(x, var) * Evaluate(I, i, x) * Evaluate(I, j, x); };
                    u(i, j) = integrator.integrate(F, I.x_l, I.x_u);
                }
        };

        Matrix MassMatrix(Interval const &I) const
        {
            return Matrix::Identity(k + 1, k + 1);
        }

        Matrix MassMatrix(Interval const &I, std::function<double(double)> const &w) const
        {
            Matrix u(k + 1, k + 1);
            for (Index i = 0; i < u.rows(); i++)
                for (Index j = 0; j < u.cols(); j++)
                {
                    auto F = [&](double x)
                    { return w(x) * Evaluate(I, i, x) * Evaluate(I, j, x); };
                    u(i, j) = integrator.integrate(F, I.x_l, I.x_u);
                }
            return u;
        }


        void DerivativeMatrix(Interval const &I, MatrixRef D) const
        {
            for (Index i = 0; i < D.rows(); i++)
                for (Index j = 0; j < D.cols(); j++)
                {
                    auto F = [&](double x)
                    { return Evaluate(I, i, x) * Prime(I, j, x); };
                    D(i, j) = integrator.integrate(F, I.x_l, I.x_u);
                }
        }

        Matrix DerivativeMatrix(Interval const &I) const
        {
            Matrix D( k + 1, k + 1 );
            DerivativeMatrix( I, D );
            return D;
        }

        void DerivativeMatrix(Interval const &I, MatrixRef D, std::function<double(double)> const &w) const
        {
            for (Index i = 0; i < D.rows(); i++)
                for (Index j = 0; j < D.cols(); j++)
                {
                    auto F = [&](double x)
                    { return w(x) * Evaluate(I, i, x) * Prime(I, j, x); };
                    D(i, j) = integrator.integrate(F, I.x_l, I.x_u);
                }
        }
};

class ChebyshevBasis
{
    private:
        unsigned int k;
        Matrix RefMass,RefDerivative; // Matrices for the reference interval
        ChebyshevBasis( unsigned int  Order ) : k(Order)
        {
            RefMass.resize( k + 1, k + 1 );
            for (Index i = 0; i < k + 1; i++)
                for (Index j = 0; j < k + 1; j++)
                {
                    auto F = [&](double x) { return Tn(i, x) * Tn(j, x); };
                    RefMass(i, j) = integrator.integrate(F, -1, 1);
                }
        };
        static std::map<unsigned int,ChebyshevBasis> singletons;
    public:

        static double Tn( unsigned int n, double x ) { return std::cos( n * std::acos( x ) ); };
        static double Un( unsigned int n, double x ) {
            double theta = std::acos( x );
            if( theta == 0 )
                return n + 1.0;
            if( theta == pi )
                return ( n % 2 == 0 ) ? ( n + 1.0 ) : -( n + 1.0 );
            return std::sin( (n + 1) * theta )/std::sin( theta );
        };

        ~ChebyshevBasis(){};

        unsigned int Order() const { return k; };

        static ChebyshevBasis getBasis( unsigned int k ) {
            if( singletons.contains( k ) )
                return singletons.at( k );
            else
            {
                singletons.insert( { k, ChebyshevBasis( k ) } );
                return singletons.at(k);
            }
        }

        static double Evaluate(Interval const &I, Index i, double x)
        {
            return Tn(i, I.toRef(x) );
        };

        static double Prime(Interval const &I, Index i, double x)
        {
            if (i == 0)
                return 0.0;

            double y = I.toRef(x);

            return (2.0 / I.h()) * i * Un( i - 1, y );
        };

        static double Evaluate(Interval const &I, const VectorRef &vCoeffs, double x)
        {
            double result = 0.0;
            for (Index i = 0; i < vCoeffs.size(); ++i)
                result += vCoeffs(i) * Evaluate(I, i, x);
            return result;
        };

        static const std::array<double,15>& abscissae() { return integrator.abscissa(); };
        static const std::array<double,15>& weights() { return integrator.weights(); };
        using IntegratorType = boost::math::quadrature::gauss<double, 30>;
        static IntegratorType integrator;

        double CellProduct(Interval const &I, std::function<double(double)> f, std::function<double(double)> g) const
        {
            auto u = [&](double x)
            { return f(x) * g(x); };
            return integrator.integrate(u, I.x_l, I.x_u);
        };

        Vector ProjectOntoBasis( Interval const& I, std::function<double(double)> f ) const
        {
            Vector out( k + 1 );
            for( Index i = 0; i < k + 1; i++ )
            {
                auto u = [&,this](double x) { return f(x) * Evaluate(I, i, x); };
                out( i ) = integrator.integrate(u, I.x_l, I.x_u);
            }
            return out;
        };

        double EdgeProduct(Interval const &I, std::function<double(double)> f, std::function<double(double)> g) const
        {
            return f(I.x_l) * g(I.x_l) + f(I.x_u) * g(I.x_u);
        };

        void MassMatrix(Interval const &I, MatrixRef u) const
        {
            u = (I.h()/2.0) * RefMass;
        };

        void MassMatrix(Interval const &I, MatrixRef u, std::function<double(double)> const &w) const
        {
            for (Index i = 0; i < u.rows(); i++)
                for (Index j = 0; j < u.cols(); j++)
                {
                    auto F = [&](double x)
                    { return w(x) * Evaluate(I, i, x) * Evaluate(I, j, x); };
                    u(i, j) = integrator.integrate(F, I.x_l, I.x_u);
                }
        };

        void MassMatrix(Interval const &I, MatrixRef u, std::function<double(double, int)> const &w, int var) const
        {
            for (Index i = 0; i < u.rows(); i++)
                for (Index j = 0; j < u.cols(); j++)
                {
                    auto F = [&](double x)
                    { return w(x, var) * Evaluate(I, i, x) * Evaluate(I, j, x); };
                    u(i, j) = integrator.integrate(F, I.x_l, I.x_u);
                }
        };

        Matrix MassMatrix(Interval const &I) const
        {
            return (I.h()/2.0) * RefMass;
        }

        Matrix MassMatrix(Interval const &I, std::function<double(double)> const &w) const
        {
            Matrix u(k + 1, k + 1);
            for (Index i = 0; i < u.rows(); i++)
                for (Index j = 0; j < u.cols(); j++)
                {
                    auto F = [&](double x)
                    { return w(x) * Evaluate(I, i, x) * Evaluate(I, j, x); };
                    u(i, j) = integrator.integrate(F, I.x_l, I.x_u);
                }
            return u;
        }

        void DerivativeMatrix(Interval const &I, MatrixRef D) const
        {
            for (Index i = 0; i < D.rows(); i++)
                for (Index j = 0; j < D.cols(); j++)
                {
                    auto F = [&](double x)
                    { return Evaluate(I, i, x) * Prime(I, j, x); };
                    D(i, j) = integrator.integrate(F, I.x_l, I.x_u);
                }
        }

        Matrix DerivativeMatrix(Interval const &I) const
        {
            Matrix D( k + 1, k + 1 );
            DerivativeMatrix( I, D );
            return D;
        }

        void DerivativeMatrix(Interval const &I, MatrixRef D, std::function<double(double)> const &w) const
        {
            for (Index i = 0; i < D.rows(); i++)
                for (Index j = 0; j < D.cols(); j++)
                {
                    auto F = [&](double x)
                    { return w(x) * Evaluate(I, i, x) * Prime(I, j, x); };
                    D(i, j) = integrator.integrate(F, I.x_l, I.x_u);
                }
        }
};

// Uses LGL points
class NodalBasis
{
    private:
        unsigned int k;
        Matrix Vandermonde,Vr,RefMass,RefDerivative; // Matrices for the reference interval
        Vector LGLNodes;
        Vector BarycentricWeights;

        NodalBasis( unsigned int  Order ) : k(Order)
        {
            if( Order == 0 ) {
                LGLNodes.resize( 1 );
                LGLNodes( 0 ) = 0.0;
                RefMass.resize( 1, 1 );
                RefMass( 0, 0 ) = 1.0;
            }

            LGLNodes.resize( k + 1 );
            Jacobi jac( 1, 1 );
            Eigen::VectorXd nodes( k - 1 );
            std::tie( nodes, std::ignore ) = jac.GaussQuadrature( k - 1 );
            std::sort( nodes.begin(), nodes.end() );
            LGLNodes.segment( 1, k - 1 ) = nodes;
            LGLNodes( 0 ) = -1.0;
            LGLNodes( k ) =  1.0;

            // LGLNodes now contains the Legendre-Gauss-Lobatto nodes on [-1,1]

            Vandermonde.resize( k + 1, k + 1 );
            Vr.resize( k + 1, k + 1 );

            for( Index i = 0; i < k + 1; ++i ) {
                for( Index j = 0; j < k + 1; ++j ) {
                    Vandermonde( i, j ) = LegendreBasis::Evaluate( j, LGLNodes( i ) );
                    Vr( i, j ) = LegendreBasis::Prime( j, LGLNodes( i ) );
                }
            }

            RefMass = ( Vandermonde * Vandermonde.transpose() ).inverse();

            // Our 'derivative' matrix is the stiffness matrix ( phi_i, phi_j' )
            // So we can use  S = (M Dr) & Dr = Vr * V^-1 from Hesthaven & Warburton p.52
            RefDerivative = RefMass * ( Vr * Vandermonde.inverse() );

            BarycentricWeights.resize( k + 1 );

            for( Index i = 0; i < k + 1; ++i ) {
                BarycentricWeights( i ) = 1.0;
                for( Index j = 0; j < k + 1; ++j ) {
                    if( j == i )
                        continue;
                    BarycentricWeights( i ) *= 1.0/(LGLNodes(j) - LGLNodes(i));
                }
            }

        };
        static std::map<unsigned int,NodalBasis> singletons;
    public:
        ~NodalBasis(){};

        unsigned int Order() const { return k; };

        static NodalBasis getBasis( unsigned int k ) {
            if( singletons.contains( k ) )
                return singletons.at( k );
            else
            {
                singletons.insert( { k, NodalBasis( k ) } );
                return singletons.at(k);
            }
        }

        double Evaluate(Index i, double x) const
        {
            if( x == LGLNodes( i ) )
                return 1.0;

            double numerator = BarycentricWeights( i ) / ( x - LGLNodes( i ) );
            double denominator = 0.0;

            for( Index j = 0; j < k + 1; ++j ) {

                double y = ( x - LGLNodes( j ) );

                if( y == 0.0 )
                    return 0.0;
                denominator += BarycentricWeights( j ) / y;
            }

            return numerator/denominator;
        };


        double Evaluate(Interval const &I, Index i, double X) const
        {
            double x = I.toRef(X);
            return Evaluate( i, x );
        };

        double Evaluate(Interval const &I, const VectorRef &vCoeffs, double x) const
        {
            double result = 0.0;
            for (Index i = 0; i < vCoeffs.size(); ++i)
                result += vCoeffs(i) * Evaluate(I, i, x);
            return result;
        };

        // l_i'(x) = l_i(x) * Sum_(m != i) (x - x_m)^-1
        // on the ref interval
        double Prime(Index i, double x) const
        {
            double l = Evaluate(i,x);
            double sum = 0;
            for( Index j = 0; j < k + 1; ++j ) {
                double y = (x - LGLNodes(j));
                if( y == 0 ) {
                    return ( BarycentricWeights( i )/BarycentricWeights( j ) )/( LGLNodes( j ) - LGLNodes( i ) );
                }
                sum += 1.0/y;
            }

            return l * sum;
        };

        double Prime(Interval const &I, Index i, double X) const
        {
            double x = I.toRef(X);
            return (2.0/I.h())*Prime(i,x);
        };

        static const std::array<double,15>& abscissae() { return integrator.abscissa(); };
        static const std::array<double,15>& weights() { return integrator.weights(); };
        using IntegratorType = boost::math::quadrature::gauss<double, 30>;
        static IntegratorType integrator;

        double CellProduct(Interval const &I, std::function<double(double)> f, std::function<double(double)> g) const
        {
            auto u = [&](double x)
            { return f(x) * g(x); };
            return integrator.integrate(u, I.x_l, I.x_u);
        };

        Vector ProjectOntoBasis( Interval const& I, std::function<double(double)> f ) const
        {
            Vector out( k + 1 );
            Vector vals( k + 1 );
            for( Index i = 0; i < k + 1; i++ )
                vals( i ) = f( I.fromRef( LGLNodes( i ) ) );
            out = (I.h()/2.0) * RefMass * vals;
            return out;
        };

        double EdgeProduct(Interval const &I, std::function<double(double)> f, std::function<double(double)> g) const
        {
            return f(I.x_l) * g(I.x_l) + f(I.x_u) * g(I.x_u);
        };

        void MassMatrix(Interval const &I, MatrixRef u) const
        {
            u = (I.h()/2.0) * RefMass;
        };

        void MassMatrix(Interval const &I, MatrixRef u, std::function<double(double)> const &w) const
        {
            for (Index i = 0; i < u.rows(); i++)
                for (Index j = 0; j < u.cols(); j++)
                {
                    auto F = [&](double x)
                    { return w(x) * Evaluate(I, i, x) * Evaluate(I, j, x); };
                    u(i, j) = integrator.integrate(F, I.x_l, I.x_u);
                }
        };

        void MassMatrix(Interval const &I, MatrixRef u, std::function<double(double, int)> const &w, int var) const
        {
            for (Index i = 0; i < u.rows(); i++)
                for (Index j = 0; j < u.cols(); j++)
                {
                    auto F = [&](double x)
                    { return w(x, var) * Evaluate(I, i, x) * Evaluate(I, j, x); };
                    u(i, j) = integrator.integrate(F, I.x_l, I.x_u);
                }
        };

        Matrix MassMatrix(Interval const &I) const
        {
            return (I.h()/2.0) * RefMass;
        }

        Matrix MassMatrix(Interval const &I, std::function<double(double)> const &w) const
        {
            Matrix u(k + 1, k + 1);
            for (Index i = 0; i < u.rows(); i++)
                for (Index j = 0; j < u.cols(); j++)
                {
                    auto F = [&](double x)
                    { return w(x) * Evaluate(I, i, x) * Evaluate(I, j, x); };
                    u(i, j) = integrator.integrate(F, I.x_l, I.x_u);
                }
            return u;
        }

        void DerivativeMatrix(Interval const &, MatrixRef D) const
        {
            // No rescaling, the stiffness matrix is invariant across intervals;
            D = RefDerivative;
        }

        Matrix DerivativeMatrix(Interval const &I) const
        {
            Matrix D( k + 1, k + 1 );
            DerivativeMatrix( I, D );
            return D;
        }

        void DerivativeMatrix(Interval const &I, MatrixRef D, std::function<double(double)> const &w) const
        {
            for (Index i = 0; i < D.rows(); i++)
                for (Index j = 0; j < D.cols(); j++)
                {
                    auto F = [&](double x)
                    { return w(x) * Evaluate(I, i, x) * Prime(I, j, x); };
                    D(i, j) = integrator.integrate(F, I.x_l, I.x_u);
                }
        }
};

#endif
