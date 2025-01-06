#ifndef BASIS_HPP
#define BASIS_HPP

#include "Types.hpp"
#include <boost/math/special_functions/legendre.hpp>

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

        static double Prime(Interval const &I, Index i, double x)
        {
            if (i == 0)
                return 0.0;

            double y = I.toRef(x);

            if (y == 1.0)
                return i * (i + 1.0) / 2.0;
            if (y == -1.0)
                return (i % 2 == 0 ? i * (i + 1.0) / 2.0 : -i * (i + 1.0) / 2.0);

            return ::sqrt((2 * i + 1) / (I.h())) * (2 * i / I.h()) * (1.0 / (y * y - 1.0)) * (y * std::legendre(i, y) - std::legendre(i - 1, y));
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
public:
	NodalBasis(){};
	~NodalBasis(){};

	static double Evaluate(Interval const &I, Index i, double x)
	{
		return ::sqrt((2 * i + 1) / (I.h())) * std::legendre(i, 2 * (x - I.x_l) / I.h() - 1.0);
	};

	static double Prime(Interval const &I, Index i, double x)
	{
		if (i == 0)
			return 0.0;

		double y = 2 * (x - I.x_l) / I.h() - 1.0;

		if (y == 1.0)
			return i * (i + 1.0) / 2.0;
		if (y == -1.0)
			return (i % 2 == 0 ? i * (i + 1.0) / 2.0 : -i * (i + 1.0) / 2.0);

		return ::sqrt((2 * i + 1) / (I.h())) * (2 * i / I.h()) * (1.0 / (y * y - 1.0)) * (y * std::legendre(i, y) - std::legendre(i - 1, y));
	};

	static double Evaluate(Interval const &I, const VectorRef &vCoeffs, double x)
	{
		double result = 0.0;
		for (Index i = 0; i < vCoeffs.size(); ++i)
			result += vCoeffs(i) * Evaluate(I, i, x);
		return result;
	};

	static std::function<double(double)> phi(Interval const &I, Index i)
	{
		return [=](double x)
		{
			return ::sqrt((2 * i + 1) / (I.h())) * std::legendre(i, 2 * (x - I.x_l) / I.h() - 1.0);
		};
	}

	static std::function<double(double)> phiPrime(Interval const &I, Index i)
	{
		if (i == 0)
			return [](double)
			{ return 0.0; };

		return [=](double x)
		{
			double y = 2 * (x - I.x_l) / I.h() - 1.0;

			if (y == 1.0)
				return i * (i + 1.0) / 2.0;
			if (y == -1.0)
				return (i % 2 == 0 ? i * (i + 1.0) / 2.0 : -i * (i + 1.0) / 2.0);

			return ::sqrt((2 * i + 1) / (I.h())) * (2 * i / I.h()) * (1.0 / (y * y - 1.0)) * (y * std::legendre(i, y) - std::legendre(i - 1, y));
		};
	}
};

#endif
