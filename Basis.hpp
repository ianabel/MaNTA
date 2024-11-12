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

	static std::function<double(double)> phi(Interval const &I, Index i)
	{
		return [=](double x)
		{
			return ::sqrt((2 * i + 1) / (I.h())) * std::legendre(i, I.toRef(x));
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


    static const std::array<double,15>& abscissae() { return integrator.abscissa(); };
    static const std::array<double,15>& weights() { return integrator.weights(); };
    using IntegratorType = boost::math::quadrature::gauss<double, 30>;
    static IntegratorType integrator;

    /*
    Vector Project( Interval const& I, std::function<double(double)> f )
    {
        Vector out( k+1 );
        for( unsigned int i = 0; i < k + 1; ++i )
        {

    }
    */
};

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
