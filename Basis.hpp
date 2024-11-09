#ifndef BASIS_HPP
#define BASIS_HPP

#include "Types.hpp"
#include <boost/math/special_functions/legendre.hpp>

class Interval;

class LegendreBasis
{
public:
	LegendreBasis(){};
	~LegendreBasis(){};

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
