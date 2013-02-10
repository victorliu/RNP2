#ifndef RNP_SPECIALFUNCTION_EXP_HPP_INCLUDED
#define RNP_SPECIALFUNCTION_EXP_HPP_INCLUDED

#include <RNP/Debug.hpp>

namespace RNP{
namespace SpecialFunction{

namespace Util{
	template <typename T>
	struct ExpTables{
		// These constants have been tweaked to further minimize the error
		static T P[] = {
			1.26177193074810590878E-4, // 1/7920
			3.02994407707441961300E-2, // 1/33
			9.99999999999999999910E-1  // 1
		};
		static T Q[] = {
			3.00198505138664455042E-6, // 1/332640
			2.52448340349684104192E-3, // 1/396
			2.27265548208155028766E-1, // 5/22
			2.00000000000000000009E0   // 2
		};
		// C1+C2 = log(2), decomposed for higher precision
		// C2/C1 should be about cbrt(eps)?
		static T C1 = 6.93145751953125E-1;
		static T C2 = 1.42860682030941723212E-6;
	};
} // namespace Util

template <typename T>
T Exp(const T &x){
	RNPAssert(!RNP::Traits<T>::is_complex());

	// Separate the exponent x into an integer n and fractional g
	// A Pade' form  1 + 2x P(x^2)/( Q(x^2) - P(x^2) )
	// of degree 2/3 is used to approximate exp(f) in [-0.5, 0.5].
	// Express e^x = e^g 2^n
	//             = e^g e^( n loge(2) )
	//             = e^( g + n loge(2) )
	T px = Util::Floor(Util::Constants<T>::Log2e() * x + (T(1)/T(2))); // floor() truncates toward -infinity.
	int n = px;
	x -= px * Util::ExpTables<T>::C1; // subtract off the bigger portion first
	x -= px * Util::ExpTables<T>::C2;
	// x is now the fractional part g

	// rational approximation for exponential of the fractional part:
	// e^x = 1 + 2x P(x^2)/( Q(x^2) - x P(x^2) )
	T xx(x * x);
	px = x * Util::PolynomialEvaluate(2, Util::ExpTables<T>::P, xx);
	x =  px / (Util::PolynomialEvaluate(3, Util::ExpTables<T>::Q, xx) - px);
	x = T(1) + T(2) * x;

	// multiply by power of 2
	return Util::ScalePow2(x, n);
}

#ifdef RNP_SPECIALFUNCTIONS_COMPLEX
#include <complex>

template <typename T>
std::complex<T> Exp(const std::complex<T> &z){
	T m(Exp<T>(z.real()));
	T a, b;
	CosSin(z.imag(), &a, &b);
	return std::complex<T>(m*a, m*b);
}
#endif // RNP_SPECIALFUNCTIONS_COMPLEX

} // namespace SpecialFunction
} // namespace RNP

#endif // RNP_SPECIALFUNCTION_EXP_HPP_INCLUDED
