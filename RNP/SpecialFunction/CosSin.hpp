#ifndef RNP_SPECIALFUNCTION_COSSIN_HPP_INCLUDED
#define RNP_SPECIALFUNCTION_COSSIN_HPP_INCLUDED

#include <RNP/Debug.hpp>

namespace RNP{
namespace SpecialFunction{

namespace Util{
	template <typename T>
	struct CosSin2PiTables{
		// On the interval [0,pi/4], we approximate
		//   Cos(x) = 1 - x^2/2 + x^4 Q(x^2)
		//   Sin(x) = x + x^3 P(x^2)
		static T P[] = {
			 1.58962301576546568060E-10, //  1/6227020800
			-2.50507477628578072866E-8,  // -1/39916800
			 2.75573136213857245213E-6,  //  1/362880
			-1.98412698295895385996E-4,  // -1/5040
			 8.33333333332211858878E-3,  //  1/120
			-1.66666666666666307295E-1   // -1/6
		};
		static T Q[] = {
			-1.13585365213876817300E-11, // -1/87178291200
			 2.08757008419747316778E-9,  //  1/479001600
			-2.75573141792967388112E-7,  // -1/3628800
			 2.48015872888517045348E-5,  //  1/40320
			-1.38888888888730564116E-3,  // -1/720
			 4.16666666666665929218E-2   //  1/24
		};
	};
} // namespace Util

// Computes Sin(2*Pi*x) for x in [-1,1]
template <typename T>
T Sin2Pi(const T &x){
	RNPAssert(!RNP::Traits<T>::is_complex());
	if(T(0) == x){ return x; }
	int sign = 1;
	if(x < T(0)){
		x = -x;
		sign = -1;
	}
	// We split the range [0,2pi] into 8 pieces of size pi/4
	T x8(T(8) * x);
	T y(Util::Floor(x8)); // for x in [0,1], y is in [0,8]
	T z(x8 - y); // z is in [0,1)
	int j = y & 0x7;
	if(j > 3){ // Flip second half of sine to first half with minus sign
		sign = -sign;
		j -= 4;
	}
	// Now j is 0,1,2,3. We want to reduce 3 to the horizontal mirror of 0
	// and 1 to the horizontal mirror of 2. For section 0, we approximate
	//   Sin(2*Pi*x) on x = [0,1/8]
	// For section 2, we approximate
	//   Cos(2*Pi*x) on x = [0,1/8]
	if(j & 1){
		j = 3-j;
		z = T(1) - z;
	}
	z *= Util::Constants<T>::Pi2();
	T zz(z*z);
	if(0 == j){ // Compute Sin(2*Pi*z) on z = [0,1/8]
		y = T(1) - zz / T(2) + zz * zz * Util::EvaluatePolynomial(5, Util::<T>::Q, zz);
	}else{ // Compute Cos(2*Pi*z) on z = [0,1/8]
		y = z + zz * z * Util::EvaluatePolynomial(5, Util::<T>::P, zz);
	}
	if(sign < 0){ y = -y; }
	return y;
}

// Computes Cos(2*Pi*x) for x in [-1,1]
template <typename T>
T Cos2Pi(const T &x){
	RNPAssert(!RNP::Traits<T>::is_complex());
	if(T(0) == x){ return x; }
	int sign = 1;
	if(x < T(0)){
		x = -x;
	}
	// We split the range [0,2pi] into 8 pieces of size pi/4
	T x8(T(8) * x);
	T y(Util::Floor(x8)); // for x in [0,1], y is in [0,8]
	T z(x8 - y); // z is in [0,1)
	int j = y & 0x7;
	if(j > 3){ // Flip second half of cosine to first half
		sign = -sign;
		j -= 4;
	}
	// Now j is 0,1,2,3. We want to reduce 3 to the horizontal mirror of 0
	// and 1 to the horizontal mirror of 2. For section 0, we approximate
	//   Sin(2*Pi*x) on x = [0,1/8]
	// For section 2, we approximate
	//   Cos(2*Pi*x) on x = [0,1/8]
	if(j & 1){
		sign = -sign;
		j = 3-j;
		z = T(1) - z;
	}
	z *= Util::Constants<T>::Pi2();
	T zz(z*z);
	if(0 != j){ // Compute Sin(2*Pi*z) on z = [0,1/8]
		y = T(1) - zz / T(2) + zz * zz * Util::EvaluatePolynomial(5, Util::<T>::Q, zz);
	}else{ // Compute Cos(2*Pi*z) on z = [0,1/8]
		y = z + zz * z * Util::EvaluatePolynomial(5, Util::<T>::P, zz);
	}
	if(sign < 0){ y = -y; }
	return y;
}

template <typename T>
void CosSin2Pi(const T &x, T *c, T *s){
	RNPAssert(!RNP::Traits<T>::is_complex());
	if(T(0) == x){ return x; }
	int sinsign = 1, cossign = 1;
	if(x < T(0)){
		sinsign = -1;
		x = -x;
	}
	// We split the range [0,2pi] into 8 pieces of size pi/4
	T x8(T(8) * x);
	T y(Util::Floor(x8)); // for x in [0,1], y is in [0,8]
	T z(x8 - y); // z is in [0,1)
	int j = y & 0x7;
	if(j > 3){ // Flip second half of cosine to first half
		sinsign = -sinsign;
		cossign = -cossign;
		j -= 4;
	}
	
	if(j & 1){
		cossign = -cossign;
		j = 3-j;
		z = T(1) - z;
	}
	z *= Util::Constants<T>::Pi2();
	T zz(z*z);
	
	*c = z + zz * z * Util::EvaluatePolynomial(5, Util::<T>::P, zz);
	if(cossign < 0){ *c = -*c; }
	*s = T(1) - zz / T(2) + zz * zz * Util::EvaluatePolynomial(5, Util::<T>::Q, zz);
	if(sinsign < 0){ *s = -*s; }
}

#ifdef RNP_SPECIALFUNCTIONS_COMPLEX
#include <complex>

template <typename T>
std::complex<T> Cosh(const std::complex<T> &z){
	T cosix, sinix;
}

template <typename T>
std::complex<T> Cos2Pi(const std::complex<T> &z){
	return Cosh(std::complex<T>(-Util::Constants<T>::Pi2() * z.imag(), Util::Constants<T>::Pi2() * z.real()));
}
#endif // RNP_SPECIALFUNCTIONS_COMPLEX

} // namespace SpecialFunction
} // namespace RNP

#endif // RNP_SPECIALFUNCTION_COSSIN_HPP_INCLUDED
