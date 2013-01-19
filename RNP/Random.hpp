#ifndef RNP_RANDOM_HPP_INCLUDED
#define RNP_RANDOM_HPP_INCLUDED

///////////////////////////////////////////////////////////////////////
// Random
// ======
// Random number generation routines. These are not cryptographically
// random and are intended for numerical purposes only.
//
#include <complex>
#include <cmath>
#include <RNP/Types.hpp>

namespace RNP{
namespace Random{

///////////////////////////////////////////////////////////////////////
// UniformRealVector
// -----------------
// Returns a vector of n random real numbers from a uniform
// distribution in [0,1).
//
// This routine uses a multiplicative congruential method with modulus
// 2^48 and multiplier 33952834046453. For reference, see:
//
// > G.S.Fishman, 'Multiplicative congruential random number
// > generators with modulus 2^b: an exhaustive analysis for
// > b = 32 and a partial analysis for b = 48',
// > Math. Comp. 189, pp 331-344, 1990).
//
// 48-bit integers are stored in 4 integer array elements with 12 bits
// per element. Hence the routine is portable across machines with
// integers of 32 bits or more.
//
// This corresponds approximately to the Lapack routines _laruv.
//
// Arguments
// iseed   On entry, the seed of the random number generator; the array
//         elements must be between 0 and 4095, and iseed[3] must be
//         odd. On exit, the seed is updated.
//         If iseed is NULL, then an internal seed is used.
// n       The number of random numbers to be generated.
// x       Output vector of the generated random numbers of length n.
//
template <typename T> // T must be a real number type
void UniformRealVector(size_t n, T *x, int iseed[4] = NULL){
	// This table can be generated in Mathematica by
	// SplitNum[n_] := Module[{a, b, bb, c, cc, d},
	//   d = Mod[n, 4096];
	//   cc = (n - d)/4096;
	//   c = Mod[cc, 4096];
	//   bb = (cc - c)/4096;
	//   b = Mod[bb, 4096];
	//   a = Mod[(bb - b)/4096, 4096];
	//   {a, b, c, d}
	// ];
	// Table[SplitNum[Mod[33952834046453^i, 2^48]], {i, 128}]
    static const int mm[128][4] = {
		{ 494, 322,2508,2549}, {2637, 789,3754,1145},
		{ 255,1440,1766,2253}, {2008, 752,3572, 305},
		{1253,2859,2893,3301}, {3344, 123, 307,1065},
		{4084,1848,1297,3133}, {1739, 643,3966,2913},
		{3143,2405, 758,3285}, {3468,2638,2598,1241},
		{ 688,2344,3406,1197}, {1657,  46,2922,3729},
		{1238,3814,1038,2501}, {3166, 913,2934,1673},
		{1292,3649,2091, 541}, {3422, 339,2451,2753},
		{1270,3808,1580, 949}, {2016, 822,1958,2361},
		{ 154,2832,2055,1165}, {2862,3078,1507,4081},
		{ 697,3633,1078,2725}, {1706,2970,3273,3305},
		{ 491, 637,  17,3069}, { 931,2249, 854,3617},
		{1444,2081,2916,3733}, { 444,4019,3971, 409},
		{3577,1478,2889,2157}, {3944, 242,3831,1361},
		{2184, 481,2621,3973}, {1661,2075,1541,1865},
		{3482,4058, 893,2525}, { 657, 622, 736,1409},
		{3023,3376,3992,3445}, {3618, 812, 787,3577},
		{1267, 234,2125,  77}, {1828, 641,2364,3761},
		{ 164,4005,2460,2149}, {3798,1122, 257,1449},
		{3087,3135,1574,3005}, {2400,2640,3912, 225},
		{2870,2302,1216,  85}, {3876,  40,3248,3673},
		{1905,1832,3401,3117}, {1593,2247,2124,3089},
		{1797,2034,2762,1349}, {1234,2637, 149,2057},
		{3460,1287,2245, 413}, { 328,1691, 166,  65},
		{2861, 496, 466,1845}, {1950,1597,4018, 697},
		{ 617,2394,1399,3085}, {2070,2584, 190,3441},
		{3331,1843,2879,1573}, { 769, 336, 153,3689},
		{1558,1472,2320,2941}, {2412,2407,  18, 929},
		{2800, 433, 712, 533}, { 189,2096,2159,2841},
		{ 287,1761,2318,4077}, {2045,2810,2091, 721},
		{1227, 566,3443,2821}, {2838, 442,1510,2249},
		{ 209,  41, 449,2397}, {2770,1238,1956,2817},
		{3654,1086,2201, 245}, {3993, 603,3137,1913},
		{ 192, 840,3399,1997}, {2253,3168,1321,3121},
		{3491,1499,2271, 997}, {2889,1084,3667,1833},
		{2857,3438,2703,2877}, {2094,2408, 629,1633},
		{1818,1589,2365, 981}, { 688,2391,2431,2009},
		{1407, 288,1113, 941}, { 634,  26,3922,2449},
		{3231, 512,2554, 197}, { 815,1456, 184,2441},
		{3524, 171,2099, 285}, {1914,1677,3228,1473},
		{ 516,2657,4012,2741}, { 164,2270,1921,3129},
		{ 303,2587,3452, 909}, {2144,2961,3901,2801},
		{3480,1970, 572, 421}, { 119,1817,3309,4073},
		{3357, 676,3171,2813}, { 837,1410, 817,2337},
		{2826,3723,3039,1429}, {2332,2803,1696,1177},
		{2089,3185,1256,1901}, {3780, 184,3715,  81},
		{1700, 663,2077,1669}, {3712, 499,3019,2633},
		{ 150,3784,1497,2269}, {2000,1631,1101, 129},
		{3375,1925, 717,1141}, {1621,3912,  51, 249},
		{3090,1398, 981,3917}, {3765,1349,1978,2481},
		{1149,1441,1813,3941}, {3146,2224,3881,2217},
		{  33,2411,  76,2749}, {3082,1907,3846,3041},
		{2741,3192,3694,1877}, { 359,2786,1682, 345},
		{3316, 382, 124,2861}, {1749,  37,1660,1809},
		{ 185, 759,3997,3141}, {2784,2948, 479,2825},
		{2202,1862,1141, 157}, {2199,3802, 886,2881},
		{1364,2423,3514,3637}, {1244,2051,1301,1465},
		{2020,2295,3604,2829}, {3160,1332,1888,2161},
		{2785,1832,1836,3365}, {2772,2405,1990, 361},
		{1217,3638,2058,2685}, {1822,3661, 692,3745},
		{1245, 327,1194,2325}, {2252,3660,  20,3609},
		{3904, 716,3285,3821}, {2774,1842,2046,3537},
		{ 997,3987,2107, 517}, {2573,1368,3508,3017},
		{1148,1848,3525,2141}, { 545,2366,3801,1537}
	};

	if(0 == n){ return; }
	static int internal_iseed[4] = {2398, 691, 782, 721};
	int *_iseed = iseed;
	if(NULL == _iseed){
		_iseed = internal_iseed;
	}
	int it1, it2, it3, it4;
	
	for(size_t j = 0; j < n; ++j){
		int i1 = _iseed[0];
		int i2 = _iseed[1];
		int i3 = _iseed[2];
		int i4 = _iseed[3] | 1; // ensure oddness
		
		const size_t i = j % 128;
		
		// Multiply the seed by i-th power of the multiplier modulo 2^48
		it4 =       i4 * mm[i][3];
		it3 = it4 / 4096;
		it4 -= it3 << 12;
		it3 = it3 + i3 * mm[i][3] + i4 * mm[i][2];
		it2 = it3 / 4096;
		it3 -= it2 << 12;
		it2 = it2 + i2 * mm[i][3] + i3 * mm[i][2] + i4 * mm[i][1];
		it1 = it2 / 4096;
		it2 -= it1 << 12;
		it1 = it1 + i1 * mm[i][3] + i2 * mm[i][2] + i3 * mm[i][1] + i4 * mm[i][0];
		it1 %= 4096;

		// Convert 48-bit integer to a real number in the interval [0,1)
		x[j] = 
			((T)it1 +
			((T)it2 +
			((T)it3 +
			 (T)it4 * (T(1)/T(4096)))
					* (T(1)/T(4096)))
					* (T(1)/T(4096)))
					* (T(1)/T(4096));

		// If a real number has n bits of precision, and the first
		// n bits of the 48-bit integer above happen to be all 1 (which
		// will occur about once every 2^n calls), then x[i] will
		// be rounded to exactly 1.0. 
		// Note the case x[i] = 0 should not be possible.
		// We subtract the number from 1 to get it in the range [0,1).
		x[j] = T(1) - x[j];
		if(i+1 == 128){ // Update seed every 128 iterations
			_iseed[0] = it1;
			_iseed[1] = it2;
			_iseed[2] = it3;
			_iseed[3] = it4;
		}
	}
	// Update seed
    _iseed[0] = it1;
    _iseed[1] = it2;
    _iseed[2] = it3;
    _iseed[3] = it4;
}

template <typename T>
struct Utility{
	static T c11(){ return T(1); }
	static void Uniform01Vector(size_t n, T *x, int iseed[4] = NULL){
		UniformRealVector(n, x, iseed);
	}
	static T UniformToNormal(const T &a, const T &b){
		T angle = Traits<T>::twopi() * b;
		return sqrt(T(-2) * log1p(-a)) * cos(angle);
	}
	static T UniformToUnitDisc(const T &a, const T &b){
		return T(2)*a - T(1);
	}
	static T UniformToUnitCircle(const T &b){
		return (T(2)*b < T(1)) ? T(-1) : T(1);
	}
};
template <typename T>
struct Utility<std::complex<T> >{
	typedef std::complex<T> complex_type;
	static std::complex<T> c11(){ return std::complex<T>(T(1),T(1)); }
	static void Uniform01Vector(size_t n, std::complex<T> *x, int iseed[4] = NULL){
		UniformRealVector(2*n, reinterpret_cast<T*>(x), iseed);
	}
	static complex_type UniformToNormal(const T &a, const T &b){
		return sqrt(T(-2) * log1p(-a)) * UniformToUnitCircle(b);
	}
	static complex_type UniformToUnitDisc(const T &a, const T &b){
		return sqrt(a) * UniformToUnitCircle(b);
	}
	static complex_type UniformToUnitCircle(const T &b){
		T angle = Traits<T>::twopi() * b;
		return complex_type(cos(angle), sin(angle));
	}
};

namespace Distribution{
///////////////////////////////////////////////////////////////////////
// Distribution
// ------------
// The enumeration of allowable random number distributions
// 
// * Uniform01:  The uniform distribution on [0,1). For complex numbers
//               the real and imaginary parts are independently drawn.
// * Uniform_11: The uniform distribution on [-1,1). For complex numbers
//               the real and imaginary parts are independently drawn.
// * Normal01:   The standoard normal distribution. For complex numbers
//               the real and imaginary parts are independently drawn.
// * UnitDisc:   The uniform distribution within the unit disc. For real
//               numbers this is the same as Uniform_11.
// * UnitCircle: The uniform distribution on the unit circle. For real
//               numbers this is the uniform distribution on the set
//               {-1,1}.
//
enum Distribution{
	Uniform01,
	Uniform_11,
	Normal01,
	UnitDisc,
	UnitCircle
};
}

///////////////////////////////////////////////////////////////////////
// GenerateVector
// --------------
// Generates a vector of numbers each drawn from a specified
// distribution. This corresponds approximately to the Lapack
// routines _larnv.
//
// Arguments
// dist   The distribution to draw from (see documentation on
//        Distribution).
// n      The length of the vector.
// x      The output vector (length n, increment must be 1).
// iseed  The seed array. See documentation for UniformRealVector.
//
template <typename T>
void GenerateVector(
	Distribution::Distribution dist, size_t n, T *x,
	int iseed[4] = NULL
){
	typedef typename Traits<T>::real_type real_type;
	switch(dist){
	case Distribution::Uniform01:
		Utility<T>::Uniform01Vector(n, x, iseed);
		break;
	case Distribution::Uniform_11:
		Utility<T>::Uniform01Vector(n, x, iseed);
		for(size_t i = 0; i < n; ++i){
			x[i] = real_type(2) * x[i] - Utility<T>::c11();
		}
		break;
	case Distribution::Normal01:
		{
			const size_t chunksize = 128;
			real_type pool[chunksize];
			for(size_t i0 = 0; i0 < n; i0 += chunksize/2){
				size_t ilim = chunksize/2; if(n < i0+ilim){ ilim = n-i0; }
				UniformRealVector(2*ilim, pool, iseed);
				for(size_t i = 0; i < ilim; ++i){
					x[i0+i] = Utility<T>::UniformToNormal(pool[2*i+0], pool[2*i+1]);
				}
			}
		}
		break;
	case Distribution::UnitDisc: // for reals, this is the same as Uniform_11;
		{
			const size_t chunksize = 128;
			real_type pool[chunksize];
			for(size_t i0 = 0; i0 < n; i0 += chunksize/2){
				size_t ilim = chunksize/2; if(n < i0+ilim){ ilim = n-i0; }
				UniformRealVector(chunksize, pool, iseed);
				for(size_t i = 0; i < ilim; ++i){
					x[i0+i] = Utility<T>::UniformToUnitDisc(pool[2*i+0], pool[2*i+1]);
				}
			}
		}
		break;
	case Distribution::UnitCircle: // for reals, this is the same as Uniform_11;
		{
			const size_t chunksize = 128;
			real_type pool[chunksize];
			for(size_t i0 = 0; i0 < n; i0 += chunksize/2){
				size_t ilim = chunksize/2; if(n < i0+ilim){ ilim = n-i0; }
				UniformRealVector(chunksize, pool, iseed);
				for(size_t i = 0; i < ilim; ++i){
					x[i0+i] = Utility<T>::UniformToUnitCircle(pool[2*i+0]);
				}
			}
		}
		break;
	default:
		break;
	}
}

///////////////////////////////////////////////////////////////////////
// UniformReal
// -----------
// Generates a single number from the uniform distribution in the
// interval [0,1). This routine returns a real number.
//
// Arguments
// iseed  The seed array. See documentation for UniformRealVector.
template <typename T>
typename Traits<T>::real_type UniformReal(int iseed[4] = NULL){
	typename Traits<T>::real_type r;
	UniformRealVector(1, &r, iseed);
	return r;
}

///////////////////////////////////////////////////////////////////////
// Uniform
// -------
// Generates a single number from the uniform distribution in the
// interval [0,1).
// For complex numbers, the real and imaginary parts are each drawn
// from this distribution.
//
// Arguments
// iseed  The seed array. See documentation for UniformRealVector.
template <typename T>
T Uniform(int iseed[4] = NULL){
	T r;
	Utility<T>::Uniform01Vector(1, &r, iseed);
	return r;
}

///////////////////////////////////////////////////////////////////////
// StandardNormal
// --------------
// Generates a single number from the standard normal distribution.
// For complex numbers, the real and imaginary parts are each drawn
// from this distribution.
//
// Arguments
// iseed  The seed array. See documentation for UniformRealVector.
template <typename T>
T StandardNormal(int iseed[4] = NULL){
	typename Traits<T>::real_type r[2];
	UniformRealVector(2, &r, iseed);
	return Utility<T>::UniformToNormal(r[0], r[1]);
}

///////////////////////////////////////////////////////////////////////
// UnitDisc
// --------
// Generates a single number uniformly distributed within the unit
// circle. For real numbers, this is the uniform distribution on the
// interval [0,1).
//
// Arguments
// iseed  The seed array. See documentation for UniformRealVector.
template <typename T>
T UnitDisc(int iseed[4] = NULL){
	typename Traits<T>::real_type r[2];
	UniformRealVector(2, &r, iseed);
	return Utility<T>::UniformToUnitDisc(r[0], r[1]);
}

///////////////////////////////////////////////////////////////////////
// Unitcircle
// ----------
// Generates a single number uniformly distributed on the unit circle.
// For real numbers, this is the uniform distribution on the set
// {-1,1}.
//
// Arguments
// iseed  The seed array. See documentation for UniformRealVector.
template <typename T>
T Unitcircle(int iseed[4] = NULL){
	typename Traits<T>::real_type r[2];
	UniformRealVector(2, &r, iseed);
	return Utility<T>::Unitcircle(r[0], r[1]);
}

} // namespace Random
} // namespace RNP

#endif // RNP_RANDOM_HPP_INCLUDED
