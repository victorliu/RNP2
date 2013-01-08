#ifndef RNP_REFLECTOR_HPP_INCLUDED
#define RNP_REFLECTOR_HPP_INCLUDED

#include <cstddef>
#include <RNP/BLAS.hpp>
#include <RNP/Types.hpp>
#include <RNP/Debug.hpp>
#include <iostream>

namespace RNP{
namespace LA{

namespace Reflector{

template <typename T>
size_t LastNonzeroColumn(size_t m, size_t n, const T *a, size_t lda){ // ilazlc, iladlc, ilaclc, ilaslc
	// if n = 0, returns -1
	if(0 == n || T(0) != a[0+(n-1)*lda] || T(0) != a[m-1+(n-1)*lda]){ return n-1; }
	size_t j = n;
	while(j --> 0){
		for(size_t i = 0; i < m; ++i){
			if(T(0) != a[i+j*lda]){ return j; }
		}
	}
	return j; // should be -1, but unsigned
}

template <typename T>
size_t LastNonzeroRow(size_t m, size_t n, const T *a, size_t lda){ // ilazlr, iladlr, ilaclr, ilaslr
	// if m = 0, returns -1
	if(0 == m || T(0) != a[m-1+0*lda] || T(0) != a[m-1+(n-1)*lda]){ return m-1; }
	size_t i = m;
	while(i --> 0){
		for(size_t j = 0; j < n; ++j){
			if(T(0) != a[i+j*lda]){ return i; }
		}
	}
	return i; // should be -1, but unsigned
}

// Let nrm = norm([a ; x], 2) and beta = sigma * nrm
// where sigma = nrmsign ? -1 : 1
// Returns:
//    xnrmest = 0 if norm(x,2) is zero, nonzero otherwise
//    beta
//    tau = (beta - a) / beta
//    scale = 1 / (a - beta)
// If xnrmest is zero, then the others are not set
template <typename T>
void Norm2Ratio(
	const T &a, size_t n, const T *x, size_t incx,
	bool nrmsign,
	typename Traits<T>::real_type *xnrmest,
	typename Traits<T>::real_type *beta,
	T *tau, T *scale
){
	typedef typename Traits<T>::real_type real_type;
	static const real_type rzero(0);
	static const real_type rone(1);
	const real_type rooteps = sqrt(Traits<real_type>::eps());
	// Calculate norm of x to extended precision
	real_type xssq(1), xssq2(0), xscale(0);
	while(n --> 0){
		if(rzero != Traits<T>::real(*x)){
			real_type temp = Traits<real_type>::abs(Traits<T>::real(*x));
			if(temp > xscale){
				real_type r = xscale/temp;
				xssq = xssq*r*r + rone;
				xssq2 = xssq2*r*r;
				xscale = temp;
			}else{
				real_type r = temp/xscale;
				if(r < rooteps){
					xssq2 += r*r;
				}else{
					xssq += r*r;
				}
			}
		}
		if(rzero != Traits<T>::imag(*x)){
			real_type temp = Traits<real_type>::abs(Traits<T>::imag(*x));
			if(temp > xscale){
				real_type r = xscale/temp;
				xssq = xssq*r*r + rone;
				xssq2 = xssq2*r*r;
				xscale = temp;
			}else{
				real_type r = temp/xscale;
				if(r < rooteps){
					xssq2 += r*r;
				}else{
					xssq += r*r;
				}
			}
		}
		x += incx;
	}
	*xnrmest = xscale;
	if(rzero == xscale){ return; }
	// Add contribution of 'a' and compute beta
	real_type anrm = Traits<T>::abs(a);
	real_type rt, beta_scale;
	if(anrm > xscale){
		real_type r = xscale / anrm;
		rt = sqrt(r*r*(xssq + xssq2) + rone);
		beta_scale = anrm;
	}else{
		real_type r = anrm / xscale;
		rt = sqrt(xssq + xssq2 + r*r);
		beta_scale = xscale;
	}
	*beta = beta_scale * rt;
	if(nrmsign){ *beta = -*beta; }
	// Compute tau = (beta - a) / beta
	T ascaled = a / beta_scale;
	if(nrmsign){ // tau = (rt + ascaled) / rt
		if(Traits<T>::real(a) > rzero){ // this is the more stable case
			*tau = (rt + ascaled) / rt;
			*scale = Traits<T>::div(T(1), a - *beta);
		}else{ // (nrm + a) could lead to cancellation
			// At this point we have:
			//   nrm^2 = beta_scale^2 * rt^2
			//         = xscale^2 * (xssq + xssq2) + anrm^2
			const real_type xnrm = xscale / beta_scale * sqrt(xssq + xssq2);
			const real_type gamma = rt - Traits<T>::real(ascaled);
			const real_type delta = Traits<T>::imag(ascaled) * (Traits<T>::imag(ascaled)/gamma) + xnrm * (xnrm/gamma);
			*tau = Traits<T>::copyimag(delta/rt, ascaled/(-rt));
			*scale = Traits<T>::div(T(1), Traits<T>::copyimag(-delta, ascaled));
		}
	}else{ // tau = (rt - ascaled) / rt
		if(Traits<T>::real(a) < rzero){ // this is the more stable case
			*tau = (rt - ascaled) / rt;
			*scale = Traits<T>::div(T(1), a - *beta);
		}else{ // (nrm + a) could lead to cancellation
			// At this point we have:
			//   nrm^2 = beta_scale^2 * rt^2
			//         = xscale^2 * (xssq + xssq2) + anrm^2
			const real_type xnrm = xscale / beta_scale * sqrt(xssq + xssq2);
			const real_type gamma = rt + Traits<T>::real(ascaled);
			const real_type delta = -(Traits<T>::imag(ascaled) * (Traits<T>::imag(ascaled)/gamma) + xnrm * (xnrm/gamma));
			*tau = -Traits<T>::copyimag(delta/rt, ascaled/rt);
			*scale = Traits<T>::div(T(1), Traits<T>::copyimag(delta, ascaled));
		}
	}
}


// Purpose
// =======
// 
// Generates a complex elementary reflector H of order n, such that
// 
// H' * ( alpha ) = ( beta ),   H' * H = I.
//      (   x   )   (   0  )
// 
// where alpha and beta are scalars, with beta real, and x is an
// (n-1)-element complex vector. H is represented in the form
// 
// H' = I - tau * ( 1 ) * ( 1 v' ) ,
//                ( v )
// 
// where tau is a complex scalar and v is a complex (n-1)-element
// vector. Note that H is not hermitian.
// 
// If the elements of x are all zero and alpha is real, then tau = 0
// and H is taken to be the unit matrix.
// 
// Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1 .
// 
// Arguments
// =========
//
// vconj   If true, the input vector is conjugated (both alpha and x)
//
// n       The order of the elementary reflector.
// 
// alpha   (input/output) 
//         On entry, the value alpha.
//         On exit, it is overwritten with the value beta.
// 
// x       (input/output) vector of dimension n-1
//         On entry, the vector x.
//         On exit, it is overwritten with the vector v.
// 
// incx    The increment between elements of x.
// 
// tau     (output) The value tau.
//
//// Notes:
// The algorithm is as follows:
//   Pick beta = -sign(real(alpha)) * norm([alpha;x])
//   Set tau = (beta - alpha) / beta
//   Set v = x / (alpha - beta)
// where rescalings have been left out.
//
// When x = 0, then beta = alpha, and H is just the identity, so tau can be zero.
// When x is small, so that beta is alpha+delta, then the denominator (alpha - beta)
// is small. Basically, v has to be very big, and tau has to be correspondingly
// very small. For analysis, let's assume that norm([alpha;x]) = 1. Then if
// alpha = 1-delta, then norm(x) ~ sqrt(2 delta). If we assume that alpha is
// real and positive, then we would set beta = -1, tau = -delta, and v = x / delta.
// This results in norm(v) ~ sqrt(2/delta).
//
// If instead, we store the reflectors as
//    H = I - ( t ) * ( t w' ) ,
//            ( w )
// so that t = sqrt(tau) and w = t * v, then the algorithm is
//   Pick beta = -sign(real(alpha)) * norm([alpha;x])
//   Set sdelta = sqrt(beta - alpha)
//   Set tau = sdelta / beta
//   Set v = x / -sdelta
template <typename T> // _larfg
void Generate(size_t n, T *alpha, T *x, size_t incx, T *tau){
	using namespace RNP;
	typedef typename Traits<T>::real_type real_type;

	if(n == 0){
		*tau = T(0);
		return;
	}
	
	real_type xnorm = BLAS::Norm2(n-1, x, incx );
	if(xnorm == 0 && Traits<T>::imag(*alpha) == 0){ // H  =  I
		*tau = T(0);
	}else{ // general case
		real_type beta = Traits<real_type>::hypot3(Traits<T>::real(*alpha), Traits<T>::imag(*alpha), xnorm);
		if(Traits<T>::real(*alpha) > 0){ beta = -beta; };
		const real_type safmin = Traits<real_type>::min() / (2*Traits<real_type>::eps());
		const real_type rsafmn = real_type(1) / safmin;
		// 
		size_t knt = 0;
		if(Traits<real_type>::abs(beta) < safmin){ // XNORM, BETA may be inaccurate; scale X and recompute them
			do{
				++knt;
				BLAS::Scale( n-1, rsafmn, x, incx );
				beta *= rsafmn;
				*alpha *= rsafmn;
			}while(abs(beta) < safmin );
			// New BETA is at most 1, at least SAFMIN
			xnorm = BLAS::Norm2( n-1, x, incx );
			beta = Traits<real_type>::hypot3(Traits<T>::real(*alpha), Traits<T>::imag(*alpha), xnorm );
			if(Traits<T>::real(*alpha) > 0){ beta = -beta; }
		}
		*tau = (beta - *alpha) / beta;
		*alpha = real_type(1)/(*alpha-beta);
		BLAS::Scale( n-1, *alpha, x, incx );
		// If alpha is subnormal, it may lose relative accuracy
		while(knt --> 0){ 
			beta *= safmin;
		}
		*alpha = beta;
	}
}

// Implementation using more robust Norm2, currently broken
template <typename T>
void Generate2(size_t n, T *alpha, T *x, size_t incx, T *tau){
	typedef typename Traits<T>::real_type real_type;
	static const real_type rzero(0);

	if(n == 0){
		*tau = T(0);
		return;
	}
	
	bool nrmsign = (Traits<T>::real(*alpha) > 0); // for _larfg
	// bool nrmsign = false; // for _larfgp
	
	T scale;
	real_type beta, xnrmest;
	Reflector::Norm2Ratio(*alpha, n-1, x, incx, nrmsign, &xnrmest, &beta, tau, &scale);
	if(rzero == xnrmest){
		// H  =  [1-alpha/abs(alpha) 0; 0 I], sign chosen so alpha >= 0.
		if(Traits<T>::imag(*alpha) == rzero){
			if(Traits<T>::real(*alpha) >= rzero){
				*tau = T(0);
			}else{ // alpha was real and negative
				*tau = real_type(2);
				*alpha = -*alpha;
			}
		}else{ // real arithmetic never gets here
			// Only "reflecting" the diagonal entry to be real and non-negative.
			xnrmest = Traits<T>::abs(*alpha);
			*tau = real_type(1) - *alpha/xnrmest;
			*alpha = xnrmest;
		}
	}else{ // general case
		BLAS::Scale(n-1, scale, x, incx);
		*alpha = beta;
	}
}

// Applies an elementary reflector H to an M-by-N matrix C,
// from either the left or the right. H is represented in the form
//        H = I - tau * v * v'
// where tau is a complex scalar and v is a complex vector.
//
// To apply H' (the conjugate transpose of H), supply conj(tau) instead

// Arguments
// =========
//
// SIDE    = 'L': form  H * C
//         = 'R': form  C * H
//
// vone    If true, the first element of V is assumed to be 1 instead
//         of the actual input value.
//
// vconj   If true, elements after the first element of V are assumed
//         to be conjugated.
//
// M       The number of rows of the matrix C.
//
// N       The number of columns of the matrix C.
//
// V       Length m if SIDE = 'L'
//             or n if SIDE = 'R'
//         The vector v in the representation of H. V is not used if
//         TAU = 0.
//
// INCV    The increment between elements of v. INCV <> 0.
//
// TAU     The value tau in the representation of H.
//
// C       (input/output) array dimension (LDC,N)
//         On entry, the M-by-N matrix C.
//         On exit, C is overwritten by the matrix H * C if SIDE = 'L',
//         or C * H if SIDE = 'R'.
//
// LDC     The leading dimension of the array C. LDC >= max(1,M).
//
// WORK    (workspace) array dimension
//                        (N) if SIDE = 'L'
//                     or (M) if SIDE = 'R'

template <typename T> // _larf
void Apply(
	const char *side, bool vone, bool vconj, size_t m, size_t n,
	const T *v, size_t incv, const T &tau, T *c, size_t ldc, T *work
){
	size_t lenv = 0;
	size_t lenc = 0;
	if(T(0) != tau){
		// Set up variables for scanning V.  LASTV begins pointing to the end of V.
		if('L' == side[0]){
			lenv = m;
		}else{
			lenv = n;
		}
		if(0 == lenv){ return; }
		size_t i = 0;
		if(incv > 0){
			i = (lenv - 1) * incv;
		}
		// Look for the last non-zero row in V.
		while(lenv > 0 && (T(0) == v[i])){
			--lenv;
			i -= incv;
		}
		if('L' == side[0]){ // Scan for the last non-zero column in C(1:lastv,:).
			lenc = 1+Reflector::LastNonzeroColumn(lenv, n, c, ldc);
		}else{ // Scan for the last non-zero row in C(:,1:lastv).
			lenc = 1+Reflector::LastNonzeroRow(m, lenv, c, ldc);
		}
		if(0 == lenc){ return; }
	}
	
	const T one(1);
	
	if(!vone && !vconj){
		if('L' == side[0]){ // Form  H * C
			// w(1:lastc,1) := C(1:lastv,1:lastc)' * v(1:lastv)
			BLAS::MultMV("C", lenv, lenc, T(1), c, ldc, v, incv, T(0), work, 1);
			// C(1:lastv,1:lastc) := C(...) - v(1:lastv) * w(1:lastc,1)'
			BLAS::ConjugateRank1Update(lenv, lenc, -tau, v, incv, work, 1, c, ldc);
		}else{ // Form  C * H
			// w(1:lastc,1) := C(1:lastc,1:lastv) * v(1:lastv)
			BLAS::MultMV("N", lenc, lenv, T(1), c, ldc, v, incv, T(0), work, 1);
			// C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv)'
			BLAS::ConjugateRank1Update(lenc, lenv, -tau, work, 1, v, incv, c, ldc);
		}
	}else{
		if('L' == side[0]){ // Form  H * C
			// w(1:lastc,1) := C(1:lastv,1:lastc)' * v(1:lastv)
			if(lenv > 1){ // Add the remaining contribution
				if(vconj){
					BLAS::MultMV("T", lenv-1, lenc, T(1), &c[1+0*ldc], ldc, &v[incv], incv, T(0), work, 1);
					BLAS::Conjugate(lenc, work, 1);
					// Add in first column
					BLAS::Axpy(lenc, one, c, ldc, work, 1);
				}else{
					BLAS::Copy(lenc, c, ldc, work, 1);
					BLAS::Conjugate(lenc, work, 1);
					BLAS::MultMV("C", lenv-1, lenc, T(1), &c[1+0*ldc], ldc, &v[incv], incv, T(1), work, 1);
				}
			}else{
				BLAS::Copy(lenc, c, ldc, work, 1);
				BLAS::Conjugate(lenc, work, 1);
			}
			
			// C(1:lastv,1:lastc) := C(...) - v(1:lastv) * w(1:lastc,1)'
			// Add in first row contribution
			BLAS::ConjugateRank1Update(1, lenc, -tau, &one, incv, work, 1, c, ldc);
			// Add remaining contribution
			if(lenv > 1){
				if(vconj){
					for(size_t j = 0; j < lenc; ++j){
						for(size_t i = 1; i < lenv; ++i){
							c[i+0*ldc] -= tau * Traits<T>::conj(v[i*incv]) * Traits<T>::conj(work[j]);
						}
					}
				}else{
					BLAS::ConjugateRank1Update(lenv-1, lenc, -tau, &v[incv], incv, work, 1, &c[1+0*ldc], ldc);
				}
			}
		}else{ // Form  C * H
			// w(1:lastc,1) := C(1:lastc,1:lastv) * v(1:lastv)
			// Add in first column
			BLAS::Copy(lenc, c, 1, work, 1);
			if(lenv > 1){ // Add remaining contribution
				if(vconj){
					for(size_t j = 1; j < lenv; ++j){
						T cvj(Traits<T>::conj(v[j*incv]));
						for(size_t i = 0; i < lenc; ++i){
							work[i] += c[i+j*ldc] * cvj;
						}
					}
				}else{
					BLAS::MultMV("N", lenc, lenv-1, T(1), &c[0+1*ldc], ldc, &v[incv], incv, T(1), work, 1);
				}
			}
			// C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv)'
			// Add in first col
			BLAS::Axpy(lenc, -tau, work, 1, c, 1);
			if(lenv > 1){
				if(vconj){
					BLAS::Rank1Update(lenc, lenv-1, -tau, work, 1, &v[incv], incv, &c[0+1*ldc], ldc);
				}else{
					BLAS::ConjugateRank1Update(lenc, lenv-1, -tau, work, 1, &v[incv], incv, &c[0+1*ldc], ldc);
				}
			}
		}
	}
}


template <typename T> // zlarfp, dlarfp, clarfp, slarfp
void GeneratePositive(size_t n, T *alpha, T *x, size_t incx, T *tau){
	typedef typename Traits<T>::real_type real_type;
	
	// Purpose
	// =======
	// Like ReflectorGenerate, but beta is non-negative.

	// Generates a complex elementary reflector H of order n, such that
	//       H' * ( alpha ) = ( beta ),   H' * H = I.
	//            (   x   )   (   0  )
	// where alpha and beta are scalars, beta is real and non-negative, and
	// x is an (n-1)-element complex vector.  H is represented in the form
	//       H = I - tau * ( 1 ) * ( 1 v' )
	//                     ( v )
	// where tau is a complex scalar and v is a complex (n-1)-element
	// vector. Note that H is not hermitian.
	// If the elements of x are all zero and alpha is real, then tau = 0
	// and H is taken to be the unit matrix.
	// Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1 .

	// Arguments
	// =========

	// N       The order of the elementary reflector.

	// ALPHA   (input/output) COMPLEX*16
	//         On entry, the value alpha.
	//         On exit, it is overwritten with the value beta.

	// X       (input/output) COMPLEX*16 array, dimension
	//                        (1+(N-2)*abs(INCX))
	//         On entry, the vector x.
	//         On exit, it is overwritten with the vector v.

	// INCX    The increment between elements of X. INCX > 0.

	// TAU     The value tau.

	if(n < 1){
		*tau = 0;
		return;
	}

	real_type xnorm = BLAS::Norm2(n-1, x, incx);

	if(xnorm == 0) {
		// H  =  [1-alpha/abs(alpha) 0; 0 I], sign chosen so alpha >= 0.
		if(Traits<T>::imag(*alpha) == 0){
			if(Traits<T>::real(*alpha) >= 0){
				// When tau == 0, the vector is special-cased to be
				// all zeros in the application routines.  We do not need
				// to clear it. We will anyways
				*tau = 0;
			}else{
				// However, the application routines rely on explicit
				// zero checks when tau != 0, and we must clear X.
				*tau = 2;
				*alpha = -*alpha;
			}
			--n; // n not needed anymore
			while(n --> 0){
				*x = 0;
				x += incx;
			}
		}else{
			// Only "reflecting" the diagonal entry to be real and non-negative.
			xnorm = Traits<T>::abs(*alpha);
			*tau = real_type(1) - *alpha/xnorm;
			--n; // n not needed anymore
			while(n --> 0){
				*x = 0;
				x += incx;
			}
			*alpha = xnorm;
		}
	}else{ // general case
		real_type beta = Traits<real_type>::hypot3(Traits<T>::real(*alpha), Traits<T>::imag(*alpha), xnorm);
		if(Traits<T>::real(*alpha) < 0){ beta = -beta; }
		const real_type safmin = Traits<real_type>::min() / Traits<real_type>::eps();
		const real_type rsafmn = 1. / safmin;

		size_t knt = 0;
		if(Traits<real_type>::abs(beta) < safmin){
			// xnorm, beta may be inaccurate; scale x and recompute them
			do{
				++knt;
				BLAS::Scale(n-1, rsafmn, x, incx);
				beta *= rsafmn;
				*alpha *= rsafmn;
			}while(abs(beta) < safmin);
			// New beta is at most 1, at least safmin
			xnorm = BLAS::Norm2(n-1, x, incx);
			beta = Traits<real_type>::hypot3(Traits<T>::real(*alpha), Traits<T>::imag(*alpha), xnorm);
			if(Traits<T>::real(*alpha) < 0){ beta = -beta; }
		}
		*alpha += beta;
		if(beta < 0){
			beta = -beta;
		}else{
			// The following original 3 lines is specific to complex numbers
			//  alphr = alphi * (alphi / alpha->real()) + xnorm * (xnorm / alpha->real());
			//  *tau = complex<double>(alphr / beta, -alphi / beta);
			//  *alpha = complex<double>(-alphr, alphi);
			
			// Equivalent to:
			// alphr = -( alphi * (alphi / alpha->real()) + xnorm * (xnorm / alpha->real()) );
			// alpha = complex<double>(alphr, alphi);
			// tau = -alpha/beta
			//
			// alphr = alpha->real();
			// temp = alphi * (alphi / alphr) + xnorm * (xnorm / alphr);
			// alpha = complex<double>(alphr-(temp+alphr), alphi);
			// tau = -alpha/beta
			//
			// Note that: temp+alphr = alphr*[(alpha/alphr)^2 + (xnorm/alphr)^2]
			// 
			real_type alphr = Traits<T>::real(*alpha);
			real_type ana = Traits<T>::abs((*alpha / alphr));
			xnorm *= (xnorm/alphr);
			*alpha -= (alphr*ana)*ana;
			*alpha -= xnorm;
		}
		*tau = -*alpha/beta;
		*alpha = Traits<T>::div(T(1), *alpha);
		BLAS::Scale(n-1, (*alpha), x, incx);

		// If beta is subnormal, it may lose relative accuracy
		while(knt --> 0){
			beta *= safmin;
		}
		*alpha = beta;
	}
}


//  Purpose
//  =======
//
//  Forms the triangular factor T of a complex block reflector H
//  of order n, which is defined as a product of k elementary reflectors.
//
//  If DIR = 'F', H = H(1) H(2) . . . H(k) and T is upper triangular;
//
//  If DIR = 'B', H = H(k) . . . H(2) H(1) and T is lower triangular.
//
//  If STOREV = 'C', the vector which defines the elementary reflector
//  H(i) is stored in the i-th column of the array V, and
//
//     H  =  I - V * T * V**H
//
//  If STOREV = 'R', the vector which defines the elementary reflector
//  H(i) is stored in the i-th row of the array V, and
//
//     H  =  I - V**H * T * V
//
//  Arguments
//  =========
//
//  DIR     (input) CHARACTER*1
//          Specifies the order in which the elementary reflectors are
//          multiplied to form the block reflector:
//          = 'F': H = H(1) H(2) . . . H(k) (Forward)
//          = 'B': H = H(k) . . . H(2) H(1) (Backward)
//
//  STOREV  (input) CHARACTER*1
//          Specifies how the vectors which define the elementary
//          reflectors are stored (see also Further Details):
//          = 'C': columnwise
//          = 'R': rowwise
//
//  N       (input) INTEGER
//          The order of the block reflector H. N >= 0.
//
//  K       (input) INTEGER
//          The order of the triangular factor T (= the number of
//          elementary reflectors). K >= 1.
//
//  V       (input) dimension
//                               (LDV,K) if STOREV = 'C'
//                               (LDV,N) if STOREV = 'R'
//          The matrix V. See further details.
//
//  LDV     (input) INTEGER
//          The leading dimension of the array V.
//          If STOREV = 'C', LDV >= max(1,N); if STOREV = 'R', LDV >= K.
//
//  TAU     (input) COMPLEX*16 array, dimension (K)
//          TAU(i) must contain the scalar factor of the elementary
//          reflector H(i).
//
//  T       (output) COMPLEX*16 array, dimension (LDT,K)
//          The k by k triangular factor T of the block reflector.
//          If DIRECT = 'F', T is upper triangular; if DIRECT = 'B', T is
//          lower triangular. The rest of the array is not used.
//
//  LDT     (input) INTEGER
//          The leading dimension of the array T. LDT >= K.
//
//  Further Details
//  ===============
//
//  The shape of the matrix V and the storage of the vectors which define
//  the H(i) is best illustrated by the following example with n = 5 and
//  k = 3. The elements equal to 1 are not stored; the corresponding
//  array elements are modified but restored on exit. The rest of the
//  array is not used.
//
//  DIR    = 'F' and STOREV = 'C':         DIR    = 'F' and STOREV = 'R':
//
//               V = (  1       )                 V = (  1 v1 v1 v1 v1 )
//                   ( v1  1    )                     (     1 v2 v2 v2 )
//                   ( v1 v2  1 )                     (        1 v3 v3 )
//                   ( v1 v2 v3 )
//                   ( v1 v2 v3 )
//
//  DIR    = 'B' and STOREV = 'C':         DIR    = 'B' and STOREV = 'R':
//
//               V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
//                   ( v1 v2 v3 )                     ( v2 v2 v2  1    )
//                   (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
//                   (     1 v3 )
//                   (        1 )
//
//  =====================================================================
//
template <typename T>
void GenerateBlockTr(
	const char *dir, const char *storev,
	size_t n, size_t k, T *v, size_t ldv, const T *tau,
	T *t, size_t ldt
){
	if(0 == n){ return; }
	if('F' == dir[0]){
		size_t prevlastv = n-1;
		for(size_t i = 0; i < k; ++i){
			if(i > prevlastv){ prevlastv = i; }
			if(T(0) == tau[i]){ // H[i] = I
				for(size_t j = 0; j <= i; ++j){
					t[j+i*ldt] = T(0);
				}
			}else{ // general case
				size_t lastv;
				if('C' == storev[0]){
					for(lastv = n-1; lastv > i; --lastv){ // skip trailing zeros
						if(T(0) != v[lastv+i*ldv]){ break; }
					}
					for(size_t j = 0; j < i; ++j){
						t[j+i*ldt] = -tau[i] * Traits<T>::conj(v[i+j*ldv]);
					}
					size_t jlim = 1+(lastv < prevlastv ? lastv : prevlastv);
					// T(1:i-1,i) := - tau(i) * V(i:j,1:i-1)**H * V(i:j,i)
					BLAS::MultMV(
						"C", jlim-i, i, -tau[i], &v[i+1+0*ldv], ldv,
						&v[i+1+i*ldv], ldv, T(1), &t[0+i*ldt], 1
					);
				}else{
					for(lastv = n-1; lastv > i; --lastv){ // skip trailing zeros
						if(T(0) != v[i+lastv*ldv]){ break; }
					}
					for(size_t j = 0; j < i; ++j){
						t[j+i*ldt] = -tau[i] * v[j+i*ldv];
					}
					size_t jlim = 1+(lastv < prevlastv ? lastv : prevlastv);
					// T(1:i-1,i) := - tau(i) * V(1:i-1,i:j) * V(i,i:j)**H
					BLAS::MultMM(
						"N", "C", i, 1, jlim-i, -tau[i], &v[0+(i+1)*ldv], ldv,
						&v[i+(i+1)*ldv], ldv, T(1), &t[0+i*ldt], ldt
					);
				}
				// T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i)
				BLAS::MultTrV("U","N","N", i, t, ldt, &t[0+i*ldt], 1);
				t[i+i*ldt] = tau[i];
				if(i > 0){
					if(lastv > prevlastv){ prevlastv = lastv; }
				}else{
					prevlastv = lastv;
				}
			}
		}
	}else{
		size_t prevlastv = 0;
		size_t i = k;
		while(i --> 0){
			if(T(0) == tau[i]){ // H[i] = I
				for(size_t j = i; j < k; ++j){
					t[j+i*ldt] = T(0);
				}
			}else{ // general case
				if(i+1 < k){
					size_t lastv;
					if('C' == storev[0]){
						for(lastv = 0; lastv < i; ++lastv){ // Skip any leading zeros.
							if(T(0) != v[lastv+i*ldv]){ break; }
						}
						for(size_t j = i+1; j < k; ++j){
							t[j+i*ldt] = -tau[i] * Traits<T>::conj(v[n-k+i+j*ldv]);
						}
						size_t jlim = (lastv > prevlastv ? lastv : prevlastv);
						// T(i+1:k,i) = -tau(i) * V(j:n-k+i,i+1:k)**H * V(j:n-k+i,i)
						BLAS::MultMV(
							"C", n-k+i-jlim, k-1-i, -tau[i], &v[jlim+(i+1)], ldv,
							&v[jlim+i*ldv], 1, T(1), &t[i+1+i*ldt], 1
						);
					}else{
						for(lastv = 0; lastv < i; ++lastv){ // Skip any leading zeros.
							if(T(0) != v[i+lastv*ldv]){ break; }
						}
						for(size_t j = i+1; j < k; ++j){
							t[j+i*ldt] = -tau[i] * v[j+(n-k+i)*ldv];
						}
						size_t jlim = (lastv > prevlastv ? lastv : prevlastv);
						// T(i+1:k,i) = -tau(i) * V(i+1:k,j:n-k+i) * V(i,j:n-k+i)**H
						BLAS::MultMM(
							"N","C", k-1-i, 1, n-k+i-jlim, -tau[i], &v[i+1+jlim*ldv], ldv,
							&v[i+jlim*ldv], ldv, T(1), &t[i+1+i*ldt], ldt
						);
					}
					// T(i+1:k,i) := T(i+1:k,i+1:k) * T(i+1:k,i)
					BLAS::MultTrV("L","N","N", k-1-i, &t[i+1+(i+1)*ldt], ldt, &t[i+1+i*ldt], 1);
					if(i > 0){
						if(lastv < prevlastv){ prevlastv = lastv; }
					}else{
						prevlastv = lastv;
					}
				}
				t[i+i*ldt] = tau[i];
			}
		}
	}
}

// Purpose:
// =============
//
// Applies a complex block reflector H or its transpose H**H to a
// complex M-by-N matrix C, from either the left or the right.
//
//  Arguments
//  =========
//
//  SIDE    (input) CHARACTER*1
//          = 'L': apply H or H**H from the Left
//          = 'R': apply H or H**H from the Right
//
//  TRANS   (input) CHARACTER*1
//          = 'N': apply H (No transpose)
//          = 'C': apply H**H (Conjugate transpose)
//
//  DIRECT  (input) CHARACTER*1
//          Indicates how H is formed from a product of elementary
//          reflectors
//          = 'F': H = H(1) H(2) . . . H(k) (Forward)
//          = 'B': H = H(k) . . . H(2) H(1) (Backward)
//
//  STOREV  (input) CHARACTER*1
//          Indicates how the vectors which define the elementary
//          reflectors are stored:
//          = 'C': Columnwise
//          = 'R': Rowwise
//
//  M       (input) INTEGER
//          The number of rows of the matrix C.
//
//  N       (input) INTEGER
//          The number of columns of the matrix C.
//
//  K       (input) INTEGER
//          The order of the matrix T (= the number of elementary
//          reflectors whose product defines the block reflector).
//
//  V       (input) COMPLEX*16 array, dimension
//                                (LDV,K) if STOREV = 'C'
//                                (LDV,M) if STOREV = 'R' and SIDE = 'L'
//                                (LDV,N) if STOREV = 'R' and SIDE = 'R'
//          The matrix V. See Further Details.
//
//  LDV     (input) INTEGER
//          The leading dimension of the array V.
//          If STOREV = 'C' and SIDE = 'L', LDV >= max(1,M);
//          if STOREV = 'C' and SIDE = 'R', LDV >= max(1,N);
//          if STOREV = 'R', LDV >= K.
//
//  T       (input) COMPLEX*16 array, dimension (LDT,K)
//          The triangular K-by-K matrix T in the representation of the
//          block reflector.
//
//  LDT     (input) INTEGER
//          The leading dimension of the array T. LDT >= K.
//
//  C       (input/output) COMPLEX*16 array, dimension (LDC,N)
//          On entry, the M-by-N matrix C.
//          On exit, C is overwritten by H*C or H**H*C or C*H or C*H**H.
//
//  LDC     (input) INTEGER
//          The leading dimension of the array C. LDC >= max(1,M).
//
//  WORK    (workspace) COMPLEX*16 array, dimension (LDWORK,K)
//
//  LDWORK  (input) INTEGER
//          The leading dimension of the array WORK.
//          If SIDE = 'L', LDWORK >= max(1,N);
//          if SIDE = 'R', LDWORK >= max(1,M).
//
//  Further Details
//  ===============
//
//  The shape of the matrix V and the storage of the vectors which define
//  the H(i) is best illustrated by the following example with n = 5 and
//  k = 3. The elements equal to 1 are not stored; the corresponding
//  array elements are modified but restored on exit. The rest of the
//  array is not used.
//
//  DIRECT = 'F' and STOREV = 'C':         DIRECT = 'F' and STOREV = 'R':
//
//               V = (  1       )                 V = (  1 v1 v1 v1 v1 )
//                   ( v1  1    )                     (     1 v2 v2 v2 )
//                   ( v1 v2  1 )                     (        1 v3 v3 )
//                   ( v1 v2 v3 )
//                   ( v1 v2 v3 )
//
//  DIRECT = 'B' and STOREV = 'C':         DIRECT = 'B' and STOREV = 'R':
//
//               V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
//                   ( v1 v2 v3 )                     ( v2 v2 v2  1    )
//                   (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
//                   (     1 v3 )
//                   (        1 )
//
//  =====================================================================
template <typename T>
void ApplyBlock(
	const char *side, const char *trans, const char *dir, const char *storev,
	size_t m, size_t n, size_t k, const T *v, size_t ldv,
	const T *t, size_t ldt, T *c, size_t ldc, T *work, size_t ldwork
){
	if(0 == m || 0 == n){ return; }
	const char *transt = ('N' == trans[0] ? "C" : "N");

	if('C' == storev[0]){
		if('F' == dir[0]){
			// Let  V =  ( V1 )    (first K rows)
			//           ( V2 )
			// where  V1  is unit lower triangular.
			if('L' == side[0]){
				// Form  H * C  or  H**H * C  where  C = ( C1 )
				//                                       ( C2 )
				size_t lastv = 1+LastNonzeroRow(m, k, v, ldv);
				if(k > lastv){ lastv = k; }
				const size_t lastc = 1+LastNonzeroColumn(lastv, n, c, ldc);
				// W := C**H * V  =  (C1**H * V1 + C2**H * V2)  (stored in WORK)
				// W := C1**H
				for(size_t j = 0; j < k; ++j){
					BLAS::Copy(lastc, &c[j+0*ldc], ldc, &work[0+j*ldwork], 1);
					BLAS::Conjugate(lastc, &work[0+j*ldwork], 1);
				}
				// W := W * V1
				BLAS::MultTrM("R","L","N","U", lastc, k, T(1), v, ldv, work, ldwork);
				if(lastv > k){
					// W := W + C2**H *V2
					BLAS::MultMM(
						"C","N", lastc, k, lastv-k, T(1), &c[k+0*ldc], ldc,
						&v[k+0*ldv], ldv, T(1), work, ldwork
					);
				}
				// W := W * T**H  or  W * T
				BLAS::MultTrM("R","U",transt,"N", lastc, k, T(1), t, ldt, work, ldwork);
				// C := C - V * W**H
				if(m > k){
					// C2 := C2 - V2 * W**H
					BLAS::MultMM(
						"N","C", lastv-k, lastc, k, T(-1), &v[k+0*ldv], ldv,
						work, ldwork, T(1), &c[k+0*ldc], ldc
					);
				}
				// W := W * V1**H
				BLAS::MultTrM("R","L","C","U", lastc, k, T(1), v, ldv, work, ldwork);
				// C1 := C1 - W**H
				for(size_t j = 0; j < k; ++j){
					for(size_t i = 0; i < lastc; ++i){
						c[j+i*ldc] -= Traits<T>::conj(work[i+j*ldwork]);
					}
				}
			}else if('R' == side[0]){
				// Form  C * H  or  C * H**H  where  C = ( C1  C2 )
				size_t lastv = 1+LastNonzeroRow(n, k, v, ldv);
				if(k > lastv){ lastv = k; }
				const size_t lastc = 1+LastNonzeroRow(m, lastv, c, ldc);
				// W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)
				// W := C1
				for(size_t j = 0; j < k; ++j){
					BLAS::Copy(lastc, &c[0+j*ldc], 1, &work[0+j*ldwork], 1);
				}
				// W := W * V1
				BLAS::MultTrM("R","L","N","U", lastc, k, T(1), v, ldv, work, ldwork);
				if(lastv > k ){
					// W := W + C2 * V2
					BLAS::MultMM(
						"N","N", lastc, k, lastv-k, T(1), &c[0+(k)*ldc], ldc,
						&v[k+0*ldv], ldv, T(1), work, ldwork
					);
				}
				// W := W * T  or  W * T**H
				BLAS::MultTrM("R","U",trans,"N", lastc, k, T(1), t, ldt, work, ldwork);
				// C := C - W * V**H
				if(lastv > k){
					// C2 := C2 - W * V2**H
					BLAS::MultMM(
						"N","C", lastc, lastv-k, k, T(-1), work, ldwork,
						&v[k+0*ldv], ldv, T(1), &c[0+(k)*ldc], ldc
					);
				}
				// W := W * V1**H
				BLAS::MultTrM("R","L","C","U", lastc, k, T(1), v, ldv, work, ldwork);
				// C1 := C1 - W
				for(size_t j = 0; j < k; ++j){
					for(size_t i = 0; i < lastc; ++i){
						c[i+j*ldc] -= work[i+j*ldwork];
					}
				}
			}
		}else{
			// Let  V =  ( V1 )
			//           ( V2 )    (last K rows)
			// where  V2  is unit upper triangular.
			if('L' == side[0]){
				// Form  H * C  or  H**H * C  where  C = ( C1 )
				//                                       ( C2 )
				const size_t lastc = 1+LastNonzeroColumn(m, n, c, ldc);
				// W := C**H * V  =  (C1**H * V1 + C2**H * V2)  (stored in WORK)
				// W := C2**H
				for(size_t j = 0; j < k; ++j){
					BLAS::Copy(lastc, &c[m-k+j+0*ldc], ldc, &work[0+j*ldwork], 1);
					BLAS::Conjugate(lastc, &work[0+j*ldwork], 1);
				}
				// W := W * V2
				BLAS::MultTrM(
					"R","U","N","U", lastc, k, T(1), &v[m-k+0*ldv], ldv, work, ldwork
				);
				if(m > k){
					// W := W + C1**H*V1
					BLAS::MultMM(
						"C","N", lastc, k, m-k, T(1), c, ldc,
						v, ldv, T(1), work, ldwork
					);
				}
				// W := W * T**H  or  W * T
				BLAS::MultTrM("R","L",transt,"N", lastc, k, T(1), t, ldt, work, ldwork);
				// C := C - V * W**H
				if(m > k){
					// C1 := C1 - V1 * W**H
					BLAS::MultMM(
						"N","C", m-k, lastc, k, T(-1), v, ldv,
						work, ldwork, T(1), c, ldc
					);
				}
				// W := W * V2**H
				BLAS::MultTrM(
					"R","U","C","U", lastc, k, T(1), &v[m-k+0*ldv], ldv, work, ldwork
				);
				// C2 := C2 - W**H
				for(size_t j = 0; j < k; ++j){
					for(size_t i = 0; i < lastc; ++i){
						c[m-k+j+i*ldc] -= Traits<T>::conj(work[i+j*ldwork]);
					}
				}
			}else if('R' == side[0]){
				// Form  C * H  or  C * H**H  where  C = (C1  C2)
				const size_t lastc = 1+LastNonzeroRow(m, n, c, ldc);
				// W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)
				// W := C2
				for(size_t j = 0; j < k; ++j){
					BLAS::Copy(lastc, &c[0+(n-k+j)*ldc], 1, &work[0+j*ldwork], 1);
				}
				// W := W * V2
				BLAS::MultTrM(
					"R","U","N","U", lastc, k, T(1), &v[n-k+0*ldv], ldv, work, ldwork
				);
				if(n > k){
					// W := W + C1 * V1
					BLAS::MultMM(
						"N","N", lastc, k, n-k, T(1), c, ldc, v, ldv, T(1), work, ldwork
					);
				}
				// W := W * T  or  W * T**H
				BLAS::MultTrM("R","L",trans,"N", lastc, k, T(1), t, ldt, work, ldwork);
				// C := C - W * V**H
				if(n > k){
					// C1 := C1 - W * V1**H
					BLAS::MultMM(
						"N","C", lastc, n-k, k, T(-1), work, ldwork,
						v, ldv, T(1), c, ldc
					);
				}
				// W := W * V2**H
				BLAS::MultTrM(
					"R","U","C","U", lastc, k, T(1), &v[n-k+0*ldv], ldv, work, ldwork
				);
				// C2 := C2 - W
				for(size_t j = 0; j < k; ++j){
					for(size_t i = 0; i < lastc; ++i){
						c[i+(n-k+j)*ldc] -= work[i+j*ldwork];
					}
				}
			}
		}
	}else if('R' == storev[0]){
		if('F' == dir[0]){
			// Let  V =  (V1  V2)    (V1: first K columns)
			// where  V1  is unit upper triangular.
			if('L' == side[0]){
				// Form  H * C  or  H**H * C  where  C = (C1)
				//                                       (C2)
				size_t lastv = 1+LastNonzeroColumn(k, m, v, ldv);
				if(k > lastv){ lastv = k; }
				const size_t lastc = 1+LastNonzeroColumn(lastv, n, c, ldc);
				// W := C**H * V**H  =  (C1**H * V1**H + C2**H * V2**H) (stored in WORK)
				// W := C1**H
				for(size_t j = 0; j < k; ++j){
					BLAS::Copy(lastc, &c[j+0*ldc], ldc, &work[0+j*ldwork], 1);
					BLAS::Conjugate(lastc, &work[0+j*ldwork], 1);
				}
				// W := W * V1**H
				BLAS::MultTrM("R","U","C","U", lastc, k, T(1), v, ldv, work, ldwork);
				if(lastv > k){
					// W := W + C2**H*V2**H
					BLAS::MultMM(
						"C","C", lastc, k, lastv-k, T(1), &c[k+0*ldc], ldc,
						&v[0+(k)*ldv], ldv, T(1), work, ldwork
					);
				}
				// W := W * T**H  or  W * T
				BLAS::MultTrM("R","U",transt,"N", lastc, k, T(1), t, ldt, work, ldwork);
				// C := C - V**H * W**H
				if(lastv > k){
					// C2 := C2 - V2**H * W**H
					BLAS::MultMM(
						"C","C", lastv-k, lastc, k, T(-1), &v[0+(k)*ldv], ldv,
						work, ldwork, T(1), &c[k+0*ldc], ldc
					);
				}
				// W := W * V1
				BLAS::MultTrM("R","U","N","U", lastc, k, T(1), v, ldv, work, ldwork);
				// C1 := C1 - W**H
				for(size_t j = 0; j < k; ++j){
					for(size_t i = 0; i < lastc; ++i){
						c[j+i*ldc] -= Traits<T>::conj(work[i+j*ldwork]);
					}
				}
			}else if('R' == side[0]){
				// Form  C * H  or  C * H**H  where  C = (C1  C2)
				size_t lastv = 1+LastNonzeroColumn(k, n, v, ldv);
				if(k > lastv){ lastv = k; }
				const size_t lastc = 1+LastNonzeroRow(m, lastv, c, ldc);
				// W := C * V**H  =  (C1*V1**H + C2*V2**H)  (stored in WORK)
				// W := C1
				for(size_t j = 0; j < k; ++j){
					BLAS::Copy(lastc, &c[0+j*ldc], 1, &work[0+j*ldwork], 1);
				}
				// W := W * V1**H
				BLAS::MultTrM("R","U","C","U", lastc, k, T(1), v, ldv, work, ldwork);
				if(lastv > k){
					// W := W + C2 * V2**H
					BLAS::MultMM(
						"N","C", lastc, k, lastv-k, T(1), &c[0+(k)*ldc], ldc,
						&v[0+(k)*ldv], ldv, T(1), work, ldwork
					);
				}
				// W := W * T  or  W * T**H
				BLAS::MultTrM("R","U",trans,"N", lastc, k, T(1), t, ldt, work, ldwork);
				// C := C - W * V
				if(lastv > k){
					// C2 := C2 - W * V2
					BLAS::MultMM(
						"N","N", lastc, lastv-k, k, T(-1), work, ldwork,
						&v[0+(k)*ldv], ldv, T(1), &c[0+(k)*ldc], ldc
					);
				}
				// W := W * V1
				BLAS::MultTrM("R","U","N","U", lastc, k, T(1), v, ldv, work, ldwork);
				// C1 := C1 - W
				for(size_t j = 0; j < k; ++j){
					for(size_t i = 0; i < lastc; ++i){
						c[i+j*ldc] -= work[i+j*ldwork];
					}
				}
			}
		}else{
			// Let  V =  (V1  V2)    (V2: last K columns)
			// where  V2  is unit lower triangular.
			if('L' == side[0]){
				// Form  H * C  or  H**H * C  where  C = (C1)
				//                                       (C2)
				const size_t lastc = 1+LastNonzeroColumn(m, n, c, ldc);
				// W := C**H * V**H  =  (C1**H * V1**H + C2**H * V2**H) (stored in WORK)
				// W := C2**H
				for(size_t j = 0; j < k; ++j){
					BLAS::Copy(lastc, &c[m-k+j+0*ldc], ldc, &work[0+j*ldwork], 1);
					BLAS::Conjugate(lastc, &work[0+j*ldwork], 1);
				}
				// W := W * V2**H
				BLAS::MultTrM(
					"R","L","C","U", lastc, k, T(1), &v[0+(m-k)*ldv], ldv, work, ldwork
				);
				if(m > k){
					// W := W + C1**H * V1**H
					BLAS::MultMM(
						"C","C", lastc, k, m-k, T(1), c, ldc, v, ldv, T(1), work, ldwork
					);
				}
				// W := W * T**H  or  W * T
				BLAS::MultTrM("R","L",transt,"N", lastc, k, T(1), t, ldt, work, ldwork);
				// C := C - V**H * W**H
				if(m > k){
					// C1 := C1 - V1**H * W**H
					BLAS::MultMM(
						"C","C", m-k, lastc, k, T(-1), v, ldv, work, ldwork, T(1), c, ldc
					);
				}
				// W := W * V2
				BLAS::MultTrM(
					"R","L","N","U", lastc, k, T(1), &v[0+(m-k)*ldv], ldv, work, ldwork
				);
				// C2 := C2 - W**H
				for(size_t j = 0; j < k; ++j){
					for(size_t i = 0; i < lastc; ++i){
						c[m-k+j+i*ldc] -= Traits<T>::conj(work[i+j*ldwork]);
					}
				}
			}else if('R' == side[0]){
				// Form  C * H  or  C * H**H  where  C = (C1  C2)
				const size_t lastc = 1+LastNonzeroRow(m, n, c, ldc);
				// W := C * V**H  =  (C1*V1**H + C2*V2**H)  (stored in WORK)
				// W := C2
				for(size_t j = 0; j < k; ++j){
					BLAS::Copy(lastc, &c[0+(n-k+j)*ldc], 1, &work[0+j*ldwork], 1);
				}
				// W := W * V2**H
				BLAS::MultTrM(
					"R","L","C","U", lastc, k, T(1), &v[0+(n-k)*ldv], ldv, work, ldwork
				);
				if(n > k){
					// W := W + C1 * V1**H
					BLAS::MultMM(
						"N","C", lastc, k, n-k, T(1), c, ldc, v, ldv, T(1), work, ldwork
					);
				}
				// W := W * T  or  W * T**H
				BLAS::MultTrM("R","L",trans,"N", lastc, k, T(1), t, ldt, work, ldwork);
				// C := C - W * V
				if(n > k){
					// C1 := C1 - W * V1
					BLAS::MultMM(
						"N","N", lastc, n-k, k, T(-1), work, ldwork, v, ldv, T(1), c, ldc
					);
				}
				// W := W * V2
				BLAS::MultTrM(
					"R","L","N","U", lastc, k, T(1), &v[0+(n-k)*ldv], ldv, work, ldwork
				);
				// C1 := C1 - W
				for(size_t j = 0; j < k; ++j){
					for(size_t i = 0; i < lastc; ++i){
						c[i+(n-k+j)*ldc] -= work[i+j*ldwork];
					}
				}
			}
		}
	}
}
 
} // namespace Reflector
} // namespace LA
} // namespace RNP

#endif // RNP_REFLECTOR_HPP_INCLUDED
