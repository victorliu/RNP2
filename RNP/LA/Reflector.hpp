#ifndef RNP_REFLECTOR_HPP_INCLUDED
#define RNP_REFLECTOR_HPP_INCLUDED


///////////////////////////////////////////////////////////////////////
// RNP::LA::Reflector
// ==================
// Computes elementary (Householder) reflectors and their effects.
// Both blocked and unblocked routines are given here.
//

#include <cstddef>
#include <RNP/BLAS.hpp>
#include <RNP/Types.hpp>
#include <RNP/Debug.hpp>
#include <iostream>

namespace RNP{
namespace LA{
namespace Reflector{


///////////////////////////////////////////////////////////////////////
// LastNonzeroColumnLimit
// ----------------------
// Scans a matrix for its last non-zero column. Returns one plus the
// column index (therefore it is the end of a range).
// This corresponds to Lapack routines ila_lc.
//
// Arguments
// m   Number of rows of the matrix.
// n   Number of columns of the matrix.
// a   Pointer to the first element of the matrix.
// lda Leading dimension of the array containing the matrix, lda >= m.
//
template <typename T>
size_t LastNonzeroColumnLimit(size_t m, size_t n, const T *a, size_t lda){
	RNPAssert(lda >= m);
	// if n = 0, returns -1
	if(0 == n || T(0) != a[0+(n-1)*lda] || T(0) != a[m-1+(n-1)*lda]){ return n; }
	size_t j = n;
	while(j --> 0){
		for(size_t i = 0; i < m; ++i){
			if(T(0) != a[i+j*lda]){ return j+1; }
		}
	}
	return j+1; // should be 0
}

///////////////////////////////////////////////////////////////////////
// LastNonzeroRowLimit
// ----------------------
// Scans a matrix for its last non-zero row. Returns one plus the
// row index (therefore it is the end of a range).
// This corresponds to Lapack routines ila_lr.
//
// Arguments
// m   Number of rows of the matrix.
// n   Number of columns of the matrix.
// a   Pointer to the first element of the matrix.
// lda Leading dimension of the array containing the matrix, lda >= m.
//
template <typename T>
size_t LastNonzeroRowLimit(size_t m, size_t n, const T *a, size_t lda){
	RNPAssert(lda >= m);
	// if m = 0, returns -1
	if(0 == m || T(0) != a[m-1+0*lda] || T(0) != a[m-1+(n-1)*lda]){ return m; }
	size_t i = m;
	while(i --> 0){
		for(size_t j = 0; j < n; ++j){
			if(T(0) != a[i+j*lda]){ return i+1; }
		}
	}
	return i+1; // should be 0
}

///////////////////////////////////////////////////////////////////////
// Generate
// --------
// Generates an elementary reflector H of order n, such that
// 
//     H' * [ alpha ] = [ beta ],   H' * H = I.
//          [   x   ]   [   0  ]
// 
// where alpha and beta are scalars, with beta real, and x is an
// (n-1)-element vector. H is represented in the form
// 
//     H' = I - tau * [ 1 ] * [ 1 v' ],
//                    [ v [
// 
// where tau is a scalar and v is an (n-1)-element vector.
// Note that H may be non-Hermitian.
// 
// If the elements of x are all zero and alpha is real, then tau = 0
// and H is taken to be the identity matrix.
// 
// Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1.
// 
// The algorithm is as follows:
//
//     Pick beta = -sign(real(alpha)) * norm([alpha;x])
//     Set tau = (beta - alpha) / beta
//     Set v = x / (alpha - beta)
//
// where rescalings have been left out. This corresponds to Lapack
// routines _larfg.
// 
// Arguments
// n       The order of the elementary reflector.
// alpha   On entry, the value alpha. On exit, it is overwritten
//         with the value beta.
// x       Vector of length n-1. On entry, the vector x.
//         On exit, it is overwritten with the vector v.
// incx    The increment between elements of x.
// tau     The value tau.
//
template <typename T>
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

///////////////////////////////////////////////////////////////////////
// Apply
// -----
// Applies an elementary reflector H to an m-by-n matrix C,
// from either the left or the right. H is represented in the form
//        H = I - tau * v * v'
// where tau is a scalar and v is a vector.
//
// To apply H' (the conjugate transpose of H), supply conj(tau)
// instead.
// This corresponds to Lapack routines _larf.
//
// Arguments
// side    If "L", form  H * C. If "R", form C * H.
// vone    If true, the first element of V is assumed to be 1 instead
//         of the actual input value.
// vconj   If true, elements after the first element of V are assumed
//         to be conjugated.
// m       The number of rows of the matrix C.
// n       The number of columns of the matrix C.
// v       Length m if side = "L" or length n if side = "R"
//         The vector v in the representation of H. v is not used if
//         tau = 0.
// incv    The increment between elements of v, incv > 0.
// tau     The value tau in the representation of H.
// c       On entry, the m-by-n matrix C.
//         On exit, C is overwritten by the matrix H * C if side = "L",
//         or C * H if side = "R".
// ldc     The leading dimension of the array containing C, ldc >= m.
// work    Workspace of size n if side = "L" or size m if side = "R".
//
template <typename T>
void Apply(
	const char *side, int vone, bool vconj, size_t m, size_t n,
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
			lenc = Reflector::LastNonzeroColumnLimit(lenv, n, c, ldc);
		}else{ // Scan for the last non-zero row in C(:,1:lastv).
			lenc = Reflector::LastNonzeroRowLimit(m, lenv, c, ldc);
		}
		if(0 == lenc){ return; }
	}
	
	const T one(1);
	
	if(0 == vone && !vconj){
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
	}else if(1 == vone){ // assume top of vector is 1
		if('L' == side[0]){ // Form  H * C
			// w(1:lastc,1) := C(1:lastv,1:lastc)' * v(1:lastv)
			if(lenv > 1){ // Add the remaining contribution
				if(vconj){
					BLAS::MultMV("T", lenv-1, lenc, T(1), &c[1+0*ldc], ldc, &v[incv], incv, T(0), work, 1);
					// Add in first column
					BLAS::Axpy(lenc, one, c, ldc, work, 1);
					BLAS::Conjugate(lenc, work, 1);
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
	}else{ // assume bottom of vector is 1
		if('L' == side[0]){ // Form  H * C
			// w(1:lastc,1) := C(1:lastv,1:lastc)' * v(1:lastv)
			if(lenv > 1){ // Add the remaining contribution
				if(vconj){
					BLAS::MultMV("T", lenv-1, lenc, T(1), c, ldc, v, incv, T(0), work, 1);
					// Add in last column
					BLAS::Axpy(lenc, one, &c[lenv-1+0*ldc], ldc, work, 1);
					BLAS::Conjugate(lenc, work, 1);
				}else{
					BLAS::Copy(lenc, &c[lenv-1+0*ldc], ldc, work, 1);
					BLAS::Conjugate(lenc, work, 1);
					BLAS::MultMV("C", lenv-1, lenc, T(1), c, ldc, v, incv, T(1), work, 1);
				}
			}else{
				BLAS::Copy(lenc, c, ldc, work, 1);
				BLAS::Conjugate(lenc, work, 1);
			}
			
			// C(1:lastv,1:lastc) := C(...) - v(1:lastv) * w(1:lastc,1)'
			// Add in last row contribution
			BLAS::ConjugateRank1Update(1, lenc, -tau, &one, incv, work, 1, &c[lenv-1+0*ldc], ldc);
			// Add remaining contribution
			if(lenv > 1){
				if(vconj){ // need to do a double-conjugate rank-1 update
					for(size_t j = 0; j < lenc; ++j){
						for(size_t i = 0; i+1 < lenv; ++i){
							c[i+j*ldc] -= tau * Traits<T>::conj(v[i*incv]) * Traits<T>::conj(work[j]);
						}
					}
				}else{
					BLAS::ConjugateRank1Update(lenv-1, lenc, -tau, v, incv, work, 1, c, ldc);
				}
			}
		}else{ // Form  C * H
			// w(1:lastc,1) := C(1:lastc,1:lastv) * v(1:lastv)
			// Add in last column
			BLAS::Copy(lenc, &c[0+(lenv-1)*ldc], 1, work, 1);
			if(lenv > 1){ // Add remaining contribution
				if(vconj){
					for(size_t j = 0; j+1 < lenv; ++j){
						T cvj(Traits<T>::conj(v[j*incv]));
						for(size_t i = 0; i < lenc; ++i){
							work[i] += c[i+j*ldc] * cvj;
						}
					}
				}else{
					BLAS::MultMV("N", lenc, lenv-1, T(1), c, ldc, v, incv, T(1), work, 1);
				}
			}
			// C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv)'
			// Add in last col
			BLAS::Axpy(lenc, -tau, work, 1, &c[0+(lenv-1)*ldc], 1);
			if(lenv > 1){
				if(vconj){
					BLAS::Rank1Update(lenc, lenv-1, -tau, work, 1, v, incv, c, ldc);
				}else{
					BLAS::ConjugateRank1Update(lenc, lenv-1, -tau, work, 1, v, incv, c, ldc);
				}
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////
// GeneratePositive
// ----------------
// Similar to Generate, except that beta is guaranteed to be positive.
// Generates an elementary reflector H of order n, such that
// 
//     H' * [ alpha ] = [ beta ],   H' * H = I.
//          [   x   ]   [   0  ]
// 
// where alpha and beta are scalars, with beta real and positive,
// and x is an (n-1)-element vector. H is represented in the form
// 
//     H' = I - tau * [ 1 ] * [ 1 v' ],
//                    [ v [
// 
// where tau is a scalar and v is an (n-1)-element vector.
// Note that H may be non-Hermitian.
// 
// If the elements of x are all zero and alpha is real, then tau = 0
// and H is taken to be the identity matrix.
// 
// Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1.
// 
// The algorithm is as follows:
//
//     Pick beta = -sign(real(alpha)) * norm([alpha;x])
//     Set tau = (beta - alpha) / beta
//     Set v = x / (alpha - beta)
//
// where rescalings have been left out. This corresponds to Lapack
// routines _larfp. Note that this routine is significantly less
// robust than Generate, so unless positve beta is absolutely required,
// it is recommended to use Generate instead.
// 
// Arguments
// n       The order of the elementary reflector.
// alpha   On entry, the value alpha. On exit, it is overwritten
//         with the value beta.
// x       Vector of length n-1. On entry, the vector x.
//         On exit, it is overwritten with the vector v.
// incx    The increment between elements of x.
// tau     The value tau.
//
template <typename T> // zlarfp, dlarfp, clarfp, slarfp
void GeneratePositive(size_t n, T *alpha, T *x, size_t incx, T *tau){
	typedef typename Traits<T>::real_type real_type;
	
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


///////////////////////////////////////////////////////////////////////
// GenerateBlockTr
// ---------------
// Forms the triangular factor T of a complex block reflector H
// of order n, which is defined as a product of k elementary reflectors.
//
// If DIR = 'F', H = H(1) H(2) . . . H(k) and T is upper triangular;
//
// If DIR = 'B', H = H(k) . . . H(2) H(1) and T is lower triangular.
//
// If STOREV = 'C', the vector which defines the elementary reflector
// H(i) is stored in the i-th column of the array V, and
//
//    H  =  I - V * T * V^H
//
// If STOREV = 'R', the vector which defines the elementary reflector
// H(i) is stored in the i-th row of the array V, and
//
//    H  =  I - V^H * T * V
//
// ### Further Details
//
// The shape of the matrix V and the storage of the vectors which define
// the H(i) is best illustrated by the following example with n = 5 and
// k = 3. The elements equal to 1 are not stored; the corresponding
// array elements are modified but restored on exit. The rest of the
// array is not used.
//
//     dir = "F" and storev = "C":        dir = "F" and storev = "R":
//
//              V = [  1       ]             V = [  1 v1 v1 v1 v1 ]
//                  [ v1  1    ]                 [     1 v2 v2 v2 ]
//                  [ v1 v2  1 ]                 [        1 v3 v3 ]
//                  [ v1 v2 v3 ]
//                  [ v1 v2 v3 ]
//
//     dir = "B" and storev = "C":        dir = "B" and storev = "R":
//
//              V = [ v1 v2 v3 ]             V = [ v1 v1  1       ]
//                  [ v1 v2 v3 ]                 [ v2 v2 v2  1    ]
//                  [  1 v2 v3 ]                 [ v3 v3 v3 v3  1 ]
//                  [     1 v3 ]
//                  [        1 ]
//
// Arguments
// dir     Specifies the order in which the elementary reflectors are
//         multiplied to form the block reflector.
//         If "F", H = H[0] H[1] ... H[k-1] (Forward).
//         If "B", H = H[k-1] ... H[1] H[0] (Backward).
// storev  Specifies how the vectors which define the elementary
//         reflectors are stored (see also Further Details):
//         If "C": columnwise. If "R": rowwise.
// n       The order of the block reflector H.
// k       The order of the triangular factor T (the number of
//         elementary reflectors), k >= 1.
// v       Pointer to the first element of the matrix V.
//         If storev = "C", V has dimensions n-by-k.
//         If storev = "R", V has dimensions k-by-n.
// ldv     The leading dimension of the array containing V.
//         If storev = "C", ldv >= n. If storev = "R", ldv >= k.
// tau     tau[i] must contain the scalar factor of the elementary
//         reflector H[i], length k.
// t       The k-by-k triangular factor T of the block reflector.
//         If dir = "F", T is upper triangular; if dir = "B", T is
//         lower triangular. The rest of the array is not used.
// ldt     The leading dimension of the array containing T, ldt >= k.
//
template <typename T>
void GenerateBlockTr(
	const char *dir, const char *storev,
	size_t n, size_t k, const T *v, size_t ldv, const T *tau,
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
				if('C' == storev[0]){ // v should be n-by-k
					for(lastv = n-1; lastv > i; --lastv){ // skip trailing zeros
						if(T(0) != v[lastv+i*ldv]){ break; }
					}
					for(size_t j = 0; j < i; ++j){
						t[j+i*ldt] = -tau[i] * Traits<T>::conj(v[i+j*ldv]);
					}
					size_t jlim = (lastv < prevlastv ? lastv : prevlastv);
					// T(1:i-1,i) := - tau(i) * V(i:j,1:i-1)^H * V(i:j,i)
					BLAS::MultMV(
						"C", jlim-i, i, -tau[i], &v[i+1+0*ldv], ldv,
						&v[i+1+i*ldv], 1, T(1), &t[0+i*ldt], 1
					);
				}else{ // v should be k-by-n
					for(lastv = n-1; lastv > i; --lastv){ // skip trailing zeros
						if(T(0) != v[i+lastv*ldv]){ break; }
					}
					for(size_t j = 0; j < i; ++j){
						t[j+i*ldt] = -tau[i] * v[j+i*ldv];
					}
					size_t jlim = (lastv < prevlastv ? lastv : prevlastv);
					// T(1:i-1,i) := - tau(i) * V(1:i-1,i:j) * V(i,i:j)^H
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
						// T(i+1:k,i) = -tau(i) * V(j:n-k+i,i+1:k)^H * V(j:n-k+i,i)
						BLAS::MultMV(
							"C", n-k+i-jlim, k-1-i, -tau[i], &v[jlim+(i+1)*ldv], ldv,
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
						// T(i+1:k,i) = -tau(i) * V(i+1:k,j:n-k+i) * V(i,j:n-k+i)^H
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

///////////////////////////////////////////////////////////////////////
// ApplyBlock
// ----------
// Applies a block reflector H or its conjugate transpose H^H to an
// m-by-n matrix C, from either the left or the right. See the
// documentation for GenerateBlockTr for how the vectors are packed
// into the V matrix.
//
// Arguments
// =========
//
// side    If "L", apply H or H^H from the left.
//         If "R", apply H or H^H from the right.
// trans   If "N", apply H (No transpose)
//         If "C", apply H^H (Conjugate transpose)
// dir     Indicates how H is formed from a product of elementary
//         reflectors.
//         If "F", H = H[0] H[1] ... H[k-1] (Forward).
//         If "B", H = H[k-1] ... H[1] H[0] (Backward).
// storev  Indicates how the vectors which define the elementary
//         reflectors are stored:
//         If "C", columnwise. If "R", rowwise.
// m       The number of rows of the matrix C.
// n       The number of columns of the matrix C.
// k       The order of the matrix T (the number of elementary
//         reflectors whose product defines the block reflector).
// v       Pointer to the first element of the matrix V.
//         V has k columns if storev = "C",
//               m columns if storev = "R" and side = "L",
//            or n columns if storev = "R" and side = "R"
// ldv     The leading dimension of the array containing V.
//         If storev = "C" and side = "L", ldv >= m.
//         if storev = "C" and side = "R", ldv >= n.
//         if storev = "R", ldv >= k.
// t       The triangular k-by-k matrix T in the representation of the
//         block reflector.
// ldt     The leading dimension of the array T, ldt >= k.
// c       Pointer to the first element of C. On entry, the m-by-n
//         matrix C. On exit, C is overwritten by H * C or H^H * C
//         or C * H or C * H^H.
// ldc     The leading dimension of the array containing C, ldc >= m.
// work    Workspace of dimension ldwork-by-k.
// ldwork  The leading dimension of the array containing work.
//         If side = 'L', ldwork >= max(1,N);
//         if side = 'R', ldwork >= max(1,M).
//
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
				// Form  H * C  or  H^H * C  where  C = ( C1 )
				//                                      ( C2 )
				size_t lastv = LastNonzeroRowLimit(m, k, v, ldv);
				if(k > lastv){ lastv = k; }
				const size_t lastc = LastNonzeroColumnLimit(lastv, n, c, ldc);
				// W := C^H * V  =  (C1^H * V1 + C2^H * V2)  (stored in WORK)
				// W := C1^H
				for(size_t j = 0; j < k; ++j){
					BLAS::Copy(lastc, &c[j+0*ldc], ldc, &work[0+j*ldwork], 1);
					BLAS::Conjugate(lastc, &work[0+j*ldwork], 1);
				}
				// W := W * V1
				BLAS::MultTrM("R","L","N","U", lastc, k, T(1), v, ldv, work, ldwork);
				if(lastv > k){
					// W := W + C2^H *V2
					BLAS::MultMM(
						"C","N", lastc, k, lastv-k, T(1), &c[k+0*ldc], ldc,
						&v[k+0*ldv], ldv, T(1), work, ldwork
					);
				}
				// W := W * T^H  or  W * T
				BLAS::MultTrM("R","U",transt,"N", lastc, k, T(1), t, ldt, work, ldwork);
				// C := C - V * W^H
				if(m > k){
					// C2 := C2 - V2 * W^H
					BLAS::MultMM(
						"N","C", lastv-k, lastc, k, T(-1), &v[k+0*ldv], ldv,
						work, ldwork, T(1), &c[k+0*ldc], ldc
					);
				}
				// W := W * V1^H
				BLAS::MultTrM("R","L","C","U", lastc, k, T(1), v, ldv, work, ldwork);
				// C1 := C1 - W^H
				for(size_t j = 0; j < k; ++j){
					for(size_t i = 0; i < lastc; ++i){
						c[j+i*ldc] -= Traits<T>::conj(work[i+j*ldwork]);
					}
				}
			}else if('R' == side[0]){
				// Form  C * H  or  C * H^H  where  C = ( C1  C2 )
				size_t lastv = LastNonzeroRowLimit(n, k, v, ldv);
				if(k > lastv){ lastv = k; }
				const size_t lastc = LastNonzeroRowLimit(m, lastv, c, ldc);
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
				// W := W * T  or  W * T^H
				BLAS::MultTrM("R","U",trans,"N", lastc, k, T(1), t, ldt, work, ldwork);
				// C := C - W * V^H
				if(lastv > k){
					// C2 := C2 - W * V2^H
					BLAS::MultMM(
						"N","C", lastc, lastv-k, k, T(-1), work, ldwork,
						&v[k+0*ldv], ldv, T(1), &c[0+(k)*ldc], ldc
					);
				}
				// W := W * V1^H
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
				// Form  H * C  or  H^H * C  where  C = ( C1 )
				//                                      ( C2 )
				const size_t lastc = LastNonzeroColumnLimit(m, n, c, ldc);
				// W := C^H * V  =  (C1^H * V1 + C2^H * V2)  (stored in WORK)
				// W := C2^H
				for(size_t j = 0; j < k; ++j){
					BLAS::Copy(lastc, &c[m-k+j+0*ldc], ldc, &work[0+j*ldwork], 1);
					BLAS::Conjugate(lastc, &work[0+j*ldwork], 1);
				}
				// W := W * V2
				BLAS::MultTrM(
					"R","U","N","U", lastc, k, T(1), &v[m-k+0*ldv], ldv, work, ldwork
				);
				if(m > k){
					// W := W + C1^H*V1
					BLAS::MultMM(
						"C","N", lastc, k, m-k, T(1), c, ldc,
						v, ldv, T(1), work, ldwork
					);
				}
				// W := W * T^H  or  W * T
				BLAS::MultTrM("R","L",transt,"N", lastc, k, T(1), t, ldt, work, ldwork);
				// C := C - V * W^H
				if(m > k){
					// C1 := C1 - V1 * W^H
					BLAS::MultMM(
						"N","C", m-k, lastc, k, T(-1), v, ldv,
						work, ldwork, T(1), c, ldc
					);
				}
				// W := W * V2^H
				BLAS::MultTrM(
					"R","U","C","U", lastc, k, T(1), &v[m-k+0*ldv], ldv, work, ldwork
				);
				// C2 := C2 - W^H
				for(size_t j = 0; j < k; ++j){
					for(size_t i = 0; i < lastc; ++i){
						c[m-k+j+i*ldc] -= Traits<T>::conj(work[i+j*ldwork]);
					}
				}
			}else if('R' == side[0]){
				// Form  C * H  or  C * H^H  where  C = (C1  C2)
				const size_t lastc = LastNonzeroRowLimit(m, n, c, ldc);
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
				// W := W * T  or  W * T^H
				BLAS::MultTrM("R","L",trans,"N", lastc, k, T(1), t, ldt, work, ldwork);
				// C := C - W * V^H
				if(n > k){
					// C1 := C1 - W * V1^H
					BLAS::MultMM(
						"N","C", lastc, n-k, k, T(-1), work, ldwork,
						v, ldv, T(1), c, ldc
					);
				}
				// W := W * V2^H
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
				// Form  H * C  or  H^H * C  where  C = (C1)
				//                                      (C2)
				size_t lastv = LastNonzeroColumnLimit(k, m, v, ldv);
				if(k > lastv){ lastv = k; }
				const size_t lastc = LastNonzeroColumnLimit(lastv, n, c, ldc);
				// W := C^H * V^H  =  (C1^H * V1^H + C2^H * V2^H) (stored in WORK)
				// W := C1^H
				for(size_t j = 0; j < k; ++j){
					BLAS::Copy(lastc, &c[j+0*ldc], ldc, &work[0+j*ldwork], 1);
					BLAS::Conjugate(lastc, &work[0+j*ldwork], 1);
				}
				// W := W * V1^H
				BLAS::MultTrM("R","U","C","U", lastc, k, T(1), v, ldv, work, ldwork);
				if(lastv > k){
					// W := W + C2^H*V2^H
					BLAS::MultMM(
						"C","C", lastc, k, lastv-k, T(1), &c[k+0*ldc], ldc,
						&v[0+(k)*ldv], ldv, T(1), work, ldwork
					);
				}
				// W := W * T^H  or  W * T
				BLAS::MultTrM("R","U",transt,"N", lastc, k, T(1), t, ldt, work, ldwork);
				// C := C - V^H * W^H
				if(lastv > k){
					// C2 := C2 - V2^H * W^H
					BLAS::MultMM(
						"C","C", lastv-k, lastc, k, T(-1), &v[0+(k)*ldv], ldv,
						work, ldwork, T(1), &c[k+0*ldc], ldc
					);
				}
				// W := W * V1
				BLAS::MultTrM("R","U","N","U", lastc, k, T(1), v, ldv, work, ldwork);
				// C1 := C1 - W^H
				for(size_t j = 0; j < k; ++j){
					for(size_t i = 0; i < lastc; ++i){
						c[j+i*ldc] -= Traits<T>::conj(work[i+j*ldwork]);
					}
				}
			}else if('R' == side[0]){
				// Form  C * H  or  C * H^H  where  C = (C1  C2)
				size_t lastv = LastNonzeroColumnLimit(k, n, v, ldv);
				if(k > lastv){ lastv = k; }
				const size_t lastc = LastNonzeroRowLimit(m, lastv, c, ldc);
				// W := C * V^H  =  (C1*V1^H + C2*V2^H)  (stored in WORK)
				// W := C1
				for(size_t j = 0; j < k; ++j){
					BLAS::Copy(lastc, &c[0+j*ldc], 1, &work[0+j*ldwork], 1);
				}
				// W := W * V1^H
				BLAS::MultTrM("R","U","C","U", lastc, k, T(1), v, ldv, work, ldwork);
				if(lastv > k){
					// W := W + C2 * V2^H
					BLAS::MultMM(
						"N","C", lastc, k, lastv-k, T(1), &c[0+(k)*ldc], ldc,
						&v[0+(k)*ldv], ldv, T(1), work, ldwork
					);
				}
				// W := W * T  or  W * T^H
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
				// Form  H * C  or  H^H * C  where  C = (C1)
				//                                      (C2)
				const size_t lastc = LastNonzeroColumnLimit(m, n, c, ldc);
				// W := C^H * V^H  =  (C1^H * V1^H + C2^H * V2^H) (stored in WORK)
				// W := C2^H
				for(size_t j = 0; j < k; ++j){
					BLAS::Copy(lastc, &c[m-k+j+0*ldc], ldc, &work[0+j*ldwork], 1);
					BLAS::Conjugate(lastc, &work[0+j*ldwork], 1);
				}
				// W := W * V2^H
				BLAS::MultTrM(
					"R","L","C","U", lastc, k, T(1), &v[0+(m-k)*ldv], ldv, work, ldwork
				);
				if(m > k){
					// W := W + C1^H * V1^H
					BLAS::MultMM(
						"C","C", lastc, k, m-k, T(1), c, ldc, v, ldv, T(1), work, ldwork
					);
				}
				// W := W * T^H  or  W * T
				BLAS::MultTrM("R","L",transt,"N", lastc, k, T(1), t, ldt, work, ldwork);
				// C := C - V^H * W^H
				if(m > k){
					// C1 := C1 - V1^H * W^H
					BLAS::MultMM(
						"C","C", m-k, lastc, k, T(-1), v, ldv, work, ldwork, T(1), c, ldc
					);
				}
				// W := W * V2
				BLAS::MultTrM(
					"R","L","N","U", lastc, k, T(1), &v[0+(m-k)*ldv], ldv, work, ldwork
				);
				// C2 := C2 - W^H
				for(size_t j = 0; j < k; ++j){
					for(size_t i = 0; i < lastc; ++i){
						c[m-k+j+i*ldc] -= Traits<T>::conj(work[i+j*ldwork]);
					}
				}
			}else if('R' == side[0]){
				// Form  C * H  or  C * H^H  where  C = (C1  C2)
				const size_t lastc = LastNonzeroRowLimit(m, n, c, ldc);
				// W := C * V^H  =  (C1*V1^H + C2*V2^H)  (stored in WORK)
				// W := C2
				for(size_t j = 0; j < k; ++j){
					BLAS::Copy(lastc, &c[0+(n-k+j)*ldc], 1, &work[0+j*ldwork], 1);
				}
				// W := W * V2^H
				BLAS::MultTrM(
					"R","L","C","U", lastc, k, T(1), &v[0+(n-k)*ldv], ldv, work, ldwork
				);
				if(n > k){
					// W := W + C1 * V1^H
					BLAS::MultMM(
						"N","C", lastc, k, n-k, T(1), c, ldc, v, ldv, T(1), work, ldwork
					);
				}
				// W := W * T  or  W * T^H
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
