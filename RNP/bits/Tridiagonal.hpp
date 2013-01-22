#ifndef RNP_TRIDIAGONAL_HPP_INCLUDED
#define RNP_TRIDIAGONAL_HPP_INCLUDED

#include <cstddef>
#include <RNP/bits/Reflector.hpp>
#include <RNP/bits/Rotation.hpp>
#include <RNP/BLAS.hpp>
#include <RNP/Debug.hpp>

namespace RNP{
namespace LA{
namespace Tridiagonal{


///////////////////////////////////////////////////////////////////////
// RNP::LA::Tridiagonal
// ====================
// Routines dealing with tridiagonal matrices and reduction of square
// matrices to tridiagonal form.
//

///////////////////////////////////////////////////////////////////////
// Tuning
// ------
// Specialize this class to tune the block sizes. The optimal block
// size should be greater than or equal to the minimum block size.
// The value of the crossover determines when to enable blocking.
//
template <typename T>
struct Tuning{
	static inline size_t reduce_herm_block_size_opt(const char *uplo, size_t n){ return 32; }
	static inline size_t reduce_herm_block_size_min(const char *uplo, size_t n){ return 2; }
	static inline size_t reduce_herm_crossover_size(const char *uplo, size_t n){ return 128; }
};

///////////////////////////////////////////////////////////////////////
// NormSym
// -------
// Returns the value of the 1-norm, Frobenius norm, infinity norm, or
// the  element of largest absolute value of a symmetric tridiagonal
// matrix A. Note that the maximum element magnitude is not a
// consistent matrix norm.
// Equivalent to Lapack routines _lanst.
//
// Arguments
// norm    Specifies which norm to return:
//           If norm = "M", returns max(abs(A[i,j])).
//           If norm = "1" or "O", returns norm1(A) (max column sum).
//           If norm = "I", returns normInf(A) (max row sum).
//           If norm = "F" or "E", returns normFrob(A).
// n       Number of rows and columns of the matrix A.
// diag    Pointer to the diagonal elements of A (length n).
// offdiag Pointer to the off-diagonal elements of A (length n-1).
//
template <typename T>
typename Traits<T>::real_type NormSym(
	const char *norm, size_t n, const T *diag, const T *offdiag
){
	typedef typename Traits<T>::real_type real_type;
	real_type result(0);
	if(0 == n){ return result; }
	if('M' == norm[0]){ // max(abs(A(i,j)))
		result = Traits<T>::abs(diag[n-1]);
		for(size_t i = 0; i+1 < n; ++i){
			real_type ai = Traits<T>::abs(diag[i]);
			if(!(ai < result)){ result = ai; }
			ai = Traits<T>::abs(offdiag[i]);
			if(!(ai < result)){ result = ai; }
		}
	}else if('O' == norm[0] || '1' == norm[0] || 'I'){ // max col sum or row sum
		if(1 == n){ return Traits<T>::abs(diag[0]); }
		result = Traits<T>::abs(diag[0]) + Traits<T>::abs(offdiag[0]);
		real_type sum(Traits<T>::abs(diag[n-1]) + Traits<T>::abs(offdiag[n-2]));
		if(!(sum < result)){ result = sum; }
		for(size_t i = 1; i+1 < n; ++i){
			sum = Traits<T>::abs(diag[i]) + Traits<T>::abs(offdiag[i-1]) + Traits<T>::abs(offdiag[i]);
			if(!(sum < result)){ result = sum; }
		}
	}else if('F' == norm[0] || 'E' == norm[0]){ // Frobenius norm
		real_type scale(0);
		real_type sum(1);
		for(size_t i = 0; i+1 < n; ++i){
			real_type ca = Traits<T>::abs(offdiag[i]);
			if(scale < ca){
				real_type r = scale/ca;
				sum = real_type(1) + sum*r*r;
				scale = ca;
			}else{
				real_type r = ca/scale;
				sum += r*r;
			}
		}
		for(size_t i = 0; i < n; ++i){
			real_type ca = Traits<T>::abs(diag[i]);
			if(scale < ca){
				real_type r = scale/ca;
				sum = real_type(1) + sum*r*r;
				scale = ca;
			}else{
				real_type r = ca/scale;
				sum += r*r;
			}
		}
		result = scale*sqrt(sum);
	}
	return result;
}


namespace Util{

///////////////////////////////////////////////////////////////////////
// Util::ReduceHerm_block
// ----------------------
// Reduces a set of rows and columns of a Hermitian matrix to
// tridiagonal form, and returns the block matrices needed to transform
// the remaining portion of the matrix. This is a utility routine
// equivalent to Lapack routines _latrd. No further explanation will
// be given.
//
template <typename T>
void ReduceHerm_block(
	const char *uplo, size_t n, size_t nb, T *a, size_t lda,
	typename Traits<T>::real_type *offdiag,
	T *tau, // length n-1
	T *w, size_t ldw
){
	typedef typename Traits<T>::real_type real_type;
	if(0 == n){ return; }
	if('U' == uplo[0]){ // reduce the last nb columns of upper triangle
		size_t i = n;
		while(i --> n-nb){
			size_t iw = i-n+nb;
			if(i+1 < n){ // Update A[0..i,i]
				a[i+i*lda] = Traits<T>::real(a[i+i*lda]);
				BLAS::Conjugate(n-i-1, &w[i+(iw+1)*ldw], ldw);
				BLAS::MultMV(
					"N", i+1, n-i-1, T(-1), &a[0+(i+1)*lda], lda,
					&w[i+(iw+1)*ldw], ldw, T(1), &a[0+i*lda], 1
				);
				BLAS::Conjugate(n-i-1, &w[i+(iw+1)*ldw], ldw);
				BLAS::Conjugate(n-i-1, &a[i+(i+1)*lda], lda);
				BLAS::MultMV(
					"N", i+1, n-i-1, T(-1), &w[0+(iw+1)*ldw], ldw,
					&a[i+(i+1)*ldw], lda, T(1), &a[0+i*lda], 1
				);
				BLAS::Conjugate(n-i-1, &a[i+(i+1)*lda], lda);
				a[i+i*lda] = Traits<T>::real(a[i+i*lda]);
			}
			if(i > 0){ // generate reflector to annihilate A[0..i-1,i]
				T alpha = a[i-1+i*lda];
				Reflector::Generate(i, &alpha, &a[0+i*lda], 1, &tau[i-1]);
				offdiag[i-1] = Traits<T>::real(alpha);
				a[i-1+i*lda] = T(1);
				// Compute W[0..i,i]
				BLAS::MultHermV(
					"U", i, real_type(1), a, lda, &a[0+i*lda], 1,
					real_type(0), &w[0+iw*ldw], 1
				);
				if(i+1 < n){
					BLAS::MultMV(
						"C", i, n-i-1, T(1), &w[0+(iw+1)*ldw], ldw,
						&a[0+i*lda], 1, T(0), &w[i+1+iw*ldw], 1
					);
					BLAS::MultMV(
						"N", i, n-i-1, T(-1), &a[0+(i+1)*lda], lda,
						&w[i+1+iw*ldw], 1, T(1), &w[0+iw*ldw], 1
					);
					BLAS::MultMV(
						"C", i, n-i-1, T(1), &a[0+(i+1)*lda], lda,
						&a[0+i*lda], 1, T(0), &w[i+1+iw*ldw], 1
					);
					BLAS::MultMV(
						"N", i, n-i-1, T(-1), &w[0+(iw+1)*ldw], ldw,
						&w[i+1+iw*ldw], 1, T(1), &w[0+iw*ldw], 1
					);
				}
				BLAS::Scale(i, tau[i-1], &w[0+iw*ldw], 1);
				alpha = -(real_type(1)/real_type(2)) * tau[i-1] *
					BLAS::ConjugateDot(i, &w[0+iw*ldw], 1, &a[0+i*lda], 1);
				BLAS::Axpy(i, alpha, &a[0+i*lda], 1, &w[0+iw*ldw], 1);
			}
		}
	}else{ // reduce the first nb columns of lower triangle
		for(size_t i = 0; i < nb; ++i){
			// Update A[i..n,i]
			a[i+i*lda] = Traits<T>::real(a[i+i*lda]);
			BLAS::Conjugate(i, &w[i+0*lda], ldw);
			BLAS::MultMV(
				"N", n-i, i, T(-1), &a[i+0*lda], lda, &w[i+0*ldw], ldw,
				T(1), &a[i+i*lda], 1
			);
			BLAS::Conjugate(i, &w[i+0*lda], ldw);
			BLAS::Conjugate(i, &a[i+0*lda], lda);
			BLAS::MultMV(
				"N", n-i, i, T(-1), &w[i+0*lda], ldw, &a[i+0*lda], lda,
				T(1), &a[i+i*lda], 1
			);
			BLAS::Conjugate(i, &a[i+0*lda], lda);
			a[i+i*lda] = Traits<T>::real(a[i+i*lda]);
			if(i+1 < n){
				// Generate reflector to annihilate A[i+2..n,i]
				T alpha(a[i+1+i*lda]);
				const size_t row = (i+3 < n ? i+2 : n-1);
				Reflector::Generate(n-i-1, &alpha, &a[row+i*lda], 1, &tau[i]);
				offdiag[i] = Traits<T>::real(alpha);
				a[i+1+i*lda] = T(1);
				// Compute W[i+1..n,i]
				BLAS::MultHermV(
					"L", n-i-1, real_type(1), &a[i+1+(i+1)*lda], lda, &a[i+1+i*lda], 1,
					real_type(0), &w[i+1+i*ldw], 1
				);
				BLAS::MultMV(
					"C", n-i-1, i, T(1), &w[i+1+0*ldw], ldw, &a[i+1+i*lda], 1,
					T(0), &w[0+i*ldw], 1
				);
				BLAS::MultMV("N", n-i-1, i, T(-1), &a[i+1+0*lda], lda,
					&w[0+i*ldw], 1, T(1), &w[i+1+i*ldw], 1
				);
				BLAS::MultMV(
					"C", n-i-1, i, T(1), &a[i+1+0*lda], lda, &a[i+1+i*lda], 1,
					T(0), &w[0+i*ldw], 1
				);
				BLAS::MultMV(
					"N", n-i-1, i, T(-1), &w[i+1+0*ldw], ldw, &w[0+i*ldw], 1,
					T(1), &w[i+1+i*ldw], 1);
				BLAS::Scale(n-i-1, tau[i], &w[i+1+i*ldw], 1);
				alpha = -(real_type(1)/real_type(2)) * tau[i] *
					BLAS::ConjugateDot(n-i-1, &w[i+1+i*ldw], 1, &a[i+1+i*lda], 1);
				BLAS::Axpy(n-i-1, alpha, &a[i+1+i*lda], 1, &w[i+1+i*ldw], 1);
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////
// Util::SymmetricEigensystem2
// ---------------------------
// Computes the eigendecomposition of a 2-by-2 symmetric matrix:
//     [ a b ]
//     [ b c ]
// The eigen decomposition is given by:
//     [ cs1 sn1 ] [ a b ] [ cs1 -sn1 ] = [ rt1  0  ]
//     [-sn1 cs1 ] [ b c ] [ sn1  cs1 ]   [  0  rt2 ]
// where abs(rt1) > abs(rt2). Note that this routine only operates on
// real number types.
//
// Arguments
// a   The [0,0] element of the matrix.
// b   The [0,1] and [1,0] elements of the matrix.
// c   The [1,1] elements of the matrix.
// rt1 The returned eigenvalue of larger absolute value.
// rt2 The returned eigenvalue of smaller absolute value.
// cs1 The first element of the right eigenvector for rt1. If NULL,
//     then the eigenvector is not returned (sn1 is not referenced).
// sn1 The second element of the right eigenvector for rt1. If NULL,
//     then the eigenvector is not returned (cs1 is not referenced).
//
template <typename T> // T must be a real type
void SymmetricEigensystem2(
	const T &a, const T &b, const T &c, T *rt1, T *rt2, T *cs1, T *sn1
){
	RNPAssert(!Traits<T>::is_complex()); // T must be a real type
	typedef T real_type;
	
	T sm(a+c);
	T df(a-c);
	real_type adf(Traits<T>::abs(df));
	T b2(b+b);
	real_type ab(Traits<T>::abs(b2));
	T acmx, acmn;
	if(Traits<T>::abs(a) > Traits<T>::abs(c)){
		acmx = a; acmn = c;
	}else{
		acmx = c; acmn = a;
	}
	// Compute the square root
	T rt(0);
	if(adf > ab){
		rt = adf*sqrt(T(1) + (ab/adf)*(ab/adf));
	}else if(adf < ab){
		rt = ab*sqrt(T(1) + (adf/ab)*(adf/ab));
	}else{
		rt = ab*sqrt(T(2));
	}
	// Compute eigenvalues
	int sgn1;
	if(T(0) == sm){
		*rt1 = (T(1)/T(2)) * rt;
		*rt2 = -(*rt1);
		sgn1 = 1;
	}else{
		if(sm < T(0)){
			*rt1 = (T(1)/T(2)) * (sm-rt);
			sgn1 = -1;
		}else{ // sm > 0
			*rt1 = (T(1)/T(2)) * (sm+rt);
			sgn1 = 1;
		}
		// Order of execution important.
		// To get fully accurate smaller eigenvalue,
		// next line needs to be executed in higher precision.
		*rt2 = (acmx / (*rt1)) * acmn - (b/(*rt1)) * b;
	}
	if(NULL == cs1 || NULL == sn1){ return; }
	// Compute eigenvector
	int sgn2;
	T cs;
	if(df >= T(0)){
		cs = df + rt;
		sgn2 = 1;
	}else{
		cs = df - rt;
		sgn2 = -1;
	}
	real_type acs = Traits<T>::abs(cs);
	if(acs > ab){
		T ct = -b2/cs;
		*sn1 = T(1) / sqrt(T(1) + ct*ct);
		*cs1 = ct*(*sn1);
	}else{
		if(real_type(0) == ab){
			*cs1 = T(1);
			*sn1 = T(0);
		}else{
			T tn = -cs/b2;
			*cs1 = T(1) / sqrt(T(1) + tn*tn);
			*sn1 = tn*(*cs1);
		}
	}
	if(sgn1 == sgn2){
		T tn = *cs1;
		*cs1 = -(*sn1);
		*sn1 = tn;
	}
}

///////////////////////////////////////////////////////////////////////
// Util::SymmetricQLIteration
// --------------------------
// Performs one pass of the QL iteration for a symmetric tridiagonal
// matrix. This is one part of the original Lapack routine _steqr. No
// further explanation will be given.
//
template <typename T>
void SymmetricQLIteration(
	size_t n, size_t m, size_t *pl, size_t lend, size_t *jtot, size_t nmaxit,
	typename Traits<T>::real_type *diag,
	typename Traits<T>::real_type *offdiag,
	T *z, size_t ldz,
	typename Traits<T>::real_type *cv,
	typename Traits<T>::real_type *sv
){
	typedef typename Traits<T>::real_type real_type;
	static const real_type eps(Traits<real_type>::eps());
	static const real_type eps2(eps*eps);
	static const real_type safmin(Traits<real_type>::min());
	
	size_t l = *pl;
	do{
		bool found_small = false;
		if(l != lend){
			for(m = l; m < lend; ++m){
				real_type tst(Traits<T>::abs(offdiag[m]));
				if(tst*tst <= (eps2*Traits<T>::abs(diag[m])) * Traits<T>::abs(diag[m+1])+safmin){
					found_small = true;
					break;
				}
			}
		}
		if(!found_small){
			m = lend;
		}
		if(m < lend){
			offdiag[m] = real_type(0);
		}
		real_type p = diag[l];
		if(m != l){
			// If remaining matrix is 2-by-2, special case it
			if(m == l+1){
				real_type rt1, rt2;
				if(NULL != z){
					LA::Tridiagonal::Util::SymmetricEigensystem2(
						diag[l], offdiag[l], diag[l+1], &rt1, &rt2, &cv[l], &sv[l]
					);
					LA::Rotation::ApplySequence(
						"R","V","B", n, 2, &cv[l], &sv[l], &z[0+l*ldz], ldz
					);
				}else{
					LA::Tridiagonal::Util::SymmetricEigensystem2(
						diag[l], offdiag[l], diag[l+1], &rt1, &rt2, (real_type*)NULL, (real_type*)NULL
					);
				}
				diag[l] = rt1;
				diag[l+1] = rt2;
				offdiag[l] = real_type(0);
				l += 2;
				if(l <= lend){
					continue;
				}
				break;
			}

			if(*jtot == nmaxit){
				break;
			}
			++(*jtot);
			// Form shift.

			real_type g = (diag[l+1]-p) / (real_type(2)*offdiag[l]);
			real_type r = Traits<real_type>::hypot2(g, real_type(1));
			g = diag[m] - p + (offdiag[l] / (g+(g > 0 ? r : -r)));
			{
				real_type s(1);
				real_type c(1);
				p = real_type(0);

				// Inner loop
				for(size_t i = m-1; i+1 >= l+1; --i){ // +1's needed here
					real_type f = s*offdiag[i];
					real_type b = c*offdiag[i];
					LA::Rotation::Generate(g, f, &c, &s, &r);
					if(i+1 != m){ offdiag[i+1] = r; }
					g = diag[i+1] - p;
					r = (diag[i]-g)*s + real_type(2)*c*b;
					p = s*r;
					diag[i+1] = g + p;
					g = c*r - b;
					// If eigenvectors are desired, then save rotations.
					if(NULL != z){
						cv[i] = c;
						sv[i] = -s;
					}
				}
			}
			// If eigenvectors are desired, then apply saved rotations.
			if(NULL != z){
				LA::Rotation::ApplySequence(
					"R","V","B", n, m-l+1, &cv[l], &sv[l], &z[0+l*ldz], ldz
				);
			}

			diag[l] -= p;
			offdiag[l] = g;
			continue;
		}
		// Eigenvalue found.
		diag[l] = p;

		++l;
	}while(l <= lend);
	*pl = l;
}

///////////////////////////////////////////////////////////////////////
// Util::SymmetricQRIteration
// --------------------------
// Performs one pass of the QR iteration for a symmetric tridiagonal
// matrix. This is one part of the original Lapack routine _steqr. No
// further explanation will be given.
//
template <typename T>
void SymmetricQRIteration(
	size_t n, size_t m, size_t *pl, size_t lend, size_t *jtot, size_t nmaxit,
	typename Traits<T>::real_type *diag,
	typename Traits<T>::real_type *offdiag,
	T *z, size_t ldz,
	typename Traits<T>::real_type *cv,
	typename Traits<T>::real_type *sv
){
	typedef typename Traits<T>::real_type real_type;
	static const real_type eps(Traits<real_type>::eps());
	static const real_type eps2(eps*eps);
	static const real_type safmin(Traits<real_type>::min());

	size_t l = *pl;
	do{
		bool found_small = false;
		if(l != lend){
			for(m = l; m > lend; --m){
				real_type tst = Traits<T>::abs(offdiag[m-1]);
				if(tst*tst <= (eps2*Traits<T>::abs(diag[m])) * Traits<T>::abs(diag[m-1])+safmin){
					found_small = true;
					break;
				}
			}
		}
		if(!found_small){
			m = lend;
		}

		if(m > lend){
			offdiag[m-1] = real_type(0);
		}
		real_type p = diag[l];
		if(m != l){	
			// If remaining matrix is 2-by-2, special case it
			if(m+1 == l){
				real_type rt1, rt2;
				if(NULL != z){
					LA::Tridiagonal::Util::SymmetricEigensystem2(
						diag[l-1], offdiag[l-1], diag[l], &rt1, &rt2, &cv[m], &sv[m]
					);
					LA::Rotation::ApplySequence(
						"R","V","F", n, 2, &cv[m], &sv[m], &z[0+(l-1)*ldz], ldz
					);
				}else{
					LA::Tridiagonal::Util::SymmetricEigensystem2(
						diag[l-1], offdiag[l-1], diag[l], &rt1, &rt2, (real_type*)NULL, (real_type*)NULL
					);
				}
				diag[l-1] = rt1;
				diag[l] = rt2;
				offdiag[l-1] = real_type(0);
				l -= 2;
				if((l+1) >= (lend+1)){
					continue;
				}
				break;
			}

			if(*jtot == nmaxit){
				break;
			}
			++(*jtot);
			// Form shift.
			real_type g = (diag[l-1]-p) / (real_type(2)*offdiag[l-1]);
			real_type r = Traits<real_type>::hypot2(g, real_type(1));
			g = diag[m] - p + (offdiag[l-1] / (g+(g > 0 ? r : -r)));
			{
				real_type s(1);
				real_type c(1);
				p = real_type(0);

				// Inner loop
				for(size_t i = m; i < l; ++i){
					real_type f = s*offdiag[i];
					real_type b = c*offdiag[i];
					LA::Rotation::Generate(g, f, &c, &s, &r);
					if(i != m){ offdiag[i-1] = r; }
					g = diag[i] - p;
					r = (diag[i+1]-g)*s + real_type(2)*c*b;
					p = s*r;
					diag[i] = g + p;
					g = c*r - b;
					// If eigenvectors are desired, then save rotations.
					if(NULL != z){
						cv[i] = c;
						sv[i] = s;
					}
				}
			}
			// If eigenvectors are desired, then apply saved rotations.
			if(NULL != z){
				LA::Rotation::ApplySequence(
					"R","V","F", n, l-m+1, &cv[m], &sv[m], &z[0+m*ldz], ldz
				);
			}
			diag[l] -= p;
			offdiag[l-1] = g;
			continue;
		}
		// Eigenvalue found.
		diag[l] = p;
		--l;
	}while(l+1 >= lend+1); // +1's necessary here
	*pl = l;
}


///////////////////////////////////////////////////////////////////////
// Util::SymmetricEigensystem
// --------------------------
// Computes all eigenvvalues and, optionally, eigenvectors of a
// symmetric tridiagonal matrix using the implicit QL or QR method.
// If a matrix was reduced to tridiagonal form, the eigenvectors are
// recovered by supplying the unitary reducing matrix in Z, and the
// diagonalizing transformations are applied to Z, which will transform
// its columns into the eigenvectors. The diagonalizing transform is
//
//     T = Z^H * D * Z
//
// where T is the given tridiagonal matrix, D is the diagonal matrix of
// eigenvalues returned in diag, and Z is the matrix of eigenvectors.
// The eigenvectors of T can be found by initializing Z to the
// identity matrix.
//
// This is equivalent to Lapack routine _steqr, except eigenvalues are
// not sorted.
//
// Arguments
// n       The number of rows and columns of the matrix.
// diag    Pointer to the diagonal elements of A (length n). On exit,
//         overwritten by the eigenvalues.
// offdiag Pointer to the off-diagonal elements of A (length n-1). On
//         exit, the contents are destroyed.
// z       Pointer to the first element of the matrix Z. On entry,
//         Z should be a unitary matrix to which the diagonalizing
//         transformations are applied; this can be the identity matrix
//         if the eigenvectors of T are desired, or some other matrix
//         used to reduce a Hermitian matrix to tridiagonal form if the
//         eigenvectors of the original matrix are desired. If NULL,
//         the eigenvectors are not computed.
// ldz     The leading dimension of the array containing Z, ldz >= n.
// work    Workspace of length 2*(n-1). Not referenced if z = NULL
//
template <typename T>
int SymmetricEigensystem(
	size_t n,
	typename Traits<T>::real_type *diag,
	typename Traits<T>::real_type *offdiag,
	T *z, size_t ldz,
	typename Traits<T>::real_type *work
){
	RNPAssert(NULL != diag && NULL != offdiag);
	RNPAssert(NULL == z || ldz >= n);
	RNPAssert(NULL == z || NULL != work);
	
	typedef typename Traits<T>::real_type real_type;
	if(n <= 1){
		return 0;
	}

	// Determine the unit roundoff and over/underflow thresholds.
	static const real_type eps(Traits<real_type>::eps());
	static const real_type eps2(eps*eps);
	static const real_type safmin(Traits<real_type>::min());
	static const real_type safmax(Traits<real_type>::max());
	static const real_type ssfmax(sqrt(safmax) / real_type(3));
	static const real_type ssfmin(sqrt(safmin) / eps2);

	const size_t nmaxit = n * 30;
	size_t jtot = 0;

	// Determine where the matrix splits and choose QL or QR iteration
	// for each block, according to whether top or bottom diagonal
	// element is smaller.

	size_t l1 = 0;
	size_t l, m;
	size_t lsv, lend, lendsv;

	real_type anorm;
	int iscale;
	
	do{
		if(l1+1 > n){ return 0; }
		if(l1+1 > 1){
			offdiag[l1-1] = real_type(0);
		}
		{
			bool found_small = false;
			// skip over zero and small subdiagonals
			if(l1+1 < n){
				for(m = l1; m+1 < n; ++m){
					real_type tst(Traits<T>::abs(offdiag[m]));
					if(real_type(0) == tst){
						found_small = true;
						break;
					}else if(
						tst <= eps * (
							sqrt(Traits<T>::abs(diag[m])) * sqrt(Traits<T>::abs(diag[m+1]))
						)
					){
						offdiag[m] = real_type(0);
						found_small = true;
						break;
					}
				}
			}
			if(!found_small){
				m = n-1;
			}
		}

		l = l1;
		lsv = l;
		lend = m;
		lendsv = lend;
		l1 = m+1;
		if(lend == l){
			continue;
		}

		// Scale submatrix in rows and columns (l+1) to (lend+1)
		// For the complex case, this should have been the "I" norm.
		anorm = LA::Tridiagonal::NormSym("M", lend-l+1, &diag[l], &offdiag[l]);
		iscale = 0;
		if(real_type(0) == anorm){
			continue;
		}
		if(anorm > ssfmax){
			iscale = 1;
			BLAS::Rescale("G", 0, 0, anorm, ssfmax, lend-l+1, 1, &   diag[l], n);
			BLAS::Rescale("G", 0, 0, anorm, ssfmax, lend-l+0, 1, &offdiag[l], n);
		}else if(anorm < ssfmin){
			iscale = 2;
			BLAS::Rescale("G", 0, 0, anorm, ssfmin, lend-l+1, 1, &   diag[l], n);
			BLAS::Rescale("G", 0, 0, anorm, ssfmin, lend-l+0, 1, &offdiag[l], n);
		}
		// Choose between QL and QR iteration
		if(Traits<T>::abs(diag[lend]) < Traits<T>::abs(diag[l])){
			lend = lsv;
			l = lendsv;
		}
		if(lend > l){ // QL Iteration, Look for small subdiagonal element.
			SymmetricQLIteration(
				n, m, &l, lend, &jtot, nmaxit, diag, offdiag, z, ldz, &work[0], &work[n-1]
			);
		}else{ // QR Iteration, Look for small superdiagonal element.
			SymmetricQRIteration(
				n, m, &l, lend, &jtot, nmaxit, diag, offdiag, z, ldz, &work[0], &work[n-1]
			);
		}

		// Undo scaling if necessary
		if(1 == iscale){
			BLAS::Rescale("G", 0, 0, ssfmax, anorm, lendsv-lsv+1, 1, &   diag[lsv], n);
			BLAS::Rescale("G", 0, 0, ssfmax, anorm, lendsv-lsv  , 1, &offdiag[lsv], n);
		}else if(2 == iscale){
			BLAS::Rescale("G", 0, 0, ssfmin, anorm, lendsv-lsv+1, 1, &   diag[lsv], n);
			BLAS::Rescale("G", 0, 0, ssfmin, anorm, lendsv-lsv  , 1, &offdiag[lsv], n);
		}
		// Check for no convergence to an eigenvalue after a total of N*MAXIT iterations.
	}while(jtot < nmaxit);
	
	int info = 0;
	for(size_t i = 0; i+1 < n; ++i){
		if(T(0) != offdiag[i]){
			++info;
		}
	}
	return info;
}

} // namespace Util

///////////////////////////////////////////////////////////////////////
// ReduceHerm_unblocked
// --------------------
// Reduces a Hermitian matrix A into real symmetric tridiagonal matrix
// T by unitary similarity transformation:
//
//     Q^H * A * Q = T
//
// If uplo = "U", the matrix Q is represented as a product of elementary
// reflectors Q = H[n-2] ... H[1] H[0]. Each H[i] has the form
//
//    H[i] = I - tau * v * v^H
//
// where tau is a scalar, and v is a vector with v[i+1..n] = 0 and
// v[i] = 1; v[0..i] is stored on exit in A[0..i,i+1], and tau in
// tau[i].
// If uplo = "L", the matrix Q is represented as a product of elementary
// reflectors Q = H[0] H[1] ... H[n-2]. Each H[i] has the form
//
//    H(i) = I - tau * v * v^H
//
// where tau is a scalar, and v is a vector with v[0..i+1] = 0 and
// v[i+1] = 1; v[i+2..n] is stored on exit in A[i+2..n,i], and tau
// in tau[i].
//
// The contents of A on exit are illustrated by the following examples
// with n = 5:
//
//            uplo = "U"                           uplo = "L"
//     [  d   e   v1  v2  v3 ]              [  d                  ]
//     [      d   e   v2  v3 ]              [  e   d              ]
//     [          d   e   v3 ]              [  v0  e   d          ]
//     [              d   e  ]              [  v0  v1  e   d      ]
//     [                  d  ]              [  v0  v1  v2  e   d  ]
//
// where d and e denote diagonal and off-diagonal elements of T, and vi
// denotes an element of the vector defining H[i].
// This routine uses only level 2 BLAS.
// This is equivalent to Lapack routine _hetd2 and _sytd2.
//
// Arguments
// uplo    If "U", the upper triangle of A is given.
//         If "L", the lower triangle of A is given.
// n       The number of rows and columns of the matrix A.
// a       Pointer to the first element of A. If uplo = "U", the upper
//         triangular part of A is assumed to be provided, and the
//         lower triangle is not touched. Similarly for uplo = "L".
//         On exit, the diagonal and off-diagonal are overwritten by
//         corresponding elements of the tridiagonal matrix T. The
//         elements other than the diagonal and offdiagonal are
//         overwritten by the vectors of the elementary reflectors.
// lda     Leading dimension of the array containing A, lda >= n.
// diag    Pointer to the diagonal elements of A (length n). On exit,
//         overwritten by the eigenvalues.
// offdiag Pointer to the off-diagonal elements of A (length n-1). On
//         exit, the contents are destroyed.
// tau     The scale factors of the elementary reflectors (length n-1).
//
template <typename T>
void ReduceHerm_unblocked(
	const char *uplo, size_t n, T *a, size_t lda,
	typename Traits<T>::real_type *diag,
	typename Traits<T>::real_type *offdiag,
	T *tau
){
	if(0 == n){ return; }
	if('U' == uplo[0]){
		a[n-1+(n-1)*lda] = Traits<T>::real(a[n-1+(n-1)*lda]);
		size_t i = n-1; while(i --> 0){
			T taui;
			// Generate reflector H[i] to annihilate A[0..i,i+1]
			T alpha(a[i+(i+1)*lda]);
			Reflector::Generate(i+1, &alpha, &a[0+(i+1)*lda], 1, &taui);
			offdiag[i] = Traits<T>::real(alpha);
			if(T(0) != taui){
				// Apply H[i] from both sides to A[0..i+1,0..i+1]
				a[i+(i+1)*lda] = T(1);
				// Compute x = tau * A * v, store x in tau[0..i+1]
				BLAS::MultHermV(uplo, i+1, taui, a, lda, &a[0+(i+1)*lda], 1, T(0), tau, 1);
				// Compute w = x - 1/2 * tau * (x^H * v) * v
				alpha = -(T(1)/T(2)) * taui * BLAS::ConjugateDot(i+1, tau, 1, &a[0+(i+1)*lda], 1);
				BLAS::Axpy(i+1, alpha, &a[0+(i+1)*lda], 1, tau, 1);
				// Apply A -= (v*w^H + w*v^H)
				BLAS::HermRank2Update(uplo, i+1, T(-1), &a[0+(i+1)*lda], 1, tau, 1, a, lda);
			}else{
				a[i+i*lda] = Traits<T>::real(a[i+i*lda]);
			}
			a[i+(i+1)*lda] = offdiag[i];
			diag[i+1] = Traits<T>::real(a[i+1+(i+1)*lda]);
			tau[i] = taui;
		}
		diag[0] = Traits<T>::real(a[0+0*lda]);
	}else{
		a[0+0*lda] = Traits<T>::real(a[0+0*lda]);
		for(size_t i = 0; i+1 < n; ++i){
			T taui;
			// Generate reflector H[i] to annihilate A[i+2..n,i]
			T alpha(a[i+1+i*lda]);
			size_t row = (i+3 < n ? i+2 : n-1);
			Reflector::Generate(n-i-1, &alpha, &a[row+i*lda], 1, &taui);
			offdiag[i] = Traits<T>::real(alpha);
			if(T(0) != taui){
				// Apply H[i] from both sides to A[i+1..n,i+1..n]
				a[i+1+i*lda] = T(1);
				// Compute x = tau * A * v, store x in tau[i..n-1]
				BLAS::MultHermV(uplo, n-i-1, taui, &a[i+1+(i+1)*lda], lda, &a[i+1+i*lda], 1, T(0), &tau[i], 1);
				// Compute w = x - 1/2 * tau * (x^H * v) * v
				alpha = -(T(1)/T(2)) * taui * BLAS::ConjugateDot(n-i-1, &tau[i], 1, &a[i+1+i*lda], 1);
				BLAS::Axpy(n-i-1, alpha, &a[i+1+i*lda], 1, &tau[i], 1);
				// Apply A -= (v*w^H + w*v^H)
				BLAS::HermRank2Update(uplo, n-i-1, T(-1), &a[i+1+i*lda], 1, &tau[i], 1, &a[i+1+(i+1)*lda], lda);
			}else{
				a[i+1+(i+1)*lda] = Traits<T>::real(a[i+1+(i+1)*lda]);
			}
			a[i+1+i*lda] = offdiag[i];
			diag[i] = Traits<T>::real(a[i+i*lda]);
			tau[i] = taui;
		}
		diag[n-1] = Traits<T>::real(a[n-1+(n-1)*lda]);
	}
}

///////////////////////////////////////////////////////////////////////
// ReduceHerm
// ----------
// Reduces a Hermitian matrix A into real symmetric tridiagonal matrix
// T by unitary similarity transformation:
//
//     Q^H * A * Q = T
//
// If uplo = "U", the matrix Q is represented as a product of elementary
// reflectors Q = H[n-2] ... H[1] H[0]. Each H[i] has the form
//
//    H[i] = I - tau * v * v^H
//
// where tau is a scalar, and v is a vector with v[i+1..n] = 0 and
// v[i] = 1; v[0..i] is stored on exit in A[0..i,i+1], and tau in
// tau[i].
// If uplo = "L", the matrix Q is represented as a product of elementary
// reflectors Q = H[0] H[1] ... H[n-2]. Each H[i] has the form
//
//    H(i) = I - tau * v * v^H
//
// where tau is a scalar, and v is a vector with v[0..i+1] = 0 and
// v[i+1] = 1; v[i+2..n] is stored on exit in A[i+2..n,i], and tau
// in tau[i].
//
// The contents of A on exit are illustrated by the following examples
// with n = 5:
//
//            uplo = "U"                           uplo = "L"
//     [  d   e   v1  v2  v3 ]              [  d                  ]
//     [      d   e   v2  v3 ]              [  e   d              ]
//     [          d   e   v3 ]              [  v0  e   d          ]
//     [              d   e  ]              [  v0  v1  e   d      ]
//     [                  d  ]              [  v0  v1  v2  e   d  ]
//
// where d and e denote diagonal and off-diagonal elements of T, and vi
// denotes an element of the vector defining H[i].
//
// This is equivalent to Lapack routine _hetrd and _sytrd.
//
// Arguments
// uplo    If "U", the upper triangle of A is given.
//         If "L", the lower triangle of A is given.
// n       The number of rows and columns of the matrix A.
// a       Pointer to the first element of A. If uplo = "U", the upper
//         triangular part of A is assumed to be provided, and the
//         lower triangle is not touched. Similarly for uplo = "L".
//         On exit, the diagonal and off-diagonal are overwritten by
//         corresponding elements of the tridiagonal matrix T. The
//         elements other than the diagonal and offdiagonal are
//         overwritten by the vectors of the elementary reflectors.
// lda     Leading dimension of the array containing A, lda >= n.
// diag    Pointer to the diagonal elements of A (length n). On exit,
//         overwritten by the eigenvalues.
// offdiag Pointer to the off-diagonal elements of A (length n-1). On
//         exit, the contents are destroyed.
// tau     The scale factors of the elementary reflectors (length n-1).
// lwork   Length of workspace. If *lwork == 0 then the optimal size
//         is returned in this argument. If both *lwork == 0 and
//         NULL == work, then ReduceHerm_unblocked is called.
// work    Workspace of size lwork, or NULL.
//
template <typename T>
void ReduceHerm(
	const char *uplo, size_t n, T *a, size_t lda,
	typename Traits<T>::real_type *diag,
	typename Traits<T>::real_type *offdiag,
	T *tau, size_t *lwork, T *work
){
	typedef typename Traits<T>::real_type real_type;
	RNPAssert('U' == uplo[0] || 'L' == uplo[0]);
	RNPAssert(lda >= n);
	RNPAssert(NULL != lwork);
	RNPAssert(0 == *lwork || *lwork >= n);
	
	if(0 == n){ return; }
	
	size_t nb = Tuning<T>::reduce_herm_block_size_opt(uplo, n);
	if(0 == *lwork && NULL != work){
		*lwork = n*nb;
		return;
	}
	
	// We don't require a workspace; this will always use unblocked code
	if(NULL == work){
		ReduceHerm_unblocked(uplo, n, a, lda, diag, offdiag, tau);
	}
	
	size_t nx = n;
	size_t iws = 0;
	size_t nbmin = 2;
	size_t ldwork = n;
	if(nb > 1 && nb < n){
		nx = Tuning<T>::reduce_herm_crossover_size(uplo, n);
		if(nb > nx){ nx = nb; }
		if(nx < n){
			iws = ldwork*nb;
			if(*lwork < iws){
				nb = *lwork/ldwork;
				if(0 == nb){ nb = 1; }
				nbmin = Tuning<T>::reduce_herm_block_size_min(uplo, n);
				if(nb < nbmin){ nx = n; }
			}
		}else{ nx = n; }
	}else{
		nb = 1;
	}
	
	if('U' == uplo[0]){
		size_t kk = n - ((n-nx+nb-1) / nb) * nb;
		size_t i = n-nb;
		while(i >= kk){
			Util::ReduceHerm_block(
				uplo, i+nb, nb, a, lda, offdiag, tau, work, ldwork
			);
			BLAS::HermRank2KUpdate(
				uplo, "N", i, nb, T(-1), &a[0+i*lda], lda,
				work, ldwork, real_type(1), a, lda
			);
			for(size_t j = i; j < i+nb; ++j){
				a[j-1+j*lda] = offdiag[j-1];
				diag[j] = Traits<T>::real(a[j+j*lda]);
			}
			if(0 == i){ break; }
			i -= nb;
		}
		ReduceHerm_unblocked(
			uplo, kk, a, lda, diag, offdiag, tau
		);
	}else{
		size_t i;
		for(i = 0; i+nx < n; i += nb){
			Util::ReduceHerm_block(
				uplo, n-i, nb, &a[i+i*lda], lda, &offdiag[i], &tau[i], work, ldwork
			);
			BLAS::HermRank2KUpdate(
				uplo, "N", n-i-nb, nb, T(-1), &a[i+nb+i*lda], lda,
				&work[nb], ldwork, real_type(1), &a[i+nb+(i+nb)*lda], lda
			);
			for(size_t j = 0; j < i+nb; ++j){
				a[j+1+j*lda] = offdiag[j];
				diag[j] = Traits<T>::real(a[j+j*lda]);
			}
		}
		ReduceHerm_unblocked(
			uplo, n-i, &a[i+i*lda], lda, &diag[i], &offdiag[i], &tau[i]
		);
	}
}

///////////////////////////////////////////////////////////////////////
// GenerateQHerm
// -------------
// Generates a complex unitary matrix Q which is defined as the
// product of n-1 elementary reflectors of order n, as returned by
// ReduceHerm. The resulting Q is such that
//
//     Q^H * A * Q = T
//
// This routine is equivalent to Lapack _ungtr and _orgtr.
//
// Arguments
// uplo  If "U", the upper triangle of A contains the reflectors.
//       If "L", the lower triangle of A contains the reflectors.
// n     Number of rows and columns of Q.
// a     Pointer to the first element of the matrix returned by
//       ReduceHerm which contains the vectors defining the elementary
//       reflectors.
// lda   Leading dimension of the array containing A, lda >= n.
// tau   Length n-1 array of elementary reflector scale factors
//       returned by ReduceHerm.
// lwork Length of workspace (>= n-1). If *lwork == 0 then the optimal
//       size is returned in this argument.
// work  Workspace of size lwork, or NULL for workspace query.
//
template <typename T>
void GenerateQHerm(
	const char *uplo, size_t n, T *a, size_t lda,
	const T *tau, // length n-1
	size_t *lwork, T *work
){
	RNPAssert('U' == uplo[0] || 'L' == uplo[0]);
	RNPAssert(NULL != lwork);
	RNPAssert(lda >= n);
	RNPAssert(0 == *lwork || (*lwork)+1 >= n);
	if(0 == n){ return; }
	
	if(0 == *lwork || NULL == work){
		size_t nb;
		if('U' == uplo[0]){
			nb = LA::QL::Tuning<T>::genQ_block_size_opt(n-1, n-1, n-1);
		}else{
			nb = LA::QR::Tuning<T>::genQ_block_size_opt(n-1, n-1, n-1);
		}
		*lwork = (n-1) * nb;
		return;
	}
	
	if('U' == uplo[0]){
		// Q was determined by a call to ReduceHerm("U", ...)
		// Shift the vectors which define the elementary reflectors one
		// column to the left, and set the last row and column of Q to
		// those of the unit matrix
		for(size_t j = 0; j+1 < n; ++j){
			for(size_t i = 0; i < j; ++i){
				a[i+j*lda] = a[i+(j+1)*lda];
			}
			a[n-1+j*lda] = T(0);
		}
		for(size_t i = 0; i+1 < n; ++i){
			a[i+(n-1)*lda] = T(0);
		}
		a[(n-1)+(n-1)*lda] = T(1);
		// Generate Q[0..n-1,0..n-1]
		LA::QL::GenerateQ(n-1, n-1, n-1, a, lda, tau, lwork, work);
	}else{
		// Q was determined by a call to ReduceHerm("L", ...)
		// Shift the vectors which define the elementary reflectors one
		// column to the right, and set the first row and column of Q to
		// those of the unit matrix
		size_t j = n;
		while(j --> 1){
			a[0+j*lda] = T(0);
			for(size_t i = j+1; i < n; ++i){
				a[i+j*lda] = a[i+(j-1)*lda];
			}
		}
		a[0+0*lda] = T(1);
		for(size_t i = 1; i < n; ++i){
			a[i+0*lda] = T(0);
		}
		if(n > 1){
			// Generate Q[1..n,1..n]
			LA::QR::GenerateQ(n-1, n-1, n-1, &a[1+1*lda], lda, tau, lwork, work);
		}
	}
}

///////////////////////////////////////////////////////////////////////
// MultQHerm
// ---------
// Overwrites a general m-by-n matrix C with
//
//    trans | side = "L"   | side = "R"
//    ------|--------------|------------
//     "N"  |   Q   * C    |  C * Q
//     "C"  |   Q^H * C    |  C * Q^H
//
// where Q is the unitary matrix obtained from ReduceHerm which
// performed:
//
//     Q^H * A * Q = T
//
// This routine is equivalent to Lapack _unmtr and _ormtr.
//
// Arguments
// side  If "L", apply Q or Q^H from the left. If "R", apply Q or
//       Q^H from the right.
// uplo  If "U", the upper triangle of A contains the reflectors.
//       If "L", the lower triangle of A contains the reflectors.
// trans If "N", apply Q. If "C", apply Q^H.
// m     Number of rows of C.
// n     Number of columns of C.
// a     Pointer to the first element of the matrix returned by
//       ReduceHerm which contains the vectors defining the elementary
//       reflectors.
// lda   Leading dimension of the array containing A. If side = "L",
//       lda >= m. If side = "R", lda >= n.
// tau   Array of elementary reflector scale factors returned by
//       ReduceHerm. If side = "L", length m-1. If side = "R", length
//       n-1.
// c     Pointer to the first element of the matrix C.
// ldc   Leading dimension of the array containing C.
// lwork Length of workspace; if side = "L", must be at least n, and
//       if side = "R", must be at least m. If *lwork == 0 then the
//       optimal size is returned in this argument.
// work  Workspace of size lwork, or NULL for workspace query.
//
template <typename T>
void MultQHerm(
	const char *side, const char *uplo, const char *trans,
	size_t m, size_t n, const T *a, size_t lda, const T *tau, // length n-1
	T *c, size_t ldc, size_t *lwork, T *work
){
	RNPAssert('L' == side[0] || 'R' == side[0]);
	RNPAssert('U' == uplo[0] || 'L' == uplo[0]);
	RNPAssert('N' == trans[0] || 'T' == trans[0] || 'C' == trans[0]);
	RNPAssert(NULL != lwork);
	
	const bool left  = ('L' == side[0]);
	const bool upper = ('U' == uplo[0]);
	const size_t nq = (left ? m : n);
	const size_t nw = (left ? n : m);
	RNPAssert(lda >= nq);
	RNPAssert(ldc >= m);
	RNPAssert(0 == *lwork || *lwork >= nw);
	
	if(0 == m || 0 == n || 1 == nq){ return; }
	
	if(0 == *lwork){
		size_t nb;
		if(upper){
			if(left){
				nb = LA::QL::Tuning<T>::multQ_block_size_opt(side, trans, m-1, n, m-1);
			}else{
				nb = LA::QL::Tuning<T>::multQ_block_size_opt(side, trans, m, n-1, n-1);
			}
		}else{
			if(left){
				nb = LA::QR::Tuning<T>::multQ_block_size_opt(side, trans, m-1, n, m-1);
			}else{
				nb = LA::QR::Tuning<T>::multQ_block_size_opt(side, trans, m, n-1, n-1);
			}
		}
		*lwork = nw*nb;
		return;
	}
	
	const size_t mi = (left ? m-1 : m);
	const size_t ni = (left ? n : n-1);
	if(upper){
		LA::QL::MultQ(side, trans, mi, ni, nq-1, &a[0+1*lda], lda, tau, c, ldc, lwork, work);
	}else{
		const size_t i1 = (left ? 1 : 0);
		const size_t i2 = (left ? 0 : 1);
		LA::QR::MultQ(side, trans, mi, ni, nq-1, &a[1+0*lda], lda, tau, &c[i1+i2*ldc], ldc, lwork, work);
	}
}

///////////////////////////////////////////////////////////////////////
// SymmetricEigensystem
// --------------------
// Computes all eigenvvalues and, optionally, eigenvectors of a
// symmetric tridiagonal matrix using the implicit QL or QR method.
// The diagonalizing transform is
//
//     T = Z^H * D * Z
//
// where T is the given tridiagonal matrix, D is the diagonal matrix of
// eigenvalues returned in diag, and Z is the matrix of eigenvectors.
//
// This is equivalent to Lapack routine _stev, except eigenvalues are
// not sorted.
//
// Arguments
// n       The number of rows and columns of the matrix.
// diag    Pointer to the diagonal elements of A (length n). On exit,
//         overwritten by the eigenvalues.
// offdiag Pointer to the off-diagonal elements of A (length n-1). On
//         exit, the contents are destroyed.
// z       Pointer to the first element of the matrix Z, in which the
//         eigenvectors are returned. If NULL, the eigenvectors are
//         not computed.
// ldz     The leading dimension of the array containing Z, ldz >= n.
// work    Workspace of length 2*(n-1). Not referenced if z = NULL
//
template <typename T>
int SymmetricEigensystem(
	size_t n,
	typename Traits<T>::real_type *diag,
	typename Traits<T>::real_type *offdiag,
	T *z, size_t ldz,
	typename Traits<T>::real_type *work
){
	RNPAssert(NULL != diag && NULL != offdiag);
	RNPAssert(NULL == z || ldz >= n);
	RNPAssert(NULL == z || NULL != work);
	typedef typename Traits<T>::real_type real_type;
	
	if(0 == n){ return 0; }
	if(1 == n){
		z[0] = T(1);
		return 0;
	}
	static const real_type safmin = Traits<real_type>::min();
	static const real_type eps = real_type(2)*Traits<real_type>::eps();
	static const real_type smlnum = safmin / eps;
	static const real_type bignum = real_type(1) / smlnum;
	static const real_type rmin = sqrt(smlnum);
	static const real_type rmax = sqrt(bignum);
	
	// Scale matrix to allowable range, if necessary.
	bool did_scale = false;
	real_type tnrm = NormSym("M", n, diag, offdiag);
	real_type sigma(1);
	if(tnrm > real_type(0) && tnrm < rmin){
		did_scale = true;
		sigma = rmin / tnrm;
	}else if(tnrm > rmax){
		did_scale = true;
		sigma = rmax / tnrm;
	}
	if(did_scale){
		BLAS::Scale(n, sigma, diag, 1);
		BLAS::Scale(n-1, sigma, offdiag, 1);
	}

	if(NULL != z){
		BLAS::Set(n, n, T(0), T(1), z, ldz);
	}
	int info = Util::SymmetricEigensystem(n, diag, offdiag, z, ldz, work);

	// If matrix was scaled, then rescale eigenvalues appropriately.
	if(did_scale){
		size_t nscal = n;
		if(0 != info){
			nscal = info-1;
		}
		BLAS::Scale(nscal, real_type(1)/sigma, diag, 1);
	}
	return info;
}

} // namespace Tridiagonal
} // namespace LA
} // namespace RNP

#endif // RNP_TRIDIAGONAL_HPP_INCLUDED
