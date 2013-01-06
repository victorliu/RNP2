#ifndef RNP_QR_HPP_INCLUDED
#define RNP_QR_HPP_INCLUDED

#include <iostream>
#include <cstddef>
#include <RNP/BLAS.hpp>
#include "Reflector.hpp"

#include <iostream>

namespace RNP{
namespace LA{

namespace QR{

// Specialize this class to tune the block size.
template <typename T>
struct Tuning{
	static size_t block_size_opt(size_t m, size_t n){ return 4; }
	static size_t block_size_min(size_t m, size_t n){ return 4; }
	static size_t crossover_size(size_t m, size_t n){ return 4; }
};

} // namespace QR

// Computes a QR factorization of a complex m by n matrix A = Q * R.

// Arguments
// =========

// M       The number of rows of the matrix A.  M >= 0.

// N       The number of columns of the matrix A.  N >= 0.

// A       (input/output) COMPLEX*16 array, dimension (LDA,N)
//         On entry, the m by n matrix A.
//         On exit, the elements on and above the diagonal of the array
//         contain the min(m,n) by n upper trapezoidal matrix R (R is
//         upper triangular if m >= n); the elements below the diagonal,
//         with the array TAU, represent the unitary matrix Q as a
//         product of elementary reflectors (see Further Details).

// LDA     The leading dimension of the array A.  LDA >= max(1,M).

// TAU     (output) COMPLEX*16 array, dimension (min(M,N))
//         The scalar factors of the elementary reflectors (see Further
//         Details).

// WORK    (workspace) COMPLEX*16 array, dimension (N)

// Further Details
// ===============

// The matrix Q is represented as a product of elementary reflectors
//    Q = H(1) H(2) . . . H(k), where k = min(m,n).
// Each H(i) has the form
//    H(i) = I - tau * v * v'
// where tau is a complex scalar, and v is a complex vector with
// v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
// and tau in TAU(i).
template <typename T> // _geqr2
void QRFactor_unblocked(size_t m, size_t n, T *a, size_t lda, T *tau, T *work){

	size_t k = m; if(n < k){ k = n; }
	for(size_t i = 0; i < k; ++i){
		// Generate elementary reflector H(i) to annihilate A(i+1:m,i)
		size_t row = m-1; if(i+1 < row){ row = i+1; }
		Reflector::Generate(m-i, &a[i+i*lda], &a[row+i*lda], 1, &tau[i]);
		//ReflectorGenerate2(m-i, &a[i+i*lda], &a[row+i*lda], 1, &tau[i]);
		//ReflectorGeneratePositive(m-i, &a[i+i*lda], &a[row+i*lda], 1, &tau[i]);
		if(i < n-1){
			// Apply H(i)' to A(i:m,i+1:n) from the left
			T alpha = a[i+i*lda];
			a[i+i*lda] = T(1);
			Reflector::Apply("L", false, false, m-i, n-i-1, &a[i+i*lda], 1, Traits<T>::conj(tau[i]), &a[i+(i+1)*lda], lda, work);
			//ReflectorApply("L", m-i, n-i-1, &a[i+i*lda], 1, Traits<T>::conj(tau[i]), &a[i+(i+1)*lda], lda, work);
			a[i+i*lda] = alpha;
		}
	}
}

// Computes a QR factorization of the m-by-n matrix a.
// 
// + tau should be length min(m,n)
// + When 0 == *lwork or NULL == work, a workspace query is performed and
//   the optimal lwork is placed back into lwork. Otherwise, lwork >= n.
// + work should be of length lwork when not NULL.
template <typename T> // _geqrf
void QRFactor(size_t m, size_t n, T *a, size_t lda, T *tau, size_t *lwork, T *work){
	RNPAssert(lda >= m);
	RNPAssert(NULL != lwork);
	const size_t k = (m < n ? m : n);
	if(0 == k){
		return;
	}
	size_t nb = QR::Tuning<T>::block_size_opt(m, n);
	if(NULL == work || 0 == *lwork){
		*lwork = nb*n;
		return;
	}
	size_t nbmin = 2;
	size_t nx = 0;
	size_t iws = n;
	size_t ldwork = n;
	if(nb > 1 && nb < k){
		// Determine when to cross over from blocked to unblocked code.
		nx = QR::Tuning<T>::crossover_size(m, n);
		if(nx < k){
			// Determine if workspace is large enough for blocked code.
            iws = ldwork * nb;
            if(*lwork < iws){
				// Not enough workspace to use optimal NB:  reduce NB and
				// determine the minimum value of NB.
				nb = *lwork / ldwork;
				nbmin = QR::Tuning<T>::block_size_min(m, n);
				if(2 > nbmin){ nbmin = 2; }
			}
		}
	}
	size_t i;
	if(nb >= nbmin && nb < k && nx < k){
		// Use blocked code initially
		for(i = 0; i+nx < k; i += nb){
			const size_t ib = (k-i < nb ? k-i : nb);
			// Compute the QR factorization of the current block A(i:m,i:i+ib-1)
			QRFactor_unblocked(m-i, ib, &a[i+i*lda], lda, &tau[i], work);
			if(i+ib < n){
				// Form the triangular factor of the block reflector
				// H = H(i) H(i+1) . . . H(i+ib-1)
				Reflector::GenerateBlockTr(
					"F","C", m-i, ib, &a[i+i*lda], lda, &tau[i], work, ldwork
				);
				// Apply H^H to A(i:m,i+ib:n) from the left
				Reflector::ApplyBlock(
					"L","C","F","C", m-i, n-i-ib, ib, &a[i+i*lda], lda,
					work, ldwork, &a[i+(i+ib)*lda], lda, &work[ib], ldwork
				);
			}
		}
	}else{
		i = 0;
	}
	if(i < k){
		QRFactor_unblocked(m-i, n-i, &a[i+i*lda], lda, &tau[i], work);
	}
}

template <typename T> // _unmqr, _ormqr
void QRMultQ_unblocked(
	const char *side, const char *trans, size_t m, size_t n, size_t k,
	const T *a, size_t lda, T *tau, T *c, size_t ldc, T *work
){
	// Purpose
	// =======

	// ZUNM2R overwrites the general complex m-by-n matrix C with
	//       Q * C  if SIDE = 'L' and TRANS = 'N', or
	//       Q'* C  if SIDE = 'L' and TRANS = 'C', or
	//       C * Q  if SIDE = 'R' and TRANS = 'N', or
	//       C * Q' if SIDE = 'R' and TRANS = 'C',
	// where Q is a complex unitary matrix defined as the product of k
	// elementary reflectors
	//       Q = H(1) H(2) . . . H(k)
	// as returned by ZGEQRF. Q is of order m if SIDE = 'L' and of order n
	// if SIDE = 'R'.

	// Arguments
	// =========

	// SIDE    = 'L': apply Q or Q' from the Left
	//         = 'R': apply Q or Q' from the Right

	// TRANS   = 'N': apply Q  (No transpose)
	//         = 'C': apply Q' (Conjugate transpose)

	// M       The number of rows of the matrix C. M >= 0.

	// N       The number of columns of the matrix C. N >= 0.

	// K       The number of elementary reflectors whose product defines
	//         the matrix Q.
	//         If SIDE = 'L', M >= K >= 0;
	//         if SIDE = 'R', N >= K >= 0.

	// A       (input) COMPLEX*16 array, dimension (LDA,K)
	//         The i-th column must contain the vector which defines the
	//         elementary reflector H(i), for i = 1,2,...,k, as returned by
	//         ZGEQRF in the first k columns of its array argument A.
	//         A is modified by the routine but restored on exit.

	// LDA     The leading dimension of the array A.
	//         If SIDE = 'L', LDA >= max(1,M);
	//         if SIDE = 'R', LDA >= max(1,N).

	// TAU     TAU(i) must contain the scalar factor of the elementary
	//         reflector H(i), as returned by ZGEQRF.

	// C       (input/output) COMPLEX*16 array, dimension (LDC,N)
	//         On entry, the m-by-n matrix C.
	//         On exit, C is overwritten by Q*C or Q'*C or C*Q' or C*Q.

	// LDC     The leading dimension of the array C. LDC >= max(1,M).

	// WORK    (workspace) COMPLEX*16 array, dimension
	//                                  (N) if SIDE = 'L',
	//                                  (M) if SIDE = 'R'

	const bool left = ('L' == side[0]);
	const bool notran = ('N' == trans[0]);

	if(m < 1 || n < 1 || k < 1){ return; }
	
	size_t mi, ni;
	size_t ic, jc;
	if(left){
		ni = n;
		jc = 0;
	}else{
		mi = m;
		ic = 0;
	}
	
	if((left && ! notran) || (! left && notran)){
		// loop forwards
		for(size_t i = 0; i < k; ++i){
			if(left){ // H(i) or H(i)' is applied to C(i:m,1:n)
				mi = m - i;
				ic = i;
			}else{ // H(i) or H(i)' is applied to C(1:m,i:n)
				ni = n - i;
				jc = i;
			}

			// Apply H(i) or H(i)'
			T taui;
			if(notran){
				taui = tau[i];
			}else{
				taui = Traits<T>::conj(tau[i]);
			}
			//T aii = a[i+i*lda];
			//a[i+i*lda] = 1;
			//ReflectorApply(side, mi, ni, &a[i+i*lda], 1, taui, &c[ic+jc*ldc], ldc, work);
			//a[i+i*lda] = aii;
			
			Reflector::Apply(side, true, false, mi, ni, &a[i+i*lda], 1, taui, &c[ic+jc*ldc], ldc, work);
		}
	}else{
		// loop backwards
		size_t i = k;
		while(i --> 0){
			if(left){ // H(i) or H(i)' is applied to C(i:m,1:n)
				mi = m - i;
				ic = i;
			}else{ // H(i) or H(i)' is applied to C(1:m,i:n)
				ni = n - i;
				jc = i;
			}

			// Apply H(i) or H(i)'
			T taui;
			if(notran){
				taui = tau[i];
			}else{
				taui = Traits<T>::conj(tau[i]);
			}
			//T aii = a[i+i*lda];
			//a[i+i*lda] = 1;
			//ReflectorApply(side, mi, ni, &a[i+i*lda], 1, taui, &c[ic+jc*ldc], ldc, work);
			//a[i+i*lda] = aii;
			
			Reflector::Apply(side, true, false, mi, ni, &a[i+i*lda], 1, taui, &c[ic+jc*ldc], ldc, work);
		}
	}
}

template <class T> // _ung2r
void QRGenerateQ_unblocked(size_t m, size_t n, size_t k, T *a, size_t lda, const T *tau, T *work){
	// Generates an m by n complex matrix Q with orthonormal columns,
	// which is defined as the first n columns of a product of k elementary
	// reflectors of order m
	//       Q  =  H[0] H[1] . . . H[k-1]
	// as returned by QRFactor.
	//
	// Arguments
	// =========
	//
	// m     The number of rows of the matrix Q. m >= 0
	//
	// n     The number of columns of the matrix Q. 0 <= n <= m
	//
	// k     The number of elementary reflectors whose product defines the
	//       matrix Q. 0 <= k <= n
	//
	// A     (input/output) array, dimension (LDA,N)
	//       On entry, the i-th column must contain the vector which
	//       defines the elementary reflector H[i], for i = 0,1,...,k-1, as
	//       returned by QRFactor in the first k columns of its array
	//       argument A.
	//       On exit, the m by n matrix Q.
	//
	// lda   (input) The leading dimension of the array A.
	//
	// tau   (input) array, length K
	//       tau[i] must contain the scalar factor of the elementary
	//       reflector H[i], as returned by QRFactor.
	//
	// work  (workspace) length n

	if(n < 1){ return; }

	// Initialise columns k+1:n to columns of the unit matrix
	for(size_t j = k; j < n; ++j){
		for(size_t i = 0; i < m; ++i){
			a[i+j*lda] = T(0);
		}
		a[j+j*lda] = T(1);
	}

	size_t i = k;
	while(i --> 0){
		// Apply H[i] to A(i:m,i:n) from the left
		if(i+1 < n){
			a[i+i*lda] = T(1);
			Reflector::Apply("L", false, false, m-i, n-i-1, &a[i+i*lda], 1, tau[i], &a[i+(i+1)*lda], lda, work);
		}
		if(i+1 < m){
			BLAS::Scale(m-i-1, -tau[i], &a[i+1+i*lda], 1);
		}
		a[i+i*lda] = T(1) - tau[i];

		// Set A(1:i-1,i) to zero
		for(size_t l = 0; l < i; ++l){
			a[l+i*lda] = T(0);
		}
	}
}

} // namespace LA
} // namespace RNP

#endif // RNP_QR_HPP_INCLUDED
