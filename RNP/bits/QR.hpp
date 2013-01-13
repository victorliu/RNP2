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
	static size_t factor_block_size_opt(size_t m, size_t n){ return 64; }
	static size_t factor_block_size_min(size_t m, size_t n){ return 64; }
	static size_t factor_crossover_size(size_t m, size_t n){ return 64; }
	static size_t multQ_block_size_opt(const char *side, const char *trans, size_t m, size_t n, size_t k){ return 64; }
	static size_t multQ_block_size_min(const char *side, const char *trans, size_t m, size_t n, size_t k){ return 64; }
	static size_t genQ_block_size_opt(size_t m, size_t n, size_t k){ return 64; }
	static size_t genQ_block_size_min(size_t m, size_t n, size_t k){ return 64; }
	static size_t genQ_crossover_size(size_t m, size_t n, size_t k){ return 64; }
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
	size_t nb = QR::Tuning<T>::factor_block_size_opt(m, n);
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
		nx = QR::Tuning<T>::factor_crossover_size(m, n);
		if(nx < k){
			// Determine if workspace is large enough for blocked code.
            iws = ldwork * nb;
            if(*lwork < iws){
				// Not enough workspace to use optimal NB:  reduce NB and
				// determine the minimum value of NB.
				nb = *lwork / ldwork;
				nbmin = QR::Tuning<T>::factor_block_size_min(m, n);
				if(2 > nbmin){ nbmin = 2; }
			}
		}
	}
	
	// Layout of workspace:
	//  [ T ] T is the nb-by-nb block triangular factor
	//  [ W ] W is n-nb by n
	// Thus work is treated as dimension n-by-nb
	
	size_t i;
	if(nb >= nbmin && nb < k && nx < k){
		// Use blocked code initially
		for(i = 0; i+nx < k; i += nb){
			const size_t ib = (k-i < nb ? k-i : nb);
			// Compute the QR factorization of the current block A[i..m,i..i+ib]
			QRFactor_unblocked(m-i, ib, &a[i+i*lda], lda, &tau[i], work);
			if(i+ib < n){
				// Form the triangular factor of the block reflector
				//   H = H[i] H[i+1] ...  H[i+ib-1]
				// The triangular factor goes in work[0..nb,0..nb]
				Reflector::GenerateBlockTr(
					"F","C", m-i, ib, &a[i+i*lda], lda, &tau[i], work, ldwork
				);
				// Apply H^H to A[i..m,i+ib..n] from the left
				Reflector::ApplyBlock(
					"L","C","F","C", m-i, n-i-ib, ib, &a[i+i*lda], lda,
					work, ldwork, &a[i+(i+ib)*lda], lda, &work[ib+0*ldwork], ldwork
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

// Unblocked version of QRMultQ, work is length:
//   n if side is L, m if side is R
template <typename T> // _unmr2, _ormr2
void QRMultQ_unblocked(
	const char *side, const char *trans, size_t m, size_t n, size_t k,
	const T *a, size_t lda, const T *tau, T *c, size_t ldc, T *work
){
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
			if(left){ // H[i] or H[i]' is applied to C[i..m,0..n]
				mi = m - i;
				ic = i;
			}else{ // H[i] or H[i]' is applied to C[0..m,i..n]
				ni = n - i;
				jc = i;
			}

			T taui;
			if(notran){
				taui = tau[i];
			}else{
				taui = Traits<T>::conj(tau[i]);
			}
			
			Reflector::Apply(side, true, false, mi, ni, &a[i+i*lda], 1, taui, &c[ic+jc*ldc], ldc, work);
		}
	}else{
		// loop backwards
		size_t i = k; while(i --> 0){
			if(left){ // H[i] or H[i]' is applied to C[i..m,0:n]
				mi = m - i;
				ic = i;
			}else{ // H[i] or H[i]' is applied to C[0..m,i..n]
				ni = n - i;
				jc = i;
			}

			T taui;
			if(notran){
				taui = tau[i];
			}else{
				taui = Traits<T>::conj(tau[i]);
			}
			
			Reflector::Apply(side, true, false, mi, ni, &a[i+i*lda], 1, taui, &c[ic+jc*ldc], ldc, work);
		}
	}
}

template <typename T> // _unmqr, _ormqr
void QRMultQ(
	const char *side, const char *trans, size_t m, size_t n, size_t k,
	const T *a, size_t lda, const T *tau, T *c, size_t ldc,
	size_t *lwork, T *work
){
	if(0 == m || 0 == n || 0 == k){ return; }
	const bool left = ('L' == side[0]);
	const bool notran = ('N' == trans[0]);
	const size_t nq = (left ? m : n);
	const size_t nw = (left ? n : m);
	
	size_t nb = QR::Tuning<T>::multQ_block_size_opt(side, trans, m, n, k);
	if(0 == *lwork || NULL == work){
		*lwork = nb * nw + nb*nb;
		return;
	}
	
	size_t nbmin = 2;
	size_t ldwork = nw;
	size_t iws = nw;
	T *t = NULL;
	size_t ldt = nb;
	if(nb > 1 && nb < k){
		iws = nw*nb + nb*nb;
		t = work + nb*ldwork;
		if(*lwork < iws){
			// We want to solve for nb in the equation
			//   *lwork == nb*nb + nb*nw
			// A lower bound on nb is simply *lwork/(2*nw)
			// An upper bound is *lwork/nw, so we have bounds within a
			// factor 2 of being optimal.
			nb = *lwork / (2*ldwork);
			t = work + nb*ldwork;
			ldt = nb;
			nbmin = QR::Tuning<T>::multQ_block_size_min(side, trans, m, n, k);
		}
	}
	
	if(nb < nbmin || nb >= k){ // unblocked
		QRMultQ_unblocked(side, trans, m, n, k, a,lda, tau, c, ldc, work);
	}else{
		size_t ni, mi, ic, jc;
		if(left){
			ni = n;
			jc = 0;
		}else{
			mi = m;
			ic = 0;
		}
		if((left && !notran) || (!left && notran)){	// loop forwards
			for(size_t i = 0; i < k; i += nb){
				size_t ib = (k < nb+i ? k-i : nb);
				Reflector::GenerateBlockTr("F","C", nq-i, ib, &a[i+i*lda], lda, &tau[i], t, ldt);

				if(left){
					mi = m-i;
					ic = i;
				}else{
					ni = n-i;
					jc = i;
				}
				Reflector::ApplyBlock(
					side, trans, "F", "C", mi, ni, ib,
					&a[i+i*lda], lda, t, ldt, &c[ic+jc*ldc], ldc, work, ldwork
				);
			}
		}else{
			size_t i = (k/nb)*nb;
			do{
				size_t ib = (k < nb+i ? k-i : nb);
				Reflector::GenerateBlockTr("F","C", nq-i, ib, &a[i+i*lda], lda, &tau[i], t, ldt);

				if(left){
					mi = m-i;
					ic = i;
				}else{
					ni = n-i;
					jc = i;
				}
				Reflector::ApplyBlock(
					side, trans, "F", "C", mi, ni, ib,
					&a[i+i*lda], lda, t, ldt, &c[ic+jc*ldc], ldc, work, ldwork
				);
				if(0 == i){ break; }
				i -= nb;
			}while(1);
		}
	}
}

template <class T> // _ung2r
void QRGenerateQ_unblocked(
	size_t m, size_t n, size_t k, T *a, size_t lda, const T *tau, T *work
){
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

template <class T> // _ungqr
void QRGenerateQ(
	size_t m, size_t n, size_t k, T *a, size_t lda,
	const T *tau, size_t *lwork, T *work
){
	RNPAssert(k <= n);
	RNPAssert(lda >= m);
	if(0 == n){ return; }
	
	size_t nb = QR::Tuning<T>::genQ_block_size_opt(m, n, k);
	if(0 == *lwork || NULL == work){
		*lwork = n*nb;
		return;
	}
	
	size_t nbmin = 2;
	size_t nx = 0;
	size_t iws = n;
	size_t ldwork = n;
	if(nb > 1 && nb < k){
		nx = QR::Tuning<T>::genQ_crossover_size(m, n, k);
		if(nx < k){
			iws = ldwork*nb;
			if(*lwork < iws){
				nb = *lwork / ldwork;
				nbmin = QR::Tuning<T>::genQ_block_size_min(m, n, k);
			}
		}
	}
	
	size_t ki = 0, kk = 0;
	if(nb >= nbmin && nb < k && nx < k){ // use blocked code after last block
		ki = ((k-nx-1)/nb) * nb;
		kk = (k < ki+nb ? k : ki+nb);
		for(size_t j = kk; j < n; ++j){
			for(size_t i = 0; i < kk; ++i){
				a[i+j*lda] = T(0);
			}
		}
	}
	//ki = nb;
	if(kk < n){
		QRGenerateQ_unblocked(m-kk, n-kk, k-kk, &a[kk+kk*lda], lda, &tau[kk], work);
	}
	if(kk > 0){
		size_t i = ki;
		do{
			const size_t ib = (nb+i < k ? nb : k-i);
			if(i+ib < n){
				Reflector::GenerateBlockTr("F", "C", m-i, ib, &a[i+i*lda], lda, &tau[i], work, ldwork);
				Reflector::ApplyBlock("L","N","F","C", m-i, n-i-ib, ib, &a[i+i*lda], lda, work, ldwork, &a[i+(i+ib)*lda], lda, &work[ib], ldwork);
			}
			// Apply H to rows 0..m of current block
			QRGenerateQ_unblocked(m-i, ib, ib, &a[i+i*lda], lda, &tau[i], work);
			// Set rows 0..i of current block to zero
			for(size_t j = i; j < i+ib; ++j){
				for(size_t l = 0; l < i; ++l){
					a[l+j*lda] = T(0);
				}
			}
			if(0 == i){ break; }else{ i -= nb; }
		}while(1);
	}
}

} // namespace LA
} // namespace RNP

#endif // RNP_QR_HPP_INCLUDED
