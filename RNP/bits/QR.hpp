#ifndef RNP_QR_HPP_INCLUDED
#define RNP_QR_HPP_INCLUDED

#include <cstddef>
#include <RNP/BLAS.hpp>
#include "Reflector.hpp"

namespace RNP{
namespace LA{
namespace QR{

///////////////////////////////////////////////////////////////////////
// RNP::LA::QR
// ===========
// Computes the QR factorization and operations involving Q.
// For tall matrices, R is upper triangular. For fat matrices, R is
// upper trapezoidal. The decomposition is a product A = Q * R, with Q
// packed into the lower triangle of A, and an additional tau vector
// representing the scale factors of the reflector representation of Q.
// The storate scheme is shown below, with 'A' representing elements of
// the matrix A, 'R' representing elements of the the R factor, 'Q'
// representing elements of the reflector vectors which implicitly form
// Q, and 'T' the elements of the auxiliary array tau.
//
//     A A A   R R R          A A A A A   R R R R R  T
//     A A A   Q R R          A A A A A = Q R R R R  T
//     A A A = Q Q R  T       A A A A A   Q Q R R R  T
//     A A A   Q Q Q  T
//     A A A   Q Q Q  T
//
// When m >= n, Q is m-by-n and R is n-by-n upper triangular.
// When m < n, Q is m-by-m and R is m-by-n upper trapezoidal.

///////////////////////////////////////////////////////////////////////
// Tuning
// ------
// Specialize this class to tune the block sizes. The optimal block
// size should be greater than or equal to the minimum block size.
// The value of the crossover determines when to enable blocking.
//
template <typename T>
struct Tuning{
	static size_t factor_block_size_opt(size_t m, size_t n){ return 32; }
	static size_t factor_block_size_min(size_t m, size_t n){ return 2; }
	static size_t factor_crossover_size(size_t m, size_t n){ return 128; }
	static size_t multQ_block_size_opt(const char *side, const char *trans, size_t m, size_t n, size_t k){ return 32; }
	static size_t multQ_block_size_min(const char *side, const char *trans, size_t m, size_t n, size_t k){ return 2; }
	static size_t genQ_block_size_opt(size_t m, size_t n, size_t k){ return 32; }
	static size_t genQ_block_size_min(size_t m, size_t n, size_t k){ return 2; }
	static size_t genQ_crossover_size(size_t m, size_t n, size_t k){ return 128; }
};


///////////////////////////////////////////////////////////////////////
// Factor_unblocked
// ----------------
// Computes a QR factorization of an m-by-n matrix A = Q * R.
// The matrix Q is represented as a product of elementary reflectors
//   Q = H[0] H[1] ... H[k-1], where k = min(m,n).
// Each H[i] has the form
//   H[i] = I - tau * v * v^H
// where tau is a scalar, and v is a vector with v[0..i] = 0 and
// v[i] = 1; v[i+1..m] is stored, upon exit, in A[i+1..m,i], and
// tau in tau[i].
// This routine uses only level 2 BLAS.
// Equivalent to Lapack routines _geqr2.
//
// Arguments
// m    Number of rows of the matrix A.
// n    Number of columns of the matrix A.
// a    Pointer to the first element of A. On exit, the upper
//      triangle of A contains the R factor, and the lower triangle
//      stores the vectors v of the elementary reflectors in columns.
// lda  Leading dimension of the array containing A (lda >= m).
// tau  Output vector of tau's.
// work Workspace of size n.
//
template <typename T>
void Factor_unblocked(
	size_t m, size_t n, T *a, size_t lda, T *tau, T *work
){
	size_t k = m; if(n < k){ k = n; }
	for(size_t i = 0; i < k; ++i){
		// Generate elementary reflector H[i] to annihilate A[i+1..m,i]
		size_t row = m-1; if(i+1 < row){ row = i+1; }
		// Can replace with GeneratePositive if positive diagonal
		// elements in R are required.
		Reflector::Generate(
			m-i, &a[i+i*lda], &a[row+i*lda], 1, &tau[i]
		);
		if(i < n-1){
			// Apply H[i]^H to A[i..m,i+1..n] from the left
			T alpha = a[i+i*lda];
			a[i+i*lda] = T(1);
			Reflector::Apply(
				"L", 0, false, m-i, n-i-1, &a[i+i*lda], 1,
				Traits<T>::conj(tau[i]), &a[i+(i+1)*lda], lda, work
			);
			a[i+i*lda] = alpha;
		}
	}
}

///////////////////////////////////////////////////////////////////////
// Factor
// ------
// Computes a QR factorization of an m-by-n matrix A = Q * R.
// The matrix Q is represented as a product of elementary reflectors
//   Q = H[0] H[1] ... H[k-1], where k = min(m,n).
// Each H[i] has the form
//   H[i] = I - tau * v * v^H
// where tau is a scalar, and v is a vector with v[0..i] = 0 and
// v[i] = 1; v[i+1..m] is stored, upon exit, in A[i+1..m,i], and
// tau in tau[i].
// Equivalent to Lapack routines _geqrf.
//
// Arguments
// m     Number of rows of the matrix A.
// n     Number of columns of the matrix A.
// a     Pointer to the first element of A. On exit, the upper
//       triangle of A contains the R factor, and the lower triangle
//       stores the vectors v of the elementary reflectors in columns.
// lda   Leading dimension of the array containing A (lda >= m).
// tau   Output vector of tau's.
// lwork Length of workspace (>= n). If *lwork == 0 or NULL == work,
//       then the optimal size is returned in this argument.
// work  Workspace of size lwork.
//
template <typename T>
void Factor(
	size_t m, size_t n, T *a, size_t lda, T *tau, size_t *lwork, T *work
){
	RNPAssert(lda >= m);
	RNPAssert(NULL != lwork);
	RNPAssert(0 == *lwork || *lwork >= m);
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
			Factor_unblocked(m-i, ib, &a[i+i*lda], lda, &tau[i], work);
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
		Factor_unblocked(m-i, n-i, &a[i+i*lda], lda, &tau[i], work);
	}
}



///////////////////////////////////////////////////////////////////////
// MultQ_unblocked
// ---------------
// From an existing QR factorization, multiplies a given matrix by the
// unitary matrix Q. The given m-by-n matrix C is overwritten with:
//
//    trans | side = "L"   | side = "R"
//    ------|--------------|------------
//     "N"  |   Q   * C    |  C * Q
//     "C"  |   Q^H * C    |  C * Q^H
//
// This routine uses only level 2 BLAS.
// Equivalent to Lapack routines _unm2r and _orm2r.
//
// Arguments
// side  If "L", apply Q or Q^H from the left. If "R", apply Q or
//       Q^H from the right.
// trans If "N", apply Q. If "C", apply Q'.
// m     Number of rows of the matrix C.
// n     Number of columns of the matrix C.
// k     Number of elementary reflectors to apply.
//       If side = "L", k <= m. If side = "R", k <= n.
// a     Pointer to the factorization. The i-th column should contain
//       the vector which defines the i-th elementary reflector for
//       i = 0..k-1.
// lda   Leading dimension of the array containing A.
//       If side = "L", lda >= m. If side = "R", lda >= n.
// tau   Array of tau's, length k.
// c     Pointer to the first element of the matrix C.
// ldc   Leading dimension of the array containing C.
// work  Workspace.
//       If side = "L", length n. If side = "R", length m.
//
template <typename T>
void MultQ_unblocked(
	const char *side, const char *trans, size_t m, size_t n, size_t k,
	const T *a, size_t lda, const T *tau, T *c, size_t ldc, T *work
){
	RNPAssert(ldc >= m);
	const bool left = ('L' == side[0]);
	const bool notran = ('N' == trans[0]);
	RNPAssert((left && k <= m) || (!left && k <= n));
	RNPAssert(lda >= k);

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
			
			Reflector::Apply(side, 1, false, mi, ni, &a[i+i*lda], 1, taui, &c[ic+jc*ldc], ldc, work);
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
			
			Reflector::Apply(side, 1, false, mi, ni, &a[i+i*lda], 1, taui, &c[ic+jc*ldc], ldc, work);
		}
	}
}



///////////////////////////////////////////////////////////////////////
// MultQ
// -----
// From an existing QR factorization, multiplies a given matrix by the
// unitary matrix Q. The given m-by-n matrix C is overwritten with:
//
//    trans | side = "L"   | side = "R"
//    ------|--------------|------------
//     "N"  |   Q   * C    |  C * Q
//     "C"  |   Q^H * C    |  C * Q^H
//
// Equivalent to Lapack routines _unmqr and _ormqr.
//
// Arguments
// side  If "L", apply Q or Q^H from the left. If "R", apply Q or
//       Q^H from the right.
// trans If "N", apply Q. If "C", apply Q'.
// m     Number of rows of the matrix C.
// n     Number of columns of the matrix C.
// k     Number of elementary reflectors to apply.
//       If side = "L", k <= m. If side = "R", k <= n.
// a     Pointer to the factorization. The i-th column should contain
//       the vector which defines the i-th elementary reflector for
//       i = 0..k-1.
// lda   Leading dimension of the array containing A.
//       If side = "L", lda >= m. If side = "R", lda >= n.
// tau   Array of tau's, length k.
// c     Pointer to the first element of the matrix C.
// ldc   Leading dimension of the array containing C.
// lwork Lenth of workspace.
//       If side = "L", lwork >= n. If side = "R", lwork >= m.
//       If *lwork == 0 or NULL == work, then the optimal size is
//       returned in this argument.
// work  Workspace of size lwork.
//
template <typename T>
void MultQ(
	const char *side, const char *trans, size_t m, size_t n, size_t k,
	const T *a, size_t lda, const T *tau, T *c, size_t ldc,
	size_t *lwork, T *work
){
	RNPAssert(ldc >= m);
	if(0 == m || 0 == n || 0 == k){ return; }
	const bool left = ('L' == side[0]);
	const bool notran = ('N' == trans[0]);
	
	RNPAssert((left && k <= m) || (!left && k <= n));
	RNPAssert(lda >= k);
	
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
		MultQ_unblocked(side, trans, m, n, k, a,lda, tau, c, ldc, work);
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
			size_t i = ((k-1)/nb)*nb + nb;
			do{ i -= nb;
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
			}while(i > 0);
		}
	}
}

///////////////////////////////////////////////////////////////////////
// GenerateQ_unblocked
// -------------------
// From an existing QR factorization, generates the unitary matrix Q.
// The original matrix containing the factorization is overwritten
// by Q.
// This routine uses only level 2 BLAS.
// Equivalent to Lapack routines _ung2r and _org2r.
//
// Arguments
// m     Number of rows of the matrix Q.
// n     Number of columns of the matrix Q, m >= n.
// k     Number of elementary reflectors, k <= n.
// a     Pointer to the factorization. The i-th column should contain
//       the vector which defines the i-th elementary reflector for
//       i = 0..k-1. On exit, the matrix Q.
// lda   Leading dimension of the array containing Q, lda >= m.
// tau   Array of tau's, length k.
// c     Pointer to the first element of the matrix C.
// ldc   Leading dimension of the array containing C.
// work  Workspace of size n.
//
template <class T>
void GenerateQ_unblocked(
	size_t m, size_t n, size_t k, T *a, size_t lda, const T *tau, T *work
){
	RNPAssert(m >= n);
	RNPAssert(k <= n);
	RNPAssert(lda >= m);
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
			Reflector::Apply("L", 0, false, m-i, n-i-1, &a[i+i*lda], 1, tau[i], &a[i+(i+1)*lda], lda, work);
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

///////////////////////////////////////////////////////////////////////
// GenerateQ
// ---------
// From an existing QR factorization, generates the unitary matrix Q.
// The original matrix containing the factorization is overwritten
// by Q.
// Equivalent to Lapack routines _ungqr and _orgqr.
//
// Arguments
// m     Number of rows of the matrix Q.
// n     Number of columns of the matrix Q, m >= n.
// k     Number of elementary reflectors, k <= n.
// a     Pointer to the factorization. The i-th column should contain
//       the vector which defines the i-th elementary reflector for
//       i = 0..k-1. On exit, the matrix Q.
// lda   Leading dimension of the array containing Q, lda >= m.
// tau   Array of tau's, length k.
// c     Pointer to the first element of the matrix C.
// ldc   Leading dimension of the array containing C.
// lwork Lenth of workspace, lwork >= n.
//       If *lwork == 0 or NULL == work, then the optimal size is
//       returned in this argument.
// work  Workspace of size lwork.
//
template <class T>
void GenerateQ(
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
	if(kk < n){
		GenerateQ_unblocked(m-kk, n-kk, k-kk, &a[kk+kk*lda], lda, &tau[kk], work);
	}
	if(kk > 0){
		size_t i = ki + nb;
		do{ i -= nb;
			const size_t ib = (nb+i < k ? nb : k-i);
			if(i+ib < n){
				Reflector::GenerateBlockTr("F", "C", m-i, ib, &a[i+i*lda], lda, &tau[i], work, ldwork);
				Reflector::ApplyBlock("L","N","F","C", m-i, n-i-ib, ib, &a[i+i*lda], lda, work, ldwork, &a[i+(i+ib)*lda], lda, &work[ib], ldwork);
			}
			// Apply H to rows 0..m of current block
			GenerateQ_unblocked(m-i, ib, ib, &a[i+i*lda], lda, &tau[i], work);
			// Set rows 0..i of current block to zero
			for(size_t j = i; j < i+ib; ++j){
				for(size_t l = 0; l < i; ++l){
					a[l+j*lda] = T(0);
				}
			}
		}while(i > 0);
	}
}

} // namespace QR
} // namespace LA
} // namespace RNP

#endif // RNP_QR_HPP_INCLUDED
