#ifndef RNP_QL_HPP_INCLUDED
#define RNP_QL_HPP_INCLUDED

#include <iostream>
#include <cstddef>
#include <RNP/BLAS.hpp>
#include "Reflector.hpp"

#include <iostream>

namespace RNP{
namespace LA{
namespace QL{

///////////////////////////////////////////////////////////////////////
// RNP::LA::QL
// ===========
// Computes the QL factorization and operations involving Q.
// For tall matrices, L is lower triangular. For fat matrices, L is
// lower trapezoidal.
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
	static size_t factor_block_size_opt(size_t m, size_t n){ return 32; }
	static size_t factor_block_size_min(size_t m, size_t n){ return 2; }
	static size_t factor_crossover_size(size_t m, size_t n){ return 128; }
	static size_t multQ_block_size_opt(const char *side, const char *trans, size_t m, size_t n, size_t k){ return 64; }
	static size_t multQ_block_size_min(const char *side, const char *trans, size_t m, size_t n, size_t k){ return 64; }
	static size_t genQ_block_size_opt(size_t m, size_t n, size_t k){ return 64; }
	static size_t genQ_block_size_min(size_t m, size_t n, size_t k){ return 64; }
	static size_t genQ_crossover_size(size_t m, size_t n, size_t k){ return 64; }
};


///////////////////////////////////////////////////////////////////////
// Factor_unblocked
// ----------------
// Computes a QR factorization of an m-by-n matrix A = Q * R.
// The matrix Q is represented as a product of elementary reflectors
//   Q = H[1] H[2] ... H[k-1], where k = min(m,n).
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
//      stores the vectors v of the elementary reflectors.
// lda  Leading dimension of the array containing A (lda >= m).
// tau  Output vector of tau's.
// work Workspace of size n.
//
template <typename T>
void Factor_unblocked(
	size_t m, size_t n, T *a, size_t lda, T *tau, T *work
){
	size_t k = m; if(n < k){ k = n; }
	size_t i = k; while(i --> 0){
		// Generate elementary reflector H[i] to annihilate A[0..m-k+i,n-k+i]
		T alpha(a[m-k+i+(n-k+i)*lda]);
		// Can replace with GeneratePositive if positive diagonal
		// elements in R are required.
		Reflector::Generate(
			m-k+i+1, &alpha, &a[0+(n-k+i)*lda], 1, &tau[i]
		);
		// Apply H[i]^H to A[0..m-k+i,0..n-k+i] from the left
		a[m-k+i+(n-k+i)*lda] = T(1);
		Reflector::Apply(
			"L", false, false, m-k+i+1, n-k+i, &a[0+(n-k+i)*lda], 1,
			Traits<T>::conj(tau[i]), a, lda, work
		);
		a[m-k+i+(n-k+i)*lda] = alpha;
	}
}

///////////////////////////////////////////////////////////////////////
// Factor
// ------
// Computes a QL factorization of an m-by-n matrix A = Q * L.
// The matrix Q is represented as a product of elementary reflectors
//   Q = H[1] H[2] ... H[k-1], where k = min(m,n).
// Each H[i] has the form
//   H[i] = I - tau * v * v^H
// where tau is a scalar, and v is a vector with v[0..i] = 0 and
// v[i] = 1; v[i+1..m] is stored, upon exit, in A[i+1..m,i], and
// tau in tau[i].
// Equivalent to Lapack routines _geqlf.
//
// Arguments
// m     Number of rows of the matrix A.
// n     Number of columns of the matrix A.
// a     Pointer to the first element of A. On exit, the lower
//       triangle of A contains the L factor, and the upper triangle
//       stores the vectors v of the elementary reflectors.
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
	const size_t k = (m < n ? m : n);
	if(0 == k){
		return;
	}
	size_t nb = QL::Tuning<T>::factor_block_size_opt(m, n);
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
		nx = QL::Tuning<T>::factor_crossover_size(m, n);
		if(nx < k){
			// Determine if workspace is large enough for blocked code.
            iws = ldwork * nb;
            if(*lwork < iws){
				// Not enough workspace to use optimal NB:  reduce NB and
				// determine the minimum value of NB.
				nb = *lwork / ldwork;
				nbmin = QL::Tuning<T>::factor_block_size_min(m, n);
				if(2 > nbmin){ nbmin = 2; }
			}
		}
	}
	
	// Layout of workspace:
	//  [ T ] T is the nb-by-nb block triangular factor
	//  [ W ] W is n-nb by n
	// Thus work is treated as dimension n-by-nb
	
	size_t mu, nu; // unblocked problem size

	if(nb >= nbmin && nb < k && nx < k){
		// Use blocked code initially; the last kk columns are blocked
		size_t ki = ((k-nx-1) / nb) * nb;
		size_t kk = (k < ki+nb ? k : ki+nb);
		size_t i = k - kk + ki;
		for(i = k-kk+ki; i >= k-kk; i -= nb){
			const size_t ib = (k < nb+i ? k-i : nb);
			// Compute the QR factorization of the current block A[i..m-k+i+ib,n-k+i:n-k+i+ib]
			Factor_unblocked(m-k+i+ib, ib, &a[0+(n-k+i)*lda], lda, &tau[i], work);
			if(n-k+i > 0){
				// Form the triangular factor of the block reflector
				//   H = H[i+ib-1] ...  H[i+1] H[i]
				// The triangular factor goes in work[0..nb,0..nb]
				
				Reflector::GenerateBlockTr(
					"B","C", m-k+i+ib, ib, &a[0+(n-k+i)*lda], lda, &tau[i], work, ldwork
				);
				
				// Apply H^H to A[i..m-k+i+ib,0..n-k+i] from the left
				Reflector::ApplyBlock(
					"L","C","B","C", m-k+i+ib, n-k+i, ib, &a[0+(n-k+i)*lda], lda,
					work, ldwork, a, lda, &work[ib+0*ldwork], ldwork
				);
			}
		}
		mu = m-k+i+nb;
		nu = n-k+i+nb;
	}else{
		mu = m;
		nu = n;
	}
	if(mu > 0 && nu > 0){
		Factor_unblocked(mu, nu, a, lda, tau, work);
	}
}



///////////////////////////////////////////////////////////////////////
// MultQ_unblocked
// ---------------
// From an existing QL factorization, multiplies a given matrix by the
// unitary matrix Q. The given m-by-n matrix C is overwritten with:
//
//    trans | side = "L"   | side = "R"
//    ------|--------------|------------
//     "N"  |   Q   * C    |  C * Q
//     "C"  |   Q^H * C    |  C * Q^H
//
// This routine uses only level 2 BLAS.
// Equivalent to Lapack routines _unm2l and _orm2l.
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
//       i = 1..k.
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
	const bool left = ('L' == side[0]);
	const bool notran = ('N' == trans[0]);

	if(m < 1 || n < 1 || k < 1){ return; }
	
	size_t mi, ni;
	if(left){
		ni = n;
	}else{
		mi = m;
	}
	
	if((left && notran) || (!left && !notran)){
		// loop forwards
		for(size_t i = 0; i < k; ++i){
			if(left){ // H[i] or H[i]' is applied to C[0..m-k+i+1,0..n]
				mi = m-k+i+1;
			}else{ // H[i] or H[i]' is applied to C[0..m,0..n-k+i+1]
				ni = n-k+i+1;
			}

			T taui;
			if(notran){
				taui = tau[i];
			}else{
				taui = Traits<T>::conj(tau[i]);
			}
			
			Reflector::Apply(side, -1, false, mi, ni, &a[0+i*lda], 1, taui, c, ldc, work);
		}
	}else{
		size_t i = k; while(i --> 0){
			if(left){ // H[i] or H[i]' is applied to C[0..m-k+i+1,0..n]
				mi = m-k+i+1;
			}else{ // H[i] or H[i]' is applied to C[0..m,0..n-k+i+1]
				ni = n-k+i+1;
			}

			T taui;
			if(notran){
				taui = tau[i];
			}else{
				taui = Traits<T>::conj(tau[i]);
			}
			Reflector::Apply(side, -1, false, mi, ni, &a[0+i*lda], 1, taui, c, ldc, work);
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
//       i = 0..k.
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
	RNPAssert((left && k <= m) || (!left && k <= n));
	RNPAssert((left && lda >= m) || (!left && lda >= n));
	const bool notran = ('N' == trans[0]);
	const size_t nq = (left ? m : n);
	const size_t nw = (left ? n : m);
	
	size_t nb = QL::Tuning<T>::multQ_block_size_opt(side, trans, m, n, k);
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
			nbmin = QL::Tuning<T>::multQ_block_size_min(side, trans, m, n, k);
		}
	}
	
	if(nb < nbmin || nb >= k){ // unblocked
		MultQ_unblocked(side, trans, m, n, k, a,lda, tau, c, ldc, work);
	}else{
		size_t ni, mi;
		if(left){
			ni = n;
		}else{
			mi = m;
		}
		if((left && notran) || (!left && !notran)){	// loop forwards
			for(size_t i = 0; i < k; i += nb){
				size_t ib = (k < nb+i ? k-i : nb);
				Reflector::GenerateBlockTr("B","C", nq-k+i+ib, ib, &a[0+i*lda], lda, &tau[i], t, ldt);

				if(left){
					mi = m-k+i+ib;
				}else{
					ni = n-k+i+ib;
				}
				Reflector::ApplyBlock(
					side, trans, "B", "C", mi, ni, ib,
					&a[0+i*lda], lda, t, ldt, c, ldc, work, ldwork
				);
			}
		}else{
			size_t i = ((k-1)/nb)*nb;
			do{
				size_t ib = (k < nb+i ? k-i : nb);
				Reflector::GenerateBlockTr("B","C", nq-k+i+ib, ib, &a[0+i*lda], lda, &tau[i], t, ldt);

				if(left){
					mi = m-k+i+ib;
				}else{
					ni = n-k+i+ib;
				}
				Reflector::ApplyBlock(
					side, trans, "B", "C", mi, ni, ib,
					&a[0+i*lda], lda, t, ldt, c, ldc, work, ldwork
				);
				if(0 == i){ break; }
				i -= nb;
			}while(1);
		}
	}
}

///////////////////////////////////////////////////////////////////////
// GenerateQ
// ---------
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
//       i = 0..k. On exit, the matrix Q.
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
	if(n < 1){ return; }

	// Initialise columns 0..n-k to columns of the unit matrix
	for(size_t j = 0; j+k < n; ++j){
		for(size_t i = 0; i < m; ++i){
			a[i+j*lda] = T(0);
		}
		a[m-n+j+j*lda] = T(1);
	}

	for(size_t i = 0; i < k; ++i){
		size_t ii = n-k+i;
		// Apply H[i] to A(i:m-k+i,i:n-k+i) from the left
		a[m-n+ii+ii*lda] = T(1);
		Reflector::Apply("L", 0, false, m-n+ii+1, ii, &a[0+ii*lda], 1, tau[i], a, lda, work);
		BLAS::Scale(m-n+ii, -tau[i], &a[0+ii*lda], 1);
		a[m-n+ii+ii*lda] = T(1) - tau[i];

		// Set A(m-k+i+1:m,n-k+i) to zero
		for(size_t l = m-n+ii+1; l < m; ++l){
			a[l+ii*lda] = T(0);
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
//       i = 0..k. On exit, the matrix Q.
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
	
	size_t kk = 0;
	if(nb >= nbmin && nb < k && nx < k){ // use blocked code after first block
		// last kk columns are handed by blocked method
		const size_t ki = ((k-nx+nb-1)/nb) * nb;
		kk = (k < ki ? k : ki);
		if(kk < m){
			for(size_t j = 0; j+kk < n; ++j){
				for(size_t i = m-kk; i < m; ++i){
					a[i+j*lda] = T(0);
				}
			}
		}
	}

	GenerateQ_unblocked(m-kk, n-kk, k-kk, a, lda, tau, work);
	
	if(kk > 0){
		for(size_t i = k-kk; i < k; i += nb){
			const size_t ib = (nb+i < k ? nb : k-i);
			if(n+i > k){
				Reflector::GenerateBlockTr("B", "C", m-k+i+ib, ib, &a[0+(n-k+i)*lda], lda, &tau[i], work, ldwork);
				Reflector::ApplyBlock("L","N","B","C", m-k+i+ib, n-k+i, ib, &a[0+(n-k+i)*lda], lda, work, ldwork, a, lda, &work[ib], ldwork);
			}
			// Apply H to rows 0..m of current block
			GenerateQ_unblocked(m-k+i+ib, ib, ib, &a[0+(n-k+i)*lda], lda, &tau[i], work);
			// Set rows m-k+i+ib..m of current block to zero
			for(size_t j = n-k+i; j+k < n+i+ib; ++j){
				for(size_t l = m-k+i+ib; l < m; ++l){
					a[l+j*lda] = T(0);
				}
			}
		}
	}
}

} // namespace QL
} // namespace LA
} // namespace RNP

#endif // RNP_QL_HPP_INCLUDED
