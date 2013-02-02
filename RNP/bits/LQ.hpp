#ifndef RNP_LQ_HPP_INCLUDED
#define RNP_LQ_HPP_INCLUDED

#include <cstddef>
#include <RNP/BLAS.hpp>
#include "Reflector.hpp"

namespace RNP{
namespace LA{
namespace LQ{

///////////////////////////////////////////////////////////////////////
// RNP::LA::LQ
// ===========
// Computes the LQ factorization and operations involving Q.
// For tall matrices, L is lower trapezoidal. For fat matrices, L is
// lower triangular. The decomposition is a product A = L * Q, with Q
// packed into the upper triangle of A, and an additional tau vector
// representing the scale factors of the reflector representation of Q.
// The storate scheme is shown below, with 'A' representing elements of
// the matrix A, 'L' representing elements of the the L factor, 'Q'
// representing elements of the reflector vectors which implicitly form
// Q, and 'T' the elements of the auxiliary array tau.
//
//     A A A   L Q Q          A A A A A   L Q Q Q Q  T
//     A A A   L L Q          A A A A A = L L Q Q Q  T
//     A A A = L L L  T       A A A A A   L L L Q Q  T
//     A A A   L L L  T
//     A A A   L L L  T
//
// When m >= n, L is m-by-n lower trapezoidal and Q is n-by-n.
// When m < n, L is m-by-m lower triangular and Q is m-by-n.

///////////////////////////////////////////////////////////////////////
// Tuning
// ------
// Specialize this class to tune the block sizes. The optimal block
// size should be greater than or equal to the minimum block size.
// The value of the crossover determines when to enable blocking.
//
template <typename T>
struct Tuning{
	static inline size_t factor_block_size_opt(size_t m, size_t n){ return 32; }
	static inline size_t factor_block_size_min(size_t m, size_t n){ return 2; }
	static inline size_t factor_crossover_size(size_t m, size_t n){ return 128; }
	static inline size_t multQ_block_size_opt(const char *side, const char *trans, size_t m, size_t n, size_t k){ return 32; }
	static inline size_t multQ_block_size_min(const char *side, const char *trans, size_t m, size_t n, size_t k){ return 2; }
	static size_t genQ_block_size_opt(size_t m, size_t n, size_t k){ return 32; }
	static size_t genQ_block_size_min(size_t m, size_t n, size_t k){ return 2; }
	static size_t genQ_crossover_size(size_t m, size_t n, size_t k){ return 128; }
};

///////////////////////////////////////////////////////////////////////
// Factor_unblocked
// ----------------
// Computes an LQ factorization of an m-by-n matrix A = L * Q.
// The matrix Q is represented as a product of elementary reflectors
//   Q = H[0] H[1] ... H[k-1], where k = min(m,n).
// Each H[i] has the form
//   H[i] = I - tau * v * v^H
// where tau is a scalar, and v is a vector with v[0..i] = 0 and
// v[i] = 1; v[i+1..n]' is stored, upon exit, in A[i,i+1..n], and
// tau in tau[i].
// This routine uses only level 2 BLAS.
// Equivalent to Lapack routines _gelq2.
//
// Arguments
// m    Number of rows of the matrix A.
// n    Number of columns of the matrix A.
// a    Pointer to the first element of A. On exit, the lower
//      triangle of A contains the L factor (lower triangular from
//      the bottom right of the matrix), and the upper triangle
//      stores the vectors v of the elementary reflectors in rows.
// lda  Leading dimension of the array containing A (lda >= m).
// tau  Output vector of tau's.
// work Workspace of size m.
//
template <typename T>
void Factor_unblocked(
	size_t m, size_t n,
	T *a, size_t lda,
	T *tau,
	T *work // length m
){
	const size_t k = (m < n ? m : n);
	for(size_t i = 0; i < k; ++i){
		BLAS::Conjugate(n-i, &a[i+i*lda], lda);
		T alpha = a[i+i*lda];
		Reflector::Generate(n-i, &alpha, &a[i+(i+1)*lda], lda, &tau[i]);
		if(i+1 < m){
			a[i+i*lda] = T(1);
			Reflector::Apply("R", 0, false, m-i-1, n-i, &a[i+i*lda], lda, tau[i], &a[i+1+i*lda], lda, work);
		}
		a[i+i*lda] = alpha;
		BLAS::Conjugate(n-i, &a[i+i*lda], lda);
	}
}

///////////////////////////////////////////////////////////////////////
// Factor
// ------
// Computes an LQ factorization of an m-by-n matrix A = L * Q.
// The matrix Q is represented as a product of elementary reflectors
//   Q = H[0] H[1] ... H[k-1], where k = min(m,n).
// Each H[i] has the form
//   H[i] = I - tau * v * v^H
// where tau is a scalar, and v is a vector with v[0..i] = 0 and
// v[i] = 1; v[i+1..n]' is stored, upon exit, in A[i,i+1..n], and
// tau in tau[i].
// This routine uses only level 2 BLAS.
// Equivalent to Lapack routines _gelqf.
//
// Arguments
// m     Number of rows of the matrix A.
// n     Number of columns of the matrix A.
// a     Pointer to the first element of A. On exit, the lower
//       triangle of A contains the L factor (lower triangular from
//       the bottom right of the matrix), and the upper triangle
//       stores the vectors v of the elementary reflectors in rows.
// lda   Leading dimension of the array containing A (lda >= m).
// tau   Output vector of tau's.
// lwork Length of workspace (>= m). If *lwork == 0 or NULL == work,
//       then the optimal size is returned in this argument.
// work  Workspace of size lwork.
//
template <typename T>
void Factor(size_t m, size_t n, T *a, size_t lda, T *tau, size_t *lwork, T *work){
	RNPAssert(lda >= m);
	RNPAssert(NULL != lwork);
	RNPAssert(0 == *lwork || *lwork >= m);
	const size_t k = (m < n ? m : n);
	if(0 == k){
		return;
	}
	size_t nb = LQ::Tuning<T>::factor_block_size_opt(m, n);
	if(NULL == work || 0 == *lwork){
		*lwork = nb*n;
		return;
	}
	size_t nbmin = 2;
	size_t nx = 0;
	size_t iws = n;
	size_t ldwork = m;
	if(nb > 1 && nb < k){
		// Determine when to cross over from blocked to unblocked code.
		nx = LQ::Tuning<T>::factor_crossover_size(m, n);
		if(nx < k){
			// Determine if workspace is large enough for blocked code.
            iws = ldwork * nb;
            if(*lwork < iws){
				// Not enough workspace to use optimal NB:  reduce NB and
				// determine the minimum value of NB.
				nb = *lwork / ldwork;
				nbmin = LQ::Tuning<T>::factor_block_size_min(m, n);
				if(2 > nbmin){ nbmin = 2; }
			}
		}
	}
	size_t i = 0;
	if(nb >= nbmin && nb < k && nx < k){
		// Use blocked code initially
		for(i = 0; i+nx < k; i += nb){
			const size_t ib = (k < nb+i ? k-i : nb);
			// Compute the LQ factorization of the current block A(i:m,i:i+ib-1)
			Factor_unblocked(ib, n-i, &a[i+i*lda], lda, &tau[i], work);
			if(i+ib < m){
				// Form the triangular factor of the block reflector
				// H = H[i] H[i+1] ... H[i+ib-1]
				Reflector::GenerateBlockTr(
					"F","R", n-i, ib, &a[i+i*lda], lda, &tau[i], work, ldwork
				);
				// Apply H to A[i+ib..m,i..n] from the right
				
				Reflector::ApplyBlock(
					"R","N","F","R", m-i-ib, n-i, ib, &a[i+i*lda], lda,
					work, ldwork, &a[(i+ib)+i*lda], lda, &work[ib], ldwork
				);
			}
		}
	}
	if(i < k){
		Factor_unblocked(m-i, n-i, &a[i+i*lda], lda, &tau[i], work);
	}
}


///////////////////////////////////////////////////////////////////////
// MultQ_unblocked
// ---------------
// From an existing LQ factorization, multiplies a given matrix by the
// unitary matrix Q. The given m-by-n matrix C is overwritten with:
//
//    trans | side = "L"   | side = "R"
//    ------|--------------|------------
//     "N"  |   Q   * C    |  C * Q
//     "C"  |   Q^H * C    |  C * Q^H
//
// This routine uses only level 2 BLAS.
// Equivalent to Lapack routines _unml2 and _orml2.
//
// Arguments
// side  If "L", apply Q or Q^H from the left. If "R", apply Q or
//       Q^H from the right.
// trans If "N", apply Q. If "C", apply Q^H.
// m     Number of rows of the matrix C.
// n     Number of columns of the matrix C.
// k     Number of elementary reflectors to apply.
//       If side = "L", k <= m. If side = "R", k <= n.
// a     Pointer to the factorization. The i-th row should contain
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
	const char *side, const char *trans,
	size_t m, size_t n, size_t k,
	const T *a, size_t lda,
	const T *tau,
	T *c, size_t ldc,
	T *work // size n if side is L, m if side is R
){
	RNPAssert(ldc >= m);
	if(0 == m || 0 == n || 0 == k){ return; }
	const bool left = ('L' == side[0]);
	const bool notran = ('N' == trans[0]);
	RNPAssert((left && k <= m) || (!left && k <= n));
	RNPAssert(lda >= k);
	
	size_t mi, ni, ic, jc;
	if(left){
		ni = n;
		jc = 0;
	}else{
		mi = m;
		ic = 0;
	}
	
	if((left && notran) || (!left && !notran)){ // do forward loop
		for(size_t i = 0; i < k; ++i){
			if(left){
				mi = m-i;
				ic = i;
			}else{
				ni = n-i;
				jc = i;
			}
			T taui = (notran ? Traits<T>::conj(tau[i]) : tau[i]);
			Reflector::Apply(side, 1, true, mi, ni, &a[i+i*lda], lda, taui, &c[ic+jc*ldc], ldc, work);
		}
	}else{ // do backwards loop
		size_t i = k;
		while(i --> 0){
			if(left){
				mi = m-i;
				ic = i;
			}else{
				ni = n-i;
				jc = i;
			}
			T taui = (notran ? Traits<T>::conj(tau[i]) : tau[i]);
			Reflector::Apply(side, 1, true, mi, ni, &a[i+i*lda], lda, taui, &c[ic+jc*ldc], ldc, work);
		}
	}
}


///////////////////////////////////////////////////////////////////////
// MultQ
// -----
// From an existing :Q factorization, multiplies a given matrix by the
// unitary matrix Q. The given m-by-n matrix C is overwritten with:
//
//    trans | side = "L"   | side = "R"
//    ------|--------------|------------
//     "N"  |   Q   * C    |  C * Q
//     "C"  |   Q^H * C    |  C * Q^H
//
// Equivalent to Lapack routines _unmlq and _ormlq.
//
// Arguments
// side  If "L", apply Q or Q^H from the left. If "R", apply Q or
//       Q^H from the right.
// trans If "N", apply Q. If "C", apply Q^H.
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
// lwork Lenth of workspace.
//       If side = "L", lwork >= n. If side = "R", lwork >= m.
//       If *lwork == 0 or NULL == work, then the optimal size is
//       returned in this argument.
// work  Workspace of size lwork.
//
template <typename T>
void MultQ(
	const char *side, const char *trans,
	size_t m, size_t n, size_t k,
	const T *a, size_t lda,
	const T *tau,
	T *c, size_t ldc,
	size_t *lwork,
	T *work
){
	RNPAssert(ldc >= m);
	if(0 == m || 0 == n || 0 == k){ return; }
	const bool left = ('L' == side[0]);
	const bool notran = ('N' == trans[0]);
	RNPAssert((left && k <= m) || (!left && k <= n));
	RNPAssert(lda >= k);
	
	const size_t nq = (left ? m : n);
	const size_t nw = (left ? n : m);
	
	size_t nb = LQ::Tuning<T>::multQ_block_size_opt(side, trans, m, n, k);
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
			nbmin = LQ::Tuning<T>::multQ_block_size_min(side, trans, m, n, k);
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
		const char *transt = (notran ? "C" : "N");
		if((left && notran) || ((!left) && (!notran))){	// loop forwards
			for(size_t i = 0; i < k; i += nb){
				size_t ib = (k < nb+i ? k-i : nb);
				Reflector::GenerateBlockTr("F","R", nq-i, ib, &a[i+i*lda], lda, &tau[i], t, ldt);

				if(left){
					mi = m-i;
					ic = i;
				}else{
					ni = n-i;
					jc = i;
				}
				Reflector::ApplyBlock(
					side, transt, "F", "R", mi, ni, ib,
					&a[i+i*lda], lda, t, ldt, &c[ic+jc*ldc], ldc, work, ldwork
				);
			}
		}else{
			size_t i = (k/nb)*nb + nb;
			do{ i -= nb;
				size_t ib = (k < nb+i ? k-i : nb);
				Reflector::GenerateBlockTr("F","R", nq-i, ib, &a[i+i*lda], lda, &tau[i], t, ldt);

				if(left){
					mi = m-i;
					ic = i;
				}else{
					ni = n-i;
					jc = i;
				}
				Reflector::ApplyBlock(
					side, transt, "F", "R", mi, ni, ib,
					&a[i+i*lda], lda, t, ldt, &c[ic+jc*ldc], ldc, work, ldwork
				);
			}while(i > 0);
		}
	}
}

///////////////////////////////////////////////////////////////////////
// GenerateQ_unblocked
// -------------------
// From an existing LQ factorization, generates the unitary matrix Q.
// The original matrix containing the factorization is overwritten
// by Q.
// This routine uses only level 2 BLAS.
// Equivalent to Lapack routines _ungl2 and _orgl2.
//
// Arguments
// m     Number of rows of the matrix Q.
// n     Number of columns of the matrix Q, n >= m.
// k     Number of elementary reflectors, k <= m.
// a     Pointer to the factorization. The i-th row should contain
//       the vector which defines the i-th elementary reflector for
//       i = 1..k. On exit, the matrix Q. When m > n, a should not
//       point to the first element of the matrix passed to Factor;
//       it should start at row m-n.
// lda   Leading dimension of the array containing Q, lda >= m.
// tau   Array of tau's, length k.
// work  Workspace of size m.
//
template <class T>
void GenerateQ_unblocked(
	size_t m, size_t n, size_t k, T *a, size_t lda, const T *tau, T *work
){
	RNPAssert(n >= m);
	RNPAssert(k <= m);
	RNPAssert(lda >= m);
	if(0 == m){ return; }

	// Initialise rows k+1:m to rows of the unit matrix
	if(k < m){
		for(size_t j = 0; j < n; ++j){
			for(size_t i = k; i < m; ++i){
				a[i+j*lda] = T(0);
			}
			if(j >= k && j < m){ 
				a[j+j*lda] = T(1);
			}
		}
	}

	size_t i = k;
	while(i --> 0){
		if(i+1 < n){
			BLAS::Conjugate(n-i-1, &a[i+(i+1)*lda], lda);
			if(i+1 < m){
				a[i+i*lda] = T(1);
				Reflector::Apply("R", 0, false, m-i-1, n-i, &a[i+i*lda], lda, Traits<T>::conj(tau[i]), &a[i+1+i*lda], lda, work);
			}
			BLAS::Scale(n-i-1, -tau[i], &a[i+(i+1)*lda], lda);
			BLAS::Conjugate(n-i-1, &a[i+(i+1)*lda], lda);
		}
		a[i+i*lda] = T(1) - Traits<T>::conj(tau[i]);

		for(size_t l = 0; l < i; ++l){
			a[i+l*lda] = T(0);
		}
	}
}

///////////////////////////////////////////////////////////////////////
// GenerateQ
// ---------
// From an existing LQ factorization, generates the unitary matrix Q.
// The original matrix containing the factorization is overwritten
// by Q.
// Equivalent to Lapack routines _unglq and _orglq.
//
// Arguments
// m     Number of rows of the matrix Q.
// n     Number of columns of the matrix Q, n >= m.
// k     Number of elementary reflectors, k <= m.
// a     Pointer to the factorization. The i-th column should contain
//       the vector which defines the i-th elementary reflector for
//       i = 0..k.
// lda   Leading dimension of the array containing Q, lda >= m.
// tau   Array of tau's, length k.
// lwork Lenth of workspace, lwork >= m.
//       If *lwork == 0 or NULL == work, then the optimal size is
//       returned in this argument.
// work  Workspace of size lwork.
//
template <class T>
void GenerateQ(
	size_t m, size_t n, size_t k, T *a, size_t lda,
	const T *tau, size_t *lwork, T *work
){
	RNPAssert(k <= m);
	RNPAssert(lda >= m);
	if(0 == m){ return; }
	
	size_t nb = LQ::Tuning<T>::genQ_block_size_opt(m, n, k);
	if(0 == *lwork || NULL == work){
		*lwork = n*nb;
		return;
	}
	
	size_t nbmin = 2;
	size_t nx = 0;
	size_t iws = n;
	size_t ldwork = n;
	if(nb > 1 && nb < k){
		nx = LQ::Tuning<T>::genQ_crossover_size(m, n, k);
		if(nx < k){
			iws = ldwork*nb;
			if(*lwork < iws){
				nb = *lwork / ldwork;
				nbmin = LQ::Tuning<T>::genQ_block_size_min(m, n, k);
			}
		}
	}
	
	size_t ki = 0, kk = 0;
	if(nb >= nbmin && nb < k && nx < k){ // use blocked code after last block
		ki = ((k-nx-1)/nb) * nb;
		kk = (k < ki+nb ? k : ki+nb);
		for(size_t j = 0; j < kk; ++j){
			for(size_t i = kk; i < m; ++i){
				a[i+j*lda] = T(0);
			}
		}
	}
	//ki = nb;
	if(kk < m){
		GenerateQ_unblocked(m-kk, n-kk, k-kk, &a[kk+kk*lda], lda, &tau[kk], work);
	}
	if(kk > 0){
		size_t i = ki + nb;
		do{ i -= nb;
			const size_t ib = (nb+i < k ? nb : k-i);
			if(i+ib < m){
				Reflector::GenerateBlockTr("F", "R", n-i, ib, &a[i+i*lda], lda, &tau[i], work, ldwork);
				Reflector::ApplyBlock("R","C","F","R", m-i-ib, n-i, ib, &a[i+i*lda], lda, work, ldwork, &a[(i+ib)+i*lda], lda, &work[ib], ldwork);
			}
			// Apply H to rows 0..m of current block
			GenerateQ_unblocked(ib, n-i, ib, &a[i+i*lda], lda, &tau[i], work);
			// Set rows 0..i of current block to zero
			for(size_t j = 0; j < i; ++j){
				for(size_t l = i; l < i+ib; ++l){
					a[l+j*lda] = T(0);
				}
			}
		}while(i > 0);
	}
}

} // namespace LQ
} // namespace LA
} // namespace RNP

#endif // RNP_LQ_HPP_INCLUDED
