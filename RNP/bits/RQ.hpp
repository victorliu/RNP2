#ifndef RNP_RQ_HPP_INCLUDED
#define RNP_RQ_HPP_INCLUDED

#include <cstddef>
#include <RNP/BLAS.hpp>
#include "Reflector.hpp"

namespace RNP{
namespace LA{
namespace RQ{

///////////////////////////////////////////////////////////////////////
// RNP::LA::RQ
// ===========
// Computes the RQ factorization and operations involving Q.
// For tall matrices, R is upper trapezoidal. For fat matrices, R is
// upper triangular. The decomposition is a product A = R * Q, with Q
// packed into the lower triangle of A, and an additional tau vector
// representing the scale factors of the reflector representation of Q.
// The storate scheme is shown below, with 'A' representing elements of
// the matrix A, 'R' representing elements of the the R factor, 'Q'
// representing elements of the reflector vectors which implicitly form
// Q, and 'T' the elements of the auxiliary array tau.
//
//     A A A   R R R          A A A A A   Q Q R R R  T
//     A A A   R R R          A A A A A = Q Q Q R R  T
//     A A A = R R R  T       A A A A A   Q Q Q Q R  T
//     A A A   Q R R  T
//     A A A   Q Q R  T
//
// When m >= n, R is m-by-n upper trapezoidal and Q is n-by-n.
// When m < n, R is m-by-m upper triangular and Q is m-by-n.

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
// Computes an RQ factorization of an m-by-n matrix A = R * Q.
// The matrix Q is represented as a product of elementary reflectors
//   Q = H[0] H[1] ... H[k-1], where k = min(m,n).
// Each H[i] has the form
//   H[i] = I - tau * v * v^H
// where tau is a scalar, and v is a vector with v[n-k+i+1..n] = 0 and
// v[n-k+i] = 1; v[i+1..m]' is stored, upon exit, in A[m-k+i,0..n-k+i],
// and tau in tau[i].
// This routine uses only level 2 BLAS.
// Equivalent to Lapack routines _gerq2.
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
	size_t m, size_t n, T *a, size_t lda, T *tau, T *work
){
	const size_t k = (m < n ? m : n);
	if(0 == k){ return; }
	size_t i = k; while(i --> 0){
		BLAS::Conjugate(n-k+i+1, &a[m-k+i+0*lda], lda); // note the +1
		T alpha(a[m-k+i+(n-k+i)*lda]);
		// Can replace with GeneratePositive if positive diagonal
		// elements in R are required.
		Reflector::Generate(
			n-k+i+1, &alpha, &a[(m-k+i)+0*lda], lda, &tau[i]
		);
		
		a[m-k+i+(n-k+i)*lda] = T(1);
		Reflector::Apply(
			"R", 0, false, m-k+i, n-k+i+1, &a[(m-k+i)+0*lda], lda,
			tau[i], a, lda, work
		);
		a[m-k+i+(n-k+i)*lda] = alpha;
		BLAS::Conjugate(n-k+i, &a[m-k+i+0*lda], lda);
	}
}

///////////////////////////////////////////////////////////////////////
// Factor
// ------
// Computes an RQ factorization of an m-by-n matrix A = R * Q.
// The matrix Q is represented as a product of elementary reflectors
//   Q = H[0] H[1] ... H[k-1], where k = min(m,n).
// Each H[i] has the form
//   H[i] = I - tau * v * v^H
// where tau is a scalar, and v is a vector with v[n-k+i+1..n] = 0 and
// v[n-k+i] = 1; v[i+1..m]' is stored, upon exit, in A[m-k+i,0..n-k+i],
// and tau in tau[i].
// Equivalent to Lapack routines _gerqf.
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
	size_t nb = RQ::Tuning<T>::factor_block_size_opt(m, n);
	if(NULL == work || 0 == *lwork){
		*lwork = nb*m;
		return;
	}
	size_t nbmin = 2;
	size_t nx = 0;
	size_t iws = m;
	size_t ldwork = m;
	if(nb > 1 && nb < k){
		// Determine when to cross over from blocked to unblocked code.
		nx = RQ::Tuning<T>::factor_crossover_size(m, n);
		if(nx < k){
			// Determine if workspace is large enough for blocked code.
            iws = ldwork * nb;
            if(*lwork < iws){
				// Not enough workspace to use optimal NB:  reduce NB and
				// determine the minimum value of NB.
				nb = *lwork / ldwork;
				nbmin = RQ::Tuning<T>::factor_block_size_min(m, n);
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
			// Compute the QR factorization of the current block A[m-k+i..m-k+i+ib,0..n-k+i+ib]
			Factor_unblocked(ib, n-k+i+ib, &a[(m-k+i)+0*lda], lda, &tau[i], work);
			if(m-k+i > 0){
				// Form the triangular factor of the block reflector
				//   H = H[i+ib-1] ...  H[i+1] H[i]
				
				Reflector::GenerateBlockTr(
					"B","R", n-k+i+ib, ib, &a[(m-k+i)+0*lda], lda, &tau[i], work, ldwork
				);
				
				// Apply H^H to A[0..m-k+i,0..n-k+i+ib] from the right
				Reflector::ApplyBlock(
					"R","N","B","R", m-k+i, n-k+i+ib, ib, &a[(m-k+i)+0*lda], lda,
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
// From an existing RQ factorization, multiplies a given matrix by the
// unitary matrix Q. The given m-by-n matrix C is overwritten with:
//
//    trans | side = "L"   | side = "R"
//    ------|--------------|------------
//     "N"  |   Q   * C    |  C * Q
//     "C"  |   Q^H * C    |  C * Q^H
//
// This routine uses only level 2 BLAS.
// Equivalent to Lapack routines _unmr2 and _ormr2.
//
// Arguments
// side  If "L", apply Q or Q^H from the left. If "R", apply Q or
//       Q^H from the right.
// trans If "N", apply Q. If "C", apply Q'.
// m     Number of rows of the matrix C.
// n     Number of columns of the matrix C.
// k     Number of elementary reflectors to apply.
//       If side = "L", k <= m. If side = "R", k <= n.
// a     Pointer to the factorization. The i-th row should contain
//       the vector which defines the i-th elementary reflector for
//       i = 1..k. When m > n, a should not point to the first element
//       of the matrix passed to Factor; it should start at row m-n.
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
	if(left){
		ni = n;
	}else{
		mi = m;
	}
	
	if((left && !notran) || (!left && notran)){
		// loop forwards
		for(size_t i = 0; i < k; ++i){
			if(left){ // H[i] or H[i]' is applied to C[0..m-k+i+1,0..n]
				mi = m-k+i+1;
			}else{ // H[i] or H[i]' is applied to C[0..m,0..n-k+i+1]
				ni = n-k+i+1;
			}

			T taui;
			if(notran){
				taui = Traits<T>::conj(tau[i]);
			}else{
				taui = tau[i];
			}
			Reflector::Apply(side, -1, true, mi, ni, &a[i+0*lda], lda, taui, c, ldc, work);
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
				taui = Traits<T>::conj(tau[i]);
			}else{
				taui = tau[i];
			}
			Reflector::Apply(side, -1, true, mi, ni, &a[i+0*lda], lda, taui, c, ldc, work);
			//BLAS::Conjugate(nq-k+i, &a[i+0*lda], lda);
			//T aii(a[i+(nq-k+i)*lda]);
			//a[i+(nq-k+i)*lda] = T(1);
			//Reflector::Apply(side, 0, false, mi, ni, &a[i+0*lda], lda, taui, c, ldc, work);
			//a[i+(nq-k+i)*lda] = aii;
			//BLAS::Conjugate(nq-k+i, &a[i+0*lda], lda);
		}
	}
}


///////////////////////////////////////////////////////////////////////
// MultQ
// -----
// From an existing RQ factorization, multiplies a given matrix by the
// unitary matrix Q. The given m-by-n matrix C is overwritten with:
//
//    trans | side = "L"   | side = "R"
//    ------|--------------|------------
//     "N"  |   Q   * C    |  C * Q
//     "C"  |   Q^H * C    |  C * Q^H
//
// Equivalent to Lapack routines _unmrq and _ormrq.
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
//       i = 1..k. When m > n, a should not point to the first element
//       of the matrix passed to Factor; it should start at row m-n.
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
	
	size_t nb = RQ::Tuning<T>::multQ_block_size_opt(side, trans, m, n, k);
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
			nbmin = RQ::Tuning<T>::multQ_block_size_min(side, trans, m, n, k);
		}
	}
	
	const char *transt = (notran ? "C" : "N");
	
	if(nb < nbmin || nb >= k){ // unblocked
		MultQ_unblocked(side, trans, m, n, k, a,lda, tau, c, ldc, work);
	}else{
		size_t ni, mi;
		if(left){
			ni = n;
		}else{
			mi = m;
		}
		if((left && !notran) || (!left && notran)){	// loop forwards
			for(size_t i = 0; i < k; i += nb){
				size_t ib = (k < nb+i ? k-i : nb);
				Reflector::GenerateBlockTr("B","R", nq-k+i+ib, ib, &a[i+0*lda], lda, &tau[i], t, ldt);

				if(left){
					mi = m-k+i+ib;
				}else{
					ni = n-k+i+ib;
				}
				Reflector::ApplyBlock(
					side, transt, "B", "R", mi, ni, ib,
					&a[i+0*lda], lda, t, ldt, c, ldc, work, ldwork
				);
			}
		}else{
			size_t i = ((k-1)/nb)*nb + nb;
			do{ i -= nb;
				size_t ib = (k < nb+i ? k-i : nb);
				Reflector::GenerateBlockTr(
					"B","R", nq-k+i+ib, ib, &a[i+0*lda], lda, &tau[i], t, ldt
				);

				if(left){
					mi = m-k+i+ib;
				}else{
					ni = n-k+i+ib;
				}
				Reflector::ApplyBlock(
					side, transt, "B", "R", mi, ni, ib,
					&a[i+0*lda], lda, t, ldt, c, ldc, work, ldwork
				);
			}while(i > 0);
		}
	}
}

///////////////////////////////////////////////////////////////////////
// GenerateQ_unblocked
// -------------------
// From an existing RQ factorization, generates the unitary matrix Q.
// The original matrix containing the factorization is overwritten
// by Q.
// This routine uses only level 2 BLAS.
// Equivalent to Lapack routines _ungr2 and _orgr2.
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
// c     Pointer to the first element of the matrix C.
// ldc   Leading dimension of the array containing C.
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

	// Initialise columns 0..n-k to columns of the unit matrix
	if(k < m){
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i+k < m; ++i){
				a[i+j*lda] = T(0);
			}
			if(j >= n-m && j < n-k){
				a[m-n+j+j*lda] = T(1);
			}
		}
	}
	
	for(size_t i = 0; i < k; ++i){
		size_t ii = m-k+i;
		// Apply H[i] to A[] from the left
		BLAS::Conjugate(n-m+ii, &a[ii+0*lda], lda);
		a[ii+(n-m+ii)*lda] = T(1);
		Reflector::Apply("R", 0, false, ii, n-m+ii+1, &a[ii+0*lda], lda, Traits<T>::conj(tau[i]), a, lda, work);
		BLAS::Scale(n-m+ii, -tau[i], &a[ii+0*lda], lda);
		BLAS::Conjugate(n-m+ii, &a[ii+0*lda], lda);
		a[ii+(n-m+ii)*lda] = T(1) - Traits<T>::conj(tau[i]);

		// Set A(m-k+i+1:m,n-k+i) to zero
		for(size_t l = n-m+ii+1; l < n; ++l){
			a[ii+l*lda] = T(0);
		}
	}
}

///////////////////////////////////////////////////////////////////////
// GenerateQ
// ---------
// From an existing RQ factorization, generates the unitary matrix Q.
// The original matrix containing the factorization is overwritten
// by Q.
// Equivalent to Lapack routines _ungrq and _orgrq.
//
// Arguments
// m     Number of rows of the matrix Q.
// n     Number of columns of the matrix Q, n >= m.
// k     Number of elementary reflectors, k <= m.
// a     Pointer to the factorization. The i-th column should contain
//       the vector which defines the i-th elementary reflector for
//       i = 0..k. On exit, the matrix Q. When m > n, a should not
//       point to the first element of the matrix passed to Factor;
//       it should start at row m-n.
// lda   Leading dimension of the array containing Q, lda >= m.
// tau   Array of tau's, length k.
// c     Pointer to the first element of the matrix C.
// ldc   Leading dimension of the array containing C.
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
	RNPAssert(n >= m);
	RNPAssert(k <= m);
	RNPAssert(lda >= m);
	if(0 == m){ return; }
	
	size_t nb = QR::Tuning<T>::genQ_block_size_opt(m, n, k);
	if(0 == *lwork || NULL == work){
		*lwork = m*nb;
		return;
	}
	
	size_t nbmin = 2;
	size_t nx = 0;
	size_t iws = m;
	size_t ldwork = m;
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
		for(size_t j = n-kk; j < n; ++j){
			for(size_t i = 0; i < m-kk; ++i){
				a[i+j*lda] = T(0);
			}
		}
	}

	GenerateQ_unblocked(m-kk, n-kk, k-kk, a, lda, tau, work);
	
	if(kk > 0){
		for(size_t i = k-kk; i < k; i += nb){
			const size_t ib = (nb+i < k ? nb : k-i);
			const size_t ii = m-k+i;
			if(ii > 0){
				Reflector::GenerateBlockTr("B", "R", n-k+i+ib, ib, &a[ii+0*lda], lda, &tau[i], work, ldwork);
				Reflector::ApplyBlock("R","C","B","R", ii, n-k+i+ib, ib, &a[ii+0*lda], lda, work, ldwork, a, lda, &work[ib], ldwork);
			}
			// Apply H to 
			GenerateQ_unblocked(ib, n-k+i+ib, ib, &a[ii+0*lda], lda, &tau[i], work);
			// Set rows 
			for(size_t l = n-k+i+ib; l < n; ++l){
				for(size_t j = ii; j < ii+ib; ++j){
					a[j+l*lda] = T(0);
				}
			}
		}
	}
}

} // namespace RQ
} // namespace LA
} // namespace RNP

#endif // RNP_QL_HPP_INCLUDED
