#ifndef RNP_EIGENSYSTEMS_HPP_INCLUDED
#define RNP_EIGENSYSTEMS_HPP_INCLUDED

#include <RNP/BLAS.hpp>
#include <RNP/bits/Tridiagonal.hpp>
#include <iostream>

namespace RNP{
namespace LA{

///////////////////////////////////////////////////////////////////////
// Eigensystems
// ============
// Computes various eigenvalue decompositions, which are basically
// similarity transformations which bring a square matrix to diagonal
// form.


///////////////////////////////////////////////////////////////////////
// HermitianEigensystem
// --------------------
// Computes all eigenvalues and, optionally, eigenvectors of a
// Hermitian matrix A. This routine works for complex Hermitian or real
// symmetric matrices. The right eigenvectors are returned as columns
// of the same matrix A.
//
// Equivalent to Lapack routines zheev/cheev and dsyev/ssyev.
//
// Arguments
// job   If "N", only eigenvalues are computed. If "V" the
//       eigenvectors are returned in the columns of a.
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of the matrix A.
// a     Pointer to the first element of A. On exit, if job = "V",
//       this is overwritten by the matrix whose columns are the
//       eigenvectors corresponding to the eigenvalues in w.
// lda   Leading dimension of the array containing A (lda >= n).
// w     Array of returned eigenvalues (length n).
// lwork Length of workspace. If *lwork == 0 then the optimal size
//       is returned in this argument. lwork must be at least 2*(n-1)
//       for complex matrices, and 3*(n-1) for real matrices.
// work  Workspace of size lwork, or NULL for workspace query.
//
template <typename T>
int HermitianEigensystem(
	const char *job, const char *uplo, size_t n, T *a, size_t lda,
	typename Traits<T>::real_type *w,
	size_t *lwork,
	T *work
){
	typedef typename Traits<T>::real_type real_type;
	
	RNPAssert('V' == job[0] || 'N' == job[0]);
	RNPAssert('U' == uplo[0] || 'L' == uplo[0]);
	RNPAssert(NULL != a);
	RNPAssert(lda >= n);
	RNPAssert(NULL != lwork);
	
	static const bool is_real = !Traits<T>::is_complex();
	
	RNPAssert(0 == *lwork || *lwork >= 2*(n-1) + (is_real ? n-1 : 0));
	
	if(0 == n){ return 0; }
	if(1 == n){
		w[0] = Traits<T>::real(a[0]);
		if('V' == job[0]){
			a[0] = T(1);
		}
		return 0;
	}
	
	// Workspace
	//  Real symmetric:
	//   [ offdiag | tau | reduce_work ] : Reduction phase
	//       n-1     n-1    variable
	//   [ offdiag | qr_work           ] : QR phase
	//       n-1     2*(n-1)
	//  Complex symmetric:
	//   [ offdiag | tau | reduce_work ] : Reduction phase
	//     n-1 (r)   n-1    variable
	//   [ offdiag | qr_work           ]
	//     n-1 (r)   2*(n-1) (r)
	//  We assume that ReduceHerm and GenerateQHerm can use the same
	//  blocked workspace. The QR iteration workspace can overwrite tau
	//  since it is no longer needed at that stage.
	//   Therefore we see that 2*(n-1)+var is enough, where var >= n-1
	//  in the real case.
	//  The only assumption here is that
	//     sizeof(complex) >= 2*sizeof(real)
	
	// Determine lwork
	if(0 == *lwork || NULL == work){
		// Tridiagonal reduction needs a workspace, and
		// symmetric QR needs a workspace.
		T dummy; // ReduceHerm's work can't be NULL for a workspace query.
		Tridiagonal::ReduceHerm(uplo, n, a, lda, w, (real_type*)NULL, (T*)NULL, lwork, &dummy);
		if(is_real && *lwork < n-1){ *lwork = (n-1); }
		*lwork += 2*(n-1);
		return 0;
	}
	
	// Set workspace
	real_type *offdiag;
	T *tau;
	T *reduce_work;
	size_t reduce_lwork;
	real_type *qr_work;
	if(is_real){ // real symmetric
		// Since T and real_type should be the same, these casts should be ok.
		offdiag = reinterpret_cast<real_type*>(work);
		tau = reinterpret_cast<T*>(offdiag + (n-1));
		reduce_work = tau + (n-1);
		reduce_lwork = *lwork - 2*(n-1);
		qr_work = reinterpret_cast<real_type*>(tau); // This will run beyond tau into reduce_work, which is at least n-1.
	}else{ // complex hermitian
		offdiag = reinterpret_cast<real_type*>(work);
		tau = work + (n-1);
		reduce_work = tau + (n-1);
		reduce_lwork = *lwork - 2*(n-1);
		qr_work = reinterpret_cast<real_type*>(tau);
	}
	
	const real_type safmin(Traits<real_type>::min());
	const real_type eps(real_type(2)*Traits<real_type>::eps());
	const real_type smlnum(safmin / eps);
	const real_type bignum(real_type(1) / smlnum);
	const real_type rmin(sqrt(smlnum));
	const real_type rmax(sqrt(bignum));
	
	const real_type anrm = MatrixNormHerm("M", uplo, n, a, lda);
	bool scaled = false;
	real_type sigma(1);
	if(anrm > 0 && anrm < rmin){
		scaled = true;
		sigma = rmin/anrm;
	}else if(anrm > rmax){
		scaled = true;
		sigma = rmax/anrm;
	}
	if(scaled){
		BLAS::Rescale(uplo, 0, 0, real_type(1), sigma, n, n, a, lda);
	}
	
	Tridiagonal::ReduceHerm(uplo, n, a, lda, w, offdiag, tau, &reduce_lwork, reduce_work);
	
	int info;
	if('V' == job[0]){
		Tridiagonal::GenerateQHerm(uplo, n, a, lda, tau, &reduce_lwork, reduce_work);
		info = Tridiagonal::Util::SymmetricEigensystem(
			n, w, offdiag, a, lda, qr_work
		);
	}else{
		info = Tridiagonal::Util::SymmetricEigensystem(
			n, w, offdiag, (real_type*)NULL, lda, (real_type*)NULL
		);
	}
	
	if(scaled){
		size_t wlim = n;
		if(0 != info){
			wlim = info - 1;
		}
		BLAS::Scale(wlim, real_type(1)/sigma, w, 1);
	}
	return info;
}

} // namespace LA
} // namespace RNP

#endif // RNP_EIGENSYSTEMS_HPP_INCLUDED
