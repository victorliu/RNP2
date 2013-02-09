#ifndef RNP_HESSENBERG_HPP_INCLUDED
#define RNP_HESSENBERG_HPP_INCLUDED

#include <cstddef>
#include <RNP/BLAS.hpp>
#include <RNP/bits/Rotation.hpp>
#include <RNP/bits/Reflector.hpp>
#include <RNP/bits/QR.hpp>

namespace RNP{
namespace LA{
namespace Hessenberg{

///////////////////////////////////////////////////////////////////////
// RNP::LA::Hessenberg
// ===================
// Routines dealing with Hessenberg matrices and reduction of square
// matrices to Hessenberg form.
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
	static size_t reduce_block_size_opt(size_t n, size_t ilo, size_t ihi){ return 64; }
	static size_t reduce_block_size_min(size_t n, size_t ilo, size_t ihi){ return 2; }
	static size_t reduce_crossover_size(size_t n, size_t ilo, size_t ihi){ return 128; }
};

///////////////////////////////////////////////////////////////////////
// Norm
// ----
// Returns the value of the 1-norm, Frobenius norm, infinity norm, or
// the  element of largest absolute value of an upper Hessenberg
// matrix A. Note that the maximum element magnitude is not a
// consistent matrix norm.
// Equivalent to Lapack routines _lanhs.
//
// Arguments
// norm Specifies which norm to return:
//        If norm = "M", returns max(abs(A[i,j])).
//        If norm = "1" or "O", returns norm1(A) (max column sum).
//        If norm = "I", returns normInf(A) (max row sum).
//        If norm = "F" or "E", returns normFrob(A).
// n    Number of rows and columns of the matrix A.
// a    Pointer to the first element of A.
// lda  Leading dimension of the array containing A (lda >= n).
// work Optional workspace of size n when norm = "I". If work = NULL,
//      the norm is computed slightly less efficiently.
//
template <typename T>
typename Traits<T>::real_type Norm(
	const char *norm, size_t n, const T *a, size_t lda,
	typename Traits<T>::real_type *work = NULL
){
	typedef typename Traits<T>::real_type real_type;
	
	RNPAssert(NULL != norm);
	RNPAssert(
		'M' == norm[0] || '1' == norm[0] || 'O' == norm[0] ||
		'I' == norm[0] || 'F' == norm[0] || 'E' == norm[0]
	);
	RNPAssert(NULL != a);
	RNPAssert(lda >= n);
	
	const real_type rzero(0);
	if(n < 1){ return rzero; }
	real_type result(0);
	if('M' == norm[0]){ // max(abs(A(i,j)))
		for(size_t j = 0; j < n; ++j){
			size_t ilimit = j+2; if(n < ilimit){ ilimit = n; }
			for(size_t i = 0; i < ilimit; ++i){
				real_type ca = Traits<T>::abs(a[i+j*lda]);
				if(!(ca < result)){ result = ca; }
			}
		}
	}else if('O' == norm[0] || '1' == norm[0]){ // max col sum
		for(size_t j = 0; j < n; ++j){
			size_t ilimit = j+2; if(n < ilimit){ ilimit = n; }
			real_type sum(0);
			for(size_t i = 0; i < ilimit; ++i){
				sum += Traits<T>::abs(a[i+j*lda]);
			}
			if(!(sum < result)){ result = sum; }
		}
	}else if('I' == norm[0]){ // max row sum
		if(NULL == work){ // can't accumulate row sums
			for(size_t i = 0; i < n; ++i){
				size_t jstart = 0; if(i > 0){ jstart = i-1; }
				real_type sum = 0;
				for(size_t j = 0; j < n; ++j){
					sum += Traits<T>::abs(a[i+j*lda]);
				}
				if(!(sum < result)){ result = sum; }
			}
		}else{ // accumulate row sums in a cache-friendlier traversal order
			for(size_t i = 0; i < n; ++i){ work[i] = 0; }
			for(size_t j = 0; j < n; ++j){
				size_t ilimit = j+2; if(n < ilimit){ ilimit = n; }
				for(size_t i = 0; i < ilimit; ++i){
					work[i] += Traits<T>::abs(a[i+j*lda]);
				}
			}
			for(size_t i = 0; i < n; ++i){
				if(!(work[i] < result)){ result = work[i]; }
			}
		}
	}else if('F' == norm[0] || 'E' == norm[0]){ // Frobenius norm
		real_type scale = 0;
		real_type sum = 1;
		for(size_t j = 0; j < n; ++j){
			size_t ilimit = j+2; if(n < ilimit){ ilimit = n; }
			real_type sum = 0;
			for(size_t i = 0; i < ilimit; ++i){
				real_type ca = Traits<T>::abs(a[i+j*lda]);
				if(scale < ca){
					real_type r = scale/ca;
					sum = real_type(1) + sum*r*r;
					scale = ca;
				}else{
					real_type r = ca/scale;
					sum += r*r;
				}
			}
		}
		result = scale*sqrt(sum);
	}
	return result;
}

///////////////////////////////////////////////////////////////////////
// Reduce_unblocked
// ----------------
// Reduces a general square matrix A to upper Hessenberg form H
// by a unitary similarity transformation: Q^H * A * Q = H.
// This routine uses only level 2 BLAS.
// Equivalent to Lapack routines _gehd2.
//
// The matrix Q is represented as a product of (ihi-ilo-1) elementary
// reflectors
//
//     Q = H[ilo] H[ilo+1] ... H[ihi-2].
//
// Each H[i] has the form
//
//     H[i] = I - tau * v * v'
//
// where tau is a scalar, and v is a vector with v[0:i+1] = 0,
// v[i+1] = 1 and v[ihi:n] = 0; v[i+2:ihi] is stored on exit in
// A[i+2:ihi,i], and tau in tau[i].
//
// The contents of A are illustrated by the following example, with
// n = 7, ilo = 1 and ihi = 6:
//
//     On entry:                        On exit:
//     [ a   a   a   a   a   a   a ]    [  a   a   h   h   h   h   a ]
//     [     a   a   a   a   a   a ]    [      a   h   h   h   h   a ]
//     [     a   a   a   a   a   a ]    [      h   h   h   h   h   h ]
//     [     a   a   a   a   a   a ]    [      v1  h   h   h   h   h ]
//     [     a   a   a   a   a   a ]    [      v1  v2  h   h   h   h ]
//     [     a   a   a   a   a   a ]    [      v1  v2  v3  h   h   h ]
//     [                         a ]    [                          a ]
//
// where a denotes an element of the original matrix A, h denotes a
// modified element of the upper Hessenberg matrix H, and vi denotes an
// element of the vector defining H[i].
//
// Arguments
// n    Number of rows and columns of the matrix A.
// ilo  It is assumed that A is already upper triangular in rows and
// ihi  columns 0:ilo-1 and ihi:n (that is, the reduction is performed
//      on the range ilo:ihi. Usually these are returned by a balancing
//      routine. 0 <= ilo < ihi <= n.
// a    Pointer to the first element of A.
// lda  Leading dimension of the array containing A (lda >= n).
// tau  The scale factors of the elementary reflectors, length n-1.
// work Workspace of size n.
//
template <typename T>
void Reduce_unblocked(
	size_t n, size_t ilo, size_t ihi, T *a, size_t lda, T *tau, T *work
){
	RNPAssert(ilo <= n);
	RNPAssert(ilo < ihi);
	RNPAssert(ihi <= n);
	RNPAssert(lda >= n);
	RNPAssert(NULL != a);
	RNPAssert(NULL != tau);
	RNPAssert(NULL != work);
	for(size_t i = ilo; i+1 < ihi; ++i){
		// Compute elementary reflector H(i) to annihilate A(i+2:ihi,i)
		T alpha = a[i+1+i*lda];
		size_t row = i+2; if(n <= row){ row = n-1; }
		Reflector::Generate(ihi-i-1, &alpha, &a[row+i*lda], 1, &tau[i]);
		a[i+1+i*lda] = T(1);
		// Apply H(i) to A(1:ihi,i+1:ihi) from the right
		Reflector::Apply("R", false, false, ihi, ihi-i-1, &a[i+1+i*lda], 1, tau[i], &a[0+(i+1)*lda], lda, work);
		// Apply H(i)' to A(i+1:ihi,i+1:n) from the left
		Reflector::Apply("L", false, false, ihi-i-1, n-i-1, &a[i+1+i*lda], 1, Traits<T>::conj(tau[i]), &a[i+1+(i+1)*lda], lda, work);
		a[i+1+i*lda] = alpha;
	}
}

namespace Util{

///////////////////////////////////////////////////////////////////////
// Util::Reduce_block
// ------------------
// Reduces the first nb columns of a general n-by-(n-k+1) matrix A so
// that elements below the k-th subdiagonal are zero. The reduction is
// performed by a unitary similarity transformation Q^H * A * Q.
// The routine returns the matrices V and T which determine Q as a
// block reflector I - V*T*V^H, and also the matrix Y = A * V * T.
// This is equivalent to Lapack routine _lahr2. No further explanation
// will be given.
//
template <typename T>
void Reduce_block(
	size_t n, size_t k, size_t nb, T *a, size_t lda,
	T *tau, T *t, size_t ldt, T *y, size_t ldy
){
	if(n <= 1){ return; }
	T ei;
	for(size_t i = 0; i < nb; ++i){
		if(i > 0){ // Update A[k+1..n,i]
			// Update i-th column of A - Y*V^H
			BLAS::Conjugate(i, &a[k+i+0*lda], lda);
			BLAS::MultMV(
				"N", n-k-1, i, T(-1), &y[k+1+0*ldy], ldy,
				&a[k+i+0*lda], lda, T(1), &a[k+1+i*lda], 1
			);
            BLAS::Conjugate(i, &a[k+i+0*lda], lda);
			// Apply I - V * T^H * V^H to this column (call it b) from the
			// left, using the last column of T as workspace
			// Let  V = ( V1 )   and   b = ( b1 )   (first I-1 rows)
			//          ( V2 )             ( b2 )
			// where V1 is unit lower triangular
			// w := V1^H * b1
			BLAS::Copy(i, &a[k+1+i*lda], 1, &t[0+(nb-1)*ldt], 1);
			BLAS::MultTrV("L","C","U", i, &a[k+1+0*lda], lda, &t[0+(nb-1)*ldt], 1);
			// w := w + V2^H * b2
			BLAS::MultMV(
				"C", n-k-1-i, i, T(1), &a[k+i+1+0*lda], lda,
				&a[k+i+1+i*lda], 1, T(1), &t[0+(nb-1)*ldt], 1
			);
			// w := T^H * w
			BLAS::MultTrV("U","C","N", i, t, ldt, &t[0+(nb-1)*ldt], 1);
			// b2 := b2 - V2*w
			BLAS::MultMV(
				"N", n-k-1-i, i, T(-1), &a[k+i+1+0*lda], lda,
				&t[0+(nb-1)*ldt], 1, T(1), &a[k+i+1+i*lda], 1
			);
			// b1 := b1 - V1*w
			BLAS::MultTrV("L","N","U", i, &a[k+1+0*lda], lda, &t[0+(nb-1)*ldt], 1);
			BLAS::Axpy(i, T(-1), &t[0+(nb-1)*ldt], 1, &a[k+1+i*lda], 1);
            a[k+i+(i-1)*lda] = ei; // ei is set in previous loop iteration
		}
		// Generate the elementary reflector H[I] to annihilate A[k+i+2..n,i]
		const size_t row = (k+i+2 < n-1 ? k+i+2 : n-1);
		Reflector::Generate(n-k-1-i, &a[k+i+1+i*lda], &a[row+i*lda], 1, &tau[i]);
		ei = a[k+i+1+i*lda];
		a[k+i+1+i*lda] = T(1);
		// Compute  Y[k+1..n,i]
		BLAS::MultMV(
			"N", n-k-1, n-k-1-i, T(1), &a[k+1+(i+1)*lda], lda,
			&a[k+i+1+i*lda], 1, T(0), &y[k+1+i*ldy], 1
		);
		BLAS::MultMV(
			"C", n-k-1-i, i, T(1), &a[k+i+1+0*lda], lda,
			&a[k+i+1+i*lda], 1, T(0), &t[0+i*ldt], 1
		);
		BLAS::MultMV(
			"N", n-k-1, i, T(-1), &y[k+1+0*ldy], ldy,
			&t[0+i*ldt], 1, T(1), &y[k+1+i*ldy], 1
		);
		BLAS::Scale(n-k-1, tau[i], &y[k+1+i*ldy], 1);
		// Compute T[0..i+1,i]
		BLAS::Scale(i, -tau[i], &t[0+i*ldt], 1);
		BLAS::MultTrV("U", "N", "N", i, t, ldt, &t[0+i*ldt], 1);
		t[i+i*ldt] = tau[i];
	}
	a[k+nb+(nb-1)*lda] = ei;
	// Compute Y[0..k+1,0..nb]
	BLAS::Copy(k+1, nb, &a[0+1*lda], lda, y, ldy);
	BLAS::MultTrM("R", "L", "N", "U", k+1, nb, T(1), &a[k+1+0*lda], lda, y, ldy);
	if(n > k+1+nb){
		BLAS::MultMM(
			"N", "N", k+1, nb, n-k-1-nb, T(1), &a[0+(nb+1)*lda], lda,
			&a[k+1+nb+0*lda], lda, T(1), y, ldy
		);
	}
	BLAS::MultTrM("R", "U", "N", "N", k+1, nb, T(1), t, ldt, y, ldy);
}

} // namespace Util

///////////////////////////////////////////////////////////////////////
// Reduce
// ------
// Reduces a general square matrix A to upper Hessenberg form H
// by a unitary similarity transformation: Q^H * A * Q = H.
// Equivalent to Lapack routines _gehrd.
//
// The matrix Q is represented as a product of (ihi-ilo-1) elementary
// reflectors
//
//     Q = H[ilo] H[ilo+1] ... H[ihi-2].
//
// Each H[i] has the form
//
//     H[i] = I - tau * v * v'
//
// where tau is a scalar, and v is a vector with v[0:i+1] = 0,
// v[i+1] = 1 and v[ihi:n] = 0; v[i+2:ihi] is stored on exit in
// A[i+2:ihi,i], and tau in tau[i].
//
// The contents of A are illustrated by the following example, with
// n = 7, ilo = 1 and ihi = 6:
//
//     On entry:                        On exit:
//     [ a   a   a   a   a   a   a ]    [  a   a   h   h   h   h   a ]
//     [     a   a   a   a   a   a ]    [      a   h   h   h   h   a ]
//     [     a   a   a   a   a   a ]    [      h   h   h   h   h   h ]
//     [     a   a   a   a   a   a ]    [      v1  h   h   h   h   h ]
//     [     a   a   a   a   a   a ]    [      v1  v2  h   h   h   h ]
//     [     a   a   a   a   a   a ]    [      v1  v2  v3  h   h   h ]
//     [                         a ]    [                          a ]
//
// where a denotes an element of the original matrix A, h denotes a
// modified element of the upper Hessenberg matrix H, and vi denotes an
// element of the vector defining H[i].
//
// Arguments
// n     Number of rows and columns of the matrix A.
// ilo   It is assumed that A is already upper triangular in rows and
// ihi   columns 0:ilo-1 and ihi:n (that is, the reduction is performed
//       on the range ilo:ihi. Usually these are returned by a
//       balancing routine. 0 <= ilo < ihi <= n.
// a     Pointer to the first element of A.
// lda   Leading dimension of the array containing A (lda >= n).
// tau   The scale factors of the elementary reflectors, length n-1.
// lwork Length of workspace (>= n). If *lwork == 0 or NULL == work,
//       then the optimal size is returned in this argument.
// work  Workspace of size lwork.
//
template <typename T>
void Reduce(
	size_t n, size_t ilo, size_t ihi, T *a, size_t lda,
	T *tau, size_t *lwork, T *work
){
	RNPAssert(ilo <= n);
	RNPAssert(ilo < ihi);
	RNPAssert(ihi <= n);
	RNPAssert(lda >= n);
	RNPAssert(NULL != lwork);
	RNPAssert(0 == *lwork || *lwork >= n);
	
	const size_t nh = ihi-ilo;
	size_t nb = Tuning<T>::reduce_block_size_opt(n, ilo, ihi);
	
	if(0 == *lwork || NULL == work){
		*lwork = n*nb + nb*nb;
		return;
	}
	RNPAssert(NULL != a);
	RNPAssert(NULL != tau);
	
	for(size_t i = 0; i < ilo; ++i){ tau[i] = 0; }
	for(size_t i = ihi-1; i+1 < n; ++i){ tau[i] = 0; }
	
	if(nh <= 1){ return; }
	
	size_t nbmin = 2;
	size_t nx = 0;
	if(nb > 1 && nb < nh){
		nx = Tuning<T>::reduce_crossover_size(n, ilo, ihi);
		if(nx < nh){
			if(*lwork < n*nb + nb*nb){
				nbmin = Tuning<T>::reduce_block_size_min(n, ilo, ihi);
				if(*lwork >= n*nbmin){
					// We want to solve for nb in the equation
					//   *lwork == nb*nb + nb*n
					// A lower bound on nb is simply *lwork/(2*n)
					// An upper bound is *lwork/n, so we have bounds within a
					// factor 2 of being optimal.
					nb = *lwork / (2*n);
					if(0 == nb){ nb = 1; }
				}else{
					nb = 1;
				}
			}
		}
	}
	const size_t ldwork = n;
	
	T *t = work + n*nb;
	const size_t ldt = nb;
	
	size_t i = ilo;
	if(nb >= nbmin && nb < nh){ // use blocked code
		for(i = ilo; i+1+nx < ihi; i += nb){
			const size_t ib = (nb < ihi-i ? nb : ihi-i);
			// Reduce columns i..i+ib to Hessenberg form, returning the
			// matrices V and T of the block reflector H = I - V*T*V'
			// which performs the reduction, and also the matrix Y = A*V*T
			Util::Reduce_block(ihi, i, ib, &a[0+i*lda], lda, &tau[i], t, ldt, work, ldwork);
			// Apply the block reflector H to A[0..ihi,i+ib..ihi) from the
			// right, computing  A := A - Y * V'. V[i+ib,ib-2] must be set
			// to 1
			T ei(a[i+ib+(i+ib-1)*lda]);
			a[i+ib+(i+ib-1)*lda] = T(1);
			BLAS::MultMM("N","C", ihi, ihi-i-ib, ib, T(-1), work, ldwork, &a[i+ib+i*lda], lda, T(1), &a[0+(i+ib)*lda], lda);
			a[i+ib+(i+ib-1)*lda] = ei;
			// Apply the block reflector H to A[0:i,i+1..i+ib] from the right
			BLAS::MultTrM("R","L","C","U", i+1, ib-1, T(1), &a[i+1+i*lda], lda, work, ldwork);
			for(size_t j = 0; j+1 < ib; ++j){
				BLAS::Axpy(i+1, T(-1), &work[0+j*ldwork], 1, &a[0+(i+j+1)*lda], 1);
			}
			// Apply the block reflector H to A(i+1:ihi,i+ib:n) from the left
			Reflector::ApplyBlock(
				"L","C","F","C", ihi-i-1, n-i-ib, ib, &a[i+1+i*lda], lda,
				t, ldt, &a[i+1+(i+ib)*lda], lda, work, ldwork
			);
		}
	}
	Reduce_unblocked(n, i, ihi, a, lda, tau, work);
}

///////////////////////////////////////////////////////////////////////
// MultQ
// -----
// From an existing Hessenberg reduction, multiplies a given matrix by
// the unitary matrix Q. The given m-by-n matrix C is overwritten with:
//
//    trans | side = "L"   | side = "R"
//    ------|--------------|------------
//     "N"  |   Q   * C    |  C * Q
//     "C"  |   Q^H * C    |  C * Q^H
//
// where Q is a complex unitary matrix of order nq, with nq = m if
// side = "L" and nq = n if side = "R". Q is defined as the product of
// ihi-ilo-1 elementary reflectors, as returned by Reduce:
//
// Q = H[ilo] H[ilo+1] ... H[ihi-2].
// Equivalent to Lapack routines _unmhr and _ormhr.
//
// Arguments
// side  If "L", apply Q or Q^H from the left. If "R", apply Q or
//       Q^H from the right.
// trans If "N", apply Q. If "C", apply Q^H.
// m     Number of rows of the matrix C.
// n     Number of columns of the matrix C.
// ilo   ilo and ihi should have the same values as in the previous
// ihi   call to Reduce. Q is equal to the identity matrix except in
//       submatrix range Q[ilo+1:ihi,ilo+1:ihi]. If side = "L", then
//       0 <= ilo < ihi <= m. If side = "R", then 0 <= ilo < ihi <= n.
// a     Pointer to the reduction, as returned by Reduce. If side = "L",
//       then it should m columns. If side = "R", then it should have n
//       columns.
// lda   Leading dimension of the array containing A.
//       If side = "L", lda >= m. If side = "R", lda >= n.
// tau   Array of tau's. If side = "L", length m-1.
//       If side = "R", length n-1.
// c     Pointer to the first element of the matrix C.
// ldc   Leading dimension of the array containing C, ldc >= m.
// lwork Lenth of workspace.
//       If side = "L", lwork >= n. If side = "R", lwork >= m.
//       If *lwork == 0 or NULL == work, then the optimal size is
//       returned in this argument.
// work  Workspace of size lwork.
//
template <typename T>
void MultQ(
	const char *side, const char *trans, size_t m, size_t n,
	size_t ilo, size_t ihi, const T *a, size_t lda, const T *tau,
	T *c, size_t ldc, size_t *lwork, T *work // length n when side is L, else m
){
	const bool left = ('L' == side[0]);
	const size_t nh = ihi-ilo-1;
	const size_t nq = (left ? m : n);
	const size_t nw = (left ? n : m);
	
	RNPAssert('L' == side[0] || 'R' == side[0]);
	RNPAssert('N' == trans[0] || 'T' == trans[0] || 'C' == trans[0]);
	RNPAssert(ilo <= nq);
	RNPAssert(ilo < ihi);
	RNPAssert(ihi <= nq);
	RNPAssert(NULL != lwork);
	RNPAssert(0 == *lwork || *lwork >= nw);
	
	// We will go ahead and pass workspace queries directly on.
	
	size_t mi, ni, i1, i2;
	if(left){
		mi = nh;
		ni = n;
		i1 = ilo+1;
		i2 = 0;
	}else{
		mi = m;
		ni = nh;
		i1 = 0;
		i2 = ilo+1;
	}
	QR::MultQ(
		side, trans, mi, ni, nh, &a[ilo+1+ilo*lda], lda,
		&tau[ilo], &c[i1+i2*ldc], ldc, lwork, work
	);
}

///////////////////////////////////////////////////////////////////////
// GenerateQ
// ---------
// From an existing Hessenberg reduction, generates the unitary matrix
// Q. The original matrix containing the factorization is overwritten
// by Q.
// Equivalent to Lapack routines _unghr and _orghr.
//
// Arguments
// n     Number of rows and columns of the matrix Q.
// ilo   ilo and ihi should have the same values as in the previous
// ihi   call to Reduce. Q is equal to the identity matrix except in
//       submatrix range Q[ilo+1:ihi,ilo+1:ihi], 0 <= ilo < ihi <= n.
// a     Pointer to the reduction. The i-th column should contain
//       the vector which defines the i-th elementary reflector for
//       i = 0..n-1. On exit, the n-by-n matrix Q.
// lda   Leading dimension of the array containing Q, lda >= n.
// tau   Array of tau's, length n-1.
// lwork Lenth of workspace, lwork >= ihi-ilo-1.
//       If *lwork == 0 or NULL == work, then the optimal size is
//       returned in this argument.
// work  Workspace of size lwork.
//
template <typename T>
void GenerateQ(
	size_t n, size_t ilo, size_t ihi, T *a, size_t lda,
	const T *tau, size_t *lwork, T *work
){
	const size_t nh = ihi-ilo-1;
	RNPAssert(ilo <= n);
	RNPAssert(ilo < ihi);
	RNPAssert(ihi <= n);
	RNPAssert(lda >= n);
	RNPAssert(NULL != lwork);
	RNPAssert(0 == *lwork || *lwork >= nh);
	
	if(0 == *lwork || NULL == work){
		size_t nb = LA::QR::Tuning<T>::genQ_block_size_opt(nh, nh, nh);
		*lwork = nh*nb;
		return;
	}
	if(0 == n){ return; }
	
	// Shift the vectors which define the elementary reflectors one
	// column to the right, and set the first ilo+1 and the last n-ihi
	// rows and columns to those of the unit matrix
	size_t j = ihi;
	while(j --> ilo+1){
		for(size_t i = 0; i < j; ++i){
			a[i+j*lda] = T(0);
		}
		for(size_t i = j+1; i < ihi; ++i){
			a[i+j*lda] = a[i+(j-1)*lda];
		}
		for(size_t i = ihi; i < n; ++i){
			a[i+j*lda] = T(0);
		}
	}
	for(j = 0; j <= ilo; ++j){
		for(size_t i = 0; i < n; ++i){
			a[i+j*lda] = T(0);
		}
		a[j+j*lda] = T(1);
	}
	for(j = ihi; j < n; ++j){
		for(size_t i = 0; i < n; ++i){
			a[i+j*lda] = T(0);
		}
		a[j+j*lda] = T(1);
	}

	if(nh > 0){ // Generate Q[ilo+1..ihi,ilo+1..ihi]
		QR::GenerateQ(nh, nh, nh, &a[ilo+1+(ilo+1)*lda], lda, &tau[ilo], lwork, work);
	}
}


///////////////////////////////////////////////////////////////////////
// ReduceGeneralized_unblocked
// ---------------------------
// Reduces a pair of matrices (A,B) to generalized upper Hessenberg
// form using unitary transformations, where A is a general matrix
// and B is upper triangular.  The form of the generalized eigenvalue
// problem is
//
//     A*x = lambda*B*x,
//
// and B is typically made upper triangular by computing its QR
// factorization and moving the unitary matrix Q to the left side
// of the equation.
//
// This subroutine simultaneously reduces A to a Hessenberg matrix H:
//
//     Q^H*A*Z = H
//
// and transforms B to another upper triangular matrix T:
//
//     Q^H*B*Z = T
//
// in order to reduce the problem to its standard form
//
//     H*y = lambda*T*y
//
// where y = Z^H*x.
//
// The unitary matrices Q and Z are determined as products of Givens
// rotations. They may either be formed explicitly, or they may be
// postmultiplied into input matrices Q1 and Z1, so that
//
//     Q1 * A * Z1^H = (Q1*Q) * H * (Z1*Z)^H
//     Q1 * B * Z1^H = (Q1*Q) * T * (Z1*Z)^H
//
// If Q1 is the unitary matrix from the QR factorization of B in the
// original equation A*x = lambda*B*x, then this routine reduces the
// original problem to generalized Hessenberg form.
//
// This is equivalent to Lapack routine _gghrd.
//
// This routine performs the Hessenberg-triangular reduction by
// an unblocked reduction, as described in "Matrix_Computations",
// by Golub and van Loan (Johns Hopkins Press).
//
// Arguments
// n     The number of rows and columns and A and B.
// ilo   It is assumed that A is already upper triangular in rows and
// ihi   columns 0:ilo-1 and ihi:n (that is, the reduction is performed
//       on the range ilo:ihi. Usually these are returned by a
//       balancing routine. 0 <= ilo < ihi <= n.
// a     Pointer to the first element of A. On exit, the upper triangle
//       and the first subdiagonal of A are overwritten with the upper
//       Hessenberg matrix H, and the rest is set to zero.
// lda   Leading dimension of the array containing A (lda >= n).
// b     Pointer to the first element of B. On exit, the upper
//       triangular matrix T = Q^H B Z.  The elements below the
//       diagonal are set to zero.
// ldb   Leading dimension of the array containing B (ldb >= n).
// q     Pointer to the first element of Q. If NULL, then Q is not
//       returned. If initialized to the identity matrix, then the
//       matrix Q is returned. If initialized to Q1, then the product
//       Q1*Q is returned.
// ldq   Leading dimension of the array containing Q (ldq >= n).
// z     Pointer to the first element of Z. If NULL, then Z is not
//       returned. If initialized to the identity matrix, then the
//       matrix Z is returned. If initialized to Z1, then the product
//       Z1*Z is returned.
// ldz   Leading dimension of the array containing Z (ldz >= n).
//
template <typename T>
void ReduceGeneralized_unblocked(
	size_t n, size_t ilo, size_t ihi,
	T *a, size_t lda, T *b, size_t ldb,
	T *q, size_t ldq, T *z, size_t ldz
){
	if(n <= 1){ return; }

	// Zero out lower triangle of B
	for(size_t jcol = 0; jcol+1 < n; ++jcol){
		for(size_t jrow = jcol+1; jrow < n; ++jrow){
			b[jrow+jcol*ldb] = T(0);
		}
	}

	// Reduce A and B
	for(size_t jcol = ilo; jcol+3 <= ihi; ++jcol){
		size_t jrow = ihi;
		while(jrow --> jcol+2){
			typename Traits<T>::real_type c;
			T s, ctemp;
			
			// Step 1: rotate rows jrow-1, jrow to kill a(jrow,jcol)
			ctemp = a[jrow-1+jcol*lda];
			Rotation::Generate(ctemp, a[jrow+jcol*lda], &c, &s, &a[jrow-1+jcol*lda]);
			a[jrow+jcol*lda] = T(0);
			Rotation::Apply(n-jcol-1, &a[jrow-1+(jcol+1)*lda], lda, &a[jrow+(jcol+1)*lda], lda, c, s);
			Rotation::Apply(n+1-jrow, &b[jrow-1+(jrow-1)*ldb], ldb, &b[jrow+(jrow-1)*ldb], ldb, c, s);
			if(NULL != q){
				Rotation::Apply(n, &q[0+(jrow-1)*ldq], 1, &q[0+jrow*ldq], 1, c, Traits<T>::conj(s));
			}

			// Step 2: rotate columns jrow, jrow-1 to kill b(jrow,jrow-1)
			ctemp = b[jrow+jrow*ldb];
			Rotation::Generate(ctemp, b[jrow+(jrow-1)*ldb], &c, &s, &b[jrow+jrow*ldb]);
			b[jrow+(jrow-1)*ldb] = T(0);
			Rotation::Apply(ihi , &a[0+jrow*lda], 1, &a[0+(jrow-1)*lda], 1, c, s);
			Rotation::Apply(jrow, &b[0+jrow*ldb], 1, &b[0+(jrow-1)*ldb], 1, c, s);
			if(NULL != z){
				Rotation::Apply(n, &z[0+jrow*ldz], 1, &z[0+(jrow-1)*ldz], 1, c, s);
			}
		}
	}
}


} // namespace Hessenberg
} // namespace LA
} // namespace RNP

#endif // RNP_HESSENBERG_HPP_INCLUDED
