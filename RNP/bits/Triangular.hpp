#ifndef RNP_TRIANGULAR_HPP_INCLUDED
#define RNP_TRIANGULAR_HPP_INCLUDED

#include <cstddef>
#include <RNP/Types.hpp>
#include <RNP/BLAS.hpp>
#include <RNP/Debug.hpp>


namespace RNP{
namespace LA{
namespace Triangular{

///////////////////////////////////////////////////////////////////////
// RNP::LA::Triangular
// ===================
// Utility routines dealing with triangular matrices.
//

///////////////////////////////////////////////////////////////////////
// Tuning
// ------
// Specialize this class to tune the block sizes.
//
template <typename T>
struct Tuning{
	static inline size_t invert_block_size(const char *uplo, const char *diag, size_t n){ return 64; }
};

///////////////////////////////////////////////////////////////////////
// Invert_unblocked
// ----------------
// Inverts a triangular matrix in-place.
// This corresponds to Lapackk routines _trti2.
// This routine uses only level 2 BLAS.
//
// Arguments
// uplo If "U", the matrix is upper triangular.
//      If "L", the matrix is lower triangular.
// diag If "U", the matrix is assumed to have only 1's on the diagonal.
//      If "N", the diagonal is given.
// n   Number of rows and columns of the matrix.
// a   Pointer to the first element of the matrix.
// lda Leading dimension of the array containing the matrix, lda >= n.
//
template <typename T>
void Invert_unblocked(
	const char *uplo, const char *diag,
	size_t n, T *a, size_t lda
){
	RNPAssert(lda >= n);
	RNPAssert('U' == uplo[0] || 'L' == uplo[0]);
	RNPAssert('U' == diag[0] || 'N' == diag[0]);
	if('U' == uplo[0]){
		for(size_t j = 0; j < n; ++j){
			T Ajj;
			if('U' != diag[0]){
				a[j+j*lda] = T(1) / a[j+j*lda];
				Ajj = -a[j+j*lda];
			}else{
				Ajj = T(-1);
			}
			if(j > 0){
				// Compute elements 1:j-1 of j-th column
				BLAS::MultTrV("U","N",diag, j, a, lda, &a[0+j*lda], 1);
				BLAS::Scale(j, Ajj, &a[0+j*lda], 1);
			}
		}
	}else{
		size_t j = n;
		while(j --> 0){
			T Ajj;
			if('U' != diag[0]){
				a[j+j*lda] = T(1) / a[j+j*lda];
				Ajj = -a[j+j*lda];
			}else{
				Ajj = T(-1);
			}
			if(j+1 < n){
				// Compute elements 1:j-1 of j-th column
				BLAS::MultTrV("L","N",diag, n-1-j, &a[j+1+(j+1)*lda], lda, &a[j+1+j*lda], 1);
				BLAS::Scale(n-1-j, Ajj, &a[j+1+j*lda], 1);
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////
// Invert
// ------
// Inverts a triangular matrix in-place.
// This corresponds to Lapackk routines _trtri.
//
// Arguments
// uplo If "U", the matrix is upper triangular.
//      If "L", the matrix is lower triangular.
// diag If "U", the matrix is assumed to have only 1's on the diagonal.
//      If "N", the diagonal is given.
// n   Number of rows and columns of the matrix.
// a   Pointer to the first element of the matrix.
// lda Leading dimension of the array containing the matrix, lda >= n.
//
template <typename T>
int Invert(
	const char *uplo, const char *diag,
	size_t n, T *a, size_t lda
){
	RNPAssert(lda >= n);
	RNPAssert('U' == uplo[0] || 'L' == uplo[0]);
	RNPAssert('U' == diag[0] || 'N' == diag[0]);
	if(0 == n){ return 0; }
	if('N' == diag[0]){
		for(size_t i = 0; i < n; ++i){
			if(T(0) == a[i+i*lda]){
				return -(int)(i+1);
			}
		}
	}
	
	size_t nb = RNP::LA::Triangular::Tuning<T>::invert_block_size(uplo, diag, n);
	if(nb <= 1 || nb >= n){
		Invert_unblocked(uplo, diag, n, a, lda);
	}else{
		if('U' == uplo[0]){
			for(size_t j = 0; j < n; j += nb){
				const size_t jb = (nb+j < n ? nb : n-j);
				// Compute rows 1:j-1 of current block column
				BLAS::MultTrM("L","U","N",diag, j, jb, T(1), a, lda, &a[0+j*lda], lda);
				BLAS::SolveTrM("R","U","N",diag, j, jb, T(-1), &a[j+j*lda], lda, &a[0+j*lda], lda);
				// Compute inverse of current diagonal block
				Invert_unblocked("U",diag, jb, &a[j+j*lda], lda);
			}
		}else{
			size_t j = (n / nb) * nb;
			while(j --> 0){
				const size_t jb = (nb+j < n ? nb : n-j);
				if(j+jb < n){ // comput rows j+jb:n of current block column
					BLAS::MultTrM("L","L","N",diag, n-j-jb, jb, T(1), &a[j+jb+(j+jb)*lda], lda, &a[j+jb+j*lda], lda);
					BLAS::SolveTrM("R","L","N",diag, n-j-jb, jb, T(-1), &a[j+j*lda], lda, &a[j+jb+j*lda], lda);
				}
				// Compute inverse of current diagonal block
				Invert_unblocked("L", diag, jb, &a[j+j*lda], lda);
			}
		}
	}
	return 0;
}

///////////////////////////////////////////////////////////////////////
// Copy
// ----
// Copies a triangular matrix.
//
// Arguments
// uplo  If "U", the matrix is upper triangular.
//       If "L", the matrix is lower triangular.
// diag  If "U", the matrix is assumed to have only 1's on the diagonal.
//       If "N", the diagonal is given.
// m     Number of rows of the matrix.
// n     Number of columns of the matrix.
// src   Pointer to the first element of the source matrix.
// ldsrc Leading dimension of the array containing the source
//       matrix, ldsrc >= m.
// dst   Pointer to the first element of the destination matrix.
// lddst Leading dimension of the array containing the destination
//       matrix, lddst >= m.
//
template <typename T>
void Copy(
	const char *uplo, const char *diag, size_t m, size_t n,
	const T* src, size_t ldsrc,
	T* dst, size_t lddst
){
	if('L' == uplo[0]){
		for(size_t j = 0; j < n; ++j){
			size_t i0 = ('N' == diag[0] ? j : j+1);
			for(size_t i = i0; i < m; ++i){
				dst[i+j*lddst] = src[i+j*ldsrc];
			}
		}
	}else{
		for(size_t j = 0; j < n; ++j){
			size_t ilim = ('N' == diag[0] ? j+1 : j);
			if(m < ilim){ ilim = m; }
			for(size_t i = 0; i < ilim; ++i){
				dst[i+j*lddst] = src[i+j*ldsrc];
			}
		}
	}
}




///////////////////////////////////////////////////////////////////////
// Solve
// -----
// Solves a triangular system of equations with the scale factor set
// to prevent overflow. This routine solves one of the triangular
// systems
// 
//     A * x = s*b,  A^T * x = s*b,  or  A^H * x = s*b,
// 
// with scaling to prevent overflow. Here A is an upper or lower
// triangular matrix, A^T denotes the transpose of A, A^H denotes the
// conjugate transpose of A, x and b are n-element vectors, and s is a
// scaling factor, usually less than or equal to 1, chosen so that the
// components of x will be less than the overflow threshold.  If the
// unscaled problem will not cause overflow, the Level 2 BLAS routine
// SolveTrV is called. If the matrix A is singular (A[j,j] = 0 for
// some j), then s is set to 0 and a non-trivial solution to A*x = 0
// is returned.
// This corresponds to Lapackk routines _latrs.
//
//
// ### Authors
//
// * Univ. of Tennessee
// * Univ. of California Berkeley
// * Univ. of Colorado Denver
// * NAG Ltd.
//
// ### Details
// 
// A rough bound on x is computed; if that is less than overflow,
// SolveTrV is called, otherwise, specific code is used which checks
// for possible overflow or divide-by-zero at every operation.
// 
// A columnwise scheme is used for solving A*x = b. The basic algorithm
// if A is lower triangular is
// 
//     x[0..n] := b[0..n]
//     for j = 0..n
//         x[j] /= A[j,j]
//         x[j+1..n] -= x[j] * A[j+1..n,j]
//     end
// 
// Define bounds on the components of x after j iterations of the loop:
//
//     M[j] = bound on x[0..j]
//     G[j] = bound on x[j+1..n]
//
// Initially, let M[0] = 0 and G[0] = max{x[i], i=0..n}. 
// Then for iteration j+1 we have
//
//     M[j+1] <= G[j] / | A[j+1,j+1] |
//     G[j+1] <= G[j] + M[j+1] * | A[j+2..n,j+1] |
//            <= G[j] ( 1 + cnorm(j+1) / | A[j+1,j+1] | )
// 
// where CNORM(j+1) is greater than or equal to the infinity-norm of
// column j+1 of A, not counting the diagonal.  Hence
// 
//     G[j] <= G[0]  Prod ( 1 + cnorm[i] / | A[i,i] | )
//                  i=1..j+1
//
// and
// 
//     |x[j]| <= ( G[0] / |A[j,j]| ) Prod ( 1 + cnorm[i] / |A[i,i]| )
//                                  i=1..j
// 
// Since |x[j]| <= M[j], we use the Level 2 BLAS routine SolveTrV
// if the reciprocal of the largest M(j), j=1,..,n, is larger than
// max(underflow, 1/overflow).
// 
// The bound on x[j] is also used to determine when a step in the
// columnwise method can be performed without fear of overflow.  If
// the computed bound is greater than a large constant, x is scaled to
// prevent overflow, but if the bound overflows, x is set to 0, x[j] to
// 1, and scale to 0, and a non-trivial solution to A*x = 0 is found.
// 
// Similarly, a row-wise scheme is used to solve A^T *x = b  or
// A^H *x = b.  The basic algorithm for A upper triangular is
// 
//     for j = 0..n
//         x[j] := ( b[j] - A[0..j,j]' * x[0..j] ) / A[j,j]
//     end
// 
// We simultaneously compute two bounds
//
//     G[j] = bound on ( b[i] - A[0..i,i]' * x[0..i] ), i=0..j+1
//     M[j] = bound on x[i], i=0..j+1
// 
// The initial values are G[0] = 0, M[0] = max{b[i], i=0..n}, and we
// add the constraint G[j] >= G[j-1] and M[j] >= M[j-1] for j >= 1.
// Then the bound on x[j] is
// 
//     M[j] <= M[j-1] * ( 1 + cnorm[j] ) / | A[j,j] |
//          <= M[0] *  Prod ( ( 1 + cnorm[i] ) / |A[i,i]| )
//                   i=1..j+1
// 
// and we can safely call SolveTrV if 1/M[n] and 1/G[n] are both greater
// than max(underflow, 1/overflow).
//
// Arguments
// uplo   If "U", the upper triangle of A is given.
//        If "L", the lower triangle of A is given.
// trans  If "N", op(A) = A. If "T", op(A) = A^T. If "C", op(A) = A^H.
// diag   If "U", the diagonal of A is assumed to be all 1's.
//        If "N", the diagonal of A is given.
// normin If "Y", cnorm contains column norms on entry.
//        If "N", cnorm will be filled in.
// n      Number of rows and columns of A.
// a      Pointer to the first element of A.
// lda    Leading dimension of the array containing A, lda >= n.
// x      Pointer to the first element of the x vector. On entry, it
//        is the right hand side vector b. On exit, it is overwritten
//        by the solution x.
// incx  Increment between elements of the x vector, incx > 0.
// scale Returned scaling factor s for the triangular system. If zero,
//       the matrix A is singular or badly scaled, and the vector x
//       is an exact or approximate solution to A*x = 0.
// cnorm Length n array of column norms. If normin = "Y", cnorm is an
//       input argument and cnorm[j] contains the norm of the off-
//       diagonal part of the j-th column of A. If trans = "N",
//       cnorm[j] must be greater than or equal to the infinity-norm,
//       and if trans = "T" or "C", cnorm[j] must be greater than or
//       equal to the 1-norm. If normin = "N", cnorm is an output
//       argument and cnorm[j] returns the 1-norm of the offdiagonal
//       part of the j-th column of A.
//
template <typename T>
void Solve(
	const char *uplo, const char *trans,
	const char *diag, const char *normin,
	size_t n, T *a, size_t lda, T *x,
	typename Traits<T>::real_type *scale,
	typename Traits<T>::real_type *cnorm
){
	typedef typename Traits<T>::real_type real_type;
	static const real_type zero(0);
	static const real_type one(1);
	static const real_type two(2);
	static const real_type half(one / two);
	
	using namespace std;
	
	
	RNPAssert(NULL != uplo);
	RNPAssert('U' == uplo[0] || 'L' == uplo[0]);
	RNPAssert(NULL != trans);
	RNPAssert('N' == trans[0] || 'T' == trans[0] || 'C' == trans[0]);
	RNPAssert(NULL != diag);
	RNPAssert('U' == diag[0] || 'N' == diag[0]);
	RNPAssert(NULL != normin);
	RNPAssert('Y' == normin[0] || 'N' == normin[0]);
	RNPAssert(lda >= n);

    real_type grow;
    real_type tscal;
    int jfirst, jlast, jinc;

	if(0 == n){ return; }
	
	const bool upper  = ('U' == uplo [0]);
	const bool notran = ('N' == trans[0]);
	const bool nounit = ('N' == diag [0]);

	// On a cray, these should be sqrt(min)/2*eps and sqrt(1/smlnum)
	static const real_type smlnum(Traits<real_type>::min() / (real_type(2) * Traits<real_type>::min()));
	static const real_type bignum(real_type(1)/smlnum);
	*scale = real_type(1);

	if('N' == normin[0]){
		// Compute the 1-norm of each column, not including the diagonal.
		if(upper){
			for(size_t j = 0; j < n; ++j){
				cnorm[j] = BLAS::Asum(j, &a[0+j*lda], 1);
			}
		}else{
			for(size_t j = 0; j+1 < n; ++j){
				cnorm[j] = BLAS::Asum(n-j-1, &a[(j+1)+j*lda], 1);
			}
			cnorm[n-1] = real_type(0);
		}
	}

	// Scale the column norms by TSCAL if the maximum element in CNORM is
	// greater than BIGNUM/2.
	{
		const size_t imax = BLAS::MaximumIndex(n, cnorm, 1);
		real_type tmax = cnorm[imax];
		if(tmax <= bignum * half){
			tscal = one;
		}else{
			tscal = half / (smlnum * tmax);
			BLAS::Scale(n, tscal, cnorm, 1);
		}
	}

	// Compute a bound on the computed solution vector to see if the
	// Level 2 BLAS routine ZTRSV can be used.

    real_type xmax = zero;
    for(size_t j = 0; j < n; ++j){
		real_type d1 = Traits<T>::norm1(half * x[j]);
		if(d1 > xmax){ xmax = d1; }
    }
	real_type xbnd = xmax;

    if(notran){ // Compute the growth in A * x = b.
		if(upper){
			jfirst = n;
			jlast = 1;
			jinc = -1;
		}else{
			jfirst = 1;
			jlast = n;
			jinc = 1;
		}

		if(tscal != one){
			grow = zero;
		}else{
			if(nounit){
				// Compute GROW = 1/G(j) and XBND = 1/M(j).
				// Initially, G(0) = max{x(i), i=1,...,n}.
				grow = half / (xbnd > smlnum ? xbnd : smlnum);
				xbnd = grow;
				bool toosmall = false;
				for(int jj = jfirst; jinc < 0 ? jj >= jlast : jj <= jlast; jj += jinc){
					size_t j = jj-1;
					// Exit the loop if the growth factor is too small.
					if(grow <= smlnum){
						toosmall = true;
						break;
					}

					T tjjs(a[j+j*lda]);
					real_type tjj = Traits<T>::norm1(tjjs);

					if(tjj >= smlnum){ // M(j) = G(j-1) / abs(A(j,j))
						real_type d1 = (one < tjj ? one : tjj) * grow;
						if(d1 < xbnd){ xbnd = d1; }
					}else{ // M(j) could overflow, set XBND to 0.
						xbnd = zero;
					}

					if(tjj + cnorm[j] >= smlnum){
						// G(j) = G(j-1)*( 1 + CNORM(j) / abs(A(j,j)) )
						grow *= tjj / (tjj + cnorm[j]);
					}else{ // G(j) could overflow, set GROW to 0.
						grow = zero;
					}
				}
				if(!toosmall){
					grow = xbnd;
				}
			}else{ // A is unit triangular.
				// Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.
				grow = half / (xbnd > smlnum ? xbnd : smlnum);
				if(one < grow){ grow = one; }
				for(int jj = jfirst; jinc < 0 ? jj >= jlast : jj <= jlast; jj += jinc){
					size_t j = jj-1;
					// Exit the loop if the growth factor is too small.
					if(grow <= smlnum){
						break;
					}

					// G(j) = G(j-1)*( 1 + CNORM(j) )
					grow *= one / (cnorm[j] + 1.);
				}
			}
		}
    }else{ // Compute the growth in A**T * x = b  or  A**H * x = b.
		if(upper){
			jfirst = 1;
			jlast = n;
			jinc = 1;
		}else{
			jfirst = n;
			jlast = 1;
			jinc = -1;
		}

		if(tscal != one){
			grow = zero;
		}else{
			if(nounit){ // A is non-unit triangular.
				// Compute GROW = 1/G(j) and XBND = 1/M(j).
				// Initially, M(0) = max{x(i), i=1,...,n}.

				grow = half / (xbnd > smlnum ? xbnd : smlnum);
				xbnd = grow;
				bool toosmall = false;
				for(int jj = jfirst; jinc < 0 ? jj >= jlast : jj <= jlast; jj += jinc){
					size_t j = jj-1;
					// Exit the loop if the growth factor is too small.
					if(grow <= smlnum){
						toosmall = true;
						break;
					}
					// G(j) = max( G(j-1), M(j-1)*( 1 + CNORM(j) ) )
					real_type xj = cnorm[j] + one;
					real_type d1 = xbnd/xj;
					if(d1 < grow){ grow = d1; }

					T tjjs(a[j+j*lda]);
					real_type tjj = Traits<T>::norm1(tjjs);

					if(tjj >= smlnum){
						// M(j) = M(j-1)*( 1 + CNORM(j) ) / abs(A(j,j))
						if(xj > tjj){
							xbnd *= tjj / xj;
						}
					}else{ // M(j) could overflow, set XBND to 0.
						xbnd = zero;
					}
				}
				if(!toosmall){
					if(xbnd < grow){ grow = xbnd; }
				}
			}else{ // A is unit triangular.
				// Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.
				grow = half / (xbnd > smlnum ? xbnd : smlnum);
				if(one < grow){ grow = one; }
				for(int jj = jfirst; jinc < 0 ? jj >= jlast : jj <= jlast; jj += jinc){
					size_t j = jj-1;
					// Exit the loop if the growth factor is too small.
					if(grow <= smlnum){
						break;
					}
					// G(j) = ( 1 + CNORM(j) )*G(j-1)
					real_type xj = cnorm[j] + one;
					grow /= xj;
				}
			}
		}
    }

    if(grow * tscal > smlnum){
		// Use the Level 2 BLAS solve if the reciprocal of the bound on
		// elements of X is not too small.

		BLAS::SolveTrV(uplo, trans, diag, n, a, lda, x, 1);
    }else{ // Use a Level 1 BLAS solve, scaling intermediate results.
		if(xmax > bignum * half){
			// Scale X so that its components are less than or equal to
			// BIGNUM in absolute value.
			*scale = bignum * half / xmax;
			BLAS::Scale(n, *scale, x, 1);
			xmax = bignum;
		}else{
			xmax *= two;
		}
		
		T tjjs;

		if(notran){ // Solve A * x = b
			for(int jj = jfirst; jinc < 0 ? jj >= jlast : jj <= jlast; jj += jinc){
				size_t j = jj-1;
				// Compute x(j) = b(j) / A(j,j), scaling x if necessary.
				real_type xj = Traits<T>::norm1(x[j]);
				bool skipscaling = false;
				if(nounit){
					tjjs = tscal * a[j+j*lda];
				}else{
					tjjs = tscal;
					if(tscal == one){
						skipscaling = true;
					}
				}
				if(!skipscaling){
					real_type tjj = Traits<T>::norm1(tjjs);
					if(tjj > smlnum){ // abs(A(j,j)) > SMLNUM:
						if(tjj < one){
							if(xj > tjj * bignum){
								// Scale x by 1/b(j).
								real_type rec = one / xj;
								BLAS::Scale(n, rec, x, 1);
								*scale *= rec;
								xmax *= rec;
							}
						}
						x[j] = Traits<T>::div(x[j], tjjs);
						xj = Traits<T>::norm1(x[j]);
					}else if(tjj > zero){
						// 0 < abs(A(j,j)) <= SMLNUM:
						if(xj > tjj * bignum){
							// Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM
							// to avoid overflow when dividing by A(j,j).

							real_type rec = tjj * bignum / xj;
							if(cnorm[j] > one){
								// Scale by 1/CNORM(j) to avoid overflow when
								// multiplying x(j) times column j.

								rec /= cnorm[j];
							}
							BLAS::Scale(n, rec, x, 1);
							*scale *= rec;
							xmax *= rec;
						}
						x[j] = Traits<T>::div(x[j], tjjs);
						xj = Traits<T>::norm1(x[j]);
					}else{
						// A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
						// scale = 0, and compute a solution to A*x = 0.
						for(size_t i = 0; i < n; ++i){
							x[i] = zero;
						}
						x[j] = one;
						xj = one;
						*scale = zero;
						xmax = zero;
					}
				}

				// Scale x if necessary to avoid overflow when adding a
				// multiple of column j of A.
				if(xj > one){
					real_type rec = one / xj;
					if(cnorm[j] > (bignum - xmax) * rec){
						// Scale x by 1/(2*abs(x(j))).

						rec *= half;
						BLAS::Scale(n, rec, x, 1);
						*scale *= rec;
					}
				}else if(xj * cnorm[j] > bignum - xmax){
					// Scale x by 1/2.
					BLAS::Scale(n, half, x, 1);
					*scale *= half;
				}

				if(upper){
					if(j > 0){
						// Compute the update
						// x(1:j-1) := x(1:j-1) - x(j) * A(1:j-1,j)
						BLAS::Axpy(j, -tscal * x[j], &a[0+j*lda], 1, x, 1);
						size_t i = BLAS::MaximumIndex(j, x, 1);
						xmax = Traits<T>::norm1(x[i]);
					}
				}else{
					if(j+1 < n){
						// Compute the update
						// x(j+1:n) := x(j+1:n) - x(j) * A(j+1:n,j)
						BLAS::Axpy(n-j-1, -tscal * x[j], &a[(j+1)+j*lda], 1, &x[j+1], 1);
						size_t i = j+1 + BLAS::MaximumIndex(n-j-1, &x[j+1], 1);
						xmax = Traits<T>::norm1(x[i]);
					}
				}
			}
		}else if ('T' == trans[0]){ // Solve A**T * x = b
			for(int jj = jfirst; jinc < 0 ? jj >= jlast : jj <= jlast; jj += jinc){
				size_t j = jj-1;
				// Compute x(j) = b(j) - sum A(k,j)*x(k).
				// k<>j

				real_type xj = Traits<T>::norm1(x[j]);
				T uscal = tscal;
				real_type rec = one / (xmax > one ? xmax : one);
				if(cnorm[j] > (bignum - xj) * rec){

					// If x(j) could overflow, scale x by 1/(2*XMAX).

					rec *= half;
					if(nounit){
						tjjs = tscal * a[j+j*lda];
					}else{
						tjjs = tscal;
					}
					real_type tjj = Traits<T>::norm1(tjjs);
					if(tjj > one){
						// Divide by A(j,j) when scaling x if A(j,j) > 1.
						rec *= tjj;
						if(one < rec){ rec = one; }
						uscal = Traits<T>::div(uscal, tjjs);
					}
					if(rec < one){
						BLAS::Scale(n, rec, x, 1);
						*scale *= rec;
						xmax *= rec;
					}
				}

				T csumj(0);
				if(one == uscal){
					// If the scaling needed for A in the dot product is 1,
					// call Dot to perform the dot product.
					if(upper){
						csumj = BLAS::Dot(j, &a[0+j*lda], 1, x, 1);
					}else if(j+1 < n){
						csumj = BLAS::Dot(n-j-1, &a[(j+1)+j*lda], 1, &x[j+1], 1);
					}
				}else{
					// Otherwise, use in-line code for the dot product.
					if(upper){
						for(size_t i = 0; i < j; ++i){
							csumj += (a[i+j*lda]*uscal)*x[i];
						}
					}else if(j+1 < n){
						for(size_t i = j+1; i < n; ++i){
							csumj += (a[i+j*lda]*uscal)*x[i];
						}
					}
				}

				if(uscal == tscal){
					// Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j)
					// was not used to scale the dotproduct.
					x[j] -= csumj;
					xj = Traits<T>::norm1(x[j]);
					bool skipscaling = false;
					if(nounit){
						tjjs = tscal * a[j+j*lda];
					}else{
						tjjs = tscal;
						if(tscal == one){
							skipscaling = true;
						}
					}
					if(!skipscaling){
						// Compute x(j) = x(j) / A(j,j), scaling if necessary.
						real_type tjj = Traits<T>::norm1(tjjs);
						if(tjj > smlnum){ // abs(A(j,j)) > SMLNUM:

							if(tjj < one){
								if(xj > tjj * bignum){ // Scale X by 1/abs(x(j)).

									real_type rec = one / xj;
									BLAS::Scale(n, rec, x, 1);
									*scale *= rec;
									xmax *= rec;
								}
							}
							x[j] = Traits<T>::div(x[j], tjjs);
						}else if(tjj > zero){ // 0 < abs(A(j,j)) <= SMLNUM:
							if(xj > tjj * bignum){
								// Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM.
								real_type rec = tjj * bignum / xj;
								BLAS::Scale(n, rec, x, 1);
								*scale *= rec;
								xmax *= rec;
							}
							x[j] = Traits<T>::div(x[j], tjjs);
						}else{
							// A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
							// scale = 0 and compute a solution to A**T *x = 0.
							for(size_t i = 0; i < n; ++i){
								x[i] = zero;
							}
							x[j] = one;
							*scale = zero;
							xmax = zero;
						}
					}
				}else{
					// Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot
					// product has already been divided by 1/A(j,j).
					x[j] = Traits<T>::div(x[j], tjjs) - csumj;
					
				}
				real_type d1 = Traits<T>::norm1(x[j]);
				if(d1 > xmax){ xmax = d1; }
			}
		}else{ // Solve A**H * x = b
			for(int jj = jfirst; jinc < 0 ? jj >= jlast : jj <= jlast; jj += jinc){
				size_t j = jj-1;
				// Compute x(j) = b(j) - sum A(k,j)*x(k).
				// k<>j
				real_type xj = Traits<T>::norm1(x[j]);
				T uscal = tscal;
				real_type rec = one / (xmax > one ? xmax : one);
				if(cnorm[j] > (bignum - xj) * rec){
					// If x(j) could overflow, scale x by 1/(2*XMAX).
					rec *= half;
					if(nounit){
						tjjs = tscal * Traits<T>::conj(a[j+j*lda]);
					}else{
						tjjs = tscal;
					}
					real_type tjj = Traits<T>::norm1(tjjs);
					if(tjj > one){ // Divide by A(j,j) when scaling x if A(j,j) > 1.
						rec *= tjj;
						if(one < rec){ rec = one; }
						uscal = Traits<T>::div(uscal, tjjs);
					}
					if(rec < one){
						BLAS::Scale(n, rec, x, 1);
						*scale *= rec;
						xmax *= rec;
					}
				}

				T csumj(0);
				if(uscal == one){
					// If the scaling needed for A in the dot product is 1,
					// call ZDOTC to perform the dot product.
					if(upper){
						csumj = BLAS::ConjugateDot(j, &a[0+j*lda], 1, x, 1);
					}else if(j+1 < n){
						csumj = BLAS::ConjugateDot(n-j-1, &a[(j+1)+j*lda], 1, &x[j+1], 1);
					}
				}else{ // Otherwise, use in-line code for the dot product.
					if(upper){
						for(size_t i = 0; i < j; ++i){
							csumj += (Traits<T>::conj(a[i+j*lda]) * uscal) * x[i];
						}
					}else if(j+1 < n){
						for(size_t i = j+1; i < n; ++i){
							csumj += (Traits<T>::conj(a[i+j*lda]) * uscal) * x[i];
						}
					}
				}

				if(tscal == uscal){
					// Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j)
					// was not used to scale the dotproduct.
					x[j] -= csumj;
					xj = Traits<T>::norm1(x[j]);
					bool skipscaling = false;
					if(nounit){
						tjjs = Traits<T>::conj(a[j+j*lda])*tscal;
					}else{
						tjjs = tscal;
						if(tscal == one){
							skipscaling = true;
						}
					}

					if(!skipscaling){
						// Compute x(j) = x(j) / A(j,j), scaling if necessary.
						real_type tjj = Traits<T>::norm1(tjjs);
						if(tjj > smlnum){ // abs(A(j,j)) > SMLNUM:
							if(tjj < one){
								if(xj > tjj * bignum){

								// Scale X by 1/abs(x(j)).

								real_type rec = one / xj;
								BLAS::Scale(n, rec, x, 1);
								*scale *= rec;
								xmax *= rec;
								}
							}
							x[j] = Traits<T>::div(x[j], tjjs);
						}else if(tjj > zero){
							// 0 < abs(A(j,j)) <= SMLNUM:
							if(xj > tjj * bignum){
								// Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM.
								real_type rec = tjj * bignum / xj;
								BLAS::Scale(n, rec, x, 1);
								*scale *= rec;
								xmax *= rec;
							}
							x[j] = Traits<T>::div(x[j], tjjs);
						}else{
							// A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
							// scale = 0 and compute a solution to A**H *x = 0.
							for(size_t i = 0; i < n; ++i){
								x[i] = zero;
							}
							x[j] = one;
							*scale = zero;
							xmax = zero;
						}
					}
				}else{
					// Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot
					// product has already been divided by 1/A(j,j).
					x[j] = Traits<T>::div(x[j], tjjs) - csumj;
				}
				real_type d1 = Traits<T>::norm1(x[j]);
				if(d1 > xmax){ xmax = d1; }
			}
		}
		*scale /= tscal;
    }

    if(one != tscal){
		BLAS::Scale(n, one / tscal, cnorm, 1);
    }
}

} // namespace Triangular
} // namespace LA
} // namespace RNP

#endif // RNP_TRIANGULAR_HPP_INCLUDED
