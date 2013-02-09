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
	const char *uplo, size_t m, size_t n,
	const T* src, size_t ldsrc,
	T* dst, size_t lddst
){
	if('L' == uplo[0]){
		for(size_t j = 0; j < n; ++j){
			for(size_t i = j; i < m; ++i){
				dst[i+j*lddst] = src[i+j*ldsrc];
			}
		}
	}else{
		for(size_t j = 0; j < n; ++j){
			size_t ilim = (m < j+1 ? m : j+1);
			for(size_t i = 0; i < ilim; ++i){
				dst[i+j*lddst] = src[i+j*ldsrc];
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////
// Set
// ---
// Sets a triangular matrix.
//
// Arguments
// uplo    If "U", the matrix is upper triangular.
//         If "L", the matrix is lower triangular.
// m       Number of rows of the matrix.
// n       Number of columns of the matrix.
// offdiag Value to set the offdiagonal elements.
// diag    Value to set the diagonal elements.
// a       Pointer to the first element of the matrix.
// lda     Leading dimension of the array containing the matrix,
//         lda >= m.
//
template <typename T, typename TS>
void Set(
	const char *uplo, size_t m, size_t n,
	const TS &offdiag, const TS &diag,
	T* a, size_t lda
){
	if('L' == uplo[0]){
		for(size_t j = 0; j < n; ++j){
			if(j < m){ a[j+j*lda] = diag; }
			for(size_t i = j+1; i < m; ++i){
				a[i+j*lda] = offdiag;
			}
		}
	}else{
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < j; ++i){
				a[i+j*lda] = offdiag;
			}
			if(j < m){ a[j+j*lda] = diag; }
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


///////////////////////////////////////////////////////////////////////
// Eigenvectors
// ------------
// Computes some or all of the right and/or left eigenvectors of an
// upper triangular matrix T.
// Matrices of this type are produced by the Schur factorization of
// a complex general matrix: A = Q*T*Q^H.
// 
// The right eigenvector x and the left eigenvector y of T corresponding
// to an eigenvalue w are defined by:
// 
//     T*x = w*x,     (y^H)*T = w*(y^H)
// 
// where y^H denotes the conjugate transpose of the vector y.
// The eigenvalues are not input to this routine, but are read directly
// from the diagonal of T.
// 
// This routine returns the matrices X and/or Y of right and left
// eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
// input matrix.  If Q is the unitary factor that reduces a matrix A to
// Schur form T, then Q*X and Q*Y are the matrices of right and left
// eigenvectors of A.
//
// ### Details
//
// The algorithm used in this program is basically backward (forward)
// substitution, with scaling to make the the code robust against
// possible overflow.
//
// Each eigenvector is normalized so that the element of largest
// magnitude has magnitude 1; here the magnitude of a complex number
// (x,y) is taken to be |x| + |y|.
//
// ### Authors
//
// * Univ. of Tennessee
// * Univ. of California Berkeley
// * Univ. of Colorado Denver
// * NAG Ltd.
//
// Arguments
// howmny If "A", compute all right and/or left eigenvectors.
//        If "B", compute all right and/or left eigenvectors,
//        backtransformed using the matrices supplied in vr and/or vl.
//        If "S", compute selected right and/or left eigenvectors,
//        as indicated by the array select.
// select Length n array. If howmny = "S", then if select[j] is non-
//        zero, the eigenvector corresponding to the j-th eigenvalue
//        computed. Not referenced otherwise.
// n      Number of rows and columns of T.
// t      Pointer to the first element of T.
// ldt    Leading dimension of the array containing T, ldt >= n.
// vl     Pointer to the first element of the matrix of left
//        eigenvectorts. If NULL, the left eigenvectors are not
//        computed. If howmny = "S", then each eigenvector is stored
//        consecutively on the columns in the same order as the
//        eigenvalues, and the required columns are referenced.
//        Otherwise, all eigenvectors are computed, requiring n
//        columns. If howmny = "B", then on entry, vl should contain
//        an n-by-n matrix Q.
// ldvl   Leading dimension of the array containing vl, ldvl >= n.
// vr     Pointer to the first element of the matrix of right
//        eigenvectorts. If NULL, the right eigenvectors are not
//        computed. If howmny = "S", then each eigenvector is stored
//        consecutively on the columns in the same order as the
//        eigenvalues, and the required columns are referenced.
//        Otherwise, all eigenvectors are computed, requiring n
//        columns. If howmny = "B", then on entry, vr should contain
//        an n-by-n matrix Q.
// ldvr   Leading dimension of the array containing vr, ldvr >= n.
// work   Workspace of size 2*n.
// rwork  Workspace of size n.
//
template <typename T>
void Eigenvectors(
	const char *howmny, const int *select,
	size_t n, T *t, size_t ldt, T *vl, size_t ldvl, T *vr, size_t ldvr,
	T *work, typename Traits<T>::real_type *rwork
){
	typedef typename Traits<T>::real_type real_type;

	RNPAssert('A' == howmny[0] || 'B' == howmny[0] || 'S' == howmny[0]);
	RNPAssert(ldt >= n);
	RNPAssert(NULL == vl || ldvl >= n);
	RNPAssert(NULL == vr || ldvr >= n);
	
	const bool over  = ('B' == howmny[0]);
	const bool somev = ('S' == howmny[0]);

	// Set m to the number of columns required to store the selected
	// eigenvectors.
	size_t m = n;
	if(somev){
		m = 0;
		for(size_t j = 0; j < n; ++j){
			if(select[j]){
				++m;
			}
		}
	}

	if(0 == n){ return; }

	// Set the constants to control overflow.
	static const real_type unfl(Traits<real_type>::min()); // take sqrt on Cray
	static const real_type ulp(real_type(2) * Traits<real_type>::eps());
	static const real_type smlnum(unfl * (real_type(n) / ulp));

	// Store the diagonal elements of T in working array WORK.
	for(size_t i = 0; i < n; ++i){
		work[n+i] = t[i+i*ldt];
	}

	// Compute 1-norm of each column of strictly upper triangular
	// part of T to control overflow in triangular solver.
	rwork[0] = real_type(0);
	for(size_t j = 1; j < n; ++j){
		rwork[j] = BLAS::Asum(j, &t[0+j*ldt], 1);
	}

	if(NULL != vr){ // Compute right eigenvectors.
		size_t is = m-1;
		size_t ki = n; while(ki --> 0){
			if(somev && !select[ki]){ continue; }
			real_type smin = ulp * (Traits<T>::norm1(t[ki+ki*ldt]));
			if(smlnum > smin){ smin = smlnum; }

			work[0] = T(1);

			// Form right-hand side.
			for(size_t k = 0; k < ki; ++k){
				work[k] = -t[k+ki*ldt];
			}

			// Solve the triangular system:
			// (T(1:KI-1,1:KI-1) - T(KI,KI))*X = SCALE*WORK.
			for(size_t k = 0; k < ki; ++k){
				t[k + k * ldt] -= t[ki + ki * ldt];
				if(Traits<T>::norm1(t[k + k * ldt]) < smin){
					t[k+k*ldt] = smin;
				}
			}

			real_type scale(1);
			if(ki+1 > 1){
				Triangular::Solve(
					"U", "N", "N", "Y", ki+1 - 1, t, ldt,
					work, &scale, rwork
				);
				work[ki] = scale;
			}

			// Copy the vector x or Q*x to VR and normalize.

			if(!over){
				BLAS::Copy(ki+1, work, 1, &vr[0+is*ldvr], 1);

				const size_t ii = BLAS::MaximumIndex(ki+1, &vr[0+is*ldvr], 1);
				const real_type remax(real_type(1) / Traits<T>::norm1(vr[ii+is*ldvr]));
				BLAS::Scale(ki+1, remax, &vr[0+is*ldvr], 1);

				for(size_t k = ki+1; k < n; ++k){
					vr[k+is*ldvr] = T(0);
				}
			}else{
				if(ki+1 > 1){
					BLAS::MultMV("N", n, ki+1 - 1, T(1), vr, ldvr, work, 1, scale, &vr[0+ki*ldvr], 1);
				}
				const size_t ii = BLAS::MaximumIndex(n, &vr[0+ki*ldvr], 1);
				const real_type remax(real_type(1) / Traits<T>::norm1(vr[ii+ki*ldvr]));
				BLAS::Scale(n, remax, &vr[0+ki*ldvr], 1);
			}

			// Set back the original diagonal elements of T.
			for(size_t k = 0; k < ki; ++k){
				t[k+k*ldt] = work[k+n];
			}
			--is;
		}
	}

	if(NULL != vl){ // Compute left eigenvectors.
		size_t is = 0;
		for(size_t ki = 0; ki < n; ++ki){
			if(somev && !select[ki]){ continue; }
			real_type smin = ulp * Traits<T>::norm1(t[ki + ki * ldt]);
			if(smlnum > smin){ smin = smlnum; }
			work[n-1] = T(1);

			// Form right-hand side.
			for(size_t k = ki+1; k < n; ++k){
				work[k] = -Traits<T>::conj(t[ki+k*ldt]);
			}

			// Solve the triangular system:
			// (T(KI+1:N,KI+1:N) - T(KI,KI))**H * X = SCALE*WORK.

			for(size_t k = ki+1; k < n; ++k){
				t[k+k*ldt] -= t[ki+ki*ldt];
				if(Traits<T>::norm1(t[k+k*ldt]) < smin){
					t[k+k*ldt] = smin;
				}
			}

			real_type scale(1);
			if(ki+1 < n){
				Triangular::Solve(
					"U", "C", "N", "Y", n - ki-1, &t[ki + 1 + (ki + 1) * ldt], ldt,
					&work[ki + 1], &scale, rwork
				);
				work[ki] = scale;
			}

			// Copy the vector x or Q*x to VL and normalize.
			if(!over){
				BLAS::Copy(n - ki, &work[ki], 1, &vl[ki+is*ldvl], 1);

				const size_t ii = BLAS::MaximumIndex(n - ki, &vl[ki+is*ldvl], 1) + ki;
				const real_type remax(real_type(1) / Traits<T>::norm1(vl[ii+is*ldvl]));
				BLAS::Scale(n - ki, remax, &vl[ki+is*ldvl], 1);

				for(size_t k = 0; k < ki; ++k){
					vl[k+is*ldvl] = T(0);
				}
			}else{
				if(ki+1 < n){
					BLAS::MultMV("N", n, n - ki-1, T(1), &vl[0+(ki + 1) * ldvl], 
						ldvl, &work[ki + 1], 1, scale, &vl[0+ki*ldvl], 1);
				}
				const size_t ii = BLAS::MaximumIndex(n, &vl[0+ki*ldvl], 1);
				const real_type remax(real_type(1) / Traits<T>::norm1(vl[ii+ki*ldvl]));
				BLAS::Scale(n, remax, &vl[0+ki*ldvl], 1);
			}

			// Set back the original diagonal elements of T.
			for(size_t k = ki + 1; k < n; ++k){
				t[k+k*ldt] = work[k+n];
			}

			++is;
		}
	}
}


///////////////////////////////////////////////////////////////////////
// GeneralizedEigenvectors
// -----------------------
// Computes some or all of the right and/or left eigenvectors of
// a pair of complex matrices (S,P), where S and P are upper triangular.
// Matrix pairs of this type are produced by the generalized Schur
// factorization of a complex matrix pair (A,B):
//
//     A = Q*S*Z^H,  B = Q*P*Z^H
//
// as computed by Hessenberg::ReduceGeneralized and QZ iteration.
//
// The right eigenvector x and the left eigenvector y of (S,P)
// corresponding to an eigenvalue w are defined by:
//
//     S*x = w*P*x,  (y^H)*S = w*(y^H)*P,
//
// where y^H denotes the conjugate tranpose of y.
// The eigenvalues are not input to this routine, but are computed
// directly from the diagonal elements of S and P.
//
// This routine returns the matrices X and/or Y of right and left
// eigenvectors of (S,P), or the products Z*X and/or Q*Y,
// where Z and Q are input matrices.
// If Q and Z are the unitary factors from the generalized Schur
// factorization of a matrix pair (A,B), then Z*X and Q*Y
// are the matrices of right and left eigenvectors of (A,B).
//
// Arguments
// howmny If "A", compute all right and/or left eigenvectors.
//        If "B", compute all right and/or left eigenvectors,
//        backtransformed using the matrices supplied in vr and/or vl.
//        If "S", compute selected right and/or left eigenvectors,
//        as indicated by the array select.
// select Length n array. If howmny = "S", then if select[j] is non-
//        zero, the eigenvector corresponding to the j-th eigenvalue
//        computed. Not referenced otherwise.
// n      Number of rows and columns of S and P.
// s      Pointer to the first element of S.
// lds    Leading dimension of the array containing S, lds >= n.
// p      Pointer to the first element of P.
// ldp    Leading dimension of the array containing P, ldp >= n.
// vl     Pointer to the first element of the matrix of left
//        eigenvectorts. If NULL, the left eigenvectors are not
//        computed. If howmny = "S", then each eigenvector is stored
//        consecutively on the columns in the same order as the
//        eigenvalues, and the required columns are referenced.
//        Otherwise, all eigenvectors are computed, requiring n
//        columns. If howmny = "B", then on entry, vl should contain
//        an n-by-n matrix Q.
// ldvl   Leading dimension of the array containing vl, ldvl >= n.
// vr     Pointer to the first element of the matrix of right
//        eigenvectorts. If NULL, the right eigenvectors are not
//        computed. If howmny = "S", then each eigenvector is stored
//        consecutively on the columns in the same order as the
//        eigenvalues, and the required columns are referenced.
//        Otherwise, all eigenvectors are computed, requiring n
//        columns. If howmny = "B", then on entry, vr should contain
//        an n-by-n matrix Q.
// ldvr   Leading dimension of the array containing vr, ldvr >= n.
// work   Workspace of size 2*n.
// rwork  Workspace of size 2*n.
//
template <typename T>
void GeneralizedEigenvectors(
	const char *howmny, const int *select, size_t n,
	const T *s, size_t lds, const T *p, size_t ldp,
	T *vl, size_t ldvl, T *vr, size_t ldvr,
	T *work, typename Traits<T>::real_type *rwork
){
	typedef typename Traits<T>::real_type real_type;
	
	static const real_type rzero(0);
	static const real_type rone (1);
	static const T zero(rzero);
	static const T one ( rone);

	// Count the number of eigenvectors
	size_t im = n;
	if('S' == howmny[0]){
		im = 0;
		for(size_t j = 0; j < n; ++j){
			if(select[j]){
				++im;
			}
		}
	}
	
	// Check diagonal of B
	for(size_t j = 0; j < n; ++j){
		RNPAssert(rzero == Traits<T>::imag(p[j+j*ldp]));
	}
	
	if(0 == n){
		return;
	}
	
	// Machine Constants
	const real_type safmin(Traits<real_type>::min());
	const real_type ulp(real_type(2) * Traits<real_type>::eps());
	const real_type small(safmin * real_type(n) / ulp);
	const real_type big(real_type(1) / small);
	const real_type bignum(real_type(1) / (safmin * real_type(n)));

	// Compute the 1-norm of each column of the strictly upper triangular
	// part of A and B to check for possible overflow in the triangular
	// solver.
	real_type anorm = Traits<T>::norm1(s[0+0*lds]);
	real_type bnorm = Traits<T>::norm1(p[0+0*ldp]);
	rwork[0] = rzero;
	rwork[n] = rzero;
	
	for(size_t j = 1; j < n; ++j){
		rwork[  j] = rzero;
		rwork[n+j] = rzero;
		for(size_t i = 0; i < j; ++i){
			rwork[  j] += Traits<T>::norm1(s[i+j*lds]);
			rwork[n+j] += Traits<T>::norm1(p[i+j*ldp]);
		}
		real_type temp;
		temp = rwork[  j] + Traits<T>::norm1(s[j+j*lds]);
		if(temp > anorm){ anorm = temp; }
		temp = rwork[n+j] + Traits<T>::norm1(p[j+j*ldp]);
		if(temp > bnorm){ bnorm = temp; }
	}

	const real_type ascale = 1. / (anorm > safmin ? anorm : safmin);
	const real_type bscale = 1. / (bnorm > safmin ? bnorm : safmin);

	if(NULL != vl){ // Left eigenvectors
		size_t ieig = 0;
	
		// Main loop over eigenvalues
		for(size_t je = 0; je < n; ++je, ++ieig){
			if('S' == howmny[0] && !select[je]){ continue; }

			if(Traits<T>::norm1(s[je+je*lds]) <= safmin && Traits<real_type>::abs(Traits<T>::real(p[je+je*ldp])) <= safmin){
				// Singular matrix pencil -- return unit eigenvector
				for(size_t jr = 0; jr < n; ++jr){
					vl[jr+ieig*ldvl] = zero;
				}
				vl[ieig+ieig*ldvl] = one;
				continue;
			}

			// Non-singular eigenvalue:
			// Compute coefficients a and b in y^H ( a A - b B ) = 0
			const real_type as(Traits<T>::norm1(s[je+je*lds]) * ascale);
			const real_type ps(Traits<real_type>::abs(Traits<T>::real(p[je+je*ldp])) * bscale);
			const real_type ms = (as > ps ? as : ps);
			const real_type temp = rone / (ms > safmin ? ms : safmin);
			T salpha = temp*s[je+je*lds] * ascale;
			real_type sbeta = temp * Traits<T>::real(p[je+je*ldp]) * bscale;
			real_type acoeff = sbeta * ascale;
			T bcoeff = bscale * salpha;

			// Scale to avoid underflow
			const bool lsa = abs(sbeta) >= safmin && abs(acoeff) < small;
			const bool lsb = Traits<T>::norm1(salpha) >= safmin && Traits<T>::norm1(bcoeff) < small;

			real_type scale(1);
			if(lsa){
				scale = small / abs(sbeta) * (anorm < big ? anorm : big);
			}
			if(lsb){
				real_type scale2(small / Traits<T>::norm1(salpha) * (bnorm < big ? bnorm : big));
				if(scale2 > scale){ scale = scale2; }
			}
			if(lsa || lsb){
				real_type aa(Traits<T>::abs(acoeff));
				if(rone > aa){ aa = rone; }
				real_type bb(Traits<T>::norm1(bcoeff));
				real_type scale2(rone / (safmin * (aa > bb ? aa : bb)));
				if(scale2 < scale){ scale = scale2; }
				if(lsa){
					acoeff = ascale * (scale * sbeta);
				}else{
					acoeff = scale * acoeff;
				}
				if(lsb){
					bcoeff = bscale * (scale * salpha);
				}else{
					bcoeff *= scale;
				}
			}

			const real_type acoefa = Traits<T>::abs(acoeff);
			const real_type bcoefa = Traits<T>::norm1(bcoeff);
			real_type xmax(1);
			for(size_t jr = 0; jr < n; ++jr){
				work[jr] = zero;
			}
			work[je] = one;
			real_type ua = ulp * acoefa * anorm;
			real_type ub = ulp * bcoefa * bnorm;
			real_type um = (ua > ub ? ua : ub);
			const real_type dmin = (um > safmin ? um : safmin);

			// Triangular solve of  (a A - b B)^H y = 0
			// (rowwise in (a A - b B)^H , or columnwise in a A - b B)


			for(size_t j = je+1; j < n; ++j){
				// Compute
				//       j-1
				// SUM = sum  conj( a*S(k,j) - b*P(k,j) )*x(k)
				//       k=je
				// (Scale if necessary)

				const real_type temp = 1. / xmax;
				if(acoefa * rwork[j] + bcoefa * rwork[n+j] > bignum * temp) {
					for(size_t jr = je; jr < j; ++jr){
						work[jr] *= temp;
					}
					xmax = rone;
				}
				T suma(0), sumb(0);

				for(size_t jr = je; jr < j; ++jr){
					suma += Traits<T>::conj(s[jr+j*lds]) * work[jr];
					sumb += Traits<T>::conj(p[jr+j*ldp]) * work[jr];
				}
				T sum = acoeff*suma - Traits<T>::conj(bcoeff)*sumb;


				// Form x(j) = - SUM / conj( a*S(j,j) - b*P(j,j) )
				// with scaling and perturbation of the denominator
				T d = Traits<T>::conj(acoeff*s[j+j*lds] - bcoeff*p[j+j*ldp]);
				if(Traits<T>::norm1(d) <= dmin){
					d = dmin;
				}

				real_type abs1d = Traits<T>::norm1(d);
				if(abs1d < rone) {
					real_type abs1sum = Traits<T>::norm1(sum);
					if(abs1sum >= bignum * abs1d) {
						const real_type temp = rone / abs1sum;
						for(size_t jr = je; jr < j; ++jr){
							work[jr] *= temp;
						}
						xmax *= temp;
						sum *= temp;
					}
				}
				work[j] = -sum/d;

				real_type nwj(Traits<T>::norm1(work[j]));
				if(nwj > xmax){ xmax = nwj; }
			}

			// Back transform eigenvector if HOWMNY='B'.
			size_t isrc = 0, ibeg = je;
			if('B' == howmny[0]){
				BLAS::MultMV("N", n, n-je, rone, &vl[0+je*ldvl], ldvl, &work[je], 1, rzero, &work[n], 1);
				isrc = n;
				ibeg = 0;
			}

			// Copy and scale eigenvector into column of VL
			xmax = rzero;
			for(size_t jr = ibeg; jr < n; ++jr){
				real_type nwj(Traits<T>::norm1(work[isrc+jr]));
				if(nwj > xmax){ xmax = nwj; }
			}

			if(xmax > safmin){
				const real_type temp = rone / xmax;
				for(size_t jr = ibeg; jr < n; ++jr){
					vl[jr+ieig*ldvl] = temp * work[isrc+jr];
				}
			}else{
				ibeg = n;
			}

			for(size_t jr = 0; jr < ibeg; ++jr){
				vl[jr+ieig*ldvl] = zero;
			}
		}
	}

	if(NULL != vr){ // Right eigenvectors
		size_t ieig = im;

		// Main loop over eigenvalues
		size_t je = n;
		while(je --> 0){
			if('S' == howmny[0] && !select[je]){ continue; }
			--ieig;

			if (Traits<T>::norm1(s[je+je*lds]) <= safmin && Traits<T>::abs(Traits<T>::real(p[je+je*ldp])) <= safmin) {
				// Singular matrix pencil -- return unit eigenvector
				for(size_t jr = 0; jr < n; ++jr){
					vr[jr+ieig*ldvr] = zero;
				}
				vr[ieig+ieig*ldvr] = one;
				continue;
			}

			// Non-singular eigenvalue:
			// Compute coefficients a and b in ( a A - b B ) x  = 0
			const real_type as(Traits<T>::norm1(s[je+je*lds]) * ascale);
			const real_type ps(Traits<real_type>::abs(Traits<T>::real(p[je+je*ldp])) * bscale);
			const real_type ms = (as > ps ? as : ps);
			const real_type temp = rone / (ms > safmin ? ms : safmin);
			T salpha = temp*s[je+je*lds] * ascale;
			real_type sbeta = temp * Traits<T>::real(p[je+je*ldp]) * bscale;
			real_type acoeff = sbeta * ascale;
			T bcoeff = bscale * salpha;

			// Scale to avoid underflow
			const bool lsa = abs(sbeta) >= safmin && abs(acoeff) < small;
			const bool lsb = Traits<T>::norm1(salpha) >= safmin && Traits<T>::norm1(bcoeff) < small;

			real_type scale(1);
			if(lsa){
				scale = small / abs(sbeta) * (anorm < big ? anorm : big);
			}
			if(lsb){
				real_type scale2(small / Traits<T>::norm1(salpha) * (bnorm < big ? bnorm : big));
				if(scale2 > scale){ scale = scale2; }
			}
			if(lsa || lsb){
				real_type aa(Traits<T>::abs(acoeff));
				if(rone > aa){ aa = rone; }
				real_type bb(Traits<T>::norm1(bcoeff));
				real_type scale2(rone / (safmin * (aa > bb ? aa : bb)));
				if(scale2 < scale){ scale = scale2; }
				if(lsa){
					acoeff = ascale * (scale * sbeta);
				}else{
					acoeff = scale * acoeff;
				}
				if(lsb){
					bcoeff = bscale * (scale * salpha);
				}else{
					bcoeff *= scale;
				}
			}

			const real_type acoefa = abs(acoeff);
			const real_type bcoefa = Traits<T>::norm1(bcoeff);
			real_type xmax(1);
			for(size_t jr = 0; jr < n; ++jr){
				work[jr] = zero;
			}
			work[je] = one;
			real_type ua = ulp * acoefa * anorm;
			real_type ub = ulp * bcoefa * bnorm;
			real_type um = (ua > ub ? ua : ub);
			const real_type dmin = (um > safmin ? um : safmin);
			

			// Triangular solve of  (a A - b B) x = 0  (columnwise)
			// work[0:j] contains sums w, work[j+1:je+1] contains x
			for(size_t jr = 0; jr < je; ++jr){
				work[jr] = acoeff*s[jr+je*lds] - bcoeff*p[jr+je*ldp];
			}
			work[je] = one;

			size_t j = je;
			while(j --> 0){
				// Form x(j) := - w(j) / d
				// with scaling and perturbation of the denominator
				T d = acoeff*s[j+j*lds] - bcoeff*p[j+j*ldp];
				if(Traits<T>::norm1(d) <= dmin){
					d = dmin;
				}

				double abs1d = Traits<T>::norm1(d);
				if(abs1d < rone){
					double abs1workj = Traits<T>::norm1(work[j]);
					if(abs1workj >= bignum * abs1d){
						const real_type temp = rone / abs1workj;
						for(size_t jr = 0; jr <= je; ++jr){
							work[jr] *= temp;
						}
					}
				}
				work[j] = -work[j]/d;

				if(j > 0){
					// w += x(j)*(a S(*,j) - b P(*,j) ) with scaling
					double abs1workj = Traits<T>::norm1(work[j]);
					if(abs1workj > rone){
						const real_type temp = rone / abs1workj;
						if(acoefa * rwork[j] + bcoefa * rwork[n + j] >= bignum * temp){
							for(size_t jr = 0; jr <= je; ++jr){
								work[jr] *= temp;
							}
						}
					}

					T ca = acoeff*work[j];
					T cb = bcoeff*work[j];
					for(size_t jr = 0; jr < j; ++jr){
						work[jr] += ca*s[jr+j*lds] - cb*p[jr+j*ldp];
					}
				}
			}

			// Back transform eigenvector if HOWMNY='B'.
			size_t isrc = 0;
			size_t iend = je+1;
			if('B' == howmny[0]){
				BLAS::MultMV("N", n, je+1, rone, vr, ldvr, work, 1, rzero, &work[n], 1);
				isrc = n;
				iend = n;
			}

			// Copy and scale eigenvector into column of VR
			xmax = rzero;
			for(size_t jr = 0; jr < iend; ++jr){
				real_type nwj(Traits<T>::norm1(work[isrc+jr]));
				if(nwj > xmax){ xmax = nwj; }
			}

			if(xmax > safmin){
				const real_type temp = rone / xmax;
				for(size_t jr = 0; jr < iend; ++jr) {
					vr[jr+ieig*ldvr] = temp * work[isrc+jr];
				}
			}else{
				iend = 0;
			}

			for(size_t jr = iend; jr < n; ++jr){
				vr[jr+ieig*ldvr] = zero;
			}
		}
	}
}



template <typename T>
void ExchangeDiagonal(
	size_t n, T *t, size_t ldt, T *q, size_t ldq, size_t ifst, size_t ilst
){

    // Purpose
    // =======
    // 
    // ZTREXC reorders the Schur factorization of a complex matrix
    // A = Q*T*Q^H, so that the diagonal element of T with row index IFST
    // is moved to row ILST.
    // 
    // The Schur form T is reordered by a unitary similarity transformation
    // Z^H*T*Z, and optionally the matrix Q of Schur vectors is updated by
    // postmultplying it with Z.
    // 
    // Arguments
    // =========
    // 
    // COMPQ   (input) char*1
    // = 'V':  update the matrix Q of Schur vectors;
    // = 'N':  do not update Q.
    // 
    // N       (input) int
    // The order of the matrix T. N >= 0.
    // 
    // T       (input/output) std::complex<double> array, dimension (LDT,N)
    // On entry, the upper triangular matrix T.
    // On exit, the reordered upper triangular matrix.
    // 
    // LDT     (input) int
    // The leading dimension of the array T. LDT >= max(1,N).
    // 
    // Q       (input/output) std::complex<double> array, dimension (LDQ,N)
    // On entry, if COMPQ = 'V', the matrix Q of Schur vectors.
    // On exit, if COMPQ = 'V', Q has been postmultiplied by the
    // unitary transformation matrix Z which reorders T.
    // If COMPQ = 'N', Q is not referenced.
    // 
    // LDQ     (input) int
    // The leading dimension of the array Q.  LDQ >= max(1,N).
    // 
    // IFST    (input) int
    // ILST    (input) int
    // Specify the reordering of the diagonal elements of T:
    // The element with row index IFST is moved to row ILST by a
    // sequence of transpositions between adjacent elements.
    // 1 <= IFST <= N; 1 <= ILST <= N.
	if(n <= 1 || ifst == ilst){ return; }

	if(ifst < ilst){ // Move the IFST-th diagonal element forward down the diagonal.
		for(size_t k = ifst; k < ilst; ++k){
			// Interchange the k-th and (k+1)-th diagonal elements.
			T t11 = t[k+k*ldt];
			T t22 = t[(k+1)+(k+1)*ldt];
			
			// Determine the transformation to perform the interchange.
			double cs;
			T sn, temp;
			Rotation::Generate(t[k+(k+1)*ldt], t22-t11, &cs, &sn, &temp);

			// Apply transformation to the matrix T.
			if(k+2 < n){
				Rotation::Apply(n-k-2, &t[k+(k+2)*ldt], ldt, &t[(k+1)+(k+2)*ldt], ldt, cs, sn);
			}
			Rotation::Apply(k, &t[0+k*ldt], 1, &t[0+(k+1)*ldt], 1, cs, std::conj(sn));

			t[k+k*ldt] = t22;
			t[(k+1)+(k+1)*ldt] = t11;
			
			if(NULL != q){ // Accumulate transformation in the matrix Q.
				Rotation::Apply(n, &q[0+k*ldq], 1, &q[0+(k+1)*ldq], 1, cs, std::conj(sn));
			}
		}
	}else{ // Move the IFST-th diagonal element backward up the diagonal.
		size_t k = ifst;
		while(k --> ilst){
			// Interchange the k-th and (k+1)-th diagonal elements.
			T t11 = t[k+k*ldt];
			T t22 = t[(k+1)+(k+1)*ldt];
			
			// Determine the transformation to perform the interchange.
			double cs;
			T sn, temp;
			Rotation::Generate(t[k+(k+1)*ldt], t22-t11, &cs, &sn, &temp);

			// Apply transformation to the matrix T.
			if(k+2 < n){
				Rotation::Apply(n-k-2, &t[k+(k+2)*ldt], ldt, &t[(k+1)+(k+2)*ldt], ldt, cs, sn);
			}
			Rotation::Apply(k, &t[0+k*ldt], 1, &t[0+(k+1)*ldt], 1, cs, std::conj(sn));

			t[k+k*ldt] = t22;
			t[(k+1)+(k+1)*ldt] = t11;
			
			if(NULL != q){ // Accumulate transformation in the matrix Q.
				Rotation::Apply(n, &q[0+k*ldq], 1, &q[0+(k+1)*ldq], 1, cs, std::conj(sn));
			}
		}
	}
}




} // namespace Triangular
} // namespace LA
} // namespace RNP

#endif // RNP_TRIANGULAR_HPP_INCLUDED
