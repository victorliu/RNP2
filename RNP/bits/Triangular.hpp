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

} // namespace Triangular
} // namespace LA
} // namespace RNP

#endif // RNP_TRIANGULAR_HPP_INCLUDED
