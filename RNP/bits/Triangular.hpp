#ifndef RNP_TRIANGULAR_HPP_INCLUDED
#define RNP_TRIANGULAR_HPP_INCLUDED

#include <iostream>
#include <cstddef>
#include <RNP/BLAS.hpp>
#include <RNP/Debug.hpp>

namespace RNP{
namespace LA{
namespace Triangular{

// Specialize this class to tune the block size.
template <typename T>
struct Tuning{
	static inline size_t invert_block_size(const char *uplo, const char *diag, size_t n){ return 64; }
};

template <typename T> // _trti2
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

template <typename T> // _trtri
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

template <typename T>
void Copy(const char *uplo, const char *diag, size_t m, size_t n, const T* RNP_RESTRICT src, size_t ldsrc, T* RNP_RESTRICT dst, size_t lddst){
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
