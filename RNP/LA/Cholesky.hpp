#ifndef RNP_CHOLESKY_HPP_INCLUDED
#define RNP_CHOLESKY_HPP_INCLUDED

#include <RNP/BLAS.hpp>

namespace RNP{
namespace LA{
namespace Cholesky{

template <typename T>
struct Tuning{
	static inline size_t factor_block_size(const char *uplo, size_t n){ return 64; }
};

template <typename T>
int Factor_unblocked(const char *uplo, size_t n, T *a, size_t lda){
	typedef typename Traits<T>::real_type real_type;
	if(0 == n){ return 0; }
	if('U' == uplo[0]){ // Compute the Cholesky factorization A = U^H * U.
		for(size_t j = 0; j < n; ++j){
			// Compute U[j,j] and test for non-positive-definiteness.
			real_type Ajj = Traits<T>::real(a[j+j*lda]) - Traits<T>::real(BLAS::ConjugateDot(j, &a[0+j*lda], 1, &a[0+j*lda], 1));
			if(!(Ajj > real_type(0))){ // handles NaN case as well
				a[j+j*lda] = Ajj;
				return j+1;
			}
			Ajj = sqrt(Ajj);
			a[j+j*lda] = Ajj;
			// Compute elements j+1:n of row j.
			if(j+1 < n){
				BLAS::Conjugate(j, &a[0+j*lda], 1);
				BLAS::MultMV("T", j, n-j-1, T(-1), &a[0+(j+1)*lda], lda, &a[0+j*lda], 1, T(1), &a[j+(j+1)*lda], lda);
				BLAS::Conjugate(j, &a[0+j*lda], 1);
				BLAS::Scale(n-j-1, real_type(1) / Ajj, &a[j+(j+1)*lda], lda);
			}
		}
	}else{ // Compute the Cholesky factorization A = L * L^H.
		for(size_t j = 0; j < n; ++j){
			// Compute L[j,j] and test for non-positive-definiteness.
			real_type Ajj = Traits<T>::real(a[j+j*lda]) - Traits<T>::real(BLAS::ConjugateDot(j, &a[j+0*lda], lda, &a[j+0*lda], lda));
			if(!(Ajj > real_type(0))){ // handles NaN case as well
				a[j+j*lda] = Ajj;
				return j+1;
			}
			Ajj = sqrt(Ajj);
			a[j+j*lda] = Ajj;
			// Compute elements j+1:n of column j.
			if(j+1 < n){
				BLAS::Conjugate(j, &a[j+0*lda], lda);
				BLAS::MultMV("N", n-j-1, j, T(-1), &a[j+1+0*lda], lda, &a[j+0*lda], lda, T(1), &a[j+1+j*lda], 1);
				BLAS::Conjugate(j, &a[j+0*lda], lda);
				BLAS::Scale(n-j-1, real_type(1) / Ajj, &a[j+1+j*lda], 1);
			}
		}
	}
	return 0;
}

template <typename T>
int Factor(const char *uplo, size_t n, T *a, size_t lda){
	typedef typename Traits<T>::real_type real_type;
	if(0 == n){ return 0; }
	const size_t nb = Tuning<T>::factor_block_size(uplo, n);
	if(nb <= 1 || nb >= n){
		return Factor_unblocked(uplo, n, a, lda);
	}else{
		if('U' == uplo[0]){
			for(size_t j = 0; j < n; j += nb){
				// Update and factorize the current diagonal block
				// and test for non-positive-definiteness.
				const size_t jb = (nb+j < n ? nb : n-j);
				BLAS::HermRankKUpdate("U", "C", jb, j, real_type(-1), &a[0+j*lda], lda, real_type(1), &a[j+j*lda], lda);
				int info = Factor_unblocked("U", jb, &a[j+j*lda], lda);
				if(0 != info){ return info+j; }
				if(j+jb < n){ // Compute the current block row
					BLAS::MultMM("C", "N", nb, n-j-jb, j, T(-1), &a[0+j*lda], lda, &a[0+(j+jb)*lda], lda, T(1), &a[j+(j+jb)*lda], lda);
					BLAS::SolveTrM("L", "U", "C", "N", jb, n-j-jb, T(1), &a[j+j*lda], lda, &a[j+(j+jb)*lda], lda);
				}
			}
		}else{
			for(size_t j = 0; j < n; j += nb){
				const size_t jb = (nb+j < n ? nb : n-j);
				BLAS::HermRankKUpdate("L", "N", jb, j, real_type(-1), &a[j+0*lda], lda, real_type(1), &a[j+j*lda], lda);
				int info = Factor_unblocked("L", jb, &a[j+j*lda], lda);
				if(0 != info){ return info+j; }
				if(j+jb < n){ // Compute the current block column
					BLAS::MultMM("N", "C", n-j-jb, jb, j, T(-1), &a[j+jb+0*lda], lda, &a[j+0*lda], lda, T(1), &a[j+jb+j*lda], lda);
					BLAS::SolveTrM("R", "L", "C", "N", n-j-jb, jb, T(1), &a[j+j*lda], lda, &a[j+jb+j*lda], lda);
				}
			}
		}
	}
	return 0;
}

} // namespace Cholesky
} // namespace LA
} // namespace RNP

#endif // RNP_CHOLESKY_HPP_INCLUDED
