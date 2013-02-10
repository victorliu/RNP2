#ifndef RNP_LU_HPP_INCLUDED
#define RNP_LU_HPP_INCLUDED

#include <cstddef>
#include <RNP/BLAS.hpp>
#include "Permutation.hpp"

namespace RNP{
namespace LA{
namespace LU{

// Specialize this class to tune the block size.
template <typename T>
struct Tuning{
	static inline size_t factor_block_size(size_t m, size_t n){ return 64; }
	static inline size_t invert_block_size_opt(size_t n){ return 64; }
	static inline size_t invert_block_size_min(size_t n){ return 64; }
	static inline size_t refine_max_iters(){ return 5; }
};

namespace Util{

// The Norm1Estimator is used in the iterative refinement process
// for estimating the forward error.
template <typename T>
struct Norm1EstimatorData{
	const T *af;
	size_t ldaf;
	size_t *ipiv;
	const char *transt, *transn;
	typename Traits<T>::real_type *diagw;
};

template <typename T>
void Norm1Estimator(const char *trans, size_t n, T *x, void *data_){
	Norm1EstimatorData<T> *data = reinterpret_cast<Norm1EstimatorData<T>*>(data_);
	if('N' == trans[0]){
		PLUSolve(data->transt, n, 1, data->af, data->ldav, data->ipiv, x);
		for(size_t i = 0; i < n; ++i){
			x[i] *= data->diagw[i];
		}
	}else{
		for(size_t i = 0; i < n; ++i){
			x[i] *= data->diagw[i];
		}
		PLUSolve(data->transn, n, 1, data->af, data->ldav, data->ipiv, x);
	}
}

} // namespace Util

template <typename T>
int Factor_unblocked(size_t m, size_t n, T *a, size_t lda, size_t *pivots){
	int info = 0;
	size_t min_dim = (m < n ? m : n);
	for(size_t j = 0; j < min_dim; ++j){
		size_t jp = j + BLAS::MaximumIndex(m-j, &a[j+j*lda], 1);
		pivots[j] = jp;
		if(T(0) != a[jp+j*lda]){
			if(jp != j){
				BLAS::Swap(n, &a[j+0*lda], lda, &a[jp+0*lda], lda);
			}
			if(j < m){
				BLAS::Scale(m-j-1, T(1)/a[j+j*lda], &a[j+1+j*lda], 1); // possible overflow when inverting A(j,j)
			}
		}else{
			info = j;
		}
		if(j < min_dim){
			BLAS::Rank1Update(m-j-1, n-j-1, T(-1), &a[j+1+j*lda], 1, &a[j+(j+1)*lda], lda, &a[j+1+(j+1)*lda], lda);
		}
	}
	if(0 != info){ return info+1; }
	return 0;
}

template <typename T>
int Factor(size_t m, size_t n, T *a, size_t lda, size_t *pivots){
	//return PLUFactor_unblocked(m, n, a, lda, pivots);
	
	static const size_t nb = LU::Tuning<T>::factor_block_size(m, n);
	const size_t min_dim = (m < n ? m : n);
	if(min_dim <= nb || nb <= 1){
		return Factor_unblocked(m, n, a, lda, pivots);
	}
	// Use blocked code
	int info = 0;
	for(size_t j = 0; j < min_dim; j += nb){
		// Size of the block
		const size_t jb = (min_dim < nb+j ? min_dim-j : nb);
		// Factor the diagonal and subdiagonal blocks
		int iinfo = Factor_unblocked(m-j, jb, &a[j+j*lda], lda, &pivots[j]);
		if(0 == info && iinfo > 0){ // Adjust info
			info = iinfo + j;
		}
		const size_t ilim = (m < j+jb ? m : j+jb);
		for(size_t i = j; i < ilim; ++i){ // Adjust pivots
			pivots[i] += j;
		}
		// Apply row swaps to first j columns and rows j to j+jb
		ApplyPermutations("L", "F", jb, j, a, lda, pivots, j);
		if(j+jb < n){
			// Apply row swaps to columns after j+jb
			ApplyPermutations("L", "F", jb, n-j-jb, &a[0+(j+jb)*lda], lda, pivots, j);
			// Make block row of U
			BLAS::SolveTrM("L","L","N","U", jb, n-j-jb, T(1), &a[j+j*lda], lda, &a[j+(j+jb)*lda], lda);
			if(j+jb < m){
				// Update trailing submatrix
				BLAS::MultMM("N","N", m-j-jb, n-j-jb, jb, T(-1), &a[j+jb+j*lda], lda, &a[j+(j+jb)*lda], lda, T(1), &a[j+jb+(j+jb)*lda], lda);
			}
		}
	}
	if(0 != info){ return info+1; }
	return 0;
}

template <typename T>
void Solve(const char *trans, size_t n, size_t nRHS, const T *a, size_t lda, size_t *ipiv, T *b, size_t ldb){
	if(0 == n || nRHS == 0){ return; }
	
	if('N' == trans[0]){
		for(size_t i = 0; i < n; ++i){
			if(ipiv[i] != i){
				BLAS::Swap(nRHS, &b[i+0*ldb], ldb, &b[ipiv[i]+0*ldb], ldb);
			}
		}
		BLAS::SolveTrM("L","L","N","U", n, nRHS, T(1), a, lda, b, ldb);
		BLAS::SolveTrM("L","U","N","N", n, nRHS, T(1), a, lda, b, ldb);
	}else{
		if('T' == trans[0]){
			BLAS::SolveTrM("L","U","T","N", n, nRHS, T(1), a, lda, b, ldb);
			BLAS::SolveTrM("L","L","T","U", n, nRHS, T(1), a, lda, b, ldb);
		}else if('C' == trans[0]){
			BLAS::SolveTrM("L","U","C","N", n, nRHS, T(1), a, lda, b, ldb);
			BLAS::SolveTrM("L","L","C","U", n, nRHS, T(1), a, lda, b, ldb);
		}
		size_t i = n;
		while(i --> 0){
			if(ipiv[i] != i){
				BLAS::Swap(nRHS, &b[i+0*ldb], ldb, &b[ipiv[i]+0*ldb], ldb);
			}
		}
	}
}

template <typename T>
int Invert(size_t n, T *a, size_t lda, const size_t *ipiv, size_t *lwork, T *work){
	RNPAssert(NULL != lwork);
	RNPAssert((0 == *lwork || NULL == work) || *lwork >= n);
	if(0 == n){ return 0; }
	size_t nb = LU::Tuning<T>::invert_block_size_opt(n);
	if(NULL == work || 0 == *lwork){
		*lwork = nb * n;
		return 0;
	}
	int info = Triangular::Invert("U","N", n, a, lda);
	if(0 != info){ return info; }
	
	size_t nbmin = 2;
	size_t iws = n;
	const size_t ldwork = n;
	if(nb > 1 && nb < n){
		iws = ldwork * nb;
		if(*lwork < iws){
			nb = *lwork / ldwork;
			nbmin = LU::Tuning<T>::invert_block_size_min(n);
		}
	}else{
		iws = n;
	}
	// Solve the equation inv(A)*L = inv(U) for inv(A).
	if(nb < nbmin || nb >= n){ // unblocked
		// We build the columns of inv(A) from right to left
		size_t j = n;
		while(j --> 0){
			// Copy current column of L to work, and replace with zeros
			for(size_t i = j+1; i < n; ++i){
				work[i] = a[i+j*lda];
				a[i+j*lda] = T(0);
			}
			// Compute current column
			if(j+1 < n){
				BLAS::MultMV("N", n, n-1-j, T(-1), &a[0+(j+1)*lda], lda, &work[j+1], 1, T(1), &a[0+j*lda], 1);
			}
		}
	}else{ // blocked
		size_t j = (n / nb) * nb; // round n down to a multiple of nb
		do{
			// size of current block
			const size_t jb = (n < nb+j ? n-j : nb);
			// Copy current block column of L to work and replace with zeros
			for(size_t jj = 0; jj < jb; ++jj){
				for(size_t ii = jj+1; ii < n; ++ii){
					work[ii+jj*ldwork] = a[ii+jj*lda];
					a[ii+jj*lda] = T(0);
				}
			}
			// Compute current block column
			if(j+jb < n){
				BLAS::MultMM("N","N", n, jb, n-j-jb,
					T(-1), &a[0+(j+jb)*lda], lda, &work[j+jb], ldwork,
					T(1), &a[0+j*lda], lda
				);
			}
			BLAS::SolveTrM("R","L","N","U", n, jb, T(1), &work[j], ldwork, &a[0+j*lda], lda);
			if(0 == j){ break; }
			j -= nb;
		}while(1);
	}
	// Apply column interchanges
	LA::ApplyPermutations("R", "B", n, n, a, lda, ipiv);
	*lwork = iws;
	return 0;
}

template <typename T>
void RefineSolution(
	const char *trans, size_t n, size_t nRHS, const T *a, size_t lda,
	const T *af, size_t ldaf, const size_t *ipiv,
	const T *b, size_t ldb, T *x, size_t ldx,
	typename Traits<T>::real_type *ferr, // length nRHS
	typename Traits<T>::real_type *berr, // length nRHS
	T *work, // length 2*n
	typename Traits<T>::real_type *rwork
){
	typedef typename Traits<T>::real_type real_type;
	
	if(0 == n || 0 == nRHS){
		if(NULL != ferr){
			for(size_t i = 0; i < nRHS; ++i){
				ferr[i] = real_type(0);
			}
		}
		if(NULL != berr){
			for(size_t i = 0; i < nRHS; ++i){
				berr[i] = real_type(0);
			}
		}
		return;
	}
	const char *transn = ('N' == trans[0] ? "N" : "C");
	const char *transt = ('N' == trans[0] ? "C" : "N");
	// nz = maximum number of nonzero elements in each row of A, plus 1
	size_t nz = n+1;
	const real_type safe1 = nz * Traits<real_type>::min();
	const real_type safe2 = safe1 / Traits<real_type>::eps();
	
	for(size_t j = 0; j < nRHS; ++j){
		size_t count = 1;
		real_type lstres = real_type(3);
		do{ // Loop until stopping criterion is satisfied.
			// Compute residual R = B - op(A) * X,
			// where op(A) is given by trans.
			BLAS::Copy(n, &b[0+j*ldb], 1, work, 1);
			BLAS::MultMV(trans, n, n, T(-1), a, lda, &x[0+j*ldx], 1, T(1), work, 1);
			// Compute componentwise relative backward error from formula
			//   max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )
			// where abs(Z) is the componentwise absolute value of the matrix
			// or vector Z.  If the i-th component of the denominator is less
			// than safe2, then safe1 is added to the i-th components of the
			// numerator and denominator before dividing.
			for(size_t i = 0; i < n; ++i){
				rwork[i] = Traits<T>::norm1(b[i+j*ldb]);
			}
			// Compute abs(op(A))*abs(X) + abs(B).
			if('N' == trans[0]){
				for(size_t k = 0; k < n; ++k){
					real_type xk = Traits<T>::norm1(x[k+j*ldx]);
					for(size_t i = 0; i < n; ++i){
						rwork[i] += Traits<T>::norm1(a[i+k*lda]) * xk;
					}
				}
			}else{
				for(size_t k = 0; k < n; ++k){
					real_type s(0);
					for(size_t i = 0; i < n; ++i){
						s += Traits<T>::norm1(a[i+k*lda]) * Traits<T>::norm1(x[i+j*ldx]);;
					}
					rwork[k] += s;
				}
			}
			real_type s(0);
			for(size_t i = 0; i < n; ++i){
				if(rwork[i] > safe2){
					real_type r = Traits<T>::norm1(work[i]) / rwork[i];
					if(r > s){ s = r; }
				}else{
					real_type r = (Traits<T>::norm1(work[i]) + safe1) / (rwork[i] + safe1);
					if(r > s){ s = r; }
				}
			}
			if(NULL != berr){ berr[j] = s; }
			// Test stopping criterion. Continue iterating if
			//   1) The residual BERR(J) is larger than machine epsilon, and
			//   2) BERR(J) decreased by at least a factor of 2 during the
			//      last iteration, and
			//   3) At most ITMAX iterations tried.
			if(
				s > Traits<real_type>::eps() &&
				real_type(2)*s <= lstres &&
				count < LU::Tuning<T>::refine_max_iters()
			){
				// Update solution and try again
				PLUSolve(trans, n, 1, af, ldaf, ipiv, work, n);
				BLAS::Axpy(n, T(1), work, 1, &x[0+j*ldx], 1);
				lstres = s;
				++count;
				continue;
			}else{ break; }
		}while(1);
		// Bound error from formula
		// 
		// norm(X - XTRUE) / norm(X) .le. FERR =
		// norm( abs(inv(op(A)))*
		//    ( abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) ))) / norm(X)
		//
		// where
		//   norm(Z) is the magnitude of the largest component of Z
		//   inv(op(A)) is the inverse of op(A)
		//   abs(Z) is the componentwise absolute value of the matrix or
		//      vector Z
		//   NZ is the maximum number of nonzeros in any row of A, plus 1
		//   EPS is machine epsilon
		//
		// The i-th component of abs(R)+NZ*EPS*(abs(op(A))*abs(X)+abs(B))
		// is incremented by SAFE1 if the i-th component of
		// abs(op(A))*abs(X) + abs(B) is less than SAFE2.
		//
		// Use ZLACN2 to estimate the infinity-norm of the matrix
		//    inv(op(A)) // diag(W),
		// where W = abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) )))
		for(size_t i = 0; i < n; ++i){
			if(rwork[i] > safe2){
				rwork[i] = Traits<T>::norm1(work[i]) + nz*Traits<real_type>::eps()*rwork[i];
			}else{
				rwork[i] = Traits<T>::norm1(work[i]) + nz*Traits<real_type>::eps()*rwork[i] * safe1;
			}
		}
		
		Util::Norm1EstimatorData<T> estdata;
		estdata.af = af;
		estdata.ldaf = ldaf;
		estdata.ipiv = ipiv;
		estdata.diagw = rwork;
		estdata.transn = transn;
		estdata.transt = transt;
		real_type ferrj = MatrixNorm1Estimate(n, &Util::Norm1Estimator, &estdata, work);
		
		// Normalize error
		lstres = real_type(0);
		for(size_t i = 0; i < n; ++i){
			real_type nxij = Traits<T>::norm1(x[i+j*ldx]);
			if(nxij > lstres){ lstres = nxij; }
		}
		if(NULL != ferr){
			if(0 != lstres){
				ferr[j] = ferrj / lstres;
			}else{
				ferr[j] = ferrj;
			}
		}
	}
}

} // namespace LU
} // namespace LA
} // namespace RNP

#endif // RNP_LU_HPP_INCLUDED
