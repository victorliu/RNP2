#ifndef RNP_LU_OBJ_HPP_INCLUDED
#define RNP_LU_OBJ_HPP_INCLUDED

#include <RNP/Types.hpp>
#include <RNP/Debug.hpp>
#include <RNP/BLAS.hpp>
#include "LU.hpp"

namespace RNP{
namespace LA{

template <typename T, typename Allocator = std::allocator<T> >
class PLUFactors : public Base<T, Allocator>{
	T *a;
	size_t m, n, lda;
	size_t mindim;
	size_t *ipiv;
	int err;
public:
	typedef Base<T, Allocator> base_type;
	typedef typename Traits<T>::real_type real_type;

	PLUFactors(const Matrix<T> &A):a(A.ptr()),m(A.rows()),n(A.cols()),lda(A.ldim()),ipiv(NULL){
		A.Invalidate(); A.Release();
		mindim = (m < n ? m : n);
		ipiv = this->template Alloc<size_t>(mindim);
		Compute();
	}
	~PLUFactors(){
		this->template Dealloc<size_t>(ipiv, mindim);
	}
	void Release() const{
		this->Invalidate();
	}
	
	bool IsValid() const{ return base_type::IsValid() && 0 == err; }

	void Solve(Side side, Trans trans, Matrix<T> &B, const T alpha = T(1)) const{
		RNPAssert(m == n && n == B.rows());
		if(Left == side){
			if(NoTranspose == trans){
				for(size_t i = 0; i < m; ++i){
					if(ipiv[i] != i){
						BLAS::Swap(B.cols(), &B(i,0), B.ldim(), &B(ipiv[i],0), B.ldim());
					}
				}
				BLAS::SolveTrM("L","L","N","U", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				BLAS::SolveTrM("L","U","N","N", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
			}else{
				if(Transpose == trans){
					BLAS::SolveTrM("L","U","T","N", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					BLAS::SolveTrM("L","L","T","U", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}else{ // "C"
					BLAS::SolveTrM("L","U","C","N", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					BLAS::SolveTrM("L","L","C","U", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}
				size_t i = m; while(i --> 0){
					if(ipiv[i] != i){
						BLAS::Swap(B.cols(), &B(i,0), B.ldim(), &B(ipiv[i],0), B.ldim());
					}
				}
			}
		}else{
			if(NoTranspose == trans){
				BLAS::SolveTrM("R","U","N","N", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				BLAS::SolveTrM("R","L","N","U", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				for(size_t i = 0; i < n; ++i){
					if(ipiv[i] != i){
						BLAS::Swap(B.rows(), &B(0,i), 1, &B(0,ipiv[i]), 1);
					}
				}
			}else{
				size_t i = n; while(i --> 0){
					if(ipiv[i] != i){
						BLAS::Swap(B.rows(), &B(0,i), 1, &B(0,ipiv[i]), 1);
					}
				}
				if(Transpose == trans){
					BLAS::SolveTrM("R","U","T","N", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					BLAS::SolveTrM("R","L","T","U", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}else{ // "C"
					BLAS::SolveTrM("R","U","C","N", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					BLAS::SolveTrM("R","L","C","U", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}
			}
		}
	}
	void Mult(Trans trans, Side side, Matrix<T> &B, const T &alpha = T(1)) const{
		if(Left == side){
			if(NoTranspose == trans){
				BLAS::MultTrM("L","U","N","N", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				BLAS::MultTrM("L","L","N","U", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				size_t i = n; while(i --> 0){
					if(ipiv[i] != i){
						BLAS::Swap(B.cols(), &B(i,0), B.ldim(), &B(ipiv[i],0), B.ldim());
					}
				}
			}else{
				for(size_t i = 0; i < n; ++i){
					if(ipiv[i] != i){
						BLAS::Swap(B.cols(), &B(i,0), B.ldim(), &B(ipiv[i],0), B.ldim());
					}
				}
				if(Transpose == trans){
					BLAS::MultTrM("L","L","T","U", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					BLAS::MultTrM("L","U","T","N", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}else{
					BLAS::MultTrM("L","L","C","U", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					BLAS::MultTrM("L","U","C","N", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}
			}
		}else{
			if(NoTranspose == trans){
				for(size_t i = 0; i < n; ++i){
					if(ipiv[i] != i){
						BLAS::Swap(B.rows(), &B(0,i), 1, &B(0,ipiv[i]), 1);
					}
				}
				BLAS::MultTrM("R","L","N","U", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				BLAS::MultTrM("R","U","N","N", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
			}else{
				if(Transpose == trans){
					BLAS::MultTrM("R","U","T","N", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					BLAS::MultTrM("R","L","T","U", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}else{
					BLAS::MultTrM("R","U","C","N", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					BLAS::MultTrM("R","L","C","U", B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}
				size_t i = n; while(i --> 0){
					if(ipiv[i] != i){
						BLAS::Swap(B.rows(), &B(0,i), 1, &B(0,ipiv[i]), 1);
					}
				}
			}
		}
	}
	void MakeL(Matrix<T> &L) const{
		RNPAssert(L.rows() == m && L.cols() == n);
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				if(i == j){
					L(i,j) = T(1);
				}else if(i > j){
					L(i,j) = a[i+j*lda];
				}else{
					L(i,j) = T(0);
				}
			}
		}
	}
	void MakeU(Matrix<T> &L) const{
		RNPAssert(L.rows() == m && L.cols() == n);
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				if(i <= j){
					L(i,j) = a[i+j*lda];
				}else{
					L(i,j) = T(0);
				}
			}
		}
	}
	template <typename IntType>
	void MakeP(Matrix<IntType> &P) const{
		RNP::TBLAS::Kernel K;
		size_t k = (m < n ? m : n);
		RNPAssert(P.rows() == P.cols() && P.rows() == k);
		P.Identity();
		for(size_t i = 0; i < k; ++i){
			if(ipiv[i] != i){
				BLAS::Swap(P.rows(), &P(0,i), 1, &P(0,ipiv[i]), 1);
			}
		}
	}

	void Determinant(T *mant, real_type *base, int *expo) const{
		static const real_type b = 16.;
		*base = b;
		*expo = 0;
		*mant = T(1);
		for(size_t i = 0; i < n; ++i){
			*mant *= a[i+i*lda];
			if(T(0) == *mant){ break; }
			while(Traits<T>::abs(*mant) < 1){
				*mant *= b;
				(*expo)--;
			}
			while(Traits<T>::abs(*mant) >= b){
				*mant /= b;
				(*expo)++;
			}
			if(ipiv[i] != i){ *mant = -(*mant); }
		}
	}
protected:
	// Computes A = P*L*U
	void Compute(){
		err = PLUFactor(m, n, a, lda, ipiv);
	}
};

}; // namespace LA
}; // namespace RNP

#endif // RNP_LU_OBJ_HPP_INCLUDED
