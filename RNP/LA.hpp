#ifndef RNP_LA_HPP_INCLUDED
#define RNP_LA_HPP_INCLUDED

#include <RNP/Types.hpp>
#include <RNP/Debug.hpp>
#include <RNP/TBLAS.hpp>
#include <RNP/TBLAS_Kernel.hpp>

namespace RNP{
namespace LA{

template <class T>
class LUFactors : public Base{
	T *a;
	size_t m, n, lda;
	size_t *ipiv;
	int err;
public:
	LUFactors(const Matrix<T> &A):a(A.ptr()),m(A.rows()),n(A.cols()),lda(A.ldim()),ipiv(NULL){
		ipiv = (size_t*)malloc(sizeof(size_t) * (m < n ? m : n));
		Compute();
	}
	~LUFactors(){
		free(ipiv);
	}
	void Release(){
		free(ipiv); ipiv = NULL;
		Invalidate();
	}
	
	bool IsValid() const{ return Base::IsValid() && 0 == err; }

	void Solve(Trans trans, Side side, Matrix<T> &B, const T alpha = T(1)) const{
		RNP::TBLAS::Kernel K;
		if(Left == side){
			if(NoTranspose == trans){
				for(size_t i = 0; i < n; ++i){
					if(ipiv[i] != i){
						K.Swap(B.cols(), &B(i,0), B.ldim(), &B(ipiv[i],0), B.ldim());
					}
				}
				K.SolveTrM('L','L','N','U', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				K.SolveTrM('L','U','N','N', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
			}else{
				if(Transpose == trans){
					K.SolveTrM('L','U','T','N', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					K.SolveTrM('L','L','T','U', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}else{ // 'C'
					K.SolveTrM('L','U','C','N', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					K.SolveTrM('L','L','C','U', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}
				size_t i = n; while(i --> 0){
					if(ipiv[i] != i){
						K.Swap(B.cols(), &B(i,0), B.ldim(), &B(ipiv[i],0), B.ldim());
					}
				}
			}
		}else{
			if(NoTranspose == trans){
				K.SolveTrM('R','U','N','N', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				K.SolveTrM('R','L','N','U', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				for(size_t i = 0; i < n; ++i){
					if(ipiv[i] != i){
						K.Swap(B.rows(), &B(0,i), 1, &B(0,ipiv[i]), 1);
					}
				}
			}else{
				size_t i = n; while(i --> 0){
					if(ipiv[i] != i){
						K.Swap(B.rows(), &B(0,i), 1, &B(0,ipiv[i]), 1);
					}
				}
				if(Transpose == trans){
					K.SolveTrM('R','U','T','N', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					K.SolveTrM('R','L','T','U', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}else{ // 'C'
					K.SolveTrM('R','U','C','N', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					K.SolveTrM('R','L','C','U', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}
			}
		}
	}
	void Mult(Trans trans, Side side, Matrix<T> &B, const T &alpha = T(1)) const{
		RNP::TBLAS::Kernel K;
		if(Left == side){
			if(NoTranspose == trans){
				K.MultTrM('L','U','N','N', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				K.MultTrM('L','L','N','U', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				size_t i = n; while(i --> 0){
					if(ipiv[i] != i){
						K.Swap(B.cols(), &B(i,0), B.ldim(), &B(ipiv[i],0), B.ldim());
					}
				}
			}else{
				for(size_t i = 0; i < n; ++i){
					if(ipiv[i] != i){
						K.Swap(B.cols(), &B(i,0), B.ldim(), &B(ipiv[i],0), B.ldim());
					}
				}
				if(Transpose == trans){
					K.MultTrM('L','L','T','U', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					K.MultTrM('L','U','T','N', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}else{
					K.MultTrM('L','L','C','U', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					K.MultTrM('L','U','C','N', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}
			}
		}else{
			if(NoTranspose == trans){
				for(size_t i = 0; i < n; ++i){
					if(ipiv[i] != i){
						K.Swap(B.rows(), &B(0,i), 1, &B(0,ipiv[i]), 1);
					}
				}
				K.MultTrM('R','L','N','U', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				K.MultTrM('R','U','N','N', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
			}else{
				if(Transpose == trans){
					K.MultTrM('R','U','T','N', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					K.MultTrM('R','L','T','U', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}else{
					K.MultTrM('R','U','C','N', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
					K.MultTrM('R','L','C','U', B.rows(), B.cols(), alpha, a, lda, B.ptr(), B.ldim());
				}
				size_t i = n; while(i --> 0){
					if(ipiv[i] != i){
						K.Swap(B.rows(), &B(0,i), 1, &B(0,ipiv[i]), 1);
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
				K.Swap(P.rows(), &P(0,i), 1, &P(0,ipiv[i]), 1);
			}
		}
	}
protected:
	// Computes A = P*L*U
	void Compute(){
		RNP::TBLAS::Kernel K;
		err = 0;
		size_t min_dim = (m < n ? m : n);
		for(size_t j = 0; j < min_dim; ++j){
			size_t jp = j + K.MaximumIndex(m-j, &a[j+j*lda], 1);
			ipiv[j] = jp;
			if(T(0) != a[jp+j*lda]){
				if(jp != j){
					K.Swap(n, &a[j+0*lda], lda, &a[jp+0*lda], lda);
				}
				if(j < m){
					K.Scale(m-j-1, T(1)/a[j+j*lda], &a[j+1+j*lda], 1); // possible overflow when inverting A(j,j)
				}
			}else{
				err = j;
			}
			if(j < min_dim){
				K.Rank1Update(m-j-1, n-j-1, T(-1), &a[j+1+j*lda], 1, &a[j+(j+1)*lda], lda, &a[j+1+(j+1)*lda], lda);
			}
		}
	}
};

class QRFactors : public Base{
};

class EigenFactors : public Base{
};

class SVD : public Base{
};

}; // namespace LA
}; // namespace RNP

#endif // RNP_LA_HPP_INCLUDED
