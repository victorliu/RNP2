#include <RNP/LA.hpp>
#include <RNP/BLAS.hpp>
#include <RNP/MatrixGen.hpp>
#include <RNP/IOBasic.hpp>
#include <iostream>
#include <cstdlib>

using namespace RNP;

template <typename T>
void test_tri(size_t n){
	typedef typename Traits<T>::real_type real_type;
	
	const size_t lda = n;
	T *A = new T[lda*n];
	T *Ainv = new T[lda*n];
	
	// Generate some simple random upper triangular matrix
	BLAS::Set(n, n, T(0), T(0), A, lda);
	for(size_t j = 0; j < n; ++j){
		Random::GenerateVector(Random::Distribution::Uniform_11, j+1, &A[0+j*lda]);
		A[j+j*lda] += T(2);
	}
	if(0){
		Matrix<T> mA(n, n, A, lda);
		std::cout << mA << std::endl;
	}
	
	// Invert the matrix
	BLAS::Copy(n, n, A, lda, Ainv, lda);
	LA::Triangular::Invert("U", "N", n, Ainv, lda);
	if(0){
		Matrix<T> mAinv(n, n, Ainv, lda);
		std::cout << mAinv << std::endl;
	}
	
	BLAS::MultTrM("L","U","N","N", n, n, T(-1), A, lda, Ainv, lda);
	for(size_t i = 0; i < n; ++i){
		Ainv[i+i*lda] += T(1);
	}
	if(0){
		Matrix<T> mAinv(n, n, Ainv, lda);
		std::cout << mAinv << std::endl;
	}
	
	real_type resid = LA::MatrixNorm("1", n, n, Ainv, lda);
	resid = resid / real_type(n);
	std::cout << "norm1(I - T*inv(T)) = " << resid << std::endl;
	
	delete [] Ainv;
	delete [] A;
}

int main(){
	test_tri<double>(400);
	return 0;
}
