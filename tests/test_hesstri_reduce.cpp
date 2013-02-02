#include <RNP/LA.hpp>
#include <RNP/BLAS.hpp>
#include <RNP/bits/Hessenberg.hpp>
#include <RNP/IOBasic.hpp>
#include <iostream>
#include <cstdlib>

using namespace RNP;

template <typename T>
void test_reduce(size_t n){
	typedef typename RNP::Traits<T>::real_type real_type;
	real_type rsnrm(1./((n*n) * RNP::Traits<real_type>::eps()));
	
	const size_t lda = n;
	T *A = new T[lda*n];
	T *B = new T[lda*n];
	for(size_t j = 0; j < n; ++j){
		RNP::Random::GenerateVector(Random::Distribution::Uniform_11, n, &A[0+j*lda]);
	}
	for(size_t j = 0; j < n; ++j){
		RNP::Random::GenerateVector(Random::Distribution::Uniform_11, j+1, &B[0+j*lda]);
	}
	T *Q = new T[n*n];
	T *Z = new T[n*n];
	T *W1 = new T[n*n];
	T *W2 = new T[n*n];
	T *Ared = new T[lda*n];
	T *Bred = new T[lda*n];
	RNP::BLAS::Copy(n, n, A, lda, Ared, lda);
	RNP::BLAS::Copy(n, n, B, lda, Bred, lda);
	
	if(0){
		std::cout << "A:" << std::endl;
		RNP::Matrix<T> mH(n, n, A, lda);
		std::cout << RNP::IO::Chop(mH) << std::endl << std::endl;
	}
	if(0){
		std::cout << "B:" << std::endl;
		RNP::Matrix<T> mT(n, n, B, lda);
		std::cout << RNP::IO::Chop(mT) << std::endl << std::endl;
	}
	
	RNP::BLAS::Set(n, n, T(0), T(1), Q, n);
	RNP::BLAS::Set(n, n, T(0), T(1), Z, n);
	RNP::LA::Hessenberg::ReduceGeneralized_unblocked(n, 0, n, Ared, lda, Bred, lda, Q, n, Z, n);
	
	if(0){
		std::cout << "H:" << std::endl;
		RNP::Matrix<T> mH(n, n, Ared, lda);
		std::cout << RNP::IO::Chop(mH) << std::endl << std::endl;
	}
	if(0){
		std::cout << "T:" << std::endl;
		RNP::Matrix<T> mT(n, n, Bred, lda);
		std::cout << RNP::IO::Chop(mT) << std::endl << std::endl;
	}
	if(0){
		std::cout << "Q:" << std::endl;
		RNP::Matrix<T> mH(n, n, Q, n);
		std::cout << RNP::IO::Chop(mH) << std::endl << std::endl;
	}
	if(0){
		std::cout << "Z:" << std::endl;
		RNP::Matrix<T> mT(n, n, Z, n);
		std::cout << RNP::IO::Chop(mT) << std::endl << std::endl;
	}

	{ // Form Q*H*Z' - A
		RNP::BLAS::Set(n, n, T(0), T(0), W1, n);
		RNP::BLAS::MultMM("N","N", n, n, n, T(1), Q, n, Ared, lda, T(0), W1, n);
		RNP::BLAS::Copy(n, n, A, lda, W2, n);
		RNP::BLAS::MultMM("N","C", n, n, n, T(1), W1, n, Z, n, T(-1), W2, n);
		real_type maxdiff = rsnrm * LA::MatrixNorm("M", n, n, W2, n);
		std::cout << maxdiff << std::endl;
	}
	{ // Form Q*T*Z' - B
		RNP::BLAS::Set(n, n, T(0), T(0), W1, n);
		RNP::BLAS::MultMM("N","N", n, n, n, T(1), Q, n, Bred, lda, T(0), W1, n);
		RNP::BLAS::Copy(n, n, B, lda, W2, n);
		RNP::BLAS::MultMM("N","C", n, n, n, T(1), W1, n, Z, n, T(-1), W2, n);
		real_type maxdiff = rsnrm * LA::MatrixNorm("M", n, n, W2, n);
		std::cout << maxdiff << std::endl;
	}
	{ // Form Q*Q' - I
		RNP::BLAS::MultMM("N", "C", n, n, n, T(1), Q, n, Q, n, T(0), W1, n);
		for(size_t j = 0; j < n; ++j){ W1[j+j*n] -= T(1); }
		real_type maxdiff = rsnrm * LA::MatrixNorm("M", n, n, W1, n);
		std::cout << maxdiff << std::endl;
	}
	{ // Form Z*Z' - I
		RNP::BLAS::MultMM("N", "C", n, n, n, T(1), Z, n, Z, n, T(0), W1, n);
		for(size_t j = 0; j < n; ++j){ W1[j+j*n] -= T(1); }
		real_type maxdiff = rsnrm * LA::MatrixNorm("M", n, n, W1, n);
		std::cout << maxdiff << std::endl;
	}
	
	delete [] W1;
	delete [] W2;
	delete [] Q;
	delete [] Z;
	delete [] Bred;
	delete [] Ared;
	delete [] B;
	delete [] A;
}

int main(){
	test_reduce<double>(200);
	test_reduce<std::complex<double> >(200);
}
