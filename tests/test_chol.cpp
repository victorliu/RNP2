#include <RNP/LA/Cholesky.hpp>
#include <RNP/LA.hpp>
#include <RNP/BLAS.hpp>
#include <RNP/IOBasic.hpp>
#include <RNP/Random.hpp>
#include <iostream>
#include <cstdlib>

template <typename T>
void test_chol(const char *uplo, size_t n){
	typedef typename RNP::Traits<T>::real_type real_type;
	real_type rsnrm(1./((n*n) * RNP::Traits<real_type>::eps()));
	T *Afac = new T[n*n];
	for(size_t j = 0; j < n; ++j){
		RNP::Random::GenerateVector(RNP::Random::Distribution::Uniform_11, n, &Afac[0+j*n]);
	}
	T *A = new T[n*n];
	RNP::BLAS::MultMM("C", "N", n, n, n, T(1), Afac, n, Afac, n, T(0), A, n);

	// Workspace
	T *B = new T[n*n];
	T *C = new T[n*n];
	
	if(0){
		std::cout << "Original A:" << std::endl;
		RNP::Matrix<T> mA(n, n, A, n);
		std::cout << RNP::IO::Chop(mA) << std::endl << std::endl;
	}
	
	RNP::BLAS::Copy(n, n, A, n, Afac, n);
	RNP::LA::Cholesky::Factor(uplo, n, Afac, n);
	
	if(0){
		std::cout << "Factored A:" << std::endl;
		RNP::Matrix<T> mA(n, n, Afac, n);
		std::cout << RNP::IO::Chop(mA) << std::endl << std::endl;
	}

	RNP::BLAS::Copy(n, n, A, n, B, n);
	RNP::BLAS::Set(n, n, T(0), T(0), C, n);
	RNP::LA::Triangular::Copy(uplo, n, n, Afac, n, C, n);
	
	if('U' == uplo[0]){
		RNP::BLAS::MultMM("C", "N", n, n, n, T(1), C, n, C, n, T(-1), B, n);
	}else{
		RNP::BLAS::MultMM("N", "C", n, n, n, T(1), C, n, C, n, T(-1), B, n);
	}
	
	if(0){
		std::cout << "F*F - A:" << std::endl;
		RNP::Matrix<T> mB(n, n, B, n);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	// Check to see if the lower triangle is correct
	if(1){
		T sum = 0;
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < n; ++i){
				sum += RNP::Traits<T>::abs(B[i+j*n]);
			}
		}
		std::cout << "diff norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	delete [] C;
	delete [] B;
	delete [] Afac;
	delete [] A;
}

int main(){
	srand(0);
	size_t n = 150;
	test_chol<double>("U", n);
	test_chol<std::complex<double> >("U", n);
	test_chol<double>("L", n);
	test_chol<std::complex<double> >("L", n);
	return 0;
}
