#include <RNP/LA.hpp>
#include <RNP/BLAS.hpp>
#include <RNP/IOBasic.hpp>
#include <RNP/Random.hpp>
#include <iostream>
#include <cstdlib>

template <typename T>
void test_heev(const char *uplo, size_t n){
	typedef typename RNP::Traits<T>::real_type real_type;
	real_type rsnrm(1./((n*n) * RNP::Traits<real_type>::eps()));
	T *A = new T[n*n];
	for(size_t j = 0; j < n; ++j){
		RNP::Random::GenerateVector(RNP::Random::Distribution::Uniform_11, n, &A[0+j*n]);
	}
	for(size_t j = 0; j < n; ++j){
		A[j+j*n] = RNP::Traits<T>::real(A[j+j*n]);
		for(size_t i = j+1; i < n; ++i){
			A[i+j*n] = RNP::Traits<T>::conj(A[j+i*n]);
		}
	}
	
	if(0){
		std::cout << "Original A:" << std::endl;
		RNP::Matrix<T> mA(n, n, A, n);
		std::cout << RNP::IO::Chop(mA) << std::endl << std::endl;
	}
	
	T *Q = new T[n*n];
	T *M1 = new T[n*n];
	T *M2 = new T[n*n];
	real_type *D = new real_type[n];
	T *work = NULL;
	
	{ // Perform decomposition
		RNP::BLAS::Copy(n, n, A, n, Q, n);
		size_t lwork = 0;
		RNP::LA::HermitianEigensystem("V", uplo, n, Q, n, D, &lwork, work);
		std::cout << "lwork = " << lwork << std::endl;
		work = new T[lwork];
		RNP::LA::HermitianEigensystem("V", uplo, n, Q, n, D, &lwork, work);
	}
	
	if(0){
		std::cout << "D:" << std::endl;
		RNP::Vector<real_type> vD(n, D, 1);
		std::cout << RNP::IO::Chop(vD) << std::endl << std::endl;
	}
	if(0){
		std::cout << "Q:" << std::endl;
		RNP::Matrix<T> mA(n, n, Q, n);
		std::cout << RNP::IO::Chop(mA) << std::endl << std::endl;
	}
	
	// Multiply Q*D*Q' and check that we get A
	{
		RNP::BLAS::Copy(n, n, Q, n, M1, n);
		for(size_t j = 0; j < n; ++j){
			RNP::BLAS::Scale(n, D[j], &M1[0+j*n], 1);
		}
		RNP::BLAS::Copy(n, n, A, n, M2, n);
		RNP::BLAS::MultMM("N", "C", n, n, n, T(1), M1, n, Q, n, T(-1), M2, n);
		
		if(0){
			std::cout << "Q*D*Q':" << std::endl;
			RNP::Matrix<T> mA(n, n, M2, n);
			std::cout << RNP::IO::Chop(mA) << std::endl << std::endl;
		}
		if(1){
			T sum = 0;
			for(size_t j = 0; j < n; ++j){
				for(size_t i = 0; i < n; ++i){
					sum += RNP::Traits<T>::abs(M2[i+j*n]);
				}
			}
			std::cout << "Q*D*Q'-A norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
		}
	}
	
	delete [] work;
	delete [] M2;
	delete [] M1;
	delete [] Q;
	delete [] A;
}

int main(){
	srand(0);
	size_t n = 200;
	const char *uplo = "L";
	test_heev<double>(uplo, n);
	test_heev<std::complex<double> >(uplo, n);
	uplo = "U";
	test_heev<double>(uplo, n);
	test_heev<std::complex<double> >(uplo, n);
	return 0;
}
