#include <RNP/LA.hpp>
#include <RNP/BLAS.hpp>
#include <RNP/IOBasic.hpp>
#include <RNP/Random.hpp>
#include <iostream>
#include <cstdlib>

extern "C" void dorgtr_(
	const char *uplo, const int &n, double *a, const int &lda,
	double *tau, double *work, const int &lwork, int *info);
extern "C" void dsytrd_(
	const char *uplo, const int &n, double *a, const int &lda,
	double *d, double *e, double *tau, double *work, const int &lwork, int *info);

template <typename T>
void test_hetrd(const char *uplo, size_t n){
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
	
	real_type *diag = new real_type[n];
	real_type *offdiag = new real_type[n-1];
	T *tau = new T[n-1];
	T *Ared = new T[n*n];
	T *Q = new T[n*n];
	T *M1 = new T[n*n];
	T *M2 = new T[n*n];
	T *work = NULL;
	
	{ // Perform reduction
		RNP::BLAS::Copy(n, n, A, n, Ared, n);
		size_t lwork = 0;
		T dummy;
		RNP::LA::Tridiagonal::ReduceHerm(uplo, n, Ared, n, diag, offdiag, tau, &lwork, &dummy);
		work = new T[lwork];
		RNP::LA::Tridiagonal::ReduceHerm(uplo, n, Ared, n, diag, offdiag, tau, &lwork, work);
		//RNP::LA::Tridiagonal::ReduceHerm_unblocked(uplo, n, Ared, n, diag, offdiag, tau);
	}
	
	if(0){
		std::cout << "Ared:" << std::endl;
		RNP::Matrix<T> mA(n, n, Ared, n);
		std::cout << RNP::IO::Chop(mA) << std::endl << std::endl;
	}
	
	// Multiply Q'*A*Q and check for tridiagonal
	{
		RNP::BLAS::Copy(n, n, A, n, M2, n);
		if(NULL != work){ delete [] work; work = NULL; }
		size_t lwork = 0;
		RNP::LA::Tridiagonal::MultQHerm("L", uplo, "C", n, n, Ared, n, tau, M2, n, &lwork, work);
		work = new T[lwork];
		RNP::LA::Tridiagonal::MultQHerm("L", uplo, "C", n, n, Ared, n, tau, M2, n, &lwork, work);
		RNP::LA::Tridiagonal::MultQHerm("R", uplo, "N", n, n, Ared, n, tau, M2, n, &lwork, work);
		
		if(0){
			std::cout << "Q'*A*Q:" << std::endl;
			RNP::Matrix<T> mA(n, n, M2, n);
			std::cout << RNP::IO::Chop(mA) << std::endl << std::endl;
		}
		if(1){
			T sum = 0;
			for(size_t j = 0; j < n; ++j){
				for(size_t i = 0; i+1 < j; ++i){
					sum += RNP::Traits<T>::abs(M2[i+j*n]);
				}
				for(size_t i = j+2; i < n; ++i){
					sum += RNP::Traits<T>::abs(M2[i+j*n]);
				}
			}
			std::cout << "Q'*A*Q zeros norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
		}
	}
	
	// Generate Q and check that Q'*A*Q is tridiagonal
	{
		RNP::BLAS::Copy(n, n, Ared, n, Q, n);
		if(NULL != work){ delete [] work; work = NULL; }
		size_t lwork = 0;
		RNP::LA::Tridiagonal::GenerateQHerm(uplo, n, Q, n, tau, &lwork, work);
		work = new T[lwork];
		RNP::LA::Tridiagonal::GenerateQHerm(uplo, n, Q, n, tau, &lwork, work);
		
		if(0){
			std::cout << "Q:" << std::endl;
			RNP::Matrix<T> mA(n, n, Q, n);
			std::cout << RNP::IO::Chop(mA) << std::endl << std::endl;
		}
		
		RNP::BLAS::MultMM("C", "N", n, n, n, T(1),  Q, n, A, n, T(0), M1, n);
		RNP::BLAS::MultMM("N", "N", n, n, n, T(1), M1, n, Q, n, T(0), M2, n);
		
		if(0){
			std::cout << "Q'*A*Q:" << std::endl;
			RNP::Matrix<T> mA(n, n, M2, n);
			std::cout << RNP::IO::Chop(mA) << std::endl << std::endl;
		}
		
		if(1){
			T sum = 0;
			for(size_t j = 0; j < n; ++j){
				for(size_t i = 0; i+1 < j; ++i){
					sum += RNP::Traits<T>::abs(M2[i+j*n]);
				}
				for(size_t i = j+2; i < n; ++i){
					sum += RNP::Traits<T>::abs(M2[i+j*n]);
				}
			}
			std::cout << "Q'*A*Q zeros norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
		}
	}
	
	delete [] work;
	delete [] M2;
	delete [] M1;
	delete [] Q;
	delete [] Ared;
	delete [] tau;
	delete [] offdiag;
	delete [] diag;
	delete [] A;
}

int main(){
	srand(0);
	size_t n = 200;
	const char *uplo = "L";
	test_hetrd<double>(uplo, n);
	test_hetrd<std::complex<double> >(uplo, n);
	uplo = "U";
	test_hetrd<double>(uplo, n);
	test_hetrd<std::complex<double> >(uplo, n);
	return 0;
}
