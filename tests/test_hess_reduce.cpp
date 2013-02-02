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
	for(size_t j = 0; j < n; ++j){
		RNP::Random::GenerateVector(Random::Distribution::Uniform_11, n, &A[0+j*lda]);
	}
	T *Ared = new T[lda*n];
	T *B = new T[lda*n];
	T *tau = new T[n];
	T *work = NULL;
	RNP::BLAS::Copy(n, n, A, lda, Ared, lda);
	
	{ // Perform reduction
		size_t lwork = 0;
		RNP::LA::Hessenberg::Reduce(n, 0, n, Ared, lda, tau, &lwork, work);
		work = new T[lwork];
		RNP::LA::Hessenberg::Reduce(n, 0, n, Ared, lda, tau, &lwork, work);
		//RNP::LA::Hessenberg::Reduce_unblocked(n, 0, n, Ared, lda, tau, work);
	}
	
	{ // Form Q*H*Q' using MultQ
		RNP::BLAS::Set(n, n, T(0), T(0), B, n);
		for(size_t j = 0; j < n; ++j){
			size_t ilimit = j+2; if(n < ilimit){ ilimit = n; }
			for(size_t i = 0; i < ilimit; ++i){
				B[i+j*lda] = Ared[i+j*lda];
			}
		} // B = H
		size_t lwork = 0;
		LA::Hessenberg::MultQ("R", "C", n, n, 0, n, Ared, lda, tau, B, lda, &lwork, work);
		if(NULL != work){ delete [] work; } work = new T[lwork];
		LA::Hessenberg::MultQ("R", "C", n, n, 0, n, Ared, lda, tau, B, lda, &lwork, work);
		LA::Hessenberg::MultQ("L", "N", n, n, 0, n, Ared, lda, tau, B, lda, &lwork, work);
		// Subtract off A
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < n; ++i){
				B[i+j*lda] -= A[i+j*lda];
			}
		}
		real_type maxdiff = rsnrm * LA::MatrixNorm("M", n, n, B, lda);
		std::cout << maxdiff << std::endl;
	}
	
	{ // Form Q*H*Q' using GenerateQ
		T *Q = new T[n*n];
		T *C = new T[n*n];
		size_t lwork = 0;
		BLAS::Copy(n, n, Ared, lda, Q, n);
		LA::Hessenberg::GenerateQ(n, 0, n, Q, n, tau, &lwork, work);
		std::cout << "lwork = " << lwork << std::endl;
		if(NULL != work){ delete [] work; } work = new T[lwork];
		work = new T[lwork];
		LA::Hessenberg::GenerateQ(n, 0, n, Q, n, tau, &lwork, work);
		
		if(0){
			std::cout << "Q:" << std::endl;
			RNP::Matrix<T> mQ(n, n, Q, n);
			std::cout << RNP::IO::Chop(mQ) << std::endl << std::endl;
		}
		
		for(size_t j = 0; j < n; ++j){
			size_t ilimit = j+2; if(n < ilimit){ ilimit = n; }
			for(size_t i = 0; i < ilimit; ++i){
				B[i+j*lda] = Ared[i+j*lda];
			}
		} // B = H
		
		RNP::BLAS::MultMM("N", "N", n, n, n, T(1), Q, n, B, n, T(0), C, n);
		RNP::BLAS::MultMM("N", "C", n, n, n, T(1), C, n, Q, n, T(0), B, n);

		// Subtract off A
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < n; ++i){
				B[i+j*lda] -= A[i+j*lda];
			}
		}
		real_type maxdiff = rsnrm * LA::MatrixNorm("M", n, n, B, lda);
		std::cout << maxdiff << std::endl;
		delete [] C;
		delete [] Q;
	}
	
	{
		// Form Q'*A*Q using MultQ
		BLAS::Copy(n, n, A, lda, B, lda);
		size_t lwork = 0;
		LA::Hessenberg::MultQ("R", "N", n, n, 0, n, Ared, lda, tau, B, lda, &lwork, work);
		if(NULL != work){ delete [] work; } work = new T[lwork];
		LA::Hessenberg::MultQ("R", "N", n, n, 0, n, Ared, lda, tau, B, lda, &lwork, work);
		LA::Hessenberg::MultQ("L", "C", n, n, 0, n, Ared, lda, tau, B, lda, &lwork, work);
		// Subtract off H
		for(size_t j = 0; j < n; ++j){
			size_t ilimit = j+2; if(n < ilimit){ ilimit = n; }
			for(size_t i = 0; i < ilimit; ++i){
				B[i+j*lda] -= Ared[i+j*lda];
			}
		}
		real_type maxdiff = rsnrm * LA::MatrixNorm("M", n, n, B, lda);
		std::cout << maxdiff << std::endl;
	}
	
	{
		// Form Q'*A*Q using GenerateQ
		T *Q = new T[n*n];
		T *C = new T[n*n];
		size_t lwork = 0;
		BLAS::Copy(n, n, Ared, lda, Q, n);
		LA::Hessenberg::GenerateQ(n, 0, n, Q, n, tau, &lwork, work);
		std::cout << "lwork = " << lwork << std::endl;
		if(NULL != work){ delete [] work; } work = new T[lwork];
		work = new T[lwork];
		LA::Hessenberg::GenerateQ(n, 0, n, Q, n, tau, &lwork, work);
		
		RNP::BLAS::MultMM("C", "N", n, n, n, T(1), Q, n, A, n, T(0), C, n);
		RNP::BLAS::MultMM("N", "N", n, n, n, T(1), C, n, Q, n, T(0), B, n);
		
		// Subtract off H
		for(size_t j = 0; j < n; ++j){
			size_t ilimit = j+2; if(n < ilimit){ ilimit = n; }
			for(size_t i = 0; i < ilimit; ++i){
				B[i+j*lda] -= Ared[i+j*lda];
			}
		}
		real_type maxdiff = rsnrm * LA::MatrixNorm("M", n, n, B, lda);
		std::cout << maxdiff << std::endl;
		delete [] C;
		delete [] Q;
	}
	
	delete [] work;
	delete [] tau;
	delete [] B;
	delete [] Ared;
	delete [] A;
}

int main(){
	test_reduce<double>(200);
	test_reduce<std::complex<double> >(200);
}
