#include <cstdlib>
#include <cmath>
#include <complex>
#include <limits>
#include <iostream>
#include <ctime>
#include <RNP/LA/GeneralizedEigensystems.hpp>
#include <RNP/Random.hpp>

double frand(){
	return (double)rand()/(double)RAND_MAX;
}

template <typename T>
void test_zggev(size_t n, const T *A, const T *B){
	typedef typename RNP::Traits<T>::real_type real_type;
	
	std::complex<double> *Ared = new std::complex<double>[n*n];
	std::complex<double> *Bred = new std::complex<double>[n*n];
	std::complex<double> *alpha = new std::complex<double>[n];
	std::complex<double> *beta  = new std::complex<double>[n];
	std::complex<double> *U = new std::complex<double>[n*n];
	std::complex<double> *V = new std::complex<double>[n*n];
	
	RNP::BLAS::Copy(n, n, A, n, Ared, n);
	RNP::BLAS::Copy(n, n, B, n, Bred, n);
	
	size_t lwork = 3*n;
	std::complex<double> *work = new std::complex<double>[lwork];
	int *iwork = new int[2*n];
	int info;
	info = RNP::LA::ComplexGeneralizedEigensystem(n,Ared,n,Bred,n,alpha,beta,U,n,V,n,&lwork,work,iwork);
	
	if(0){
		for(size_t i = 0; i < 2*n; ++i){
			std::cout << "iw[" << i << "] = " << iwork[i] << std::endl;
		}
	}
	
	delete [] iwork;
	
	std::cout << "Info = " << info << std::endl;
	
	// Verify
	std::cout << "Checking right eigenvectors" << std::endl;
	for(size_t j = 0; j < n; ++j){
		real_type vnrm = RNP::BLAS::Norm1(n, &V[0+j*n], 1);
		// Compute beta*A*v - alpha*B*v
		RNP::BLAS::MultMV("N", n, n,   beta[j], A, n, &V[0+j*n], 1, 0., work, 1);
		RNP::BLAS::MultMV("N", n, n, -alpha[j], B, n, &V[0+j*n], 1, 1., work, 1);
		double resid = RNP::BLAS::Norm1(n, work, 1) / (n*vnrm*RNP::Traits<real_type>::eps());
		if(resid > 30){ std::cout << "bad" << std::endl; }
		//std::cout << "maxerr = " << resid << std::endl;
	}
	std::cout << "Checking left eigenvectors" << std::endl;
	for(size_t j = 0; j < n; ++j){
		real_type unrm = RNP::BLAS::Norm1(n, &U[0+j*n], 1);
		// Compute beta*u^H*A - alpha*u^H*B
		RNP::BLAS::MultMV("C", n, n,  std::conj( beta[j]), A, n, &U[0+j*n], 1, 0., work, 1);
		RNP::BLAS::MultMV("C", n, n, -std::conj(alpha[j]), B, n, &U[0+j*n], 1, 1., work, 1);
		double resid = RNP::BLAS::Norm1(n, work, 1) / (n*unrm*RNP::Traits<real_type>::eps());
		if(resid > 30){ std::cout << "bad" << std::endl; }
		//std::cout << "maxerr = " << resid << std::endl;
	}
	
	delete [] work;
	delete [] Ared;
	delete [] Bred;
	delete [] alpha;
	delete [] beta;
	delete [] U;
	delete [] V;
}

int main(){
	srand(time(0));
	size_t n = 150;
	std::complex<double> *A = new std::complex<double>[n*n];
	std::complex<double> *B = new std::complex<double>[n*n];
	
	for(size_t j = 0; j < n; ++j){
		RNP::Random::GenerateVector(
			RNP::Random::Distribution::Uniform_11, n, &A[0+j*n]
		);
	}
	for(size_t j = 0; j < n; ++j){
		RNP::Random::GenerateVector(
			RNP::Random::Distribution::Uniform_11, n, &B[0+j*n]
		);
	}
	
	// Add random scaling to test balancing
	for(size_t i = 0; i < n; ++i){
		RNP::BLAS::Scale(n, frand(), &A[i+0*n], n);
	}
	for(size_t j = 0; j < n; ++j){
		RNP::BLAS::Scale(n, frand(), &A[0+j*n], 1);
	}
	
	test_zggev(n, A, B);
	
	delete [] A;
	delete [] B;
	return 0;
}
