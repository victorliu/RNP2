#include <ctime>
#include <iostream>
#include <RNP/bits/ComplexEigensystem.hpp>
#include <RNP/Random.hpp>
#include "nep/read_mm.hpp"

template <typename T>
void test_complex_eigensystem(size_t n, const T *A){
	typedef typename RNP::Traits<T>::real_type real_type;
	
	T *Afac = new T[n*n];
	T *vl = new T[n*n];
	T *vr = new T[n*n];
	T *work = NULL;
	T *evals = new T[n];
	int *iwork = new int[n];
	
	
	RNP::BLAS::Copy(n, n, A, n, Afac, n);
	size_t lwork = 0;
	RNP::LA::ComplexEigensystem(n, Afac, n, evals, vl, n, vr, n, &lwork, work, iwork);
	work = new T[lwork];
	RNP::LA::ComplexEigensystem(n, Afac, n, evals, vl, n, vr, n, &lwork, work, iwork);
	
	for(size_t j = 0; j < n; ++j){
		// A * vr = eval*vr
		real_type vnrm = RNP::BLAS::Norm1(n, &vr[0+j*n], 1);
		RNP::BLAS::MultMV("N", n, n, T(1), A, n, &vr[0+j*n], 1, T(0), work, 1);
		RNP::BLAS::Axpy(n, -evals[j], &vr[0+j*n], 1, work, 1);
		// Compute residual
		real_type resid = RNP::BLAS::Norm1(n, work, 1) / (n*vnrm*RNP::Traits<real_type>::eps());
		if(resid > 30.){ std::cout << "bad" << std::endl; }
		//std::cout << resid << std::endl;
		
		// A' * vl = conj(eval)*vl
		vnrm = RNP::BLAS::Norm1(n, &vl[0+j*n], 1);
		RNP::BLAS::MultMV("C", n, n, T(1), A, n, &vl[0+j*n], 1, T(0), work, 1);
		RNP::BLAS::Axpy(n, -RNP::Traits<T>::conj(evals[j]), &vl[0+j*n], 1, work, 1);
		// Compute residual
		resid = RNP::BLAS::Norm1(n, work, 1) / (n*vnrm*RNP::Traits<real_type>::eps());
		if(resid > 30.){ std::cout << "bad" << std::endl; }
		//std::cout << resid << std::endl;
	}
	
	delete [] evals;
	delete [] iwork;
	delete [] work;
	delete [] vr;
	delete [] vl;
}

int main(){
	srand(time(NULL));
	/*
	const size_t n = 200;
	std::complex<double> *A = new std::complex<double>[n*n];
	for(size_t j = 0; j < n; ++j){
		RNP::Random::GenerateVector(
			RNP::Random::Distribution::Uniform_11, n, &A[0+j*n]
		);
	}
	*/
	
	int nr, nc;
	std::complex<double> *A = read_mm<std::complex<double> >(
		//"nep/rdb200.mtx"
		"nep/qc324.mtx",
		&nr, &nc
	);
	const size_t n = nr;
	
	test_complex_eigensystem<std::complex<double> >(n, A);
	delete [] A;
	return 0;
}
