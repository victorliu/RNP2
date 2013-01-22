
#include <iostream>
#include <RNP/LA.hpp>
#include <RNP/Random.hpp>

template <typename T>
void test_tri_eigvec(size_t n){
	typedef typename RNP::Traits<T>::real_type real_type;
	
	T *A = new T[n*n];
	T *vl = new T[n*n];
	T *vr = new T[n*n];
	T *work = new T[2*n];
	T *evals = new T[n];
	real_type *rwork = new real_type[n];
	
	for(size_t j = 0; j < n; ++j){
		RNP::Random::GenerateVector(
			RNP::Random::Distribution::Uniform_11, 1+j, &A[0+j*n]
		);
	}
	for(size_t j = 0; j < n; ++j){
		evals[j] = A[j+j*n];
	}
	
	RNP::LA::Triangular::Eigenvectors("A", NULL, n, A, n, vl, n, vr, n, work, rwork);
	for(size_t j = 0; j < n; ++j){
		// A * vr = eval*vr
		RNP::BLAS::Copy(n, &vr[0+j*n], 1, work, 1);
		RNP::BLAS::MultTrV("U", "N", "N", n, A, n, work, 1);
		RNP::BLAS::Axpy(n, -evals[j], &vr[0+j*n], 1, work, 1);
		// Compute residual
		real_type resid = RNP::BLAS::Norm1(n, work, 1) / (n*RNP::Traits<real_type>::eps());
		if(resid > 30.){ std::cout << "bad" << std::endl; }
		//std::cout << resid << std::endl;
		
		// A' * vl = conj(eval)*vl
		RNP::BLAS::Copy(n, &vl[0+j*n], 1, work, 1);
		RNP::BLAS::MultTrV("U", "C", "N", n, A, n, work, 1);
		RNP::BLAS::Axpy(n, -RNP::Traits<T>::conj(evals[j]), &vl[0+j*n], 1, work, 1);
		// Compute residual
		resid = RNP::BLAS::Norm1(n, work, 1) / (n*RNP::Traits<real_type>::eps());
		if(resid > 30.){ std::cout << "bad" << std::endl; }
		//std::cout << resid << std::endl;
		
	}
	
	delete [] evals;
	delete [] rwork;
	delete [] work;
	delete [] vr;
	delete [] vl;
	delete [] A;
}

int main(){
	srand(0);
	const size_t n = 100;
	test_tri_eigvec<double>(n);
	test_tri_eigvec<std::complex<double> >(n);
	return 0;
}
