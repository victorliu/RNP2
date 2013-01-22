#include <iostream>
#include <RNP/Random.hpp>
#include <RNP/LA.hpp>

template <typename T>
void test_trisolve(
	const char *uplo, const char *trans,
	const char *diag, size_t n, size_t nx
){
	typedef typename RNP::Traits<T>::real_type real_type;
	
	T *A = new T[n*n];
	T *x = new T[n];
	T *y = new T[n];
	real_type scale;
	real_type *cnorm = new real_type[n];
	if('L' == uplo[0]){
		for(size_t j = 0; j < n; ++j){
			RNP::Random::GenerateVector(
				RNP::Random::Distribution::Uniform_11, n-j, &A[j+j*n]
			);
		}
	}else{
		for(size_t j = 0; j < n; ++j){
			RNP::Random::GenerateVector(
				RNP::Random::Distribution::Uniform_11, 1+j, &A[0+j*n]
			);
		}
	}
	
	for(size_t ix = 0; ix < nx; ++ix){
		RNP::Random::GenerateVector(
			RNP::Random::Distribution::Uniform_11, n, y
		);
		// Solve A*x = y for x
		RNP::BLAS::Copy(n, y, 1, x, 1);
		/*{
			int nn = n;
			int info;
			dlatrs_(uplo, trans, diag, "N", &nn, A, &nn, x, &scale, cnorm, &info,1,1,1,1);
		}*/
		
		RNP::LA::Triangular::Solve(
			uplo, trans, diag, "N", n, A, n, x, &scale, cnorm
		);
		//RNP::BLAS::SolveTrV(uplo, trans, diag, n, A, n, x, 1);
		// Compute x <- A*x
		RNP::BLAS::MultTrV(uplo, trans, diag, n, A, n, x, 1);
		// Set x = A*x - y
		RNP::BLAS::Axpy(n, -scale, y, 1, x, 1);
		// Compute residual
		real_type resid = RNP::BLAS::Norm1(n, x, 1) / (n*RNP::Traits<real_type>::eps());
		if(resid > 30.){ std::cout << "bad" << std::endl; }
	}
	
	delete [] y;
	delete [] x;
	delete [] A;
}

int main(){
	srand(0);
	const size_t n = 100, nx = 10;
	test_trisolve<double>("U", "N", "N", n, nx);
	test_trisolve<double>("U", "N", "U", n, nx);
	test_trisolve<double>("U", "T", "N", n, nx);
	test_trisolve<double>("U", "T", "U", n, nx);
	test_trisolve<double>("U", "C", "N", n, nx);
	test_trisolve<double>("U", "C", "U", n, nx);
	
	test_trisolve<double>("L", "N", "N", n, nx);
	test_trisolve<double>("L", "N", "U", n, nx);
	test_trisolve<double>("L", "T", "N", n, nx);
	test_trisolve<double>("L", "T", "U", n, nx);
	test_trisolve<double>("L", "C", "N", n, nx);
	test_trisolve<double>("L", "C", "U", n, nx);
	
	typedef std::complex<double> complex_t;
	test_trisolve<complex_t>("U", "N", "N", n, nx);
	test_trisolve<complex_t>("U", "N", "U", n, nx);
	test_trisolve<complex_t>("U", "T", "N", n, nx);
	test_trisolve<complex_t>("U", "T", "U", n, nx);
	test_trisolve<complex_t>("U", "C", "N", n, nx);
	test_trisolve<complex_t>("U", "C", "U", n, nx);
	
	test_trisolve<complex_t>("L", "N", "N", n, nx);
	test_trisolve<complex_t>("L", "N", "U", n, nx);
	test_trisolve<complex_t>("L", "T", "N", n, nx);
	test_trisolve<complex_t>("L", "T", "U", n, nx);
	test_trisolve<complex_t>("L", "C", "N", n, nx);
	test_trisolve<complex_t>("L", "C", "U", n, nx);
	return 0;
}
