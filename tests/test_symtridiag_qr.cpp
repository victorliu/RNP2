#include <RNP/BLAS.hpp>
#include <RNP/LA.hpp>
#include <RNP/IOBasic.hpp>
#include <iostream>
#include <cstdlib>
using namespace RNP;

template <typename T>
void test_tridiag(size_t n){
	typedef typename Traits<T>::real_type real_type;
	
	T *diag = new T[n];
	T *offdiag = new T[n-1];
	T *z = new T[n*n];
	T *work = new T[n*n];
	T *work2 = new T[n*n];
	T *d = new T[n];
	T *e = new T[n-1];
	
	// Generate some simple random tridiagonal matrix
	Random::GenerateVector(Random::Distribution::Uniform01, n, diag);
	Random::GenerateVector(Random::Distribution::Uniform_11, n-1, offdiag);

	for(size_t i = 1; i < n; ++i){
		//diag[i] = diag[i-1] / 0.9;
	}
	BLAS::Copy(n,    diag, 1, d, 1);
	BLAS::Copy(n-1, offdiag, 1, e, 1);
	if(0){
		Vector<T> vdiag(n, diag, 1);
		Vector<T> voffdiag(n-1, offdiag, 1);
		std::cout << "Diag:\n" << IO::Chop(vdiag) << std::endl;
		std::cout << "Offdiag:\n" << IO::Chop(voffdiag) << std::endl;
	}
	
	LA::Tridiagonal::SymmetricEigensystem(n, d, e, z, n, work);
	
	// Form Z*Lambda*Z'
	BLAS::Copy(n, n, z, n, work, n);
	for(size_t i = 0; i < n; ++i){
		BLAS::Scale(n, d[i], &work[0+i*n], 1);
	}
	BLAS::MultMM("N", "C", n, n, n, T(1), work, n, z, n, T(0), work2, n);
	
	if(0){
		Matrix<T> mresult(n, n, work2, n);
		std::cout << IO::Chop(mresult) << std::endl;
	}
	
	for(size_t i = 0; i < n; ++i){
		work2[i+i*n] -= diag[i];
	}
	for(size_t i = 0; i+1 < n; ++i){
		work2[i+1+i*n] -= offdiag[i];
		work2[i+(i+1)*n] -= offdiag[i];
	}
	
	real_type diff = LA::MatrixNorm("M", n, n, work2, n);
	std::cout << diff << std::endl;
	
	delete [] work2;
	delete [] e;
	delete [] d;
	delete [] work;
	delete [] z;
	delete [] offdiag;
	delete [] diag;
}

int main(){
	for(size_t i = 0; i < 20; ++i){
		test_tridiag<double>(200);
	}
	return 0;
}
