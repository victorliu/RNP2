#include <RNP/LA.hpp>
#include <RNP/BLAS.hpp>
#include <RNP/MatrixGen.hpp>
#include <RNP/IOBasic.hpp>
#include <iostream>
#include <cstdlib>

using namespace RNP;

double drand(){
	//return drand48() - 0.5;
	return (double)rand() / (double)RAND_MAX - 0.5;
}

// Types:
//   0: diagonal
//   1: upper trapezoidal
//   2: lower trapezoidal
//   3: full
// Condition types:
//   0: condition number = 0.1 / eps
//   1: condition number = sqrt(0.1 / eps)
//   2: condition number = 2
// Anorm types:
//   0: Anorm = min
//   1: Anorm = 1/min
//   2: Anorm = 1
template <typename T>
void test_lu(size_t m, size_t n, size_t nrhs,
	const typename Traits<T>::real_type &cond,
	const typename Traits<T>::real_type &anrm
){
	typedef typename Traits<T>::real_type real_type;
	const size_t mindim = (m < n ? m : n);
	//const size_t maxdim = (m > n ? m : n);
	real_type *d = new real_type[mindim];
	const size_t lda = m;
	T *A = new T[lda*n];
	T *Afac = new T[lda*n];
	T *Ainv = new T[lda*n];
	size_t *ipiv = new size_t[mindim];
	
	// Generate the matrix
	MatrixGen::RandomMatrix(
		MatrixGen::Symmetry::Nonsymmetric,
		Random::Distribution::Uniform_11,
		MatrixGen::DiagonalMode::GradedExponentially,
		false, // reverse
		cond, anrm, d,
		m, n, A, lda
	);
	
	const real_type Anorm = LA::MatrixNorm("1", m, n, A, lda);
	
	BLAS::Copy(m, n, A, lda, Afac, lda);
	int info = LA::PLUFactor(m, n, Afac, lda, ipiv);
	{ // Test 1: reconstruct matrix from factors
		BLAS::Copy(m, n, Afac, lda, Ainv, lda);
		size_t k = n;
		while(k --> 0){
			if(k >= m){
				BLAS::MultTrV("L","N","U", m, Ainv, lda, &Ainv[0+k*lda], 1);
			}else{	
				// Compute elements (k+1:m,k)
				T t = Ainv[k+k*lda];
				if(k+1 < m){
					BLAS::Scale(m-1-k, t, &Ainv[k+1+k*lda], 1);
					BLAS::MultMV("N", m-1-k, k, T(1), &Ainv[k+1+0*lda], lda, &Ainv[0+k*lda], 1, T(1), &Ainv[k+1+k*lda], 1);
				}
				// Compute element (k,k)
				Ainv[k+k*lda] = t + BLAS::Dot(k, &Ainv[k+0*lda], lda, &Ainv[0+k*lda], 1);
				// Compute elements (1:k-1,k)
				BLAS::MultTrV("L","N","U", k, Ainv, lda, &Ainv[0+k*lda], 1);
			}
		}
		LA::ApplyPermutations("L","I", m, n, Ainv, lda, ipiv);
		// Compute difference
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				Ainv[i+j*lda] -= A[i+j*lda];
			}
		}
		real_type resid = LA::MatrixNorm("1", m, n, Ainv, lda);
		if(resid <= 0){
			resid = real_type(1) / Traits<T>::eps();
		}else{
			resid = ((resid / real_type(n)) / Anorm) / Traits<T>::eps();
		}
		std::cout << "resid(L*U - A) = " << resid << std::endl;
	}
	if(m == n && 0 == info){
		// Test 2: Form inverse, and check that A * Ainv == I
		{
			BLAS::Copy(m, n, Afac, lda, Ainv, lda);
			
			size_t lwork = 0;
			T *work = NULL;
			LA::PLUInvert(n, Ainv, lda, ipiv, &lwork, work);
			work = new T[lwork];
			LA::PLUInvert(n, Ainv, lda, ipiv, &lwork, work);
			delete [] work;
			
			const real_type Ainvnorm = LA::MatrixNorm("1", m, n, Ainv, lda);
			real_type rcond(0), resid;
			if(Anorm <= real_type(0) || Ainvnorm <= real_type(0)){
				resid = real_type(1) / Traits<T>::eps();
			}else{
				rcond = (real_type(1) / Anorm) / Ainvnorm;
				// Compute I - A*Ainv
				work = new T[n*n];
				BLAS::MultMM("N","N", n, n, n, T(-1), Ainv, lda, A, lda, T(0), work, n);
				for(size_t i = 0; i < n; ++i){
					work[i+i*n] += T(1);
				}
				resid = LA::MatrixNorm("1", n, n, work, n);
				resid = ((resid*rcond) / Traits<T>::eps()) / real_type(n);
				delete [] work;
			}
			std::cout << "rcond(I - A*inv(A)) = " << rcond << std::endl;
			std::cout << "resid(I - A*inv(A)) = " << resid << std::endl;
		}
		
		// Test 3: Solve A*X == B
		T *B = new T[n*nrhs];
		T *X = new T[n*nrhs];

		// Test 4: Check that iterative refinement works
		delete [] X;
		delete [] B;
	}
	
	delete [] ipiv;
	delete [] Ainv;
	delete [] Afac;
	delete [] A;
	delete [] d;
}

int main(){
	test_lu<double>(5, 5, 13, 1, 1);
	return 0;
}
