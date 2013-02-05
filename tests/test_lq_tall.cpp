extern "C" void dlarfb_(
	const char *side, const char *trans, const char *direc, const char *storev,
	const int &m, const int &n, const int &k, double *v, const int &ldv,
	double *t, const int &ldt,
	double *c, const int &ldc, double *work, const int &ldwork);
extern "C" void dlarft_(
	const char *direc, const char *storev,
	const int &m, const int &n, double *a, const int &lda,
	double *tau, double *work, const int &ldwork);

#include <RNP/LA.hpp>
#include <RNP/BLAS.hpp>
#include <RNP/IOBasic.hpp>
#include <RNP/Random.hpp>
#include <iostream>
#include <cstdlib>

extern "C" void dgelq2_(const int &m, const int &n, double *a, const int &lda, double *tau, double *work, int *info);
extern "C" void dorml2_(
	const char *side, const char *trans,
	const int &m, const int &n, const int &k, double *a, const int &lda, double *tau,
	double *c, const int &ldc, double *work, int *info);
extern "C" void dorgl2_(
	const int &m, const int &n, const int &k, double *a, const int &lda, double *tau,
	double *work, int *info);
	
template <typename T>
void test_lq(size_t m, size_t n){
	typedef typename RNP::Traits<T>::real_type real_type;
	real_type rsnrm(1./((m*n) * RNP::Traits<real_type>::eps()));
	T *A = new T[m*n];
	for(size_t j = 0; j < n; ++j){
		RNP::Random::GenerateVector(RNP::Random::Distribution::Uniform_11, m, &A[0+j*m]);
	}
	T *Afac = new T[m*n];

	// Workspace
	T *B = new T[m*n];
	
	if(0){
		std::cout << "Original A:" << std::endl;
		RNP::Matrix<T> mA(m, n, A, m);
		std::cout << RNP::IO::Chop(mA) << std::endl << std::endl;
	}
	
	T *tau = new T[m];
	T *work = NULL;
	size_t lwork = 0;
	RNP::BLAS::Copy(m, n, A, m, Afac, m);
	RNP::LA::LQ::Factor(m, n, Afac, m, tau, &lwork, work);
	//lwork = m;
	std::cout << "lwork = " << lwork << std::endl;
	work = new T[lwork];
	RNP::LA::LQ::Factor(m, n, Afac, m, tau, &lwork, work);

	if(0){
		std::cout << "Factored A:" << std::endl;
		RNP::Matrix<T> mA(m, n, Afac, m);
		std::cout << RNP::IO::Chop(mA) << std::endl << std::endl;
	}
	
	// Apply Q' to the right of original A (use B for workspace)
	RNP::BLAS::Copy(m, n, A, m, B, m);
	delete [] work; work = NULL; lwork = 0;
	RNP::LA::LQ::MultQ("R", "C", m, n, n, Afac, m, tau, B, m, &lwork, work);
	//lwork = m;
	work = new T[lwork];
	RNP::LA::LQ::MultQ("R", "C", m, n, n, Afac, m, tau, B, m, &lwork, work);
	
	if(0){
		std::cout << "origA * Q':" << std::endl;
		RNP::Matrix<T> mB(m, n, B, m);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	// Check to see if the lower triangle is correct
	if(1){
		T sum = 0;
		for(size_t j = 0; j < n; ++j){
			for(size_t i = j; i < m; ++i){
				sum += RNP::Traits<T>::abs(Afac[i+j*m] - B[i+j*m]);
			}
		}
		std::cout << "L norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	// Apply Q to the right of L
	RNP::BLAS::Set(m, n, T(0), T(0), B, m);
	RNP::LA::Triangular::Copy("L", m, n, Afac, m, B, m);
	if(0){
		std::cout << "B = L:" << std::endl;
		RNP::Matrix<T> mB(m, n, B, m);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	RNP::LA::LQ::MultQ("R", "N", m, n, n, Afac, m, tau, B, m, &lwork, work);
	if(0){
		std::cout << "B = L*Q:" << std::endl;
		RNP::Matrix<T> mB(m, n, B, m);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	// We should recover the original matrix
	if(1){
		T sum = 0;
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				sum += RNP::Traits<T>::abs(A[i+j*m] - B[i+j*m]);
			}
		}
		std::cout << "(A - L*Q) norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	// Now treat B as a n-by-m matrix, and copy A' into it,
	// and apply Q from the left
	for(size_t j = 0; j < m; ++j){
		for(size_t i = 0; i < n; ++i){
			B[i+j*n] = RNP::Traits<T>::conj(A[j+i*m]);
		}
	}
	if(0){
		std::cout << "B = A':" << std::endl;
		RNP::Matrix<T> mB(n, m, B, n);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	RNP::LA::LQ::MultQ("L", "N", n, m, n, Afac, m, tau, B, n, &lwork, work);
	if(0){
		std::cout << "B = L':" << std::endl;
		RNP::Matrix<T> mB(n, m, B, n);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	// We should recover L'
	if(1){
		T sum = 0;
		for(size_t j = 0; j < n; ++j){
			for(size_t i = j; i < m; ++i){
				sum += RNP::Traits<T>::abs(Afac[i+j*m] - RNP::Traits<T>::conj(B[j+i*n]));
			}
		}
		std::cout << "L' norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	// Now set B = L', and apply Q' from the left to get A'
	RNP::BLAS::Set(n, m, T(0), T(0), B, n);
	for(size_t j = 0; j < n; ++j){
		for(size_t i = j; i < m; ++i){
			B[j+i*n] = RNP::Traits<T>::conj(Afac[i+j*m]);
		}
	}
	RNP::LA::LQ::MultQ("L", "C", n, m, n, Afac, m, tau, B, n, &lwork, work);
	// We should recover A'
	if(1){
		T sum = 0;
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				sum += RNP::Traits<T>::abs(A[i+j*m] - RNP::Traits<T>::conj(B[j+i*n]));
			}
		}
		std::cout << "A' norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	
	// Make Q
	T *Q = new T[n*m];
	RNP::BLAS::Copy(n, n, Afac, m, Q, n);
	delete [] work; work = NULL; lwork = 0;
	RNP::LA::LQ::GenerateQ(n, n, n, Q, n, tau, &lwork, work);
	//lwork = m;
	work = new T[lwork];
	RNP::LA::LQ::GenerateQ(n, n, n, Q, n, tau, &lwork, work);
	//RNP::LA::LQ::GenerateQ_unblocked(n, n, n, Q, n, tau, work);
	
	if(0){
		std::cout << "Q:" << std::endl;
		RNP::Matrix<T> mQ(n, n, Q, n);
		std::cout << RNP::IO::Chop(mQ) << std::endl << std::endl;
	}
	
	// Form Q'*Q
	T *QQ = new T[m*n];
	RNP::BLAS::MultMM("C", "N", n, n, n, 1., Q, n, Q, n, 0., QQ, n);
	
	if(0){
		std::cout << "Q' * Q:" << std::endl;
		RNP::Matrix<T> mQQ(n, n, QQ, n);
		std::cout << RNP::IO::Chop(mQQ) << std::endl << std::endl;
	}
	
	// Check to see if we get I
	if(1){
		T sum = 0;
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < n; ++i){
				T delta = (i == j ? 1 : 0);
				sum += RNP::Traits<T>::abs(QQ[i+j*n] - delta);
			}
		}
		std::cout << "Q' * Q - I norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	// Form L*Q in B
	//  Form L in QQ
	RNP::BLAS::Set(m, n, T(0), T(0), QQ, m);
	RNP::LA::Triangular::Copy("L", m, n, Afac, m, QQ, m);
	if(0){
		std::cout << "QQ = L:" << std::endl;
		RNP::Matrix<T> mB(m, n, QQ, m);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	RNP::BLAS::MultMM("N", "N", m, n, n, T(1), QQ, m, Q, n, T(0), B, m);
	
	if(0){
		std::cout << "B = L*Q:" << std::endl;
		RNP::Matrix<T> mB(m, n, B, m);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	// We should recover the original matrix
	if(1){
		T sum = 0;
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				sum += RNP::Traits<T>::abs(A[i+j*m] - B[i+j*m]);
			}
		}
		std::cout << "(A - R*Q) norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	delete [] QQ;
	delete [] Q;
	delete [] B;
	delete [] Afac;
	delete [] A;
	delete [] tau;
	delete [] work;
}

int main(){
	srand(0);
	size_t m = 170; // must be larger than n
	size_t n = 150;
	test_lq<double>(m, n);
	test_lq<std::complex<double> >(m, n);
	return 0;
}
