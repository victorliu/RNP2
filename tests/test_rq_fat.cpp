#include <complex>
extern "C" void dlarfb_(
	const char *side, const char *trans, const char *direc, const char *storev,
	const int &m, const int &n, const int &k, double *v, const int &ldv,
	double *t, const int &ldt,
	double *c, const int &ldc, double *work, const int &ldwork);
extern "C" void dlarft_(
	const char *direc, const char *storev,
	const int &m, const int &n, double *a, const int &lda,
	const double *tau, double *work, const int &ldwork);
extern "C" void dlarf_(const char *side,
	const int &m, const int &n, const double *v, const int &ldv, const double &tau,
	double *c, const int &ldc, double* work);
extern "C" void zlarf_(const char *side,
	const int &m, const int &n, const std::complex<double> *v, const int &ldv, const std::complex<double> &tau,
	std::complex<double> *c, const int &ldc, std::complex<double>* work);

#include <RNP/LA.hpp>
#include <RNP/BLAS.hpp>
#include <RNP/IOBasic.hpp>
#include <RNP/Random.hpp>
#include <iostream>
#include <cstdlib>

extern "C" void zgerq2_(const int &m, const int &n, std::complex<double> *a, const int &lda, std::complex<double> *tau, std::complex<double> *work, int *info);
extern "C" void dgerq2_(const int &m, const int &n, double *a, const int &lda, double *tau, double *work, int *info);
extern "C" void dormr2_(
	const char *side, const char *trans,
	const int &m, const int &n, const int &k, double *a, const int &lda, double *tau,
	double *c, const int &ldc, double *work, int *info);
extern "C" void zunmr2_(
	const char *side, const char *trans,
	const int &m, const int &n, const int &k, std::complex<double> *a, const int &lda, std::complex<double> *tau,
	std::complex<double> *c, const int &ldc, std::complex<double> *work, int *info);

template <typename T>
void test_rq(size_t m, size_t n){
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
	
	RNP::LA::RQ::Factor(m, n, Afac, m, tau, &lwork, work);
	//lwork = m;
	std::cout << "lwork = " << lwork << std::endl;
	work = new T[lwork];
	RNP::LA::RQ::Factor(m, n, Afac, m, tau, &lwork, work);
	//RNP::LA::RQ::Factor_unblocked(m, n, Afac, m, tau, work);
	//int info; zgerq2_(m, n, Afac, m, tau, work, &info);
	//int info;
	if(0){
		std::cout << "Factored A:" << std::endl;
		RNP::Matrix<T> mA(m, n, Afac, m);
		std::cout << RNP::IO::Chop(mA) << std::endl << std::endl;
	}
	
	// Apply Q' to the right of original A (use B for workspace)
	RNP::BLAS::Copy(m, n, A, m, B, m);
	delete [] work; work = NULL; lwork = 0;
	RNP::LA::RQ::MultQ("R", "C", m, n, m, Afac, m, tau, B, m, &lwork, work);
	//lwork = m;
	work = new T[lwork];
	RNP::LA::RQ::MultQ("R", "C", m, n, m, Afac, m, tau, B, m, &lwork, work);
	//RNP::LA::RQ::MultQ_unblocked("R", "C", m, n, n, &Afac[m-n+0*m], m, tau, B, m, work);
	//dormr2_("R", "T", m, n, n, &Afac[m-n+0*m], m, tau, B, m, work, &info);
	//zunmr2_("R", "C", m, n, n, &Afac[m-n+0*m], m, tau, B, m, work, &info);
	
	if(0){
		std::cout << "Q' * origA:" << std::endl;
		RNP::Matrix<T> mB(m, n, B, m);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	// Check to see if the upper trapezoid is correct
	if(1){
		T sum = 0;
		for(size_t j = n-m; j < n; ++j){
			size_t i;
			for(i = 0; i <= j-(n-m); ++i){
				sum += RNP::Traits<T>::abs(Afac[i+j*m] - B[i+j*m]);
			}
			for(; i < m; ++i){ // check for zero lower triangle
				sum += RNP::Traits<T>::abs(B[i+j*m]);
			}
		}
		std::cout << "R norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	// Apply Q to the right of R
	RNP::BLAS::Set(m, n, T(0), T(0), B, m);
	RNP::LA::Triangular::Copy("U", m, m, &Afac[0+(n-m)*m], m, &B[0+(n-m)*m], m);
	if(0){
		std::cout << "B = R:" << std::endl;
		RNP::Matrix<T> mB(m, n, B, m);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	RNP::LA::RQ::MultQ("R", "N", m, n, m, Afac, m, tau, B, m, &lwork, work);
	
	if(0){
		std::cout << "B = R*Q:" << std::endl;
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
	RNP::LA::RQ::MultQ("L", "N", n, m, m, Afac, m, tau, B, n, &lwork, work);
	if(0){
		std::cout << "B = R':" << std::endl;
		RNP::Matrix<T> mB(n, m, B, n);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	// We should recover R'
	if(1){
		T sum = 0;
		for(size_t j = n-m; j < n; ++j){
			for(size_t i = 0; i <= j-(n-m); ++i){
				sum += RNP::Traits<T>::abs(Afac[i+j*m] - RNP::Traits<T>::conj(B[j+i*n]));
			}
		}
		std::cout << "R' norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	// Now set B = L', and apply Q' from the left to get A'
	RNP::BLAS::Set(n, m, T(0), T(0), B, n);
	for(size_t j = n-m; j < n; ++j){
		for(size_t i = 0; i <= j-(n-m); ++i){
			B[j+i*n] = RNP::Traits<T>::conj(Afac[i+j*m]);
		}
	}
	RNP::LA::RQ::MultQ("L", "C", n, m, m, Afac, m, tau, B, n, &lwork, work);
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
	T *Q = new T[n*n];
	RNP::BLAS::Copy(m, n, Afac, m, Q, m);
	delete [] work; work = NULL; lwork = 0;
	
	RNP::LA::RQ::GenerateQ(m, n, m, Q, m, tau, &lwork, work);
	//lwork = n;
	work = new T[lwork];
	RNP::LA::RQ::GenerateQ(m, n, m, Q, m, tau, &lwork, work);
	//RNP::LA::RQ::GenerateQ_unblocked(m, n, m, Q, m, tau, work);
	
	if(0){
		std::cout << "Q:" << std::endl;
		RNP::Matrix<T> mQ(m, n, Q, m);
		std::cout << RNP::IO::Chop(mQ) << std::endl << std::endl;
	}
	
	// Form Q'*Q
	T *QQ = new T[n*n];
	RNP::BLAS::MultMM("N", "C", m, m, n, 1., Q, m, Q, m, 0., QQ, m);
	
	if(0){
		std::cout << "Q' * Q:" << std::endl;
		RNP::Matrix<T> mQQ(m, m, QQ, m);
		std::cout << RNP::IO::Chop(mQQ) << std::endl << std::endl;
	}
	// Check to see if we get I
	if(1){
		T sum = 0;
		for(size_t j = 0; j < m; ++j){
			for(size_t i = 0; i < m; ++i){
				T delta = (i == j ? 1 : 0);
				sum += RNP::Traits<T>::abs(QQ[i+j*m] - delta);
			}
		}
		std::cout << "Q' * Q - I norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	// Form R*Q in B
	//  Form R in QQ
	RNP::BLAS::Set(m, m, T(0), T(0), QQ, m);
	RNP::LA::Triangular::Copy("U", m, m, &Afac[0+(n-m)*m], m, QQ, m);
	if(0){
		std::cout << "QQ = R:" << std::endl;
		RNP::Matrix<T> mB(m, m, QQ, m);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	RNP::BLAS::MultMM("N", "N", m, n, m, T(1), QQ, m, Q, m, T(0), B, m);
	
	if(0){
		std::cout << "B = R*Q:" << std::endl;
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
	
	// Generate the rows of Q corresponding to the nullspace of A
	RNP::BLAS::Set(n, n, T(0), T(1), Q, n);
	RNP::LA::RQ::MultQ("L", "N", n, n, m, Afac, m, tau, Q, n, &lwork, work);
	if(0){
		std::cout << "Q:" << std::endl;
		RNP::Matrix<T> mQ(n, n, Q, n);
		std::cout << RNP::IO::Chop(mQ) << std::endl << std::endl;
	}
	// The first n-m rows of Q span the nullspace of A
	RNP::BLAS::MultMM("N", "C", m, n-m, n, T(1), A, m, Q, n, T(0), QQ, m);
	
	if(1){
		T sum = 0;
		for(size_t j = 0; j < n-m; ++j){
			for(size_t i = 0; i < m; ++i){
				sum += RNP::Traits<T>::abs(QQ[i+j*m]);
			}
		}
		std::cout << "A*Qn norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
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
	size_t m = 150;
	size_t n = 170; // must be larger than m
	test_rq<double>(m, n);
	test_rq<std::complex<double> >(m, n);
	return 0;
}
