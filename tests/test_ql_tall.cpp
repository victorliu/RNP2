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

extern "C" void dgeql2_(const int &m, const int &n, double *a, const int &lda, double *tau, double *work, int *info);
extern "C" void dorm2l_(
	const char *side, const char *trans,
	const int &m, const int &n, const int &k, double *a, const int &lda, double *tau,
	double *c, const int &ldc, double *work, int *info);

template <typename T>
void test_ql(size_t m, size_t n){
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
	
	RNP::LA::QL::Factor(m, n, Afac, m, tau, &lwork, work);
	std::cout << "lwork = " << lwork << std::endl;
	work = new T[lwork];
	RNP::LA::QL::Factor(m, n, Afac, m, tau, &lwork, work);

	if(0){
		std::cout << "Factored A:" << std::endl;
		RNP::Matrix<T> mA(m, n, Afac, m);
		std::cout << RNP::IO::Chop(mA) << std::endl << std::endl;
	}
	
	// Apply Q' to the left of original A (use B for workspace)
	RNP::BLAS::Copy(m, n, A, m, B, m);
	delete [] work; work = NULL; lwork = 0;
	RNP::LA::QL::MultQ("L", "C", m, n, n, Afac, m, tau, B, m, &lwork, work);
	work = new T[lwork];
	RNP::LA::QL::MultQ("L", "C", m, n, n, Afac, m, tau, B, m, &lwork, work);
	
	if(0){
		std::cout << "Q' * origA:" << std::endl;
		RNP::Matrix<T> mB(m, n, B, m);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	// Check to see if the lower triangle is correct
	if(1){
		T sum = 0;
		for(size_t j = 0; j < n; ++j){
			for(size_t i = m-n+j; i < m; ++i){
				sum += RNP::Traits<T>::abs(Afac[i+j*m] - B[i+j*m]);
			}
		}
		std::cout << "L norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	// Apply Q to the left of L
	RNP::BLAS::Set(m, n, T(0), T(0), B, m);
	RNP::LA::Triangular::Copy("L", n, n, &Afac[m-n+0*m], m, &B[m-n+0*m], m);
	if(0){
		std::cout << "B = L:" << std::endl;
		RNP::Matrix<T> mB(m, n, B, m);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	RNP::LA::QL::MultQ("L", "N", m, n, n, Afac, m, tau, B, m, &lwork, work);
	
	if(0){
		std::cout << "B = Q*L:" << std::endl;
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
		std::cout << "(A - Q*L) norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	// Now treat B as a n-by-m matrix, and copy A' into it,
	// and apply Q from the right
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
	RNP::LA::QL::MultQ("R", "N", n, m, n, Afac, m, tau, B, n, &lwork, work);
	if(0){
		std::cout << "B = L':" << std::endl;
		RNP::Matrix<T> mB(n, m, B, n);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	// We should recover L'
	if(1){
		T sum = 0;
		for(size_t j = 0; j < n; ++j){
			for(size_t i = m-n+j; i < m; ++i){
				sum += RNP::Traits<T>::abs(Afac[i+j*m] - RNP::Traits<T>::conj(B[j+i*n]));
			}
		}
		std::cout << "L' norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	// Now set B = L', and apply Q' from the right to get A'
	RNP::BLAS::Set(n, m, T(0), T(0), B, n);
	for(size_t j = 0; j < n; ++j){
		for(size_t i = m-n+j; i < m; ++i){
			B[j+i*n] = RNP::Traits<T>::conj(Afac[i+j*m]);
		}
	}
	RNP::LA::QL::MultQ("R", "C", n, m, n, Afac, m, tau, B, n, &lwork, work);
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
	T *Q = new T[m*m];
	RNP::BLAS::Copy(m, n, Afac, m, Q, m);
	delete [] work; work = NULL; lwork = 0;
	
	RNP::LA::QL::GenerateQ(m, n, n, Q, m, tau, &lwork, work);
	//lwork = n;
	work = new T[lwork];
	RNP::LA::QL::GenerateQ(m, n, n, Q, m, tau, &lwork, work);
	
	if(0){
		std::cout << "Q:" << std::endl;
		RNP::Matrix<T> mQ(m, m, Q, m);
		std::cout << RNP::IO::Chop(mQ) << std::endl << std::endl;
	}
	
	// Form Q'*Q
	T *QQ = new T[m*m];
	RNP::BLAS::MultMM("C", "N", n, n, m, 1., Q, m, Q, m, 0., QQ, n);
	
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
	
	// Form Q*L
	//  Put L in QQ for now
	RNP::BLAS::Set(n, n, T(0), T(0), QQ, n);
	RNP::LA::Triangular::Copy("L", n, n, &Afac[m-n+0*m], m, QQ, n);
	if(0){
		std::cout << "QQ = L:" << std::endl;
		RNP::Matrix<T> mB(n, n, QQ, n);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	RNP::BLAS::MultMM("N", "N", m, n, n, T(1), Q, m, QQ, n, T(0), B, m);
	if(0){
		std::cout << "B = Q*L:" << std::endl;
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
		std::cout << "(A - Q*L) norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	// Generate the columns of Q corresponding to the nullspace of A'
	RNP::BLAS::Set(m, m, T(0), T(1), Q, m);
	RNP::LA::QL::MultQ("L", "N", m, m, n, Afac, m, tau, Q, m, &lwork, work);
	
	RNP::BLAS::MultMM("C", "N", n, m-n, m, T(1), A, m, &Q[0+0*m], m, T(0), QQ, n);
	// Check to see if we get 0
	if(1){
		T sum = 0;
		for(size_t j = 0; j < m-n; ++j){
			for(size_t i = 0; i < n; ++i){
				sum += RNP::Traits<T>::abs(QQ[i+j*n]);
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
	size_t m = 170; // must be larger than n
	size_t n = 150;
	test_ql<double>(m, n);
	test_ql<std::complex<double> >(m, n);
	return 0;
}
