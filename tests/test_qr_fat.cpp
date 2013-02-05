#include <RNP/LA.hpp>
#include <RNP/BLAS.hpp>
#include <RNP/IOBasic.hpp>
#include <RNP/Random.hpp>
#include <iostream>
#include <cstdlib>

template <typename T>
void test_qr(size_t m, size_t n){
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
	RNP::LA::QR::Factor(m, n, Afac, m, tau, &lwork, work);
	std::cout << "lwork = " << lwork << std::endl;
	work = new T[lwork];
	RNP::LA::QR::Factor(m, n, Afac, m, tau, &lwork, work);
	
	if(0){
		std::cout << "Factored A:" << std::endl;
		RNP::Matrix<T> mA(m, n, Afac, m);
		std::cout << RNP::IO::Chop(mA) << std::endl << std::endl;
	}
	
	// Apply Q to the left of original A (use B for workspace)
	RNP::BLAS::Copy(m, n, A, m, B, m);
	delete [] work; work = NULL; lwork = 0;
	RNP::LA::QR::MultQ("L", "C", m, n, m, Afac, m, tau, B, m, &lwork, work);
	work = new T[lwork];
	RNP::LA::QR::MultQ("L", "C", m, n, m, Afac, m, tau, B, m, &lwork, work);
	
	if(0){
		std::cout << "Q' * origA:" << std::endl;
		RNP::Matrix<T> mB(m, n, B, m);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	// Check to see if the upper triangle is correct
	if(1){
		T sum = 0;
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i <= std::min(j,m-1); ++i){
				sum += RNP::Traits<T>::abs(Afac[i+j*m] - B[i+j*m]);
			}
		}
		std::cout << "R norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	// Apply Q to the left of R
	RNP::BLAS::Set(m, n, T(0), T(0), B, m);
	RNP::LA::Triangular::Copy("U", m, n, Afac, m, B, m);
	if(0){
		std::cout << "B = R:" << std::endl;
		RNP::Matrix<T> mB(m, n, B, m);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	RNP::LA::QR::MultQ("L", "N", m, n, m, Afac, m, tau, B, m, &lwork, work);
	if(0){
		std::cout << "B = Q*R:" << std::endl;
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
		std::cout << "(A - Q*R) norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
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
	RNP::LA::QR::MultQ("R", "N", n, m, m, Afac, m, tau, B, n, &lwork, work);
	if(0){
		std::cout << "B = R':" << std::endl;
		RNP::Matrix<T> mB(n, m, B, n);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	// We should recover R'
	if(1){
		T sum = 0;
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < std::min(j+1,m); ++i){
				sum += RNP::Traits<T>::abs(Afac[i+j*m] - RNP::Traits<T>::conj(B[j+i*n]));
			}
		}
		std::cout << "R' norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
	}
	
	// Now set B = R', and apply Q' from the right to get A'
	RNP::BLAS::Set(n, m, T(0), T(0), B, n);
	for(size_t j = 0; j < m; ++j){
		for(size_t i = j; i < n; ++i){
			B[i+j*n] = RNP::Traits<T>::conj(Afac[j+i*m]);
		}
	}
	RNP::LA::QR::MultQ("R", "C", n, m, m, Afac, m, tau, B, n, &lwork, work);
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
	RNP::BLAS::Copy(m, m, Afac, m, Q, m);
	delete [] work; work = NULL; lwork = 0;
	RNP::LA::QR::GenerateQ(m, m, m, Q, m, tau, &lwork, work);
	//lwork = n;
	work = new T[lwork];
	RNP::LA::QR::GenerateQ(m, m, m, Q, m, tau, &lwork, work);
	
	if(0){
		std::cout << "Q:" << std::endl;
		RNP::Matrix<T> mQ(m, m, Q, m);
		std::cout << RNP::IO::Chop(mQ) << std::endl << std::endl;
	}
	
	// Form Q'*Q
	T *QQ = new T[m*n];
	RNP::BLAS::MultMM("C", "N", m, m, m, 1., Q, m, Q, m, 0., QQ, m);
	
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
	
	// Form Q*R
	//  Put R in QQ for now
	RNP::BLAS::Set(m, n, T(0), T(0), QQ, m);
	RNP::LA::Triangular::Copy("U", m, n, Afac, m, QQ, m);
	if(0){
		std::cout << "QQ = R:" << std::endl;
		RNP::Matrix<T> mB(m, n, QQ, m);
		std::cout << RNP::IO::Chop(mB) << std::endl << std::endl;
	}
	RNP::BLAS::MultMM("N", "N", m, n, m, T(1), Q, m, QQ, m, T(0), B, m);
	if(0){
		std::cout << "B = Q*R:" << std::endl;
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
		std::cout << "(A - Q*R) norm-1 error: " << std::abs(sum)*rsnrm << std::endl;
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
	size_t m = 166;
	size_t n = 170; // must be larger than m
	test_qr<double>(m, n);
	test_qr<std::complex<double> >(m, n);
	return 0;
}
