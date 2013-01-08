#ifndef MATRIX_GEN_HPP_INCLUDED
#define MATRIX_GEN_HPP_INCLUDED

#include <RNP/Types.hpp>
#include <RNP/BLAS.hpp>
#include <RNP/Random.hpp>

namespace RNP{
namespace MatrixGen{

namespace DiagonalMode{
enum Mode{
	Given,
	First1RemainingCond,
	LastCondRemaining1,
	GradedExponentially,
	GradedLinearly,
	RandomExponential,
	RandomDist
};
}

// mode:
//   0: d is given as input
//   1: d[0] is 1, remaining are 1/cond
//   2: d[last] = 1/cond, remaining are 1
//   3: d[i] = cond^{-t}, t = i/(n-1)
//   4: d[i] = (1-t)*1 - t*1/cond), t defined above
//   5: d[i] is random in range [1,1/cond], logs are uniformly dist
//   6: d[i] is determined by dist
template <typename T> // T must be a real type
void GenerateDiagonal(
	Random::Distribution::Distribution dist,
	DiagonalMode::Mode mode,
	bool reverse, bool randsign,
	const T &cond, // cond >= 1
	const T &dscale, // if d is not Given and not RandomDist, d is scaled to have maximum dscale
	size_t n, T *d,
	int iseed[4] = NULL
){
	typedef typename Traits<T>::real_type real_type;
	
	if(0 == n){ return; }
	const T rcond = T(1)/cond;
	
	switch(mode){
	case DiagonalMode::First1RemainingCond:
		d[0] = T(1);
		for(size_t i = 1; i < n; ++i){
			d[i] = rcond;
		}
		break;
	case DiagonalMode::LastCondRemaining1:
		for(size_t i = 0; i+1 < n; ++i){
			d[i] = T(1);
		}
		d[n-1] = rcond;
		break;
	case DiagonalMode::GradedExponentially:
		for(size_t i = 0; i < n; ++i){
			T t = T(i) / T(n-1);
			d[i] = pow(rcond, t);
		}
		break;
	case DiagonalMode::GradedLinearly:
		for(size_t i = 0; i < n; ++i){
			T t = T(i) / T(n-1);
			T t1 = T(1) - t;
			d[i] = t1 + t*rcond;
		}
		break;
	case DiagonalMode::RandomExponential:
		Random::GenerateVector(Random::Distribution::Uniform01, n, d, iseed);
		for(size_t i = 0; i < n; ++i){
			d[i] = pow(rcond, d[i]);
		}
		break;
	case DiagonalMode::RandomDist:
		Random::GenerateVector(dist, n, d, iseed);
		break;
	case DiagonalMode::Given:
		// d specified as input
		break;
	default: break;
	}
	if(reverse){ // reverse d if necessary
		for(size_t i = 0; i < n; ++i){
			T t = d[i];
			d[i] = d[n-1-i];
			d[n-1-i] = t;
		}
	}
	if(randsign){
		for(size_t i = 0; i < n; ++i){
			if(T(2)*Random::UniformReal<T>(iseed) > T(1)){
				d[i] = -d[i];
			}
		}
	}
	// scale to dscale
	if(DiagonalMode::Given != mode && DiagonalMode::RandomDist != mode){
		real_type maxabs = RNP::Traits<T>::abs(d[0]);
		for(size_t i = 1; i < n; ++i){
			real_type curabs = RNP::Traits<T>::abs(d[i]);
			if(curabs > maxabs){ maxabs = curabs; }
		}
		BLAS::Scale(n, dscale / maxabs, d, 1);
	}
}

namespace Symmetry{
enum Sym{
	Nonsymmetric,
	Hermitian,
	HermitianPositiveDefinite,
	Symmetric
};
}

template <typename T>
void RandomMatrix(
	Symmetry::Sym sym,
	Random::Distribution::Distribution dist,
	DiagonalMode::Mode mode,
	bool reverse,
	const typename Traits<T>::real_type &cond, // cond >= 1
	const typename Traits<T>::real_type &dmax,
	typename Traits<T>::real_type *d_,
	size_t m, size_t n, T *a, size_t lda,
	int iseed[4] = NULL,
	T *work_ = NULL // length m+n
){
	typedef typename Traits<T>::real_type real_type;
	
	RNPAssert(!(DiagonalMode::Given == mode && NULL == d_));
	RNPAssert(Symmetry::Nonsymmetric == sym || m == n);

	const size_t mindim = (m < n ? m : n);
	if(0 == mindim){ return; }
	
	real_type *d = d_;
	if(NULL == d_){
		d = new real_type[mindim];
	}
	T *work = work_;
	if(NULL == work_){
		work = new T[m+n];
	}
	
	const bool randsign = (Symmetry::Hermitian == sym);
	GenerateDiagonal(dist, mode, reverse, randsign, cond, dmax, mindim, d, iseed);

	// Set A to diagonal
	for(size_t j = 0; j < n; ++j){
		for(size_t i = 0; i < m; ++i){
			if(i == j){
				a[i+j*lda] = d[i];
			}else{
				a[i+j*lda] = T(0);
			}
		}
	}
	if(NULL == d_){
		delete [] d;
	}
	
	// Apply random Householder reflectors from either side
	if(Symmetry::Nonsymmetric == sym){
		size_t i = mindim;
		while(i --> 0){
			if(i+1 < m){ // reflect on left
				// Generate the reflection
				Random::GenerateVector(Random::Distribution::Normal01, m-i, work, iseed);
				real_type wn = BLAS::Norm2(m-i, work, 1);
				T wa = (wn / Traits<real_type>::abs(work[0])) * work[0];
				T tau(0);
				if(real_type(0) != wn){
					T wb = work[0] + wa;
					BLAS::Scale(m-i-1, real_type(1)/wb, &work[1], 1);
					work[0] = T(1);
					tau = Traits<T>::real(wb / wa);
				}
				// Apply from left
				BLAS::MultMV("C", m-i, n-i, T(1), &a[i+i*lda], lda, work, 1, T(0), &work[m], 1);
				BLAS::ConjugateRank1Update(m-i, n-i, -tau, work, 1, &work[m], 1, &a[i+i*lda], lda);
			}
			if(i+1 < n){ // reflect on right
				// Generate the reflection
				Random::GenerateVector(Random::Distribution::Normal01, n-i, work, iseed);
				real_type wn = BLAS::Norm2(n-i, work, 1);
				T wa = (wn / Traits<real_type>::abs(work[0])) * work[0];
				T tau(0);
				if(real_type(0) != wn){
					T wb = work[0] + wa;
					BLAS::Scale(n-i-1, real_type(1)/wb, &work[1], 1);
					work[0] = T(1);
					tau = Traits<T>::real(wb / wa);
				}
				// Apply from left
				BLAS::MultMV("N", m-i, n-i, T(1), &a[i+i*lda], lda, work, 1, T(0), &work[n], 1);
				BLAS::ConjugateRank1Update(m-i, n-i, -tau, &work[n], 1, work, 1, &a[i+i*lda], lda);
			}
		}
	}else if(Symmetry::Hermitian == sym || Symmetry::HermitianPositiveDefinite == sym){
		size_t i = n;
		while(i --> 0){
			// Generate the reflection
			Random::GenerateVector(Random::Distribution::Normal01, n-i, work, iseed);
			real_type wn = BLAS::Norm2(n-i, work, 1);
			T wa = (wn / Traits<real_type>::abs(work[0])) * work[0];
			T tau(0);
			if(real_type(0) != wn){
				T wb = work[0] + wa;
				BLAS::Scale(n-i-1, real_type(1)/wb, &work[1], 1);
				work[0] = T(1);
				tau = Traits<T>::real(wb / wa);
			}
			// Apply from left and right
			// Compute  y := tau * A * u
			BLAS::MultHermV("L", n-i, tau, &a[i+i*lda], lda, work, 1, T(0), &work[n], 1);
			// Compute  v := y - 1/2 * tau * ( y, u ) * u
			const T alpha = -(real_type(1)/real_type(2))*tau*BLAS::ConjugateDot(n-i, &work[n], 1, work, 1);
			BLAS::Axpy(n-i, alpha, work, 1, &work[n], 1);
			// Apply the transformation as a rank-2 update to A(i:n,i:n)
			BLAS::HermRank2Update("L", n-i, T(-1), work, 1, &work[n], 1, &a[i+i*lda], lda);
		}
		// Generate upper triangle
		for(size_t j = 0; j < n; ++j){
			for(i = j+1; i < n; ++i){
				a[j+i*lda] = Traits<T>::conj(a[i+j*lda]);
			}
		}
	}else if(Symmetry::Symmetric == sym){
		size_t i = n;
		while(i --> 0){
			// Generate the reflection
			Random::GenerateVector(Random::Distribution::Normal01, n-i, work, iseed);
			real_type wn = BLAS::Norm2(n-i, work, 1);
			T wa = (wn / Traits<real_type>::abs(work[0])) * work[0];
			T tau(0);
			if(real_type(0) != wn){
				T wb = work[0] + wa;
				BLAS::Scale(n-i-1, real_type(1)/wb, &work[1], 1);
				work[0] = T(1);
				tau = Traits<T>::real(wb / wa);
			}
			// Apply from left and right
			// Compute  y := tau * A * u
			BLAS::MultSymV("L", n-i, tau, &a[i+i*lda], lda, work, 1, T(0), &work[n], 1);
			// Compute  v := y - 1/2 * tau * ( y, u ) * u
			const T alpha = -(real_type(1)/real_type(2))*tau*BLAS::ConjugateDot(n-i, &work[n], 1, work, 1);
			BLAS::Axpy(n-i, alpha, work, 1, &work[n], 1);
			// Apply the transformation as a rank-2 update to A(i:n,i:n)
			BLAS::SymRank2Update("L", n-i, T(-1), work, 1, &work[n], 1, &a[i+i*lda], lda);
		}
		// Generate upper triangle
		for(size_t j = 0; j < n; ++j){
			for(i = j+1; i < n; ++i){
				a[j+i*lda] = a[i+j*lda];
			}
		}
	}
	
	if(NULL == work_){
		delete [] work;
	}
}

} // namespace MatrixGen
} // namespace RNP

#endif // MATRIX_GEN_HPP_INCLUDED
