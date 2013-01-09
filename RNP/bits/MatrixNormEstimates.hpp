#ifndef RNP_MATRIX_NORM_ESTIMATES_HPP_INCLUDED
#define RNP_MATRIX_NORM_ESTIMATES_HPP_INCLUDED

#include <cstddef>
#include <RNP/BLAS.hpp>
#include <RNP/Random.hpp>
#include <RNP/Sort.hpp>

namespace RNP{
namespace LA{

// Estimates the 1-norm of a square matrix
// Aop must perform:
//   x <- A   * x          when 'N' == trans[0]
//   x <- A^H * x          when 'C' == trans[0]
template <typename T> // _lacon
typename Traits<T>::real_type MatrixNorm1Estimate_unblocked(
	size_t n,
	void (*Aop)(const char *trans, size_t n, T *x, void *data),
	void *data,
	T *work // length 2*n
){
	static const size_t itmax = 5;
	typedef typename Traits<T>::real_type real_type;
	const real_type safmin = Traits<real_type>::min();
	
	// Set up workspaces
	T *v = work;
	T *x = v + n;
	
	for(size_t i = 0; i < n; ++i){
		x[i] = real_type(1) / real_type(n);
	}
	Aop("N", n, x, data);
	if(1 == n){
		return Traits<T>::abs(x[0]);
	}
	real_type est = BLAS::Norm1(n, x, 1);
	for(size_t i = 0; i < n; ++i){
		real_type absxi = Traits<T>::abs(x[i]);
		if(absxi > safmin){
			x[i] /= absxi;
		}else{
			x[i] = T(1);
		}
	}
	Aop("C", n, x, data);
	size_t imax = 0;
	{ // compute index of maximum absolute value of real part
		real_type armax(0);
		for(size_t i = 0; i < n; ++i){
			real_type ar = Traits<real_type>::abs(Traits<T>::real(x[i]));
			if(ar > armax){
				armax = ar;
				imax = i;
			}
		}
	}
	size_t iter = 1;
	size_t jlast;
	do{
		++iter;
		for(size_t i = 0; i < n; ++i){ x[i] = T(0); }
		x[imax] = T(1);
		Aop("N", n, x, data);
		
		BLAS::Copy(n, x, 1, v, 1);
		real_type estold = est;
		est = BLAS::Norm1(n, v, 1);
		if(est <= estold){
			break;
		}
		for(size_t i = 0; i < n; ++i){
			real_type absxi = Traits<T>::abs(x[i]);
			if(absxi > safmin){
				x[i] /= absxi;
			}else{
				x[i] = T(1);
			}
		}
		Aop("C", n, x, data);
		jlast = imax;
		imax = 0;
		{ // compute index of maximum absolute value of real part
			real_type armax(0);
			for(size_t i = 0; i < n; ++i){
				real_type ar = Traits<real_type>::abs(Traits<T>::real(x[i]));
				if(ar > armax){
					armax = ar;
					imax = i;
				}
			}
		}
	}while(Traits<T>::abs(x[jlast]) != Traits<T>::abs(x[imax]) && iter < itmax);
	real_type altsgn(1);
	for(size_t i = 0; i < n; ++i){
		x[i] = altsgn*( real_type(1) + real_type(i) / real_type(n-1) );
		altsgn = -altsgn;
	}
	Aop("N", n, x, data);
	real_type temp(real_type(2) * BLAS::Norm1(n, x, 1) / real_type(3*n));
	if(temp >= est){
		BLAS::Copy(n, x, 1, v, 1);
		est = temp;
	}
	return est;
}

namespace Util{

template <typename T>
void ReplaceColumns(
	size_t n,
	size_t t,
	T *x, size_t ldx, const T *xold, size_t ldxold, // xold can be NULL
	int iseed[4] = NULL
){
	typedef typename Traits<T>::real_type real_type;
	// Purpose
	// =======
	//
	// Looks for and replaces columns of x which are parallel to
	// columns of xold and itself. If xold == NULL, then x is
	// only checked against itself.
	//
	// Arguments
	// =========
	//
	// N      (input) INTEGER
	//        The number of rows.  N >= 1.
	//
	// T      (input) INTEGER
	//        The number of columns of X and XOLD
	//
	// X      (input/output) DOUBLE PRECISION array, dimension (N,T)
	//        On return, X will have full column rank.
	//
	// LDX    (input) INTEGER
	//        The leading dimension of X.  LDX >= max(1,N).
	//
	// XOLD   (input/output) DOUBLE PRECISION array, dimension (N,T)
	//
	// LDXOLD (input) INTEGER
	//        The leading dimension of XOLD.  LDXOLD >= max(1,N).
	//
	// ISEED  (input/output) INTEGER array, dimension (4)
	//         On entry, the seed of the random number generator; the array
	//         elements must be between 0 and 4095, and ISEED(4) must be
	//         odd.
	//         On exit, the seed is updated.

	for(size_t j = 1; j < t; ++j){
		size_t ngen = 0;
		do{
			// Compute inner product of j-th column of x with xold[0..t], and x[1..j]
			const real_type thres(real_type(n) - (real_type(1)/real_type(2)));
			bool is_parallel = true;
			if(NULL != xold){
				for(size_t k = 0; k < t; ++k){ // check against xold[0..t]
					real_type overlap(Traits<T>::abs(
						BLAS::ConjugateDot(n, &x[0+j*ldx], 1, &xold[0+k*ldxold], 1)
					));
					if(overlap < thres){
						is_parallel = false;
						break;
					}
				}
				if(!is_parallel){ break; }
			}
			for(size_t k = 1; k < j; ++k){ // check against x[1..j]
				real_type overlap(Traits<T>::abs(
					BLAS::ConjugateDot(n, &x[0+j*ldx], 1, &x[0+k*ldx], 1)
				));
				if(overlap < thres){
					is_parallel = false;
					break;
				}
			}
			if(!is_parallel){ break; }
			++ngen;
			Random::GenerateVector(Random::Distribution::UnitCircle, n, &x[0+j*ldx], iseed);
		}while(ngen < n/t);
	}
}

} // namespace Util

template <typename T>
int MatrixNorm1Estimate(
	size_t n,
	size_t t,
	void (*Aop)(
		const char *trans, size_t n, size_t nb, T *x, size_t ldx, void *data
	),
	void *data,
	typename Traits<T>::real_type *est, // returned estimate
	T *x, size_t ldx, T *xold, size_t ldxold, // xold can be NULL
	typename Traits<T>::real_type *h, size_t *ind, size_t *indh,
	int iseed[4] = NULL
){
	typedef typename Traits<T>::real_type real_type;
	//  Purpose
	//  =======
	//
	//  Estimates the 1-norm of a square, real matrix A.
	//
	//  Arguments
	//  =========
	//
	//  N      (input) INTEGER
	//         The order of the matrix.  N >= 1.
	//
	//  T      (input) INTEGER
	//         The number of columns used at each step.
	//
	//  V      (output) COMPLEX*16 array, dimension (N).
	//         On the final return, V = A*W,  where  EST = norm(V)/norm(W)
	//         (W is not returned).
	//
	//  X      (input/output) COMPLEX*16 array, dimension (N,T)
	//         On an intermediate return, X should be overwritten by
	//               A * X,   if KASE=1,
	//               A' * X,  if KASE=2,
	//         and ZLACN1 must be re-called with all the other parameters
	//         unchanged.
	//
	//  LDX    (input) INTEGER
	//         The leading dimension of X.  LDX >= max(1,N).
	//
	//  XOLD   (workspace) COMPLEX*16 array, dimension (N,T)
	//
	//  LDXOLD (input) INTEGER
	//         The leading dimension of XOLD.  LDXOLD >= max(1,N).
	//
	//  H      (workspace) DOUBLE PRECISION array, dimension (N)
	//
	//  IND    (workspace) INTEGER array, dimension (N)
	//
	//  INDH   (workspace) INTEGER array, dimension (N)
	//
	//  EST    (output) DOUBLE PRECISION
	//         An estimate (a lower bound) for norm(A).
	//
	//  ISEED  (input/output) INTEGER array, dimension (4)
	//          On entry, the seed of the random number generator; the array
	//          elements must be between 0 and 4095, and ISEED(4) must be
	//          odd.
	//          On exit, the seed is updated.
	//
	//  Return: 0: Normal termination
	//          1: iteration limit reached. (possibly inaccurate estimate)
	//          2: estimate not increased.
	//          3: repeated sign matrix (good estimate)
	//          4: power method convergence test (exact estimate)
	//          5: repeated unit vectors.
	const size_t itmax = 5;

	const real_type safmin = Traits<T>::min();

	real_type estold(0);
	size_t iter = 0;
	size_t itemp = 0;
	int info = 0;

	for(size_t i = 0; i < n; ++i){
		x[i+0*ldx] = T(1);
		ind[i]  = i;
		indh[i] = 0;
	}

	for(size_t j = 1; j < t; ++j){
		Random::GenerateVector(Random::Distribution::UnitCircle, n, &x[0+j*ldx], iseed);
	}
	
	if(t > 1 && !Traits<T>::is_complex()){
		Util::ReplaceColumns(n, t, x, ldx, xold, ldxold, iseed);
	}
	
	BLAS::Rescale("G", 0, 0, real_type(n), real_type(1), n, t, x, ldx);
	Aop("N", n, t, x, ldx, data);

	size_t ibest = 0; // only really initialized when iter >= 1
	do{
		if(0 == iter && 1 == n){
			//v[0] = x[0];
			*est = Traits<T>::abs(x[0]);
			return 0;
		}
		*est = real_type(0);
		for(size_t j = 0; j < t; ++j){
			real_type temp = BLAS::Norm1(n, &x[0+j*ldx], 1);
			if(temp > *est){
				*est = temp;
				itemp = j;
			}
		}

		if(*est > estold  || 1 == iter){
			ibest = ind[itemp];
		}
		if(*est <= estold && iter >= 1){
			*est = estold;
			return 2;
		}
		estold = *est;
		//BLAS::Copy(n, &x[0+itemp*ldx], 1, v, 1);

		if(iter >= itmax){ return 1; }

		// Computing the sign matrix 
		for(size_t j = 0; j < t; ++j){
			for(size_t i = 0; i < n; ++i){
				real_type absxij = Traits<T>::abs(x[i+j*ldx]);
				if(absxij > safmin){
					x[i+j*ldx] /= absxij;
				}else{
					x[i+j*ldx] = T(1);
				}
			}
		}
		if(!Traits<T>::is_complex()){
			if(iter > 0 && NULL != xold){
				// If all columns of x parallel to xold, exit.
				// This is determined by X'*Xold having at least one value of n in every column
				// We check against n-0.5 just to be safe
				bool all_parallel = true;
				const real_type thres(real_type(n) - (real_type(1)/real_type(2)));
				for(size_t j = 0; j < t; ++j){
					for(size_t k = 0; k < t; ++k){
						real_type overlap(Traits<T>::abs(
							BLAS::ConjugateDot(n, &xold[0+k*ldxold], 1, &x[0+j*ldx], 1)
						));
						if(overlap < thres){
							all_parallel = false;
							break;
						}
					}
					if(!all_parallel){ break; }
				}
				if(all_parallel){ return 3; }
			}
			if(t > 1){
				Util::ReplaceColumns(n, t, x, ldx, (iter > 0 ? xold : NULL), ldxold, iseed);
			}
			if(NULL != xold){
				BLAS::Copy(n, t, x, ldx, xold, ldxold);
			}
		}

		Aop("N", n, t, x, ldx, data);
		for(size_t i = 0; i < n; ++i){
			// Determine element of largest magnitude on the i-th row
			size_t jmax = 0;
			real_type jmaxv = 0;
			for(size_t j = 0; j < t; ++j){
				real_type val = Traits<T>::abs(x[i+j*ldx]);
				if(val > jmaxv){
					jmax = j;
					jmaxv = val;
				}
			}
			h[i] = Traits<T>::abs(x[i+jmax*ldx]);
			ind[i] = i;
		}

		if(iter >= 1 && h[BLAS::MaximumIndex(n, h, 1)] == h[ibest]){
			return 4;
		}

		// Sort so that h(i) >= h(j) for i < j
		Sort::ByIndex("D", n, h, ind);
		if(0 == iter){
			itemp = t-1;
		}else{
			// If ind[0..t] is contained in indh, terminate.
			if(t > 1){
				for(size_t j = 0; j < t; ++j){
					for(size_t i = 0; i < iter*t; ++i){
						if(i >= n || ind[j] == indh[i]){
							 return 5;
						}
					}
				}
				// Replace ind[0..t] by the first t indices in ind that
				// are not in indh.
				itemp = 0;
				for(size_t j = 0; j < n; ++j){
					for(size_t i = 0; i < iter*t; ++i){
						if(i >= n || ind[j] == indh[i]){
							continue;
						}
					}
					ind[itemp] = ind[j];
					if(itemp+1 == t){
						++itemp; // compensate for the decrement before exiting the if clause
						break;
					}
					++itemp;
				}
			}
			--itemp;
		}
		if(iter*t >= n){
			for(size_t j = 0; j <= itemp; ++j){
				indh[iter*t+j] = ind[j];
			}
		}

		for(size_t j = 0; j < t; ++j){
			for(size_t i = 0; i < n; ++i){
				x[i+j*ldx] = T(0);
			}
			x[ind[j]+j*ldx] = T(1);
		}

		++iter;
		Aop("N", n, t, x, ldx, data);
	}while(1);
}

} // namespace LA
} // namespace RNP

#endif // RNP_MATRIX_NORM_ESTIMATES_HPP_INCLUDED
