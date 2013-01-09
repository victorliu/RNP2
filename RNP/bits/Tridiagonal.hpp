#ifndef RNP_TRIDIAGONAL_HPP_INCLUDED
#define RNP_TRIDIAGONAL_HPP_INCLUDED

#include <iostream>
#include <cstddef>
#include <RNP/bits/Rotation.hpp>
#include <RNP/BLAS.hpp>
#include <RNP/Debug.hpp>

namespace RNP{
namespace LA{
namespace Tridiagonal{

template <typename T> // _lange
typename Traits<T>::real_type Norm(
	const char *norm, size_t n, const T *diag, const T *offdiag
){
	typedef typename Traits<T>::real_type real_type;
	real_type result(0);
	if(0 == n){ return result; }
	if('M' == norm[0]){ // max(abs(A(i,j)))
		result = Traits<T>::abs(diag[n-1]);
		for(size_t i = 0; i+1 < n; ++i){
			real_type ai = Traits<T>::abs(diag[i]);
			if(!(ai < result)){ result = ai; }
			ai = Traits<T>::abs(offdiag[i]);
			if(!(ai < result)){ result = ai; }
		}
	}else if('O' == norm[0] || '1' == norm[0] || 'I'){ // max col sum or row sum
		if(1 == n){ return Traits<T>::abs(diag[0]); }
		result = Traits<T>::abs(diag[0]) + Traits<T>::abs(offdiag[0]);
		real_type sum(Traits<T>::abs(diag[n-1]) + Traits<T>::abs(offdiag[n-2]));
		if(!(sum < result)){ result = sum; }
		for(size_t i = 1; i+1 < n; ++i){
			sum = Traits<T>::abs(diag[i]) + Traits<T>::abs(offdiag[i-1]) + Traits<T>::abs(offdiag[i]);
			if(!(sum < result)){ result = sum; }
		}
	}else if('F' == norm[0] || 'E' == norm[0]){ // Frobenius norm
		real_type scale(0);
		real_type sum(1);
		for(size_t i = 0; i+1 < n; ++i){
			real_type ca = Traits<T>::abs(offdiag[i]);
			if(scale < ca){
				real_type r = scale/ca;
				sum = real_type(1) + sum*r*r;
				scale = ca;
			}else{
				real_type r = ca/scale;
				sum += r*r;
			}
		}
		for(size_t i = 0; i < n; ++i){
			real_type ca = Traits<T>::abs(diag[i]);
			if(scale < ca){
				real_type r = scale/ca;
				sum = real_type(1) + sum*r*r;
				scale = ca;
			}else{
				real_type r = ca/scale;
				sum += r*r;
			}
		}
		result = scale*sqrt(sum);
	}
	return result;
}


namespace Util{

// Computes eigendecomposition
//   [ cs1 sn1 ] [ a b ] [ cs1 -sn1 ] = [ rt1  0  ]
//   [-sn1 cs1 ] [ b c ] [ sn1  cs1 ]   [  0  rt2 ], rt1 > rt2
template <typename T> // T must be a real type
void SymmetricEigensystem2(
	const T &a, const T &b, const T &c, T *rt1, T *rt2, T *cs1, T *sn1
){
	RNPAssert(!Traits<T>::is_complex()); // T must be a real type
	typedef T real_type;
	
	T sm(a+c);
	T df(a-c);
	real_type adf(Traits<T>::abs(df));
	T b2(b+b);
	real_type ab(Traits<T>::abs(b2));
	T acmx, acmn;
	if(Traits<T>::abs(a) > Traits<T>::abs(c)){
		acmx = a; acmn = c;
	}else{
		acmx = c; acmn = a;
	}
	// Compute the square root
	T rt(0);
	if(adf > ab){
		rt = adf*sqrt(T(1) + (ab/adf)*(ab/adf));
	}else if(adf < ab){
		rt = ab*sqrt(T(1) + (adf/ab)*(adf/ab));
	}else{
		rt = ab*sqrt(T(2));
	}
	// Compute eigenvalues
	int sgn1;
	if(T(0) == sm){
		*rt1 = (T(1)/T(2)) * rt;
		*rt2 = -(*rt1);
		sgn1 = 1;
	}else{
		if(sm < T(0)){
			*rt1 = (T(1)/T(2)) * (sm-rt);
			sgn1 = -1;
		}else{ // sm > 0
			*rt1 = (T(1)/T(2)) * (sm+rt);
			sgn1 = 1;
		}
		// Order of execution important.
		// To get fully accurate smaller eigenvalue,
		// next line needs to be executed in higher precision.
		*rt2 = (acmx / (*rt1)) * acmn - (b/(*rt1)) * b;
	}
	if(NULL == cs1 || NULL == sn1){ return; }
	// Compute eigenvector
	int sgn2;
	T cs;
	if(df >= T(0)){
		cs = df + rt;
		sgn2 = 1;
	}else{
		cs = df - rt;
		sgn2 = -1;
	}
	real_type acs = Traits<T>::abs(cs);
	if(acs > ab){
		T ct = -b2/cs;
		*sn1 = T(1) / sqrt(T(1) + ct*ct);
		*cs1 = ct*(*sn1);
	}else{
		if(real_type(0) == ab){
			*cs1 = T(1);
			*sn1 = T(0);
		}else{
			T tn = -cs/b2;
			*cs1 = T(1) / sqrt(T(1) + tn*tn);
			*sn1 = tn*(*cs1);
		}
	}
	if(sgn1 == sgn2){
		T tn = *cs1;
		*cs1 = -(*sn1);
		*sn1 = tn;
	}
}

template <typename T> // T must be a real type
int SymmetricQRIteration(
	size_t n,
	T *diag, T *offdiag,
	T *z, size_t ldz,
	T *work // only needed if NULL != z, size 2*(n-1)
){
	RNPAssert(!Traits<T>::is_complex()); // T must be a real type
	typedef T real_type;
	if(n <= 1){
		return 0;
	}

	// Determine the unit roundoff and over/underflow thresholds.
	static const real_type eps(Traits<real_type>::eps());
	static const real_type eps2(eps*eps);
	static const real_type safmin(Traits<real_type>::min());
	static const real_type safmax(Traits<real_type>::max());
	static const real_type ssfmax(sqrt(safmax) / real_type(3));
	static const real_type ssfmin(sqrt(safmin) / eps2);

	const size_t nmaxit = n * 30;
	size_t jtot = 0;

	// Determine where the matrix splits and choose QL or QR iteration
	// for each block, according to whether top or bottom diagonal
	// element is smaller.

	size_t l1 = 0;
	size_t l, m;
	size_t lsv, lend, lendsv;

	real_type anorm;
	int iscale;
	
outer_loop_start:
	if(l1+1 > n){ return 0; }
	if(l1+1 > 1){
		offdiag[l1-1] = T(0);
	}
	{
		bool found_small = false;
		// skip over zero and small subdiagonals
		if(l1+1 < n){
			for(m = l1; m+1 < n; ++m){
				real_type tst(Traits<T>::abs(offdiag[m]));
				if(real_type(0) == tst){
					found_small = true;
					break;
				}else if(
					tst <= eps * (
						sqrt(Traits<T>::abs(diag[m])) * sqrt(Traits<T>::abs(diag[m+1]))
					)
				){
					offdiag[m] = T(0);
					found_small = true;
					break;
				}
			}
		}
		if(!found_small){
			m = n-1;
		}
	}

	l = l1;
	lsv = l;
	lend = m;
	lendsv = lend;
	l1 = m+1;
	if(lend == l){
		goto outer_loop_start;
	}

	// Scale submatrix in rows and columns (l+1) to (lend+1)
	anorm = LA::Tridiagonal::Norm("M", lend-l+1, &diag[l], &offdiag[l]);
	iscale = 0;
	if(real_type(0) == anorm){
		goto outer_loop_start;
	}
	if(anorm > ssfmax){
		iscale = 1;
		BLAS::Rescale("G", 0, 0, anorm, ssfmax, lend-l+1, 1, &   diag[l], n);
		BLAS::Rescale("G", 0, 0, anorm, ssfmax, lend-l+0, 1, &offdiag[l], n);
	}else if(anorm < ssfmin){
		iscale = 2;
		BLAS::Rescale("G", 0, 0, anorm, ssfmin, lend-l+1, 1, &   diag[l], n);
		BLAS::Rescale("G", 0, 0, anorm, ssfmin, lend-l+0, 1, &offdiag[l], n);
	}
	// Choose between QL and QR iteration
	if(Traits<T>::abs(diag[lend]) < Traits<T>::abs(diag[l])){
		lend = lsv;
		l = lendsv;
	}
	if(lend > l){ // QL Iteration, Look for small subdiagonal element.
		do{
			bool found_small = false;
			if(l != lend){
				for(m = l; m < lend; ++m){
					real_type tst(Traits<T>::abs(offdiag[m]));
					if(tst*tst <= (eps2*Traits<T>::abs(diag[m])) * Traits<T>::abs(diag[m+1])+safmin){
						found_small = true;
						break;
					}
				}
			}
			if(!found_small){
				m = lend;
			}
			if(m < lend){
				offdiag[m] = T(0);
			}
			T p = diag[l];
			if(m != l){
				// If remaining matrix is 2-by-2, special case it
				if(m == l+1){
					T rt1, rt2;
					if(NULL != z){
						T c, s;
						LA::Tridiagonal::Util::SymmetricEigensystem2(
							diag[l], offdiag[l], diag[l+1], &rt1, &rt2, &c, &s
						);
						work[l] = c;
						work[n-1+l] = s;
						LA::Rotation::ApplySequence(
							"R","V","B", n, 2, &work[l], &work[n-1+l], &z[0+l*ldz], ldz
						);
					}else{
						LA::Tridiagonal::Util::SymmetricEigensystem2(
							diag[l], offdiag[l], diag[l+1], &rt1, &rt2, (T*)NULL, (T*)NULL
						);
					}
					diag[l] = rt1;
					diag[l+1] = rt2;
					offdiag[l] = T(0);
					l += 2;
					if(l <= lend){
						continue;
					}
					goto end_outer_loop;
				}

				if(jtot == nmaxit){
					goto end_outer_loop;
				}
				++jtot;
				// Form shift.

				T g = (diag[l+1]-p) / (T(2)*offdiag[l]);
				T r = Traits<T>::hypot2(g, T(1));
				g = diag[m] - p + (offdiag[l] / (g+(g > 0 ? r : -r)));
				{
					T s = 1.;
					T c = 1.;
					p = 0.;

					// Inner loop
					for(size_t i = m-1; i+1 >= l+1; --i){ // +1's needed here
						T f = s*offdiag[i];
						T b = c*offdiag[i];
						LA::Rotation::Generate(g, f, &c, &s, &r);
						if(i+1 != m){ offdiag[i+1] = r; }
						g = diag[i+1] - p;
						r = (diag[i]-g)*s + T(2)*c*b;
						p = s*r;
						diag[i+1] = g + p;
						g = c*r - b;
						// If eigenvectors are desired, then save rotations.
						if(NULL != z){
							work[i] = c;
							work[n-1+i] = -s;
						}
					}
				}
				// If eigenvectors are desired, then apply saved rotations.
				if(NULL != z){
					LA::Rotation::ApplySequence(
						"R","V","B", n, m-l+1, &work[l], &work[n-1+l], &z[0+l*ldz], ldz
					);
				}

				diag[l] -= p;
				offdiag[l] = g;
				continue;
			}
			// Eigenvalue found.
			diag[l] = p;

			++l;
		}while(l <= lend);
		goto end_outer_loop;
	}else{ // QR Iteration, Look for small superdiagonal element.

		do{
			bool found_small = false;
			if(l != lend){
				for(m = l; m > lend; --m){
					real_type tst = Traits<T>::abs(offdiag[m-1]);
					if(tst*tst <= (eps2*Traits<T>::abs(diag[m])) * Traits<T>::abs(diag[m-1])+safmin){
						found_small = true;
						break;
					}
				}
			}
			if(!found_small){
				m = lend;
			}

			if(m > lend){
				offdiag[m-1] = 0.;
			}
			T p = diag[l];
			if(m != l){	
				// If remaining matrix is 2-by-2, special case it
				if(m+1 == l){
					T rt1, rt2;
					if(NULL != z){
						T c, s;
						LA::Tridiagonal::Util::SymmetricEigensystem2(
							diag[l-1], offdiag[l-1], diag[l], &rt1, &rt2, &c, &s
						);
						work[m] = c;
						work[n-1+m] = s;
						LA::Rotation::ApplySequence(
							"R","V","F", n, 2, &work[m], &work[n-1+m], &z[0+(l-1)*ldz], ldz
						);
					}else{
						LA::Tridiagonal::Util::SymmetricEigensystem2(
							diag[l-1], offdiag[l-1], diag[l], &rt1, &rt2, (T*)NULL, (T*)NULL
						);
					}
					diag[l-1] = rt1;
					diag[l] = rt2;
					offdiag[l-1] = T(0);
					l -= 2;
					if((l+1) >= (lend+1)){
						continue;
					}
					goto end_outer_loop;
				}

				if(jtot == nmaxit){
					goto end_outer_loop;
				}
				++jtot;
				// Form shift.
				T g = (diag[l-1]-p) / (T(2)*offdiag[l-1]);
				T r = Traits<T>::hypot2(g, T(1));
				g = diag[m] - p + (offdiag[l-1] / (g+(g > 0 ? r : -r)));
				{
					T s = 1.;
					T c = 1.;
					p = 0.;

					// Inner loop
					for(size_t i = m; i < l; ++i){
						T f = s*offdiag[i];
						T b = c*offdiag[i];
						LA::Rotation::Generate(g, f, &c, &s, &r);
						if(i != m){ offdiag[i-1] = r; }
						g = diag[i] - p;
						r = (diag[i+1]-g)*s + T(2)*c*b;
						p = s*r;
						diag[i] = g + p;
						g = c*r - b;
						// If eigenvectors are desired, then save rotations.
						if(NULL != z){
							work[i] = c;
							work[n-1+i] = s;
						}
					}
				}
				// If eigenvectors are desired, then apply saved rotations.
				if(NULL != z){
					LA::Rotation::ApplySequence(
						"R","V","F", n, l-m+1, &work[m], &work[n-1+m], &z[0+m*ldz], ldz
					);
				}
				diag[l] -= p;
				offdiag[l-1] = g;
				continue;
			}
			// Eigenvalue found.
			diag[l] = p;
			--l;
		}while(l+1 >= lend+1); // +1's necessary here
		goto end_outer_loop;
	}

	// Undo scaling if necessary
end_outer_loop:
	if(1 == iscale){
		BLAS::Rescale("G", 0, 0, ssfmax, anorm, lendsv-lsv+1, 1, &   diag[lsv], n);
		BLAS::Rescale("G", 0, 0, ssfmax, anorm, lendsv-lsv  , 1, &offdiag[lsv], n);
	}else if(2 == iscale){
		BLAS::Rescale("G", 0, 0, ssfmin, anorm, lendsv-lsv+1, 1, &   diag[lsv], n);
		BLAS::Rescale("G", 0, 0, ssfmin, anorm, lendsv-lsv  , 1, &offdiag[lsv], n);
	}
	// Check for no convergence to an eigenvalue after a total of N*MAXIT iterations.
	if(jtot < nmaxit){
		goto outer_loop_start;
	}
	int info = 0;
	for(size_t i = 0; i+1 < n; ++i){
		if(T(0) != offdiag[i]){
			++info;
		}
	}
	return info;
}

} // namespace Util
} // namespace Tridiagonal
} // namespace LA
} // namespace RNP

#endif // RNP_TRIDIAGONAL_HPP_INCLUDED
