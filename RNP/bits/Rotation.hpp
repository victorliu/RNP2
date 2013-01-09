#ifndef RNP_ROTATION_HPP_INCLUDED
#define RNP_ROTATION_HPP_INCLUDED

#include <cstddef>
#include <RNP/BLAS.hpp>
#include <RNP/Types.hpp>

namespace RNP{
namespace LA{
namespace Rotation{

// Purpose
// =======

// Generates a plane rotation so that

//    [  CS  SN  ]     [ F ]     [ R ]
//    [  __      ]  .  [   ]  =  [   ]   where CS**2 + |SN|**2 = 1.
//    [ -SN  CS  ]     [ G ]     [ 0 ]

// This is a faster version of the BLAS1 routine _rotg, except for
// the following differences:
//    F and G are unchanged on return.
//    If G=0, then CS=1 and SN=0.
//    If F=0, then CS=0 and SN is chosen so that R is real.
//
// Further Details
// ======= =======
// 3-5-96 - Modified with a new algorithm by W. Kahan and J. Demmel
template <typename T>// _lartg
void Generate(
	const T &f, const T &g, typename Traits<T>::real_type *cs, T *sn, T *r
){
	typedef typename Traits<T>::real_type real_type;

	const real_type safmin = Traits<real_type>::min();
	const real_type eps    = Traits<real_type>::eps();
	const real_type safmn2 = sqrt(safmin/eps);
	const real_type safmx2 = real_type(1) / safmn2;
	
	real_type aif = Traits<T>::norminf(f);
	real_type aig = Traits<T>::norminf(g);
	real_type scale = ((aif > aig) ? aif : aig);
	T fs = f;
	T gs = g;
	int count = 0;
	// Edit by vkl: check for inf; otherwise never terminates
	if(scale >= safmx2){
		do{
			++count;
			fs *= safmn2;
			gs *= safmn2;
			scale *= safmn2;
		}while(scale >= safmx2 && scale <= Traits<real_type>::max());
	}else if(scale <= safmn2){
		if(T(0) == g){
			*cs = real_type(1);
			*sn = 0;
			*r = f;
			return;
		}
		do{
			--count;
			fs *= safmx2;
			gs *= safmx2;
			scale *= safmx2;
		}while(scale <= safmn2 && scale <= Traits<real_type>::max());
	}
	real_type f2 = Traits<T>::abs2(fs);
	real_type g2 = Traits<T>::abs2(gs);
	real_type f2limit = g2; if(1 > f2limit){ f2limit = 1; }
	if(f2 <= f2limit * safmin){
		// This is a rare case: F is very small.
		if(T(0) == f){
			*cs = 0;
			*r = Traits<T>::abs(g);
			// Do complex/real division explicitly with real divisions
			{
				real_type absgs(Traits<T>::abs(gs));
				*sn = gs / absgs;
			}
			return;
		}
		real_type f2s = abs(fs);
		// g2 and g2s are accurate
		// g2 is at least safmin, and g2s is at least safmn2
		real_type g2s = sqrt(g2);
		// Error in cs from underflow in f2s is at most
		// unfl / safmn2 < sqrt(unfl*eps) < eps
		// If max(g2,one)=g2, then f2 < g2*safmin,
		// and so cs < sqrt(safmin)
		// If max(g2,one)=one, then f2 < safmin
		// and so cs < sqrt(safmin)/safmn2 = sqrt(eps)
		// Therefore, cs = f2s/g2s / sqrt( 1 + (f2s/g2s)**2 ) = f2s/g2s
		*cs = f2s / g2s;
		// Make sure abs(ff) = 1
		// Do complex/real division explicitly with 2 real divisions
		T ff(f);
		if(Traits<T>::norminf(f) <= real_type(1)) {
			ff *= safmx2;
		}
		{
			real_type absff = Traits<T>::abs(ff);
			ff /= absff;
		}
		*sn = ff * Traits<T>::conj(gs/g2s);
		*r = *cs*f + *sn*g;
	}else{
		// This is the most common case.
		// Neither f2 nor f2/g2 are less than safmin
		// f2s cannot overflow, and it is accurate
		real_type f2s = sqrt(g2 / f2 + real_type(1));
		// Do the F2S(real)*FS(complex) multiply with two real multiplies
		*r = f2s*fs;
		*cs = real_type(1) / f2s;
		// Do complex/real division explicitly with two real divisions
		{
			real_type sum(f2+g2);
			*sn = (*r / sum) * Traits<T>::conj(gs);
		}
		while(count > 0){
			*r *= safmx2;
			--count;
		}
		while(count < 0){
			*r *= safmn2;
			++count;
		}
	}
}

template <typename T> // _rot
void Apply(
	size_t n, T *cx, size_t incx, T *cy, size_t incy,
	const typename Traits<T>::real_type &c, const T &s
){
	BLAS::RotApply(n, cx, incx, cy, incy, c, s);
}

// Applies a sequence of real plane rotations to a matrix A,
// from either the left or the right.
//
// When SIDE = 'L', the transformation takes the form
//
//    A := P*A
//
// and when SIDE = 'R', the transformation takes the form
//
//    A := A*P**T
//
// where P is an orthogonal matrix consisting of a sequence of z plane
// rotations, with z = M when SIDE = 'L' and z = N when SIDE = 'R',
// and P**T is the transpose of P.
// 
// When DIRECT = 'F' (Forward sequence), then
// 
//    P = P(z-1) * ... * P(2) * P(1)
// 
// and when DIRECT = 'B' (Backward sequence), then
// 
//    P = P(1) * P(2) * ... * P(z-1)
// 
// where P(k) is a plane rotation matrix defined by the 2-by-2 rotation
// 
//    R(k) = (  c(k)  s(k) )
//         = ( -s(k)  c(k) ).
// 
// When PIVOT = 'V' (Variable pivot), the rotation is performed
// for the plane (k,k+1), i.e., P(k) has the form
// 
//    P(k) = (  1                                            )
//           (       ...                                     )
//           (              1                                )
//           (                   c(k)  s(k)                  )
//           (                  -s(k)  c(k)                  )
//           (                                1              )
//           (                                     ...       )
//           (                                            1  )
// 
// where R(k) appears as a rank-2 modification to the identity matrix in
// rows and columns k and k+1.
// 
// When PIVOT = 'T' (Top pivot), the rotation is performed for the
// plane (1,k+1), so P(k) has the form
// 
//    P(k) = (  c(k)                    s(k)                 )
//           (         1                                     )
//           (              ...                              )
//           (                     1                         )
//           ( -s(k)                    c(k)                 )
//           (                                 1             )
//           (                                      ...      )
//           (                                             1 )
// 
// where R(k) appears in rows and columns 1 and k+1.
// 
// Similarly, when PIVOT = 'B' (Bottom pivot), the rotation is
// performed for the plane (k,z), giving P(k) the form
// 
//    P(k) = ( 1                                             )
//           (      ...                                      )
//           (             1                                 )
//           (                  c(k)                    s(k) )
//           (                         1                     )
//           (                              ...              )
//           (                                     1         )
//           (                 -s(k)                    c(k) )
// 
// where R(k) appears in rows and columns k and z.  The rotations are
// performed without ever forming P(k) explicitly.
template <typename T> // _lasr
void ApplySequence(
	const char *side, const char *piv, const char *dir,
	size_t m, size_t n,
	const typename Traits<T>::real_type *c,
	const typename Traits<T>::real_type *s,
	T *a, size_t lda
){
	typedef typename Traits<T>::real_type real_type;
	RNPAssert('L' == side[0] || 'R' == side[0]);
	RNPAssert('V' == piv[0] || 'T' == piv[0] || 'B' == piv[0]);
	RNPAssert('F' == dir[0] || 'B' == dir[0]);
	RNPAssert(lda >= m);
	if(0 == m || 0 == n){ return; }
	
	static const real_type one(1), zero(0);
	
	if('L' == side[0]){ // form P * A
		if(1 == m){ return; }
		if('V' == piv[0]){
			if('F' == dir[0]){
				for(size_t j = 0; j+1 < m; ++j){
					const real_type cj(c[j]), sj(s[j]);
					if(one == cj && zero == sj){ continue; }
					for(size_t i = 0; i < n; ++i){
						T as(a[j+1+i*lda]);
						a[j+1+i*lda] = cj*as - sj*a[j+i*lda];
						a[j+0+i*lda] = sj*as + cj*a[j+i*lda];
					}
				}
			}else{
				size_t j = m-1; while(j --> 0){
					const real_type cj(c[j]), sj(s[j]);
					if(one == cj && zero == sj){ continue; }
					for(size_t i = 0; i < n; ++i){
						T as(a[j+1+i*lda]);
						a[j+1+i*lda] = cj*as - sj*a[j+i*lda];
						a[j+0+i*lda] = sj*as + cj*a[j+i*lda];
					}
				}
			}
		}else if('T' == piv[0]){
			if('F' == dir[0]){
				for(size_t j = 1; j < m; ++j){
					const real_type cj(c[j-1]), sj(s[j-1]);
					if(one == cj && zero == sj){ continue; }
					for(size_t i = 0; i < n; ++i){
						T as(a[j+1+i*lda]);
						a[j+i*lda] = cj*as - sj*a[0+i*lda];
						a[0+i*lda] = sj*as + cj*a[0+i*lda];
					}
				}
			}else{
				size_t j = m; while(j --> 1){
					const real_type cj(c[j-1]), sj(s[j-1]);
					if(one == cj && zero == sj){ continue; }
					for(size_t i = 0; i < n; ++i){
						T as(a[j+1+i*lda]);
						a[j+i*lda] = cj*as - sj*a[0+i*lda];
						a[0+i*lda] = sj*as + cj*a[0+i*lda];
					}
				}
			}
		}else{ // 'B'
			if('F' == dir[0]){
				for(size_t j = 0; j+1 < m; ++j){
					const real_type cj(c[j]), sj(s[j]);
					if(one == cj && zero == sj){ continue; }
					for(size_t i = 0; i < n; ++i){
						T as(a[j+i*lda]);
						a[ j +i*lda] = sj*a[m-1+i*lda] + cj*as;
						a[m-1+i*lda] = cj*a[m-1+i*lda] - sj*as;
					}
				}
			}else{
				size_t j = m-1; while(j --> 0){
					const real_type cj(c[j]), sj(s[j]);
					if(one == cj && zero == sj){ continue; }
					for(size_t i = 0; i < n; ++i){
						T as(a[j+i*lda]);
						a[ j +i*lda] = sj*a[m-1+i*lda] + cj*as;
						a[m-1+i*lda] = cj*a[m-1+i*lda] - sj*as;
					}
				}
			}
		}
	}else{ // form A * P^T
		if(1 == n){ return; }
		if('V' == piv[0]){
			if('F' == dir[0]){
				for(size_t j = 0; j+1 < n; ++j){
					const real_type cj(c[j]), sj(s[j]);
					if(one == cj && zero == sj){ continue; }
					for(size_t i = 0; i < m; ++i){
						T as(a[i+(j+1)*lda]);
						a[i+(j+1)*lda] = cj*as - sj*a[i+j*lda];
						a[i+(j+0)*lda] = sj*as + cj*a[i+j*lda];
					}
				}
			}else{
				size_t j = n-1; while(j --> 0){
					const real_type cj(c[j]), sj(s[j]);
					if(one == cj && zero == sj){ continue; }
					for(size_t i = 0; i < m; ++i){
						T as(a[i+(j+1)*lda]);
						a[i+(j+1)*lda] = cj*as - sj*a[i+j*lda];
						a[i+(j+0)*lda] = sj*as + cj*a[i+j*lda];
					}
				}
			}
		}else if('T' == piv[0]){
			if('F' == dir[0]){
				for(size_t j = 1; j < n; ++j){
					const real_type cj(c[j-1]), sj(s[j-1]);
					if(one == cj && zero == sj){ continue; }
					for(size_t i = 0; i < m; ++i){
						T as(a[i+j*lda]);
						a[i+j*lda] = cj*as - sj*a[i+0*lda];
						a[i+0*lda] = sj*as + cj*a[i+0*lda];
					}
				}
			}else{
				size_t j = n; while(j --> 1){
					const real_type cj(c[j-1]), sj(s[j-1]);
					if(one == cj && zero == sj){ continue; }
					for(size_t i = 0; i < m; ++i){
						T as(a[i+j*lda]);
						a[i+j*lda] = cj*as - sj*a[i+0*lda];
						a[i+0*lda] = sj*as + cj*a[i+0*lda];
					}
				}
			}
		}else{ // 'B'
			if('F' == dir[0]){
				for(size_t j = 0; j+1 < n; ++j){
					const real_type cj(c[j]), sj(s[j]);
					if(one == cj && zero == sj){ continue; }
					for(size_t i = 0; i < m; ++i){
						T as(a[i+j*lda]);
						a[i+  j  *lda] = sj*a[i+(n-1)*lda] + cj*as;
						a[i+(n-1)*lda] = cj*a[i+(n-1)*lda] - sj*as;
					}
				}
			}else{
				size_t j = n-1; while(j --> 0){
					const real_type cj(c[j]), sj(s[j]);
					if(one == cj && zero == sj){ continue; }
					for(size_t i = 0; i < m; ++i){
						T as(a[i+j*lda]);
						a[i+  j  *lda] = sj*a[i+(n-1)*lda] + cj*as;
						a[i+(n-1)*lda] = cj*a[i+(n-1)*lda] - sj*as;
					}
				}
			}
		}
	}
}

} // namespace Rotation
} // namespace LA
} // namespace RNP

#endif // RNP_ROTATION_HPP_INCLUDED
