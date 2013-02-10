#ifndef RNP_BLAS_MIX_HPP_INCLUDED
#define RNP_BLAS_MIX_HPP_INCLUDED

///////////////////////////////////////////////////////////////////////
// RNP::BLAS (mixed implementation)
// ===========================
// This header provides templated implementations of low level BLAS
// and declares prototypes to call external BLAS for high level
// routines.
//
// For efficiency, we assume that arguments are not aliased. That is,
// they do not point to overlapping regions of memory. These arguments
// are indicated by the RNP_RESTRICT marker.
//

///////////////////////////////////////////////////////////////////////
// Name mappings from fortran BLAS
// -------------------------------
//
// ### Level 1
//
// single | double | complex | zomplex | RNP name
// -------|--------|---------|---------|-----------------
// srotg  | drotg  | crotg   | zrotg   | RotGen
// srotmg | drotmg |         |         | ModifiedRotGen
// srot   | drot   |         |         | RotApply
// srotm  | drotm  |         |         | ModifiedRotApply
// sswap  | dswap  | cswap   | zswap   | Swap
// sscal  | dscal  | cscal   | zscal   | Scale
// scopy  | dcopy  | ccopy   | zcopy   | Copy
// saxpy  | daxpy  | caxpy   | zaxpy   | Axpy
// sdot   | ddot   | cdotu   | zdotu   | Dot
// dsdot  | sdsdot |         |         | DotEx
//        |        | cdotc   | zdotc   | ConjugateDot
// snrm2  | dnrm2  | scnrm2  | dznrm2  | Norm2
// sasum  | dasum  | scasum  | dzasum  | Asum
// isamax | idamax | icamax  | izamax  | MaximumIndex
//
// ### Level 2
//
// single | double | complex | zomplex | RNP name
// -------|--------|---------|---------|-----------------
// sgemv  | dgemv  | cgemv   | zgemv   | MultMV
// sgbmv  | dgbmv  | cgbmv   | zgbmv   | MultBandedV
//        |        | chemv   | zhemv   | MultHermV
//        |        | chbmv   | zhbmv   | MultBandedHermV
//        |        | chpmv   | zhpmv   | MultPackedHermV
// ssymv  | dsymv  |         |         | MultSymV
// ssbmv  | dsbmv  |         |         | MultBandedSymV
// sspmv  | dspmv  |         |         | MultPackedSymV
// strmv  | dtrmv  | ctrmv   | ztrmv   | MultTrV
// stbmv  | dtbmv  | ctbmv   | ztbmv   | MultBandedTrV
// stpmv  | dtpmv  | ctpmv   | ztpmv   | MultPackedTrV
// strsv  | dtrsv  | ctrsv   | ztrsv   | SolveTrV
// stbsv  | dtbsv  | ctbsv   | ztbsv   | SolveBandedTrV
// stpsv  | dtpsv  | ctpsv   | ztpsv   | SolvePackedTrV
//        |        |         |         |
// sger   | dger   | cgeru   | zgeru   | Rank1Update
//        |        | cgerc   | zgerc   | ConjugateRank1Update
//        |        | cher    | zher    | HermRank1Update
//        |        | chpr    | zhpr    | PackedHermRank1Update
//        |        | cher2   | zher2   | HermRank2Update
//        |        | chpr2   | zhpr2   | PackedHermRank2Update
// ssyr   | dsyr   |         |         | SymRank1Update
// sspr   | dspr   |         |         | PackedSymRank1Update
// ssyr2  | dsyr2  |         |         | SymRank2Update
// sspr2  | dspr2  |         |         | PackedSymRank2Update
//
// ### Level 3
//
// single | double | complex | zomplex | RNP name
// -------|--------|---------|---------|-----------------
// sgemm  | dgemm  | cgemm   | zgemm   | MultMM
// ssymm  | dsymm  | csymm   | zsymm   | MultSyM
//        |        | chemm   | zhemm   | MultHermM
// ssyrk  | dsyrk  | csyrk   | zsyrk   | SymRankKUpdate
//        |        | cherk   | zherk   | HermRankKUpdate
// ssyr2k | dsyr2k | csyr2k  | zsyr2k  | SymRank2KUpdate
//        |        | cher2k  | zher2k  | HermRank2KUpdate
// strmm  | dtrmm  | ctrmm   | ztrmm   | MultTrM
// strsm  | dtrsm  | ctrsm   | ztrsm   | SolveTrM
//
// ### Extra routines
//
// Level 1,2 | Description
// ----------|---------------------------------------------------------
// Set       | Sets entries in a vector or (parts of a) matrix (_laset)
// Copy      | Copies a matrix, possilby transposed
// Conjugate | Conjugates a vector (_lacgv)
// Rescale   | Rescales a matrix (_lascl)
// Norm1     | True 1-norm of a vector
//

#include <complex>
#include <cstring>
#include <RNP/Types.hpp>
#include <RNP/Debug.hpp>

namespace RNP{
namespace BLAS{

///////////////////////////////////////////////////////////////////////
// Set (vector)
// ------------
// Sets each element of a vector to the same value.
//
// Arguments
// n     Length of vector.
// val   Value to set.
// x     Pointer to the first element of the vector.
// incx  Increment between elements of the vector, incx > 0.
//
template <typename T>
void Set(size_t n, const T &val, T* RNP_RESTRICT x, size_t incx){
	while(n --> 0){ *x = val; x += incx; }
}

///////////////////////////////////////////////////////////////////////
// Set (matrix)
// ------------
// Sets elements of a rectangular matrix to the specified off-diagonal
// and diagonal values.
//
// Arguments
// m       Number of rows of the matrix.
// n       Number of columns of the matrix.
// offdiag Value to which offdiagonal elements are set.
// diag    Value to which diagonal elements are set.
// a       Pointer to the first element of the matrix.
// lda     Leading dimension of the array containing the matrix,
//         lda >= m.
//
template <typename T, typename TV>
void Set(
	size_t m, size_t n, const TV &offdiag, const TV &diag,
	T* RNP_RESTRICT a, size_t lda
){
	for(size_t j = 0; j < n; ++j){
		for(size_t i = 0; i < m; ++i){
			a[i+j*lda] = (i == j ? diag : offdiag);
		}
	}
}

///////////////////////////////////////////////////////////////////////
// Copy (matrix)
// -------------
// Copies values from one rectangular matrix to another of identical
// size.
//
// Arguments
// m       Number of rows of the matrices.
// n       Number of columns of the matrices.
// src     Pointer to the first element of the source matrix.
// ldsrc   Leading dimension of the array containing the source
//         matrix, ldsrc >= m.
// dst     Pointer to the first element of the destination matrix.
// lddst   Leading dimension of the array containing the destination
//         matrix, ldsrc >= m.
//
template <typename T>
void Copy(
	size_t m, size_t n, const T* RNP_RESTRICT src, size_t ldsrc,
	T* RNP_RESTRICT dst, size_t lddst
){
	if(m == ldsrc && m == lddst){
		memcpy(dst, src, sizeof(T) * m*n);
	}else{
		for(size_t j = 0; j < n; ++j){
			memcpy(&dst[j*lddst], &src[j*ldsrc], sizeof(T) * m);
		}
	}
}

///////////////////////////////////////////////////////////////////////
// Copy (matrix, transposed)
// -------------------------
// Copies values from one rectangular matrix to another of identical
// size, possibly transposed. This version supports copying matrices
// with different types of elements (so long as they are compatible
// with the assignment operator).
//
// Arguments
// trans   If "N", source is not transposed.
//         If "T", source is transposed.
//         If "C", source is conjugate-transposed.
// m       Number of rows of the matrices.
// n       Number of columns of the matrices.
// src     Pointer to the first element of the source matrix.
// ldsrc   Leading dimension of the array containing the source
//         matrix, ldsrc >= m.
// dst     Pointer to the first element of the destination matrix.
// lddst   Leading dimension of the array containing the destination
//         matrix, ldsrc >= m.
//
template <typename TS, typename TD>
void Copy(
	const char *trans, size_t m, size_t n,
	const TS* RNP_RESTRICT src, size_t ldsrc,
	TD* RNP_RESTRICT dst, size_t lddst
){
	if('N' == trans[0]){
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				dst[i+j*lddst] = src[i+j*ldsrc];
			}
		}
	}else if('T' == trans[0]){
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				dst[j+i*lddst] = src[i+j*ldsrc];
			}
		}
	}else if('C' == trans[0]){
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				dst[j+i*lddst] = Traits<TS>::conj(src[i+j*ldsrc]);
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////
// Conjugate
// ---------
// Conjugates each element of a vector.
//
// Arguments
// n     Number of elements in the vector.
// x     Pointer to the first element of the vector.
// incx  Increment between elements of the vector, incx > 0.
//
template <typename T>
void Conjugate(size_t n, T* RNP_RESTRICT x, size_t incx){
	while(n --> 0){
		*x = Traits<T>::conj(*x);
		x += incx;
	}
}

///////////////////////////////////////////////////////////////////////
// Rescale
// -------
// Rescales every element of a vector by an integer power of 2.
//
// Arguments
// n     Number of elements in the vector.
// p     The power of 2 of the scale factor (2^p).
// x     Pointer to the first element of the vector.
// incx  Increment between elements of the vector, incx > 0.
//
template <typename T>
void Rescale(size_t n, int p, T *x, size_t incx){
	while(n --> 0){
		*x = std::ldexp(*x, p);
		x += incx;
	}
}
template <typename T>
void Rescale(size_t n, int p, std::complex<T> *x, size_t incx){
	while(n --> 0){
		*x = std::complex<T>(std::ldexp(x->real(), p), std::ldexp(x->imag(), p));
		x += incx;
	}
}

///////////////////////////////////////////////////////////////////////
// Rescale
// -------
// Rescales every element of a matrix safely. The scale factor is
// specified as a ratio cto/cfrom. One typically specifies cfrom
// as the element norm of the existing matrix, and cto as the target
// element norm scaling.
//
// Arguments
// type  Type of matrix to scale.
//       If "G", the matrix is a general rectangular matrix.
//       If "L", the matrix is assumed to be lower triangular.
//       If "U", the matrix is assumed to be upper triangular.
//       If "H", the matrix is assumed to be upper Hessenberg.
//       If "B", the matrix is assumed to be the lower half of a
//               symmetric banded matrix (kl is the lower bandwidth).
//       If "Q", the matrix is assumed to be the upper half of a
//               symmetric banded matrix (ku is the lower bandwidth).
//       If "Z", the matrix is assumed to be banded with lower and
//               upper bandwidths kl and ku, respectively.
// cfrom The denominator of the scale factor to apply.
// cto   The numerator of the scale factor to apply.
// m     Number of rows of the matrix.
// n     Number of columns of the matrix.
// a     Pointer to the first element of the matrix.
// lda   Leading dimension of the array containing the matrix, lda > 0.
//
template <typename TS, typename T>
void Rescale(
	const char *type, size_t kl, size_t ku,
	const TS &cfrom, const TS &cto,
	size_t m, size_t n, T* RNP_RESTRICT a, size_t lda
){
	if(n == 0 || m == 0){ return; }

	const TS smlnum = Traits<TS>::min();
	const TS bignum = TS(1) / smlnum;

	TS cfromc = cfrom;
	TS ctoc = cto;

	bool done = true;
	do{
		const TS cfrom1 = cfromc * smlnum;
		TS mul;
		if(cfrom1 == cfromc){
			// CFROMC is an inf.  Multiply by a correctly signed zero for
			// finite CTOC, or a NaN if CTOC is infinite.
			mul = ctoc / cfromc;
			done = true;
			//cto1 = ctoc;
		}else{
			const TS cto1 = ctoc / bignum;
			if(cto1 == ctoc){
				// CTOC is either 0 or an inf.  In both cases, CTOC itself
				// serves as the correct multiplication factor.
				mul = ctoc;
				done = true;
				cfromc = TS(1);
			}else if(Traits<TS>::abs(cfrom1) > Traits<TS>::abs(ctoc) && ctoc != TS(0)){
				mul = smlnum;
				done = false;
				cfromc = cfrom1;
			}else if(Traits<TS>::abs(cto1) > Traits<TS>::abs(cfromc)){
				mul = bignum;
				done = false;
				ctoc = cto1;
			}else{
				mul = ctoc / cfromc;
				done = true;
			}
		}

		switch(type[0]){
		case 'G': // Full matrix
			for(size_t j = 0; j < n; ++j){
				for(size_t i = 0; i < m; ++i){
					a[i+j*lda] *= mul;
				}
			}
			break;
		case 'L': // Lower triangular matrix
			for(size_t j = 0; j < n; ++j){
				for(size_t i = j; i < m; ++i){
					a[i+j*lda] *= mul;
				}
			}
			break;
		case 'U': // Upper triangular matrix
			for(size_t j = 0; j < n; ++j){
				size_t ilimit = j+1; if(m < ilimit){ ilimit = m; }
				for(size_t i = 0; i < ilimit; ++i){
					a[i+j*lda] *= mul;
				}
			}
			break;
		case 'H': // Upper Hessenberg matrix
			for(size_t j = 0; j < n; ++j) {
				size_t ilimit = j+2; if(m < ilimit){ ilimit = m; };
				for(size_t i = 0; i < ilimit; ++i){
					a[i+j*lda] *= mul;
				}
			}
			break;
		case 'B': // Lower half of a symmetric band matrix
			for(size_t j = 0; j < n; ++j){
				size_t ilimit = n-j; if(kl+1 < ilimit){ ilimit = kl+1; }
				for(size_t i = 0; i < ilimit; ++i){
					a[i+j*lda] *= mul;
				}
			}
			break;
		case 'Q': // Upper half of a symmetric band matrix
			for(size_t j = 0; j < n; ++j){
				size_t istart = (ku > j) ? ku-j : 0;
				for(size_t i = istart; i <= ku; ++i){
					a[i+j*lda] *= mul;
				}
			}
		case 'Z': // Band matrix
			{ size_t k3 = 2*kl + ku + 1;
			for(size_t j = 0; j < n; ++j){
				size_t istart = kl+ku-j;
				if(kl > istart){ istart = kl; }
				size_t ilimit = kl + ku + m-j;
				if(k3 < ilimit){ ilimit = k3; }
				for(size_t i = istart; i < ilimit; ++i){
					a[i+j*lda] *= mul;
				}
			} }
			break;
		default: break;
		}
	}while(!done);
}

///////////////////////////////////////////////////////////////////////
// Norm1
// -----
// Returns the 1-norm of a vector (sum of absolute values of each
// element).
//
// Arguments
// n     Number of elements in the vector.
// x     Pointer to the first element of the vector.
// incx  Increment between elements of the vector, incx > 0.
//
template <typename T>
typename Traits<T>::real_type Norm1(
	size_t n, const T* RNP_RESTRICT x, size_t incx
){
	typedef typename Traits<T>::real_type real_type;
	real_type sum(0);
	while(n --> 0){
		sum += Traits<T>::abs(*x);
		x += incx;
	}
	return sum;
}

///////////////////////////////////////////////////////////////////////
// RotGen
// ------
// Computes the elements of a plane (Givens) rotation matrix such that
//
//     [      c     s ] * [ a ] = [ r ]
//     [ -congj(s)  c ]   [ b ] = [ 0 ]
//
// where r = (a / sqrt(conjg(a)*a)) * sqrt ( conjg(a)*a + conjg(b)*b ).
// The plane rotation can be used to introduce zero elements into a
// matrix selectively.
//
// Arguments
// a     First element of the vector. On exit, it is overwritten with
//       the value r of the rotated vector.
// b     Second element of the vector that is to be zeroed.
// c     The (real) value c in the rotation matrix. This corresponds
//       to the cosine of the angle of rotation.
// s     The value s in the rotation matrix. This corresponds to the
//       sine of the angle of rotation.
//
template <typename T>
void RotGen(
	T *a, const T &b,
	typename Traits<T>::real_type* c, T* s
){
	typedef typename Traits<T>::real_type real_type;
	real_type absa = Traits<T>::abs(*a);
	if(real_type(0) == absa){
		*c = real_type(0);
		*s = T(real_type(1));
		*a = *b;
	}else{
		real_type absb = Traits<T>::abs(*b);
		real_type scale = absa + absb;
		real_type norm = scale * Traits<T>::sqrt( (*a/scale) * (*a/scale) + (*b/scale) * (*b/scale) );
		T alpha = *a / absa;
		*c = absa / norm;
		*s = alpha * Traits<T>::conj(*b) / norm;
		*a = alpha * norm;
	}
}


///////////////////////////////////////////////////////////////////////
// ModifiedRotGen (float)
// ----------------------
// Calls out to BLAS routine srotmg.
//
void ModifiedRotGen(
	float *d1, float *d2, float *x1, const float &x2,
	float param[5]
);

///////////////////////////////////////////////////////////////////////
// ModifiedRotGen (double)
// -----------------------
// Calls out to BLAS routine drotmg.
//
void ModifiedRotGen(
	double *d1, double *d2, double *x1, const double &x2,
	double param[5]
);

///////////////////////////////////////////////////////////////////////
// RotApply
// --------
// Applies a plane rotation to a pair of vectors. It treats each
// corresponding pair of elements of the two vectors as a 2-vector
// upon which to apply the rotation. These will typically be rows
// of a matrix when applying a rotation from the left, or columns of
// a matrix when applying a rotation from the right.
//
// Arguments
// n    Number of elements in the vectors.
// x    Pointer to the first element of the first vector.
// incx Increment between elements of the first vector.
// y    Pointer to the first element of the second vector. This
//      typically corresponds to the row or column of the element
//      being zeroed.
// incy Increment between elements of the second vector.
// c    The (real) cosine of the angle of the rotation.
// s    The sine of the angle of the rotation.
//
template <typename T>
void RotApply(
	size_t n, T* RNP_RESTRICT x, size_t incx,
	T* RNP_RESTRICT y, size_t incy,
	const typename Traits<T>::real_type &c, const T &s
){
	while(n --> 0){
		T temp = c*(*x) + s*(*y);
		*y = c*(*y) - Traits<T>::conj(s)*(*x);
		*x = temp;
		x += incx; y += incy;
	}
}


///////////////////////////////////////////////////////////////////////
// ModifiedRotApply (float)
// ------------------------
// Calls out to BLAS routine srotm.
//
void ModifiedRotApply(
	size_t n, float  *x, size_t incx, float  *y, size_t incy,
	const float  param[5]
);

///////////////////////////////////////////////////////////////////////
// ModifiedRotApply (double)
// -------------------------
// Calls out to BLAS routine drotm.
//
void ModifiedRotApply(
	size_t n, double *x, size_t incx, double *y, size_t incy,
	const double param[5]
);

///////////////////////////////////////////////////////////////////////
// Swap
// ----
// Swaps the elements between two vectors of equal length.
//
// Arguments
// n    Number of elements in the vectors.
// x    Pointer to the first element of the first vector.
// incx Increment between elements of the first vector.
// y    Pointer to the first element of the second vector.
// incy Increment between elements of the second vector.
//
template <typename T>
void Swap(
	size_t n, T* RNP_RESTRICT x, size_t incx,
	T* RNP_RESTRICT y, size_t incy
){
	while(n --> 0){
		std::swap(*x, *y);
		x += incx; y += incy;
	}
}

///////////////////////////////////////////////////////////////////////
// Scale
// -----
// Multiplies each element of a vector by a constant alpha.
//
// Arguments
// n     Number of elements in the vector.
// alpha The scale factor to apply to the vector.
// x     Pointer to the first element of the vector.
// incx  Increment between elements of the vector.
//
template <typename TS, typename T>
void Scale(size_t n, const TS &alpha, T* RNP_RESTRICT x, size_t incx){
	while(n --> 0){
		*x *= alpha;
		x += incx;
	}
}

///////////////////////////////////////////////////////////////////////
// Copy (vector)
// -------------
// Copies a source vector into a destination vector.
//
// Arguments
// n      Number of elements in the vectors.
// src    Pointer to the first element of the source vector.
// incsrc Increment between elements of the source vector.
// dst    Pointer to the first element of the destination vector.
// incdst Increment between elements of the destination vector.
//
template <typename TS, typename TD>
void Copy(
	size_t n, const TS* RNP_RESTRICT src, size_t incsrc,
	TD* RNP_RESTRICT dst, size_t incdst
){
	while(n --> 0){
		*dst = *src;
		src += incsrc; dst += incdst;
	}
}

///////////////////////////////////////////////////////////////////////
// Axpy
// ----
// Adds a multiple alpha of one vector x to another vector y, so that
//
//     y <- alpha*x + y
//
// Arguments
// n      Number of elements in the vectors.
// alpha  Scale factor to apply to x.
// x      Pointer to the first element of x.
// incx   Increment between elements of the x vector.
// y      Pointer to the first element of y.
// incy   Increment between elements of the y vector.
//
template <typename TS, typename T>
void Axpy(
	size_t n, const TS &alpha, const T* RNP_RESTRICT x, size_t incx,
	T* RNP_RESTRICT y, size_t incy
){
	while(n --> 0){
		*y += alpha * (*x);
		x += incx; y += incy;
	}
}

///////////////////////////////////////////////////////////////////////
// Dot
// ---
// Returns the dot product of two vectors (sum of products of
// corresponding elements).
//
// Arguments
// n      Number of elements in the vectors.
// x      Pointer to the first element of the x vector.
// incx   Increment between elements of the x vector.
// y      Pointer to the first element of the y vector.
// incy   Increment between elements of the y vector.
//
template <typename T>
T Dot(
	size_t n, const T* RNP_RESTRICT x, size_t incx,
	const T* RNP_RESTRICT y, size_t incy
){
	T sum(0);
	while(n --> 0){
		sum += ((*x)*(*y));
		x += incx; y += incy;
	}
	return sum;
}

///////////////////////////////////////////////////////////////////////
// DotEx (float to double)
// -----------------------
// Calls out to the BLAS routine dsdot. Computes the dot product to
// double precision and returns it.
//
double DotEx(
	size_t n, const float* RNP_RESTRICT x, size_t incx,
	const float* RNP_RESTRICT y, size_t incy
);

///////////////////////////////////////////////////////////////////////
// DotEx (float to float)
// ----------------------
// Calls out to the BLAS routine sdsdot. Computes the dot product to
// double precision internally, returns it sum with b in float
// precision.
//
float DotEx(
	size_t n, const float &b, const float* RNP_RESTRICT x, size_t incx,
	const float* RNP_RESTRICT y, size_t incy
);


///////////////////////////////////////////////////////////////////////
// ConjugateDot
// ------------
// Returns the conjugate dot product of two vectors (a dot product
// where the first vector x is conjugated).
//
// Arguments
// n      Number of elements in the vectors.
// x      Pointer to the first element of the x vector (conjugated).
// incx   Increment between elements of the x vector.
// y      Pointer to the first element of the y vector.
// incy   Increment between elements of the y vector.
//
template <typename T>
T ConjugateDot(
	size_t n, const T* RNP_RESTRICT x, size_t incx,
	const T* RNP_RESTRICT y, size_t incy
){
	// Don't call BLAS due to return arg problem
	T sum = 0.f;
	while(n --> 0){
		sum += (Traits<T>::conj(*x)*(*y));
		x += incx; y += incy;
	}
	return sum;
}

///////////////////////////////////////////////////////////////////////
// Norm2
// -----
// Returns the 2-norm of a vector (square root of sum of squares of
// absolute values of each element).
//
// Arguments
// n     Number of elements in the vector.
// x     Pointer to the first element of the vector.
// incx  Increment between elements of the vector, incx > 0.
//
template <typename T>
typename Traits<T>::real_type Norm2(size_t n, const T* RNP_RESTRICT x, size_t incx){
	typedef typename Traits<T>::real_type real_type;
	static const real_type rzero(0);
	static const real_type rone(1);
	real_type ssq(1), scale(0);
	while(n --> 0){
		if(rzero != Traits<T>::real(*x)){
			real_type temp = Traits<real_type>::abs(Traits<T>::real(*x));
			if(scale < temp){
				real_type r = scale/temp;
				ssq = ssq*r*r + rone;
				scale = temp;
			}else{
				real_type r = temp/scale;
				ssq += r*r;
			}
		}
		if(rzero != Traits<T>::imag(*x)){
			real_type temp = Traits<real_type>::abs(Traits<T>::imag(*x));
			if(scale < temp){
				real_type r = scale/temp;
				ssq = ssq*r*r + rone;
				scale = temp;
			}else{
				real_type r = temp/scale;
				ssq += r*r;
			}
		}
		x += incx;
	}
	using namespace std;
	return scale*sqrt(ssq);
}

///////////////////////////////////////////////////////////////////////
// Asum
// ----
// Returns the sum of the 1-norms of each element of a vector. This is
// equivalent to Norm1 for real vectors, but for complex vectors, the
// 1-norm of each element is the sum of absolute values of the real
// and imaginary parts.
//
// Arguments
// n     Number of elements in the vector.
// x     Pointer to the first element of the vector.
// incx  Increment between elements of the vector, incx > 0.
//
template <typename T>
typename Traits<T>::real_type Asum(size_t n, const T* RNP_RESTRICT x, size_t incx){
	typedef typename Traits<T>::real_type real_type;
	real_type sum(0);
	while(n --> 0){
		sum += Traits<T>::norm1(*x);
		x += incx;
	}
	return sum;
}

///////////////////////////////////////////////////////////////////////
// MaximumIndex
// ------------
// Returns 0-based index of the element with maximum 1-norm. For
// complex vectors, the 1-norm is the sum of absolute values of the
// real and imaginary parts. This is not equivalent to the BLAS
// routines i_amax, which return a 1-based index.
//
// Arguments
// n     Number of elements in the vector.
// x     Pointer to the first element of the vector.
// incx  Increment between elements of the vector, incx > 0.
//
template <typename T>
size_t MaximumIndex(size_t n, const T* RNP_RESTRICT x, size_t incx){
	if(n < 1){ return 0; }
	typedef typename Traits<T>::real_type real_type;
	real_type mv = Traits<T>::norm1(*x);
	size_t mi = 0;
	for(size_t i = 1; i < n; ++i){
		x += incx;
		real_type cv = Traits<T>::norm1(*x);
		if(cv > mv){ mi = i; mv = cv; }
	}
	return mi;
}

///////////////////////////////////////////////////////////////////////
// MultMV
// ------
// Computes the product of a general rectangular matrix with a vector.
//
//     y <- alpha * op(A) * x + beta * y
//
// Arguments
// trans If "N", op(A) = A. If "T", op(A) = A^T. If "C", op(A) = A^H.
// m     Number of rows of A.
// n     Number of columns of A.
// alpha Scale factor to apply to op(A).
// a     Pointer to the first element of A.
// lda   Leading dimension of the array containing A, lda >= m.
// x     Pointer to the first element of the x vector.
// incx  Increment between elements of the x vector, incx > 0.
// beta  Scale factor to apply to y. Note that if A has a zero
//       dimension, then the scale factor is not applied to y.
//       You must detect this special case and handle it separately.
// y     Pointer to the first element of the y vector.
// incy  Increment between elements of the y vector, incy > 0.
//
void MultMV(const char *trans, size_t m, size_t n,
	const float &alpha, const float *a, size_t lda,
	const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultMV(const char *trans, size_t m, size_t n,
	const double &alpha, const double *a, size_t lda,
	const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
void MultMV(const char *trans, size_t m, size_t n,
	const std::complex<float> &alpha,
	const std::complex<float> *a, size_t lda,
	const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta,
	std::complex<float> *y, size_t incy);
void MultMV(const char *trans, size_t m, size_t n,
	const std::complex<double> &alpha,
	const std::complex<double> *a, size_t lda,
	const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta,
	std::complex<double> *y, size_t incy);

///////////////////////////////////////////////////////////////////////
// MultBandedV
// -----------
// Computes the product of a banded rectangular matrix with a vector.
//
//     y <- alpha * op(A) * x + beta * y
//
// Arguments
// trans If "N", op(A) = A. If "T", op(A) = A^T. If "C", op(A) = A^H.
// m     Number of rows of A.
// n     Number of columns of A.
// kl    The lower bandwidth of A (not counting the diagonal), kl >= 0.
// ku    The upper bandwidth of A (not counting the diagonal), ku >= 0.
// alpha Scale factor to apply to op(A).
// a     Pointer to the first element of A.
// lda   Leading dimension of the array containing A, lda >= m.
// x     Pointer to the first element of the x vector.
// incx  Increment between elements of the x vector, incx > 0.
// beta  Scale factor to apply to y. Note that if A has a zero
//       dimension, then the scale factor is not applied to y.
//       You must detect this special case and handle it separately.
// y     Pointer to the first element of the y vector.
// incy  Increment between elements of the y vector, incy > 0.
//
void MultBandedV(
	const char *trans, size_t m, size_t n, size_t kl, size_t ku,
	const float &alpha, const float *a, size_t lda,
	const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultBandedV(
	const char *trans, size_t m, size_t n, size_t kl, size_t ku,
	const double &alpha, const double *a, size_t lda,
	const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
void MultBandedV(
	const char *trans, size_t m, size_t n, size_t kl, size_t ku,
	const std::complex<float> &alpha,
	const std::complex<float> *a, size_t lda,
	const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta,
	std::complex<float> *y, size_t incy);
void MultBandedV(
	const char *trans, size_t m, size_t n, size_t kl, size_t ku,
	const std::complex<double> &alpha,
	const std::complex<double> *a, size_t lda,
	const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta,
	std::complex<double> *y, size_t incy);

///////////////////////////////////////////////////////////////////////
// MultHermV
// ---------
// Computes the product of a Hermitian square matrix with a vector.
//
//     y <- alpha * A * x + beta * y
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of A.
// alpha Scale factor to apply to A.
// a     Pointer to the first element of A.
// lda   Leading dimension of the array containing A, lda >= n.
// x     Pointer to the first element of the x vector.
// incx  Increment between elements of the x vector, incx > 0.
// beta  Scale factor to apply to y. Note that if A has a zero
//       dimension, then the scale factor is not applied to y.
//       You must detect this special case and handle it separately.
// y     Pointer to the first element of the y vector.
// incy  Increment between elements of the y vector, incy > 0.
//
void MultHermV(const char *uplo, size_t n,
	const float &alpha, const float *a, size_t lda,
	const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultHermV(const char *uplo, size_t n,
	const double &alpha, const double *a, size_t lda,
	const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
void MultHermV(const char *uplo, size_t n,
	const std::complex<float> &alpha,
	const std::complex<float> *a, size_t lda,
	const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta,
	std::complex<float> *y, size_t incy);
void MultHermV(const char *uplo, size_t n,
	const std::complex<double> &alpha,
	const std::complex<double> *a, size_t lda,
	const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta,
	std::complex<double> *y, size_t incy);

///////////////////////////////////////////////////////////////////////
// MultBandedHermV
// ---------------
// Computes the product of a Hermitian banded matrix with a vector.
//
//     y <- alpha * A * x + beta * y
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of A.
// k     Number of sub- or super-diagonals of A, k >= 0.
// alpha Scale factor to apply to A.
// a     Pointer to the first element of A.
// lda   Leading dimension of the array containing A, lda >= n.
// x     Pointer to the first element of the x vector.
// incx  Increment between elements of the x vector, incx > 0.
// beta  Scale factor to apply to y. Note that if A has a zero
//       dimension, then the scale factor is not applied to y.
//       You must detect this special case and handle it separately.
// y     Pointer to the first element of the y vector.
// incy  Increment between elements of the y vector, incy > 0.
//
void MultBandedHermV(const char *uplo, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda,
	const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultBandedHermV(const char *uplo, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda,
	const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
void MultBandedHermV(const char *uplo, size_t n, size_t k,
	const std::complex<float> &alpha,
	const std::complex<float> *a, size_t lda,
	const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta,
	std::complex<float> *y, size_t incy);
void MultBandedHermV(const char *uplo, size_t n, size_t k,
	const std::complex<double> &alpha,
	const std::complex<double> *a, size_t lda,
	const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta,
	std::complex<double> *y, size_t incy);

///////////////////////////////////////////////////////////////////////
// MultPackedHermV
// ---------------
// Computes the product of a Hermitian square matrix with a vector.
// The Hermitian matrix is assumed to be in packed form.
//
//     y <- alpha * A * x + beta * y
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of A.
// alpha Scale factor to apply to A.
// ap    Pointer to the first element of A, length n*(n+1)/2.
//       If uplo = "U", the columns of the upper triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[0,1],
//       and ap[2] is A[1,1], etc.
//       If uplo = "L", the columns of the lower triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[1,0],
//       and ap[2] is A[2,0], etc.
// x     Pointer to the first element of the x vector.
// incx  Increment between elements of the x vector, incx > 0.
// beta  Scale factor to apply to y. Note that if A has a zero
//       dimension, then the scale factor is not applied to y.
//       You must detect this special case and handle it separately.
// y     Pointer to the first element of the y vector.
// incy  Increment between elements of the y vector, incy > 0.
//
void MultPackedHermV(const char *uplo, size_t n,
	const float &alpha, const float *ap,
	const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultPackedHermV(const char *uplo, size_t n,
	const double &alpha, const double *ap,
	const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
void MultPackedHermV(const char *uplo, size_t n,
	const std::complex<float> &alpha,
	const std::complex<float> *ap,
	const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta,
	std::complex<float> *y, size_t incy);
void MultPackedHermV(const char *uplo, size_t n,
	const std::complex<double> &alpha,
	const std::complex<double> *ap,
	const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta,
	std::complex<double> *y, size_t incy);

///////////////////////////////////////////////////////////////////////
// MultSymV
// --------
// Computes the product of a symmetric square matrix with a vector.
//
//     y <- alpha * A * x + beta * y
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of A.
// alpha Scale factor to apply to A.
// a     Pointer to the first element of A.
// lda   Leading dimension of the array containing A, lda >= n.
// x     Pointer to the first element of the x vector.
// incx  Increment between elements of the x vector, incx > 0.
// beta  Scale factor to apply to y. Note that if A has a zero
//       dimension, then the scale factor is not applied to y.
//       You must detect this special case and handle it separately.
// y     Pointer to the first element of the y vector.
// incy  Increment between elements of the y vector, incy > 0.
//
void MultSymV(const char *uplo, size_t n,
	const float &alpha, const float *a, size_t lda,
	const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultSymV(const char *uplo, size_t n,
	const double &alpha, const double *a, size_t lda,
	const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
/* These don't exist in BLAS
void MultSymV(const char *uplo, size_t n,
	const std::complex<float> &alpha,
	const std::complex<float> *a, size_t lda,
	const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta,
	std::complex<float> *y, size_t incy);
void MultSymV(const char *uplo, size_t n,
	const std::complex<double> &alpha,
	const std::complex<double> *a, size_t lda,
	const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta,
	std::complex<double> *y, size_t incy);
*/

///////////////////////////////////////////////////////////////////////
// MultBandedSymV
// --------------
// Computes the product of a symmetric banded matrix with a vector.
//
//     y <- alpha * A * x + beta * y
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of A.
// k     Number of sub- or super-diagonals of A, k >= 0.
// alpha Scale factor to apply to A.
// a     Pointer to the first element of A.
// lda   Leading dimension of the array containing A, lda >= n.
// x     Pointer to the first element of the x vector.
// incx  Increment between elements of the x vector, incx > 0.
// beta  Scale factor to apply to y. Note that if A has a zero
//       dimension, then the scale factor is not applied to y.
//       You must detect this special case and handle it separately.
// y     Pointer to the first element of the y vector.
// incy  Increment between elements of the y vector, incy > 0.
//
void MultBandedSymV(const char *uplo, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda,
	const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultBandedSymV(const char *uplo, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda,
	const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
/* These don't exist in BLAS
void MultBandedSymV(const char *uplo, size_t n, size_t k,
	const std::complex<float> &alpha,
	const std::complex<float> *a, size_t lda,
	const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta,
	std::complex<float> *y, size_t incy);
void MultBandedSymV(const char *uplo, size_t n, size_t k,
	const std::complex<double> &alpha,
	const std::complex<double> *a, size_t lda,
	const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta,
	std::complex<double> *y, size_t incy);
*/

///////////////////////////////////////////////////////////////////////
// MultPackedSymV
// --------------
// Computes the product of a symmetric square matrix with a vector.
// The symmetric matrix is assumed to be in packed form.
//
//     y <- alpha * A * x + beta * y
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of A.
// alpha Scale factor to apply to A.
// ap    Pointer to the first element of A, length n*(n+1)/2.
//       If uplo = "U", the columns of the upper triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[0,1],
//       and ap[2] is A[1,1], etc.
//       If uplo = "L", the columns of the lower triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[1,0],
//       and ap[2] is A[2,0], etc.
// x     Pointer to the first element of the x vector.
// incx  Increment between elements of the x vector, incx > 0.
// beta  Scale factor to apply to y. Note that if A has a zero
//       dimension, then the scale factor is not applied to y.
//       You must detect this special case and handle it separately.
// y     Pointer to the first element of the y vector.
// incy  Increment between elements of the y vector, incy > 0.
//
void MultPackedSymV(const char *uplo, size_t n,
	const float &alpha, const float *ap,
	const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultPackedSymV(const char *uplo, size_t n,
	const double &alpha, const double *ap,
	const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
/* These don't exist in BLAS
void MultPackedSymV(const char *uplo, size_t n,
	const std::complex<float> &alpha,
	const std::complex<float> *ap,
	const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta,
	std::complex<float> *y, size_t incy);
void MultPackedSymV(const char *uplo, size_t n,
	const std::complex<double> &alpha,
	const std::complex<double> *ap,
	const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta,
	std::complex<double> *y, size_t incy);
*/

///////////////////////////////////////////////////////////////////////
// MultTrV
// -------
// Computes the product of a triangular matrix with a vector.
//
//     x <- op(A) * x
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// trans If "N", op(A) = A. If "T", op(A) = A^T. If "C", op(A) = A^H.
// diag  If "U", the diagonal of A is assumed to be all 1's.
//       If "N", the diagonal of A is given.
// n     Number of rows and columns of A.
// a     Pointer to the first element of A.
// lda   Leading dimension of the array containing A, lda >= n.
// x     Pointer to the first element of the x vector. On exit, it is
//       overwritten by op(A) * x.
// incx  Increment between elements of the x vector, incx > 0.
//
void MultTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const float *a, size_t lda, float *x, size_t incx);
void MultTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const double *a, size_t lda, double *x, size_t incx);
void MultTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<float> *a, size_t lda,
	std::complex<float> *x, size_t incx);
void MultTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<double> *a, size_t lda,
	std::complex<double> *x, size_t incx);

///////////////////////////////////////////////////////////////////////
// MultBandedTrV
// -------------
// Computes the product of a banded triangular matrix with a vector.
//
//     x <- op(A) * x
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// trans If "N", op(A) = A. If "T", op(A) = A^T. If "C", op(A) = A^H.
// diag  If "U", the diagonal of A is assumed to be all 1's.
//       If "N", the diagonal of A is given.
// n     Number of rows and columns of A.
// k     Bandwidth of A (not counting the diagonal), k >= 0.
// a     Pointer to the first element of A.
// lda   Leading dimension of the array containing A, lda >= n.
// x     Pointer to the first element of the x vector. On exit, it is
//       overwritten by op(A) * x.
// incx  Increment between elements of the x vector, incx > 0.
//
void MultBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const float *a, size_t lda,
	float *x, size_t incx);
void MultBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const double *a, size_t lda,
	double *x, size_t incx);
void MultBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const std::complex<float> *a, size_t lda,
	std::complex<float> *x, size_t incx);
void MultBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const std::complex<double> *a, size_t lda,
	std::complex<double> *x, size_t incx);

///////////////////////////////////////////////////////////////////////
// MultPackedTrV
// -------------
// Computes the product of a packed triangular matrix with a vector.
//
//     x <- op(A) * x
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// trans If "N", op(A) = A. If "T", op(A) = A^T. If "C", op(A) = A^H.
// diag  If "U", the diagonal of A is assumed to be all 1's.
//       If "N", the diagonal of A is given.
// n     Number of rows and columns of A.
// ap    Pointer to the first element of A, length n*(n+1)/2.
//       If uplo = "U", the columns of the upper triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[0,1],
//       and ap[2] is A[1,1], etc.
//       If uplo = "L", the columns of the lower triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[1,0],
//       and ap[2] is A[2,0], etc.
// lda   Leading dimension of the array containing A, lda >= n.
// x     Pointer to the first element of the x vector. On exit, it is
//       overwritten by op(A) * x.
// incx  Increment between elements of the x vector, incx > 0.
//
void MultPackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const float *ap, float *x, size_t incx);
void MultPackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const double *ap, double *x, size_t incx);
void MultPackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<float> *ap,
	std::complex<float> *x, size_t incx);
void MultPackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<double> *ap,
	std::complex<double> *x, size_t incx);

///////////////////////////////////////////////////////////////////////
// SolveTrV
// --------
// Computes the product of the inverse of a triangular matrix with
// a vector. Solves for y in
//
//     op(A) * y = x
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// trans If "N", op(A) = A. If "T", op(A) = A^T. If "C", op(A) = A^H.
// diag  If "U", the diagonal of A is assumed to be all 1's.
//       If "N", the diagonal of A is given.
// n     Number of rows and columns of A.
// a     Pointer to the first element of A.
// lda   Leading dimension of the array containing A, lda >= n.
// x     Pointer to the first element of the x vector. On exit, it is
//       overwritten by the solution y.
// incx  Increment between elements of the x vector, incx > 0.
//
void SolveTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const float *a, size_t lda, float *x, size_t incx);
void SolveTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const double *a, size_t lda, double *x, size_t incx);
void SolveTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<float> *a, size_t lda,
	std::complex<float> *x, size_t incx);
void SolveTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<double> *a, size_t lda,
	std::complex<double> *x, size_t incx);

///////////////////////////////////////////////////////////////////////
// SolveBandedTrV
// --------------
// Computes the product of the inverse of a banded triangular matrix
// with a vector. Solves for x in
//
//     op(A) * x = y
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// trans If "N", op(A) = A. If "T", op(A) = A^T. If "C", op(A) = A^H.
// diag  If "U", the diagonal of A is assumed to be all 1's.
//       If "N", the diagonal of A is given.
// n     Number of rows and columns of A.
// k     Bandwidth of A (not counting the diagonal), k >= 0.
// a     Pointer to the first element of A.
// lda   Leading dimension of the array containing A, lda >= n.
// x     Pointer to the first element of the y vector. On exit, it is
//       overwritten by the solution x.
// incx  Increment between elements of the x vector, incx > 0.
//
void SolveBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const float *a, size_t lda, float *x, size_t incx);
void SolveBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const double *a, size_t lda, double *x, size_t incx);
void SolveBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const std::complex<float> *a, size_t lda,
	std::complex<float> *x, size_t incx);
void SolveBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const std::complex<double> *a, size_t lda,
	std::complex<double> *x, size_t incx);

///////////////////////////////////////////////////////////////////////
// SolvePackedTrV
// --------------
// Computes the product of the inverse of a packed triangular matrix
// with a vector. Solves for x in
//
//     op(A) * x = y
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// trans If "N", op(A) = A. If "T", op(A) = A^T. If "C", op(A) = A^H.
// diag  If "U", the diagonal of A is assumed to be all 1's.
//       If "N", the diagonal of A is given.
// n     Number of rows and columns of A.
// ap    Pointer to the first element of A, length n*(n+1)/2.
//       If uplo = "U", the columns of the upper triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[0,1],
//       and ap[2] is A[1,1], etc.
//       If uplo = "L", the columns of the lower triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[1,0],
//       and ap[2] is A[2,0], etc.
// lda   Leading dimension of the array containing A, lda >= n.
// x     Pointer to the first element of the x vector. On exit, it is
//       overwritten by op(A) * x.
// incx  Increment between elements of the x vector, incx > 0.
//
void SolvePackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const float *ap, float *x, size_t incx);
void SolvePackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const double *ap, double *x, size_t incx);
void SolvePackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<float> *ap,
	std::complex<float> *x, size_t incx);
void SolvePackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<double> *ap,
	std::complex<double> *x, size_t incx);

///////////////////////////////////////////////////////////////////////
// Rank1Update
// -----------
// Computes a rank-1 update to a general rectangular matrix.
//
//     A <- alpha * x * y^T + A
//
// Arguments
// m     Number of rows of A.
// n     Number of columns of A.
// alpha Scale factor to apply to the rank-1 update.
// x     Pointer to the first element of the x vector, length m.
// incx  Increment between elements of the x vector, incx > 0.
// y     Pointer to the first element of the y vector, length n.
// incy  Increment between elements of the y vector, incy > 0.
// a     Pointer to the first element of the matrix A.
// lda   Leading dimension of the array containing A, lda >= m.
//
void Rank1Update(size_t m, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy,
	float *a, size_t lda);
void Rank1Update(size_t m, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy,
	double *a, size_t lda);
void Rank1Update(size_t m, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx,
	const std::complex<float> *y, size_t incy,
	std::complex<float> *a, size_t lda);
void Rank1Update(size_t m, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx,
	const std::complex<double> *y, size_t incy,
	std::complex<double> *a, size_t lda);

///////////////////////////////////////////////////////////////////////
// ConjugateRank1Update
// --------------------
// Computes a rank-1 update to a general rectangular matrix, where the
// row vector is conjugated.
//
//     A <- alpha * x * y^H + A
//
// Arguments
// m     Number of rows of A.
// n     Number of columns of A.
// alpha Scale factor to apply to the rank-1 update.
// x     Pointer to the first element of the x vector, length m.
// incx  Increment between elements of the x vector, incx > 0.
// y     Pointer to the first element of the y vector (conjugated),
//       length n.
// incy  Increment between elements of the y vector, incy > 0.
// a     Pointer to the first element of the matrix A.
// lda   Leading dimension of the array containing A, lda >= m.
//
void ConjugateRank1Update(size_t m, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy,
	float *a, size_t lda);
void ConjugateRank1Update(size_t m, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy,
	double *a, size_t lda);
void ConjugateRank1Update(size_t m, size_t n,
	const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx,
	const std::complex<float> *y, size_t incy,
	std::complex<float> *a, size_t lda);
void ConjugateRank1Update(size_t m, size_t n,
	const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx,
	const std::complex<double> *y, size_t incy,
	std::complex<double> *a, size_t lda);

///////////////////////////////////////////////////////////////////////
// HermRank1Update
// ---------------
// Computes a rank-1 update to a Hermitian square matrix, where the
// row vector is conjugated.
//
//     A <- alpha * x * x^H + A
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of A.
// alpha Scale factor to apply to the rank-1 update.
// x     Pointer to the first element of the x vector, length n.
// incx  Increment between elements of the x vector, incx > 0.
// a     Pointer to the first element of the matrix A.
// lda   Leading dimension of the array containing A, lda >= n.
//
void HermRank1Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, float *a, size_t lda);
void HermRank1Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, double *a, size_t lda);
void HermRank1Update(const char *uplo, size_t n, const float &alpha,
	const std::complex<float> *x, size_t incx,
	std::complex<float> *a, size_t lda);
void HermRank1Update(const char *uplo, size_t n, const double &alpha,
	const std::complex<double> *x, size_t incx,
	std::complex<double> *a, size_t lda);

///////////////////////////////////////////////////////////////////////
// PackedHermRank1Update
// ---------------------
// Computes a rank-1 update to a Hermitian square matrix, where the
// row vector is conjugated. The matrix is stored in packed format.
//
//     A <- alpha * x * x^H + A
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of A.
// alpha Scale factor to apply to the rank-1 update.
// x     Pointer to the first element of the x vector, length n.
// incx  Increment between elements of the x vector, incx > 0.
// ap    Pointer to the first element of A, length n*(n+1)/2.
//       If uplo = "U", the columns of the upper triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[0,1],
//       and ap[2] is A[1,1], etc.
//       If uplo = "L", the columns of the lower triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[1,0],
//       and ap[2] is A[2,0], etc.
//
void PackedHermRank1Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, float *ap);
void PackedHermRank1Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, double *ap);
void PackedHermRank1Update(const char *uplo, size_t n, const float &alpha,
	const std::complex<float> *x, size_t incx, std::complex<float> *ap);
void PackedHermRank1Update(const char *uplo, size_t n, const double &alpha,
	const std::complex<double> *x, size_t incx, std::complex<double> *ap);

///////////////////////////////////////////////////////////////////////
// HermRank2Update
// ---------------
// Computes a rank-2 update to a Hermitian square matrix.
//
//     A <- alpha * x * y^H + conj(alpha) * y * x^H + A
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of A.
// alpha Scale factor to apply to the rank-1 updates.
// x     Pointer to the first element of the x vector, length n.
// incx  Increment between elements of the x vector, incx > 0.
// y     Pointer to the first element of the y vector, length n.
// incy  Increment between elements of the y vector, incy > 0.
// a     Pointer to the first element of the matrix A.
// lda   Leading dimension of the array containing A, lda >= n.
//
void HermRank2Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy,
	float *a, size_t lda);
void HermRank2Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy,
	double *a, size_t lda);
void HermRank2Update(const char *uplo, size_t n,
	const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx,
	const std::complex<float> *y, size_t incy,
	std::complex<float> *a, size_t lda);
void HermRank2Update(const char *uplo, size_t n,
	const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx,
	const std::complex<double> *y, size_t incy,
	std::complex<double> *a, size_t lda);

///////////////////////////////////////////////////////////////////////
// PackedHermRank2Update
// ---------------------
// Computes a rank-2 update to a Hermitian square matrix.
// The matrix is stored in packed format.
//
//     A <- alpha * x * y^H + conj(alpha) * y * x^H + A
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of A.
// alpha Scale factor to apply to the rank-1 updates.
// x     Pointer to the first element of the x vector, length n.
// incx  Increment between elements of the x vector, incx > 0.
// y     Pointer to the first element of the y vector, length n.
// incy  Increment between elements of the y vector, incy > 0.
// ap    Pointer to the first element of A, length n*(n+1)/2.
//       If uplo = "U", the columns of the upper triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[0,1],
//       and ap[2] is A[1,1], etc.
//       If uplo = "L", the columns of the lower triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[1,0],
//       and ap[2] is A[2,0], etc.
//
void PackedHermRank2Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy, float *ap);
void PackedHermRank2Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy, double *ap);
void PackedHermRank2Update(const char *uplo, size_t n,
	const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx, const std::complex<float> *y,
	size_t incy, std::complex<float> *ap);
void PackedHermRank2Update(const char *uplo, size_t n,
	const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx, const std::complex<double> *y,
	size_t incy, std::complex<double> *ap);

///////////////////////////////////////////////////////////////////////
// SymRank1Update
// --------------
// Computes a rank-1 update to a symmetric square matrix.
//
//     A <- alpha * x * x^T + A
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of A.
// alpha Scale factor to apply to the rank-1 update.
// x     Pointer to the first element of the x vector, length n.
// incx  Increment between elements of the x vector, incx > 0.
// a     Pointer to the first element of the matrix A.
// lda   Leading dimension of the array containing A, lda >= n.
//
void SymRank1Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, float *a, size_t lda);
void SymRank1Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, double *a, size_t lda);
/*
void SymRank1Update(const char *uplo, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx, std::complex<float> *a, size_t lda);
void SymRank1Update(const char *uplo, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx, std::complex<double> *a, size_t lda);
*/

///////////////////////////////////////////////////////////////////////
// PackedSymRank1Update
// --------------------
// Computes a rank-1 update to a symmetric square matrix.
// The matrix is stored in packed format.
//
//     A <- alpha * x * x^T + A
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of A.
// alpha Scale factor to apply to the rank-1 update.
// x     Pointer to the first element of the x vector, length n.
// incx  Increment between elements of the x vector, incx > 0.
// ap    Pointer to the first element of A, length n*(n+1)/2.
//       If uplo = "U", the columns of the upper triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[0,1],
//       and ap[2] is A[1,1], etc.
//       If uplo = "L", the columns of the lower triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[1,0],
//       and ap[2] is A[2,0], etc.
//
void PackedSymRank1Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, float *a);
void PackedSymRank1Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, double *a);
/*
void PackedSymRank1Update(const char *uplo, size_t n,
	const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx, std::complex<float> *a);
void PackedSymRank1Update(const char *uplo, size_t n,
	const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx, std::complex<double> *a);
*/

///////////////////////////////////////////////////////////////////////
// SymRank2Update
// --------------
// Computes a rank-2 update to a symmetric square matrix.
//
//     A <- alpha * x * y^T + alpha * y * x^T + A
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of A.
// alpha Scale factor to apply to the rank-1 updates.
// x     Pointer to the first element of the x vector, length n.
// incx  Increment between elements of the x vector, incx > 0.
// y     Pointer to the first element of the y vector, length n.
// incy  Increment between elements of the y vector, incy > 0.
// a     Pointer to the first element of the matrix A.
// lda   Leading dimension of the array containing A, lda >= n.
//
void SymRank2Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy,
	float *a, size_t lda);
void SymRank2Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy,
	double *a, size_t lda);
/*
void SymRank2Update(const char *uplo, size_t n,
	const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx,
	const std::complex<float> *y, size_t incy,
	std::complex<float> *a, size_t lda);
void SymRank2Update(const char *uplo, size_t n,
	const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx,
	const std::complex<double> *y, size_t incy,
	std::complex<double> *a, size_t lda);
*/

///////////////////////////////////////////////////////////////////////
// PackedSymRank2Update
// --------------------
// Computes a rank-2 update to a symmetric square matrix.
// The matrix is stored in packed format.
//
//     A <- alpha * x * y^T + alpha * y * x^T + A
//
// Arguments
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of A.
// alpha Scale factor to apply to the rank-1 updates.
// x     Pointer to the first element of the x vector, length n.
// incx  Increment between elements of the x vector, incx > 0.
// y     Pointer to the first element of the y vector, length n.
// incy  Increment between elements of the y vector, incy > 0.
// ap    Pointer to the first element of A, length n*(n+1)/2.
//       If uplo = "U", the columns of the upper triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[0,1],
//       and ap[2] is A[1,1], etc.
//       If uplo = "L", the columns of the lower triangle of A are
//       stored sequentially, so ap[0] is A[0,0], ap[1] is A[1,0],
//       and ap[2] is A[2,0], etc.
//
void PackedSymRank2Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy, float *ap);
void PackedSymRank2Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy, double *ap);
/*
void PackedSymRank2Update(const char *uplo, size_t n,
const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx,
	const std::complex<float> *y, size_t incy, std::complex<float> *ap);
void PackedSymRank2Update(const char *uplo, size_t n,
const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx,
	const std::complex<double> *y, size_t incy, std::complex<double> *ap);
*/


///////////////////////////////////////////////////////////////////////
// MultMM
// ------
// Computes the product of two general rectangular matrices.
//
//     C = alpha * op(A) * op(B) + beta * C
//
// Arguments
// transa If "N", op(A) = A. If "T", op(A) = A^T. If "C", op(A) = A^H.
// transb If "N", op(B) = B. If "T", op(B) = B^T. If "C", op(B) = B^H.
// m      Number of rows of C, and number of rows of op(A).
// n      Number of columns of C, and number of columns of op(B).
// k      Number of columns of op(A), and number of rows of op(B).
// alpha  Scale factor to apply to the product.
// a      Pointer to the first element of A.
// lda    Leading dimension of the array containing A.
//        If transa = "N", lda >= m, otherwise lda >= k.
// b      Pointer to the first element of B.
// ldb    Leading dimension of the array containing B.
//        If transb = "N", ldb >= k, otherwise lda >= n.
// beta   Scale factor to apply to C.
// c      Pointer to the first element of C.
// ldc    Leading dimension of the array containing C, ldc >= m.
//
void MultMM(
	const char *transa, const char *transb, size_t m, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda, const float *b, size_t ldb,
	const float &beta, float *c, size_t ldc);
void MultMM(
	const char *transa, const char *transb, size_t m, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda, const double *b, size_t ldb,
	const double &beta, double *c, size_t ldc);
void MultMM(
	const char *transa, const char *transb, size_t m, size_t n, size_t k,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda,
	const std::complex<float> *b, size_t ldb,
	const std::complex<float> &beta, std::complex<float> *c, size_t ldc);
void MultMM(
	const char *transa, const char *transb, size_t m, size_t n, size_t k,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda,
	const std::complex<double> *b, size_t ldb,
	const std::complex<double> &beta, std::complex<double> *c, size_t ldc);

///////////////////////////////////////////////////////////////////////
// MultSymM
// --------
// Computes the product of a symmetric square matrix with a general
// rectangular matrix
//
//     C = alpha * A * B + beta * C     if side = "L"
//     C = alpha * B * A + beta * C     if side = "R"
//
// Arguments
// side   If "L", the symmetric matrix A is applied from the left.
//        If "R", the symmetric matrix A is applied from the right.
// uplo   If "U", the upper triangle of C is given.
//        If "L", the upper triangle of C is given.
// m      Number of rows of C.
// n      Number of columns of C.
// alpha  Scale factor to apply to the product.
// a      Pointer to the first element of the symmetric matrix A.
// lda    Leading dimension of the array containing A.
//        If side = "L", lda >= m, otherwise lda >= n.
// b      Pointer to the first element of B.
// ldb    Leading dimension of the array containing B, ldb >= m.
// beta   Scale factor to apply to C.
// c      Pointer to the first element of C.
// ldc    Leading dimension of the array containing C, ldc >= m.
//
void MultSymM(const char *side, const char *uplo, size_t m, size_t n,
	const float &alpha, const float *a, size_t lda, const float *b, size_t ldb,
	const float &beta, float *c, size_t ldc);
void MultSymM(const char *side, const char *uplo, size_t m, size_t n,
	const double &alpha, const double *a, size_t lda, const double *b, size_t ldb,
	const double &beta, double *c, size_t ldc);
void MultSymM(const char *side, const char *uplo, size_t m, size_t n,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda,
	const std::complex<float> *b, size_t ldb,
	const std::complex<float> &beta, std::complex<float> *c, size_t ldc);
void MultSymM(const char *side, const char *uplo, size_t m, size_t n,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda,
	const std::complex<double> *b, size_t ldb,
	const std::complex<double> &beta, std::complex<double> *c, size_t ldc);

///////////////////////////////////////////////////////////////////////
// MultHermM
// ---------
// Computes the product of a Hermitian square matrix with a general
// rectangular matrix
//
//     C = alpha * A * B + beta * C     if side = "L"
//     C = alpha * B * A + beta * C     if side = "R"
//
// Arguments
// side   If "L", the Hermitian matrix A is applied from the left.
//        If "R", the Hermitian matrix A is applied from the right.
// uplo   If "U", the upper triangle of C is given.
//        If "L", the upper triangle of C is given.
// m      Number of rows of C.
// n      Number of columns of C.
// alpha  Scale factor to apply to the product.
// a      Pointer to the first element of the Hermitian matrix A.
// lda    Leading dimension of the array containing A.
//        If side = "L", lda >= m, otherwise lda >= n.
// b      Pointer to the first element of B.
// ldb    Leading dimension of the array containing B, ldb >= m.
// beta   Scale factor to apply to C.
// c      Pointer to the first element of C.
// ldc    Leading dimension of the array containing C, ldc >= m.
//
void MultHermM(const char *side, const char *uplo, size_t m, size_t n,
	const float &alpha, const float *a, size_t lda, const float *b, size_t ldb,
	const float &beta, float *c, size_t ldc);
void MultHermM(const char *side, const char *uplo, size_t m, size_t n,
	const double &alpha, const double *a, size_t lda, const double *b, size_t ldb,
	const double &beta, double *c, size_t ldc);
void MultHermM(const char *side, const char *uplo, size_t m, size_t n,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda,
	const std::complex<float> *b, size_t ldb,
	const std::complex<float> &beta, std::complex<float> *c, size_t ldc);
void MultHermM(const char *side, const char *uplo, size_t m, size_t n,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda,
	const std::complex<double> *b, size_t ldb,
	const std::complex<double> &beta, std::complex<double> *c, size_t ldc);

///////////////////////////////////////////////////////////////////////
// SymRankKUpdate
// --------------
// Computes a rank-k update to a symmetric square matrix.
//
//     C <- alpha * op(A) * op(A)^T + beta * C
//
// Arguments
// uplo  If "U", the upper triangle of C is given.
//       If "L", the lower triangle of C is given.
// trans If "N", op(A) = A. If "T", op(A) = A^T. If "C", op(A) = A^H.
// n     Number of rows and columns of C.
// k     If trans = "N", k is the number of columns of A, otherwise
//       it is the number of rows of A.
// alpha Scale factor to apply to the rank-1 updates.
// a     Pointer to the first element of the matrix A.
// lda   Leading dimension of the array containing A.
//       If trans = "N", lda >= n, otherwise lda >= k.
// beta  Scale factor to apply to C.
// c     Pointer to the first element of the matrix C.
// ldc   Leading dimension of the array containing C, ldc >= n.
//
void SymRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda,
	const float &beta, float *c, size_t ldc);
void SymRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda,
	const double &beta, double *c, size_t ldc);
void SymRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda,
	const std::complex<float> &beta, std::complex<float> *c, size_t ldc);
void SymRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda,
	const std::complex<double> &beta, std::complex<double> *c, size_t ldc);

///////////////////////////////////////////////////////////////////////
// HermRankKUpdate
// ---------------
// Computes a rank-k update to a Hermitian square matrix.
//
//     C <- alpha * op(A) * op(A)^H + beta * C
//
// Arguments
// uplo  If "U", the upper triangle of C is given.
//       If "L", the lower triangle of C is given.
// trans If "N", op(A) = A. If "C", op(A) = A^H.
// n     Number of rows and columns of C.
// k     If trans = "N", k is the number of columns of A, otherwise
//       it is the number of rows of A.
// alpha Scale factor to apply to the rank-1 updates.
// a     Pointer to the first element of the matrix A.
// lda   Leading dimension of the array containing A.
//       If trans = "N", lda >= n, otherwise lda >= k.
// beta  Scale factor to apply to C.
// c     Pointer to the first element of the matrix C.
// ldc   Leading dimension of the array containing C, ldc >= n.
//
void HermRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda,
	const float &beta, float *c, size_t ldc);
void HermRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda,
	const double &beta, double *c, size_t ldc);
void HermRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const float &alpha, const std::complex<float> *a, size_t lda,
	const float &beta, std::complex<float> *c, size_t ldc);
void HermRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const double &alpha, const std::complex<double> *a, size_t lda,
	const double &beta, std::complex<double> *c, size_t ldc);

///////////////////////////////////////////////////////////////////////
// SymRank2KUpdate
// ---------------
// Computes a rank-2k update to a symmetric square matrix.
//
//     C <- alpha * A * B^T + alpha * B * A^T + beta * C
//
// or
//
//     C <- alpha * A^T * B + alpha * B^T * A + beta * C
//
// Arguments
// uplo  If "U", the upper triangle of C is given.
//       If "L", the lower triangle of C is given.
// trans If "N", then C <- alpha * A * B^T + alpha * B * A^T + beta * C
//       If "T", then C <- alpha * A^T * B + alpha * B^T * A + beta * C
// n     Number of rows and columns of C.
// k     If trans = "N", k is the number of columns of A and B,
//       otherwise it is the number of rows of A and B.
// alpha Scale factor to apply to the rank-1 updates.
// a     Pointer to the first element of the matrix A.
// lda   Leading dimension of the array containing A.
//       If trans = "N", lda >= n, otherwise lda >= k.
// b     Pointer to the first element of the matrix B.
// ldb   Leading dimension of the array containing B.
//       If trans = "N", lda >= n, otherwise ldb >= k.
// beta  Scale factor to apply to C.
// c     Pointer to the first element of the matrix C.
// ldc   Leading dimension of the array containing C, ldc >= n.
//
void SymRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda,
	const float *b, size_t ldb,
	const float &beta, float *c, size_t ldc);
void SymRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda,
	const double *b, size_t ldb,
	const double &beta, double *c, size_t ldc);
void SymRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda,
	const std::complex<float> *b, size_t ldb,
	const std::complex<float> &beta, std::complex<float> *c, size_t ldc);
void SymRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda,
	const std::complex<double> *b, size_t ldb,
	const std::complex<double> &beta, std::complex<double> *c, size_t ldc);

///////////////////////////////////////////////////////////////////////
// HermRank2KUpdate
// ----------------
// Computes a rank-2k update to a Hermitian square matrix.
//
//     C <- alpha * A * B^H + conj(alpha) * B * A^H + beta * C
//
// or
//
//     C <- alpha * A^H * B + conj(alpha) * B^H * A + beta * C
//
// Arguments
// uplo  If "U", the upper triangle of C is given.
//       If "L", the lower triangle of C is given.
// trans If "N", then C <- alpha * A * B^H + conj(alpha) * B * A^H + beta * C
//       If "C", then C <- alpha * A^H * B + conj(alpha) * B^H * A + beta * C
// n     Number of rows and columns of C.
// k     If trans = "N", k is the number of columns of A and B,
//       otherwise it is the number of rows of A and B.
// alpha Scale factor to apply to the rank-1 updates.
// a     Pointer to the first element of the matrix A.
// lda   Leading dimension of the array containing A.
//       If trans = "N", lda >= n, otherwise lda >= k.
// b     Pointer to the first element of the matrix B.
// ldb   Leading dimension of the array containing B.
//       If trans = "N", lda >= n, otherwise ldb >= k.
// beta  Scale factor to apply to C.
// c     Pointer to the first element of the matrix C.
// ldc   Leading dimension of the array containing C, ldc >= n.
//
void HermRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda,
	const float *b, size_t ldb,
	const float &beta, float *c, size_t ldc);
void HermRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda,
	const double *b, size_t ldb,
	const double &beta, double *c, size_t ldc);
void HermRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda,
	const std::complex<float> *b, size_t ldb,
	const float &beta, std::complex<float> *c, size_t ldc);
void HermRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda,
	const std::complex<double> *b, size_t ldb,
	const double &beta, std::complex<double> *c, size_t ldc);

///////////////////////////////////////////////////////////////////////
// MultTrM
// --------
// Computes the product of a triangular matrix with a general
// rectangular matrix.
//
//     B <- alpha * op(A) * B     or     B <- alpha * B * op(A)
//
// Arguments
// side   If "L", then op(A) is applied from the left.
//        If "R", then op(A) is applied from the right.
// uplo   If "U", then A is upper triangular.
//        If "L", then A is upper triangular.
// trans  If "N", op(A) = A. If "T", op(A) = A^T. If "C", op(A) = A^H.
// diag   If "U", the diagonal of A is assumed to be all 1's.
//        If "N", the digonal of A is given.
// m      Number of rows of B, and the dimension of A when side = "L".
// n      Number of columns of B, and the dimension of A when
//        side = "R".
// alpha  Scale factor to apply.
// a      Pointer to the first element of A.
// lda    Leading dimension of the array containing A.
//        If trans = "N", lda >= m, otherwise lda >= n.
// b      Pointer to the first element of B. On exit, B is overwritten
//        with the product.
// ldb    Leading dimension of the array containing B, ldb >= m.
//
void MultTrM(
	const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const float &alpha, const float *a, size_t lda,
	float *b, size_t ldb);
void MultTrM(
	const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const double &alpha, const double *a, size_t lda,
	double *b, size_t ldb);
void MultTrM(
	const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *a, size_t lda,
	std::complex<float> *b, size_t ldb);
void MultTrM(
	const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *a, size_t lda,
	std::complex<double> *b, size_t ldb);

///////////////////////////////////////////////////////////////////////
// SolveTrM
// --------
// Computes the product of the inverse of a triangular matrix with a
// general rectangular matrix. Solves:
//
//     op(A) * X = alpha * B     or     X * op(A) = alpha * B
//
// Arguments
// side   If "L", then op(A) and its inverse is applied from the left.
//        If "R", then op(A) and its inverse is applied from the right.
// uplo   If "U", then A is upper triangular.
//        If "L", then A is upper triangular.
// trans  If "N", op(A) = A. If "T", op(A) = A^T. If "C", op(A) = A^H.
// diag   If "U", the diagonal of A is assumed to be all 1's.
//        If "N", the digonal of A is given.
// m      Number of rows of B, and the dimension of A when side = "L".
// n      Number of columns of B, and the dimension of A when
//        side = "R".
// alpha  Scale factor to apply.
// a      Pointer to the first element of A.
// lda    Leading dimension of the array containing A.
//        If trans = "N", lda >= m, otherwise lda >= n.
// b      Pointer to the first element of B. On exit, B is overwritten
//        with the result X.
// ldb    Leading dimension of the array containing B, ldb >= m.
//
void SolveTrM(
	const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const float &alpha, const float *a, size_t lda,
	float *b, size_t ldb);
void SolveTrM(
	const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const double &alpha, const double *a, size_t lda,
	double *b, size_t ldb);
void SolveTrM(
	const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *a, size_t lda,
	std::complex<float> *b, size_t ldb);
void SolveTrM(
	const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *a, size_t lda,
	std::complex<double> *b, size_t ldb);

#undef RNP_RESTRICT

} // namespace BLAS
} // namespace RNP

#endif // RNP_BLAS_MIX_HPP_INCLUDED
