#ifndef RNP_BLAS_MIX_HPP_INCLUDED
#define RNP_BLAS_MIX_HPP_INCLUDED

#include <complex>
#include <cstring>
#include <RNP/Types.hpp>

#define RNP_RESTRICT __restrict

////// Name mappings from fortran BLAS:
//
//// Level 1
//
// srotg,  drotg,  crotg,  zrotg  : RotGen
// srotmg, drotmg                 : ModifiedRotGen
// srot,   drot                   : RotApply
// srotm,  drotm                  : ModifiedRotApply
// sswap,  dswap,  cswap,  zswap  : Swap
// sscal,  dscal,  cscal,  zscal  : Scale
// scopy,  dcopy,  ccopy,  zcopy  : Copy
// saxpy,  daxpy,  caxpy,  zaxpy  : Axpy
// sdot,   ddot,   cdotu,  zdotu  : Dot
// dsdot,  sdsdot                 : DotEx
//                 cdotc,  zdotc  : ConjugateDot
// snrm2,  dnrm2,  scnrm2, dznrm2 : Norm2
// sasum,  dasum,  scasum, dzasum : Asum
// isamax, idamax, icamax, izamax : MaximumIndex
//
//// Level 2
//
// sgemv, dgemv, cgemv, zgemv     : MultMV
// sgbmv, dgbmv, cgbmv, zgbmv     : MultBandedV
//               chemv, zhemv     : MultHermV
//               chbmv, zhbmv     : MultBandedHermV
//               chpmv, zhpmv     : MultPackedHermV
// ssymv, dsymv                   : MultSymV
// ssbmv, dsbmv                   : MultBandedSymV
// sspmv, dspmv                   : MultPackedSymV
// strmv, dtrmv, ctrmv, ztrmv     : MultTrV
// stbmv, dtbmv, ctbmv, ztbmv     : MultBandedTrV
// stpmv, dtpmv, ctpmv, ztpmv     : MultPackedTrV
// strsv, dtrsv, ctrsv, ztrsv     : SolveTrV
// stbsv, dtbsv, ctbsv, ztbsv     : SolveBandedTrV
// stpsv, dtpsv, ctpsv, ztpsv     : SolvePackedTrV
//
// sger,  dger, cgeru, zgeru      : Rank1Update
//              cgerc, zgerc      : ConjugateRank1Update
//              cher,  zher       : HermRank1Update
//              chpr,  zhpr       : PackedHermRank1Update
//              cher2, zher2      : HermRank2Update
//              chpr2, zhpr2      : PackedHermRank2Update
// ssyr,  dsyr                    : SymRank1Update
// sspr,  dspr                    : PackedSymRank1Update
// ssyr2, dsyr2                   : SymRank2Update
// sspr2, dspr2                   : PackedSymRank2Update
//
//// Level 3
//
// sgemm,  dgemm,  cgemm,  zgemm  : MultMM
// ssymm,  dsymm,  csymm,  zsymm  : MultSyM
//                 chemm,  zhemm  : MultHermM
// ssyrk,  dsyrk,  csyrk,  zsyrk  : SymRankKUpdate
//                 cherk,  zherk  : HermRankKUpdate
// ssyr2k, dsyr2k, csyr2k, zsyr2k : SymRank2KUpdate
//                 cher2k, zher2k : HermRank2KUpdate
// strmm,  dtrmm,  ctrmm,  ztrmm  : MultTrM
// strsm,  dtrsm,  ctrsm,  ztrsm  : SolveTrM
//
////// Extra routines
//
//// Level 1/2
// Set       : Sets entries in a vector or (parts of a) matrix (_laset)
// Copy      : Copies a matrix, possilby transposed
// Conjugate : Conjugates a vector (_lacgv)
// Rescale   : Rescales a matrix (_lascl)
// Norm1     : True 1-norm of a vector
//

namespace RNP{
namespace BLAS{

template <typename T>
static void Set(size_t n, const T &val, T* RNP_RESTRICT x, size_t incx){
	while(n --> 0){ *x = val; x += incx; }
}

template <typename T>
void Set(size_t m, size_t n, const T &offdiag, const T &diag, T* RNP_RESTRICT a, size_t lda){
	for(size_t j = 0; j < n; ++j){
		for(size_t i = 0; i < m; ++i){
			a[i+j*lda] = (i == j ? diag : offdiag);
		}
	}
}

template <typename T>
void Copy(size_t m, size_t n, const T* RNP_RESTRICT src, size_t ldsrc, T* RNP_RESTRICT dst, size_t lddst){
	if(m == ldsrc && m == lddst){
		memcpy(dst, src, sizeof(T) * m*n);
	}else{
		for(size_t j = 0; j < n; ++j){
			memcpy(&dst[j*lddst], &src[j*lddst], sizeof(T) * m);
		}
	}
}

template <typename T>
void Copy(const char *trans, size_t m, size_t n, const T* RNP_RESTRICT src, size_t ldsrc, T* RNP_RESTRICT dst, size_t lddst){
	if('N' == trans[0]){
		Copy_matrix_generic(m, n, src, ldsrc, dst, lddst);
	}else{
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				dst[j+i*lddst] = src[i+j*ldsrc];
			}
		}
	}
}

template <typename T>
void Conjugate(size_t n, T* RNP_RESTRICT x, size_t incx){
	while(n --> 0){
		*x = Traits<T>::conj(*x);
		x += incx;
	}
}

template <typename TS, typename T>
void Rescale(const char *type, size_t kl, size_t ku, const TS &cfrom, const TS &cto, size_t m, size_t n, T* RNP_RESTRICT a, size_t lda){
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

template <typename T>
typename Traits<T>::real_type Norm1(size_t n, const T* RNP_RESTRICT x, size_t incx){
	typedef typename Traits<T>::real_type real_type;
	real_type sum(0);
	while(n --> 0){
		sum += Traits<T>::abs(*x);
		x += incx;
	}
	return sum;
}

// Level 1 rotations

template <typename T>
void RotGen(T* RNP_RESTRICT a, const T &b, typename Traits<T>::real_type* RNP_RESTRICT c, T* RNP_RESTRICT s){
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

void ModifiedRotGen(float  *d1, float  *d2, float  *x1, const float  &x2, float  param[5]);
void ModifiedRotGen(double *d1, double *d2, double *x1, const double &x2, double param[5]);

template <typename T>
void RotApply(size_t n, T* RNP_RESTRICT x, size_t incx, T* RNP_RESTRICT y, size_t incy, const typename Traits<T>::real_type &c, const T &s){
	while(n --> 0){
		T temp = c*(*x) + s*(*y);
		*y = c*(*y) - Traits<T>::conj(s)*(*x);
		*x = temp;
		x += incx; y += incy;
	}
}

void ModifiedRotApply(size_t n, float  *x, size_t incx, float  *y, size_t incy, const float  param[5]);
void ModifiedRotApply(size_t n, double *x, size_t incx, double *y, size_t incy, const double param[5]);

// Level 1 utility
template <typename T>
void Swap(size_t n, T* RNP_RESTRICT x, size_t incx, T* RNP_RESTRICT y, size_t incy){
	while(n --> 0){
		std::swap(*x, *y);
		x += incx; y += incy;
	}
}

template <typename TS, typename T>
void Scale(size_t n, const TS &alpha, T* RNP_RESTRICT x, size_t incx){
	while(n --> 0){
		*x *= alpha;
		x += incx;
	}
}

template <typename TS, typename T>
void Copy(size_t n, const TS* RNP_RESTRICT src, size_t incsrc, T* RNP_RESTRICT dst, size_t incdst){
	while(n --> 0){
		*dst = *src;
		src += incsrc; dst += incdst;
	}
}

template <typename TS, typename T>
void Axpy(size_t n, const TS &alpha, const T* RNP_RESTRICT x, size_t incx, T* RNP_RESTRICT y, size_t incy){
	while(n --> 0){
		*y += alpha * (*x);
		x += incx; y += incy;
	}
}

template <typename T>
T Dot(size_t n, const T* RNP_RESTRICT x, size_t incx, const T* RNP_RESTRICT y, size_t incy){
	T sum(0);
	while(n --> 0){
		sum += ((*x)*(*y));
		x += incx; y += incy;
	}
	return sum;
}

double DotEx(size_t n, const float* RNP_RESTRICT x, size_t incx, const float* RNP_RESTRICT y, size_t incy);
float DotEx(size_t n, const float &b, const float* RNP_RESTRICT x, size_t incx, const float* RNP_RESTRICT y, size_t incy);

template <typename T>
T ConjugateDot(size_t n, const T* RNP_RESTRICT x, size_t incx, const T* RNP_RESTRICT y, size_t incy){
	// Don't call BLAS due to return arg problem
	T sum = 0.f;
	while(n --> 0){
		sum += (Traits<T>::conj(*x)*(*y));
		x += incx; y += incy;
	}
	return sum;
}

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

// Level 2
void MultMV(const char *trans, size_t m, size_t n,
	const float &alpha, const float *a, size_t lda, const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultMV(const char *trans, size_t m, size_t n,
	const double &alpha, const double *a, size_t lda, const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
void MultMV(const char *trans, size_t m, size_t n,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta, std::complex<float> *y, size_t incy);
void MultMV(const char *trans, size_t m, size_t n,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta, std::complex<double> *y, size_t incy);

void MultBandedV(const char *trans, size_t m, size_t n, size_t kl, size_t ku,
	const float &alpha, const float *a, size_t lda, const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultBandedV(const char *trans, size_t m, size_t n, size_t kl, size_t ku,
	const double &alpha, const double *a, size_t lda, const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
void MultBandedV(const char *trans, size_t m, size_t n, size_t kl, size_t ku,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta, std::complex<float> *y, size_t incy);
void MultBandedV(const char *trans, size_t m, size_t n, size_t kl, size_t ku,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta, std::complex<double> *y, size_t incy);

void MultHermV(const char *uplo, size_t n,
	const float &alpha, const float *a, size_t lda, const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultHermV(const char *uplo, size_t n,
	const double &alpha, const double *a, size_t lda, const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
void MultHermV(const char *uplo, size_t n,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta, std::complex<float> *y, size_t incy);
void MultHermV(const char *uplo, size_t n,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta, std::complex<double> *y, size_t incy);

void MultBandedHermV(const char *uplo, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda, const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultBandedHermV(const char *uplo, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda, const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
void MultBandedHermV(const char *uplo, size_t n, size_t k,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta, std::complex<float> *y, size_t incy);
void MultBandedHermV(const char *uplo, size_t n, size_t k,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta, std::complex<double> *y, size_t incy);

void MultPackedHermV(const char *uplo, size_t n,
	const float &alpha, const float *ap, const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultPackedHermV(const char *uplo, size_t n,
	const double &alpha, const double *ap, const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
void MultPackedHermV(const char *uplo, size_t n,
	const std::complex<float> &alpha, const std::complex<float> *ap, const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta, std::complex<float> *y, size_t incy);
void MultPackedHermV(const char *uplo, size_t n,
	const std::complex<double> &alpha, const std::complex<double> *ap, const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta, std::complex<double> *y, size_t incy);

void MultSymV(const char *uplo, size_t n,
	const float &alpha, const float *a, size_t lda, const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultSymV(const char *uplo, size_t n,
	const double &alpha, const double *a, size_t lda, const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
/*
void MultSymV(const char *uplo, size_t n,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta, std::complex<float> *y, size_t incy);
void MultSymV(const char *uplo, size_t n,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta, std::complex<double> *y, size_t incy);
*/

void MultBandedSymV(const char *uplo, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda, const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultBandedSymV(const char *uplo, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda, const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
/*
void MultBandedSymV(const char *uplo, size_t n, size_t k,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta, std::complex<float> *y, size_t incy);
void MultBandedSymV(const char *uplo, size_t n, size_t k,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta, std::complex<double> *y, size_t incy);
*/

void MultPackedSymV(const char *uplo, size_t n,
	const float &alpha, const float *ap, const float *x, size_t incx,
	const float &beta, float *y, size_t incy);
void MultPackedSymV(const char *uplo, size_t n,
	const double &alpha, const double *ap, const double *x, size_t incx,
	const double &beta, double *y, size_t incy);
/*
void MultPackedSymV(const char *uplo, size_t n,
	const std::complex<float> &alpha, const std::complex<float> *ap, const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta, std::complex<float> *y, size_t incy);
void MultPackedSymV(const char *uplo, size_t n,
	const std::complex<double> &alpha, const std::complex<double> *ap, const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta, std::complex<double> *y, size_t incy);
*/

void MultTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const float *a, size_t lda, float *x, size_t incx);
void MultTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const double *a, size_t lda, double *x, size_t incx);
void MultTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<float> *a, size_t lda, std::complex<float> *x, size_t incx);
void MultTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<double> *a, size_t lda, std::complex<double> *x, size_t incx);

void MultBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const float *a, size_t lda, float *x, size_t incx);
void MultBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const double *a, size_t lda, double *x, size_t incx);
void MultBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const std::complex<float> *a, size_t lda, std::complex<float> *x, size_t incx);
void MultBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const std::complex<double> *a, size_t lda, std::complex<double> *x, size_t incx);

void MultPackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const float *ap, float *x, size_t incx);
void MultPackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const double *ap, double *x, size_t incx);
void MultPackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<float> *ap, std::complex<float> *x, size_t incx);
void MultPackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<double> *ap, std::complex<double> *x, size_t incx);

void SolveTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const float *a, size_t lda, float *x, size_t incx);
void SolveTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const double *a, size_t lda, double *x, size_t incx);
void SolveTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<float> *a, size_t lda, std::complex<float> *x, size_t incx);
void SolveTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<double> *a, size_t lda, std::complex<double> *x, size_t incx);

void SolveBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const float *a, size_t lda, float *x, size_t incx);
void SolveBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const double *a, size_t lda, double *x, size_t incx);
void SolveBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const std::complex<float> *a, size_t lda, std::complex<float> *x, size_t incx);
void SolveBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const std::complex<double> *a, size_t lda, std::complex<double> *x, size_t incx);

void SolvePackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const float *ap, float *x, size_t incx);
void SolvePackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const double *ap, double *x, size_t incx);
void SolvePackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<float> *ap, std::complex<float> *x, size_t incx);
void SolvePackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<double> *ap, std::complex<double> *x, size_t incx);

void Rank1Update(size_t m, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy,
	float *a, size_t lda);
void Rank1Update(size_t m, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy,
	double *a, size_t lda);
void Rank1Update(size_t m, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx, const std::complex<float> *y, size_t incy,
	std::complex<float> *a, size_t lda);
void Rank1Update(size_t m, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx, const std::complex<double> *y, size_t incy,
	std::complex<double> *a, size_t lda);

void ConjugateRank1Update(size_t m, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy,
	float *a, size_t lda);
void ConjugateRank1Update(size_t m, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy,
	double *a, size_t lda);
void ConjugateRank1Update(size_t m, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx, const std::complex<float> *y, size_t incy,
	std::complex<float> *a, size_t lda);
void ConjugateRank1Update(size_t m, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx, const std::complex<double> *y, size_t incy,
	std::complex<double> *a, size_t lda);

void HermRank1Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, float *a, size_t lda);
void HermRank1Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, double *a, size_t lda);
void HermRank1Update(const char *uplo, size_t n, const float &alpha,
	const std::complex<float> *x, size_t incx, std::complex<float> *a, size_t lda);
void HermRank1Update(const char *uplo, size_t n, const double &alpha,
	const std::complex<double> *x, size_t incx, std::complex<double> *a, size_t lda);

void PackedHermRank1Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, float *a);
void PackedHermRank1Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, double *a);
void PackedHermRank1Update(const char *uplo, size_t n, const float &alpha,
	const std::complex<float> *x, size_t incx, std::complex<float> *a);
void PackedHermRank1Update(const char *uplo, size_t n, const double &alpha,
	const std::complex<double> *x, size_t incx, std::complex<double> *a);

void HermRank2Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy,
	float *a, size_t lda);
void HermRank2Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy,
	double *a, size_t lda);
void HermRank2Update(const char *uplo, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx, const std::complex<float> *y, size_t incy,
	std::complex<float> *a, size_t lda);
void HermRank2Update(const char *uplo, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx, const std::complex<double> *y, size_t incy,
	std::complex<double> *a, size_t lda);

void PackedHermRank2Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy, float *a);
void PackedHermRank2Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy, double *a);
void PackedHermRank2Update(const char *uplo, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx, const std::complex<float> *y, size_t incy, std::complex<float> *a);
void PackedHermRank2Update(const char *uplo, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx, const std::complex<double> *y, size_t incy, std::complex<double> *a);

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

void PackedSymRank1Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, float *a);
void PackedSymRank1Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, double *a);
/*
void PackedSymRank1Update(const char *uplo, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx, std::complex<float> *a);
void PackedSymRank1Update(const char *uplo, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx, std::complex<double> *a);
*/

void SymRank2Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy,
	float *a, size_t lda);
void SymRank2Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy,
	double *a, size_t lda);
/*
void SymRank2Update(const char *uplo, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx, const std::complex<float> *y, size_t incy,
	std::complex<float> *a, size_t lda);
void SymRank2Update(const char *uplo, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx, const std::complex<double> *y, size_t incy,
	std::complex<double> *a, size_t lda);
*/

void PackedSymRank2Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy, float *a);
void PackedSymRank2Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy, double *a);
/*
void PackedSymRank2Update(const char *uplo, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx, const std::complex<float> *y, size_t incy, std::complex<float> *a);
void PackedSymRank2Update(const char *uplo, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx, const std::complex<double> *y, size_t incy, std::complex<double> *a);
*/

// Level 3
void MultMM(const char *transa, const char *transb, size_t m, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda, const float *b, size_t ldb,
	const float &beta, float *c, size_t ldc);
void MultMM(const char *transa, const char *transb, size_t m, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda, const double *b, size_t ldb,
	const double &beta, double *c, size_t ldc);
void MultMM(const char *transa, const char *transb, size_t m, size_t n, size_t k,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *b, size_t ldb,
	const std::complex<float> &beta, std::complex<float> *c, size_t ldc);
void MultMM(const char *transa, const char *transb, size_t m, size_t n, size_t k,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *b, size_t ldb,
	const std::complex<double> &beta, std::complex<double> *c, size_t ldc);

void MultSymM(const char *side, const char *uplo, size_t m, size_t n,
	const float &alpha, const float *a, size_t lda, const float *b, size_t ldb,
	const float &beta, float *c, size_t ldc);
void MultSymM(const char *side, const char *uplo, size_t m, size_t n,
	const double &alpha, const double *a, size_t lda, const double *b, size_t ldb,
	const double &beta, double *c, size_t ldc);
void MultSymM(const char *side, const char *uplo, size_t m, size_t n,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *b, size_t ldb,
	const std::complex<float> &beta, std::complex<float> *c, size_t ldc);
void MultSymM(const char *side, const char *uplo, size_t m, size_t n,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *b, size_t ldb,
	const std::complex<double> &beta, std::complex<double> *c, size_t ldc);

void MultHermM(const char *side, const char *uplo, size_t m, size_t n,
	const float &alpha, const float *a, size_t lda, const float *b, size_t ldb,
	const float &beta, float *c, size_t ldc);
void MultHermM(const char *side, const char *uplo, size_t m, size_t n,
	const double &alpha, const double *a, size_t lda, const double *b, size_t ldb,
	const double &beta, double *c, size_t ldc);
void MultHermM(const char *side, const char *uplo, size_t m, size_t n,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *b, size_t ldb,
	const std::complex<float> &beta, std::complex<float> *c, size_t ldc);
void MultHermM(const char *side, const char *uplo, size_t m, size_t n,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *b, size_t ldb,
	const std::complex<double> &beta, std::complex<double> *c, size_t ldc);

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
	const std::complex<float> &beta, std::complex<float> *c, size_t ldc);
void HermRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda,
	const std::complex<double> *b, size_t ldb,
	const std::complex<double> &beta, std::complex<double> *c, size_t ldc);

void MultTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const float &alpha, const float *a, size_t lda,
	float *b, size_t ldb);
void MultTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const double &alpha, const double *a, size_t lda,
	double *b, size_t ldb);
void MultTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const std::complex<float> &alpha, const std::complex<float> *a, size_t lda,
	std::complex<float> *b, size_t ldb);
void MultTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const std::complex<double> &alpha, const std::complex<double> *a, size_t lda,
	std::complex<double> *b, size_t ldb);

void SolveTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const float &alpha, const float *a, size_t lda,
	float *b, size_t ldb);
void SolveTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const double &alpha, const double *a, size_t lda,
	double *b, size_t ldb);
void SolveTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const std::complex<float> &alpha, const std::complex<float> *a, size_t lda,
	std::complex<float> *b, size_t ldb);
void SolveTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const std::complex<double> &alpha, const std::complex<double> *a, size_t lda,
	std::complex<double> *b, size_t ldb);

} // namespace BLAS
} // namespace RNP

#endif // RNP_BLAS_MIX_HPP_INCLUDED
