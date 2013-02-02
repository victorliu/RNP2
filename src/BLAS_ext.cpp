#include <RNP/FBLAS.hpp>
#include <RNP/Types.hpp>
#include <cstring>
#include <limits>

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
// sasum,  dasum,  scasum, dzasum : Norm1
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
//

namespace RNP{
namespace BLAS{

using namespace FBLAS;
using namespace RNP;

template <typename T>
static void Set_vector_generic(size_t n, const T &val, T *x, size_t incx){
	while(n --> 0){ *x = val; x += incx; }
}

// Extra
void Set(size_t n, const float &val, float *x, size_t incx){
	Set_vector_generic(n, val, x, incx);
}
void Set(size_t n, const double &val, double *x, size_t incx){
	Set_vector_generic(n, val, x, incx);
}
void Set(size_t n, const std::complex<float> &val, std::complex<float> *x, size_t incx){
	Set_vector_generic(n, val, x, incx);
}
void Set(size_t n, const std::complex<double> &val, std::complex<double> *x, size_t incx){
	Set_vector_generic(n, val, x, incx);
}

template <typename T>
void Set_matrix_generic(size_t m, size_t n, const T &offdiag, const T &diag, T *a, size_t lda){
	for(size_t j = 0; j < n; ++j){
		for(size_t i = 0; i < m; ++i){
			a[i+j*lda] = (i == j ? diag : offdiag);
		}
	}
}

void Set(size_t m, size_t n, const float &offdiag, const float &diag, float *a, size_t lda){
	Set_matrix_generic(m, n, offdiag, diag, a, lda);
}
void Set(size_t m, size_t n, const double &offdiag, const double &diag, double *a, size_t lda){
	Set_matrix_generic(m, n, offdiag, diag, a, lda);
}
void Set(size_t m, size_t n, const std::complex<float> &offdiag, const std::complex<float> &diag, std::complex<float> *a, size_t lda){
	Set_matrix_generic(m, n, offdiag, diag, a, lda);
}
void Set(size_t m, size_t n, const std::complex<double> &offdiag, const std::complex<double> &diag, std::complex<double> *a, size_t lda){
	Set_matrix_generic(m, n, offdiag, diag, a, lda);
}

template <typename T>
void Copy_matrix_generic(size_t m, size_t n, const T *src, size_t ldsrc, T *dst, size_t lddst){
	if(m == ldsrc && m == lddst){
		memcpy(dst, src, sizeof(T) * m*n);
	}else{
		for(size_t j = 0; j < n; ++j){
			memcpy(&dst[j*lddst], &src[j*lddst], sizeof(T) * m);
		}
	}
}

void Copy(size_t m, size_t n, const float *src, size_t ldsrc, float *dst, size_t lddst){
	Copy_matrix_generic(m, n, src, ldsrc, dst, lddst);
}
void Copy(size_t m, size_t n, const double *src, size_t ldsrc, double *dst, size_t lddst){
	Copy_matrix_generic(m, n, src, ldsrc, dst, lddst);
}
void Copy(size_t m, size_t n, const std::complex<float> *src, size_t ldsrc, std::complex<float> *dst, size_t lddst){
	Copy_matrix_generic(m, n, src, ldsrc, dst, lddst);
}
void Copy(size_t m, size_t n, const std::complex<double> *src, size_t ldsrc, std::complex<double> *dst, size_t lddst){
	Copy_matrix_generic(m, n, src, ldsrc, dst, lddst);
}

template <typename T>
void Copy_trans_generic(const char *trans, size_t m, size_t n, const T *src, size_t ldsrc, T *dst, size_t lddst){
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

void Copy(const char *trans, size_t m, size_t n, const float *src, size_t ldsrc, float *dst, size_t lddst){
	Copy_trans_generic(trans, m, n, src, ldsrc, dst, lddst);
}
void Copy(const char *trans, size_t m, size_t n, const double *src, size_t ldsrc, double *dst, size_t lddst){
	Copy_trans_generic(trans, m, n, src, ldsrc, dst, lddst);
}
void Copy(const char *trans, size_t m, size_t n, const std::complex<float> *src, size_t ldsrc, std::complex<float> *dst, size_t lddst){
	Copy_trans_generic(trans, m, n, src, ldsrc, dst, lddst);
}
void Copy(const char *trans, size_t m, size_t n, const std::complex<double> *src, size_t ldsrc, std::complex<double> *dst, size_t lddst){
	Copy_trans_generic(trans, m, n, src, ldsrc, dst, lddst);
}

template <typename T>
void Conjugate_generic(size_t n, T *x, size_t incx){
	while(n --> 0){
		*x = Traits<T>::conj(*x);
		x += incx;
	}
}

void Conjugate(size_t n, float *x, size_t incx){
	Conjugate_generic(n, x, incx);
}
void Conjugate(size_t n, double *x, size_t incx){
	Conjugate_generic(n, x, incx);
}
void Conjugate(size_t n, std::complex<float> *x, size_t incx){
	Conjugate_generic(n, x, incx);
}
void Conjugate(size_t n, std::complex<double> *x, size_t incx){
	Conjugate_generic(n, x, incx);
}

template <typename T>
void Rescale_generic(size_t n, int p, T *x, size_t incx){
	while(n --> 0){
		*x = std::ldexp(*x, p);
		x += incx;
	}
}
void Rescale(size_t n, int p, float *x, size_t incx){
	Rescale_generic(n, p, x, incx);
}
void Rescale(size_t n, int p, double *x, size_t incx){
	Rescale_generic(n, p, x, incx);
}
template <typename T>
void Rescale_generic_complex(size_t n, int p, std::complex<T> *x, size_t incx){
	while(n --> 0){
		*x = std::complex<T>(std::ldexp(x->real(), p), std::ldexp(x->imag(), p));
		x += incx;
	}
}
void Rescale(size_t n, int p, std::complex<float> *x, size_t incx){
	Rescale_generic_complex(n, p, x, incx);
}
void Rescale(size_t n, int p, std::complex<double> *x, size_t incx){
	Rescale_generic_complex(n, p, x, incx);
}

template <typename TS, typename T>
void Rescale_generic(const char *type, size_t kl, size_t ku, const TS &cfrom, const TS &cto, size_t m, size_t n, T *a, size_t lda){
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

void Rescale(const char *type, size_t kl, size_t ku, const float &cfrom, const float &cto, size_t m, size_t n, float *a, size_t lda){
	Rescale_generic(type, kl, ku, cfrom, cto, m, n, a, lda);
}
void Rescale(const char *type, size_t kl, size_t ku, const double &cfrom, const double &cto, size_t m, size_t n, double *a, size_t lda){
	Rescale_generic(type, kl, ku, cfrom, cto, m, n, a, lda);
}
void Rescale(const char *type, size_t kl, size_t ku, const float &cfrom, const float &cto, size_t m, size_t n, std::complex<float> *a, size_t lda){
	Rescale_generic(type, kl, ku, cfrom, cto, m, n, a, lda);
}
void Rescale(const char *type, size_t kl, size_t ku, const double &cfrom, const double &cto, size_t m, size_t n, std::complex<double> *a, size_t lda){
	Rescale_generic(type, kl, ku, cfrom, cto, m, n, a, lda);
}

// Level 1 rotations
void RotGen(float  *a, const float  &b, float  *c, float  *s){
	BLASNAME(srotg,SROTG)(a, b, c, s);
}
void RotGen(double *a, const double &b, double *c, double *s){
	BLASNAME(drotg,DROTG)(a, b, c, s);
}
void RotGen(std::complex<float>  *a, const std::complex<float>  &b, float  *c, std::complex<float>  *s){
	BLASNAME(crotg,CROTG)(a, b, c, s);
}
void RotGen(std::complex<double> *a, const std::complex<double> &b, double *c, std::complex<double> *s){
	BLASNAME(zrotg,ZROTG)(a, b, c, s);
}

void ModifiedRotGen(float  *d1, float  *d2, float  *x1, const float  &x2, float  param[5]){
	BLASNAME(srotmg,SROTMG)(d1, d2, x1, x2, param);
}
void ModifiedRotGen(double *d1, double *d2, double *x1, const double &x2, double param[5]){
	BLASNAME(drotmg,DROTMG)(d1, d2, x1, x2, param);
}

void RotApply(size_t n, float  *x, size_t incx, float  *y, size_t incy, const float  &c, const float  &s){
	BLASNAME(srot,SROT)(n, x, incx, y, incy, c, s);
}
void RotApply(size_t n, double *x, size_t incx, double *y, size_t incy, const double &c, const double &s){
	BLASNAME(drot,DROT)(n, x, incx, y, incy, c, s);
}
void RotApply(size_t n, std::complex<float>  *x, size_t incx, std::complex<float>  *y, size_t incy, const std::complex<float>  &c, const std::complex<float>  &s){
	// Not in BLAS
	while(n --> 0){
		std::complex<float> temp = c*(*x) + s*(*y);
		*y = c*(*y) - std::conj(s)*(*x);
		*x = temp;
		x += incx; y += incy;
	}
}
void RotApply(size_t n, std::complex<double> *x, size_t incx, std::complex<double> *y, size_t incy, const std::complex<double> &c, const std::complex<double> &s){
	// Not in BLAS
	while(n --> 0){
		std::complex<double> temp = c*(*x) + s*(*y);
		*y = c*(*y) - std::conj(s)*(*x);
		*x = temp;
		x += incx; y += incy;
	}
}

void ModifiedRotApply(size_t n, float  *x, size_t incx, float  *y, size_t incy, const float  param[5]){
	BLASNAME(srotm,SROTM)(n, x, incx, y, incy, param);
}
void ModifiedRotApply(size_t n, double *x, size_t incx, double *y, size_t incy, const double param[5]){
	BLASNAME(drotm,DROTM)(n, x, incx, y, incy, param);
}

// Level 1 utility
void Swap(size_t n, float *x, size_t incx, float *y, size_t incy){
	BLASNAME(sswap,SSWAP)(n, x, incx, y, incy);
}
void Swap(size_t n, double *x, size_t incx, double *y, size_t incy){
	BLASNAME(dswap,DSWAP)(n, x, incx, y, incy);
}
void Swap(size_t n, std::complex<float> *x, size_t incx, std::complex<float> *y, size_t incy){
	BLASNAME(cswap,CSWAP)(n, x, incx, y, incy);
}
void Swap(size_t n, std::complex<double> *x, size_t incx, std::complex<double> *y, size_t incy){
	BLASNAME(zswap,ZSWAP)(n, x, incx, y, incy);
}

void Scale(size_t n, const float &alpha, float *x, size_t incx){
	BLASNAME(sscal,SSCAL)(n, alpha, x, incx);
}
void Scale(size_t n, const double &alpha, double *x, size_t incx){
	BLASNAME(dscal,DSCAL)(n, alpha, x, incx);
}
void Scale(size_t n, const std::complex<float> &alpha, std::complex<float> *x, size_t incx){
	BLASNAME(cscal,CSCAL)(n, alpha, x, incx);
}
void Scale(size_t n, const std::complex<double> &alpha, std::complex<double> *x, size_t incx){
	BLASNAME(zscal,ZSCAL)(n, alpha, x, incx);
}
void Scale(size_t n, const float &alpha, std::complex<float> *x, size_t incx){
	BLASNAME(csscal,CSSCAL)(n, alpha, x, incx);
}
void Scale(size_t n, const double &alpha, std::complex<double> *x, size_t incx){
	BLASNAME(zdscal,ZDSCAL)(n, alpha, x, incx);
}

void Copy(size_t n, const float *src, size_t incsrc, float *dst, size_t incdst){
	BLASNAME(scopy,SCOPY)(n, src, incsrc, dst, incdst);
}
void Copy(size_t n, const double *src, size_t incsrc, double *dst, size_t incdst){
	BLASNAME(dcopy,DCOPY)(n, src, incsrc, dst, incdst);
}
void Copy(size_t n, const std::complex<float> *src, size_t incsrc, std::complex<float> *dst, size_t incdst){
	BLASNAME(ccopy,CCOPY)(n, src, incsrc, dst, incdst);
}
void Copy(size_t n, const std::complex<double> *src, size_t incsrc, std::complex<double> *dst, size_t incdst){
	BLASNAME(zcopy,ZCOPY)(n, src, incsrc, dst, incdst);
}

void Axpy(size_t n, const float &alpha, const float *x, size_t incx, float *y, size_t incy){
	BLASNAME(saxpy,SAXPY)(n, alpha, x, incx, y, incy);
}
void Axpy(size_t n, const double &alpha, const double *x, size_t incx, double *y, size_t incy){
	BLASNAME(daxpy,DAXPY)(n, alpha, x, incx, y, incy);
}
void Axpy(size_t n, const std::complex<float> &alpha, const std::complex<float> *x, size_t incx, std::complex<float> *y, size_t incy){
	BLASNAME(caxpy,CAXPY)(n, alpha, x, incx, y, incy);
}
void Axpy(size_t n, const std::complex<double> &alpha, const std::complex<double> *x, size_t incx, std::complex<double> *y, size_t incy){
	BLASNAME(zaxpy,ZAXPY)(n, alpha, x, incx, y, incy);
}

float Dot(size_t n, const float *x, size_t incx, const float *y, size_t incy){
	return BLASNAME(sdot,SDOT)(n, x, incx, y, incy);
}
double Dot(size_t n, const double *x, size_t incx, const double *y, size_t incy){
	return BLASNAME(ddot,DDOT)(n, x, incx, y, incy);
}
std::complex<float> Dot(size_t n, const std::complex<float> *x, size_t incx, const std::complex<float> *y, size_t incy){
	// Don't call BLAS due to return arg problem
	std::complex<float> sum = 0.f;
	while(n --> 0){
		sum += ((*x)*(*y));
		x += incx; y += incy;
	}
	return sum;
}
std::complex<double> Dot(size_t n, const std::complex<double> *x, size_t incx, const std::complex<double> *y, size_t incy){
	// Don't call BLAS due to return arg problem
	std::complex<double> sum = 0.f;
	while(n --> 0){
		sum += ((*x)*(*y));
		x += incx; y += incy;
	}
	return sum;
}

double DotEx(size_t n, const float *x, size_t incx, const float *y, size_t incy){
	return BLASNAME(dsdot,DSDOT)(n, x, incx, y, incy);
}
float  DotEx(size_t n, const float &b, const float *x, size_t incx, const float *y, size_t incy){
	return BLASNAME(sdsdot,SDSDOT)(n, b, x, incx, y, incy);
}

float ConjugateDot(size_t n, const float *x, size_t incx, const float *y, size_t incy){
	return BLASNAME(sdot,SDOT)(n, x, incx, y, incy);
}
double ConjugateDot(size_t n, const double *x, size_t incx, const double *y, size_t incy){
	return BLASNAME(ddot,DDOT)(n, x, incx, y, incy);
}
std::complex<float> ConjugateDot(size_t n, const std::complex<float> *x, size_t incx, const std::complex<float> *y, size_t incy){
	// Don't call BLAS due to return arg problem
	std::complex<float> sum = 0.f;
	while(n --> 0){
		sum += (std::conj(*x)*(*y));
		x += incx; y += incy;
	}
	return sum;
}
std::complex<double> ConjugateDot(size_t n, const std::complex<double> *x, size_t incx, const std::complex<double> *y, size_t incy){
	// Don't call BLAS due to return arg problem
	std::complex<double> sum = 0.f;
	while(n --> 0){
		sum += (std::conj(*x)*(*y));
		x += incx; y += incy;
	}
	return sum;
}

float Norm2(size_t n, const float *x, size_t incx){
	return BLASNAME(snrm2,SNRM2)(n, x, incx);
}
double Norm2(size_t n, const double *x, size_t incx){
	return BLASNAME(dnrm2,DNRM2)(n, x, incx);
}
float Norm2(size_t n, const std::complex<float> *x, size_t incx){
	return BLASNAME(scnrm2,SCNRM2)(n, x, incx);
}
double Norm2(size_t n, const std::complex<double> *x, size_t incx){
	return BLASNAME(dznrm2,DZNRM2)(n, x, incx);
}

float Norm1(size_t n, const float *x, size_t incx){
	return BLASNAME(sasum,SASUM)(n, x, incx);
}
double Norm1(size_t n, const double *x, size_t incx){
	return BLASNAME(dasum,DASUM)(n, x, incx);
}
float Norm1(size_t n, const std::complex<float> *x, size_t incx){
	return BLASNAME(scasum,SCASUM)(n, x, incx);
}
double Norm1(size_t n, const std::complex<double> *x, size_t incx){
	return BLASNAME(dzasum,DZASUM)(n, x, incx);
}

// These return a 0-based index
size_t MaximumIndex(size_t n, const float *x, size_t incx){
	return BLASNAME(isamax,ISAMAX)(n, x, incx) - 1;
}
size_t MaximumIndex(size_t n, const double *x, size_t incx){
	return BLASNAME(idamax,IDAMAX)(n, x, incx) - 1;
}
size_t MaximumIndex(size_t n, const std::complex<float> *x, size_t incx){
	return BLASNAME(icamax,ICAMAX)(n, x, incx) - 1;
}
size_t MaximumIndex(size_t n, const std::complex<double> *x, size_t incx){
	return BLASNAME(izamax,IZAMAX)(n, x, incx) - 1;
}

// Level 2
void MultMV(const char *trans, size_t m, size_t n,
	const float &alpha, const float *a, size_t lda, const float *x, size_t incx,
	const float &beta, float *y, size_t incy){
	BLASNAME(sgemv,SGEMV)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
void MultMV(const char *trans, size_t m, size_t n,
	const double &alpha, const double *a, size_t lda, const double *x, size_t incx,
	const double &beta, double *y, size_t incy){
	BLASNAME(dgemv,DGEMV)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
void MultMV(const char *trans, size_t m, size_t n,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta, std::complex<float> *y, size_t incy){
	BLASNAME(cgemv,CGEMV)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
void MultMV(const char *trans, size_t m, size_t n,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta, std::complex<double> *y, size_t incy){
	BLASNAME(zgemv,ZGEMV)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void MultBandedV(const char *trans, size_t m, size_t n, size_t kl, size_t ku,
	const float &alpha, const float *a, size_t lda, const float *x, size_t incx,
	const float &beta, float *y, size_t incy){
	BLASNAME(sgbmv,SGBMV)(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
void MultBandedV(const char *trans, size_t m, size_t n, size_t kl, size_t ku,
	const double &alpha, const double *a, size_t lda, const double *x, size_t incx,
	const double &beta, double *y, size_t incy){
	BLASNAME(dgbmv,DGBMV)(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
void MultBandedV(const char *trans, size_t m, size_t n, size_t kl, size_t ku,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta, std::complex<float> *y, size_t incy){
	BLASNAME(cgbmv,CGBMV)(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
void MultBandedV(const char *trans, size_t m, size_t n, size_t kl, size_t ku,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta, std::complex<double> *y, size_t incy){
	BLASNAME(zgbmv,ZGBMV)(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void MultHermV(const char *uplo, size_t n,
	const float &alpha, const float *a, size_t lda, const float *x, size_t incx,
	const float &beta, float *y, size_t incy){
	BLASNAME(ssymv,SSYMV)(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
void MultHermV(const char *uplo, size_t n,
	const double &alpha, const double *a, size_t lda, const double *x, size_t incx,
	const double &beta, double *y, size_t incy){
	BLASNAME(dsymv,DSYMV)(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
void MultHermV(const char *uplo, size_t n,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta, std::complex<float> *y, size_t incy){
	BLASNAME(chemv,CHEMV)(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
void MultHermV(const char *uplo, size_t n,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta, std::complex<double> *y, size_t incy){
	BLASNAME(zhemv,ZHEMV)(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void MultBandedHermV(const char *uplo, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda, const float *x, size_t incx,
	const float &beta, float *y, size_t incy){
	BLASNAME(ssbmv,SSBMV)(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
void MultBandedHermV(const char *uplo, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda, const double *x, size_t incx,
	const double &beta, double *y, size_t incy){
	BLASNAME(dsbmv,DSBMV)(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
void MultBandedHermV(const char *uplo, size_t n, size_t k,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta, std::complex<float> *y, size_t incy){
	BLASNAME(chbmv,CHBMV)(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
void MultBandedHermV(const char *uplo, size_t n, size_t k,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta, std::complex<double> *y, size_t incy){
	BLASNAME(zhbmv,ZHBMV)(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void MultPackedHermV(const char *uplo, size_t n,
	const float &alpha, const float *ap, const float *x, size_t incx,
	const float &beta, float *y, size_t incy){
	BLASNAME(sspmv,SSPMV)(uplo, n, alpha, ap, x, incx, beta, y, incy);
}
void MultPackedHermV(const char *uplo, size_t n,
	const double &alpha, const double *ap, const double *x, size_t incx,
	const double &beta, double *y, size_t incy){
	BLASNAME(dspmv,DSPMV)(uplo, n, alpha, ap, x, incx, beta, y, incy);
}
void MultPackedHermV(const char *uplo, size_t n,
	const std::complex<float> &alpha, const std::complex<float> *ap, const std::complex<float> *x, size_t incx,
	const std::complex<float> &beta, std::complex<float> *y, size_t incy){
	BLASNAME(chpmv,CHPMV)(uplo, n, alpha, ap, x, incx, beta, y, incy);
}
void MultPackedHermV(const char *uplo, size_t n,
	const std::complex<double> &alpha, const std::complex<double> *ap, const std::complex<double> *x, size_t incx,
	const std::complex<double> &beta, std::complex<double> *y, size_t incy){
	BLASNAME(zhpmv,ZHPMV)(uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void MultSymV(const char *uplo, size_t n,
	const float &alpha, const float *a, size_t lda, const float *x, size_t incx,
	const float &beta, float *y, size_t incy){
	BLASNAME(ssymv,SSYMV)(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
void MultSymV(const char *uplo, size_t n,
	const double &alpha, const double *a, size_t lda, const double *x, size_t incx,
	const double &beta, double *y, size_t incy){
	BLASNAME(dsymv,DSYMV)(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void MultBandedSymV(const char *uplo, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda, const float *x, size_t incx,
	const float &beta, float *y, size_t incy){
	BLASNAME(ssbmv,SSBMV)(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
void MultBandedSymV(const char *uplo, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda, const double *x, size_t incx,
	const double &beta, double *y, size_t incy){
	BLASNAME(dsbmv,DSBMV)(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void MultPackedSymV(const char *uplo, size_t n,
	const float &alpha, const float *ap, const float *x, size_t incx,
	const float &beta, float *y, size_t incy){
	BLASNAME(sspmv,SSPMV)(uplo, n, alpha, ap, x, incx, beta, y, incy);
}
void MultPackedSymV(const char *uplo, size_t n,
	const double &alpha, const double *ap, const double *x, size_t incx,
	const double &beta, double *y, size_t incy){
	BLASNAME(dspmv,DSPMV)(uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void MultTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const float *a, size_t lda, float *x, size_t incx){
	BLASNAME(strmv,STRMV)(uplo, trans, diag, n, a, lda, x, incx);
}
void MultTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const double *a, size_t lda, double *x, size_t incx){
	BLASNAME(dtrmv,DTRMV)(uplo, trans, diag, n, a, lda, x, incx);
}
void MultTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<float> *a, size_t lda, std::complex<float> *x, size_t incx){
	BLASNAME(ctrmv,CTRMV)(uplo, trans, diag, n, a, lda, x, incx);
}
void MultTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<double> *a, size_t lda, std::complex<double> *x, size_t incx){
	BLASNAME(ztrmv,ZTRMV)(uplo, trans, diag, n, a, lda, x, incx);
}

void MultBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const float *a, size_t lda, float *x, size_t incx){
	BLASNAME(stbmv,STBMV)(uplo, trans, diag, n, k, a, lda, x, incx);
}
void MultBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const double *a, size_t lda, double *x, size_t incx){
	BLASNAME(dtbmv,DTBMV)(uplo, trans, diag, n, k, a, lda, x, incx);
}
void MultBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const std::complex<float> *a, size_t lda, std::complex<float> *x, size_t incx){
	BLASNAME(ctbmv,CTBMV)(uplo, trans, diag, n, k, a, lda, x, incx);
}
void MultBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const std::complex<double> *a, size_t lda, std::complex<double> *x, size_t incx){
	BLASNAME(ztbmv,ZTBMV)(uplo, trans, diag, n, k, a, lda, x, incx);
}

void MultPackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const float *ap, float *x, size_t incx){
	BLASNAME(stpmv,STPMV)(uplo, trans, diag, n, ap, x, incx);
}
void MultPackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const double *ap, double *x, size_t incx){
	BLASNAME(dtpmv,DTPMV)(uplo, trans, diag, n, ap, x, incx);
}
void MultPackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<float> *ap, std::complex<float> *x, size_t incx){
	BLASNAME(ctpmv,CTPMV)(uplo, trans, diag, n, ap, x, incx);
}
void MultPackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<double> *ap, std::complex<double> *x, size_t incx){
	BLASNAME(ztpmv,ZTPMV)(uplo, trans, diag, n, ap, x, incx);
}


void SolveTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const float *a, size_t lda, float *x, size_t incx){
	BLASNAME(strsv,STRSV)(uplo, trans, diag, n, a, lda, x, incx);
}
void SolveTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const double *a, size_t lda, double *x, size_t incx){
	BLASNAME(dtrsv,DTRSV)(uplo, trans, diag, n, a, lda, x, incx);
}
void SolveTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<float> *a, size_t lda, std::complex<float> *x, size_t incx){
	BLASNAME(ctrsv,CTRSV)(uplo, trans, diag, n, a, lda, x, incx);
}
void SolveTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<double> *a, size_t lda, std::complex<double> *x, size_t incx){
	BLASNAME(ztrsv,ZTRSV)(uplo, trans, diag, n, a, lda, x, incx);
}

void SolveBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const float *a, size_t lda, float *x, size_t incx){
	BLASNAME(stbsv,STBSV)(uplo, trans, diag, n, k, a, lda, x, incx);
}
void SolveBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const double *a, size_t lda, double *x, size_t incx){
	BLASNAME(dtbsv,DTBSV)(uplo, trans, diag, n, k, a, lda, x, incx);
}
void SolveBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const std::complex<float> *a, size_t lda, std::complex<float> *x, size_t incx){
	BLASNAME(ctbsv,CTBSV)(uplo, trans, diag, n, k, a, lda, x, incx);
}
void SolveBandedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, size_t k, const std::complex<double> *a, size_t lda, std::complex<double> *x, size_t incx){
	BLASNAME(ztbsv,ZTBSV)(uplo, trans, diag, n, k, a, lda, x, incx);
}

void SolvePackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const float *ap, float *x, size_t incx){
	BLASNAME(stpsv,STPSV)(uplo, trans, diag, n, ap, x, incx);
}
void SolvePackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const double *ap, double *x, size_t incx){
	BLASNAME(dtpsv,DTPSV)(uplo, trans, diag, n, ap, x, incx);
}
void SolvePackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<float> *ap, std::complex<float> *x, size_t incx){
	BLASNAME(ctpsv,CTPSV)(uplo, trans, diag, n, ap, x, incx);
}
void SolvePackedTrV(const char *uplo, const char *trans, const char *diag,
	size_t n, const std::complex<double> *ap, std::complex<double> *x, size_t incx){
	BLASNAME(ztpsv,ZTPSV)(uplo, trans, diag, n, ap, x, incx);
}

void Rank1Update(size_t m, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy,
	float *a, size_t lda){
	BLASNAME(sger,SGER)(m, n, alpha, x, incx, y, incy, a, lda);
}
void Rank1Update(size_t m, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy,
	double *a, size_t lda){
	BLASNAME(dger,DGER)(m, n, alpha, x, incx, y, incy, a, lda);
}
void Rank1Update(size_t m, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx, const std::complex<float> *y, size_t incy,
	std::complex<float> *a, size_t lda){
	BLASNAME(cgeru,CGERU)(m, n, alpha, x, incx, y, incy, a, lda);
}
void Rank1Update(size_t m, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx, const std::complex<double> *y, size_t incy,
	std::complex<double> *a, size_t lda){
	BLASNAME(zgeru,ZGERU)(m, n, alpha, x, incx, y, incy, a, lda);
}

void ConjugateRank1Update(size_t m, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy,
	float *a, size_t lda){
	BLASNAME(sger,SGER)(m, n, alpha, x, incx, y, incy, a, lda);
}
void ConjugateRank1Update(size_t m, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy,
	double *a, size_t lda){
	BLASNAME(dger,DGER)(m, n, alpha, x, incx, y, incy, a, lda);
}
void ConjugateRank1Update(size_t m, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx, const std::complex<float> *y, size_t incy,
	std::complex<float> *a, size_t lda){
	BLASNAME(cgerc,CGERC)(m, n, alpha, x, incx, y, incy, a, lda);
}
void ConjugateRank1Update(size_t m, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx, const std::complex<double> *y, size_t incy,
	std::complex<double> *a, size_t lda){
	BLASNAME(zgerc,ZGERC)(m, n, alpha, x, incx, y, incy, a, lda);
}

void HermRank1Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, float *a, size_t lda){
	BLASNAME(ssyr,SSYR)(uplo, n, alpha, x, incx, a, lda);
}
void HermRank1Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, double *a, size_t lda){
	BLASNAME(dsyr,DSYR)(uplo, n, alpha, x, incx, a, lda);
}
void HermRank1Update(const char *uplo, size_t n, const float &alpha,
	const std::complex<float> *x, size_t incx, std::complex<float> *a, size_t lda){
	BLASNAME(cher,CHER)(uplo, n, alpha, x, incx, a, lda);
}
void HermRank1Update(const char *uplo, size_t n, const double &alpha,
	const std::complex<double> *x, size_t incx, std::complex<double> *a, size_t lda){
	BLASNAME(zher,ZHER)(uplo, n, alpha, x, incx, a, lda);
}

void PackedHermRank1Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, float *ap){
	BLASNAME(sspr,SSPR)(uplo, n, alpha, x, incx, ap);
}
void PackedHermRank1Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, double *ap){
	BLASNAME(dspr,DSPR)(uplo, n, alpha, x, incx, ap);
}
void PackedHermRank1Update(const char *uplo, size_t n, const float &alpha,
	const std::complex<float> *x, size_t incx, std::complex<float> *ap){
	BLASNAME(chpr,CHPR)(uplo, n, alpha, x, incx, ap);
}
void PackedHermRank1Update(const char *uplo, size_t n, const double &alpha,
	const std::complex<double> *x, size_t incx, std::complex<double> *ap){
	BLASNAME(zhpr,ZHPR)(uplo, n, alpha, x, incx, ap);
}

void HermRank2Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy,
	float *a, size_t lda){
	BLASNAME(ssyr2,SSYR2)(uplo, n, alpha, x, incx, y, incy, a, lda);
}
void HermRank2Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy,
	double *a, size_t lda){
	BLASNAME(dsyr2,DSYR2)(uplo, n, alpha, x, incx, y, incy, a, lda);
}
void HermRank2Update(const char *uplo, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx, const std::complex<float> *y, size_t incy,
	std::complex<float> *a, size_t lda){
	BLASNAME(cher2,CHER2)(uplo, n, alpha, x, incx, y, incy, a, lda);
}
void HermRank2Update(const char *uplo, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx, const std::complex<double> *y, size_t incy,
	std::complex<double> *a, size_t lda){
	BLASNAME(zher2,ZHER2)(uplo, n, alpha, x, incx, y, incy, a, lda);
}

void PackedHermRank2Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy, float *a){
	BLASNAME(sspr2,SSPR2)(uplo, n, alpha, x, incx, y, incy, a);
}
void PackedHermRank2Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy, double *a){
	BLASNAME(dspr2,DSPR2)(uplo, n, alpha, x, incx, y, incy, a);
}
void PackedHermRank2Update(const char *uplo, size_t n, const std::complex<float> &alpha,
	const std::complex<float> *x, size_t incx, const std::complex<float> *y, size_t incy, std::complex<float> *a){
	BLASNAME(chpr2,CHPR2)(uplo, n, alpha, x, incx, y, incy, a);
}
void PackedHermRank2Update(const char *uplo, size_t n, const std::complex<double> &alpha,
	const std::complex<double> *x, size_t incx, const std::complex<double> *y, size_t incy, std::complex<double> *a){
	BLASNAME(zhpr2,ZHPR2)(uplo, n, alpha, x, incx, y, incy, a);
}

void SymRank1Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, float *a, size_t lda){
	BLASNAME(ssyr,SSYR)(uplo, n, alpha, x, incx, a, lda);
}
void SymRank1Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, double *a, size_t lda){
	BLASNAME(dsyr,DSYR)(uplo, n, alpha, x, incx, a, lda);
}

void PackedSymRank1Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, float *ap){
	BLASNAME(sspr,SSPR)(uplo, n, alpha, x, incx, ap);
}
void PackedSymRank1Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, double *ap){
	BLASNAME(dspr,DSPR)(uplo, n, alpha, x, incx, ap);
}

void SymRank2Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy,
	float *a, size_t lda){
	BLASNAME(ssyr2,SSYR2)(uplo, n, alpha, x, incx, y, incy, a, lda);
}
void SymRank2Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy,
	double *a, size_t lda){
	BLASNAME(dsyr2,DSYR2)(uplo, n, alpha, x, incx, y, incy, a, lda);
}

void PackedSymRank2Update(const char *uplo, size_t n, const float &alpha,
	const float *x, size_t incx, const float *y, size_t incy, float *a){
	BLASNAME(sspr2,SSPR2)(uplo, n, alpha, x, incx, y, incy, a);
}
void PackedSymRank2Update(const char *uplo, size_t n, const double &alpha,
	const double *x, size_t incx, const double *y, size_t incy, double *a){
	BLASNAME(dspr2,DSPR2)(uplo, n, alpha, x, incx, y, incy, a);
}

// Level 3
void MultMM(const char *transa, const char *transb, size_t m, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda, const float *b, size_t ldb,
	const float &beta, float *c, size_t ldc){
	BLASNAME(sgemm,SGEMM)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void MultMM(const char *transa, const char *transb, size_t m, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda, const double *b, size_t ldb,
	const double &beta, double *c, size_t ldc){
	BLASNAME(dgemm,DGEMM)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void MultMM(const char *transa, const char *transb, size_t m, size_t n, size_t k,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *b, size_t ldb,
	const std::complex<float> &beta, std::complex<float> *c, size_t ldc){
	BLASNAME(cgemm,CGEMM)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void MultMM(const char *transa, const char *transb, size_t m, size_t n, size_t k,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *b, size_t ldb,
	const std::complex<double> &beta, std::complex<double> *c, size_t ldc){
	BLASNAME(zgemm,ZGEMM)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void MultSymM(const char *side, const char *uplo, size_t m, size_t n,
	const float &alpha, const float *a, size_t lda, const float *b, size_t ldb,
	const float &beta, float *c, size_t ldc){
	BLASNAME(ssymm,SSYMM)(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
void MultSymM(const char *side, const char *uplo, size_t m, size_t n,
	const double &alpha, const double *a, size_t lda, const double *b, size_t ldb,
	const double &beta, double *c, size_t ldc){
	BLASNAME(dsymm,DSYMM)(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
void MultSymM(const char *side, const char *uplo, size_t m, size_t n,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *b, size_t ldb,
	const std::complex<float> &beta, std::complex<float> *c, size_t ldc){
	BLASNAME(csymm,CSYMM)(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
void MultSymM(const char *side, const char *uplo, size_t m, size_t n,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *b, size_t ldb,
	const std::complex<double> &beta, std::complex<double> *c, size_t ldc){
	BLASNAME(zsymm,ZSYMM)(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void MultHermM(const char *side, const char *uplo, size_t m, size_t n,
	const float &alpha, const float *a, size_t lda, const float *b, size_t ldb,
	const float &beta, float *c, size_t ldc){
	BLASNAME(ssymm,SSYMM)(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
void MultHermM(const char *side, const char *uplo, size_t m, size_t n,
	const double &alpha, const double *a, size_t lda, const double *b, size_t ldb,
	const double &beta, double *c, size_t ldc){
	BLASNAME(dsymm,DSYMM)(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
void MultHermM(const char *side, const char *uplo, size_t m, size_t n,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda, const std::complex<float> *b, size_t ldb,
	const std::complex<float> &beta, std::complex<float> *c, size_t ldc){
	BLASNAME(chemm,CHEMM)(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
void MultHermM(const char *side, const char *uplo, size_t m, size_t n,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda, const std::complex<double> *b, size_t ldb,
	const std::complex<double> &beta, std::complex<double> *c, size_t ldc){
	BLASNAME(zhemm,ZHEMM)(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SymRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda,
	const float &beta, float *c, size_t ldc){
	BLASNAME(ssyrk,SSYRK)(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
void SymRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda,
	const double &beta, double *c, size_t ldc){
	BLASNAME(dsyrk,DSYRK)(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
void SymRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda,
	const std::complex<float> &beta, std::complex<float> *c, size_t ldc){
	BLASNAME(csyrk,CSYRK)(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
void SymRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda,
	const std::complex<double> &beta, std::complex<double> *c, size_t ldc){
	BLASNAME(zsyrk,ZSYRK)(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void HermRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda,
	const float &beta, float *c, size_t ldc){
	BLASNAME(ssyrk,SSYRK)(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
void HermRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda,
	const double &beta, double *c, size_t ldc){
	BLASNAME(dsyrk,DSYRK)(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
void HermRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const float &alpha, const std::complex<float> *a, size_t lda,
	const float &beta, std::complex<float> *c, size_t ldc){
	BLASNAME(cherk,CHERK)(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
void HermRankKUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const double &alpha, const std::complex<double> *a, size_t lda,
	const double &beta, std::complex<double> *c, size_t ldc){
	BLASNAME(zherk,ZHERK)(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void SymRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda,
	const float *b, size_t ldb,
	const float &beta, float *c, size_t ldc){
	BLASNAME(ssyr2k,SSYR2K)(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void SymRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda,
	const double *b, size_t ldb,
	const double &beta, double *c, size_t ldc){
	BLASNAME(dsyr2k,DSYR2K)(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void SymRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda,
	const std::complex<float> *b, size_t ldb,
	const std::complex<float> &beta, std::complex<float> *c, size_t ldc){
	BLASNAME(csyr2k,CSYR2K)(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void SymRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda,
	const std::complex<double> *b, size_t ldb,
	const std::complex<double> &beta, std::complex<double> *c, size_t ldc){
	BLASNAME(zsyr2k,ZSYR2K)(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void HermRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const float &alpha, const float *a, size_t lda,
	const float *b, size_t ldb,
	const float &beta, float *c, size_t ldc){
	BLASNAME(ssyr2k,SSYR2K)(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void HermRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const double &alpha, const double *a, size_t lda,
	const double *b, size_t ldb,
	const double &beta, double *c, size_t ldc){
	BLASNAME(dsyr2k,DSYR2K)(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void HermRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const std::complex<float> &alpha, const std::complex<float> *a, size_t lda,
	const std::complex<float> *b, size_t ldb,
	const float &beta, std::complex<float> *c, size_t ldc){
	BLASNAME(cher2k,CHER2K)(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void HermRank2KUpdate(const char *uplo, const char *trans, size_t n, size_t k,
	const std::complex<double> &alpha, const std::complex<double> *a, size_t lda,
	const std::complex<double> *b, size_t ldb,
	const double &beta, std::complex<double> *c, size_t ldc){
	BLASNAME(zher2k,ZHER2K)(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void MultTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const float &alpha, const float *a, size_t lda,
	float *b, size_t ldb){
	BLASNAME(strmm,STRMM)(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}
void MultTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const double &alpha, const double *a, size_t lda,
	double *b, size_t ldb){
	BLASNAME(dtrmm,DTRMM)(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}
void MultTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const std::complex<float> &alpha, const std::complex<float> *a, size_t lda,
	std::complex<float> *b, size_t ldb){
	BLASNAME(ctrmm,CTRMM)(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}
void MultTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const std::complex<double> &alpha, const std::complex<double> *a, size_t lda,
	std::complex<double> *b, size_t ldb){
	BLASNAME(ztrmm,ZTRMM)(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}

void SolveTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const float &alpha, const float *a, size_t lda,
	float *b, size_t ldb){
	BLASNAME(strsm,STRSM)(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}
void SolveTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const double &alpha, const double *a, size_t lda,
	double *b, size_t ldb){
	BLASNAME(dtrsm,DTRSM)(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}
void SolveTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const std::complex<float> &alpha, const std::complex<float> *a, size_t lda,
	std::complex<float> *b, size_t ldb){
	BLASNAME(ctrsm,CTRSM)(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}
void SolveTrM(const char *side, const char *uplo, const char *trans, const char *diag,
	size_t m, size_t n, const std::complex<double> &alpha, const std::complex<double> *a, size_t lda,
	std::complex<double> *b, size_t ldb){
	BLASNAME(ztrsm,ZTRSM)(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}

} // namespace BLAS
} // namespace RNP
