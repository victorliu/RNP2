#ifndef RNP_BLAS_EXT_HPP_INCLUDED
#define RNP_BLAS_EXT_HPP_INCLUDED

#include <complex>

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

// Extra
void Set(size_t n, const float &val, float *x, size_t incx);
void Set(size_t n, const double &val, double *x, size_t incx);
void Set(size_t n, const std::complex<float> &val, std::complex<float> *x, size_t incx);
void Set(size_t n, const std::complex<double> &val, std::complex<double> *x, size_t incx);

void Set(size_t m, size_t n, const float &offdiag, const float &diag, float *a, size_t lda);
void Set(size_t m, size_t n, const double &offdiag, const double &diag, double *a, size_t lda);
void Set(size_t m, size_t n, const std::complex<float> &offdiag, const std::complex<float> &diag, std::complex<float> *a, size_t lda);
void Set(size_t m, size_t n, const std::complex<double> &offdiag, const std::complex<double> &diag, std::complex<double> *a, size_t lda);

void Copy(size_t m, size_t n, const float *src, size_t ldsrc, float *dst, size_t lddst);
void Copy(size_t m, size_t n, const double *src, size_t ldsrc, double *dst, size_t lddst);
void Copy(size_t m, size_t n, const std::complex<float> *src, size_t ldsrc, std::complex<float> *dst, size_t lddst);
void Copy(size_t m, size_t n, const std::complex<double> *src, size_t ldsrc, std::complex<double> *dst, size_t lddst);

void Copy(const char *trans, size_t m, size_t n, const float *src, size_t ldsrc, float *dst, size_t lddst);
void Copy(const char *trans, size_t m, size_t n, const double *src, size_t ldsrc, double *dst, size_t lddst);
void Copy(const char *trans, size_t m, size_t n, const std::complex<float> *src, size_t ldsrc, std::complex<float> *dst, size_t lddst);
void Copy(const char *trans, size_t m, size_t n, const std::complex<double> *src, size_t ldsrc, std::complex<double> *dst, size_t lddst);

void Conjugate(size_t n, float *x, size_t incx);
void Conjugate(size_t n, double *x, size_t incx);
void Conjugate(size_t n, std::complex<float> *x, size_t incx);
void Conjugate(size_t n, std::complex<double> *x, size_t incx);

void Rescale(const char *type, size_t kl, size_t ku, const float &cfrom, const float &cto, size_t m, size_t n, float *a, size_t lda);
void Rescale(const char *type, size_t kl, size_t ku, const double &cfrom, const double &cto, size_t m, size_t n, double *a, size_t lda);
void Rescale(const char *type, size_t kl, size_t ku, const float &cfrom, const float &cto, size_t m, size_t n, std::complex<float> *a, size_t lda);
void Rescale(const char *type, size_t kl, size_t ku, const double &cfrom, const double &cto, size_t m, size_t n, std::complex<double> *a, size_t lda);

float Norm1(size_t n, const float *x, size_t incx);
double Norm1(size_t n, const double *x, size_t incx);
float Norm1(size_t n, const std::complex<float> *x, size_t incx);
double Norm1(size_t n, const std::complex<double> *x, size_t incx);

// Level 1 rotations
void RotGen(float  *a, const float  &b, float  *c, float  *s);
void RotGen(double *a, const double &b, double *c, double *s);
void RotGen(std::complex<float>  *a, const std::complex<float>  &b, std::complex<float>  *c, std::complex<float>  *s);
void RotGen(std::complex<double> *a, const std::complex<double> &b, std::complex<double> *c, std::complex<double> *s);

void ModifiedRotGen(float  *d1, float  *d2, float  *x1, const float  &x2, float  param[5]);
void ModifiedRotGen(double *d1, double *d2, double *x1, const double &x2, double param[5]);

void RotApply(size_t n, float  *x, size_t incx, float  *y, size_t incy, const float  &c, const float  &s);
void RotApply(size_t n, double *x, size_t incx, double *y, size_t incy, const double &c, const double &s);
void RotApply(size_t n, std::complex<float>  *x, size_t incx, std::complex<float>  *y, size_t incy, const std::complex<float>  &c, const std::complex<float>  &s);
void RotApply(size_t n, std::complex<double> *x, size_t incx, std::complex<double> *y, size_t incy, const std::complex<double> &c, const std::complex<double> &s);

void ModifiedRotApply(size_t n, float  *x, size_t incx, float  *y, size_t incy, const float  param[5]);
void ModifiedRotApply(size_t n, double *x, size_t incx, double *y, size_t incy, const double param[5]);

// Level 1 utility
void Swap(size_t n, float *x, size_t incx, float *y, size_t incy);
void Swap(size_t n, double *x, size_t incx, double *y, size_t incy);
void Swap(size_t n, std::complex<float> *x, size_t incx, std::complex<float> *y, size_t incy);
void Swap(size_t n, std::complex<double> *x, size_t incx, std::complex<double> *y, size_t incy);

void Scale(size_t n, const float &alpha, float *x, size_t incx);
void Scale(size_t n, const double &alpha, double *x, size_t incx);
void Scale(size_t n, const std::complex<float> &alpha, std::complex<float> *x, size_t incx);
void Scale(size_t n, const std::complex<double> &alpha, std::complex<double> *x, size_t incx);
void Scale(size_t n, const float &alpha, std::complex<float> *x, size_t incx);
void Scale(size_t n, const double &alpha, std::complex<double> *x, size_t incx);

void Copy(size_t n, const float *src, size_t incsrc, float *dst, size_t incdst);
void Copy(size_t n, const double *src, size_t incsrc, double *dst, size_t incdst);
void Copy(size_t n, const std::complex<float> *src, size_t incsrc, std::complex<float> *dst, size_t incdst);
void Copy(size_t n, const std::complex<double> *src, size_t incsrc, std::complex<double> *dst, size_t incdst);

void Axpy(size_t n, const float &alpha, const float *x, size_t incx, float *y, size_t incy);
void Axpy(size_t n, const double &alpha, const double *x, size_t incx, double *y, size_t incy);
void Axpy(size_t n, const std::complex<float> &alpha, const std::complex<float> *x, size_t incx, std::complex<float> *y, size_t incy);
void Axpy(size_t n, const std::complex<double> &alpha, const std::complex<double> *x, size_t incx, std::complex<double> *y, size_t incy);

float Dot(size_t n, const float *x, size_t incx, const float *y, size_t incy);
double Dot(size_t n, const double *x, size_t incx, const double *y, size_t incy);
std::complex<float> Dot(size_t n, const std::complex<float> *x, size_t incx, const std::complex<float> *y, size_t incy);
std::complex<double> Dot(size_t n, const std::complex<double> *x, size_t incx, const std::complex<double> *y, size_t incy);

double DotEx(size_t n, const float *x, size_t incx, const float *y, size_t incy);
float  DotEx(size_t n, const float &b, const float *x, size_t incx, const float *y, size_t incy);

float ConjugateDot(size_t n, const float *x, size_t incx, const float *y, size_t incy);
double ConjugateDot(size_t n, const double *x, size_t incx, const double *y, size_t incy);
std::complex<float> ConjugateDot(size_t n, const std::complex<float> *x, size_t incx, const std::complex<float> *y, size_t incy);
std::complex<double> ConjugateDot(size_t n, const std::complex<double> *x, size_t incx, const std::complex<double> *y, size_t incy);

float Norm2(size_t n, const float *x, size_t incx);
double Norm2(size_t n, const double *x, size_t incx);
float Norm2(size_t n, const std::complex<float> *x, size_t incx);
double Norm2(size_t n, const std::complex<double> *x, size_t incx);

float Asum(size_t n, const float *x, size_t incx);
double Asum(size_t n, const double *x, size_t incx);
float Asum(size_t n, const std::complex<float> *x, size_t incx);
double Asum(size_t n, const std::complex<double> *x, size_t incx);

// These return a 0-based index
size_t MaximumIndex(size_t n, const float *x, size_t incx);
size_t MaximumIndex(size_t n, const double *x, size_t incx);
size_t MaximumIndex(size_t n, const std::complex<float> *x, size_t incx);
size_t MaximumIndex(size_t n, const std::complex<double> *x, size_t incx);

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

#endif // RNP_BLAS_EXT_HPP_INCLUDED
