#ifndef FBLAS_HPP_INCLUDED
#define FBLAS_HPP_INCLUDED

#include <complex>

namespace FBLAS{

typedef int integer;
typedef unsigned int uint;
typedef std::complex<double> dcomplex;
typedef std::complex<float>  fcomplex;

// 138 routines
#define BLASNAME(lo,up) lo ## _

#ifdef BLAS_RETURN_ARG

extern "C" void BLASNAME(cdotc,CDOTC)(
	fcomplex *dotval, const uint &n,
	const fcomplex *cx, const integer &incx,
	const fcomplex *cy, const integer &incy
);

extern "C" void BLASNAME(cdotu,CDOTU)(
	fcomplex *dotval, const uint &n,
	const fcomplex *cx, const integer &incx,
	const fcomplex *cy, const integer &incy
);

extern "C" void BLASNAME(zdotc,ZDOTC)(
	dcomplex *dotval, const uint &n,
	const dcomplex *cx, const integer &incx,
	const dcomplex *cy, const integer &incy
);

extern "C" void BLASNAME(zdotu,ZDOTU)(
	dcomplex *dotval, const uint &n,
	const dcomplex *cx, const integer &incx,
	const dcomplex *cy, const integer &incy
);

#else // !BLAS_RETURN_ARG

extern "C" fcomplex BLASNAME(cdotc,CDOTC)(
	const uint &n,
	const fcomplex *cx, const integer &incx,
	const fcomplex *cy, const integer &incy
);

extern "C" fcomplex BLASNAME(cdotu,CDOTU)(
	const uint &n,
	const fcomplex *cx, const integer &incx,
	const fcomplex *cy, const integer &incy
);

extern "C" dcomplex BLASNAME(zdotc,ZDOTC)(
	const uint &n,
	const dcomplex *cx, const integer &incx,
	const dcomplex *cy, const integer &incy
);

extern "C" dcomplex BLASNAME(zdotu,ZDOTU)(
	const uint &n,
	const dcomplex *cx, const integer &incx,
	const dcomplex *cy, const integer &incy
);

#endif // BLAS_RETURN_ARG

extern "C" integer BLASNAME(isamax,ISAMAX)(
	const uint &n, const float *x, const integer &incx
);
extern "C" integer BLASNAME(idamax,IDAMAX)(
	const uint &n, const double *x, const integer &incx
);
extern "C" integer BLASNAME(icamax,ICAMAX)(
	const uint &n, const std::complex<float> *x, const integer &incx
);
extern "C" integer BLASNAME(izamax,IZAMAX)(
	const uint &n, const std::complex<double> *x, const integer &incx
);

extern "C" float BLASNAME(scasum,SCASUM)(
	const uint &n, const fcomplex *cx, const integer &incx
);

extern "C" double BLASNAME(dzasum,DZASUM)(
	const uint &n, const dcomplex *cx, const integer &incx
);

extern "C" float BLASNAME(scnrm2,SCNRM2)(
	const uint &n, const fcomplex *x, const integer &incx
);

extern "C" double BLASNAME(dznrm2,DZNRM2)(
	const uint &n, const dcomplex *x, const integer &incx
);

extern "C" float BLASNAME(snrm2,SNRM2)(
	const uint &n, const float *x, const integer &incx
);

extern "C" double BLASNAME(dnrm2,DNRM2)(
	const uint &n, const double *x, const integer &incx
);

extern "C" float BLASNAME(sasum,SASUM)(
	const uint &n, const float *sx, const integer &incx
);

extern "C" float BLASNAME(dasum,DASUM)(
	const uint &n, const double *sx, const integer &incx
);

extern "C" float BLASNAME(ddot,DDOT)(
	const uint &n,
	const double *sx, const integer &incx,
	const double *sy, const integer &incy
);

extern "C" float BLASNAME(sdot,SDOT)(
	const uint &n,
	const float *sx, const integer &incx,
	const float *sy, const integer &incy
);

extern "C" float BLASNAME(sdsdot,SDSDOT)(
	const uint &n, const float &b,
	const float *sx, const integer &incx,
	const float *sy, const integer &incy
);

extern "C" double BLASNAME(dsdot,DSDOT)(
	const uint &n,
	const float *sx, const integer &incx,
	const float *sy, const integer &incy
);

extern "C" integer BLASNAME(caxpy,CAXPY)(
	const uint &n,
	const fcomplex &ca, const fcomplex *cx, const integer &incx,
	fcomplex *cy, const integer &incy
);

extern "C" integer BLASNAME(ccopy,CCOPY)(
	const uint &n,
	const fcomplex *cx, const integer &incx,
	fcomplex *cy, const integer &incy
);

extern "C" integer BLASNAME(cgbmv,CGBMV)(
	const char *trans,
	const uint &m, const uint &n, const uint &kl, const uint &ku,
	const fcomplex &alpha, const fcomplex *a, const uint &lda,
	const fcomplex *x, const integer &incx,
	const fcomplex &beta, fcomplex *y, const integer &incy
);

extern "C" integer BLASNAME(cgemm,CGEMM)(
	const char *transa, const char *transb,
	const uint &m, const uint &n, const uint &k,
	const fcomplex &alpha, const fcomplex *a, const uint &lda,
	const fcomplex *b, const uint &ldb,
	const fcomplex &beta, fcomplex *c, const uint &ldc
);

extern "C" integer BLASNAME(cgemv,CGEMV)(
	const char *trans,
	const uint &m, const uint &n,
	const fcomplex &alpha, const fcomplex *a, const uint &lda,
	const fcomplex *x, const integer &incx,
	const fcomplex &beta, fcomplex *y, const integer &incy
);

extern "C" integer BLASNAME(cgerc,CGERC)(
	const uint &m, const uint &n,
	const fcomplex &alpha, const fcomplex *x, const integer &incx,
	const fcomplex *y, const integer &incy,
	fcomplex *a, const uint &lda
);

extern "C" integer BLASNAME(cgeru,CGERU)(
	const uint &m, const uint &n,
	const fcomplex &alpha, const fcomplex *x, const integer &incx,
	const fcomplex *y, const integer &incy,
	fcomplex *a, const uint &lda
);

extern "C" integer BLASNAME(chbmv,CHBMV)(
	const char *uplo,
	const uint &n, const uint &k,
	const fcomplex &alpha, const fcomplex *a, const uint &lda,
	const fcomplex *x, const integer &incx,
	const fcomplex &beta, fcomplex *y, const integer &incy
);

extern "C" integer BLASNAME(chemm,CHEMM)(
	const char *side, const char *uplo,
	const uint &m, const uint &n,
	const fcomplex &alpha, const fcomplex *a, const uint &lda,
	const fcomplex *b, const uint &ldb,
	const fcomplex &beta, fcomplex *c, const uint &ldc
);

extern "C" integer BLASNAME(chemv,CHEMV)(
	const char *uplo,
	const uint &n,
	const fcomplex &alpha, const fcomplex *a, const uint &lda,
	const fcomplex *x, const integer &incx,
	const fcomplex &beta, fcomplex *y, const integer &incy
);

extern "C" integer BLASNAME(cher,CHER)(
	const char *uplo,
	const uint &n,
	const float &alpha, const fcomplex *x, const integer &incx,
	fcomplex *a, const uint &lda
);

extern "C" integer BLASNAME(cher2,CHER2)(
	const char *uplo,
	const uint &n,
	const fcomplex &alpha, const fcomplex *x, const integer &incx,
	const fcomplex *y, const integer &incy,
	fcomplex *a, const uint &lda
);

extern "C" integer BLASNAME(cher2k,CHER2K)(
	const char *uplo, const char *trans,
	const uint &n, const uint &k,
	const fcomplex &alpha, const fcomplex *a, const uint &lda,
	const fcomplex *b, const uint &ldb,
	const float &beta, fcomplex *c, const uint &ldc
);

extern "C" integer BLASNAME(cherk,CHERK)(
	const char *uplo, const char *trans,
	const uint &n, const uint &k,
	const fcomplex &alpha, const fcomplex *a, const uint &lda,
	const fcomplex &beta, fcomplex *c, const uint &ldc
);

extern "C" integer BLASNAME(chpmv,CHPMV)(
	const char *uplo,
	const uint &n,
	const fcomplex &alpha, const fcomplex *ap,
	const fcomplex *x, const integer &incx,
	const fcomplex &beta,
	fcomplex *y, const integer &incy
);

extern "C" integer BLASNAME(chpr,CHPR)(
	const char *uplo,
	const uint &n,
	const float &alpha, const fcomplex *x, const integer &incx,
	fcomplex *ap
);

extern "C" integer BLASNAME(chpr2,CHPR2)(
	const char *uplo,
	const uint &n,
	const fcomplex &alpha, const fcomplex *x, const integer &incx,
	const fcomplex *y, const integer &incy,
	fcomplex *ap
);

extern "C" integer BLASNAME(crotg,CROTG)(
	fcomplex *ca, const fcomplex &cb, float *c, fcomplex *s
);

extern "C" integer BLASNAME(cscal,CSCAL)(
	const uint &n, const fcomplex &ca,
	fcomplex *cx, const integer &incx
);

extern "C" integer BLASNAME(csscal,CSSCAL)(
	const uint &n,
	const float &sa,
	fcomplex *cx, const integer &incx
);

extern "C" integer BLASNAME(cswap,CSWAP)(
	const uint &n,
	fcomplex *cx, const integer &incx,
	fcomplex *cy, const integer &incy
);

extern "C" integer BLASNAME(csymm,CSYMM)(
	const char *side, const char *uplo,
	const uint &m, const uint &n,
	const fcomplex &alpha, const fcomplex *a, const uint &lda,
	const fcomplex *b, const uint &ldb,
	const fcomplex &beta, fcomplex *c, const uint &ldc
);

extern "C" integer BLASNAME(csyr2k,CSYR2K)(
	const char *uplo, const char *trans,
	const uint &n, const uint &k,
	const fcomplex &alpha, const fcomplex *a, const uint &lda,
	const fcomplex *b, const uint &ldb,
	const fcomplex &beta, fcomplex *c, const uint &ldc
);

extern "C" integer BLASNAME(csyrk,CSYRK)(
	const char *uplo, const char *trans,
	const uint &n, const uint &k,
	const fcomplex &alpha, const fcomplex *a, const uint &lda,
	const fcomplex &beta, fcomplex *c, const uint &ldc
);

extern "C" integer BLASNAME(ctbmv,CTBMV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const uint &k,
	const fcomplex *a, const uint &lda,
	fcomplex *x, const integer &incx
);

extern "C" integer BLASNAME(ctbsv,CTBSV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const uint &k,
	const fcomplex *a, const uint &lda,
	fcomplex *x, const integer &incx
);

extern "C" integer BLASNAME(ctpmv,CTPMV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const fcomplex *ap,
	fcomplex *x, const integer &incx
);

extern "C" integer BLASNAME(ctpsv,CTPSV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const fcomplex *ap,
	fcomplex *x, const integer &incx
);

extern "C" integer BLASNAME(ctrmm,CTRMM)(
	const char *side, const char *uplo, const char *transa, const char *diag,
	const uint &m, const uint &n,
	const fcomplex &alpha, const fcomplex *a, const uint &lda,
	fcomplex *b, const uint &ldb
);

extern "C" integer BLASNAME(ctrmv,CTRMV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const fcomplex *a, const uint &lda,
	fcomplex *x, const integer &incx
);

extern "C" integer BLASNAME(ctrsm,CTRSM)(
	const char *side, const char *uplo, const char *transa, const char *diag,
	const uint &m, const uint &n,
	const fcomplex &alpha, const fcomplex *a, const uint &lda, 
	fcomplex *b, const uint &ldb
);

extern "C" integer BLASNAME(ctrsv,CTRSV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const fcomplex *a, const uint &lda,
	fcomplex *x, const integer &incx
);

// Double precision routines

extern "C" integer BLASNAME(daxpy,DAXPY)(
	const uint &n,
	const double &a, const double *x, const integer &incx,
	double *y, const integer &incy
);

extern "C" integer BLASNAME(dcopy,DCOPY)(
	const uint &n, const double *src, const integer &incsrc,
	double *dst, const integer &incdst
);

extern "C" integer BLASNAME(dgbmv,DGBMV)(
	const char *trans,
	const uint &m, const uint &n, const uint &kl, const uint &ku,
	const double &alpha, const double *a, const uint &lda,
	const double *x, const integer &incx,
	const double &beta, double *y, const integer &incy
);

extern "C" integer BLASNAME(dgemm,DGEMM)(
	const char *transa, const char *transb,
	const uint &m, const uint &n, const uint &k,
	const double &alpha, const double *a, const uint &lda,
	const double *b, const uint &ldb,
	const double &beta, double *c, const uint &ldc
);

extern "C" integer BLASNAME(dgemv,DGEMV)(
	const char *trans,
	const uint &m, const uint &n,
	const double &alpha, const double *a, const uint &lda,
	const double *x, const integer &incx,
	const double &beta, double *y, const integer &incy
);

extern "C" integer BLASNAME(dger,DGER)(
	const uint &m, const uint &n,
	const double &alpha, const double *x, const integer &incx,
	const double *y, const integer &incy,
	double *a, const uint &lda
);

extern "C" integer BLASNAME(drot,DROT)(
	const uint &n,
	double *sx, const integer &incx,
	double *sy, const integer &incy,
	const double &c, const double &s
);

extern "C" integer BLASNAME(drotg,DROTG)(
	double *sa, const double &sb, double *c, double *s
);

extern "C" integer BLASNAME(drotm,DROTM)(
	const uint &n, double *x, const integer &incx,
	double *y, const integer &incy, const double param[5]
);

extern "C" integer BLASNAME(drotmg,DROTMG)(
	double *d1, double *d2, double *x1, const double &x2, double param[5]
);

extern "C" integer BLASNAME(dsbmv,DSBMV)(
	const char *uplo,
	const uint &n, const uint &k,
	const double &alpha, const double *a, const uint &lda,
	const double *x, const integer &incx,
	const double &beta, double *y, const integer &incy
);

extern "C" integer BLASNAME(dscal,DSCAL)(
	const uint &n,
	const double &sa,
	double *sx, const integer &incx
);

extern "C" integer BLASNAME(dspmv,DSPMV)(
	const char *uplo,
	const uint &n,
	const double &alpha, const double *ap,
	const double *x, const integer &incx,
	const double &beta, double *y, const integer &incy
);

extern "C" integer BLASNAME(dspr,DSPR)(
	const char *uplo,
	const uint &n,
	const double &alpha, const double *x, const integer &incx,
	double *ap
);

extern "C" integer BLASNAME(dspr2,DSPR2)(
	const char *uplo,
	const uint &n,
	const double &alpha, const double *x, const integer &incx,
	const double *y, const integer &incy,
	double *ap
);

extern "C" integer BLASNAME(dswap,DSWAP)(
	const uint &n,
	double *sx, const integer &incx,
	double *sy, const integer &incy
);

extern "C" integer BLASNAME(dsymm,DSYMM)(
	const char *side, const char *uplo,
	const uint &m, const uint &n,
	const double &alpha, const double *a, const uint &lda,
	const double *b, const uint &ldb,
	const double &beta, double *c, const uint &ldc
);

extern "C" integer BLASNAME(dsymv,DSYMV)(
	const char *uplo,
	const uint &n,
	const double &alpha, const double *a, const uint &lda,
	const double *x, const integer &incx,
	const double &beta, double *y, const integer &incy
);

extern "C" integer BLASNAME(dsyr,DSYR)(
	const char *uplo,
	const uint &n,
	const double &alpha, const double *x, const integer &incx,
	double *a, const uint &lda
);

extern "C" integer BLASNAME(dsyr2,DSYR2)(
	const char *uplo,
	const uint &n,
	const double &alpha, const double *x, const integer &incx,
	const double *y, const integer &incy,
	double *a, const uint &lda
);

extern "C" integer BLASNAME(dsyr2k,DSYR2K)(
	const char *uplo, const char *trans,
	const uint &n, const uint &k,
	const double &alpha, const double *a, const uint &lda,
	const double *b, const uint &ldb,
	const double &beta, double *c, const uint &ldc
);

extern "C" integer BLASNAME(dsyrk,DSYRK)(
	const char *uplo, const char *trans,
	const uint &n, const uint &k,
	const double &alpha, const double *a, const uint &lda,
	const double &beta, double *c, const uint &ldc
);

extern "C" integer BLASNAME(dtbmv,DTBMV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const uint &k,
	const double *a, const uint &lda,
	double *x, const integer &incx
);

extern "C" integer BLASNAME(dtbsv,DTBSV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const uint &k,
	const double *a, const uint &lda,
	double *x, const integer &incx
);

extern "C" integer BLASNAME(dtpmv,DTPMV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const double *ap,
	double *x, const integer &incx
);

extern "C" integer BLASNAME(dtpsv,DTPSV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const double *ap,
	double *x, const integer &incx
);

extern "C" integer BLASNAME(dtrmm,DTRMM)(
	const char *side, const char *uplo, const char *transa, const char *diag,
	const uint &m, const uint &n,
	const double &alpha, const double *a, const uint &lda,
	double *b, const uint &ldb
);

extern "C" integer BLASNAME(dtrmv,DTRMV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n,
	const double *a, const uint &lda,
	double *x, const integer &incx
);

extern "C" integer BLASNAME(dtrsm,DTRSM)(
	const char *side, const char *uplo, const char *transa, const char *diag,
	const uint &m, const uint &n,
	const double &alpha, const double *a, const uint &lda,
	double *b, const uint &ldb
);

extern "C" integer BLASNAME(dtrsv,DTRSV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const double *a, const uint &lda,
	double *x, const integer &incx
);

// Float precision routines

extern "C" integer BLASNAME(saxpy,SAXPY)(
	const uint &n,
	const float &sa, const float *sx, const integer &incx,
	float *sy, const integer &incy
);

extern "C" integer BLASNAME(scopy,SCOPY)(
	const uint &n,
	const float *sx, const integer &incx,
	float *sy, const integer &incy
);

extern "C" integer BLASNAME(sgbmv,SGBMV)(
	const char *trans,
	const uint &m, const uint &n, const uint &kl, const uint &ku,
	const float &alpha, const float *a, const uint &lda,
	const float *x, const integer &incx,
	const float &beta, float *y, const integer &incy
);

extern "C" integer BLASNAME(sgemm,SGEMM)(
	const char *transa, const char *transb,
	const uint &m, const uint &n, const uint &k,
	const float &alpha, const float *a, const uint &lda,
	const float *b, const uint &ldb,
	const float &beta, float *c, const uint &ldc
);

extern "C" integer BLASNAME(sgemv,SGEMV)(
	const char *trans,
	const uint &m, const uint &n,
	const float &alpha, const float *a, const uint &lda,
	const float *x, const integer &incx,
	const float &beta, float *y, const integer &incy
);

extern "C" integer BLASNAME(sger,SGER)(
	const uint &m, const uint &n,
	const float &alpha, const float *x, const integer &incx,
	const float *y, const integer &incy,
	float *a, const uint &lda
);

extern "C" integer BLASNAME(srot,SROT)(
	const uint &n,
	float *sx, const integer &incx,
	float *sy, const integer &incy,
	const float &c, const float &s
);

extern "C" integer BLASNAME(srotg,SROTG)(
	float *sa, const float &sb, float *c, float *s
);

extern "C" integer BLASNAME(srotm,SROTM)(
	const uint &n, float *x, const integer &incx,
	float *y, const integer &incy, const float param[5]
);

extern "C" integer BLASNAME(srotmg,SROTMG)(
	float *d1, float *d2, float *x1, const float &x2, float param[5]
);

extern "C" integer BLASNAME(ssbmv,SSBMV)(
	const char *uplo,
	const uint &n, const uint &k,
	const float &alpha, const float *a, const uint &lda,
	const float *x, const integer &incx,
	const float &beta, float *y, const integer &incy
);

extern "C" integer BLASNAME(sscal,SSCAL)(
	const uint &n,
	const float &sa, float *sx, const integer &incx
);

extern "C" integer BLASNAME(sspmv,SSPMV)(
	const char *uplo,
	const uint &n,
	const float &alpha, const float *ap,
	const float *x, const integer &incx,
	const float &beta, float *y, const integer &incy
);

extern "C" integer BLASNAME(sspr,SSPR)(
	const char *uplo,
	const uint &n,
	const float &alpha, const float *x, const integer &incx,
	float *ap
);

extern "C" integer BLASNAME(sspr2,SSPR2)(
	const char *uplo,
	const uint &n,
	const float &alpha, const float *x, const integer &incx,
	const float *y, const integer &incy,
	float *ap
);

extern "C" integer BLASNAME(sswap,SSWAP)(
	const uint &n,
	float *sx, const integer &incx,
	float *sy, const integer &incy
);

extern "C" integer BLASNAME(ssymm,SSYMM)(
	const char *side, const char *uplo,
	const uint &m, const uint &n,
	const float &alpha, const float *a, const uint &lda,
	const float *b, const uint &ldb,
	const float &beta, float *c, const uint &ldc
);

extern "C" integer BLASNAME(ssymv,SSYMV)(
	const char *uplo,
	const uint &n,
	const float &alpha, const float *a, const uint &lda,
	const float *x, const integer &incx,
	const float &beta, float *y, const integer &incy
);

extern "C" integer BLASNAME(ssyr,SSYR)(
	const char *uplo,
	const uint &n,
	const float &alpha, const float *x, const integer &incx,
	float *a, const uint &lda
);

extern "C" integer BLASNAME(ssyr2,SSYR2)(
	const char *uplo,
	const uint &n,
	const float &alpha, const float *x, const integer &incx,
	const float *y, const integer &incy,
	float *a, const uint &lda
);

extern "C" integer BLASNAME(ssyr2k,SSYR2K)(
	const char *uplo, const char *trans,
	const uint &n, const uint &k,
	const float &alpha, const float *a, const uint &lda,
	const float *b, const uint &ldb,
	const float &beta, float *c, const uint &ldc
);

extern "C" integer BLASNAME(ssyrk,SSYRK)(
	const char *uplo, const char *trans,
	const uint &n, const uint &k,
	const float &alpha, const float *a, const uint &lda,
	const float &beta, float *c, const uint &ldc
);

extern "C" integer BLASNAME(stbmv,STBMV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const uint &k,
	const float *a, const uint &lda,
	float *x, const integer &incx
);

extern "C" integer BLASNAME(stbsv,STBSV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const uint &k,
	const float *a, const uint &lda,
	float *x, const integer &incx
);

extern "C" integer BLASNAME(stpmv,STPMV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n,
	const float *ap,
	float *x, const integer &incx
);

extern "C" integer BLASNAME(stpsv,STPSV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n,
	const float *ap,
	float *x, const integer &incx
);

extern "C" integer BLASNAME(strmm,STRMM)(
	const char *side, const char *uplo, const char *transa, const char *diag,
	const uint &m, const uint &n,
	const float &alpha, const float *a, const uint &lda,
	float *b, const uint &ldb
);

extern "C" integer BLASNAME(strmv,STRMV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n,
	const float *a, const uint &lda,
	float *x, const integer &incx
);

extern "C" integer BLASNAME(strsm,STRSM)(
	const char *side, const char *uplo, const char *transa, const char *diag,
	const uint &m, const uint &n,
	const float &alpha, const float *a, const uint &lda,
	float *b, const uint &ldb
);

extern "C" integer BLASNAME(strsv,STRSV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n,
	const float *a, const uint &lda,
	float *x, const integer &incx
);

// Double complex routines

extern "C" integer BLASNAME(zaxpy,ZAXPY)(
	const uint &n,
	const dcomplex &ca, const dcomplex *cx, const integer &incx,
	dcomplex *cy, const integer &incy
);

extern "C" integer BLASNAME(zcopy,ZCOPY)(
	const uint &n,
	const dcomplex *cx, const integer &incx,
	dcomplex *cy, const integer &incy
);

extern "C" integer BLASNAME(zdscal,ZDSCAL)(
	const uint &n, const double &sa, dcomplex *cx, const integer &incx
);

extern "C" integer BLASNAME(zgbmv,ZGBMV)(
	const char *trans,
	const uint &m, const uint &n, const uint &kl, const uint &ku,
	const dcomplex &alpha, const dcomplex *a, const uint &lda,
	const dcomplex *x, const integer &incx,
	const dcomplex &beta, dcomplex *y, const integer &incy
);

extern "C" integer BLASNAME(zgemm,ZGEMM)(
	const char *transa, const char *transb,
	const uint &m, const uint &n, const uint &k,
	const dcomplex &alpha, const dcomplex *a, const uint &lda,
	const dcomplex *b, const uint &ldb,
	const dcomplex &beta, dcomplex *c, const uint &ldc
);

extern "C" integer BLASNAME(zgemv,ZGEMV)(
	const char *trans,
	const uint &m, const uint &n,
	const dcomplex &alpha, const dcomplex *a, const uint &lda,
	const dcomplex *x, const integer &incx,
	const dcomplex &beta, dcomplex *y, const integer &incy
);

extern "C" integer BLASNAME(zgerc,ZGERC)(
	const uint &m, const uint &n,
	const dcomplex &alpha, const dcomplex *x, const integer &incx,
	const dcomplex *y, const integer &incy,
	dcomplex *a, const uint &lda
);

extern "C" integer BLASNAME(zgeru,ZGERU)(
	const uint &m, const uint &n,
	const dcomplex &alpha, const dcomplex *x, const integer &incx,
	const dcomplex *y, const integer &incy,
	dcomplex *a, const uint &lda
);

extern "C" integer BLASNAME(zhbmv,ZHBMV)(
	const char *uplo,
	const uint &n, const uint &k,
	const dcomplex &alpha, const dcomplex *a, const uint &lda,
	const dcomplex *x, const integer &incx,
	const dcomplex &beta, dcomplex *y, const integer &incy
);

extern "C" integer BLASNAME(zhemm,ZHEMM)(
	const char *side, const char *uplo,
	const uint &m, const uint &n,
	const dcomplex &alpha, const dcomplex *a, const uint &lda,
	const dcomplex *b, const uint &ldb,
	const dcomplex &beta, dcomplex *c, const uint &ldc
);

extern "C" integer BLASNAME(zhemv,ZHEMV)(
	const char *uplo,
	const uint &n,
	const dcomplex &alpha, const dcomplex *a, const uint &lda,
	const dcomplex *x, const integer &incx,
	const dcomplex &beta, dcomplex *y, const integer &incy
);

extern "C" integer BLASNAME(zher,ZHER)(
	const char *uplo,
	const uint &n,
	const double &alpha, const dcomplex *x, const integer &incx,
	dcomplex *a, const uint &lda
);

extern "C" integer BLASNAME(zher2,ZHER2)(
	const char *uplo,
	const uint &n,
	const dcomplex &alpha, const dcomplex *x, const integer &incx,
	const dcomplex *y, const integer &incy,
	dcomplex *a, const uint &lda
);

extern "C" integer BLASNAME(zher2k,ZHER2K)(
	const char *uplo, const char *trans,
	const uint &n, const uint &k,
	const dcomplex &alpha, const dcomplex *a, const uint &lda,
	const dcomplex *b, const uint &ldb,
	const double &beta, dcomplex *c, const uint &ldc
);

extern "C" integer BLASNAME(zherk,ZHERK)(
	const char *uplo, const char *trans,
	const uint &n, const uint &k,
	const double &alpha, const dcomplex *a, const uint &lda,
	const double &beta, dcomplex *c, const uint &ldc
);

extern "C" integer BLASNAME(zhpmv,ZHPMV)(
	const char *uplo,
	const uint &n,
	const dcomplex &alpha, const dcomplex *ap,
	const dcomplex *x, const integer &incx,
	const dcomplex &beta, dcomplex *y, const integer &incy
);

extern "C" integer BLASNAME(zhpr,ZHPR)(
	const char *uplo,
	const uint &n,
	const double &alpha, const dcomplex *x, const integer &incx,
	const dcomplex *ap
);

extern "C" integer BLASNAME(zhpr2,ZHPR2)(
	const char *uplo,
	const uint &n,
	const dcomplex &alpha, const dcomplex *x, const integer &incx,
	const dcomplex *y, const integer &incy,
	dcomplex *ap
);

extern "C" integer BLASNAME(zrotg,ZROTG)(
	dcomplex *ca, const dcomplex &cb, double *c, dcomplex *s
);

extern "C" integer BLASNAME(zscal,ZSCAL)(
	const uint &n,
	const dcomplex &ca, dcomplex *cx, const integer &incx
);

extern "C" integer BLASNAME(zswap,ZSWAP)(
	const uint &n,
	dcomplex *cx, const integer &incx,
	dcomplex *cy, const integer &incy
);

extern "C" integer BLASNAME(zsymm,ZSYMM)(
	const char *side, const char *uplo,
	const uint &m, const uint &n,
	const dcomplex &alpha, const dcomplex *a, const uint &lda,
	const dcomplex *b, const uint &ldb,
	const dcomplex &beta, dcomplex *c, const uint &ldc
);

extern "C" integer BLASNAME(zsyr2k,ZSYR2K)(
	const char *uplo, const char *trans,
	const uint &n, const uint &k,
	const dcomplex &alpha, const dcomplex *a, const uint &lda,
	const dcomplex *b, const uint &ldb,
	const dcomplex &beta, dcomplex *c, const uint &ldc
);

extern "C" integer BLASNAME(zsyrk,ZSYRK)(
	const char *uplo, const char *trans,
	const uint &n, const uint &k,
	const dcomplex &alpha, const dcomplex *a, const uint &lda,
	const dcomplex &beta, dcomplex *c, const uint &ldc
);

extern "C" integer BLASNAME(ztbmv,ZTBMV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const uint &k,
	const dcomplex *a, const uint &lda,
	dcomplex *x, const integer &incx
);

extern "C" integer BLASNAME(ztbsv,ZTBSV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const uint &k,
	const dcomplex *a, const uint &lda,
	dcomplex *x, const integer &incx
);

extern "C" integer BLASNAME(ztpmv,ZTPMV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n,
	const dcomplex *ap,
	dcomplex *x, const integer &incx
);

extern "C" integer BLASNAME(ztpsv,ZTPSV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const dcomplex *ap,
	dcomplex *x, const integer &incx
);

extern "C" integer BLASNAME(ztrmm,ZTRMM)(
	const char *side, const char *uplo, const char *transa, const char *diag,
	const uint &m, const uint &n,
	const dcomplex &alpha, const dcomplex *a, const uint &lda,
	dcomplex *b, const uint &ldb
);

extern "C" integer BLASNAME(ztrmv,ZTRMV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n,
	const dcomplex *a, const uint &lda,
	dcomplex *x, const integer &incx
);

extern "C" integer BLASNAME(ztrsm,ZTRSM)(
	const char *side, const char *uplo, const char *transa, const char *diag,
	const uint &m, const uint &n,
	const dcomplex &alpha, const dcomplex *a, const uint &lda,
	dcomplex *b, const uint &ldb
);

extern "C" integer BLASNAME(ztrsv,ZTRSV)(
	const char *uplo, const char *trans, const char *diag,
	const uint &n, const dcomplex *a, const uint &lda,
	dcomplex *x, const integer &incx
);

} // namespace FBLAS

#endif // FBLAS_HPP_INCLUDED
