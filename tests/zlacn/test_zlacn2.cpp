#include <RNP/BLAS.hpp>
#include <RNP/Random.hpp>
#include <RNP/LA.hpp>
#include <complex>
#include <iostream>
using namespace RNP;

typedef int integer;
typedef std::complex<double> complex_t;

extern "C" void zlacn2_(
	const integer &n, complex_t *v, complex_t *x, double *est,
	integer *kase, integer *isave
);
extern "C" void dlacn2_(
	const integer &n, double *v, double *x, integer *isgn,
	double *est, integer *kase, integer *isave
);
extern "C" void dlacn1_(
	const integer &n, const integer &t, double *v, double *x, const integer &ldx,
	double *xold, const integer &ldxold, double *work, double *h,
	integer *ind, integer *indh, double *est, integer *kase,
	integer *iseed, integer *info
);
extern "C" void zlacn1_(
	const integer &n, const integer &t, complex_t *v, complex_t *x, const integer &ldx,
	complex_t *xold, const integer &ldxold, double *h,
	integer *ind, integer *indh, double *est, integer *kase,
	integer *iseed, integer *info
);

void Aop(const char *trans, size_t n, double *x, void *data){
	double *A = (double*)data;
	double *temp = new double[n];
	BLAS::Copy(n, x, 1, temp, 1);
	BLAS::MultMV(trans, n, n, 1., A, n, temp, 1, 0., x, 1);
	delete [] temp;
}
void Aop2(const char *trans, size_t n, size_t t, double *x, size_t ldx, void *data){
	double *A = (double*)data;
	double *temp = new double[n*t];
	BLAS::Copy(n, t, (const double*)x, ldx, temp, n);
	BLAS::MultMM(trans, "N", n, t, n, 1., A, n, temp, n, 0., x, ldx);
	delete [] temp;
}
void Aopz(const char *trans, size_t n, complex_t *x, void *data){
	complex_t *A = (complex_t*)data;
	complex_t *temp = new complex_t[n];
	BLAS::Copy(n, x, 1, temp, 1);
	BLAS::MultMV(trans, n, n, 1., A, n, temp, 1, 0., x, 1);
	delete [] temp;
}
void Aop2z(const char *trans, size_t n, size_t t, complex_t *x, size_t ldx, void *data){
	complex_t *A = (complex_t*)data;
	complex_t *temp = new complex_t[n*t];
	BLAS::Copy(n, t, (const complex_t*)x, ldx, temp, n);
	BLAS::MultMM(trans, "N", n, t, n, 1., A, n, temp, n, 0., x, ldx);
	delete [] temp;
}

const size_t nb = 4;

void test_dcn(size_t n, double *A){
	double est = 0;
	// Use dlacn2
	{
		size_t iter = 0;
		integer kase = 0;
		integer *isave = new integer[3];
		integer *isgn = new integer[n];
		double *v = new double[n];
		double *x = new double[n];
		double *temp = new double[n];
		dlacn2_(n, v, x, isgn, &est, &kase, isave);
		do{
			BLAS::Copy(n, x, 1, temp, 1);
			if(1 == kase){
				BLAS::MultMV("N", n, n, 1., A, n, temp, 1, 0., x, 1);
			}else{
				BLAS::MultMV("C", n, n, 1., A, n, temp, 1, 0., x, 1);
			}
			dlacn2_(n, v, x, isgn, &est, &kase, isave);
		}while(0 != kase && iter++ < 99);
		delete [] temp;
		delete [] isave;
		delete [] x;
		delete [] v;
	}
	std::cout << "dlacn2 returned " << est << std::endl;
	// Use ours
	est = 0;
	{
		double *work = new double[2*n];
		est = LA::MatrixNorm1Estimate_unblocked(n, &Aop, A, work);
		delete [] work;
	}
	std::cout << "we returned " << est << std::endl;
	// Use dlacn1
	{
		size_t t = nb;
		double *x = new double[n*t];
		double *xold = new double[n*t];
		double *work = new double[t];
		double *h = new double[n];
		double *v = new double[n];
		const size_t ldx = n, ldxold = n;
		integer *ind = new integer[n];
		integer *indh = new integer[n];
		integer iseed[4] = {234,432,5225,3223};
		integer info, kase;
		dlacn1_(n, t, v, x, ldx, xold, ldxold, work, h, ind, indh, &est, &kase, iseed, &info);
		do{
			if(1 == kase){
				Aop2("N", n, t, x, ldx, A);
			}else{
				Aop2("C", n, t, x, ldx, A);
			}
			dlacn1_(n, t, v, x, ldx, xold, ldxold, work, h, ind, indh, &est, &kase, iseed, &info);
		}while(0 != kase);
		delete [] v;
		delete [] indh;
		delete [] ind;
		delete [] h;
		delete [] work;
		delete [] xold;
		delete [] x;
	}
	std::cout << "dlacn1 returned " << est << std::endl;
	// Use blocked
	est = 0;
	{
		size_t t = nb;
		double *x = new double[n*t];
		double *xold = new double[n*t];
		double *h = new double[n];
		size_t *ind = new size_t[n];
		size_t *indh = new size_t[n];
		LA::MatrixNorm1Estimate(n, t, &Aop2, A, &est, x, n, xold, n, h, ind, indh);
		delete [] indh;
		delete [] ind;
		delete [] xold;
		delete [] x;
	}
	std::cout << "we2 returned " << est << std::endl;
}

void test_zcn(size_t n, complex_t *A){
	double est = 0;
	// Use dlacn2
	{
		size_t iter = 0;
		integer kase = 0;
		integer *isave = new integer[3];
		complex_t *v = new complex_t[n];
		complex_t *x = new complex_t[n];
		complex_t *temp = new complex_t[n];
		zlacn2_(n, v, x, &est, &kase, isave);
		do{
			BLAS::Copy(n, x, 1, temp, 1);
			if(1 == kase){
				BLAS::MultMV("N", n, n, 1., A, n, temp, 1, 0., x, 1);
			}else{
				BLAS::MultMV("C", n, n, 1., A, n, temp, 1, 0., x, 1);
			}
			zlacn2_(n, v, x, &est, &kase, isave);
		}while(0 != kase && iter++ < 99);
		delete [] temp;
		delete [] isave;
		delete [] x;
		delete [] v;
	}
	std::cout << "zlacn2 returned " << est << std::endl;
	// Use ours
	est = 0;
	{
		complex_t *work = new complex_t[2*n];
		est = LA::MatrixNorm1Estimate_unblocked(n, &Aopz, A, work);
		delete [] work;
	}
	std::cout << "we returned " << est << std::endl;
	// Use zlacn1
	{
		size_t t = nb;
		complex_t *x = new complex_t[n*t];
		complex_t *xold = new complex_t[n*t];
		double *h = new double[n];
		complex_t *v = new complex_t[n];
		const size_t ldx = n, ldxold = n;
		integer *ind = new integer[n];
		integer *indh = new integer[n];
		integer iseed[4] = {234,432,5225,3223};
		integer info, kase;
		zlacn1_(n, t, v, x, ldx, xold, ldxold, h, ind, indh, &est, &kase, iseed, &info);
		do{
			if(1 == kase){
				Aop2z("N", n, t, x, ldx, A);
			}else{
				Aop2z("C", n, t, x, ldx, A);
			}
			zlacn1_(n, t, v, x, ldx, xold, ldxold, h, ind, indh, &est, &kase, iseed, &info);
		}while(0 != kase);
		delete [] v;
		delete [] indh;
		delete [] ind;
		delete [] h;
		delete [] xold;
		delete [] x;
	}
	std::cout << "zlacn1 returned " << est << std::endl;
	// Use blocked
	est = 0;
	{
		size_t t = nb;
		complex_t *x = new complex_t[n*t];
		//complex_t *xold = new complex_t[n*t];
		complex_t *xold = NULL;
		double *h = new double[n];
		size_t *ind = new size_t[n];
		size_t *indh = new size_t[n];
		LA::MatrixNorm1Estimate(n, t, &Aop2z, A, &est, x, n, xold, n, h, ind, indh);
		delete [] indh;
		delete [] ind;
		//delete [] xold;
		delete [] x;
	}
	std::cout << "we2 returned " << est << std::endl;
}

int main(){
	const size_t ntry = 10;
	for(size_t itry = 0; itry < ntry; ++itry){
		const size_t n = 100;
		double *A = new double[n*n];
		for(size_t j = 0; j < n; ++j){
			Random::GenerateVector(Random::Distribution::Uniform_11, n, &A[0+j*n]);
		}
		test_dcn(n, A);
		delete [] A;
	}
	for(size_t itry = 0; itry < ntry; ++itry){
		const size_t n = 100;
		complex_t *A = new complex_t[n*n];
		for(size_t j = 0; j < n; ++j){
			Random::GenerateVector(Random::Distribution::Uniform_11, n, &A[0+j*n]);
		}
		test_zcn(n, A);
		delete [] A;
	}
}
