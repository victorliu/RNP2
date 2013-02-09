#ifndef RNP_GENERALIZED_EIGENSYSTEMS_HPP_INCLUDED
#define RNP_GENERALIZED_EIGENSYSTEMS_HPP_INCLUDED

#include <RNP/bits/MatrixNorms.hpp>
#include <RNP/bits/Rotation.hpp>
#include <RNP/bits/Hessenberg.hpp>
#include <RNP/bits/Triangular.hpp>

namespace RNP{
namespace LA{

namespace NonsymmetricGeneralizedEigensystem{

// same for real and complex
template <typename T>
void Balance(
	const char *job, size_t n, T *a, size_t lda, T *b, size_t ldb,
	size_t *ilo, size_t *ihi, int *lscale, int *rscale
){
	typedef typename Traits<T>::real_type real_type;
	static const T zero(0);
	static const real_type two(2);
	static const real_type half(real_type(1)/two);
	static const real_type four(4);
	static const size_t max_iters = 5;
	
	size_t k, l;
	*ilo = k = 0;
	*ihi = l = n;

	if('N' == job[0] || n <= 1){
		for(size_t i = 0; i < n; ++i){
			lscale[i] = 0;
		}
		for(size_t i = 0; i < n; ++i){
			rscale[i] = 0;
		}
		return;
	}

	if('S' != job[0]){
		// Permute the matrices A and B to isolate the eigenvalues.

		// Find row with one nonzero in columns 1 through L
		bool found_row_perm;
		do{
			found_row_perm = false;
			size_t i = l;
			while(i --> 0){
				bool found = false;
				size_t j, jp1;
				for(j = 0; j < l; ++j){
					jp1 = j+1;
					if((T(0) != a[i+j*lda]) || (T(0) != b[i+j*ldb])){
						// We found a nonzero in column j, to the left of (i,l)
						found = true;
						break;
					}
				}
				if(!found){
					j = l-1;
				}else{
					bool found2 = false;
					for(j = jp1; j < l; ++j){
						if((T(0) != a[i+j*lda]) || (T(0) != b[i+j*ldb])){
							// We found another nonzero in column j, between the first nonzero and (i,l+1)
							found2 = true;
							break;
						}
					}
					if(found2){
						// We found more than 1 nonzero, so let's try the next row up
						continue;
					}else{
						j = jp1-1;
					}
				}
				size_t m = l-1;
				// Permute rows M and I
				lscale[m] = i;
				if(i != m){
					BLAS::Swap(n, &a[i+0*lda], lda, &a[m+0*lda], lda);
					BLAS::Swap(n, &b[i+0*ldb], ldb, &b[m+0*ldb], ldb);
				}
				// Permute columns M and J
				rscale[m] = j;
				if(j != m){
					BLAS::Swap(l, &a[0+j*lda], 1, &a[0+m*lda], 1);
					BLAS::Swap(l, &b[0+j*ldb], 1, &b[0+m*ldb], 1);
				}

				--l;
				if(l == 1){
					// We have completely deflated the matrix from bottom up
					rscale[0] = 0;
					lscale[0] = 0;
				}else{
					found_row_perm = true;
				}
				break;
			}
		}while(found_row_perm);

		// Find column with one nonzero in rows K through N
		bool found_col_perm;
		do{
			found_col_perm = false;
			for(size_t j = k; j < l; ++j){
				bool found = false;
				size_t i, ip1;
				for(i = k; i+1 < l; ++i){
					ip1 = i+1;
					if((T(0) != a[i+j*lda]) || (T(0) != b[i+j*ldb])){
						found = true;
						break;
					}
				}
				if(!found){
					i = l-1;
				}else{
					bool found2 = false;
					for(i = ip1; i < l; ++i){
						if((T(0) != a[i+j*lda]) || (T(0) != b[i+j*ldb])){
							found2 = true;
							break;
						}
					}
					if(found2){
						continue;
					}else{
						i = ip1-1;
					}
				}
				size_t m = k;
				// Permute rows M and I
				lscale[m] = i;
				if(i != m){
					BLAS::Swap(n-k, &a[i+k*lda], lda, &a[m+k*lda], lda);
					BLAS::Swap(n-k, &b[i+k*ldb], ldb, &b[m+k*ldb], ldb);
				}
				// Permute columns M and J
				rscale[m] = j;
				if(j != m){
					BLAS::Swap(l, &a[0+j*lda], 1, &a[0+m*lda], 1);
					BLAS::Swap(l, &b[0+j*ldb], 1, &b[0+m*ldb], 1);
				}

				++k;
				found_col_perm = true;
				break;
			}
		}while(found_col_perm);
	}

	// End of permutations
	*ilo = k;
	*ihi = l;

	for(size_t i = k; i < l; ++i){
		lscale[i] = 0;
	}
	for(size_t i = k; i < l; ++i){
		rscale[i] = 0;
	}

	if('P' == job[0] || k+1 == l){ return; }

	// Balance the submatrix in rows k..l.
	// Iterative loop for norm reduction
	
	const real_type sfmin1 = Traits<real_type>::min() / (two*Traits<real_type>::eps());
	const real_type sfmax1 = real_type(1) / sfmin1;
	const real_type sfmin2 = sfmin1 * two;
	const real_type sfmax2 = real_type(1) / sfmin2;

	for(size_t iter = 0; iter < max_iters; ++iter){
		int emax = 0, emin = 0;
		// Scale rows of A.^2 + B.^2 to have approximate row sum 1
		for(size_t i = k; i < l; ++i){
			real_type sum(0);
			for(size_t j = k; j < l; ++j){
				sum += Traits<T>::abs2(a[i+j*lda]) + Traits<T>::abs2(b[i+j*ldb]);
			}
			int ip = 0;
			while(sum > two){
				ip--;
				sum /= four;
			}
			while(sum < half){
				ip++;
				sum *= four;
			}
			BLAS::Rescale(n-k, ip, &a[i+k*lda], lda);
			BLAS::Rescale(n-k, ip, &b[i+k*ldb], ldb);
			lscale[i] += ip;
			if(ip > emax){ emax = ip; }
			if(ip < emin){ emin = ip; }
		}
		// Scale cols of A.^2 + B.^2 to have approximate col sum 1
		for(size_t j = k; j < l; ++j){
			real_type sum(0);
			for(size_t i = k; i < l; ++i){
				sum += Traits<T>::abs2(a[i+j*lda]) + Traits<T>::abs2(b[i+j*ldb]);
			}
			int ip = 0;
			while(sum > two){
				ip--;
				sum /= four;
			}
			while(sum < half){
				ip++;
				sum *= four;
			}
			BLAS::Rescale(l, ip, &a[0+j*lda], 1);
			BLAS::Rescale(l, ip, &b[0+j*ldb], 1);
			rscale[j] += ip;
			if(ip > emax){ emax = ip; }
			if(ip < emin){ emin = ip; }
		}
		// Stop if all norms are between 1/2 and 2
		if(emax <= emin+2){ break; }
	}
}

template <typename T>
void BalanceUndo(
	const char *job, const char *side,
	size_t n, size_t ilo, size_t ihi,
	int *lscale, int *rscale, size_t m, T *v, size_t ldv
){
	RNPAssert('N' == job[0] || 'P' == job[0] || 'S' == job[0] || 'B' == job[0]);
	RNPAssert('L' == side[0] || 'R' == side[0]);
	const bool left = ('L' == side[0]);

	if(0 == n || 0 == m || 'N' == job[0]){
		return;
	}

	if(ilo+1 != ihi){ // Backward balance
		if('S' == job[0] || 'B' == job[0]){
			if(left){
				for(size_t i = ilo; i < ihi; ++i){
					BLAS::Rescale(m, lscale[i], &v[i+0*ldv], ldv);
				}
			}else{
				for(size_t i = ilo; i < ihi; ++i){
					BLAS::Rescale(m, rscale[i], &v[i+0*ldv], ldv);
				}
			}
		}
	}
	
	if('P' == job[0] || 'B' == job[0]){
		// Backward permutation
		// For  I = ILO-1 step -1 until 1,
		//        IHI+1 step 1 until N do
		if(left){
			if(ilo > 0){
				size_t i = ilo;
				while(i --> 0){
					size_t k = lscale[i];
					if(k != i){
						BLAS::Swap(m, &v[i+0*ldv], ldv, &v[k+0*ldv], ldv);
					}
				}
			}
			if(ihi < n){
				for(size_t i = ihi; i < n; ++i){
					size_t k = lscale[i];
					if(k != i){
						BLAS::Swap(m, &v[i+0*ldv], ldv, &v[k+0*ldv], ldv);
					}
				}
			}
		}else{
			if(ilo > 0){
				size_t i = ilo;
				while(i --> 0){
					size_t k = rscale[i];
					if(k != i){
						BLAS::Swap(m, &v[i+0*ldv], ldv, &v[k+0*ldv], ldv);
					}
				}
			}
			if(ihi < n){
				for(size_t i = ihi; i < n; ++i){
					size_t k = rscale[i];
					if(k != i){
						BLAS::Swap(m, &v[i+0*ldv], ldv, &v[k+0*ldv], ldv);
					}
				}
			}
		}
	}
}

template <typename T>
void HessenbergQZSweep(
	bool wantSchur,
	size_t n, size_t ilast, size_t ilastm, size_t ifirst, size_t ifrstm,
	T ascale, T bscale, T atolr, std::complex<T> *eshift,
	std::complex<T> *h, size_t ldh, std::complex<T> *t, size_t ldt,
	std::complex<T> *q, size_t ldq, std::complex<T> *z, size_t ldz
){
	typedef std::complex<T> complex_type;
	typedef T real_type;
	typedef Traits<complex_type> complex_traits;
	typedef Traits<real_type> real_traits;
	
	complex_type shift;

	// This iteration only involves rows/columns IFIRST:ILAST.  We
	// assume IFIRST < ILAST, and that the diagonal of B is non-zero.
	if(!wantSchur){
		ifrstm = ifirst;
	}

	// Compute the Shift.
	// At this point, IFIRST < ILAST, and the diagonal elements of
	// T(IFIRST:ILAST,IFIRST,ILAST) are larger than BTOLR (in magnitude)
	if(NULL == eshift){
		// The Wilkinson shift (AEP p.512), i.e., the eigenvalue of
		// the bottom-right 2x2 block of A inv(B) which is nearest to
		// the bottom-right element.

		// We factor B as U*D, where U has unit diagonals, and
		// compute (A*inv(D))*inv(U).
		complex_type u12  = (bscale*t[ilast-1+ ilast   *ldt]) / (bscale*t[ilast  + ilast   *ldt]);
		complex_type ad11 = (ascale*h[ilast-1+(ilast-1)*ldh]) / (bscale*t[ilast-1+(ilast-1)*ldt]);
		complex_type ad21 = (ascale*h[ilast  +(ilast-1)*ldh]) / (bscale*t[ilast-1+(ilast-1)*ldt]);
		complex_type ad12 = (ascale*h[ilast-1+ ilast   *ldh]) / (bscale*t[ilast  + ilast   *ldt]);
		complex_type ad22 = (ascale*h[ilast  + ilast   *ldh]) / (bscale*t[ilast  + ilast   *ldt]);
		complex_type abi22 = ad22-u12*ad21;
		
		complex_type t1 = (ad11+abi22) / real_type(2);
		complex_type rtdisc = sqrt(t1*t1 + ad12*ad21 - ad11*ad22);
		if(complex_traits::real((t1-abi22)*std::conj(rtdisc)) <= real_type(0)){
			shift = t1 + rtdisc;
		}else{
			shift = t1 - rtdisc;
		}
	}else{
		// Exceptional shift.  Chosen for no particularly good reason.
		*eshift += std::conj((ascale*h[ilast-1+ilast*ldh]) / (bscale*t[ilast-1+(ilast-1)*ldt]));
		shift = *eshift;
	}

	// Now check for two consecutive small subdiagonals.
	bool found_two_small = false;
	complex_type ctemp;
	size_t istart;
	for(size_t j = ilast-1; j > ifirst; --j){
		istart = j;
		ctemp = ascale*h[j+j*ldh] - shift*(bscale*t[j+j*ldt]);
		real_type temp = complex_traits::norm1(ctemp);
		real_type temp2 = ascale * complex_traits::norm1(h[j+1+j*ldh]);
		real_type tempr = (temp > temp2 ? temp : temp2);
		if(tempr < real_type(1) && tempr != real_type(0)){
			temp /= tempr;
			temp2 /= tempr;
		}
		if(complex_traits::norm1(h[j+(j-1)*ldh]) * temp2 <= temp * atolr){
			found_two_small = true;
			break;
		}
	}
	if(!found_two_small){
		istart = ifirst;
		ctemp = ascale*h[ifirst+ifirst*ldh] - shift*(bscale*t[ifirst+ifirst*ldt]);
	}

	// Do an implicit-shift QZ sweep.
	real_type c;
	complex_type s;

	// Initial Q
	{
		std::complex<double> ctemp3;
		Rotation::Generate(ctemp, ascale * h[istart+1+istart*ldh], &c, &s, &ctemp3);
	}

	// Sweep
	for(size_t j = istart; j < ilast; ++j){

		if(j > istart){
			complex_type ctemp = h[j+(j-1)*ldh];
			Rotation::Generate(ctemp, h[j+1+(j-1)*ldh], &c, &s, &h[j+(j-1)*ldh]);
			h[j+1+(j-1)*ldh] = real_type(0);
		}

		for(int jc = j; jc < ilastm; ++jc){
			complex_type ctemp = c*h[j+jc*ldh] + s*h[j+1+jc*ldh];
			h[j+1+jc*ldh] = -std::conj(s)*h[j+jc*ldh] + c*h[j+1+jc*ldh];
			h[j+jc*ldh] = ctemp;
			ctemp = c*t[j+jc*ldt] + s*t[j+1+jc*ldt];
			t[j+1+jc*ldt] = -std::conj(s)*t[j+jc*ldt] + c*t[j+1+jc*ldt];
			t[j+jc*ldt] = ctemp;
		}
		if(NULL != q){
			for(size_t jr = 0; jr < n; ++jr){
				complex_type ctemp = c*q[jr+j*ldq] + std::conj(s)*q[jr+(j+1)*ldq];
				q[jr+(j+1)*ldq] = -s*q[jr+j*ldq] + c*q[jr+(j+1)*ldq];
				q[jr+j*ldq] = ctemp;
			}
		}

		{
			complex_type ctemp = t[j+1+(j+1)*ldt];
			Rotation::Generate(ctemp, t[j+1+j*ldt], &c, &s, &t[j+1+(j+1)*ldt]);
			t[j+1+j*ldt] = real_type(0);
		}

		size_t jrlim = (j+2 < ilast ? j+2 : ilast);
		for(size_t jr = ifrstm; jr <= jrlim; ++jr){
			complex_type ctemp = c*h[jr+(j+1)*ldh] + s*h[jr+j*ldh];
			h[jr+j*ldh] = -std::conj(s)*h[jr+(j+1)*ldh] + c*h[jr+j*ldh];
			h[jr+(j+1)*ldh] = ctemp;
		}
		for(size_t jr = ifrstm; jr <= j; ++jr){
			complex_type ctemp = c*t[jr+(j+1)*ldt] + s*t[jr+j*ldt];
			t[jr+j*ldt] = -std::conj(s)*t[jr+(j+1)*ldt] + c*t[jr+j*ldt];
			t[jr+(j+1)*ldt] = ctemp;
		}
		if(NULL != z){
			for(size_t jr = 0; jr < n; ++jr){
				complex_type ctemp = c*z[jr+(j+1)*ldz] + s*z[jr+j*ldz];
				z[jr+j*ldz] = -complex_traits::conj(s) * z[jr+(j+1)*ldz] + c*z[jr+j*ldz];
				z[jr+(j+1)*ldz] = ctemp;
			}
		}
	}
}

// for complex only at the moment
template <typename T>
int HessenbergQZ(
	bool wantSchur,
	size_t n, size_t ilo, size_t ihi,
	std::complex<T> *h, size_t ldh, std::complex<T> *t, size_t ldt,
	std::complex<T> *alpha, std::complex<T> *beta,
	std::complex<T> *q, size_t ldq, std::complex<T> *z, size_t ldz
){
	typedef std::complex<T> complex_type;
	typedef T real_type;
	typedef Traits<complex_type> complex_traits;
	typedef Traits<real_type> real_traits;
	
	static const complex_type zero(real_type(0));
	
    if(0 == n){
		return 0;
    }

	// Machine Constants
    const size_t in = ihi - ilo;
    const real_type safmin(real_traits::min());
    const real_type ulp(real_type(2) * real_traits::eps());
    const real_type anorm(Hessenberg::Norm("F", in, &h[ilo+ilo*ldh], ldh));
    const real_type bnorm(Hessenberg::Norm("F", in, &t[ilo+ilo*ldt], ldt));
    const real_type atolr = (safmin > ulp*anorm ? safmin : ulp*anorm);
    const real_type btolr = (safmin > ulp*bnorm ? safmin : ulp*bnorm);
    const real_type ascale(real_type(1) / (safmin > anorm ? safmin : anorm));
    const real_type bscale(real_type(1) / (safmin > bnorm ? safmin : bnorm));
    
	
	// Set Eigenvalues ihi:n

	for(size_t j = ihi; j < n; ++j){
		real_type absb = complex_traits::abs(t[j+j*ldt]);
		if(absb > safmin){
			complex_type signbc = complex_traits::conj(t[j+j*ldt] / absb);
			t[j+j*ldt] = absb;
			if(wantSchur){
				BLAS::Scale(j  , signbc, &t[0+j*ldt], 1);
				BLAS::Scale(j+1, signbc, &h[0+j*ldh], 1);
			}else{
				h[j+j*ldh] *= signbc;
			}
			if(NULL != z){
				BLAS::Scale(n, signbc, &z[0+j*ldz], 1);
			}
		}else{
			t[j+j*ldt] = zero;
		}
		alpha[j] = h[j+j*ldh];
		beta[j] = t[j+j*ldt];
    }
	
    if(ihi >= ilo){

		// MAIN QZ ITERATION LOOP
		//
		// Initialize dynamic indices
		//
		// Eigenvalues ILAST+1:N have been found.
		//    Column operations modify rows IFRSTM:whatever
		//    Row operations modify columns whatever:ILASTM
		//
		// If only eigenvalues are being computed, then
		//    IFRSTM is the row of the last splitting row above row ILAST;
		//    this is always at least ILO.
		// IITER counts iterations since the last eigenvalue was found,
		//    to tell when to use an extraordinary shift.
		// MAXIT is the maximum number of QZ sweeps allowed.
		
		size_t ilast = ihi-1;
		size_t ilastm = (wantSchur ? n : ihi);
		size_t ifrstm = (wantSchur ? 0 : ilo);
		
		int iiter = 0;
		complex_type eshift(real_type(0));
		const size_t maxit = (ihi - ilo) * 30;
		bool converged = true;
		

		for(size_t jiter = 0; jiter < maxit; ++jiter){
			// Split the matrix if possible.

			bool is_tzero = false;
			bool is_hzero = false;
			// Two tests:
			//    1: H(j,j-1)=0  or  j=ILO
			//    2: T(j,j)=0

			// Special case: j=ILAST

			if(ilast == ilo){
				is_hzero = true;
			}else{
				if(complex_traits::norm1(h[ilast+(ilast-1)*ldh]) <= atolr){
					h[ilast+(ilast-1)*ldh] = zero;
					is_hzero = true;
				}
			}

			if(!is_hzero){
				if(complex_traits::abs(t[ilast+ilast*ldt]) <= btolr){
					t[ilast+ilast*ldt] = zero;
					is_tzero = true;
				}
			}

			// General case: j<ILAST
			if(!is_tzero && !is_hzero){
				size_t j = ilast;
				while(j --> ilo){
					bool ilazro = false;
					bool ilazr2 = false;
					// Test 1: for H(j,j-1)=0 or j=ILO
					if(j == ilo){
						ilazro = true;
					}else{
						if(complex_traits::norm1(h[j+(j-1)*ldh]) <= atolr){
							h[j+(j-1) * ldh] = zero;
							ilazro = true;
						}else{
							ilazro = false;
						}
					}

					// Test 2: for T(j,j)=0
					if(complex_traits::abs(t[j+j*ldt]) < btolr){
						t[j+j*ldt] = zero;

						// Test 1a: Check for 2 consecutive small subdiagonals in A
						ilazr2 = false;
						if(!ilazro){
							if(complex_traits::norm1(h[j+(j-1)*ldh])*( ascale*complex_traits::norm1(h[j+1+j*ldh])) <= complex_traits::norm1(h[j+j*ldh])*( ascale*atolr ) ){
								ilazr2 = true;
							}
						}

						// If both tests pass (1 & 2), i.e., the leading diagonal
						// element of B in the block is zero, split a 1x1 block off
						// at the top. (I.e., at the J-th row/column) The leading
						// diagonal element of the remainder can also be zero, so
						// this may have to be done repeatedly.

						if(ilazro || ilazr2){
							bool bnz = false;
							for(int jch = j; jch < ilast; ++jch){
								real_type c;
								complex_type s;
								complex_type ctemp = h[jch+jch*ldh];
								Rotation::Generate(ctemp, h[jch+1+jch*ldh], &c, &s, &h[jch+jch*ldh]);
								h[jch+1+jch*ldh] = zero;
								Rotation::Apply(ilastm-jch-1, &h[jch+(jch+1)*ldh], ldh, &h[jch+1+(jch+1)*ldh], ldh, c, s);
								Rotation::Apply(ilastm-jch-1, &t[jch+(jch+1)*ldt], ldt, &t[jch+1+(jch+1)*ldt], ldt, c, s);
								if(NULL != q){
									Rotation::Apply(n, &q[0+jch*ldq], 1, &q[0+(jch+1)*ldq], 1, c, complex_traits::conj(s));
								}
								if(ilazr2){
									h[jch+(jch-1)*ldh] *= c;
								}
								ilazr2 = false;
								if(complex_traits::norm1(t[jch+1+(jch+1)*ldt]) >= btolr){
									if(jch+1 >= ilast){
										is_hzero = true;
										bnz = true;
										break;
									}else{
										++iiter;
										HessenbergQZSweep(
											wantSchur, n, ilast, ilastm, jch+1, ifrstm, ascale, bscale, atolr,
											(iiter % 10 == 0 ? &eshift : NULL), h, ldh, t, ldt, q, ldq, z, ldz
										);
										bnz = true;
										break;
									}
								}
								t[jch+1+(jch+1)*ldt] = zero;
							}
							if(bnz){ break; }
							is_tzero = true;
							break;
						}else{
							// Only test 2 passed -- chase the zero to T(ILAST,ILAST)
							// Then process as in the case T(ILAST,ILAST)=0
							for(size_t jch = j; jch < ilast; ++jch){
								real_type c;
								complex_type s;
								complex_type ctemp = t[jch+(jch+1)*ldt];
								Rotation::Generate(ctemp, t[jch+1+(jch+1)*ldt], &c, &s, &t[jch+(jch+1)*ldt]);
								t[jch+1+(jch+1)*ldt] = zero;
								if(jch+2 < ilastm){
									Rotation::Apply(ilastm-jch-2, &t[jch+(jch+2)*ldt], ldt, &t[jch+1+(jch+2)*ldt], ldt, c, s);
								}
								Rotation::Apply(ilastm-jch+1, &h[jch+(jch-1)*ldh], ldh, &h[jch+1+(jch-1)*ldh], ldh, c, s);
								if(NULL != q){
									Rotation::Apply(n, &q[0+jch*ldq], 1, &q[0+(jch+1)*ldq], 1, c, complex_traits::conj(s));
								}
								ctemp = h[jch+1+jch*ldh];
								Rotation::Generate(ctemp, h[jch+1+(jch-1)*ldh], &c, &s, &h[jch+1+jch*ldh]);
								h[jch+1+(jch-1)*ldh] = zero;
								Rotation::Apply(jch-ifrstm+1, &h[ifrstm+jch*ldh], 1, &h[ifrstm+(jch-1)*ldh], 1, c, s);
								Rotation::Apply(jch-ifrstm  , &t[ifrstm+jch*ldt], 1, &t[ifrstm+(jch-1)*ldt], 1, c, s);
								if(NULL != z){
									Rotation::Apply(n, &z[0+jch*ldz], 1, &z[0+(jch-1)*ldz], 1, c, s);
								}
							}
							is_tzero = true;
							break;
						}
					}else if(ilazro){
						// Only test 1 passed -- work on J:ILAST
						++iiter;
						HessenbergQZSweep(
							wantSchur, n, ilast, ilastm, j, ifrstm, ascale, bscale, atolr,
							(iiter % 10 == 0 ? &eshift : NULL), h, ldh, t, ldt, q, ldq, z, ldz
						);
						break;
					}
					// Neither test passed -- try next J
				} // We should never exit the loop naturally
			}
			
			// T(ILAST,ILAST)=0 -- clear H(ILAST,ILAST-1) to split off a 1x1 block.
			if(is_tzero){
				real_type c;
				complex_type s;
				complex_type ctemp = h[ilast+ilast*ldh];
				Rotation::Generate(ctemp, h[ilast+(ilast-1)*ldh], &c, &s, &h[ilast+ilast*ldh]);
				h[ilast+(ilast-1)*ldh] = zero;
				Rotation::Apply(ilast-ifrstm, &h[ifrstm+ilast*ldh], 1, &h[ifrstm+(ilast-1)*ldh], 1, c, s);
				Rotation::Apply(ilast-ifrstm, &t[ifrstm+ilast*ldt], 1, &t[ifrstm+(ilast-1)*ldt], 1, c, s);

				if(NULL != z){
					Rotation::Apply(n, &z[0+ilast*ldz], 1, &z[0+(ilast-1)*ldz], 1, c, s);
				}
				is_hzero = true;
			}
			
			// H(ILAST,ILAST-1)=0 -- Standardize B, set ALPHA and BETA
			if(is_hzero){
				real_type absb = complex_traits::abs(t[ilast+ ilast*ldt]);
				if(absb > safmin){
					complex_type signbc = std::conj(t[ilast+ilast*ldt]/absb);

					t[ilast+ilast*ldt] = absb;
					if(wantSchur){
						BLAS::Scale(ilast-ifrstm  , signbc, &t[ifrstm+ilast*ldt], 1);
						BLAS::Scale(ilast-ifrstm+1, signbc, &h[ifrstm+ilast*ldh], 1);
					}else{
						h[ilast+ilast*ldh] *= signbc;
					}
					if(NULL != z){
						BLAS::Scale(n, signbc, &z[0+ilast*ldz], 1);
					}
				}else{
					t[ilast+ilast*ldt] = zero;
				}
				alpha[ilast] = h[ilast+ilast*ldh];
				beta [ilast] = t[ilast+ilast*ldt];

				// Go to next block -- exit if finished.
				if(ilast <= ilo){
					converged = true;
					break;
				}
				--ilast;

				// Reset counters
				iiter = 0;
				eshift = zero;
				if(!wantSchur){
					ilastm = ilast+1;
					if(ifrstm > ilast){
						ifrstm = ilo;
					}
				}
			}
		}

		
		// Drop-through = non-convergence
		if(!converged){
			return ilast;
		}
	}
	// Successful completion of all QZ steps

	// Set Eigenvalues 0:ilo
    for(int j = 0; j < ilo; ++j){
		real_type absb = complex_traits::abs(t[j+j*ldt]);
		if(absb > safmin){
			complex_type signbc = complex_traits::conj(t[j+j*ldt] / absb);
			t[j+j*ldt] = absb;
			if(wantSchur){
				BLAS::Scale(j  , signbc, &t[0+j*ldt], 1);
				BLAS::Scale(j+1, signbc, &h[0+j*ldh], 1);
			}else{
				h[j+j*ldh] *= signbc;
			}
			if(NULL != z){
				BLAS::Scale(n, signbc, &z[0+j*ldz], 1);
			}
		}else{
			t[j+j*ldt] = zero;
		}
		alpha[j] = h[j+j*ldh];
		beta [j] = t[j+j*ldt];
	}

	// Normal Termination
	// Exit (other than argument error)
    return 0;
}

} // namespace NonsymmetricGeneralizedEigensystem

// Computes for a pair of N-by-N complex nonsymmetric matrices (A,B),
// the generalized eigenvalues, and optionally, the left and/or right
// generalized eigenvectors.

// A generalized eigenvalue for a pair of matrices (A,B) is a scalar
// lambda or a ratio alpha/beta = lambda, such that A - lambda*B is
// singular. It is usually represented as the pair (alpha,beta), as
// there is a reasonable interpretation for beta=0, and even for both
// being zero.

// The right generalized eigenvector v(j) corresponding to the
// generalized eigenvalue lambda(j) of (A,B) satisfies
//              A * v(j) = lambda(j) * B * v(j).
// The left generalized eigenvector u(j) corresponding to the
// generalized eigenvalues lambda(j) of (A,B) satisfies
//              u(j)^H * A = lambda(j) * u(j)^H * B
// where u(j)^H is the conjugate-transpose of u(j).

// Arguments
// =========

// N       The order of the matrices A, B, VL, and VR.  N >= 0.
// A       (input/output) COMPLEX*16 array, dimension (LDA, N)
//         On entry, the matrix A in the pair (A,B).
//         On exit, A has been overwritten.
// LDA     The leading dimension of A.  LDA >= max(1,N).
// B       (input/output) COMPLEX*16 array, dimension (LDB, N)
//         On entry, the matrix B in the pair (A,B).
//         On exit, B has been overwritten.
// LDB     The leading dimension of B.  LDB >= max(1,N).

// ALPHA   (output) COMPLEX*16 array, dimension (N)
// BETA    (output) COMPLEX*16 array, dimension (N)
//         On exit, ALPHA(j)/BETA(j), j=1,...,N, will be the
//         generalized eigenvalues.

//         Note: the quotients ALPHA(j)/BETA(j) may easily over- or
//         underflow, and BETA(j) may even be zero.  Thus, the user
//         should avoid naively computing the ratio alpha/beta.
//         However, ALPHA will be always less than and usually
//         comparable with norm(A) in magnitude, and BETA always less
//         than and usually comparable with norm(B).

// VL      (output) COMPLEX*16 array, dimension (LDVL,N)
//         If VL != NULL, the left generalized eigenvectors u(j) are
//         stored one after another in the columns of VL, in the same
//         order as their eigenvalues.
//         Each eigenvector is scaled so the largest component has
//         abs(real part) + abs(imag. part) = 1.
// LDVL    The leading dimension of the matrix VL. LDVL >= 1, and
//         if VL != NULL, LDVL >= N.

// VR      (output) COMPLEX*16 array, dimension (LDVR,N)
//         If VR != NULL, the right generalized eigenvectors v(j) are
//         stored one after another in the columns of VR, in the same
//         order as their eigenvalues.
//         Each eigenvector is scaled so the largest component has
//         abs(real part) + abs(imag. part) = 1.
// LDVR    The leading dimension of the matrix VR. LDVR >= 1, and
//         if VR != NULL, LDVR >= N.

// WORK    (workspace/output) COMPLEX*16 array, dimension (MAX(1,2*N))
// iwork   (workspace/output) int array, dimension (2*N)

// return: = 0:  successful exit
//         < 0:  if INFO = -i, the i-th argument had an illegal value.
//         =1,...,N:
//               The QZ iteration failed.  No eigenvectors have been
//               calculated, but ALPHA(j) and BETA(j) should be
//               correct for j=INFO+1,...,N.
//         =N+1: other QZ iteration failure
int ComplexGeneralizedEigensystem(size_t n, 
	std::complex<double> *a, size_t lda, std::complex<double> *b, size_t ldb, 
	std::complex<double> *alpha, std::complex<double> *beta,
	std::complex<double> *vl, size_t ldvl, std::complex<double> *vr, size_t ldvr,
	size_t *lwork, std::complex<double> *work, int *iwork
){
	typedef std::complex<double> complex_type;
	typedef double real_type;
	
	static const complex_type one(real_type(1));
	static const complex_type zero(real_type(0));
	static const char *balancejob = "B";
	
	using namespace std;
	
	// Workspace layout:
	//   [  tau  | work ] QR::Factor
	//       n     var
	//   [  tau  | work ] QR::MultQ
	//       n     var
	//   [  tau  | work ] QR::GenerateQ
	//       n     var
	//   [ ---          ] ReduceGeneralized_unblocked
	//   [ ---          ] HessenbergQZ
	//   [ rwork | work ] GeneralizedEigenvectors
	//     2n real  2n
	// Therefore, n+max(2n,factor,multQ,genQ) is recommended,
	// and 3n is the minimum when eigenvectors are wanted.
	// If only eigenvalues are wanted, then n is the minimum.
	
	if(0 == n){
		return 0;
	}
	
	if(0 == *lwork){
		size_t sublwork = 0;
		if(NULL != vl || NULL != vr){
			*lwork = 2*n;
		}
		QR::Factor(n, n, b, ldb, work, &sublwork, work);
		if(sublwork > *lwork){ *lwork = sublwork; }
		sublwork = 0;
		QR::MultQ("L","C", n, n, n, b, ldb, work, a, lda, &sublwork, work);
		if(sublwork > *lwork){ *lwork = sublwork; }
		if(NULL != vl){
			sublwork = 0;
			QR::GenerateQ(n, n, n, vl, ldvl, work, &sublwork, work);
			if(sublwork > *lwork){ *lwork = sublwork; }
		}
		*lwork += n;
		return 0;
	}

	const double eps = std::numeric_limits<double>::epsilon() * 2;
	const double smlnum = sqrt(std::numeric_limits<double>::min()) / eps;
	const double bignum = 1. / smlnum;

	// Scale A if max element outside range [SMLNUM,BIGNUM]
	real_type anrm = MatrixNorm("M", n, n, a, lda);
	real_type anrmto;
	bool scaledA = false;
	if(anrm > 0. && anrm < smlnum){
		anrmto = smlnum;
		scaledA = true;
	}else if(anrm > bignum){
		anrmto = bignum;
		scaledA = true;
	}
	if(scaledA){
		RNP::BLAS::Rescale("G", 0, 0, anrm, anrmto, n, n, a, lda);
	}

	// Scale B if max element outside range [SMLNUM,BIGNUM]
	real_type bnrm = MatrixNorm("M", n, n, b, ldb);
	real_type bnrmto;
	bool scaledB = false;
	if(bnrm > 0. && bnrm < smlnum){
		bnrmto = smlnum;
		scaledB = true;
	}else if(bnrm > bignum){
		bnrmto = bignum;
		scaledB = true;
	}
	if(scaledB){
		BLAS::Rescale("G", 0, 0, bnrm, bnrmto, n, n, b, ldb);
	}

	// Permute the matrices A, B to isolate eigenvalues if possible
	size_t ilo, ihi;
	NonsymmetricGeneralizedEigensystem::Balance(balancejob, n, a, lda, b, ldb, &ilo, &ihi, iwork, &iwork[n]);

	const size_t irows = ihi - ilo;
	const size_t icols = (NULL != vl || NULL != vr) ? n - ilo : irows;

	// Triangularize B, and apply the transformations to A
	complex_type *tau = work;
	complex_type *work2 = tau + n;
	size_t lwork2 = *lwork - n;
	QR::Factor(irows, icols, &b[ilo+ilo*ldb], ldb, tau, &lwork2, work2);
	QR::MultQ("L","C", irows, icols, irows, &b[ilo+ilo*ldb], ldb, tau, &a[ilo+ilo*lda], lda, &lwork2, work2);

	// If we want left eigenvectors, then initialize to the Q factor
	if(NULL != vl){
		BLAS::Set(n, n, zero, one, vl, ldvl);
		if(irows > 1){
			Triangular::Copy("L", irows-1, irows-1, &b[ilo+1+ilo*ldb], ldb, &vl[ilo+1+ilo*ldvl], ldvl);
		}
		QR::GenerateQ(irows, irows, irows, &vl[ilo+ilo*ldvl], ldvl, tau, &lwork2, work2);
	}

	// The initial right eigenvectors are just the identity matrix
	if(NULL != vr){
		BLAS::Set(n, n, zero, one, vr, ldvr);
	}

	// Reduce to generalized Hessenberg form
	Hessenberg::ReduceGeneralized_unblocked(n, ilo, ihi, a, lda, b, ldb, vl, ldvl, vr, ldvr);

	int ierr = NonsymmetricGeneralizedEigensystem::HessenbergQZ(
		(NULL != vl || NULL != vr), n, ilo, ihi, a, lda, b, ldb, alpha, beta,
		vl, ldvl, vr, ldvr
	);
	if(ierr != 0){
		if(0 < ierr && ierr <= (int)n){
			return ierr;
		}else if(ierr > (int)n && ierr <= 2*(int)n){
			return ierr - n;
		}else{
			return n+1;
		}
		return ierr;
	}

	if(NULL != vl || NULL != vr){
		Triangular::GeneralizedEigenvectors(
			"B", NULL, n, a, lda, b, ldb, vl, ldvl, vr, ldvr, work2, reinterpret_cast<real_type*>(tau)
		);
		if(NULL != vl){
			NonsymmetricGeneralizedEigensystem::BalanceUndo(balancejob, "L", n, ilo, ihi, iwork, &iwork[n], n, vl, ldvl);
		}
		if(NULL != vr){
			NonsymmetricGeneralizedEigensystem::BalanceUndo(balancejob, "R", n, ilo, ihi, iwork, &iwork[n], n, vr, ldvr);
		}
	}

	// Undo scaling if necessary
	if(scaledA){
		RNP::BLAS::Rescale("G", 0, 0, anrmto, anrm, n, 1, alpha, n);
	}
	if(scaledB){
		RNP::BLAS::Rescale("G", 0, 0, bnrmto, bnrm, n, 1, beta, n);
	}

	return 0;
}


} // namespace LA
} // namespace RNP

#endif // RNP_GENERALIZED_EIGENSYSTEMS_HPP_INCLUDED
