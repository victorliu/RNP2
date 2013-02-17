#ifndef RNP_HESSENBERG_QZ_HPP_INCLUDED
#define RNP_HESSENBERG_QZ_HPP_INCLUDED

#include <RNP/LA/MatrixNorms.hpp>
#include <RNP/LA/Rotation.hpp>
#include <RNP/LA/Hessenberg.hpp>
#include <RNP/LA/Triangular.hpp>

namespace RNP{
namespace LA{
namespace HessenbergQZ{

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

		for(size_t jc = j; jc < ilastm; ++jc){
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
int GeneralizedSchurReduce(
	bool wantSchur,
	size_t n, size_t ilo, size_t ihi,
	std::complex<T> *h, size_t ldh, std::complex<T> *t, size_t ldt,
	std::complex<T> *alpha, T *beta,
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
		beta[j] = t[j+j*ldt].real();
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
		bool converged = false;
		

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
							for(size_t jch = j; jch < ilast; ++jch){
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
				real_type absb = complex_traits::abs(t[ilast+ilast*ldt]);
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
				beta [ilast] = t[ilast+ilast*ldt].real();

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
    for(size_t j = 0; j < ilo; ++j){
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
		beta [j] = t[j+j*ldt].real();
	}

	// Normal Termination
	// Exit (other than argument error)
    return 0;
}

} // namespace HessenbergQZ
} // namespace LA
} // namespace RNP

#endif // RNP_HESSENBERG_QZ_HPP_INCLUDED
