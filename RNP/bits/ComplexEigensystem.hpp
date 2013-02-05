#ifndef RNP_COMPLEX_EIGENSYSTEM_HPP_INCLUDED
#define RNP_COMPLEX_EIGENSYSTEM_HPP_INCLUDED

#include <RNP/bits/Rotation.hpp>
#include <RNP/bits/Hessenberg.hpp>
#include <RNP/bits/Triangular.hpp>
#include <RNP/bits/MatrixNorms.hpp>
#include <cstdio>

namespace RNP{
namespace LA{

namespace NonsymmetricEigensystem{

///////////////////////////////////////////////////////////////////////
// RNP::LA::NonsymmetricEigensystem
// ================================
// Auxilliary routines for the solution of nonsymmetric eigenvalue
// problems, including balancing and reduction to Schur form.
//
// The routines of this module are primarily due to Karen Braman and
// Ralph Byers, Department of Mathematics, University of Kansas, USA.
// For reference, see
//
//     K. Braman, R. Byers and R. Mathias, The Multi-Shift QR
//     Algorithm Part I: Maintaining Well Focused Shifts, and Level 3
//     Performance, SIAM Journal of Matrix Analysis, volume 23, pages
//     929-947, 2002.
//
//     K. Braman, R. Byers and R. Mathias, The Multi-Shift QR
//     Algorithm Part II: Aggressive Early Deflation, SIAM Journal
//     of Matrix Analysis, volume 23, pages 948--973, 2002.
//

///////////////////////////////////////////////////////////////////////
// Tuning
// ------
// Specialize this class to tune the various parameters of the implicit
// single/double-shift QR iteration, and the small bulge multi-shift
// QR iteration.
//
template <typename T>
struct Tuning{
	static inline size_t multishift_crossover_size(bool wantSchur, bool wantZ, size_t n, size_t ilo, size_t ihi){
		// must be at least 11
		return 75;
	}
	static inline size_t number_of_shifts(bool wantSchur, bool wantZ, size_t n, size_t ilo, size_t ihi, size_t lwork){
		// The return value should always be even.
		size_t nd = ihi-ilo;
		     if(nd <   30){ return 2; }
		else if(nd <   60){ return 4; }
		else if(nd <  150){ return 10; }
		else if(nd <  590){ return 20+((((nd-150)/10) | 1)-1); }
		else if(nd < 3000){ return 64; }
		else if(nd < 6000){ return 128; }
		else              { return 256; }
	}
	static inline size_t nibble_crossover_size(bool wantSchur, bool wantZ, size_t n, size_t ilo, size_t ihi, size_t lwork){
		return 14; // a percentage in decimal
	}
	static inline size_t deflation_window_size(bool wantSchur, bool wantZ, size_t n, size_t ilo, size_t ihi, size_t lwork){
		size_t ns = number_of_shifts(wantSchur, wantZ, n, ilo, ihi, lwork);
		if(ihi-ilo <= 500){
			return ns;
		}else{
			return 3*ns/2;
		}
	}
	
	// When BLAS Level 3 triangular routines _trmm are not efficient,
	// then general matrix multiply may be faster.
	static inline int matmul_type(bool wantSchur, bool wantZ, size_t n, size_t ilo, size_t ihi, size_t lwork){
		// 0 = No accumulation of reflections; applied with BLAS 2
		// 1 = Accumulate and apply with general matrix multiply
		// 2 = Accumulate and take advantage of banded structure with triangular matrix multiply
		int t = 0;
		size_t ns = number_of_shifts(wantSchur, wantZ, n, ilo, ihi, lwork);
		if(ns >= 14){
			t = 2;
		}
		return t;
	}
	
	static inline size_t balance_max_iters(){ return 64; }
	static inline size_t doubleshift_max_iters(size_t n, size_t ilo, size_t ihi){
		return 30;
	}
	static inline size_t multishift_max_iters(size_t n, size_t ilo, size_t ihi){
		return 30;
	}
};

///////////////////////////////////////////////////////////////////////
// Balance
// -------
// Balances a general square matrix A. This involves, first, permuting
// A by a similarity transformation to isolate eigenvalues in the
// ranges 0:ilo and ihi:n of elements on the diagonal; and second,
// applying a diagonal similarity transformation to rows and columns
// in the range ilo:ihi to make the rows and columns as close in norm
// as possible. Both steps are optional.
//
// Balancing may reduce the 1-norm of the matrix, and improve the
// accuracy of the computed eigenvalues and/or eigenvectors.
//
// ### Further Details
//
// The permutations consist of row and column interchanges which put
// the matrix in the form
//
//             [ T1   X   Y  ]
//     P A P = [  0   B   Z  ]
//             [  0   0   T2 ]
//
// where T1 and T2 are upper triangular matrices whose eigenvalues lie
// along the diagonal.  The column indices ilo and ihi mark the starting
// and 1 past ending columns of the submatrix B. Balancing consists of
// applying a diagonal similarity transformation inv(D) * B * D to make
// the 1-norms of each row of B and its corresponding column nearly
// equal. The output matrix is
//
//     [ T1     X*D          Y    ]
//     [  0  inv(D)*B*D  inv(D)*Z ]
//     [  0      0           T2   ]
//
// Information about the permutations P and the diagonal matrix D is
// returned in the vector scale.
//
// This subroutine is based on the EISPACK routine CBAL.
//
// Contributions by Tzu-Yi Chen, Computer Science Division,
//   University of California at Berkeley, USA
//
// This routine is based on Lapack _ggbal, except all information is
// stored in an integer array.
//
// Arguments
// job   If job = "N", then nothing is done and ilo is set to 0 and
//       ihi is set to n. If job = "P", then only permutations are
//       performed. If job = "S", only scaling is performed. If
//       job = "B", then both permutation and scaling are performed.
// n     The number of rows and columns of A.
// a     Pointer to the first element of A. On exit, the matrix is
//       overwritten with the balanced version.
// lda   Leading dimension of the array containing A, lda >= n.
// ilo   ilo and ihi are set such that A[i,j] = 0 if i > j and j
// ihi   in 0:ilo or ihi:n. If job = "N" or "S", ilo = 0 and ihi = n.
// scale Output array of length n, containing details of the
//       permutations and scaling factors applied to A.  If P[j] is
//       the index of the row and column interchanged with row and
//       column j and D[j] is the scaling factor applied to row and
//       column j (always a power of 2), then
//           scale[j] = P[j]         for j = 1:ilo
//                    = log(2,D[j])  for j = ilo:ihi
//                    = P[j]         for j = ihi:n
//       The order in which the interchanges are made is n-1 to ihi,
//       then 0 to ilo-1.
//
template <typename T>
void Balance(
	const char *job, size_t n, T *a, size_t lda,
	size_t *ilo, size_t *ihi, int *scale
){
	typedef typename Traits<T>::real_type real_type;
	static const T zero(0);
	static const real_type two(2);
	static const real_type threshold(real_type(95)/real_type(100));
	static const size_t max_iters = Tuning<T>::balance_max_iters();
	
	size_t k, l;
	*ilo = k = 0;
	*ihi = l = n;

	if(n == 0){
		return;
	}

	if('N' == job[0]){
		for(size_t i = 0; i < n; ++i){
			scale[i] = 0;
		}
		return;
	}

	if('S' != job[0]){
		// Permutation to isolate eigenvalues if possible
		// Search for rows isolating an eigenvalue and push them down.
		size_t j = l; while(j --> 0){
			bool found_nonzero = false;
			for(size_t i = 0; i < l; ++i){
				if(i == j){
					continue;
				}
				if(zero != a[j+i*lda]){
					found_nonzero = true;
					break;
				}
			}
			if(!found_nonzero){
				scale[l-1] = j;
				if(j+1 != l){
					BLAS::Swap(l, &a[0+j*lda], 1, &a[0+(l-1)*lda], 1);
					BLAS::Swap(n-k, &a[j+k*lda], lda, &a[l-1+k*lda], lda);
				}
				if(l == 1){
					*ilo = k;
					*ihi = l;
					return;
				}
				--l;
				j = l-1;
			}
		}

		// Search for columns isolating an eigenvalue and push them left.
		for(size_t j = k; j < l; ++j){
			bool found_nonzero = false;
			for(size_t i = k; i < l; ++i){
				if(i == j){
					continue;
				}
				if(zero != a[i+j*lda]){
					found_nonzero = true;
					break;
				}
			}
			if(!found_nonzero){
				scale[k] = j;
				if(j != k){
					BLAS::Swap(l, &a[0+j*lda], 1, &a[0+k*lda], 1);
					BLAS::Swap(n-k, &a[j+k*lda], lda, &a[k+k*lda], lda);
				}
				++k;
				j = k;
			}
		}
	}

	for(size_t i = k; i < l; ++i){
		scale[i] = real_type(1);
	}

	if('P' == job[0]){
		*ilo = k;
		*ihi = l;
		return;
	}

	// Balance the submatrix in rows k..l.
	// Iterative loop for norm reduction
	
	const real_type sfmin1 = Traits<real_type>::min() / (two*Traits<real_type>::eps());
	const real_type sfmax1 = real_type(1) / sfmin1;
	const real_type sfmin2 = sfmin1 * two;
	const real_type sfmax2 = real_type(1) / sfmin2;

	bool noconv;
	size_t iter = max_iters;
	do{
		noconv = false;

		for(size_t i = k; i < l; ++i){
			real_type c(0);
			real_type r(0);

			for(size_t j = k; j < l; ++j){
				if(j != i){
					c += Traits<T>::norm1(a[j+i*lda]);
					r += Traits<T>::norm1(a[i+j*lda]);
				}
			}
			size_t ica = BLAS::MaximumIndex(l, &a[0+i*lda], 1);
			real_type ca = std::abs(a[ica+i*lda]);
			size_t ira = BLAS::MaximumIndex(n-k, &a[i+k*lda], lda);
			real_type ra = std::abs(a[i+(ira+k)*lda]);

			// Guard against zero C or R due to underflow.
			if(real_type(0) == c || real_type(0) == r){
				continue;
			}
			real_type g = r / two;
			real_type f(1); int ifs = 0;
			real_type s = c + r;
			while(!(c >= g ||
				f >= sfmax2 || c >= sfmax2 || ca >= sfmax2 ||
				r <= sfmin2 || g <= sfmin2 || ra <= sfmin2
			)){
				f *= two; ifs++;
				c *= two;
				ca *= two;
				r /= two;
				g /= two;
				ra /= two;
			}

			g = c / two;
			while(!(g < r ||
				r >= sfmax2 || ra >= sfmax2 ||
				f <= sfmin2 || c <= sfmin2 || g <= sfmin2 || ca <= sfmin2
			)){
				f /= two; ifs--;
				c /= two;
				g /= two;
				ca /= two;
				r *= two;
				ra *= two;
			}

			// Now balance.
			if(c + r >= s * threshold){
				continue;
			}
			if(ifs < 0 && scale[i] < 0){
				real_type fac(1);
				BLAS::Rescale(1, ifs + scale[i], &fac, 0);
				if(fac <= sfmin1){
					continue;
				}
			}
			if(ifs > 0 && scale[i] > 0){
				real_type fac(1);
				BLAS::Rescale(1, scale[i], &fac, 0);
				if(fac >= sfmax1 / f){
					continue;
				}
			}
			//g = real_type(1) / f;
			scale[i] += ifs;
			noconv = true;

			BLAS::Rescale(n-k, -ifs, &a[i+k*lda], lda);
			BLAS::Rescale(l  ,  ifs, &a[0+i*lda], 1);
		}
	}while(noconv && iter --> 0);

	*ilo = k;
	*ihi = l;
}

///////////////////////////////////////////////////////////////////////
// BalanceUndo
// -----------
// Forms the right or left eigenvectors of a general matrix by
// backward transformation on the computed eigenvectors of the
// balanced matrix output by Balance.
//
// Arguments
// job   If job = "N", then nothing is done and ilo is set to 0 and
//       ihi is set to n. If job = "P", then only permutations are
//       performed. If job = "S", only scaling is performed. If
//       job = "B", then both permutation and scaling are performed.
//       This value should be the same as whatever was originally
//       passed to Balance.
// side  If side = "L", v contains the left eigenvectors.
//       If side = "R", v contains the right eigenvectors.
// n     The number of rows and columns of A.
// ilo   ilo and ihi are set such that A[i,j] = 0 if i > j and j
// ihi   in 0:ilo or ihi:n. If job = "N" or "S", ilo = 0 and ihi = n.
// scale The details of the permutations and scaling factors returned
//       by Balance.
// m     The number of columns of v.
// v     Pointer to the first element of the matrix of eigenvectors.
// ldv   Leading dimension of the array containing v, ldv >= n.
//
template <typename T>
void BalanceUndo(
	const char *job, const char *side,
	size_t n, size_t ilo, size_t ihi,
	int *scale, size_t m, T *v, size_t ldv
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
					BLAS::Rescale(m, -scale[i], &v[i+0*ldv], ldv);
				}
			}else{
				for(size_t i = ilo; i < ihi; ++i){
					BLAS::Rescale(m, scale[i], &v[i+0*ldv], ldv);
				}
			}
		}
	}
	
	if('P' == job[0] || 'B' == job[0]){
		// Backward permutation
		// For  I = ILO-1 step -1 until 1,
		//        IHI+1 step 1 until N do
		if(left){
			for(size_t ii = 0; ii < n; ++ii){
				size_t i = ii;
				if(i >= ilo && i < ihi){
					continue;
				}
				if(i < ilo){
					i = ilo - ii;
				}
				size_t k = scale[i];
				if(k != i){
					BLAS::Swap(m, &v[i+0*ldv], ldv, &v[k+0*ldv], ldv);
				}
			}
		}else{
			for(size_t ii = 0; ii < n; ++ii){
				size_t i = ii;
				if(i >= ilo && i < ihi){
					continue;
				}
				if(i < ilo){
					i = ilo - ii;
				}
				size_t k = scale[i];
				if(k != i){
					BLAS::Swap(m, &v[i+0*ldv], ldv, &v[k+0*ldv], ldv);
				}
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////
// ComplexImplicitSingleShiftQR
// ----------------------------
// Implements the complete single-shift QR algorithm for a complex
// upper Hessenberg matrix.
// This is equivalent to Lapack routine zlahqr.
//
// Returns 0 on success. A return value greater than 0 indicates a
// convergence failure after reaching the iteration limit. The elements
// in the range info:ihi of w contain those eigenvalues which have been
// successfully computed. If info > 0 and wantt is false, then the
// remaining unconverged eigenvalues are the eigenvalues of the upper
// Hessenberg matrix in the range ilo:info. If wantt is true, then
// H0*U = U*H1 where H0 is the initial input Hessenberg matrix, and H1
// is the final value, where U is a unitary matrix. The final value is
// triangular in the range info:ihi. If info > 0 and wantz is true,
// then Z1 = Z0*U, regardless of the value of wantt.
//
// Arguments
// wantt Set to true if the full Schur form T is required, otherwise
//       only eigenvalues are computed.
// wantz Set to true if the matrix of Schur vectors Z is required.
// n     The number of rows and columns of H.
// ilo   It is assumed the matrix H is already upper triangular outside
// ihi   the range ilo:ihi. Note that the back transformations are
//       applied to the full range 0:n if wantt is true.
// h     Pointer to the first element of H.
// ldh   Leading dimension of the array containing H, ldh >= n.
// w     Pointer to the array of eigenvalues. Only the range ilo:ihi
//       is touched.
// iloz  The range iloz:ihiz specify which rows of Z the transformations
// ihiz  should be applied if wantz is true.
// z     Pointer to the first element of the matrix of Schur vectors.
//       If a transformation was used to reduce a general matrix to
//       Hessenberg form, Z should contain the transformation matrix.
// ldz   Leading dimension of the array containing Z, ldz >= n.
//
template <typename T>
int ComplexImplicitSingleShiftQR(
	bool wantt, bool wantz, size_t n, size_t ilo, size_t ihi,
	std::complex<T> *h, size_t ldh,
	std::complex<T> *w, size_t iloz, size_t ihiz,
	std::complex<T> *z, size_t ldz
){
	typedef typename std::complex<T> complex_type;
	typedef                       T  real_type;
	typedef Traits<complex_type> complex_traits;
	typedef Traits<real_type>    real_traits;
	
	static const complex_type zero(real_type(0));
	static const complex_type one(real_type(1));
	static const real_type rzero(0);
	static const real_type half(real_type(1)/real_type(2));
	static const real_type dat1(real_type(3)/real_type(4));

	const size_t itmax = Tuning<complex_type>::doubleshift_max_iters(n, ilo, ihi);
	
	size_t info = 0;
	
	if(0 == n){ return 0; }
	if(ilo+1 == ihi){
		w[ilo] = h[ilo+ilo*ldh];
		return 0;
	}
	
	// clear out the trash
	for(size_t j = ilo; j+3 < ihi; ++j){
		h[(j+2)+j*ldh] = zero;
		h[(j+3)+j*ldh] = zero;
	}
	if(ilo+2 < ihi){
		h[(ihi-1)+(ihi-3)*ldh] = zero;
	}
	
	{ // Ensure that subdiagonal entries are real
		size_t jhi;
		size_t jlo;
		if(wantt){
			jlo = 0;
			jhi = n;
		}else{
			jlo = ilo;
			jhi = ihi;
		}
		for(size_t i = ilo+1; i < ihi; ++i){
			if(rzero != h[i+(i-1)*ldh].imag()){
				// The following redundant normalization
				// avoids problems with both gradual and
				// sudden underflow in abs(H(I,I-1))
				complex_type sc = h[i+(i-1)*ldh] / complex_traits::norm1(h[i+(i-1)*ldh]);
				sc = complex_traits::conj(sc) / complex_traits::abs(sc);
				h[i+(i-1)*ldh] = complex_traits::abs(h[i+(i-1)*ldh]);
				BLAS::Scale(jhi-i, sc, &h[i+i*ldh], ldh);
				const size_t mj = (jhi < i+2 ? jhi : i+2);
				BLAS::Scale(mj-jlo, complex_traits::conj(sc), &h[jlo+i*ldh], 1);
				if(wantz){
					BLAS::Scale(ihiz-iloz, complex_traits::conj(sc), &z[iloz+i*ldz], 1);
				}
			}
		}
	}
	
	const size_t nh = ihi - ilo;
	const size_t nz = ihiz - iloz;
	
	// Set machine-dependent constants for the stopping criterion.
	const real_type safmin(Traits<real_type>::min());
	//const double safmax = rone / safmin;
	const real_type ulp(real_type(2)*Traits<real_type>::eps());
	const real_type smlnum(safmin*(real_type(nh) / ulp));

	// I1 and I2 are the indices of the first row and 1 + last column of H
	// to which transformations must be applied. If eigenvalues only are
	// being computed, I1 and I2 are set inside the main loop.
	size_t i1 = 0, i2 = n;

	// The main loop begins here. I is the loop index and decreases from
	// IHI to ILO in steps of 1. Each iteration of the loop works
	// with the active submatrix in rows and columns L to I.
	// Eigenvalues I+1 to IHI have already converged. Either L = ILO, or
	// H(L,L-1) is negligible so that the matrix splits.
	size_t i = ihi-1;
	
	
	while(1){
		if(i < ilo){ break; }
		// Perform QR iterations on rows and columns ILO to I until a
		// submatrix of order 1 splits off at the bottom because a
		// subdiagonal element has become negligible.
		size_t l = ilo;
		bool converged = false;
		for(size_t its = 0; its < itmax; ++its){
			// Look for a single small subdiagonal element.
			{ size_t k;
				for(k = i; k > l; --k){
					if(complex_traits::norm1( h[k+(k-1)*ldh] ) <= smlnum ){ break; }
					real_type tst = complex_traits::norm1( h[(k-1)+(k-1)*ldh] ) + complex_traits::norm1( h[k+k*ldh] );
					if(rzero == tst){
						if(k >= ilo+2 ) tst += real_traits::abs(h[(k-1)+(k-2)*ldh].real());
						if(k+2 <= ihi ) tst += real_traits::abs(h[k+1+k*ldh].real());
					}
					// The following is a conservative small subdiagonal
					// deflation criterion due to Ahues & Tisseur (LAWN 122, 1997).
					// It has better mathematical foundation and
					// improves accuracy in some examples.
					if( real_traits::abs( h[k+(k-1)*ldh].real() ) <= ulp*tst ){
						real_type ab = complex_traits::norm1(h[  k  +(k-1)*ldh]);
						real_type ba = complex_traits::norm1(h[(k-1)+  k  *ldh]);
						if(ba > ab){ std::swap(ab, ba); }
						real_type aa = complex_traits::norm1(h[k+k*ldh]);
						real_type bb = complex_traits::norm1(h[(k-1)+(k-1)*ldh] - h[k+k*ldh]);
						if(bb > aa){ std::swap(aa, bb); }
						real_type s = aa + ab;
						real_type ulpbas = ulp*(bb*(aa/s));
						if(ba*(ab/s) <= (smlnum > ulpbas ? smlnum : ulpbas)){ break; }
					}
				}
				l = k;
			}
			if(l > ilo){ // H(L,L-1) is negligible
				h[l+(l-1)*ldh] = zero;
			}

			// Exit from loop if a submatrix of order 1 has split off.
			if(l >= i){
				converged = true;
				break;
			}

			// Now the active submatrix is in rows and columns L to I. If
			// eigenvalues only are being computed, only the active submatrix
			// need be transformed.
			if(!wantt){
				i1 = l;
				i2 = i+1;
			}
			complex_type t;
			if(10 == its){ // Exceptional shift.
				real_type s(dat1*real_traits::abs( h[l+1+l*ldh].real() ));
				t = s + h[l+l*ldh];
			}else if(20 == its){ // Exceptional shift.
				real_type s(dat1*real_traits::abs( h[i+(i-1)*ldh].real() ));
				t = s + h[i+i*ldh];
			}else{ // Wilkinson's shift.
				t = h[i+i*ldh];
				complex_type u(sqrt( h[(i-1)+i*ldh] )*sqrt( h[i+(i-1)*ldh] ));
				real_type s(complex_traits::norm1( u ));
				if(rzero != s){
					complex_type x = half*( h[(i-1)+(i-1)*ldh]-t );
					real_type sx = complex_traits::norm1(x);
					if(sx > s){ s = sx; }
					complex_type y = s*sqrt((x/s)*(x/s) + (u/s)*(u/s));
					if(sx > rzero){
						if(complex_traits::real(x/sx)*y.real()+complex_traits::imag(x/sx)*y.imag() < rzero){ y = -y; }
					}
					// Precise complex division needed here
					t -= u*complex_traits::div(u, x+y);
				}
			}

			// Look for two consecutive small subdiagonal elements.
			complex_type v[2];
			bool found_small = false;
			size_t m;

			for(m = i-1; m > l; --m){
				// Determine the effect of starting the single-shift QR
				// iteration at row M, and see if this would make H(M,M-1)
				// negligible.
				complex_type h11 = h[m+m*ldh];
				complex_type h22 = h[m+1+(m+1)*ldh];
				complex_type h11s = h11 - t;
				real_type h21 = h[m+1+m*ldh].real();
				real_type s = complex_traits::norm1( h11s ) + real_traits::abs( h21 );
				h11s = h11s / s;
				h21 = h21 / s;
				v[0] = h11s;
				v[1] = h21;
				real_type h10 = h[m+(m-1)*ldh].real();
				if(real_traits::abs( h10 )*real_traits::abs( h21 ) <= ulp* (complex_traits::norm1( h11s )*(complex_traits::norm1( h11 )+complex_traits::norm1( h22 ) ) ) ){
					found_small = true;
					break;
				}
			}
			if(!found_small){
				 complex_type h11 = h[l+l*ldh];
				 complex_type h11s = h11 - t;
				 real_type h21 = std::real( h[l+1+l*ldh] );
				 real_type s = complex_traits::norm1( h11s ) + real_traits::abs( h21 );
				 h11s = h11s / s;
				 h21 = h21 / s;
				 v[0] = h11s;
				 v[1] = h21;
			}
			
			// Single-shift QR step
			for(size_t k = m; k < i; ++k){
				// The first iteration of this loop determines a reflection G
				// from the vector V and applies it from left and right to H,
				// thus creating a nonzero bulge below the subdiagonal.
				//
				// Each subsequent iteration determines a reflection G to
				// restore the Hessenberg form in the (K-1)th column, and thus
				// chases the bulge one step toward the bottom of the active
				// submatrix.
				//
				// V(2) is always real before the call to ZLARFG, and hence
				// after the call T2 ( = T1*V(2) ) is also real.

				if(k > m){
					BLAS::Copy(2, &h[k+(k-1)*ldh], 1, v, 1);
				}
				complex_type t1;
				Reflector::Generate(2, v, &v[1], 1, &t1);
				if(k > m){
					h[k+(k-1)*ldh] = v[0];
					h[k+1+(k-1)*ldh] = zero;
				}
				complex_type v2 = v[1];
				real_type t2 = complex_traits::real(t1*v2);

				// Apply G from the left to transform the rows of the matrix
				// in columns K to I2.
				for(size_t j = k; j < i2; ++j){
					complex_type sum = complex_traits::conj(t1)*h[k+j*ldh] + t2*h[k+1+j*ldh];
					h[k+j*ldh] -= sum;
					h[k+1+j*ldh] -= sum*v2;
				}

				// Apply G from the right to transform the columns of the
				// matrix in rows I1 to min(K+2,I).
				const size_t rowlim = (k+2 < i ? k+2 : i);
				for(size_t j = i1; j <= rowlim; ++j){
					complex_type sum = t1*h[j+k*ldh] + t2*h[j+(k+1)*ldh];
					h[j+k*ldh] -= sum;
					h[j+(k+1)*ldh] -= sum * complex_traits::conj(v2);
				}

				if(wantz){ // Accumulate transformations in the matrix Z
					for(size_t j = iloz; j < ihiz; ++j){
						complex_type sum = t1*z[j+k*ldz] + t2*z[j+(k+1)*ldz];
						z[j+k*ldz] -= sum;
						z[j+(k+1)*ldz] -= sum * complex_traits::conj(v2);
					}
				}

				if(k == m && m > l){
					// If the QR step was started at row M > L because two
					// consecutive small subdiagonals were found, then extra
					// scaling must be performed to ensure that H(M,M-1) remains
					// real.
					complex_type temp = one - t1;
					temp /= complex_traits::abs(temp);
					h[m+1+m*ldh] *= complex_traits::conj(temp);
					if(m+2 <= i){
						h[m+2+(m+1)*ldh] *= temp;
					}
					for(size_t j = m; j <= i; ++j){
						if(j != m+1){
							if(i2 > j+1){
								BLAS::Scale(i2-j-1, temp, &h[j+(j+1)*ldh], ldh);
							}
							BLAS::Scale(j-i1, complex_traits::conj(temp), &h[i1+j*ldh], 1);
							if(wantz){
								BLAS::Scale(nz, complex_traits::conj(temp), &z[iloz+j*ldz], 1);
							}
						}
					}
				}
			}

			// Ensure that H(I,I-1) is real.
			complex_type temp = h[i+(i-1)*ldh];
			if(rzero != temp.imag()){
				real_type rtemp = complex_traits::abs(temp);
				h[i+(i-1)*ldh] = rtemp;
				temp /= rtemp;
				if(i2 > i){
					BLAS::Scale(i2-i-1, complex_traits::conj(temp), &h[i+(i+1)*ldh], ldh);
				}
				BLAS::Scale(i-i1, temp, &h[i1+i*ldh], 1);
				if(wantz){
					BLAS::Scale(nz, temp, &z[iloz+i*ldz], 1);
				}
			}
		}
		// Failure to converge in remaining number of iterations
		if(!converged){
			return i+1;
		}
		// H(I,I-1) is negligible: one eigenvalue has converged.
		w[i] = h[i+i*ldh];
		if(0 == l){ break; } // prevent the check at the top from failing from underflow
		// return to start of the main loop with new value of I.
		i = l-1;
	}
	return info;
}

template <typename T>
int ComplexSmallBulgeMultishiftQR(int level,
	bool wantt, bool wantz, size_t n, size_t ilo, size_t ihi,
	std::complex<T> *h, size_t ldh, std::complex<T> *w,
	size_t iloz, size_t ihiz, std::complex<T> *z, size_t ldz,
	size_t *lwork, std::complex<T> *work
);

///////////////////////////////////////////////////////////////////////
// AggressiveEarlyDeflation
// ------------------------
// This is an auxilliary routine which accepts as input an upper
// Hessenberg matrix H and performs a unitary similarity transformation
// designed to detect and deflate fully converged eigenvalues from a
// trailing principal submatrix. On output H has been over-written by
// a new Hessenberg matrix that is a perturbation of a unitary
// similarity transformation of H. It is hoped that the final version
// of H has many zero subdiagonal entries.
//
// This is equivalent to Lapack routine zlaqr3 and zlaqr2.
// Only a partial explanation of the arguments will be given here.
//
// Arguments
// level  The recursion level. Set to 0 on the highest level call, and
//        increase by one for each level. Currently it is assumed level
//        does not exceed 1. When level = 0, this is equivalent to
//        Lapack routine zlaqr3, otherwise it behaves as zlaqr2.
//
template <typename T>
int AggressiveEarlyDeflation(int level, bool wantt, bool wantz, size_t n, 
	size_t ktop, size_t kbot, size_t nw, std::complex<T> *h, 
	size_t ldh, size_t iloz, size_t ihiz, std::complex<T> *z, 
	size_t ldz, size_t *ns, size_t *nd, std::complex<T> *sh, 
	std::complex<T> *v, size_t ldv, size_t nh, std::complex<T> *t,
	size_t ldt, size_t nv, std::complex<T> *wv, size_t ldwv,
	size_t *lwork, std::complex<T> *work
){
	typedef typename std::complex<T> complex_type;
	typedef                       T  real_type;
	typedef Traits<complex_type> complex_traits;
	typedef Traits<real_type>    real_traits;
	
	static const complex_type zero(real_type(0));
	static const complex_type one (real_type(1));
	static const real_type rzero(0);

    const size_t jw = (nw < kbot-ktop ? nw : kbot-ktop);
    if(0 == *lwork){
		size_t sublwork = 0;
		Hessenberg::Reduce(jw, 0, jw-1, t, ldt, work, &sublwork, work);
		if(sublwork > *lwork){ *lwork = sublwork; }
		
		sublwork = 0;
		Hessenberg::MultQ(
			"R", "N", jw, jw, 0, jw-1, t, ldt, work,
			v, ldv, &sublwork, work
		);
		if(sublwork > *lwork){ *lwork = sublwork; }
		
		*lwork += jw;

		sublwork = 0;
		ComplexSmallBulgeMultishiftQR(
			1, true, true, jw, 0, jw, t, ldt, sh,
			0, jw, v, ldv, &sublwork, work
		);
		if(sublwork > *lwork){ *lwork = sublwork; }
		return 0;
    }


	*ns = 0;
	*nd = 0;

	if(ktop >= kbot){
		return 0;
	}
	if(nw < 1){
		return 0;
	}

	// Machine constants
	const real_type safmin = Traits<real_type>::min();
	const real_type safmax = real_type(1) / safmin;
	const real_type ulp = real_type(2)*Traits<real_type>::eps();
	const real_type smlnum = safmin * real_type(n) / ulp;

	// Setup deflation window
	size_t kwtop = kbot - jw;
	complex_type s(kwtop == ktop ? zero : h[kwtop+(kwtop-1)*ldh]);

	if(kbot == kwtop+1){ // 1-by-1 deflation window: not much to do
		sh[kwtop] = h[kwtop+kwtop*ldh];
		*ns = 1;
		*nd = 0;
		real_type threshold = ulp * complex_traits::norm1(h[kwtop+kwtop*ldh]);
		if(smlnum > threshold){ threshold = smlnum; }
		if(complex_traits::norm1(s) <= threshold){
			*ns = 0;
			*nd = 1;
			if(kwtop > ktop){
				h[kwtop+(kwtop-1)*ldh] = zero;
			}
		}
		return 0;
	}

	// Convert to spike-triangular form.  (In case of a
	// rare QR failure, this routine continues to do
	// aggressive early deflation using that part of
	// the deflation window that converged using INFQR
	// here and there to keep track.)
    Triangular::Copy("U", jw, jw, &h[kwtop+kwtop*ldh], ldh, t, ldt);
    BLAS::Copy(jw-1, &h[kwtop+1+kwtop*ldh], ldh+1, &t[1+0*ldt], ldt+1);
    BLAS::Set(jw, jw, zero, one, v, ldv);

	const size_t nmin = Tuning<T>::multishift_crossover_size(true, true, jw, 0, jw);
	int infqr;
	if(0 == level && jw >= nmin){
		infqr = ComplexSmallBulgeMultishiftQR(
			1, true, true, jw, 0, jw, t, ldt, &sh[kwtop], 0, jw, v, ldv, lwork, work
		);
    }else{
		infqr = ComplexImplicitSingleShiftQR(
			true, true, jw, 0, jw, t, ldt, &sh[kwtop], 0, jw, v, ldv
		);
    }

	// Deflation detection loop
	{
		*ns = jw;
		size_t ilst = infqr;
		for(size_t knt = infqr; knt < jw; ++knt){
			// Small spike tip deflation test
			real_type foo = complex_traits::norm1(t[(*ns-1)+(*ns-1)*ldt]);
			if(rzero == foo){
				foo = complex_traits::norm1(s);
			}
			if(complex_traits::norm1(s) * complex_traits::norm1(v[0+(*ns-1)*ldv]) <= (smlnum > ulp*foo ? smlnum : ulp*foo)){
				// One more converged eigenvalue
				--(*ns);
			}else{
				// One undeflatable eigenvalue.  Move it up out of the
				// way.   (ZTREXC can not fail in this case.)
				size_t ifst = *ns-1;
				Triangular::ExchangeDiagonal(
					jw, t, ldt, v, ldv, ifst, ilst
				);
				++ilst;
			}
		}
	}

	// Return to Hessenberg form
	if(0 == *ns){
		s = zero;
    }

    if(*ns < jw){
		// sorting the diagonal of T improves accuracy for graded matrices.
		for(size_t i = infqr; i < *ns; ++i){
			size_t ifst = i;
			for(size_t j = i + 1; j < *ns; ++j){
				if(complex_traits::norm1(t[j+j*ldt]) > complex_traits::norm1(t[ifst+ifst*ldt])){
					ifst = j;
				}
			}
			size_t ilst = i;
			if(ifst != ilst){
				Triangular::ExchangeDiagonal(jw, t, ldt, v, ldv, ifst, ilst);
			}
		}
    }
	
	// Restore shift/eigenvalue array from T
	for(size_t i = infqr; i < jw; ++i){
		sh[kwtop+i] = t[i+i*ldt];
	}

	if(*ns < jw || zero == s){
		if(*ns > 1 && zero != s){
			// Reflect spike back into lower triangle
			BLAS::Copy(*ns, v, ldv, work, 1);
			for(size_t i = 0; i < *ns; ++i){
				work[i] = complex_traits::conj(work[i]);
			}
			complex_type beta(work[0]);
			complex_type tau;
			Reflector::Generate(*ns, &beta, &work[1], 1, &tau);

			work[0] = one;
			Triangular::Set("L", jw-2, jw-2, zero, zero, &t[2+0*ldt], ldt);
			Reflector::Apply("L", 0, false, *ns,  jw, work, 1, complex_traits::conj(tau), t, ldt, &work[jw]);
			Reflector::Apply("R", 0, false, *ns, *ns, work, 1, tau, t, ldt, &work[jw]);
			Reflector::Apply("R", 0, false,  jw, *ns, work, 1, tau, v, ldv, &work[jw]);
			size_t reduce_lwork = *lwork - jw;
			Hessenberg::Reduce(jw, 0, *ns, t, ldt, work, &reduce_lwork, &work[jw]);
		}

		// Copy updated reduced window into place
		if(kwtop > 1){
			h[kwtop+(kwtop-1)*ldh] = s * std::conj(v[0+0*ldv]);
		}
		Triangular::Copy("U", jw, jw, t, ldt, &h[kwtop+kwtop*ldh], ldh);
		BLAS::Copy(jw-1, &t[1+0*ldt], ldt+1, &h[kwtop+1+kwtop*ldh], ldh+1);

		// Accumulate orthogonal matrix in order update H and Z, if requested.
		if (*ns > 1 && (s != 0.)) {
			size_t multq_lwork = *lwork - jw;
			Hessenberg::MultQ("R", "N", jw, *ns, 0, *ns, t, ldt, work, v, ldv, &multq_lwork, &work[jw]);
		}

		// Update vertical slab in H
		const size_t ltop = (wantt ? 0 : ktop);
		for(size_t krow = ltop; krow < kwtop; krow += nv){
			const size_t kln = (nv < kwtop-krow ? nv : kwtop-krow);
			BLAS::MultMM("N", "N", kln, jw, jw, one, &h[krow+kwtop*ldh], ldh, v, ldv, zero, wv, ldwv);
			BLAS::Copy(kln, jw, wv, ldwv, &h[krow+kwtop*ldh], ldh);
		}

		// Update horizontal slab in H
		if(wantt){
			for(size_t kcol = kbot; kcol < n; kcol += nh){
				const size_t kln = (nh < n-kcol ? nh : n-kcol);
				BLAS::MultMM("C", "N", jw, kln, jw, 1., v, ldv, &h[kwtop+kcol*ldh], ldh, zero, t, ldt);
				BLAS::Copy(jw, kln, t, ldt, &h[kwtop+kcol*ldh], ldh);
			}
		}

		// Update vertical slab in Z
		if(wantz){
			for(size_t krow = iloz; krow < ihiz; krow += nv){
				const size_t kln = (nv < ihiz-krow ? nv : ihiz-krow);
				BLAS::MultMM("N", "N", kln, jw, jw, one, &z[krow+kwtop*ldz], ldz, v, ldv, zero, wv, ldwv);
				BLAS::Copy(kln, jw, wv, ldwv, &z[krow+kwtop*ldz], ldz);
			}
		}
    }

	// Return the number of deflations ...
    *nd = jw - *ns;

	// ... and the number of shifts. (Subtracting
	// INFQR from the spike length takes care
	// of the case of a rare QR failure while
	// calculating eigenvalues of the deflation
	// window.)
    *ns -= infqr;

    return 0;
}

///////////////////////////////////////////////////////////////////////
// ComplexImplicitBulgeStart
// -------------------------
// Given a 2-by-2 or 3-by-3 matrix H, this routine sets v to a scalar
// multiple of the first column of the product
// 
//     K = (H - s1*I)*(H - s2*I)
// 
// scaling to avoid overflows and most underflows.
// 
// This is useful for starting double implicit shift bulges in the
// QR algorithm.
//
// This is equivalent to Lapack routine zlaqr1.
//
// Arguments
// n     Order of the matrix H. N must be either 2 or 3.
// h     Pointer to the first element of H.
// ldh   Leading dimension of the array containing H, ldh >= n.
// s1    Shift defining K.
// s2    Shift defining K.
// v     Vector of length n, a scalar multiple of the first column of
//       the matrix K.
//
template <typename T>
void ComplexImplicitBulgeStart(
	size_t n, const std::complex<T> *h, size_t ldh,
	const std::complex<T> &s1,
	const std::complex<T> &s2, std::complex<T> *v
){
	typedef typename std::complex<T> complex_type;
	typedef                       T  real_type;
	typedef Traits<complex_type> complex_traits;
	typedef Traits<real_type>    real_traits;
	
	static const real_type rzero(0);
	static const complex_type zero(rzero);
	
	if( n == 2 ){
		real_type s = complex_traits::norm1( h[0+0*ldh]-s2 ) + complex_traits::norm1( h[1+0*ldh] );
		if( s == rzero ){
			v[0] = zero;
			v[1] = zero;
		}else{
			complex_type h21s = h[1+0*ldh] / s;
			v[0] = h21s*h[0+1*ldh] + ( h[0+0*ldh]-s1 )* ( ( h[0+0*ldh]-s2 ) / s );
			v[1] = h21s*( h[0+0*ldh]+h[1+1*ldh]-s1-s2 );
		}
	}else{
		real_type s = complex_traits::norm1( h[0+0*ldh]-s2 ) + complex_traits::norm1( h[1+0*ldh] ) + complex_traits::norm1( h[2+0*ldh] );
		if( s == zero ){
			v[0] = zero;
			v[1] = zero;
			v[2] = zero;
		}else{
			complex_type h21s = h[1+0*ldh] / s;
			complex_type h31s = h[2+0*ldh] / s;
			v[0] = ( h[0+0*ldh]-s1 )*( ( h[0+0*ldh]-s2 ) / s ) + h[0+1*ldh]*h21s + h[0+2*ldh]*h31s;
			v[1] = h21s*( h[0+0*ldh]+h[1+1*ldh]-s1-s2 ) + h[1+2*ldh]*h31s;
			v[2] = h31s*( h[0+0*ldh]+h[2+2*ldh]-s1-s2 ) + h21s*h[2+1*ldh];
		}
	}
}

///////////////////////////////////////////////////////////////////////
// ComplexSmallBulgeMultishiftQRSweep
// ----------------------------------
// This routine performs a single small-bulge multi-shift QR sweep.
//
// This is equivalent to Lapack routine zlaqr5.
//
template <typename T>
void ComplexSmallBulgeMultishiftQRSweep(
	bool wantt, bool wantz, int kacc22, size_t n, int ktop, int kbot, size_t nshfts,
	std::complex<T> *s, std::complex<T> *h, size_t ldh, int iloz, int ihiz,
	std::complex<T> *z, size_t ldz, std::complex<T> *v, size_t ldv,
	std::complex<T> *u, size_t ldu, size_t nv, std::complex<T> *wv, size_t ldwv,
	size_t nh, std::complex<T> *wh, size_t ldwh
){
	typedef typename std::complex<T> complex_type;
	typedef                       T  real_type;
	typedef Traits<complex_type> complex_traits;
	typedef Traits<real_type>    real_traits;
	
	static const complex_type zero(real_type(0));
	static const complex_type one(real_type(1));
	static const real_type rzero(0);
	
	using std::min;
	using std::max;

	// If there are no shifts, then there is nothing to do.
	if( nshfts < 2 ) return;

	// If the active block is empty or 1-by-1, then there
	// .    is nothing to do.
	if( ktop+1 >= kbot ) return;
	
	// NSHFTS is supposed to be even, but if it is odd,
	// .    then simply reduce it by one. 
	const int ns = nshfts - nshfts%2;

	// Machine constants for deflation
	const real_type safmin = real_traits::min();
	//safmax = rone / safmin;
	const real_type ulp = real_type(2)*real_traits::eps();
	const real_type smlnum = safmin*( real_type(n) / ulp );

	// Use accumulated reflections to update far-from-diagonal
	// .    entries ?
	const bool accum = ( kacc22 == 1 )  ||  ( kacc22 == 2 );

	// If so, exploit the 2-by-2 block structure?
	const bool blk22 = ( ns > 2 )  &&  ( kacc22 == 2 );

	// clear trash
	if(ktop+3 <= kbot){ h[(ktop+2)+ktop*ldh] = zero; }

	// NBMPS = number of 2-shift bulges in the chain
	const int nbmps = ns / 2;

	// KDU = width of slab
	const int kdu = 6*nbmps - 3;
	
	const int nb32 = 3*nbmps-2;

	// Create and chase chains of NBMPS bulges
	for(int incol = -nb32 + ktop; incol < kbot - 2; incol += nb32){
		int ndcol = incol+1 + kdu;
		if( accum ) BLAS::Set(kdu, kdu, zero, one, u, ldu );

		// Near-the-diagonal bulge chase.  The following loop
		// .    performs the near-the-diagonal part of a small bulge
		// .    multi-shift QR sweep.  Each 6*NBMPS-2 column diagonal
		// .    chunk extends from column INCOL to column NDCOL
		// .    (including both column INCOL and column NDCOL). The
		// .    following loop chases a 3*NBMPS column long chain of
		// .    NBMPS bulges 3*NBMPS-2 columns to the right.  (INCOL
		// .    may be less than KTOP and and NDCOL may be greater than
		// .    KBOT indicating phantom columns from which to chase
		// .    bulges before they are actually introduced or to which
		// .    to chase bulges beyond column KBOT.)
		for(int krcol = incol; krcol < min( incol+nb32, kbot-2 ); ++krcol){

			// Bulges number MTOP to MBOT are active double implicit
			// .    shift bulges.  There may or may not also be small
			// .    2-by-2 bulge, if there is room.  The inactive bulges
			// .    (if any) must wait until the active bulges have moved
			// .    down the diagonal to make room.  The phantom matrix
			// .    paradigm described above helps keep track. 
			int mtop = max( 0, ( ktop+1-krcol ) / 3 );
			int mbot = min( nbmps, ( kbot-krcol-1 ) / 3 );
			int m22 = mbot;
			const bool bmp22 = (mbot < nbmps) && (krcol+1+3*m22) == (kbot-2);
			// Generate reflections to chase the chain right
			// .    one column.  (The minimum value of K is KTOP-1.)
			for(int m = mtop; m < mbot; ++m){
				int k = krcol + 3*m;
				if(k+1 == ktop){
					ComplexImplicitBulgeStart(3, &h[ktop+ktop*ldh], ldh, s[2*m], s[2*m+1], &v[0+m*ldv]);
					complex_type alpha = v[0+m*ldv];
					Reflector::Generate(3, &alpha, &v[1+m*ldv], 1, &v[0+m*ldv]);
				}else{
					complex_type beta(h[k+1+k*ldh]);
					v[1+m*ldv] = h[(k+2)+k*ldh];
					v[2+m*ldv] = h[(k+3)+k*ldh];
					Reflector::Generate(3, &beta, &v[1+m*ldv], 1, &v[0+m*ldv]);
					// 
					// A Bulge may collapse because of vigilant
					// .    deflation or destructive underflow.  In the
					// .    underflow case, try the two-small-subdiagonals
					// .    trick to try to reinflate the bulge. 
					// 
					if( h[(k+3)+k*ldh] != zero || h[(k+3)+(k+1)*ldh] != zero || h[(k+3)+(k+2)*ldh] == zero){
						// Typical case: not collapsed (yet).
						h[k+1+k*ldh] = beta;
						h[(k+2)+k*ldh] = zero;
						h[(k+3)+k*ldh] = zero;
					}else{
						// Atypical case: collapsed.  Attempt to
						// .    reintroduce ignoring H(K+1,K) and H(K+2,K).
						// .    If the fill resulting from the new
						// .    reflector is too large, then abandon it.
						// .    Otherwise, use the new one.
						complex_type vt[3];
						ComplexImplicitBulgeStart(3, &h[k+1+(k+1)*ldh], ldh, s[2*m], s[2*m+1], vt);
						complex_type alpha(vt[0]);
						Reflector::Generate(3, &alpha, &vt[1], 1, &vt[0]);
						complex_type refsum = std::conj(vt[0]) * (h[k+1+k*ldh] + std::conj(vt[1]) * h[(k+2)+k*ldh]);
						// 
						if(
							complex_traits::norm1(h[(k+2)+k*ldh]-refsum*vt[1]) +
							complex_traits::norm1(refsum*vt[2]) > ulp * (
							complex_traits::norm1(h[(k+0)+(k+0)*ldh]) +
							complex_traits::norm1(h[(k+1)+(k+1)*ldh]) +
							complex_traits::norm1(h[(k+2)+(k+2)*ldh]))
						){
							// Starting a new bulge here would
							// .    create non-negligible fill.  Use
							// .    the old one with trepidation.
							h[k+1+k*ldh] = beta;
							h[k+2+k*ldh] = zero;
							h[k+3+k*ldh] = zero;
						}else{
							// 
							// Stating a new bulge here would
							// .    create only negligible fill.
							// .    Replace the old reflector with
							// .    the new one.
							h[k+1+k*ldh] = h[k+1+(k)*ldh] - refsum;
							h[k+2+k*ldh] = zero;
							h[k+3+k*ldh] = zero;
							v[0+m*ldv] = vt[0];
							v[1+m*ldv] = vt[1];
							v[2+m*ldv] = vt[2];
						}
					}
				}
			}

			// Generate a 2-by-2 reflection, if needed.
			if(bmp22){
				int k = krcol + 3*m22;
				if(k+1 == ktop){
					ComplexImplicitBulgeStart(2, &h[k+1+(k+1)*ldh], ldh, s[2*m22], s[2*m22+1], &v[0+m22*ldv]);
					complex_type beta(v[0+m22*ldv]);
					Reflector::Generate(2, &beta, &v[1+m22*ldv], 1, &v[0+m22*ldv]);
				}else{
					complex_type beta(h[k+1+k*ldh]);
					v[1+m22*ldv] = h[(k+2)+k*ldh];
					Reflector::Generate(2, &beta, &v[1+m22*ldv], 1, &v[0+m22*ldv]);
					h[k+1+k*ldh] = beta;
					h[k+2+k*ldh] = zero;
				}
			}

			// Multiply H by reflections from the left
			int jbot;
			if(accum){
				jbot = min(ndcol, kbot);
			}else if(wantt){
				jbot = n;
			}else{
				jbot = kbot;
			}
			for(int j = max( ktop, krcol ); j < jbot; ++j){
				int mend = min( mbot, ( j-krcol+2 ) / 3 );
				for(int m = mtop; m < mend; ++m){
					int k = krcol + 3*m;
					complex_type refsum = std::conj(v[0+m*ldv])* ( h[k+1+j*ldh]+std::conj( v[1+m*ldv] )* h[(k+2)+j*ldh]+std::conj( v[2+m*ldv] )*h[(k+3)+j*ldh] );
					h[k+1+j*ldh] = h[k+1+j*ldh] - refsum;
					h[k+2+j*ldh] = h[k+2+j*ldh] - refsum*v[1+m*ldv];
					h[k+3+j*ldh] = h[k+3+j*ldh] - refsum*v[2+m*ldv];
				}
			}
			if(bmp22){
				int k = krcol + 3*m22;
				for(int j = max( k+1, ktop ); j < jbot; ++j){
					complex_type refsum = std::conj( v[0+m22*ldv] )* ( h[k+1+j*ldh]+std::conj( v[1+m22*ldv] )* h[k+2+j*ldh] );
					h[k+1+j*ldh] = h[k+1+j*ldh] - refsum;
					h[k+2+j*ldh] = h[k+2+j*ldh] - refsum*v[1+m22*ldv];
				}
			}

			// Multiply H by reflections from the right.
			// .    Delay filling in the last row until the
			// .    vigilant deflation check is complete.
			int jtop;
			if( accum ){
				jtop = max( ktop, incol );
			}else if( wantt ){
				jtop = 0;
			}else{
				jtop = ktop;
			}
			for(int m = mtop; m < mbot; ++m){
				if(zero != v[0+m*ldv]){
					int k = krcol + 3*m;
					for(int j = jtop; j < min( kbot, k+4 ); ++j){
						complex_type refsum = v[0+m*ldv]*( h[j+(k+1)*ldh]+v[1+m*ldv]* h[j+(k+2)*ldh]+v[2+m*ldv]*h[j+(k+3)*ldh] );
						h[j+(k+1)*ldh] = h[j+(k+1)*ldh] - refsum;
						h[j+(k+2)*ldh] = h[j+(k+2)*ldh] - refsum*std::conj( v[1+m*ldv] );
						h[j+(k+3)*ldh] = h[j+(k+3)*ldh] - refsum*std::conj( v[2+m*ldv] );
					}

					if( accum ){
						// Accumulate U. (If necessary, update Z later
						// .    with with an efficient matrix-matrix
						// .    multiply.)
						const int kms = k - incol;
						for(int j = max( 0, ktop-incol-1 ); j < kdu; ++j){
							complex_type refsum = v[0+m*ldv]*( u[j+kms*ldu]+v[1+m*ldv]* u[j+(kms+1)*ldu]+v[2+m*ldv]*u[j+(kms+2)*ldu] );
							u[j+(kms+0)*ldu] = u[j+(kms+0)*ldu] - refsum;
							u[j+(kms+1)*ldu] = u[j+(kms+1)*ldu] - refsum*std::conj( v[1+m*ldv] );
							u[j+(kms+2)*ldu] = u[j+(kms+2)*ldu] - refsum*std::conj( v[2+m*ldv] );
						}
					}else if( wantz ){
						// U is not accumulated, so update Z
						// .    now by multiplying by reflections
						// .    from the right.
						for(int j = iloz; j < ihiz; ++j){
							complex_type refsum = v[0+m*ldv]*( z[j+(k+1)*ldz]+v[1+m*ldv]* z[j+(k+2)*ldz]+v[2+m*ldv]*z[j+(k+3)*ldz] );
							z[j+(k+1)*ldz] = z[j+(k+1)*ldz] - refsum;
							z[j+(k+2)*ldz] = z[j+(k+2)*ldz] - refsum*std::conj( v[1+m*ldv] );
							z[j+(k+3)*ldz] = z[j+(k+3)*ldz] - refsum*std::conj( v[2+m*ldv] );
						}
					}
				}
			}

			// Special case: 2-by-2 reflection (if needed)
			if(bmp22 && (v[0+m22*ldv] != zero)){
				int k = krcol + 3*m22;
				for(int j = jtop; j < min( kbot, k+4 ); ++j){
					complex_type refsum = v[0+m22*ldv]*( h[j+(k+1)*ldh]+v[1+m22*ldv]* h[j+(k+2)*ldh] );
					h[j+(k+1)*ldh] = h[j+(k+1)*ldh] - refsum;
					h[j+(k+2)*ldh] = h[j+(k+2)*ldh] - refsum*std::conj( v[1+m22*ldv] );
				}

				if( accum ){
					const int kms = k - incol;
					for(int j = max( (int)0, ktop-incol-1 ); j < kdu; ++j){
						complex_type refsum = v[0+m22*ldv]*( u[j+kms*ldu]+v[1+m22*ldv]* u[j+(kms+1)*ldu] );
						u[j+(kms+0)*ldu] = u[j+(kms+0)*ldu] - refsum;
						u[j+(kms+1)*ldu] = u[j+(kms+1)*ldu] - refsum*std::conj( v[1+m22*ldv] );
					}
				}else if( wantz ){
					for(int j = iloz; j < ihiz; ++j){
						complex_type refsum = v[0+m22*ldv]*( z[j+(k+1)*ldz]+v[1+m22*ldv]* z[j+(k+2)*ldz] );
						z[j+(k+1)*ldz] = z[j+(k+1)*ldz] - refsum;
						z[j+(k+2)*ldz] = z[j+(k+2)*ldz] - refsum*std::conj( v[1+m22*ldv] );
					}
				}
			}

			// Vigilant deflation check
			int mstart = mtop;
			if(krcol+3*mstart < ktop){ mstart++; }
			int mend = mbot;
			if(bmp22){ mend++; }
			if(krcol+1 == kbot){ mend++; }
			
			for(int m = mstart; m < mend; ++m){
				int k = min( kbot-2, krcol+3*m);

				// The following convergence test requires that
				// .    the tradition small-compared-to-nearby-diagonals
				// .    criterion and the Ahues & Tisseur (LAWN 122, 1997)
				// .    criteria both be satisfied.  The latter improves
				// .    accuracy in some examples. Falling back on an
				// .    alternate convergence criterion when TST1 or TST2
				// .    is zero (as done here) is traditional but probably
				// .    unnecessary.
				if( h[k+1+k*ldh] != zero ){
					real_type tst1 = complex_traits::norm1( h[k+k*ldh] ) + complex_traits::norm1( h[k+1+(k+1)*ldh] );
					if( tst1 == rzero ){
						if( k >= ktop+1 ) tst1 += complex_traits::norm1( h[k+(k-1)*ldh] );
						if( k >= ktop+2 ) tst1 += complex_traits::norm1( h[k+(k-2)*ldh] );
						if( k >= ktop+3 ) tst1 += complex_traits::norm1( h[k+(k-3)*ldh] );
						if( k+3 <= kbot ) tst1 += complex_traits::norm1( h[(k+2)+(k+1)*ldh] );
						if( k+4 <= kbot ) tst1 += complex_traits::norm1( h[(k+3)+(k+1)*ldh] );
						if( k+5 <= kbot ) tst1 += complex_traits::norm1( h[(k+4)+(k+1)*ldh] );
					}
					if( complex_traits::norm1( h[k+1+k*ldh] ) <= max( smlnum, ulp*tst1 ) ){
						real_type h12 = complex_traits::norm1( h[k+1+k*ldh] );
						real_type h21 = complex_traits::norm1( h[k+(k+1)*ldh] );
						if(h21 > h12){ std::swap(h21, h12); }
						real_type h11 = complex_traits::norm1( h[k+1+(k+1)*ldh] );
						real_type h22 = complex_traits::norm1( h[k+k*ldh]-h[k+1+(k+1)*ldh] );
						if(h22 > h11){ std::swap(h22, h11); }
						real_type scl = h11 + h12;
						real_type tst2 = h22*( h11 / scl );
						// 
						if( tst2 == rzero  ||  h21*( h12 / scl ) <=  max( smlnum, ulp*tst2 ) ){ h[k+1+k*ldh] = zero; }
					}
				}
			}

			// Fill in the last row of each bulge.
			mend = min( nbmps, ( kbot-krcol-2 ) / 3 );
			for(int m = mtop; m < mend; ++m){
				int k = krcol + 3*m;
				complex_type refsum = v[0+m*ldv]*v[2+m*ldv]*h[(k+4)+(k+3)*ldh];
				h[(k+4)+(k+1)*ldh] = -refsum;
				h[(k+4)+(k+2)*ldh] = -refsum*std::conj( v[1+m*ldv] );
				h[(k+4)+(k+3)*ldh] -= refsum*std::conj( v[2+m*ldv] );
			}
			// End of near-the-diagonal bulge chase.
		}

		// Use U (if accumulated) to update far-from-diagonal
		// .    entries in H.  If required, use U to update Z as
		// .    well.
		if( accum ){
			int jbot, jtop;
			if( wantt ){
				jtop = 0;
				jbot = n;
			}else{
				jtop = ktop;
				jbot = kbot;
			}
			if( (  !blk22 )  ||  ( incol < ktop )  ||  ( ndcol > kbot )  ||  ( ns <= 2 ) ){
				// Updates not exploiting the 2-by-2 block
				// .    structure of U.  K1 and NU keep track of
				// .    the location and size of U in the special
				// .    cases of introducing bulges and chasing
				// .    bulges off the bottom.  In these special
				// .    cases and in case the number of shifts
				// .    is NS = 2, there is no 2-by-2 block
				// .    structure to exploit. 
				int k1 = max( (int)0, ktop-incol-1 );
				int nu = ( kdu-max( 0, ndcol-kbot ) ) - k1;
				// Horizontal Multiply
				for(int jcol = min( ndcol, kbot ); jcol < jbot; jcol += nh){
					int jlen = min( (int)nh, jbot-jcol );
					BLAS::MultMM("C","N", nu, jlen, nu, one, &u[k1+k1*ldu], ldu, &h[incol+k1+1+jcol*ldh], ldh, zero, wh, ldwh );
					BLAS::Copy(nu, jlen, wh, ldwh, &h[incol+k1+1+jcol*ldh], ldh );
				}
				// Vertical multiply
				for(int jrow = jtop; jrow < max( ktop, incol ); jrow += nv){
					int jlen = min( (int)nv, max( ktop, incol )-jrow );
					BLAS::MultMM("N","N", jlen, nu, nu, one, &h[jrow+(incol+k1+1)*ldh], ldh, &u[k1+k1*ldu], ldu, zero, wv, ldwv );
					BLAS::Copy(jlen, nu, wv, ldwv, &h[jrow+(incol+k1+1)*ldh], ldh );
				}
				// Z multiply (also vertical)
				if( wantz ){
					for(int jrow = iloz; jrow < ihiz; jrow += nv){
						int jlen = min( (int)nv, ihiz-jrow );
						BLAS::MultMM("N","N", jlen, nu, nu, one, &z[jrow+(incol+k1+1)*ldz], ldz, &u[k1+k1*ldu], ldu, zero, wv, ldwv );
						BLAS::Copy(jlen, nu, wv, ldwv, &z[(jrow)+(incol+k1+1)*ldz], ldz );
					}
				}
			}else{
				// Updates exploiting U's 2-by-2 block structure.
				// .    (I2, I4, J2, J4 are the last rows and columns
				// .    of the blocks.)
				int i2 = ( kdu+1 ) / 2;
				int i4 = kdu;
				int j2 = i4 - i2;
				int j4 = kdu;
				// KZS and KNZ deal with the band of zeros
				// .    along the diagonal of one of the triangular
				// .    blocks.
				int kzs = ( j4-j2 ) - ( ns+1 );
				int knz = ns + 1;
				// Horizontal multiply
				for(int jcol = min( ndcol, kbot ); jcol < jbot; jcol += nh){
					int jlen = min( (int)nh, jbot-jcol );
					// Copy bottom of H to top+KZS of scratch
					// (The first KZS rows get multiplied by zero.)
					BLAS::Copy(knz, jlen, &h[(incol+j2+1)+jcol*ldh], ldh, &wh[kzs+0*ldwh], ldwh );
					// Multiply by U21'
					BLAS::Set(kzs, jlen, zero, zero, wh, ldwh );
					BLAS::MultTrM("L","U","C","N", knz, jlen, one, &u[j2+kzs*ldu], ldu, &wh[kzs+0*ldwh], ldwh );
					// Multiply top of H by U11'
					BLAS::MultMM("C","N", i2, jlen, j2, one, u, ldu, &h[incol+1+jcol*ldh], ldh, one, wh, ldwh );
					// Copy top of H to bottom of WH
					BLAS::Copy(j2, jlen, &h[incol+1+jcol*ldh], ldh, &wh[i2+0*ldwh], ldwh );
					// Multiply by U21'
					BLAS::MultTrM("L","L","C","N", j2, jlen, one, &u[0+i2*ldu], ldu, &wh[i2+0*ldwh], ldwh );
					// Multiply by U22
					BLAS::MultMM("C","N", i4-i2, jlen, j4-j2, one, &u[j2+i2*ldu], ldu, &h[(incol+j2+1)+jcol*ldh], ldh, one, &wh[i2+0*ldwh], ldwh );
					// Copy it back
					BLAS::Copy(kdu, jlen, wh, ldwh, &h[incol+1+jcol*ldh], ldh );
				}
				// Vertical multiply
				for(int jrow = jtop; jrow < max( incol, ktop ); jrow += nv){
					int jlen = min( (int)nv, max( incol, ktop )-jrow );
					// Copy right of H to scratch (the first KZS columns get multiplied by zero)
					BLAS::Copy(jlen, knz, &h[jrow+(incol+j2+1)*ldh], ldh, &wv[0+kzs*ldwv], ldwv );
					// Multiply by U21
					BLAS::Set(jlen, kzs, zero, zero, wv, ldwv );
					BLAS::MultTrM("R","U","N","N", jlen, knz, one, &u[j2+kzs*ldu], ldu, &wv[0+kzs*ldwv], ldwv );
					// Multiply by U11
					BLAS::MultMM("N","N", jlen, i2, j2, one, &h[jrow+(incol+1)*ldh], ldh, u, ldu, one, wv, ldwv );
					// 
					// Copy left of H to right of scratch
					// 
					BLAS::Copy(jlen, j2, &h[jrow+(incol+1)*ldh], ldh, &wv[0+i2*ldwv], ldwv );
					// Multiply by U21
					BLAS::MultTrM("R","L","N","N", jlen, i4-i2, one, &u[0+i2*ldu], ldu, &wv[0+i2*ldwv], ldwv );
					// Multiply by U22
					BLAS::MultMM("N","N", jlen, i4-i2, j4-j2, one, &h[jrow+(incol+j2+1)*ldh], ldh, &u[j2+i2*ldu], ldu, one, &wv[0+i2*ldwv], ldwv );
					// Copy it back
					BLAS::Copy(jlen, kdu, wv, ldwv, &h[jrow+(incol+1)*ldh], ldh);
				}

				// Multiply Z (also vertical)
				if( wantz ){
					for(int jrow = iloz; jrow < ihiz; jrow += nv){
						int jlen = min( (int)nv, ihiz-jrow );

						// Copy right of Z to left of scratch (first
						// .     KZS columns get multiplied by zero)
						BLAS::Copy(jlen, knz, &z[jrow+(incol+j2+1)*ldz], ldz, &wv[0+kzs*ldwv], ldwv);
						// Multiply by U12
						BLAS::Set(jlen, kzs, zero, zero, wv, ldwv );
						BLAS::MultTrM("R","U","N","N", jlen, knz, one, &u[j2+kzs*ldu], ldu, &wv[0+kzs*ldwv], ldwv);
						// Multiply by U11
						BLAS::MultMM("N","N", jlen, i2, j2, one, &z[jrow+(incol+1)*ldz], ldz, u, ldu, one, wv, ldwv);
						// Copy left of Z to right of scratch
						BLAS::Copy(jlen, j2, &z[jrow+(incol+1)*ldz], ldz, &wv[0+i2*ldwv], ldwv);
						// Multiply by U21
						BLAS::MultTrM("R","L","N","N", jlen, i4-i2, one, &u[0+i2*ldu], ldu, &wv[0+i2*ldwv], ldwv);
						// Multiply by U22
						BLAS::MultMM("N","N", jlen, i4-i2, j4-j2, one, &z[jrow+(incol+j2+1)*ldz], ldz, &u[j2+(i2)*ldu], ldu, one, &wv[0+i2*ldwv], ldwv);
						// Copy the result back to Z
						BLAS::Copy(jlen, kdu, wv, ldwv, &z[jrow+(incol+1)*ldz], ldz);
					}
				}
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////
// ComplexSmallBulgeMultishiftQR
// -----------------------------
// Computes the eigenvalues of a Hessenberg matrix H and, optionally,
// the matrices T and Z from the Schur decomposition H = Z T Z^H,
// where T is an upper triangular matrix (the Schur form), and Z is
// the unitary matrix of Schur vectors.
//
// Optionally Z may be postmultiplied into an input unitary matrix Q
// so that this routine can give the Schur factorization of a matrix
// A which has been reduced to the Hessenberg form H by the unitary
// matrix Q: A = Q*H*Q^H = (QZ)*H*(QZ)^H.
//
// Only matrices of order greater than 11 may be passed to this routine.
// Smaller matrices should use the basic single-shift routine.
//
// This is equivalent to Lapack routine zlaqr0 and zlaqr4.
// Only a partial explanation of the arguments will be given here.
//
// Arguments
// level  The recursion level. Set to 0 on the highest level call, and
//        increase by one for each level. Currently it is assumed level
//        does not exceed 1. When level = 0, this is equivalent to
//        Lapack routine zlaqr0, otherwise it behaves as zlaqr4.
//
template <typename T>
int ComplexSmallBulgeMultishiftQR(int level,
	bool wantt, bool wantz, size_t n, size_t ilo, size_t ihi,
	std::complex<T> *h, size_t ldh, std::complex<T> *w,
	size_t iloz, size_t ihiz, std::complex<T> *z, size_t ldz,
	size_t *lwork, std::complex<T> *work
){
	typedef typename std::complex<T> complex_type;
	typedef                       T  real_type;
	typedef Traits<complex_type> complex_traits;
	typedef Traits<real_type>    real_traits;
	
	static const complex_type zero(real_type(0));
	static const complex_type one(real_type(1));
	static const real_type half(real_type(1)/real_type(2));
	static const real_type wilk1(real_type(3)/real_type(4));

	if(0 == n){
		return 0;
	}

	if(n < 12){ // Tiny matrices must use ZLAHQR.
		// Amswer workspace query
		if(0 == *lwork && NULL == work){ return 0; }
		// Forward the call to zlahqr
	    return ComplexImplicitSingleShiftQR(
			wantt, wantz, n, ilo, ihi, h, ldh, w, iloz, ihiz, z, ldz
		);
	}
	
	// NWR = recommended deflation window size.  At this
	// point, n >= 12, so there is enough
	// subdiagonal workspace for nwr >= 2 as required.
	// (In fact, there is enough subdiagonal space for
	// nwr >= 3.)
	size_t nwr = Tuning<T>::deflation_window_size(wantt, wantz, n, ilo, ihi, *lwork);
	if(nwr < 2){ nwr = 2; }
	if(ihi-ilo < nwr){ nwr = ihi-ilo; }
	if((n-1)/3 < nwr){ nwr = (n-1)/3; }

	// NSR = recommended number of simultaneous shifts.
	// At this point N .GT. NTINY = 11, so there is at
	// enough subdiagonal workspace for NSR to be even
	// and greater than or equal to two as required.
	size_t nsr = Tuning<T>::number_of_shifts(wantt, wantz, n, ilo, ihi, *lwork);
	if(ihi-ilo < nsr){ nsr = ihi-ilo; }
	if((n+6)/9 < nsr){ nsr = (n+6)/9; }
	nsr = (nsr - nsr%2 > 2 ? nsr - nsr%2 : 2);

	if(*lwork == 0){
		AggressiveEarlyDeflation(0,
			wantt, wantz, n, ilo, ihi, nwr+1, h, ldh, iloz,
			ihiz, z, ldz, NULL, NULL, w, h, 
			ldh, n, h, ldh, n, h, ldh, lwork, work
		);

		const size_t sweep_lwork = 3*nsr/2;
		if(sweep_lwork > *lwork){ *lwork = sweep_lwork; }
		return 0;
	}

	size_t nmin = Tuning<T>::multishift_crossover_size(wantt, wantz, n, ilo, ihi);
	if(nmin < 12){ nmin = 12; }
	size_t nibble = Tuning<T>::nibble_crossover_size(wantt, wantz, n, ilo, ihi, *lwork);

	// Accumulate reflections during ttswp?  Use block
	// 2-by-2 structure during matrix-matrix multiply?
	int kacc22 = Tuning<T>::matmul_type(wantt, wantz, n, ilo, ihi, *lwork);
	if(kacc22 < 0){ kacc22 = 0; }
	if(kacc22 > 2){ kacc22 = 2; }

	// NWMAX = the largest possible deflation window for
	// which there is sufficient workspace.
	size_t nwmax = (n-1)/3;
	if(*lwork / 2 < nwmax){ nwmax = *lwork/2; }
	size_t nw = nwmax;

	// NSMAX = the Largest number of simultaneous shifts for
	// which there is sufficient workspace.
	size_t nsmax = (n + 6) / 9;
	if((2 * (*lwork)) / 3 < nsmax){ nsmax = (2 * (*lwork)) / 3; }
	nsmax -= nsmax % 2;

	// NDFL: an iteration count restarted at deflation.
	size_t ndfl = 1;

	size_t itmax = (ihi - ilo) * 30;
	if(itmax < 10){ itmax = 10; }

	// Last row and column in the active block
	size_t kbot = ihi;
	
	int ndec = 0;
	
	for(size_t it = 0; it < itmax; ++it){ // Main Loop
		// Done when KBOT falls below ILO
	    if(kbot < ilo+1){
			return 0;
		}
		
		// Locate active block
		bool found_zero = false;
		size_t k;
		for(k = kbot-1; k > ilo; --k){
			if(zero == h[k+(k-1)*ldh]){
				found_zero = true;
				break;
			}
	    }
		if(!found_zero){ k = ilo; }
		size_t ktop = k;

		// Select deflation window size:
		// Typical Case:
		//   If possible and advisable, nibble the entire
		//   active block.  If not, use size MIN(NWR,NWMAX)
		//   or MIN(NWR+1,NWMAX) depending upon which has
		//   the smaller corresponding subdiagonal entry
		//   (a heuristic).
		//
		// Exceptional Case:
		//   If there have been no deflations in KEXNW (5) or
		//   more iterations, then vary the deflation window
		//   size.   At first, because, larger windows are,
		//   in general, more powerful than smaller ones,
		//   rapidly increase the window to the maximum possible.
		//   Then, gradually reduce the window size.

	    const size_t nh = kbot - ktop;
		const size_t nwupbd = (nh < nwmax ? nh : nwmax);
	    if(ndfl < 5){
			nw = (nwupbd < nwr ? nwupbd : nwr);
	    } else {
			nw = (nwupbd < 2*nw ? nwupbd : 2*nw);
	    }
	    if(nw < nwmax){
			if(nw+1 >= nh){
				nw = nh;
			}else{
				size_t kwtop = kbot - nw;
				if(complex_traits::norm1(h[kwtop+(kwtop-1)*ldh]) > complex_traits::norm1(h[kwtop-1+(kwtop-2)*ldh])){
					++nw;
				}
			}
	    }
	    if(ndfl < 5){
			ndec = -1;
	    } else if(ndec >= 0 || nw >= nwupbd){
			++ndec;
			if((int)nw < 2+ndec){
				ndec = 0;
			}
			if((int)nw <= ndec){
				nw = 0;
			}else{
				nw -= ndec;
			}
	    }

		// Aggressive early deflation:
		// split workspace under the subdiagonal into
		//   - an nw-by-nw work array V in the lower
		//     left-hand-corner,
		//   - an NW-by-at-least-NW-but-more-is-better
		//     (NW-by-NHO) horizontal work array along
		//     the bottom edge,
		//   - an at-least-NW-but-more-is-better (NHV-by-NW)
		//     vertical work array along the left-hand-edge.

		size_t kv = n - nw;
		size_t kt = nw;
		size_t nho = n - nw - kt;
		size_t kwv = nw + 1;
		size_t nve = n - nw - kwv;

		// Aggressive early deflation
		size_t ld, ls;
		AggressiveEarlyDeflation(level,
			wantt, wantz, n, ktop, kbot, nw,
			h, ldh, iloz, ihiz, z, ldz, &ls, &ld, w,
			&h[kv+0*ldh], ldh, nho, &h[kv + kt * ldh], ldh,
			nve, &h[kwv + 0*ldh], ldh, lwork, work
		);
		// Adjust KBOT accounting for new deflations.
	    kbot -= ld;

		// KS points to the shifts.
		size_t ks = kbot - ls;

		// Skip an expensive QR sweep if there is a (partly
		// heuristic) reason to expect that many eigenvalues
		// will deflate without it.  Here, the QR sweep is
		// skipped if many eigenvalues have just been deflated
		// or if the remaining active block is small.

		if(ld == 0 || ((ld * 100 <= nw * nibble) && (kbot - ktop > (nmin < nwmax ? nmin : nwmax)))){

			// NS = nominal number of simultaneous shifts.
			// This may be lowered (slightly) if AggressiveEarlyDeflation
			// did not provide that many shifts.
			size_t ns = kbot-ktop-1;
			if(ns < 2){ ns = 2; }
			if(nsmax < ns){ ns = nsmax; }
			if(nsr < ns){ ns = nsr; }
			ns -= ns % 2;

			// If there have been no deflations
			// in a multiple of KEXSH (6) iterations,
			// then try exceptional shifts.
			// Otherwise use shifts provided by
			// AggressiveEarlyDeflation above or from the eigenvalues
			// of a trailing principal submatrix.

			if(ndfl % 6 == 0){
				ks = kbot - ns;
				for(size_t i = kbot-1; i > ks; i -= 2){
					w[i] = h[i+i*ldh] + wilk1*complex_traits::norm1(h[i+(i-1)*ldh]);
					w[i-1] = w[i];
				}
			}else{

				// Got NS/2 or fewer shifts? Use multi-shift QR or
				// single-shift QR on a trailing principal submatrix to
				// get more. (Since NS.LE.NSMAX.LE.(N+6)/9,
				// there is enough space below the subdiagonal
				// to fit an NS-by-NS scratch array.)

				if(kbot - ks <= ns / 2){
					ks = kbot - ns;
					kt = n - ns;
					BLAS::Copy(ns, ns, &h[ks+ks*ldh], ldh, &h[kt+0*ldh], ldh);
					int inf;
					if(0 == level && ns > nmin){
						inf = ComplexSmallBulgeMultishiftQR(1, 0, 0, ns, 0, ns, &h[
							kt + 0*ldh], ldh, &w[ks], 0, 1, 
							(complex_type*)NULL, 1, lwork, work);
					} else {
						inf = ComplexImplicitSingleShiftQR(false, false, ns, 0, ns, &h[
							kt + 0*ldh], ldh, &w[ks], 0, 1, 
							(complex_type*)NULL, 0);
					}
					ks += inf;

					// In case of a rare QR failure use
					// eigenvalues of the trailing 2-by-2
					// principal submatrix.  Scale to avoid
					// overflows, underflows and subnormals.
					// (The scale factor S can not be zero,
					// because H(KBOT,KBOT-1) is nonzero.)

					if(ks+1 >= kbot){
						real_type s(
							  complex_traits::norm1(h[kbot-2+(kbot-2)*ldh])
							+ complex_traits::norm1(h[kbot-1+(kbot-2)*ldh])
							+ complex_traits::norm1(h[kbot-2+(kbot-1)*ldh])
							+ complex_traits::norm1(h[kbot-1+(kbot-1)*ldh])
						);
						complex_type aa = h[kbot-2+(kbot-2)*ldh] / s;
						complex_type cc = h[kbot-1+(kbot-2)*ldh] / s;
						complex_type bb = h[kbot-2+(kbot-1)*ldh] / s;
						complex_type dd = h[kbot-1+(kbot-1)*ldh] / s;
						complex_type tr2 = (aa+dd) * half;
						complex_type det = (aa-tr2)*(dd-tr2) - bb*cc;
						complex_type rtdisc = sqrt(-det);
						w[kbot-2] = (tr2+rtdisc)*s;
						w[kbot-1] = (tr2-rtdisc)*s;
						ks = kbot - 2;
					}
				}
			
				if(kbot - ks > ns){ // Sort the shifts (Helps a little)
					bool sorted = false;
					for(size_t k = kbot-1; k > ks; --k){
						if(sorted){ break; }
						sorted = true;

						for(size_t i = ks; i < k; ++i){
							if(complex_traits::norm1(w[i]) < complex_traits::norm1(w[i + 1])){
								sorted = false;
								complex_type swap = w[i];
								w[i] = w[i+1];
								w[i+1] = swap;
							}
						}
					}
				}
			}

			// If there are only two shifts, then use only one.
			if(kbot - ks == 2){
				if(complex_traits::norm1(w[kbot-1]-h[kbot-1+(kbot-1)*ldh]) < complex_traits::norm1(w[kbot-2]-h[kbot-1+(kbot-1)*ldh])){
					w[kbot-2] = w[kbot-1];
				}else{
					w[kbot-1] = w[kbot-2];
				}
			}

			// Use up to NS of the the smallest magnatiude
			// shifts.  If there aren't NS shifts available,
			// then use them all, possibly dropping one to
			// make the number of shifts even.
			if(kbot-ks < ns){ ns = kbot-ks; }
			ns -= ns % 2;
			ks = kbot - ns;

			// Small-bulge multi-shift QR sweep:
			// split workspace under the subdiagonal into
			// - a KDU-by-KDU work array U in the lower
			//   left-hand-corner,
			// - a KDU-by-at-least-KDU-but-more-is-better
			//   (KDU-by-NHo) horizontal work array WH along
			//   the bottom edge,
			// - and an at-least-KDU-but-more-is-better-by-KDU
			//   (NVE-by-KDU) vertical work WV arrow along
			//   the left-hand-edge.

			size_t kdu = ns * 3 - 3;
			size_t ku = n - kdu;
			size_t kwh = kdu;
			nho = n - kdu - 3 - kdu;
			kwv = kdu + 3;
			nve = n - kdu - kwv;


			// Small-bulge multi-shift QR sweep
			ComplexSmallBulgeMultishiftQRSweep(
				wantt, wantz, kacc22, n, ktop, kbot, ns, &w[ks], h, ldh,
				iloz, ihiz, z, ldz, work, 3, &h[ku+0*ldh], ldh, nve,
				&h[kwv+0*ldh], ldh, nho, &h[ku+kwh*ldh], ldh);
		}

		// Note progress (or the lack of it).
		if(ld > 0){
			ndfl = 1;
		} else {
			++ndfl;
		}
	}
	// Iteration limit exceeded. Return where the problem occurred.
    return kbot;
}



} // namespace NonsymmetricEigensystem


///////////////////////////////////////////////////////////////////////
// ComplexImplicitSingleShiftQR
// ----------------------------
// Computes the eigenvalues of a Hessenberg matrix H and, optionally,
// the matrices T and Z from the Schur decomposition H = Z T Z^H,
// where T is an upper triangular matrix (the Schur form), and Z is
// the unitary matrix of Schur vectors.
//
// Optionally Z may be postmultiplied into an input unitary matrix Q
// so that this routine can give the Schur factorization of a matrix
// A which has been reduced to the Hessenberg form H by the unitary
// matrix Q:  A = Q*H*Q^H = (QZ)*H*(QZ)^H.
//
// This is equivalent to Lapack routine _hseqr.
//
// Returns 0 on success. A return value greater than 0 indicates a
// convergence failure after reaching the iteration limit. The elements
// in the range info:ihi of w contain those eigenvalues which have been
// successfully computed. If info > 0 and job = "E", then the
// remaining unconverged eigenvalues are the eigenvalues of the upper
// Hessenberg matrix in the range ilo:info. If job = "S", then
// H0*U = U*H1 where H0 is the initial input Hessenberg matrix, and H1
// is the final value, where U is a unitary matrix. The final value is
// triangular in the range info:ihi. If info > 0 and compz = "V",
// then Z1 = Z0*U, regardless of the value of job. If info > 0 and
// compz = "I", then Z1 = U, regardless of the value of job.
//
// Arguments
// job   If job = "S", the full Schur form T is required.
//       If job = "E", only eigenvalues are computed.
// compz If compz = "N", no Schur vectors are computed. If compz = "I",
//       Z is initialized to the identity matrix and the matrix Z of
//       Schur vectors of H is returned. If compz = "V", Z should
//       contain the unitary matrix Q, and the product Q*Z is returned.
// n     The number of rows and columns of H.
// ilo   It is assumed the matrix H is already upper triangular outside
// ihi   the range ilo:ihi. Note that the back transformations are
//       applied to the full range 0:n if wantt is true.
// h     Pointer to the first element of H.
// ldh   Leading dimension of the array containing H, ldh >= n.
// w     Pointer to the array of eigenvalues. Only the range ilo:ihi
//       is touched.
// z     Pointer to the first element of the matrix of Schur vectors.
//       If a transformation was used to reduce a general matrix to
//       Hessenberg form, Z should contain the transformation matrix.
// ldz   Leading dimension of the array containing Z, ldz >= n.
// lwork Length of workspace (>= n). If *lwork == 0,
//       then the optimal size is returned in this argument.
// work  Workspace of size lwork.
//
template <typename T>
int HessenbergQR(
	const char *job, const char *compz,
	size_t n, size_t ilo, size_t ihi,
	T *h, size_t ldh,
	T *w,
	T *z, size_t ldz,
	size_t *lwork,
	T *work
){
	// Using nl = 49 allows up to six simultaneous shifts and a 16-by-16
	// deflation window.
	static const size_t nl = 49;

	const bool wantt = (job[0] == 'S');
	const bool initz = (compz[0] == 'I');
	const bool wantz = initz || (compz[0] == 'V');

	int info = 0;

	if(0 == n){
		return 0;
	}

	// copy eigenvalues isolated by balancing
	if(ilo > 0){
		BLAS::Copy(ilo, h, ldh+1, w, 1);
	}
	if(ihi < n){
		BLAS::Copy(n-ihi, &h[ihi+ihi*ldh], ldh+1, &w[ihi], 1);
	}

	// Initialize Z, if requested
	if(initz){
		BLAS::Set(n, n, T(0), T(1), z, ldz);
	}

	if(ilo+1 == ihi){
		w[ilo] = h[ilo+ilo*ldh];
		return 0;
	}
	
	const size_t nmin = NonsymmetricEigensystem::Tuning<T>::multishift_crossover_size(wantt, wantz, n, ilo, ihi);

	// small bulge multishift for big matrices; implicit single-shift for small ones
	if(n >= nmin){
		info = NonsymmetricEigensystem::ComplexSmallBulgeMultishiftQR(
			0, wantt, wantz, n, ilo, ihi, h, ldh, w, ilo, ihi, z, ldz, lwork, work
		);
	}else{ // Small matrix
		info = NonsymmetricEigensystem::ComplexImplicitSingleShiftQR(
			wantt, wantz, n, ilo, ihi, h, ldh, w, ilo, ihi, z, ldz
		);
		if(info > 0){
			// A rare single-shift failure! multi-shift QR sometimes succeeds when single-shift fails.
			size_t kbot = info;

			if(n >= nl){
				// Larger matrices have enough subdiagonal scratch
				// space to call multi-shift directly.
				info = NonsymmetricEigensystem::ComplexSmallBulgeMultishiftQR(
					0, wantt, wantz, n, ilo, kbot, h, ldh, w, ilo, ihi, z, ldz, lwork, work
				);
			}else{
				T hl[nl*nl]; // This is dangerously large to live on the stack
				T workl[nl];
				// Tiny matrices don't have enough subdiagonal
				// scratch space to benefit from multi-shift.  Hence,
				// tiny matrices must be copied into a larger
				// array before calling multi-shift.
				BLAS::Copy(n, n, h, ldh, hl, nl);
				hl[n+(n-1)*nl] = 0;
				BLAS::Set(nl, nl-n, T(0), T(0), &hl[0+n*nl], nl);
				info = NonsymmetricEigensystem::ComplexSmallBulgeMultishiftQR(
					0, wantt, wantz, nl, ilo, kbot, hl, nl, w, ilo, ihi, z, ldz, lwork, workl
				);
				if(wantt || info != 0){
					BLAS::Copy(n, n, hl, nl, h, ldh);
				}
			}
		}
	}

	// Clear out the trash, if necessary.
	if((wantt || info != 0) && n > 2){
		for(size_t j = 0; j+2 < n; ++j){
			for(size_t i = j; i+2 < n; ++i){
				h[2+i+j*ldh] = T(0);
			}
		}
	}

	return info;
}

///////////////////////////////////////////////////////////////////////
// ComplexEigensystem
// ------------------
// Computes for an N-by-N complex nonsymmetric matrix A, the
// eigenvalues and, optionally, the left and/or right eigenvectors.
//
// The right eigenvector v[j] of A satisfies
//
//     A * v[j] = lambda[j] * v[j]
//
// where lambda[j] is its eigenvalue.
// The left eigenvector u[j] of A satisfies
//
//     u[j]^H * A = lambda[j] * u[j]^H
//
// where u[j]^H denotes the conjugate transpose of u[j].
// The eigenvectors are not normalized.
//
// On success, returns 0. If the i-th argument is invalid, returns -i.
// A positive return value indicates that the QR algorithm failed to
// compute all the eigenvalues, and no eigenvectors have been computed;
// elements info:n of w contain eigenvalues which have converged.
//
// Arguments
// n     The number of rows and columns of A.
// a     Pointer to the first element of A. The matrix is destroyed
//       on exit.
// lda   Leading dimension of the array containing A, lda >= n.
// w     Array of length n containing the computed eigenvalues.
// vl    Pointer to the first element of the array containing the
//       left eigenvectors (in columns). If not NULL, the left
//       eigenvectors u[j] are stored one after another in the columns
//       of vl, in the same order as their eigenvalues.
// ldvl  Leading dimension of the array containing vl, ldvl >= n when
//       vl is non-NULL.
// vr    Pointer to the first element of the array containing the
//       right eigenvectors (in columns). If not NULL, the right
//       eigenvectors v[j] are stored one after another in the columns
//       of vr, in the same order as their eigenvalues.
// ldvr  Leading dimension of the array containing vr, ldvr >= n when
//       vr is non-NULL.
// lwork Length of workspace (>= 3*n). If *lwork == 0,
//       then the optimal size is returned in this argument.
// work  Workspace of size lwork.
// iwork Integer workspace of length n.
//
template <typename T>
int ComplexEigensystem(
	size_t n, 
	std::complex<T> *a, size_t lda,
	std::complex<T> *w,
	std::complex<T> *vl, size_t ldvl,
	std::complex<T> *vr, size_t ldvr,
	size_t *lwork, std::complex<T> *work, int *iwork
){
	typedef std::complex<T> complex_type;
	typedef T real_type;

	if(0 == n){
		return 0;
	}
	
	const bool wantvl = (vl != NULL);
	const bool wantvr = (vr != NULL);

	// Workspace:
	//  [ tau        | hrd_work    ] : Hessenberg::Reduce
	//     n            var (min: 0)
	//  [ tau        | genQ_work   ] : Hessenberg::GenerateQ
	//     n            var (min: n)
	//  [ qr_work                  ] : HessenbergQR
	//     var (min: n)
	//  [ trvec_work | trvec_rwork ] : Triangular::Eigenvectors
	//      2*n         n (real)
	// The trvec_rwork can simply cast a complex workspace. Therefore
	// we will impose a minimum workspace size of 3*n, iwork needs to be
	// at least n.
	if(0 == *lwork){
		// We will assume HessenbergQR's optimal block size is dwarfed
		// by the Hessenberg reduction and GenerateQ routines.
		// Note that this can vastly overestimate since ihi-ilo may be
		// much less than n, but we don't know ilo,ihi yet.
		size_t qwork = 0;
		complex_type dummy;
		Hessenberg::Reduce(n, 0, n, a, lda, (complex_type*)NULL, &qwork, &dummy);
		if(qwork > *lwork){ *lwork = qwork; }
		Hessenberg::GenerateQ(n, 0, n, a, lda, (complex_type*)NULL, &qwork, (complex_type*)NULL);
		if(qwork > *lwork){ *lwork = qwork; }
		*lwork += n;
		if(*lwork < 3*n){ *lwork = 3*n; }
		return 0;
	}
	
	size_t lxwork = *lwork - n;
	complex_type *xwork = work + n;
	
	RNPAssert(*lwork >= 3*n);
	
	const real_type eps = real_type(2)*Traits<real_type>::eps();
	const real_type smlnum = sqrt(Traits<real_type>::min()) / eps;
	const real_type bignum = real_type(1) / smlnum;

	// Scale A if max element outside range [SMLNUM,BIGNUM]
	real_type anrm(MatrixNorm("M", n, n, a, lda));
	bool scalea = false;
	real_type cscale(1);
	if(anrm > real_type(0) && anrm < smlnum){
		scalea = true;
		cscale = smlnum;
	}else if(anrm > bignum){
		scalea = true;
		cscale = bignum;
	}
	if(scalea){
		BLAS::Rescale("G", 0, 0, anrm, cscale, n, n, a, lda);
	}

	size_t ilo, ihi;
	NonsymmetricEigensystem::Balance("B", n, a, lda, &ilo, &ihi, iwork);

	complex_type *tau = work;
	Hessenberg::Reduce(n, ilo, ihi, a, lda, tau, &lxwork, xwork);

	int info = 0;
	if(wantvl){
		BLAS::Copy(n, n, a, lda, vl, ldvl);
		Hessenberg::GenerateQ(n, ilo, ihi, vl, ldvl, tau, &lxwork, xwork);

		info = HessenbergQR("S", "V", n, ilo, ihi, a, lda, w, vl, ldvl, lwork, work);

		if(wantvr){ // Copy Schur vectors to vr
			BLAS::Copy(n, n, vl, ldvl, vr, ldvr);
		}
	}else if(wantvr){
		// Copy Householder vectors to vr
		BLAS::Copy(n, n, a, lda, vr, ldvr);
		Hessenberg::GenerateQ(n, ilo, ihi, vr, ldvr, tau, &lxwork, xwork);

		info = HessenbergQR("S", "V", n, ilo, ihi, a, lda, w, vr, ldvr, lwork, work);
	}else{ // only eigenvalues
		HessenbergQR("E", "N", n, ilo, ihi, a, lda, w, (std::complex<T>*)NULL, 0, lwork, work);
	}

	if(0 == info){
		if(wantvl || wantvr){
			real_type *rwork = reinterpret_cast<real_type*>(work + 2*n);
			Triangular::Eigenvectors("B", NULL, n, a, lda, vl, ldvl, vr, ldvr, work, rwork);
		}

		if(wantvl){
			NonsymmetricEigensystem::BalanceUndo("B", "L", n, ilo, ihi, iwork, n, vl, ldvl);
		}

		if(wantvr){
			NonsymmetricEigensystem::BalanceUndo("B", "R", n, ilo, ihi, iwork, n, vr, ldvr);
		}
	}
	
	// Undo scaling if necessary
	if(scalea){
		BLAS::Rescale("G", 0, 0, cscale, anrm, n-info, 1, &w[info], n-info);
		if(info > 0){
			BLAS::Rescale("G", 0, 0, cscale, anrm, ilo-1, 1, w, n);
		}
	}
	
	return info;
}

} // namespace LA
} // namespace RNP

#endif // RNP_COMPLEX_EIGENSYSTEM_HPP_INCLUDED
