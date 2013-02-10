#ifndef RNP_EIGENSYSTEMS_HPP_INCLUDED
#define RNP_EIGENSYSTEMS_HPP_INCLUDED

#include <RNP/BLAS.hpp>
#include <RNP/LA/Tridiagonal.hpp>
#include <RNP/LA/MatrixNorms.hpp>
#include <RNP/LA/HessenbergQR.hpp>

namespace RNP{
namespace LA{

///////////////////////////////////////////////////////////////////////
// Eigensystems
// ============
// Computes various eigenvalue decompositions, which are basically
// similarity transformations which bring a square matrix to diagonal
// form.


///////////////////////////////////////////////////////////////////////
// HermitianEigensystem
// --------------------
// Computes all eigenvalues and, optionally, eigenvectors of a
// Hermitian matrix A. This routine works for complex Hermitian or real
// symmetric matrices. The right eigenvectors are returned as columns
// of the same matrix A.
//
// Equivalent to Lapack routines zheev/cheev and dsyev/ssyev, except
// eigenvalues are not sorted.
//
// Arguments
// job   If "N", only eigenvalues are computed. If "V" the
//       eigenvectors are returned in the columns of a.
// uplo  If "U", the upper triangle of A is given.
//       If "L", the lower triangle of A is given.
// n     Number of rows and columns of the matrix A.
// a     Pointer to the first element of A. On exit, if job = "V",
//       this is overwritten by the matrix whose columns are the
//       eigenvectors corresponding to the eigenvalues in w.
// lda   Leading dimension of the array containing A (lda >= n).
// w     Array of returned eigenvalues (length n).
// lwork Length of workspace. If *lwork == 0 then the optimal size
//       is returned in this argument. lwork must be at least 2*(n-1)
//       for complex matrices, and 3*(n-1) for real matrices.
// work  Workspace of size lwork, or NULL for workspace query.
//
template <typename T>
int HermitianEigensystem(
	const char *job, const char *uplo, size_t n, T *a, size_t lda,
	typename Traits<T>::real_type *w,
	size_t *lwork,
	T *work
){
	typedef typename Traits<T>::real_type real_type;
	
	RNPAssert('V' == job[0] || 'N' == job[0]);
	RNPAssert('U' == uplo[0] || 'L' == uplo[0]);
	RNPAssert(NULL != a);
	RNPAssert(lda >= n);
	RNPAssert(NULL != lwork);
	
	static const bool is_real = !Traits<T>::is_complex();
	
	RNPAssert(0 == *lwork || *lwork >= 2*(n-1) + (is_real ? n-1 : 0));
	
	if(0 == n){ return 0; }
	if(1 == n){
		w[0] = Traits<T>::real(a[0]);
		if('V' == job[0]){
			a[0] = T(1);
		}
		return 0;
	}
	
	// Workspace
	//  Real symmetric:
	//   [ offdiag | tau | reduce_work ] : Reduction phase
	//       n-1     n-1    variable
	//   [ offdiag | qr_work           ] : QR phase
	//       n-1     2*(n-1)
	//  Complex symmetric:
	//   [ offdiag | tau | reduce_work ] : Reduction phase
	//     n-1 (r)   n-1    variable
	//   [ offdiag | qr_work           ]
	//     n-1 (r)   2*(n-1) (r)
	//  We assume that ReduceHerm and GenerateQHerm can use the same
	//  blocked workspace. The QR iteration workspace can overwrite tau
	//  since it is no longer needed at that stage.
	//   Therefore we see that 2*(n-1)+var is enough, where var >= n-1
	//  in the real case.
	//  The only assumption here is that
	//     sizeof(complex) >= 2*sizeof(real)
	
	// Determine lwork
	if(0 == *lwork || NULL == work){
		// Tridiagonal reduction needs a workspace, and
		// symmetric QR needs a workspace.
		T dummy; // ReduceHerm's work can't be NULL for a workspace query.
		Tridiagonal::ReduceHerm(uplo, n, a, lda, w, (real_type*)NULL, (T*)NULL, lwork, &dummy);
		if(is_real && *lwork < n-1){ *lwork = (n-1); }
		*lwork += 2*(n-1);
		return 0;
	}
	
	// Set workspace
	real_type *offdiag;
	T *tau;
	T *reduce_work;
	size_t reduce_lwork;
	real_type *qr_work;
	if(is_real){ // real symmetric
		// Since T and real_type should be the same, these casts should be ok.
		offdiag = reinterpret_cast<real_type*>(work);
		tau = reinterpret_cast<T*>(offdiag + (n-1));
		reduce_work = tau + (n-1);
		reduce_lwork = *lwork - 2*(n-1);
		qr_work = reinterpret_cast<real_type*>(tau); // This will run beyond tau into reduce_work, which is at least n-1.
	}else{ // complex hermitian
		offdiag = reinterpret_cast<real_type*>(work);
		tau = work + (n-1);
		reduce_work = tau + (n-1);
		reduce_lwork = *lwork - 2*(n-1);
		qr_work = reinterpret_cast<real_type*>(tau);
	}
	
	const real_type safmin(Traits<real_type>::min());
	const real_type eps(real_type(2)*Traits<real_type>::eps());
	const real_type smlnum(safmin / eps);
	const real_type bignum(real_type(1) / smlnum);
	const real_type rmin(sqrt(smlnum));
	const real_type rmax(sqrt(bignum));
	
	const real_type anrm = MatrixNormHerm("M", uplo, n, a, lda);
	bool scaled = false;
	real_type sigma(1);
	if(anrm > 0 && anrm < rmin){
		scaled = true;
		sigma = rmin/anrm;
	}else if(anrm > rmax){
		scaled = true;
		sigma = rmax/anrm;
	}
	if(scaled){
		BLAS::Rescale(uplo, 0, 0, real_type(1), sigma, n, n, a, lda);
	}
	
	Tridiagonal::ReduceHerm(uplo, n, a, lda, w, offdiag, tau, &reduce_lwork, reduce_work);
	
	int info;
	if('V' == job[0]){
		Tridiagonal::GenerateQHerm(uplo, n, a, lda, tau, &reduce_lwork, reduce_work);
		info = Tridiagonal::Util::SymmetricEigensystem(
			n, w, offdiag, a, lda, qr_work
		);
	}else{
		info = Tridiagonal::Util::SymmetricEigensystem(
			n, w, offdiag, (real_type*)NULL, lda, (real_type*)NULL
		);
	}
	
	if(scaled){
		size_t wlim = n;
		if(0 != info){
			wlim = info - 1;
		}
		BLAS::Scale(wlim, real_type(1)/sigma, w, 1);
	}
	return info;
}


namespace NonsymmetricEigensystem{

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
	static const size_t max_iters = 32;
	
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

} // namespace NonsymmetricEigensystem

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

		info = HessenbergQR::SchurReduce("S", "V", n, ilo, ihi, a, lda, w, vl, ldvl, lwork, work);

		if(wantvr){ // Copy Schur vectors to vr
			BLAS::Copy(n, n, vl, ldvl, vr, ldvr);
		}
	}else if(wantvr){
		// Copy Householder vectors to vr
		BLAS::Copy(n, n, a, lda, vr, ldvr);
		Hessenberg::GenerateQ(n, ilo, ihi, vr, ldvr, tau, &lxwork, xwork);

		info = HessenbergQR::SchurReduce("S", "V", n, ilo, ihi, a, lda, w, vr, ldvr, lwork, work);
	}else{ // only eigenvalues
		HessenbergQR::SchurReduce("E", "N", n, ilo, ihi, a, lda, w, (std::complex<T>*)NULL, 0, lwork, work);
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

#endif // RNP_EIGENSYSTEMS_HPP_INCLUDED
