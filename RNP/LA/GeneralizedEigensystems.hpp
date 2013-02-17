#ifndef RNP_GENERALIZED_EIGENSYSTEMS_HPP_INCLUDED
#define RNP_GENERALIZED_EIGENSYSTEMS_HPP_INCLUDED

#include <RNP/LA/MatrixNorms.hpp>
#include <RNP/LA/Triangular.hpp>
#include <RNP/LA/HessenbergQZ.hpp>

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

} // namespace NonsymmetricGeneralizedEigensystem


///////////////////////////////////////////////////////////////////////
// ComplexGeneralizedEigensystem
// -----------------------------
// Computes for a pair of n-by-n complex nonsymmetric matrices (A,B),
// the generalized eigenvalues, and optionally, the left and/or right
// generalized eigenvectors.
//
// A generalized eigenvalue for a pair of matrices (A,B) is a scalar
// lambda or a ratio alpha/beta = lambda, such that A - lambda*B is
// singular. It is usually represented as the pair (alpha,beta), as
// there is a reasonable interpretation for beta=0, and even for both
// being zero.
//
// The right generalized eigenvector v[j] corresponding to the
// generalized eigenvalue lambda(j) of (A,B) satisfies
//              A * v[j] = lambda[j] * B * v[j].
// The left generalized eigenvector u[j] corresponding to the
// generalized eigenvalues lambda(j) of (A,B) satisfies
//              u[j]^H * A = lambda[j] * u[j]^H * B
// where u[j]^H is the conjugate-transpose of u[j].
//
// Returns 0 on a successful exit. A negative return value of -i means
// the i-th argument had an illegal value. If the return value is
// positive in the range i=1:n+1, then the QZ iteration failed.  No
// eigenvectors have been calculated, but alpha[j] and beta[j] should
// be correct for j=i:n. Any other value signifies some other QZ
// iteration failure.
//
// Arguments
// n     The number of rows and columns of A and B.
// a     Pointer to the first element of A.
//       On exit, A has been overwritten.
// lda   The leading dimension of the array containing A, lda >= n.
// b     Pointer to the first element of B.
//       On exit, B has been overwritten.
// ldb   The leading dimension of the array containing B, ldb >= n.
// alpha On exit, alpha[j]/beta[j], j=0:n will be the generalized
// beta  eigenvalues.
//       Note: the quotients alpha[j]/beta[j] may easily over- or
//       underflow, and beta[j] may even be zero.  Thus, the user
//       should avoid naively computing the ratio alpha/beta.
//       However, alpha will be always less than and usually
//       comparable with norm(A) in magnitude, and beta always less
//       than and usually comparable with norm(B).
//       Beta is guaranteed to be non-negative.
// vl    If vl != NULL, the left generalized eigenvectors u[j] are
//       stored one after another in the columns of vl, in the same
//       order as their eigenvalues. They are not normalized.
// ldvl  The leading dimension of the array containing vl.
//       If vl != NULL, ldvl >= n.
// vr    If vr != NULL, the right generalized eigenvectors v[j] are
//       stored one after another in the columns of vr, in the same
//       order as their eigenvalues. They are not normalized.
// ldvr  The leading dimension of the array containing vr.
//       If vr != NULL, ldvr >= n.
// lwork Length of workspace work. At least 3*n if any eigenvectors
//       are desired. Otherwise, 2*n is the minimum. Set to zero to
//       query the optimal size, returned in lwork.
// work  Workspace of size lwork.
// iwork Integer workspace of size 2*n.
//
template <typename T>
int ComplexGeneralizedEigensystem(size_t n, 
	std::complex<T> *a, size_t lda, std::complex<T> *b, size_t ldb, 
	std::complex<T> *alpha, T *beta,
	std::complex<T> *vl, size_t ldvl, std::complex<T> *vr, size_t ldvr,
	size_t *lwork, std::complex<T> *work, int *iwork
){
	typedef std::complex<T> complex_type;
	typedef T real_type;
	
	static const complex_type one(real_type(1));
	static const complex_type zero(real_type(0));
	static const char *balancejob = "P";
	
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
	// If only eigenvalues are wanted, then 2n is the minimum.
	
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

	const real_type eps = real_type(2) * Traits<real_type>::eps();
	const real_type smlnum = sqrt(Traits<real_type>::min()) / eps;
	const real_type bignum = real_type(1) / smlnum;

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
	int info;
	//zgghrd_(NULL != vl ? "V" : "N", NULL != vr ? "V" : "N", n, ilo+1, ihi, a, lda, b, ldb, vl, (1 > ldvl ? 1 : ldvl), vr, (1 > ldvr ? 1 : ldvr), &info);

	int ierr = HessenbergQZ::GeneralizedSchurReduce(
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
