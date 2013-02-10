#ifndef RNP_IMPLICITLY_RESTARTED_ARNOLDI_HPP_INCLUDED
#define RNP_IMPLICITLY_RESTARTED_ARNOLDI_HPP_INCLUDED

#include <RNP/Types.hpp>
#include <RNP/Random.hpp>
#include <RNP/BLAS.hpp>
#include <RNP/LA.hpp>
#include <algorithm>

namespace RNP{
namespace IterativeEigensystems{

template <typename T, typename Allocator>
class ComplexEigensystemImpl{
protected:
	typedef std::complex<T> complex_type;
	typedef T real_type;
	
	ComplexEigensystem *parent;
	
	size_t n, n_wanted, n_arnoldi;
	complex_type *workd;           // length 3*n
	complex_type *workl;           // length (3*n_arnoldi + 5)*n_arnoldi.
	complex_type *workev;          // length 2*n_arnoldi;
	complex_type *resid;           // length n
	complex_type *v;               // length n
	real_type *rwork;   // length n_arnoldi
	int *iwork;         // length n_arnoldi (really just booleans)
	Allocator real_allocator;
	typename Allocator::template rebind<complex_type>::other complex_allocator;
	typename Allocator::template rebind<int>::other int_allocator;
	
	bool shift_invert;
	complex_type shift;
	size_t maxiter;
	real_type tol;
	bool bsubspace;
	bool resid0_provided;
	
	bool eigvals_solved, eigvecs_solved;
	
	size_t nconv;
	
	static const size_t nb = 1; // block size
public:
	ComplexEigensystemImpl(ComplexEigensystem *sup,
		size_t nn, size_t nwanted,
		size_t narnoldi, // n_wanted+1 <= n_arnoldi <= n
		bool shift_invert_mode, const complex_type &shift_value,
		const ComplexEigensystem::Params &params,
		complex_type *resid0
	):parent(sup),
		n(nn), n_wanted(nwanted), n_arnoldi(narnoldi),
		shift_invert(shift_invert_mode), shift(shift_value),
		maxiter(params.max_iterations), tol(params.tol),
		bsubspace(params.no_eigenvectors),
		eigvals_solved(false), eigvecs_solved(false),
		nconv(0)
	{
		v      = complex_allocator.allocate(n*n_arnoldi);
		workd  = complex_allocator.allocate(3*n);
		workl  = complex_allocator.allocate((3*n_arnoldi+5)*n_arnoldi);
		workev = complex_allocator.allocate(2*n_arnoldi);
		rwork  = real_allocator.allocate(n_arnoldi);
		iwork  = int_allocator.allocate(n_arnoldi);
		if(NULL != resid0){
			resid0_provided = true;
			resid = resid0;
		}else{
			resid0_provided = false;
			resid  = complex_allocator.allocate(n);
		}
		if(tol <= real_type(0)){
			tol = Traits<real_type>::eps();
		}
	}
	~ComplexEigensystemImpl(){
		complex_allocator.deallocate(v, n*n_arnoldi);
		int_allocator.deallocate(iwork, n_arnoldi);
		real_allocator.deallocate(rwork, n_arnoldi);
		complex_allocator.deallocate(workev, 2*n_arnoldi);
		complex_allocator.deallocate(workl , (3*n_arnoldi+5)*n_arnoldi);
		complex_allocator.deallocate(workd , 3*n);
		if(!resid0_provided){
			complex_allocator.deallocate(resid , n);
		}
	}
	
protected:
	int RunArnoldi(){
		const size_t ncv = n_arnoldi;
		const size_t nev = n_wanted;
		// np is the number of additional steps to extend the
		// length nev Lanczos factorization.
		// nev0 is the local variable designating the size of the
		// invariant subspace desired.
		const size_t np = ncv - nev;
		const size_t nev0 = nev;
			
		// Zero out internal workspace
		BLAS::Set(ncv*(ncv*3 + 5), workl, 1);
		
		// workl is split up as follows (sequentially):
		//   Length        Purpose
		//   -------       -----------------------
		//   ncv*ncv       The Hessenberg matrix H
		//   ncv           Vector of Ritz values
		//   ncv           Vector of error bounds
		//   ncv*ncv       Rotation matrix Q
		//   ncv*(ncv + 3) Workspace for zneigh
		
		complex_type *H       = workl;
		complex_type *ritz    = H      + ncv*ncv;
		complex_type *bounds  = ritz   + ncv;
		complex_type *Q       = bounds + ncv;
		complex_type *w       = Q      + ncv*ncv;
		
		// kplusp is the bound on the largest
		//        Lanczos factorization built.
		// nconv is the current number of
		//        "converged" eigenvalues.
		// iter is the counter on the current
		//      iteration step.
		const size_t kplusp = nev0 + np;
		nconv = 0;
		size_t iter = 0;
		int info;
		
		const real_type eps23 = pow(Traits<real_type>::eps(), real_type(2)/real_type(3));
		
		info = GetInitialVector(v, resid, &rnorm, workd);
	}
	void ComputeEigenvectors(){
	}
	
	// If initv is true, then the initial vector is already in resid
	// j is the index of the residual vector to be generated. (j > 0 in case of a restart)
	// On exit, workd[0..n] contains B*resid
	// Returns B-norm of generated residual
	real_type GenerateInitialVector(bool initv, size_t j, real_type *rnorm){
		size_t iter = 0;
		
		if(!initv){ // Generate random starting vector if not given already
			Random::GenerateVector(Random::Distribution::Uniform_11, n, resid);
		}
		
		real_type rnorm0;
		if(!parent->IsBIdentity()){
			// Force the starting vector into the range of OP to handle
			// the generalized problem when B is possibly (singular).
			// We want to compute workd[n..2n] = OP * resid
			if(shift_invert){
				if(parent->IsBInPlace()){
					if(parent->IsAInPlace()){
						RNP::TBLAS::Copy(n, resid, 1, &workd[n], 1);
						parent->ApplyB(n, NULL, &workd[n], Bdata);
						parent->ApplyA(n, shift, NULL, &workd[n], Adata);
					}else{
						RNP::TBLAS::Copy(n, resid, 1, &workd[0], 1);
						parent->ApplyB(n, NULL, &workd[0], Bdata);
						parent->ApplyA(n, shift, &workd[0], &workd[n], Adata);
					}
				}else{
					if(parent->IsAInPlace()){
						parent->ApplyB(n, resid, &workd[n], Bdata);
						parent->ApplyA(n, shift, NULL, &workd[n], Adata);
					}else{
						parent->ApplyB(n, resid, &workd[0], Bdata);
						parent->ApplyA(n, shift, &workd[0], &workd[n], Adata);
					}
				}
			}else{
				RNP::TBLAS::Copy(n, resid, 1, workd, 1);
				parent->ApplyA(n, 0., &workd[ipntr[0]], &workd[ipntr[1]], Adata);
			}
			// Starting vector is now in the range of OP; r = OP*r;
			// Compute B-norm of starting vector.
			RNP::TBLAS::Copy(n, &workd[n], 1, resid, 1);
			complex_type cnorm;
			if(parent->IsBInPlace()){
				parent->ApplyB(n, NULL, &workd[n], Bdata);
				cnorm = BLAS::ConjugateDot(n, resid, 1, &workd[n], 1);
			}else{
				parent->ApplyB(n, &workd[n], &workd[0], Bdata);
				cnorm = BLAS::ConjugateDot(n, resid, 1, workd, 1);
			}
			rnorm0 = sqrt(Traits<complex_type>::abs(cnorm));
		}else{
			RNP::TBLAS::Copy(n, resid, 1, workd, 1);
			rnorm0 = RNP::TBLAS::Norm2(n, resid, 1);
		}
		*rnorm = rnorm0;
		// At this point, resid needs to have the starting vector, and
		// workd needs to have a copy of resid
		
		// Exit if this is the very first Arnoldi step
		if(j > 0){
			// Otherwise need to B-orthogonalize the starting vector against
			// the current Arnoldi basis using Gram-Schmidt with iter. ref.
			// This is the case where an invariant subspace is encountered
			// in the middle of the Arnoldi factorization.
			//
			//       s = V^{T}*B*r;   r = r - V*s;
			//
			// Stopping criteria used for iter. ref. is discussed in
			// Parlett's book, page 107 and in Gragg & Reichel TOMS paper.
			do{
				BLAS::MultMV("C", n, j, real_type( 1), v, ldv, &workd[0], 1, real_type(0), &workd[n], 1);
				BLAS::MultMV("N", n, j, real_type(-1), v, ldv, &workd[n], 1, real_type(1), resid, 1);

				// Compute the B-norm of the orthogonalized starting vector
				if(!IsBIdentity()){
					complex_type cnorm;
					if(parent->IsBInPlace()){
						BLAS::Copy(n, resid, 1, &workd[0], 1);
						parent->ApplyB(n, NULL, &workd[0], Bdata);
						cnorm = RNP::TBLAS::ConjugateDot(n, resid, 1, workd, 1);
					}else{
						BLAS::Copy(n, resid, 1, &workd[n], 1);
						parent->ApplyB(n, &workd[n], &workd[0], Bdata);
						cnorm = RNP::TBLAS::ConjugateDot(n, resid, 1, workd, 1);
					}
					*rnorm = sqrt(std::abs(cnorm));
				}else{
					BLAS::Copy(n, resid, 1, workd, 1);
					*rnorm = RNP::TBLAS::Norm2(n, resid, 1);
				}

				// Check for further orthogonalization.
				if(rnorm > rnorm0 * .717f){
					break;
				}

				++iter;
				if(iter > 1){
					// Iterative refinement step "failed"
					for(size_t jj = 0; jj < n; ++jj){
						resid[jj] = 0;
					}
					*rnorm = 0.;
					return -1;
				}
				// Perform iterative refinement step
				rnorm0 = *rnorm;
			}while(1);
		}
		return 0;
	}
	
	void SortRitzKernel(size_t n, complex_type *x, complex_type *y, bool reverse, bool sm){
		size_t igap = n / 2;
		while(igap != 0){
			for(size_t i = igap; i < n; ++i){
				int j = i - igap;
				while(j >= 0){
					if(j < 0){ break; }
					bool cmpres;
					if(sm){
						cmpres = (Traits<complex_t>::abs(x[j]) > Traits<complex_t>::abs(x[j+igap]));
					}else{
						cmpres = parent->EigenvalueCompare(x[j], x[j+igap])
					}
					if(reverse == cmpres){
						std::swap(x[j], x[j+igap]);
						if(NULL != y){
							std::swap(y[j], y[j+igap]);
						}
					}else{ break; }
					j -= igap;
				}
			}
			igap /= 2;
		}
	}
	// Given the eigenvalues of the upper Hessenberg matrix H,
	// computes the NP shifts AMU that are zeros of the polynomial of
	// degree NP which filters out components of the unwanted eigenvectors
	// corresponding to the AMU's based on some given criteria.
	//
	// NOTE: call this even in the case of user specified shifts in order
	// to sort the eigenvalues, and error bounds of H for later use.
	// NOTE: Complex conjugate pairs are not kept together
	void SortRitz(size_t kev, size_t np, complex_type *ritz, complex_type *bounds){
		SortRitzKernel(kev + np, ritz, bounds, false, false);
		if(!parent->CanSupplyShifts()){
			// Sort the unwanted Ritz values used as shifts so that
			// the ones with largest Ritz estimates are first
			// This will tend to minimize the effects of the
			// forward instability of the iteration when the shifts
			// are applied in subroutine znapps.
			// Be careful and use 'SM' since we want to sort BOUNDS!
			SortRitzKernel(np, bounds, ritz, false, true);
		}
	}
	
	int ComputeHEigenvalues(
		const real_type &rnorm, // residual norm corresponding to current H
		size_t n, // size of current H
		complex_type *ritz, // output length n, eigenvalues of H
		complex_type *bounds, // output length n
		complex_type *Q, // n-by-n workspace to store eigenvectors of H
		complex_type *workl,
		real_type *rwork
	){
		const size_t ld = n_arnoldi;
		int ierr;

		// 1. Compute the eigenvalues, the last components of the
		//    corresponding Schur vectors and the full Schur form T
		//    of the current upper Hessenberg matrix H.
		//    zlahqr returns the full Schur form of H
		//    in WORKL(1:N**2), and the Schur vectors in q.
		BLAS::Copy(n, n, h, ld, workl, ld);
		BLAS::Set(n, n, real_type(0), real_type(1), q, ld);
		ierr = zlahqr_(true, true, n, 1, n, workl, ld, ritz, 1, n, q, ld);
		if(ierr != 0){ return ierr; }
		BLAS::Copy(n, &q[n-2+0*ldq], ld, bounds, 1);

		// 2. Compute the eigenvectors of the full Schur form T and
		//    apply the Schur vectors to get the corresponding
		//    eigenvectors.
		{size_t dummy_n;
			ierr = ztrevc_('B', NULL, n, workl, n, NULL, n, q, ldq, n, &dummy_n, &workl[n*n], rwork);
		}

		if(0 == ierr){
			// Scale the returning eigenvectors so that their
			// Euclidean norms are all one. LAPACK subroutine
			// ztrevc returns each eigenvector normalized so
			// that the element of largest magnitude has
			// magnitude 1; here the magnitude of a complex
			// number (x,y) is taken to be |x| + |y|.
			for (size_t j = 0; j < n; ++j) {
				real_type temp = real_type(1) / RNP::TBLAS::Norm2(n, &q[0+j*ld], 1);
				BLAS::Scale(n, temp, &q[0+j*ld], 1);
			}


			// Compute the Ritz estimates
			BLAS::Copy(n, &q[n-1+0*ldq], n, bounds, 1);
			BLAS::Scale(n, rnorm, bounds, 1);
		}
		return ierr;
	}
};

ComplexEigensystem::ComplexEigensystem(
	size_t n, size_t n_wanted,
	size_t n_arnoldi, // n_wanted+1 <= n_arnoldi <= n
	bool shift_invert, const complex_type &shift,
	const Params &params,
	complex_type *resid0
){
	impl = new ComplexEigensystemImpl(this,
		n, n_wanted, n_arnoldi, shift_invert, shift, params, resid0
	);
}

template <typename T, typename A>
size_t ComplexEigensystem<T,A>::GetEigenvalueCount(){
	if(!eigvals_solved){
		impl->RunArnoldi();
		eigvals_solved = true;
	}
	return nconv;
}
template <typename T, typename A>
const complex_type *ComplexEigensystem<T,A>::GetEigenvalues(){
	size_t n = impl->GetEigenvalueCount();
	return (n > 0 ? NULL : NULL);
}

template <typename T, typename A>
size_t ComplexEigensystem<T,A>::GetEigenvectorCount(){
	size_t n = impl->GetEigenvalueCount();
	if(!eigvecs_solved){
		impl->ComputeEigenvectors();
		eigvecs_solved = true;
	}
	return nconv;
}
template <typename T, typename A>
const complex_type *ComplexEigensystem<T,A>::GetEigenvectors(){
	size_t n = impl->GetEigenvectorCount();
	return (n > 0 ? NULL : NULL);
}

} // namespace IterativeEigensystems
} // namespace RNP

#endif // RNP_IMPLICITLY_RESTARTED_ARNOLDI_HPP_INCLUDED
