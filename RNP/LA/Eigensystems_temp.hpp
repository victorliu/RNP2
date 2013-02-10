template <typename T>
int RealSchurForm(
	const char *jobvs,
	int (*cmpfun)(const T &xr, const T &xi, const T &yr, const T &yi, void *data),
	void *data,
	size_t n, 
	T *a, size_t lda,
	T *wr, T *wi,
	T *vs, size_t ldvs,
	size_t *lwork, T *work
){
}

template <typename T>
int ComplexSchurForm(
	const char *jobvs,
	int (*cmpfun)(const std::complex<T> &x, const std::complex<T> &y, void *data),
	void *data,
	size_t n, 
	std::complex<T> *a, size_t lda,
	std::complex<T> *w,
	std::complex<T> *vs, size_t ldvs,
	size_t *lwork, std::complex<T> *work, T *rwork
){
}


template <typename T>
int RealEigensystem(
	size_t n, 
	T *a, size_t lda,
	T *wr, T *wi,
	T *vl, size_t ldvl, T *vr, size_t ldvr,
	size_t *lwork, T *work
){
}
/*
template <typename T>
int ComplexEigensystemExpert(
	size_t n,
	std::complex<T> *a, size_t lda,
	std::complex<T> *w,
	std::complex<T> *vl, size_t ldvl, std::complex<T> *vr, size_t ldvr,
	// Additional expert options
	size_t ilo, size_t ihi,
	bool normeigvecs, // if false, then eigenvectors are not normalized
	// Balancing data (not done if NULL)
	size_t *balperm, T *balscale,
	T *abnrm,
	T *rconde, // length n
	T *rcondv, // length n
	// Workspace
	size_t *lwork, std::complex<T> *work, T *rwork
){
	typedef std::complex<T> complex_type;
	typedef T real_type;
	
	RNPAssert(lda >= n);
	RNPAssert(NULL != lwork);
	
	const bool wantvl = (vl != NULL);
	const bool wantvr = (vr != NULL);
	
	if(n == 0){
		return 0;
	}
	if(0 == *lwork || NULL == work){
		*lwork = 2*n; // need to do better than this
		return 0;
	}
	
	const real_type eps = real_type(2)*Traits<real_type>::epsilon();
	const real_type smlnum = sqrt(Traits<real_type>::min()) / eps;
	const real_type bignum = real_type(1) / smlnum;
	
	// Scale A if max element outside range [SMLNUM,BIGNUM]
	real_type anrm = MatrixNorm("M", n, n, a, lda);
	bool scalea = false;
	real_type cscale = 1;
	if(anrm > 0. && anrm < smlnum){
		scalea = true;
		cscale = smlnum;
	}else if(anrm > bignum){
		scalea = true;
		cscale = bignum;
	}
	if(scalea){
		BLAS::Rescale("G", 0, 0, anrm, cscale, n, n, a, lda);
	}
	
	ComplexEigensystemBalance(n, a, lda, ilo, ihi, balperm, balscale);
	
	Hessenberg::Reduce(n, ilo, ihi, a, lda, &work[0], &work[n]);
	
	int info;
	if(wantvl){
		BLAS::CopyMatrix<'L'>(n, n, a, lda, vl, ldvl);

		// Generate unitary matrix in VL
		// (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
		// (RWorkspace: none)

		Reflector::GenerateHessenberg(n, ilo-1, ihi-1, vl, ldvl, &work[0], &work[n]);

		// Perform QR iteration, accumulating Schur vectors in VL
		// (CWorkspace: need 1, prefer HSWORK (see comments) )
		// (RWorkspace: none)

		iwrk = itau;
		info = HessenbergQR('S', 'V', n, ilo, ihi, a, lda, w, vl, ldvl, &work[n]);

		if (wantvr) {
			// Want left and right eigenvectors
			// Copy Schur vectors to VR
			RNP::TBLAS::CopyMatrix<'F'>(n, n, vl, ldvl, vr, ldvr);
		}
	}else if(wantvr){
		// Want right eigenvectors
		// Copy Householder vectors to VR
		RNP::TBLAS::CopyMatrix<'L'>(n, n, a, lda, vr, ldvr);
		// Generate unitary matrix in VR
		// (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
		// (RWorkspace: none)

		Reflector::GenerateHessenberg(n, ilo-1, ihi-1, vr, ldvr, &work[0], &work[n]);

		// Perform QR iteration, accumulating Schur vectors in VR
		// (CWorkspace: need 1, prefer HSWORK (see comments) )
		// (RWorkspace: none)

		iwrk = itau;
		info = HessenbergQR('S', 'V', n, ilo, ihi, a, lda, w, vr, ldvr, &work[n]);
	}else{
		// Compute eigenvalues only
		// (CWorkspace: need 1, prefer HSWORK (see comments) )
		// (RWorkspace: none)
		iwrk = itau;
		HessenbergQR('E', 'N', n, ilo, ihi, a, lda, w, vr, ldvr, &work[n]);
	}

	if(info == 0){
		size_t irwork = ibal + n;
		if(wantvl || wantvr){
			// Compute left and/or right eigenvectors
			// (CWorkspace: need 2*N)
			// (RWorkspace: need 2*N)
			size_t nout;
			TriangularEigenvectors('B', NULL, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, &work[iwrk], &rwork[irwork]);
		}

		if(wantvl){
			// Undo balancing of left eigenvectors
			// (CWorkspace: none)
			// (RWorkspace: need N)
			ComplexEigensystemBalanceUndo(n, a, lda, ilo, ihi, balperm, balscale);
			zgebak_('B', 'L', n, ilo-1, ihi-1, &rwork[ibal], n, vl, ldvl);

			// Normalize left eigenvectors and make largest component real
			for(size_t i = 0; i < n; ++i){
				double scl = 1. / RNP::TBLAS::Norm2(n, &vl[0+i*ldvl], 1);
				RNP::TBLAS::Scale(n, scl, &vl[0+i*ldvl], 1);
				for(size_t k = 0; k < n; ++k){
					rwork[irwork+k] = std::norm(vl[k+i*ldvl]);
				}
				size_t k = RNP::TBLAS::MaximumIndex(n, &rwork[irwork], 1);
				RNP::TBLAS::Scale(n, std::conj(vl[k+i*ldvl]) / sqrt(rwork[irwork+k]), &vl[0+i*ldvl], 1);
				vl[k+i*ldvl] = vl[k+i*ldvl].real();
			}
		}

		if(wantvr){
			// Undo balancing of right eigenvectors
			// (CWorkspace: none)
			// (RWorkspace: need N)
			ComplexEigensystemBalanceUndo(n, a, lda, ilo, ihi, balperm, balscale);
			zgebak_('B', 'R', n, ilo-1, ihi-1, &rwork[ibal], n, vr, ldvr);

			// Normalize right eigenvectors and make largest component real
			for(size_t i = 0; i < n; ++i){
				double scl = 1. / RNP::TBLAS::Norm2(n, &vr[0+i*ldvr], 1);
				RNP::TBLAS::Scale(n, scl, &vr[0+i*ldvr], 1);
				for(size_t k = 0; k < n; ++k){
					rwork[irwork + k] = std::norm(vr[k+i*ldvr]);
				}
				size_t k = RNP::TBLAS::MaximumIndex(n, &rwork[irwork], 1);
				RNP::TBLAS::Scale(n, std::conj(vr[k+i*ldvr]) / sqrt(rwork[irwork+k]), &vr[0+i*ldvr], 1);
				vr[k+i*ldvr] = vr[k+i*ldvr].real();
			}
		}
	}
	
	
	// Undo scaling if necessary (the array bounds here need to be reviewed)
	if(scalea){
		BLAS::Rescale("G", 0, 0, cscale, anrm, n-info, 1, &w[info], n-info);
		if(info > 0){
			BLAS::Rescale("G", 0, 0, cscale, anrm, ilo, 1, w, n);
		}
	}
}
*/

template <typename T>
int ComplexEigensystem(
	size_t n,
	std::complex<T> *a, size_t lda,
	std::complex<T> *w,
	std::complex<T> *vl, size_t ldvl, std::complex<T> *vr, size_t ldvr,
	size_t *lwork, std::complex<T> *work, T *rwork
){
}