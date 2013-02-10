#ifndef RNP_RESCALE_HPP_INCLUDED
#define RNP_RESCALE_HPP_INCLUDED

#include <RNP/Types.hpp>
#include <cmath>

namespace RNP{

///////////////////////////////////////////////////////////////////////
// Rescale
// -------
// Rescales every element of a vector by an integer power of 2.
//
// Arguments
// n     Number of elements in the vector.
// p     The power of 2 of the scale factor (2^p).
// x     Pointer to the first element of the vector.
// incx  Increment between elements of the vector, incx > 0.
//
template <typename T>
void Rescale(size_t n, int p, T *x, size_t incx){
	while(n --> 0){
		*x = std::ldexp(*x, p);
		x += incx;
	}
}
template <typename T>
void Rescale(size_t n, int p, std::complex<T> *x, size_t incx){
	while(n --> 0){
		*x = std::complex<T>(std::ldexp(x->real(), p), std::ldexp(x->imag(), p));
		x += incx;
	}
}

///////////////////////////////////////////////////////////////////////
// Rescale
// -------
// Rescales every element of a matrix safely. The scale factor is
// specified as a ratio cto/cfrom. One typically specifies cfrom
// as the element norm of the existing matrix, and cto as the target
// element norm scaling.
//
// Arguments
// type  Type of matrix to scale.
//       If "G", the matrix is a general rectangular matrix.
//       If "L", the matrix is assumed to be lower triangular.
//       If "U", the matrix is assumed to be upper triangular.
//       If "H", the matrix is assumed to be upper Hessenberg.
//       If "B", the matrix is assumed to be the lower half of a
//               symmetric banded matrix (kl is the lower bandwidth).
//       If "Q", the matrix is assumed to be the upper half of a
//               symmetric banded matrix (ku is the lower bandwidth).
//       If "Z", the matrix is assumed to be banded with lower and
//               upper bandwidths kl and ku, respectively.
// cfrom The denominator of the scale factor to apply.
// cto   The numerator of the scale factor to apply.
// m     Number of rows of the matrix.
// n     Number of columns of the matrix.
// a     Pointer to the first element of the matrix.
// lda   Leading dimension of the array containing the matrix, lda > 0.
//
template <typename TS, typename T>
void Rescale(
	const char *type, size_t kl, size_t ku,
	const TS &cfrom, const TS &cto,
	size_t m, size_t n, T* a, size_t lda
){
	if(n == 0 || m == 0){ return; }

	const TS smlnum = Traits<TS>::min();
	const TS bignum = TS(1) / smlnum;

	TS cfromc = cfrom;
	TS ctoc = cto;

	bool done = true;
	do{
		const TS cfrom1 = cfromc * smlnum;
		TS mul;
		if(cfrom1 == cfromc){
			// CFROMC is an inf.  Multiply by a correctly signed zero for
			// finite CTOC, or a NaN if CTOC is infinite.
			mul = ctoc / cfromc;
			done = true;
			//cto1 = ctoc;
		}else{
			const TS cto1 = ctoc / bignum;
			if(cto1 == ctoc){
				// CTOC is either 0 or an inf.  In both cases, CTOC itself
				// serves as the correct multiplication factor.
				mul = ctoc;
				done = true;
				cfromc = TS(1);
			}else if(Traits<TS>::abs(cfrom1) > Traits<TS>::abs(ctoc) && ctoc != TS(0)){
				mul = smlnum;
				done = false;
				cfromc = cfrom1;
			}else if(Traits<TS>::abs(cto1) > Traits<TS>::abs(cfromc)){
				mul = bignum;
				done = false;
				ctoc = cto1;
			}else{
				mul = ctoc / cfromc;
				done = true;
			}
		}

		switch(type[0]){
		case 'G': // Full matrix
			for(size_t j = 0; j < n; ++j){
				for(size_t i = 0; i < m; ++i){
					a[i+j*lda] *= mul;
				}
			}
			break;
		case 'L': // Lower triangular matrix
			for(size_t j = 0; j < n; ++j){
				for(size_t i = j; i < m; ++i){
					a[i+j*lda] *= mul;
				}
			}
			break;
		case 'U': // Upper triangular matrix
			for(size_t j = 0; j < n; ++j){
				size_t ilimit = j+1; if(m < ilimit){ ilimit = m; }
				for(size_t i = 0; i < ilimit; ++i){
					a[i+j*lda] *= mul;
				}
			}
			break;
		case 'H': // Upper Hessenberg matrix
			for(size_t j = 0; j < n; ++j) {
				size_t ilimit = j+2; if(m < ilimit){ ilimit = m; };
				for(size_t i = 0; i < ilimit; ++i){
					a[i+j*lda] *= mul;
				}
			}
			break;
		case 'B': // Lower half of a symmetric band matrix
			for(size_t j = 0; j < n; ++j){
				size_t ilimit = n-j; if(kl+1 < ilimit){ ilimit = kl+1; }
				for(size_t i = 0; i < ilimit; ++i){
					a[i+j*lda] *= mul;
				}
			}
			break;
		case 'Q': // Upper half of a symmetric band matrix
			for(size_t j = 0; j < n; ++j){
				size_t istart = (ku > j) ? ku-j : 0;
				for(size_t i = istart; i <= ku; ++i){
					a[i+j*lda] *= mul;
				}
			}
		case 'Z': // Band matrix
			{ size_t k3 = 2*kl + ku + 1;
			for(size_t j = 0; j < n; ++j){
				size_t istart = kl+ku-j;
				if(kl > istart){ istart = kl; }
				size_t ilimit = kl + ku + m-j;
				if(k3 < ilimit){ ilimit = k3; }
				for(size_t i = istart; i < ilimit; ++i){
					a[i+j*lda] *= mul;
				}
			} }
			break;
		default: break;
		}
	}while(!done);
}

} // namespace RNP

#endif // RNP_RESCALE_HPP_INCLUDED
