#ifndef RNP_MATRIX_NORMS_HPP_INCLUDED
#define RNP_MATRIX_NORMS_HPP_INCLUDED

// We will define the following other functions here:
// MatrixNormBanded // _langb
// MatrixNormTridiagonal // _langt
// MatrixNormHessenberg // _lanhs (not here, see Hessenberg.hpp)
// MatrixNormHerm // _lanhe
// MatrixNormBandedHerm // _lanhb
// MatrixNormPackedHerm // _lanhp
// MatrixNormTridiagonalHerm // _lanht
// MatrixNormTr // _lantr (not here, see Triangular.hpp)
// MatrixNormBandedTr // _lantb
// MatrixNormPackedTr // _lantp
// The routines for particular matrix types in their respective
// namespaces will either duplicate these or call into here.

#include <cstddef>
#include <RNP/BLAS.hpp>

namespace RNP{
namespace LA{

template <typename T> // _lange
typename Traits<T>::real_type MatrixNorm(
	const char *norm, size_t m, size_t n, const T *a, size_t lda,
	typename Traits<T>::real_type *work = NULL
){
	typedef typename Traits<T>::real_type real_type;
	real_type result(0);
	if(0 == n){ return result; }
	if('M' == norm[0]){ // max(abs(A(i,j)))
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				real_type ca = Traits<T>::abs(a[i+j*lda]);
				if(!(ca < result)){ result = ca; }
			}
		}
	}else if('O' == norm[0] || '1' == norm[0]){ // max col sum
		for(size_t j = 0; j < n; ++j){
			real_type sum = 0;
			for(size_t i = 0; i < m; ++i){
				sum += Traits<T>::abs(a[i+j*lda]);
			}
			if(!(sum < result)){ result = sum; }
		}
	}else if('I' == norm[0]){ // max row sum
		if(NULL == work){ // can't accumulate row sums
			for(size_t i = 0; i < m; ++i){
				real_type sum = 0;
				for(size_t j = 0; j < n; ++j){
					sum += Traits<T>::abs(a[i+j*lda]);
				}
				if(!(sum < result)){ result = sum; }
			}
		}else{ // accumulate row sums in a cache-friendlier traversal order
			for(size_t i = 0; i < m; ++i){ work[i] = 0; }
			for(size_t j = 0; j < n; ++j){
				for(size_t i = 0; i < m; ++i){
					work[i] += Traits<T>::abs(a[i+j*lda]);
				}
			}
			for(size_t i = 0; i < m; ++i){
				if(!(work[i] < result)){ result = work[i]; }
			}
		}
	}else if('F' == norm[0] || 'E' == norm[0]){ // Frobenius norm
		real_type scale(0);
		real_type sum(1);
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				real_type ca = Traits<T>::abs(a[i+j*lda]);
				if(scale < ca){
					real_type r = scale/ca;
					sum = real_type(1) + sum*r*r;
					scale = ca;
				}else{
					real_type r = ca/scale;
					sum += r*r;
				}
			}
		}
		result = scale*sqrt(sum);
	}
	return result;
}

} // namespace LA
} // namespace RNP

#endif // RNP_MATRIX_NORMS_HPP_INCLUDED
