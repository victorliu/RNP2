#ifndef RNP_PERMUTATION_HPP_INCLUDED
#define RNP_PERMUTATION_HPP_INCLUDED

#include <cstddef>
#include <RNP/BLAS.hpp>

namespace RNP{
namespace LA{

template <typename T>
void ApplyPermutations(
	const char *side, const char *rev,
	size_t m, size_t n, T *A, size_t ldA, const size_t *ipiv,
	size_t off = 0
){
	if('L' == side[0]){
		if('N' == rev[0]){
			for(size_t i = off; i < off+m; ++i){
				size_t ip = ipiv[i];
				if(i == ip){ continue; }
				BLAS::Swap(n, &A[i],ldA, &A[ip],ldA);
			}
		}else{
			size_t i = off+m;
			while(i --> off){
				size_t ip = ipiv[i];
				if(i == ip){ continue; }
				BLAS::Swap(n, &A[i],ldA, &A[ip],ldA);
			}
		}
	}else{
		if('N' == rev[0]){
			for(size_t j = off; j < off+n; ++j){
				size_t jp = ipiv[j];
				if(j == jp){ continue; }
				BLAS::Swap(m, &A[0+j*ldA],1, &A[0+jp*ldA],1);
			}
		}else{
			size_t j = off+n;
			while(j --> off){
				size_t jp = ipiv[j];
				if(j == jp){ continue; }
				BLAS::Swap(m, &A[0+j*ldA],1, &A[0+jp*ldA],1);
			}
		}
	}
}

} // namespace LA
} // namespace RNP

#endif // RNP_PERMUTATION_HPP_INCLUDED
