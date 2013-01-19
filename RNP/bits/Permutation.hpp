#ifndef RNP_PERMUTATION_HPP_INCLUDED
#define RNP_PERMUTATION_HPP_INCLUDED

#include <cstddef>
#include <RNP/BLAS.hpp>

namespace RNP{
namespace LA{

///////////////////////////////////////////////////////////////////////
// RNP::LA::ApplyPermutations
// --------------------------
// Applies a permutation vector to the rows or columns of a matrix.
// Used for example, with the pivot vector returned by an LU
// factorization.
//
// Arguments
// side  If "L", the permutation is applied from the left (row swaps).
//       If "R", the permutation is applied from the right (column swaps).
// dir   If "F", the swaps are applied in forward order.
//       If "B", the swaps are applied in backwards order.
// m     Number of rows of the matrix.
// n     Number of columns of the matrix.
// a     Pointer to the first element of the matrix.
// lda   Leading dimension of the array containing the matrix.
// ipiv  Pivot vector. If side = "L", length m.
//       If side = "R", length n.
// off   Offset at which to start applying the swaps. This is the
//       offset both into the vector ipiv, as well as the row or
//       column offset from the pointer specified by a.
//
template <typename T>
void ApplyPermutations(
	const char *side, const char *dir,
	size_t m, size_t n, T *A, size_t ldA, const size_t *ipiv,
	size_t off = 0
){
	if('L' == side[0]){
		if('F' == dir[0]){
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
		if('F' == dir[0]){
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
