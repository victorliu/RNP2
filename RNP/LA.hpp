#ifndef RNP_LA_HPP_INCLUDED
#define RNP_LA_HPP_INCLUDED

#include <RNP/Types.hpp>
#include <RNP/Debug.hpp>
#include <RNP/TBLAS.hpp>

namespace RNP{
namespace LA{

template <class T>
class LUFactors : public Base{
	T *a;
	size_t m, n, lda;
	size_t *ipiv;
	int err;
public:
	LUFactors(const Matrix &A):a(A.ptr()),m(A.rows()),n(A.cols()),lda(A.ldim()),ipiv(NULL){
		ipiv = (size_t*)malloc(sizeof(size_t) * ipiv);
		Compute();
	}
	~LUFactors(){
		free(ipiv);
	}
	void Release(){
		free(ipiv); ipiv = NULL;
		Invalidate();
	}
	
	bool IsValid() const{ return Base::IsValid() && 0 == err; }
	
	void MultFromLeft(Matrix &A){
	}
	void MultFromRight(Matrix &A){
	}
	void MultInverseFromLeft(Matrix &A){
	}
	void MultInverseFromRight(Matrix &A){
	}
protected:
	void Compute(){
		err = 0;
		size_t min_dim = (m < n ? m : n);
		for(size_t j = 0; j < min_dim; ++j){
			size_t jp = j + RNP::TBLAS::MaximumIndex(m-j, &a[j+j*lda], 1);
			ipiv[j] = jp;
			if(T(0) != a[jp+j*lda]){
				if(jp != j){
					RNP::TBLAS::Swap(n, &a[j+0*lda], lda, &a[jp+0*lda], lda);
				}
				if(j < m){
					RNP::TBLAS::Scale(m-j-1, T(1)/a[j+j*lda], &a[j+1+j*lda], 1); // possible overflow when inverting A(j,j)
				}
			}else{
				err = j;
			}
			if(j < min_dim){
				RNP::TBLAS::Rank1Update(m-j-1, n-j-1, T(-1), &a[j+1+j*lda], 1, &a[j+(j+1)*lda], lda, &a[j+1+(j+1)*lda], lda);
			}
		}
	}
};

}; // namespace LA
}; // namespace RNP

#endif // RNP_LA_HPP_INCLUDED
