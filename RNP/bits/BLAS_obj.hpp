#ifndef RNP_BLAS_OBJ_HPP_INCLUDED
#define RNP_BLAS_OBJ_HPP_INCLUDED

#include <complex>
#include <cstring>
#include <RNP/Types.hpp>
#include <RNP/Debug.hpp>

namespace RNP{

// Extra

template <typename T1, typename A1, typename T2, typename A2>
void Copy(const Matrix<T1,A1> &src, Matrix<T2,A2> &dst){
	RNPAssert(src.rows() == dst.rows() && src.cols() == dst.cols());
	BLAS::Copy(src.rows(), src.cols(), src.ptr(), src.ldim(), dst.ptr(), dst.ldim());
}

template <typename T1, typename A1, typename T2, typename A2>
void Copy(const char *trans, const Matrix<T1,A1> &src, Matrix<T2,A2> &dst){
	if('N' == trans[0]){
		Copy(src, dst);
	}else{
		RNPAssert(src.rows() == dst.cols() && src.cols() == dst.rows());
		BLAS::Copy(trans, src.rows(), src.cols(), src.ptr(), src.ldim(), dst.ptr(), dst.ldim());
	}
}

template <typename T, typename A>
void Conjugate(Vector<T,A> &x){
	BLAS::Conjugate(x.size(), x.ptr(), x.incr());
}

// Level 1

template <typename T, typename A1, typename A2>
void Swap(Vector<T,A1> &x, Vector<T,A1> &y){
	RNPAssert(x.size() == y.size());
	BLAS::Swap(x.size(), x.ptr(), x.incr(), y.ptr(), y.incr());
}

template <typename TS, typename T, typename A>
void Scale(const TS &alpha, Vector<T,A> &x){
	BLAS::Scale(x.size(), alpha, x.ptr(), x.incr());
}

template <typename T1, typename A1, typename T2, typename A2>
void Copy(const Vector<T1,A1> &src, Vector<T2,A1> &dst){
	RNPAssert(src.size() == dst.size());
	BLAS::Copy(src.size(), src.ptr(), src.incr(), dst.ptr(), dst.incr());
}

template <typename T, typename A1, typename A2>
T Dot(const Vector<T,A1> &x, const Vector<T,A1> &y){
	RNPAssert(x.size() == y.size());
	return BLAS::Dot(x.size(), x.ptr(), x.incr(), y.ptr(), y.incr());
}

template <typename T, typename A1, typename A2>
T ConjugateDot(const Vector<T,A1> &x, const Vector<T,A1> &y){
	RNPAssert(x.size() == y.size());
	return BLAS::ConjugateDot(x.size(), x.ptr(), x.incr(), y.ptr(), y.incr());
}

template <typename T, typename A>
typename Traits<T>::real_type Norm2(const Vector<T,A> &x){
	return BLAS::Norm2(x.size(), x.ptr(), x.incr());
}

template <typename T, typename A>
typename Traits<T>::real_type Norm1(const Vector<T,A> &x){
	return BLAS::Norm1(x.size(), x.ptr(), x.incr());
}

template <typename T, typename A>
size_t MaximumIndex(const Vector<T,A> &x){
	return BLAS::MaximumIndex(x.size(), x.ptr(), x.incr());
}

// Level 2
template <typename T, typename AA, typename AX, typename AY>
void MultMV(
	Trans trans, const T &alpha, const Matrix<T,AA> &a,
	const Vector<T,AX> &x, const T &beta, Vector<T,AX> &y
){
	if(NoTranspose == trans){
		RNPAssert(a.rows() == y.size() && a.cols() == x.size());
		BLAS::MultMV(
			"N", a.rows(), a.cols(), alpha, a.ptr(), a.ldim(),
			x.ptr(), x.incr(), beta, y.ptr(), y.incr()
		);
	}else{
		RNPAssert(a.rows() == x.size() && a.cols() == y.size());
		if(Transpose == trans){
			BLAS::MultMV(
				"T", a.rows(), a.cols(), alpha, a.ptr(), a.ldim(),
				x.ptr(), x.incr(), beta, y.ptr(), y.incr()
			);
		}else{
			BLAS::MultMV(
				"C", a.rows(), a.cols(), alpha, a.ptr(), a.ldim(),
				x.ptr(), x.incr(), beta, y.ptr(), y.incr()
			);
		}
	}
}

template <typename T, typename AA, typename AB, typename AC>
void MultMM(
	Trans transa, Trans transb, const T &alpha, const Matrix<T,AA> &a,
	const Matrix<T,AB> &b, const T &beta, Matrix<T,AC> &c
){
	if(NoTranspose == transa){
		if(NoTranspose == transb){
			RNPAssert(a.rows() == c.rows() && b.cols() == c.cols() && a.cols() == b.rows());
			BLAS::MultMM(
				"N", "N", a.rows(), b.cols(), a.cols(), alpha, a.ptr(), a.ldim(),
				b.ptr(), b.ldim(), beta, c.ptr(), c.ldim()
			);
		}else{
			RNPAssert(a.rows() == c.rows() && b.rows() == c.cols() && a.cols() == b.cols());
			if(Transpose == transb){
				BLAS::MultMM(
					"N", "T", a.rows(), b.rows(), a.cols(), alpha, a.ptr(), a.ldim(),
					b.ptr(), b.ldim(), beta, c.ptr(), c.ldim()
				);
			}else{
				BLAS::MultMM(
					"N", "C", a.rows(), b.rows(), a.cols(), alpha, a.ptr(), a.ldim(),
					b.ptr(), b.ldim(), beta, c.ptr(), c.ldim()
				);
			}
		}
	}else{
		if(NoTranspose == transb){
			RNPAssert(a.cols() == c.rows() && b.cols() == c.cols() && a.rows() == b.rows());
			if(Transpose == transa){
				BLAS::MultMM(
					"T", "N", a.cols(), b.cols(), a.rows(), alpha, a.ptr(), a.ldim(),
					b.ptr(), b.ldim(), beta, c.ptr(), c.ldim()
				);
			}else{
				BLAS::MultMM(
					"C", "N", a.cols(), b.cols(), a.rows(), alpha, a.ptr(), a.ldim(),
					b.ptr(), b.ldim(), beta, c.ptr(), c.ldim()
				);
			}
		}else{
			RNPAssert(a.cols() == c.rows() && b.rows() == c.cols() && a.rows() == b.cols());
			if(Transpose == transb){
				if(Transpose == transa){
					BLAS::MultMM(
						"T", "T", a.cols(), b.rows(), a.rows(), alpha, a.ptr(), a.ldim(),
						b.ptr(), b.ldim(), beta, c.ptr(), c.ldim()
					);
				}else{
					BLAS::MultMM(
						"C", "T", a.cols(), b.rows(), a.rows(), alpha, a.ptr(), a.ldim(),
						b.ptr(), b.ldim(), beta, c.ptr(), c.ldim()
					);
				}
			}else{
				if(Transpose == transa){
					BLAS::MultMM(
						"T", "C", a.cols(), b.rows(), a.rows(), alpha, a.ptr(), a.ldim(),
						b.ptr(), b.ldim(), beta, c.ptr(), c.ldim()
					);
				}else{
					BLAS::MultMM(
						"T", "C", a.cols(), b.rows(), a.rows(), alpha, a.ptr(), a.ldim(),
						b.ptr(), b.ldim(), beta, c.ptr(), c.ldim()
					);
				}
			}
		}
	}
}

} // namespace RNP

#endif // RNP_BLAS_MIX_HPP_INCLUDED
