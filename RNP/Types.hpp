#ifndef RNP_TYPES_HPP_INCLUDED
#define RNP_TYPES_HPP_INCLUDED

#define _USE_MATH_DEFINES
#include <cmath>
#include <complex>
#include <limits>

namespace RNP{

namespace TBLAS{ class Kernel; }

// Base class for numeric traits
template <typename T> struct GenericTraits{
	enum{
		IsInteger = std::numeric_limits<T>::is_integer,
		IsSigned  = std::numeric_limits<T>::is_signed,
		IsComplex = 0
	};
	typedef T Real;
	static inline Real epsilon(){ return std::numeric_limits<T>::epsilon(); }
	static inline Real maximum(){ return std::numeric_limits<T>::max(); }
	static inline Real abs(const T &val){ return std::abs(val); }
	static inline Real norm1(const T &val){ return std::abs(val); }
	static inline Real real(const T &val){ return val; }
	static inline Real imag(const T &val){ return Real(0); }
	static inline T conj(const T &val){ return val; }
	static inline Real abs2(const T &val){ return val*val; }
};

// Traits class for scalar reals
template<typename T> struct Traits : GenericTraits<T>{
};

// Partial template specialization for complex numbers
template<typename _Real> struct Traits<std::complex<_Real> >
	: GenericTraits<std::complex<_Real> >
{
	enum{
		IsComplex = 1
	};
	typedef std::complex<_Real> Complex;
	typedef _Real Real;
	static inline Real epsilon(){ return std::numeric_limits<_Real>::epsilon(); }
	static inline Real maximum(){ return std::numeric_limits<_Real>::max(); }
	static inline Real abs(const Complex &val){ return std::abs(val); }
	static inline Real norm1(const Complex&val){ return std::abs(val); }
	static inline Real real(const Complex &val){ return std::real(val); }
	static inline Real imag(const Complex &val){ return std::imag(val); }
	static inline Complex conj(const Complex &val){ return std::conj(val); }
	static inline Real abs2(const Complex &val){ return std::norm(val); }
};

class Base{
	unsigned int flags;
	mutable bool valid;
protected:
	void Invalidate() const{ valid = false; }
	bool IsValid() const{ return valid; }
public:
	Base():flags(0),valid(true){}
	
	const static unsigned int SELF_ALLOCATED = 0x8000;
};

template <typename T>
class Vector : public Base{
	T *x;
	size_t n, inc;
	//Vector(const Vector &x){ /* do not use */ }
	friend class RNP::TBLAS::Kernel;
public:
	Vector(size_t nelem, T *pelem, size_t incelem):x(pelem),n(nelem),inc(incelem){}
	Vector(size_t nelem):x(NULL),n(nelem),inc(1){
		flags |= Base::SELF_ALLOCATED;
		x = (T*)malloc(sizeof(T) * nelem);
	}
	~Vector(){
		if(Base::SELF_ALLOCATED & flags){ free(x); x = NULL; }
	}
	T* ptr() const{ return x; }
	size_t size() const{ return n; }
	size_t inc()  const{ return inc; }
	
	void Fill(const T &value){
		size_t i = n;
		T *v = x;
		while(i --> 0){
			*v = value;
			v += inc;
		}
	}
	void Zero(){
		Fill(0);
	}
	
	T& operator[](size_t i){ return x[i]; }
	const T& operator[](size_t i) const{ return x[i]; }
};

template <typename T>
class Matrix : public Base{
	T *a;
	size_t m, n, lda;
	unsigned int flags;
	//Matrix(const Matrix &m){ /* do not use */ }
	friend class RNP::TBLAS::Kernel;
public:
	Matrix(size_t nrows, size_t ncols, T *ptr, size_t stride):a(ptr),m(nrows),n(ncols),lda(stride){}
	Matrix(size_t nrows, size_t ncols):a(NULL),m(nrows),n(ncols),lda(nrows){
		flags |= Base::SELF_ALLOCATED;
		a = (T*)malloc(sizeof(T) * nrows*ncols);
	}
	~Matrix(){
		if(Base::SELF_ALLOCATED & flags){ free(a); a = NULL; }
	}
	T* ptr() const{ return a; }
	size_t rows() const{ return m; }
	size_t cols() const{ return n; }
	size_t ldim() const{ return lda; }
	
	enum Part{
		BelowDiagonal = 1,
		AboveDiagonal = 2,
		Diagonal = 4,
		All = 7,
	};
	
	void Fill(Part part, const T &value){
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				if((i == j && Diagonal & part)
				|| (i > j && BelowDiagonal & part)
				|| (i < j && AboveDiagonal & part)){
					a[i+j*lda] = value;
				}
			}
		}
	}
	void Zero(){
		Fill(All, 0);
	}
	void Identity(const T &value){
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				if(i == j){
					a[i+j*lda] = T(1);
				}else{
					a[i+j*lda] = T(0);
				}
			}
		}
	}
	
	T& operator()(size_t i, size_t j){ return a[i+j*lda]; }
	const T& operator()(size_t i, size_t j) const{ return a[i+j*lda]; }
	
	const Vector<T> Column(size_t j) const{
		return Vector<T>(m, &a[0+j*lda], 1);
	}
	const Vector<T> Row(size_t i) const{
		return Vector<T>(n, &a[i+0*lda], lda);
	}
	Vector<T> Column(size_t j){
		return Vector<T>(m, &a[0+j*lda], 1);
	}
	Vector<T> Row(size_t i){
		return Vector<T>(n, &a[i+0*lda], lda);
	}
};

}; // namespace RNP

#endif // RNP_TYPES_HPP_INCLUDED
