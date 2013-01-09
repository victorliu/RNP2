#ifndef RNP_TYPES_HPP_INCLUDED
#define RNP_TYPES_HPP_INCLUDED

#define _USE_MATH_DEFINES
#include <cmath>
#include <complex>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <RNP/Debug.hpp>

namespace RNP{

enum Trans{
	NoTranspose = 'N',
	Transpose = 'T',
	ConjugateTranspose = 'C'
};

enum Uplo{
	Lower = 'L',
	Upper = 'U'
};

enum Side{
	Left = 'L',
	Right = 'R'
};

enum Diag{
	Unit = 'U',
	NonUnit = 'N'
};

namespace TBLAS{ class Kernel; }

// Base class for numeric traits
template <typename T> struct GenericTraits{
	typedef T real_type;
	static inline bool is_complex(){ return false; }
	static inline real_type eps(){ return std::numeric_limits<T>::epsilon(); }
	static inline real_type min(){ return std::numeric_limits<T>::min(); }
	static inline real_type max(){ return std::numeric_limits<T>::max(); }
	static inline real_type abs(const T &val){ return std::abs(val); }
	static inline real_type norm1(const T &val){ return std::abs(val); }
	static inline real_type norminf(const T &val){ return std::abs(val); }
	static inline real_type real(const T &val){ return val; }
	static inline real_type imag(const T &val){ return real_type(0); }
	static inline T conj(const T &val){ return val; }
	static inline real_type abs2(const T &val){ return val*val; }
	static inline T copyimag(const T &re, const T &usemyimag){ return re; }
	static inline T div(const T &num, const T &den){ return num/den; }
};

// Traits class for scalar reals
template<typename T> struct Traits : GenericTraits<T>{
	static T hypot2(const T &x, const T &y){
		// returns sqrt(x**2+y**2), taking care not to cause unnecessary overflow.
		T a = Traits<T>::abs(x);
		T b = Traits<T>::abs(y);
		if(a < b){ std::swap(a,b); }
		if(0 == b){
			return a;
		}else{
			b /= a;
			return a * sqrt(b*b + T(1));
		}
	}
	static T hypot3(const T &x, const T &y, const T &z){
		// returns sqrt(x**2+y**2+z**2), taking care not to cause unnecessary overflow.
		T xabs = Traits<T>::abs(x);
		T yabs = Traits<T>::abs(y);
		T zabs = Traits<T>::abs(z);
		T w = xabs;
		if(yabs > w){ w = yabs; }
		if(zabs > w){ w = zabs; }
		if(0 == w){
			// W can be zero for max(0,nan,0)
			// adding all three entries together will make sure
			// NaN will not disappear.
			return xabs + yabs + zabs;
		} else {
			xabs /= w;
			yabs /= w;
			zabs /= w;
			return w * sqrt(xabs*xabs + yabs*yabs + zabs*zabs);
		}
	}
	static T twopi(){ return T(6.28318530717958647692528676656); }
};

// Partial template specialization for complex numbers
template<typename _Real> struct Traits<std::complex<_Real> >
	: GenericTraits<std::complex<_Real> >
{
	typedef std::complex<_Real> complex_type;
	typedef _Real real_type;
	static inline bool is_complex(){ return true; }
	static inline real_type eps(){ return std::numeric_limits<_Real>::epsilon(); }
	static inline real_type min(){ return std::numeric_limits<_Real>::min(); }
	static inline real_type max(){ return std::numeric_limits<_Real>::max(); }
	static inline real_type abs(const complex_type &val){ return std::abs(val); }
	static inline real_type norm1(const complex_type &val){ return std::abs(std::real(val)) + std::abs(std::imag(val)); }
	static inline real_type norminf(const complex_type &val){
		real_type ar = Traits<real_type>::abs(val.real());
		real_type ai = Traits<real_type>::abs(val.imag());
		return (ar > ai ? ar : ai);
	}
	static inline real_type real(const complex_type &val){ return val.real(); }
	static inline real_type imag(const complex_type &val){ return val.imag(); }
	static inline complex_type conj(const complex_type &val){ return complex_type(val.real(), -val.imag()); }
	static inline real_type abs2(const complex_type &val){ return val.real()*val.real() + val.imag()*val.imag(); }
	static inline complex_type copyimag(const real_type &re, const complex_type &usemyimag){
		return complex_type(re, usemyimag.imag());
	}
	
	real_type zdiv_cr(
		real_type a, real_type b,
		real_type c, real_type d,
		real_type r, real_type t
	){
		if(real_type(0) != r){
			const real_type br = b*r;
			if(real_type(0) != br){
				return (a + br)*t;
			}else{
				return a*t + (b*t)*r;
			}
		}else{
			return (a+d*(b/c))*t;
		}
	}
	static void zdiv_sub(
		real_type  a, real_type  b,
		real_type  c, real_type  d,
		real_type *p, real_type *q
	){
		real_type r(d/c);
		real_type t(real_type(1)/(c+d*r));
		*p = zdiv_cr(a, b, c, d, r, t);
		a = -a;
		*q = zdiv_cr(b, a, c, d, r, t);
	}
	static inline complex_type div(const complex_type &num, const complex_type &den){
		const real_type a = num.real();
		const real_type b = num.imag();
		const real_type c = den.real();
		const real_type d = den.imag();
		const real_type A = Traits<real_type>::abs(a);
		const real_type B = Traits<real_type>::abs(b);
		const real_type C = Traits<real_type>::abs(c);
		const real_type D = Traits<real_type>::abs(d);
		const real_type AB = (A > B ? A : B);
		const real_type CD = (C > D ? C : D);
		static const real_type BS(2); // number base
		static const real_type half(real_type(1) / real_type(2));
		static const real_type two(2);
		real_type S(1);
		const real_type Be(BS / (Traits<real_type>::eps()*Traits<real_type>::eps()));
		if(AB > half*Traits<real_type>::max()){ // scale down a, b
			a *= half; b *= half; S *= two;
		}
		if(CD > half*Traits<real_type>::max()){ // scale down c, d
			c *= half; d *= half; S *= half;
		}
		if(AB <= Traits<real_type>::min() * BS/Traits<real_type>::eps()){ // scale up a,b
			a *= Be; b *= Be; S /= Be;
		}
		if(CD <= Traits<real_type>::min() * BS/Traits<real_type>::eps()){ // scale up a,b
			c *= Be; d *= Be; S *= Be;
		}
		real_type p, q;
		if(Traits<real_type>::abs(d) <= Traits<real_type>::abs(c)){
			zdiv_sub(a, b, c, d, &p, &q);
		}else{
			zdiv_sub(b, a, d, c, &p, &q);
			q = -q;
		}
		return complex_type(p*S, q*S);
	}
};

// Extra precision traits
template <typename T> struct GenericExtraPrecisionTraits{
	typedef T real_type;
};
template<typename T> struct ExtraPrecisionTraits : GenericExtraPrecisionTraits<T>{
	typedef T real_type;
	static inline real_type split_value(){
		// The splitting value for computing extended precision products
		return T(1) + sqrt(
			T(std::numeric_limits<T>::radix())/std::numeric_limits<T>::eps()
		);
	}
	// computes c = a*b, and d = remainder(a*b)
	static inline void product2(const T &a, const T &b, T *prod, T *rem){
		static const real_type s = split_value();
		*prod = a*b;
		// Split a
		T c = s*a;
		T ah = c - (c - a);
		T al = a - ah;
		// Split b
		c = s*b;
		T bh = c - (c - b);
		T bl = b - bh;
		*rem = al*bl - ((((*prod) - ah*bh) - al*bh) - ah*bl);
	}
	static inline void sum2(const T &a, const T &b, T *sum, T *rem){
		*sum = a + b;
		T z = *sum - a;
		*rem = (a - (*sum - z)) + (b - z);
	}
};
template<typename _Real> struct ExtraPrecisionTraits<std::complex<_Real> >
	: GenericExtraPrecisionTraits<std::complex<_Real> >
{
	typedef std::complex<_Real> complex_type;
	typedef _Real real_type;
	typedef ExtraPrecisionTraits<real_type> real_traits;
	static inline bool is_complex(){ return true; }
	
	static inline void product2(const complex_type &a, const complex_type &b, complex_type *prod, complex_type *rem){
		// Let the product be z = zr + i*zi
		// zr = ar*br - ai*bi, zi = ar*bi + ai*br
		real_type p1, p2, q1, q2, r1, r2, i1, i2;
		// Compute zr
		real_traits::product2(a.real(),  b.real(), &p1, &p2);
		real_traits::product2(a.imag(), -b.imag(), &q1, &q2);
		real_traits::sum2(p1, q1, &r1, &r2);
		r2 += p2 + q2;
		// Compute zi
		real_traits::product2(a.real(), b.imag(), &p1, &p2);
		real_traits::product2(a.imag(), b.real(), &q1, &q2);
		real_traits::sum2(p1, q1, &i1, &i2);
		i2 += p2 + q2;
		*prod = complex_type(r1, i1);
		*rem  = complex_type(r2, i2);
	}
	static inline void sum2(const complex_type &a, const complex_type &b, complex_type *sum, complex_type *rem){
		real_type r1, r2, i1, i2;
		real_traits::sum2(a.real(), b.real(), &r1, &r2);
		real_traits::sum2(a.imag(), b.imag(), &i1, &i2);
		*sum = complex_type(r1, i1);
		*rem = complex_type(r2, i2);
	}
};


template <typename T, typename Allocator = std::allocator<T> >
class Base{
	mutable bool valid;
protected:
	mutable unsigned int flags;
	Allocator the_allocator;
	
	template <typename Other>
	Other *Alloc(size_t n) const{
		typename Allocator::template rebind<Other>::other other_allocator;
		return other_allocator.allocate(n);
	}
	template <typename Other>
	void Dealloc(Other *ptr, size_t n) const{
		typename Allocator::template rebind<Other>::other other_allocator;
		other_allocator.deallocate(ptr, n);
	}
public:
	typedef Allocator                           allocator_type;
	typedef typename Allocator::size_type       size_type;
	typedef typename Allocator::difference_type difference_type;
	typedef typename Allocator::reference       reference;
	typedef typename Allocator::const_reference const_reference;
	typedef T*                                  iterator;
	typedef const T*                            const_iterator;

	Base():valid(true),flags(0){}
	Base(const Base &b):valid(true),flags(b.flags){}
	const static unsigned int SELF_ALLOCATED = 0x8000;
	
	virtual void Invalidate() const{ valid = false; }
	virtual bool IsValid() const{ return valid; }
	virtual void Release() const{ flags &= ~SELF_ALLOCATED; }
};

template <typename T, typename Allocator = std::allocator<T> >
class Vector : public Base<T, Allocator>{
	T *x;
	size_t n, inc;
	//Vector(const Vector &x){ /* do not use */ }
	friend class RNP::TBLAS::Kernel;
public:
	typedef Base<T,Allocator> base_type;
	Vector(size_t nelem, T *pelem, size_t incelem):x(pelem),n(nelem),inc(incelem){}
	Vector(size_t nelem):x(NULL),n(nelem),inc(1){
		this->flags |= base_type::SELF_ALLOCATED;
		x = this->template Alloc<T>(nelem);
	}
	~Vector(){
		if(base_type::SELF_ALLOCATED & this->flags){
			this->template Dealloc<T>(x, n);
			x = NULL;
		}
	}
	T* ptr() const{ return x; }
	size_t size() const{ return n; }
	size_t incr()  const{ return inc; }
	
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

template <typename T, typename Allocator = std::allocator<T> >
class Matrix : public Base<T, Allocator>{
	T *a;
	size_t m, n, lda;
	//Matrix(const Matrix &m){ /* do not use */ }
	friend class RNP::TBLAS::Kernel;
public:
	typedef Matrix<T,Allocator> matrix_type;
	typedef Vector<T,Allocator> vector_type;
	typedef Base<T,Allocator> base_type;
	Matrix(size_t nrows, size_t ncols, T *ptr, size_t stride):a(ptr),m(nrows),n(ncols),lda(stride){}
	Matrix(size_t nrows, size_t ncols):a(NULL),m(nrows),n(ncols),lda(nrows){
		this->flags |= base_type::SELF_ALLOCATED;
		a = this->template Alloc<T>(nrows*ncols);
	}
	~Matrix(){
		if(base_type::SELF_ALLOCATED & this->flags){
			this->template Dealloc<T>(a, m*n);
			a = NULL;
		}
	}
	T* ptr() const{ return a; }
	size_t rows() const{ return m; }
	size_t cols() const{ return n; }
	size_t ldim() const{ return lda; }
	
	template <typename A2>
	matrix_type& operator=(const Matrix<T,A2> &M){
		RNPAssert(M.rows() == m);
		RNPAssert(M.cols() == n);
		if(M.ldim() == lda && lda == m){ // copy entire array at once
			memcpy(a, M.ptr(), sizeof(T) * m*n);
		}else{ // copy by column
			for(size_t j = 0; j < n; ++j){
				memcpy(a[j*lda], &M.a[j*M.lda], sizeof(T) * m);
			}
		}
		return *this;
	}
	
	template <typename U, typename A2>
	matrix_type& operator=(const Matrix<U,A2> &M){
		RNPAssert(M.rows() == m);
		RNPAssert(M.cols() == n);
		for(size_t j = 0; j < n; ++j){
			for(size_t i = 0; i < m; ++i){
				a[i+j*lda] = (T)M(i,j);
			}
		}
		return *this;
	}
	
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
	void Identity(){
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
	
	const vector_type Column(size_t j) const{
		return vector_type(m, &a[0+j*lda], 1);
	}
	const vector_type Row(size_t i) const{
		return vector_type(n, &a[i+0*lda], lda);
	}
	vector_type Column(size_t j){
		return vector_type(m, &a[0+j*lda], 1);
	}
	vector_type Row(size_t i){
		return vector_type(n, &a[i+0*lda], lda);
	}
};

}; // namespace RNP

#endif // RNP_TYPES_HPP_INCLUDED
