#ifndef RNP_IOBASIC_HPP_INCLUDED
#define RNP_IOBASIC_HPP_INCLUDED

#include <istream>
#include <ostream>
#include <RNP/Types.hpp>

template <typename T, typename TA>
std::ostream& operator<<(std::ostream& os, const RNP::Vector<T,TA> &x){
	for(size_t i = 0; i < x.size(); ++i){
		if(i > 0){ os << '\n'; }
		os << x[i];
	}
	return os;
}

template <typename T, typename TA>
std::ostream& operator<<(std::ostream& os, const RNP::Matrix<T,TA> &a){
	for(size_t i = 0; i < a.rows(); ++i){
		for(size_t j = 0; j < a.cols(); ++j){
			if(j > 0){ os << '\t'; }
			os << a(i,j);
		}
		os << '\n';
	}
	return os;
}

template <typename T, typename TA>
std::istream& operator>>(std::istream& is, const RNP::Vector<T,TA> &x){
	for(size_t i = 0; i < x.size(); ++i){
		is >> x[i];
	}
	return is;
}

template <typename T, typename TA>
std::istream& operator<<(std::istream& is, const RNP::Matrix<T,TA> &a){
	for(size_t i = 0; i < a.rows(); ++i){
		for(size_t j = 0; j < a.cols(); ++j){
			is >> a(i,j);
		}
	}
	return is;
}

namespace RNP{
namespace IO{

template <typename T, typename TA>
Vector<T,TA>& Chop(Vector<T,TA> &x, typename Traits<T>::real_type tol = Traits<T>::eps()){
	typedef typename Traits<T>::real_type real_type;
	real_type m = real_type(0), t;
	for(size_t i = 0; i < x.size(); ++i){
		t = Traits<T>::abs(x[i]);
		if(t > m){ m = t; }
		else if(t < tol*m){
			x[0] = T(0);
		}
	}
	for(size_t i = 0; i < x.size(); ++i){
		if(T(0) != x[i]){
			t = Traits<T>::abs(x[i]);
			if(t < tol*m){
				x[0] = T(0);
			}
		}
	}
	return x;
}

template <typename T, typename TA>
Matrix<T,TA>& Chop(Matrix<T,TA> &a, typename Traits<T>::real_type tol = Traits<T>::eps()){
	typedef typename Traits<T>::real_type real_type;
	real_type m = real_type(0), t;
	for(size_t j = 0; j < a.cols(); ++j){
		for(size_t i = 0; i < a.rows(); ++i){
			t = Traits<T>::abs(a(i,j));
			if(t > m){ m = t; }
			else if(t < tol*m){
				a(i,j) = T(0);
			}
		}
	}
	for(size_t j = 0; j < a.cols(); ++j){
		for(size_t i = 0; i < a.rows(); ++i){
			if(T(0) != a(i,j)){
				t = Traits<T>::abs(a(i,j));
				if(t < tol*m){
					a(i,j) = T(0);
				}
			}
		}
	}
	return a;
}

}; // namespace IO
}; // namespace RNP

#endif // RNP_IOBASIC_HPP_INCLUDED
