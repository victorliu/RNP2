#ifndef RNP_SORT_HPP_INCLUDED
#define RNP_SORT_HPP_INCLUDED

#include <algorithm>

namespace RNP{
namespace Sort{

namespace Util{

template <typename T>
struct ByIndexIncreasing{
	const T *val;
	ByIndexIncreasing(const T *v):val(v){}
	bool operator()(size_t i, size_t j) const{ return val[i] < val[j]; }
};
template <typename T>
struct ByIndexDecreasing{
	const T *val;
	ByIndexDecreasing(const T *v):val(v){}
	bool operator()(size_t i, size_t j) const{ return val[i] > val[j]; }
};

} // namespace Util

// Returns an index vector ord such that values[ord[i]] is sorted.
// order can be "I" or "D" for increasing or decreasing.
template <typename T>
inline void ByIndex(const char *order, size_t n, const T *values, size_t *ord){
	for(size_t i = 0; i < n; ++i){
		ord[i] = i;
	}
	if('I' == order[0]){
		std::sort(ord, ord+n, Util::ByIndexIncreasing<T>(values));
	}else{
		std::sort(ord, ord+n, Util::ByIndexDecreasing<T>(values));
	}
}

} // namespace Sort
} // namespace RNP

#endif // RNP_SORT_HPP_INCLUDED
