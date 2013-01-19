#ifndef RNP_DEBUG_HPP_INCLUDED
#define RNP_DEBUG_HPP_INCLUDED

#ifndef NDEBUG

#include <iostream>

#define RNPAssertStringify(s) #s
#define RNPAssert(COND) RNP::Assert(COND, __FILE__, __LINE__, RNPAssertStringify(COND))

namespace RNP{

inline void Assert(bool cond, const char *file, int line, const char *str){
	if(!cond){
		std::cerr << "Assertion failed in " << file << ":" << line << ": " << str << std::endl;
	}
}

#else

#define RNPAssert(COND)

#endif

}; // namespace RNP

#endif // RNP_DEBUG_HPP_INCLUDED
