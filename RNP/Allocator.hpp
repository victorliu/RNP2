#ifndef RNP_ALLOCATOR_HPP_INCLUDED
#define RNP_ALLOCATOR_HPP_INCLUDED

#include <RNP/Tuning.hpp>

// By default, we assume malloc is already sufficiently aligned.
// On Windows, the CRT provides aligned allocation routines.
// If manual alignment is needed (guaranteed to work), then define
//   RNP_ALLOCATOR_MANUALLY_ALIGN

#if defined(WIN32)
# include <malloc.h>
  extern "C" {
    // Link with msvcrt
    void* _aligned_malloc(size_t size, size_t alignment);
    void  _aligned_free  (void *ptr);
  }
#elif defined(RNP_ALLOCATOR_MANUALLY_ALIGN)
# include <cstdlib>
# include <cstdio>
# include <inttypes.h>
  typedef uintptr_t malloc_aligned_ULONG_PTR;
#endif

namespace RNP{

template <class T>
class Allocator{
public:
	static T* Allocate(size_t count, bool zero = false){
#if defined(WIN32)
		return (T*)_aligned_malloc(count * sizeof(T), RNP::Tuning::MemoryAlignment);
#elif defined(RNP_ALLOCATOR_MANUALLY_ALIGN)
		void *pa, *ptr;
		// Allocate enough space plus alignment padding, plus pointer to original start of block
		static const size_t extra = RNP::Tuning::MemoryAlignment - 1 + sizeof(void*);
		pa = malloc(count*sizeof(T) + extra);
		if(!pa){ return NULL; }

		malloc_aligned_ULONG_PTR mask = (~(RNP::Tuning::MemoryAlignment-1));
		// Actual returned pointer
		ptr = (T*)( ((malloc_aligned_ULONG_PTR)pa + extra) & mask );
		// Store the original starting pointer just before
		*((void **)ptr-1) = pa;
		
		return (T*)ptr;
#else
		return (T*)malloc(count * sizeof(T));
#endif
	}
	static void Resize(T **pptr, size_t newcount){
		static const size_t alignment = RNP::Tuning::MemoryAlignment;
#if defined(WIN32)
		*pptr = (T*)_aligned_realloc(*pptr, newcount * sizeof(T), alignment);
#elif defined(RNP_ALLOCATOR_MANUALLY_ALIGN)
		void *pa = *((void **)(*ptr)-1);
		static const size_t extra = RNP::Tuning::MemoryAlignment - 1 + sizeof(void*);
		pa = realloc(pa, newcount*sizeof(T) + extra);
		if(!pa){ *pptr = NULL; }

		malloc_aligned_ULONG_PTR mask = (~(RNP::Tuning::MemoryAlignment-1));
		ptr = (T*)( ((malloc_aligned_ULONG_PTR)pa + extra) & mask );
		// Store the original starting pointer just before
		*((void **)(*ptr)-1) = pa;
#else
		*pptr = realloc(*pptr, newcount * sizeof(T));
#endif
	}
	static void Free(T *ptr){
#if defined(WIN32)
		_aligned_free(ptr);
#elif defined(RNP_ALLOCATOR_MANUALLY_ALIGN)
		if(ptr){
			free(*((void **)ptr-1));
		}
#else
		free(ptr);
#endif
	}
};

}; // namespace RNP

#endif // RNP_ALLOCATOR_HPP_INCLUDED
