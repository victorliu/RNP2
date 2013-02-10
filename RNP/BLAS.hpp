#ifndef RNP_BLAS_HPP_INCLUDED
#define RNP_BLAS_HPP_INCLUDED

///////////////////////////////////////////////////////////////////////
// BLAS
// ====
// This file is used to select the underying BLAS implementation.
// It is recommended that BLAS_mix.hpp be used, since using external
// Level 1 BLAS tends to introduce interfacing subtleties.
//

// Select this line to use external BLAS
//#include "BLAS/BLAS_ext.hpp"

// Select this line to use a mixture of templates and external BLAS
#include "BLAS/BLAS_mix.hpp"

#include "BLAS/BLAS_obj.hpp"

#endif // RNP_BLAS_HPP_INCLUDED
