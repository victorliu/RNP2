RNP (RNP2)
===
Second generation of the Rapid Numerical Prototyping library.

Why RNP?
========
RNP grew out of a frustration with the current common standard
numerical libraries, such as Lapack and Arpack. Much of the
code is written in Fortran, and is difficult to interface with,
and harder still to understand and tweak. I firmly believe that
it should be possible to do rapid algorithmic prototyping in a
low level systems language such as C or C++, where we can get
a decent level of abstraction without losing much in the way
of performance. Being able to prototype near production level
efficiency is very important in determining the worthiness of
an algorithm.

Since much of numerics is linear algebra, the Lapack and BLAS
routines are at the core of RNP. The use of BLAS is essentially
mandatory to achieve peak performance, and we try to operate
within the BLAS calling paradigm. However, Lapack is not nearly
as performance critical. Therefore we aim to improve upon the
basic Lapack interface. The main shortcomings of Lapack as I
see it are:

	1) Inscrutible routine names. For those well-versed in the
	   workings of the library, the names are somewhat more
	   recallable, but I won't go so far as to say they are
	   intuitive. I appreciate the Mathematica approach of using
	   long descriptive names that can be aliased if needed.

	2) Code duplication. There are essentially four versions of
	   every routine for the four different number types. This
	   represents a huge duplication of code and a maintenance
	   headache. Templates and traits classes allow us to write
	   a single version of most of the basic routines.
	
	3) Monolothic-ness. If you want to use a part of Lapack, you
	   need to link in the whole thing. Linking is not usually
	   a problem, but it can cause name clashes (for example,
	   xerbla multiply defined in various Fortran libraries).
	   The use of namespaces and aggressively partitioning
	   functionality into logically coherent units is absolutely
	   necessary when tackling a topic as wide as linear algebra.
	
	4) Inextensibility. Lapack works great if you don't need
	   anything it can't do. But, if you need to overload the
	   nonsymmetric eigenvalue solver with your own number type,
	   or change some internal workings of the algorithms, the
	   task is essentially impossible. Applying templates, we
	   can allow template specialization to easily generate
	   modified versions of routines.
	
	5) Lack of high level abstraction. Libraries that wrap Lapack
	   generally fall into two categories: thin wrappers and huge
	   frameworks. Thin wrappers do nothing to hide the complexity
	   of the underlying library, while huge frameworks such as
	   Eigen tend to abstract at too high a level. Most programming
	   languages do not possess a sufficiently rich syntax to
	   express all the operations of linear algebra, therefore it
	   is futile to try to achieve a one-to-one correspondence
	   between code and math. The success of the BLAS is a perfect
	   example of how high performance and interface robustness
	   tends to blend together when it comes to math libraries.
	    Instead of the usual vector, matrix, and higher level
	   objects like eigensolvers, we abstract at the level of
	   memory blocks. The basic unit of manipulation in Lapack is
	   a block of memory. Therefore we abstract the representation
	   of the memory rather than the actual object being stored.
	   As an example, consider an LU factorization. It is nothing
	   more than the block of memory originally holding the matrix
	   and an extra pivot array. We make no real distinction
	   between the LU factors held in an array, and the original
	   matrix that was stored within. This places a larger burden
	   on the programmer to keep things straight and consistent
	   (we do provide some mechanisms to make life easier). At
	   the end of the day, it is no harder to shoot yourself in
	   the foot than it was to interface with Lapack directly, but
	   at least the code is substantially simpler.

	6) Strange workspace query conventions. I have always found it
	   strange that Lapack returns the workspace size in a floating
	   point field rather than back in the lwork parameter.

	7) Monolithic tuning. The ILAENV routine is the single point
	   of entry for all things tuning related. This makes it
	   incredibly difficult to actually play with the tunings for
	   the inexperienced user. It makes far more sense to use traits
	   classes localized to each functional unit, so that template
	   overloads can provide for easy tuning.
	
	8) 1-based indexing. This is more of a consequence of the
	   Fortran language than a design decision, but it makes far
	   more sense to use 0-based indexing when it comes to slicing
	   and manipulating parts of matrices and vectors. We adopt the
	   convention that all indexes are actually offsets, and all
	   ranges follow the Dijkstra convention of being lower bound
	   inclusive and upper bound exclusive.

	9) Integer portability. It is always an uncertain game when
	   interfacing with Fortran from another language. One is never
	   quite sure of the sizes of data types of installed libraries.
	   First, by largely using a header-only system, we lift some
	   of the portability issues. Second, where integers should
	   represent sizes of things, we use the standard type size_t,
	   allowing for the maximum use of addressible memory. Likewise,
	   care is taken to ensure that the code is signed-unsigned safe
	   and consistent.

Scope
=====
RNP at its core provides the functionality of BLAS and Lapack, but
it is intended to include all things numeric, including what would
normally be considered combinatorial algorithms, such as matching
and graph algorithms. This is somewhat inevitable because a large
part of numerical code is nowadays focused on combinatorics (for
example, sparse matrix re-ordering is often a graph partitioning
problem). RNP is intended to be a one-stop shop for any numerical
algorithm, without requiring external dependencies. It is meant to
be modular, so that only the functionality that is used needs to
be included. And above all, it is meant to be relatively simple to
use. We do not use template metaprogramming (except in the case of
traits classes), and we do not use operator overloading to
represent mathematical operations. We try to remain C-like in our
use of C++, so that simple C wrappers can be made should they be
necessary.
 A second goal is to make a platform on which the very cutting edge
algorithms may be implemented, tested, and applied to relevent
problems. There is no greater feeling of frustration than knowing
a potentially useful algorithm lies unimplemented, or worse, is
implemented in an awkward way that cannot be easily used.
