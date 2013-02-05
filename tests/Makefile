CFLAGS = -Wall -O0 -ggdb -DRNP_ENABLE_DEBUG -std=c++11
DEBUGLIB = /usr/lib/libduma.a -lpthread -llapack
LIBS = -lblas $(DEBUGLIB)

TESTS = \
	test_zeig \
	test_hesstri_reduce \
	test_hess_reduce \
	test_herm_eig \
	test_tridiag_herm_reduce \
	test_symtridiag_qr \
	test_rq_tall \
	test_rq_fat \
	test_lq_tall \
	test_lq_fat \
	test_ql_tall \
	test_ql_fat \
	test_qr_tall \
	test_qr_fat \
	test_random \
	test_trigeneigvec \
	test_trieigvec \
	test_trisolve \
	test_triangular \
	test_lu

test_random: test_random.cpp
	g++ -std=c++11 -Wall test_random.cpp -I.. -o $@
test_triangular: test_triangular.cpp
	g++ -Wall test_triangular.cpp -I.. ../src/BLAS_mix.cpp $(LIBS) -o $@
test_trisolve: test_trisolve.cpp
	g++ -Wall test_trisolve.cpp -I.. ../src/BLAS_mix.cpp $(LIBS) -o $@
test_trieigvec: test_trieigvec.cpp
	g++ -Wall test_trieigvec.cpp -I.. ../src/BLAS_mix.cpp $(LIBS) -o $@
test_trigeneigvec: test_trigeneigvec.cpp
	g++ -Wall test_trigeneigvec.cpp -I.. ../src/BLAS_mix.cpp $(LIBS) -o $@
test_lu: test_lu.cpp
	g++ $(CFLAGS) test_lu.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -o $@
test_qr_fat: test_qr_fat.cpp
	g++ $(CFLAGS) test_qr_fat.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -o $@
test_qr_tall: test_qr_tall.cpp
	g++ $(CFLAGS) test_qr_tall.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -o $@
test_ql_tall: test_ql_tall.cpp
	g++ $(CFLAGS) test_ql_tall.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -llapack -o $@
test_ql_fat: test_ql_fat.cpp
	g++ $(CFLAGS) test_ql_fat.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -llapack -o $@
test_lq_tall: test_lq_tall.cpp
	g++ $(CFLAGS) test_lq_tall.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -llapack -o $@
test_lq_fat: test_lq_fat.cpp
	g++ $(CFLAGS) test_lq_fat.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -llapack -o $@
test_rq_tall: test_rq_tall.cpp
	g++ $(CFLAGS) test_rq_tall.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -llapack -o $@
test_rq_fat: test_rq_fat.cpp
	g++ $(CFLAGS) test_rq_fat.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -llapack -o $@
test_symtridiag_qr: test_symtridiag_qr.cpp
	g++ $(CFLAGS) test_symtridiag_qr.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -o $@
test_tridiag_herm_reduce: test_tridiag_herm_reduce.cpp
	g++ $(CFLAGS) test_tridiag_herm_reduce.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -o $@
test_herm_eig: test_herm_eig.cpp
	g++ $(CFLAGS) test_herm_eig.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -o $@
test_hess_reduce: test_hess_reduce.cpp
	g++ $(CFLAGS) test_hess_reduce.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -o $@
test_hesstri_reduce: test_hesstri_reduce.cpp
	g++ $(CFLAGS) test_hesstri_reduce.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -o $@
test_zeig: test_zeig.cpp
	g++ $(CFLAGS) test_zeig.cpp ../src/BLAS_mix.cpp -I.. $(LIBS) -o $@
#test_lu_obj: test_lu_obj.cpp
#	g++ $(CFLAGS) test_lu_obj.cpp ../src/BLAS_mix.cpp -I.. -lblas -o $@
#test_qr_load: test_qr_load.cpp
#	g++ $(CFLAGS) test_qr_load.cpp ../src/BLAS_mix.cpp -I.. -lblas -o $@

clean:
	rm -f $(TESTS)
