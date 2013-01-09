CFLAGS = -Wall -O0 -ggdb -DRNP_ENABLE_DEBUG

TESTS = \
	test_random \
	test_triangular \
	test_lu

test_random: test_random.cpp
	g++ -std=c++11 -Wall test_random.cpp -I.. -o $@
test_triangular: test_triangular.cpp
	g++ -Wall test_triangular.cpp -I.. ../src/BLAS_mix.cpp -lblas -o $@
test_lu: test_lu.cpp
	g++ $(CFLAGS) test_lu.cpp ../src/BLAS_mix.cpp -I.. -lblas -o $@
test_qr_fat: test_qr_fat.cpp
	g++ $(CFLAGS) test_qr_fat.cpp ../src/BLAS_mix.cpp -I.. -lblas -o $@
test_symtridiag_qr: test_symtridiag_qr.cpp
	g++ $(CFLAGS) test_symtridiag_qr.cpp ../src/BLAS_mix.cpp -I.. -lblas -o $@
#test_qr_tall: test_qr_tall.cpp
#	g++ $(CFLAGS) test_qr_tall.cpp ../src/BLAS_mix.cpp -I.. -lblas -o $@
#test_lu_obj: test_lu_obj.cpp
#	g++ $(CFLAGS) test_lu_obj.cpp ../src/BLAS_mix.cpp -I.. -lblas -o $@
#test_qr_load: test_qr_load.cpp
#	g++ $(CFLAGS) test_qr_load.cpp ../src/BLAS_mix.cpp -I.. -lblas -o $@

clean:
	rm -f $(TESTS)
