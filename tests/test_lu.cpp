#include <RNP/LA.hpp>
#include <RNP/IOBasic.hpp>
#include <iostream>
#include <cstdlib>

double drand(){
	//return drand48() - 0.5;
	return (double)rand() / (double)RAND_MAX - 0.5;
}

int main(){
	size_t n = 4;
	RNP::Matrix<double> A(n,n);
	for(size_t j = 0; j < n; ++j){
		for(size_t i = 0; i < n; ++i){
			A(i,j) = drand();
		}
	}
	RNP::Matrix<double> B(n,n);
	RNP::TBLAS::Copy(A, B);
	
	RNP::LA::LUFactors<double> LU(A);
	//LU.Solve(RNP::NoTranspose, RNP::Left, B);
	LU.Solve(RNP::NoTranspose, RNP::Right, B);

	std::cout << RNP::IO::Chop(B) << std::endl;
	
	RNP::Matrix<double> Ainv(n,n); Ainv.Identity();
	LU.Solve(RNP::NoTranspose, RNP::Left, Ainv);
	
	LU.Mult(RNP::NoTranspose, RNP::Right, Ainv);
	
	std::cout << RNP::IO::Chop(Ainv) << std::endl;
	
	return 0;
}
