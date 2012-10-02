#include <RNP/TBLAS.h>

int main(){
	double a[4] = {1,2,3,4};
	double b[4] = {10,20,30,40};
	double x[2] = {5,6};
	double y[2] = {7,8};
	
	RNP::Matrix<double> A(2,2,a,2);
	RNP::Matrix<double> B(2,2,b,2);
	RNP::Vector<double> X(2,x,1);
	RNP::Vector<double> Y(2,y,1);
	RNP::TBLAS::Copy(A,B);
	return 0;
}
