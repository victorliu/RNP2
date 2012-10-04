#include <iostream>
#include <RNP/Integration.hpp>

double f(const double &x, void *data){
	std::cout << "x = " << x << std::endl;
	return x*x;
}

int main(){
	double result, err; size_t neval;
	
	//RNP::Integration::GaussKronrod<double,double>(&f, NULL, 0., 1., 0., 0.001, &result, &err, &neval);
	
	void *work = NULL;
	RNP::Integration::GaussLegendre<double,double>(&f, NULL, 0., 1., 2, &result, &work);
	
	std::cout << result << " " << err << " " << neval << std::endl;
	return 0;
}
