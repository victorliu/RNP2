#include <cmath>
using std::log;
using std::sqrt;
using std::log1p;
#include <cstdlib>
#include <iostream>
#include <RNP/Debug.hpp>
#include <RNP/Random.hpp>

extern "C" void dlaruv_(int *iseed, const int &n, double *x);
extern "C" void dlarnv_(const int &idist, int *iseed, const int &n, double *x);
extern "C" void zlarnv_(const int &idist, int *iseed, const int &n, std::complex<double> *x);

template <typename T>
void test_real(size_t n, int iseed[4]){
	static const size_t nbuckets = 32;
	size_t bucket[nbuckets];
	T *v = new T[n];
	
	// Test UniformRealVector uniformity
	RNP::Random::GenerateVector(RNP::Random::Distribution::Uniform01, n, v, iseed);
	// Bin into buckets
	for(size_t i = 0; i < nbuckets; ++i){
		bucket[i] = 0;
	}
	for(size_t i = 0; i < n; ++i){
		int ib = int(v[i] * (T)nbuckets);
		RNPAssert(0 <= ib && ib < (int)nbuckets);
		bucket[ib]++;
	}
	// Check variations
	{
		const T expected((T)n / (T)nbuckets);
		T maxdev(0), maxadev(0);
		for(size_t i = 0; i < nbuckets; ++i){
			const T dev = (T)bucket[i] - expected;
			const T adev = RNP::Traits<T>::abs(dev);
			if(adev > maxadev){
				maxadev = adev;
				maxdev  = dev;
			}
		}
		std::cout << " Max rel. var. of " << n << " uniforms over " << nbuckets << " buckets: " << (maxdev / expected) << std::endl;
	}
	
	// Generate standard normals
	RNP::Random::GenerateVector(RNP::Random::Distribution::Normal01, n, v, iseed);
	// Check symmetry
	{
		int npos = 0, nneg = 0;
		for(size_t i = 0; i < n; ++i){
			if(v[i] < 0){ ++nneg; }
			else{ ++npos; }
		}
		std::cout << " Std. normal asymmetry: " << ((T)(nneg - npos) / (T)n) << std::endl;
	}
	// Check first second and third central deviations
	{
		size_t nbin[3] = {0,0,0};
		for(size_t i = 0; i < n; ++i){
			if(RNP::Traits<T>::abs(v[i]) < T(1)){
				++nbin[0]; ++nbin[1]; ++nbin[2];
			}else if(RNP::Traits<T>::abs(v[i]) < T(2)){
				++nbin[1]; ++nbin[2];
			}else if(RNP::Traits<T>::abs(v[i]) < T(3)){
				++nbin[2];
			}
		}
		// Should be 68-95-99.7%
		std::cout << " Std. normals within (1,2,3) stdevs: " << ((T)nbin[0] / (T)n) << ", " << ((T)nbin[1] / (T)n) << ", " << ((T)nbin[2] / (T)n) << std::endl;
	}
	
	delete [] v;
}

template <typename T>
void test_complex(size_t n, int iseed[4]){
	typedef std::complex<T> complex_type;
	typedef T real_type;
	static const size_t nbuckets = 32;
	size_t bucketr[nbuckets];
	size_t bucketi[nbuckets];
	complex_type *v = new complex_type[n];
	
	// Test UniformRealVector uniformity
	RNP::Random::GenerateVector(RNP::Random::Distribution::Uniform01, n, v, iseed);
	// Bin into buckets
	for(size_t i = 0; i < nbuckets; ++i){
		bucketr[i] = 0;
		bucketi[i] = 0;
	}
	for(size_t i = 0; i < n; ++i){
		int ib;
		ib = int(v[i].real() * (real_type)nbuckets);
		RNPAssert(0 <= ib && ib < (int)nbuckets);
		bucketr[ib]++;
		ib = int(v[i].imag() * (real_type)nbuckets);
		RNPAssert(0 <= ib && ib < (int)nbuckets);
		bucketi[ib]++;
	}
	// Check variations
	{
		const real_type expected((real_type)n / (real_type)nbuckets);
		real_type maxdev(0), maxadev(0);
		for(size_t i = 0; i < nbuckets; ++i){
			const real_type dev = (real_type)bucketr[i] - expected;
			const real_type adev = RNP::Traits<real_type>::abs(dev);
			if(adev > maxadev){
				maxadev = adev;
				maxdev  = dev;
			}
		}
		std::cout << " Max rel. var. of " << n << " uniforms over " << nbuckets << " buckets: " << (maxdev / expected) << std::endl;
	}
	{
		const real_type expected((real_type)n / (real_type)nbuckets);
		real_type maxdev(0), maxadev(0);
		for(size_t i = 0; i < nbuckets; ++i){
			const real_type dev = (real_type)bucketi[i] - expected;
			const real_type adev = RNP::Traits<real_type>::abs(dev);
			if(adev > maxadev){
				maxadev = adev;
				maxdev  = dev;
			}
		}
		std::cout << " Max rel. var. of " << n << " uniforms over " << nbuckets << " buckets: " << (maxdev / expected) << std::endl;
	}
	
	// Generate standard normals
	RNP::Random::GenerateVector(RNP::Random::Distribution::Normal01, n, v, iseed);
	// Check symmetry
	{
		int npos = 0, nneg = 0;
		for(size_t i = 0; i < n; ++i){
			if(v[i].real() < 0){ ++nneg; }
			else{ ++npos; }
		}
		std::cout << " Std. normal asymmetry: " << ((real_type)(nneg - npos) / (real_type)n) << std::endl;
	}
	{
		int npos = 0, nneg = 0;
		for(size_t i = 0; i < n; ++i){
			if(v[i].imag() < 0){ ++nneg; }
			else{ ++npos; }
		}
		std::cout << " Std. normal asymmetry: " << ((real_type)(nneg - npos) / (real_type)n) << std::endl;
	}
	// Check first second and third central deviations
	{
		size_t nbin[3] = {0,0,0};
		for(size_t i = 0; i < n; ++i){
			real_type a = RNP::Traits<real_type>::abs(v[i].real());
			if(a < real_type(1)){
				++nbin[0]; ++nbin[1]; ++nbin[2];
			}else if(a < real_type(2)){
				++nbin[1]; ++nbin[2];
			}else if(a < real_type(3)){
				++nbin[2];
			}
		}
		// Should be 68-95-99.7%
		std::cout << " Std. normals within (1,2,3) stdevs: "
			<< ((real_type)nbin[0] / (real_type)n) << ", "
			<< ((real_type)nbin[1] / (real_type)n) << ", "
			<< ((real_type)nbin[2] / (real_type)n) << std::endl;
	}
	{
		size_t nbin[3] = {0,0,0};
		for(size_t i = 0; i < n; ++i){
			real_type a = RNP::Traits<real_type>::abs(v[i].imag());
			if(a < real_type(1)){
				++nbin[0]; ++nbin[1]; ++nbin[2];
			}else if(a < real_type(2)){
				++nbin[1]; ++nbin[2];
			}else if(a < real_type(3)){
				++nbin[2];
			}
		}
		// Should be 68-95-99.7%
		std::cout << " Std. normals within (1,2,3) stdevs: "
			<< ((real_type)nbin[0] / (real_type)n) << ", "
			<< ((real_type)nbin[1] / (real_type)n) << ", "
			<< ((real_type)nbin[2] / (real_type)n) << std::endl;
	}
	
	delete [] v;
}

int main(){
	int iseed[4] = { 2398, 691, 782, 721 };
	size_t nrnd = 999999;
	
	std::cout << "Testing complex<double>" << std::endl;
	test_complex<double>(nrnd, iseed);
	std::cout << "Testing complex<float>" << std::endl;
	test_complex<float>(nrnd, iseed);
	std::cout << "Testing double" << std::endl;
	test_real<double>(nrnd, iseed);
	std::cout << "Testing float" << std::endl;
	test_real<float>(nrnd, iseed);
	
	return EXIT_SUCCESS;
}
