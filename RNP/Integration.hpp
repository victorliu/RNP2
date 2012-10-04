#ifndef RNP_INTEGRATION_HPP_INCLUDED
#define RNP_INTEGRATION_HPP_INCLUDED

namespace RNP{
namespace Integration{

template <typename DomainType, typename RangeType>
int GaussKronrod(
	RangeType (*f)(const DomainType &x, void *data), void *data,
	DomainType a, DomainType b,
	RNP::Traits<RangeType>::Real epsabs,
	RNP::Traits<RangeType>::Real epsrel,
	RangeType *result, RNP::Traits<RangeType>::Real *abserr, size_t *neval);

template <typename DomainType, typename RangeType>
int AdaptiveGaussKronrod(
	RangeType (*f)(const DomainType &x, void *data), void *data,
	DomainType a, DomainType b,
	RNP::Traits<RangeType>::Real epsabs,
	RNP::Traits<RangeType>::Real epsrel,
	size_t maxsub, int key,
	RangeType *result, RNP::Traits<RangeType>::Real *abserr, void **work);

template <typename DomainType, typename RangeType>
int GaussLegendre(
	RangeType (*f)(const DomainType &x, void *data), void *data,
	DomainType a, DomainType b,
	size_t n,
	RangeType *result, void **work);

}; // namespace Integration
}; // namespace RNP

#endif // RNP_INTEGRATION_HPP_INCLUDED
