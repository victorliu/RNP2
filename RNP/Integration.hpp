#ifndef RNP_INTEGRATION_HPP_INCLUDED
#define RNP_INTEGRATION_HPP_INCLUDED

#include <RNP/Types.hpp>
#include <limits>
#include <float.h>

namespace RNP{
namespace Integration{

namespace Tables{
// Gauss-Kronrod-Patterson quadrature coefficients for use in
// quadpack routine qng. These coefficients were calculated with
// 101 decimal digit arithmetic by L. W. Fullerton, Bell Labs, Nov
// 1981.

// x1, abscissae common to the 10-, 21-, 43- and 87-point rule
static const double x1[5] = {
  0.973906528517171720077964012084452,
  0.865063366688984510732096688423493,
  0.679409568299024406234327365114874,
  0.433395394129247190799265943165784,
  0.148874338981631210884826001129720
};

// w10, weights of the 10-point formula
static const double w10[5] = {
  0.066671344308688137593568809893332,
  0.149451349150580593145776339657697,
  0.219086362515982043995534934228163,
  0.269266719309996355091226921569469,
  0.295524224714752870173892994651338
};

// x2, abscissae common to the 21-, 43- and 87-point rule
static const double x2[5] = {
  0.995657163025808080735527280689003,
  0.930157491355708226001207180059508,
  0.780817726586416897063717578345042,
  0.562757134668604683339000099272694,
  0.294392862701460198131126603103866
};

// w21a, weights of the 21-point formula for abscissae x1
static const double w21a[5] = {
  0.032558162307964727478818972459390,
  0.075039674810919952767043140916190,
  0.109387158802297641899210590325805,
  0.134709217311473325928054001771707,
  0.147739104901338491374841515972068
};

// w21b, weights of the 21-point formula for abscissae x2
static const double w21b[6] = {
  0.011694638867371874278064396062192,
  0.054755896574351996031381300244580,
  0.093125454583697605535065465083366,
  0.123491976262065851077958109831074,
  0.142775938577060080797094273138717,
  0.149445554002916905664936468389821
};

// x3, abscissae common to the 43- and 87-point rule
static const double x3[11] = {
  0.999333360901932081394099323919911,
  0.987433402908088869795961478381209,
  0.954807934814266299257919200290473,
  0.900148695748328293625099494069092,
  0.825198314983114150847066732588520,
  0.732148388989304982612354848755461,
  0.622847970537725238641159120344323,
  0.499479574071056499952214885499755,
  0.364901661346580768043989548502644,
  0.222254919776601296498260928066212,
  0.074650617461383322043914435796506
};

// w43a, weights of the 43-point formula for abscissae x1, x3
static const double w43a[10] = {
  0.016296734289666564924281974617663,
  0.037522876120869501461613795898115,
  0.054694902058255442147212685465005,
  0.067355414609478086075553166302174,
  0.073870199632393953432140695251367,
  0.005768556059769796184184327908655,
  0.027371890593248842081276069289151,
  0.046560826910428830743339154433824,
  0.061744995201442564496240336030883,
  0.071387267268693397768559114425516
};

// w43b, weights of the 43-point formula for abscissae x3
static const double w43b[12] = {
  0.001844477640212414100389106552965,
  0.010798689585891651740465406741293,
  0.021895363867795428102523123075149,
  0.032597463975345689443882222526137,
  0.042163137935191811847627924327955,
  0.050741939600184577780189020092084,
  0.058379395542619248375475369330206,
  0.064746404951445885544689259517511,
  0.069566197912356484528633315038405,
  0.072824441471833208150939535192842,
  0.074507751014175118273571813842889,
  0.074722147517403005594425168280423
};

// x4, abscissae of the 87-point rule
static const double x4[22] = {
  0.999902977262729234490529830591582,
  0.997989895986678745427496322365960,
  0.992175497860687222808523352251425,
  0.981358163572712773571916941623894,
  0.965057623858384619128284110607926,
  0.943167613133670596816416634507426,
  0.915806414685507209591826430720050,
  0.883221657771316501372117548744163,
  0.845710748462415666605902011504855,
  0.803557658035230982788739474980964,
  0.757005730685495558328942793432020,
  0.706273209787321819824094274740840,
  0.651589466501177922534422205016736,
  0.593223374057961088875273770349144,
  0.531493605970831932285268948562671,
  0.466763623042022844871966781659270,
  0.399424847859218804732101665817923,
  0.329874877106188288265053371824597,
  0.258503559202161551802280975429025,
  0.185695396568346652015917141167606,
  0.111842213179907468172398359241362,
  0.037352123394619870814998165437704
};

// w87a, weights of the 87-point formula for abscissae x1, x2, x3
static const double w87a[21] = {
  0.008148377384149172900002878448190,
  0.018761438201562822243935059003794,
  0.027347451050052286161582829741283,
  0.033677707311637930046581056957588,
  0.036935099820427907614589586742499,
  0.002884872430211530501334156248695,
  0.013685946022712701888950035273128,
  0.023280413502888311123409291030404,
  0.030872497611713358675466394126442,
  0.035693633639418770719351355457044,
  0.000915283345202241360843392549948,
  0.005399280219300471367738743391053,
  0.010947679601118931134327826856808,
  0.016298731696787335262665703223280,
  0.021081568889203835112433060188190,
  0.025370969769253827243467999831710,
  0.029189697756475752501446154084920,
  0.032373202467202789685788194889595,
  0.034783098950365142750781997949596,
  0.036412220731351787562801163687577,
  0.037253875503047708539592001191226
};

// w87b, weights of the 87-point formula for abscissae x4
static const double w87b[23] = {
  0.000274145563762072350016527092881,
  0.001807124155057942948341311753254,
  0.004096869282759164864458070683480,
  0.006758290051847378699816577897424,
  0.009549957672201646536053581325377,
  0.012329447652244853694626639963780,
  0.015010447346388952376697286041943,
  0.017548967986243191099665352925900,
  0.019938037786440888202278192730714,
  0.022194935961012286796332102959499,
  0.024339147126000805470360647041454,
  0.026374505414839207241503786552615,
  0.028286910788771200659968002987960,
  0.030052581128092695322521110347341,
  0.031646751371439929404586051078883,
  0.033050413419978503290785944862689,
  0.034255099704226061787082821046821,
  0.035262412660156681033782717998428,
  0.036076989622888701185500318003895,
  0.036698604498456094498018047441094,
  0.037120549269832576114119958413599,
  0.037334228751935040321235449094698,
  0.037361073762679023410321241766599
};

}; // namespace Tables

template <class T>
T _rescale_error(T err, const T result_abs, const T result_asc){
	typedef typename RNP::Traits<T> TTraits;
	err = TTraits::abs(err);

	if(result_asc != 0 && err != 0){
		T scale = pow((200 * err / result_asc), 1.5);
		if(scale < 1){
			err = result_asc * scale;
		}else{
			err = result_asc;
		}
	}
	
	if(result_abs > TTraits::minimum() / (50 * TTraits::epsilon())){
		T min_err = 50 * TTraits::epsilon() * result_abs;
		if(min_err > err){
			err = min_err;
		}
	}

	return err;
}

// Returns:
//   0 on success
//  -n if n-th argument is invalid
//   1 if tolerance not achieved
template <typename DomainType, typename RangeType>
int GaussKronrod(
	RangeType (*f)(const DomainType &x, void *data), void *data,
	DomainType a, DomainType b,
	typename RNP::Traits<RangeType>::Real epsabs,
	typename RNP::Traits<RangeType>::Real epsrel,
	RangeType *result, typename RNP::Traits<RangeType>::Real *abserr, size_t *neval
){
	typedef typename RNP::Traits<RangeType> RangeTraits;
	typedef typename RNP::Traits<DomainType> DomainTraits;
	typedef typename RangeTraits::Real error_type;
	RangeType fv1[5], fv2[5], fv3[5], fv4[5];
	RangeType savfun[21];  // array of function values which have been computed
	RangeType res10, res21, res43, res87; // 10, 21, 43 and 87 point results
	RangeType result_kronrod;
	error_type err;
	RangeType resabs; // approximation to the integral of abs(f)
	RangeType resasc; // approximation to the integral of abs(f-i/(b-a))

	const DomainType half_length =  0.5 * (b - a);
	const DomainType abs_half_length = RNP::Traits<DomainType>::abs(half_length);
	const DomainType center = 0.5 * (b + a);
	const RangeType f_center = f(center, data);

	if(epsabs <= 0 && (epsrel < 50 * DBL_EPSILON || epsrel < 0.5e-28)){
		*result = 0;
		*abserr = 0;
		*neval = 0;
	};

	// Compute the integral using the 10- and 21-point formula.
	res10 = 0;
	res21 = Tables::w21b[5] * f_center;
	resabs = Tables::w21b[5] * RangeTraits::abs(f_center);

	for(size_t k = 0; k < 5; k++){
		const DomainType abscissa = half_length * Tables::x1[k];
		const RangeType fval1 = f(center + abscissa, data);
		const RangeType fval2 = f(center - abscissa, data);
		const RangeType fval = fval1 + fval2;
		res10 += Tables::w10[k] * fval;
		res21 += Tables::w21a[k] * fval;
		resabs += Tables::w21a[k] * (RangeTraits::abs(fval1) + RangeTraits::abs(fval2));
		savfun[k] = fval;
		fv1[k] = fval1;
		fv2[k] = fval2;
	}

	for(size_t k = 0; k < 5; k++){
		const DomainType abscissa = half_length * Tables::x2[k];
		const RangeType fval1 = f(center + abscissa, data);
		const RangeType fval2 = f(center - abscissa, data);
		const RangeType fval = fval1 + fval2;
		res21 += Tables::w21b[k] * fval;
		resabs += Tables::w21b[k] * (RangeTraits::abs(fval1) + RangeTraits::abs(fval2));
		savfun[k + 5] = fval;
		fv3[k] = fval1;
		fv4[k] = fval2;
	}

	resabs *= abs_half_length;

	{
		const RangeType mean = 0.5 * res21;

		resasc = Tables::w21b[5] * RangeTraits::abs(f_center - mean);

		for(size_t k = 0; k < 5; k++){
			resasc +=
				(Tables::w21a[k] * (RangeTraits::abs(fv1[k] - mean) + RangeTraits::abs(fv2[k] - mean))
				+ Tables::w21b[k] * (RangeTraits::abs(fv3[k] - mean) + RangeTraits::abs(fv4[k] - mean)));
		}
		resasc *= abs_half_length;
	}

	result_kronrod = res21 * half_length;

	err = _rescale_error((res21 - res10) * half_length, resabs, resasc);

	// test for convergence.

	if (err < epsabs || err < epsrel * RangeTraits::abs(result_kronrod)){
		*result = result_kronrod;
		*abserr = err;
		*neval = 21;
		return 0;
	}

	// compute the integral using the 43-point formula.

	res43 = Tables::w43b[11] * f_center;

	for(size_t k = 0; k < 10; k++){
		res43 += savfun[k] * Tables::w43a[k];
	}

	for(size_t k = 0; k < 11; k++){
		const DomainType abscissa = half_length * Tables::x3[k];
		const RangeType fval = (f(center + abscissa, data) + f(center - abscissa, data));
		res43 += fval * Tables::w43b[k];
		savfun[k + 10] = fval;
	}

	// test for convergence

	result_kronrod = res43 * half_length;
	err = _rescale_error((res43 - res21) * half_length, resabs, resasc);

	if(err < epsabs || err < epsrel * fabs (result_kronrod)){
		*result = result_kronrod;
		*abserr = err;
		*neval = 43;
		return 0;
	}

	// compute the integral using the 87-point formula.

	res87 = Tables::w87b[22] * f_center;

	for(size_t k = 0; k < 21; k++){
		res87 += savfun[k] * Tables::w87a[k];
	}

	for(size_t k = 0; k < 22; k++){
		const double abscissa = half_length * Tables::x4[k];
		res87 += Tables::w87b[k] * (f(center + abscissa, data) + f(center - abscissa, data));
	}

	// test for convergence

	result_kronrod = res87 * half_length ;

	err = _rescale_error((res87 - res43) * half_length, resabs, resasc);

	if(err < epsabs || err < epsrel * fabs (result_kronrod)){
		*result = result_kronrod;
		*abserr = err;
		*neval = 87;
		return 0;
	}

	// failed to converge

	*result = result_kronrod;
	*abserr = err;
	*neval = 87;
	return 1;
}

template <typename DomainType, typename RangeType>
int AdaptiveGaussKronrod(
	RangeType (*f)(const DomainType &x, void *data), void *data,
	DomainType a, DomainType b,
	typename RNP::Traits<RangeType>::Real epsabs,
	typename RNP::Traits<RangeType>::Real epsrel,
	size_t maxsub, int key,
	RangeType *result, typename RNP::Traits<RangeType>::Real *abserr, void **work
){
	return 0;
}

template <typename DomainType>
struct GaussLegendreTable{
	size_t n;
	DomainType xw[2];
};

// Exact results for polynomial integrands of order 2*n-1
template <typename DomainType, typename RangeType>
int GaussLegendre(
	RangeType (*f)(const DomainType &x, void *data), void *data,
	DomainType a, DomainType b,
	size_t n,
	RangeType *result, void **work
){
	typedef GaussLegendreTable<DomainType> TableType;
	TableType *table = (TableType*)(*work);
	const size_t m = (n + 1)/2;
	if(NULL == table || table->n != n){
		table = (TableType*)realloc(table, sizeof(TableType) + 2*sizeof(DomainType) * (m-1));
		table->n = n;
		*work = table;

		DomainType *x = table->xw + 0;
		DomainType *w = table->xw + m;
		
		{
			DomainType eps = 1e-10;
			DomainType x0,  x1,  dx; /* Abscissas */
			DomainType w0,  w1,  dw; /* Weights */
			DomainType P0, P_1, P_2; /* Legendre polynomial values */
			DomainType dpdx;   /* Legendre polynomial derivative */
			DomainType t0, t1, t2, t3;

			t0 = (1.0 - (1.0 - 1.0 / (DomainType)n) / (8.0 * (DomainType)n * (DomainType)n));
			t1 = 1.0 / (4.0 * (DomainType)n + 2.0);

			for(size_t i = 1; i <= m; i++){
				// Find i-th root of Legendre polynomial

				// Initial guess
				const double pi_to_many_places = 3.1415926535897932384626433832795028841971693993751;
				x0 = cos(pi_to_many_places * (double)((i << 2) - 1) * t1) * t0;

				/* Newton iterations, at least one */
				size_t j = 0;
				dx = dw = DBL_MAX;

				do{
					/* Compute Legendre polynomial value at x0 */
					P_1 = 1.0;
					P0  = x0;

					for(size_t k = 2; k <= n; k++){
						P_2 = P_1;
						P_1 = P0;
						t2 = x0 * P_1;
						t3 = (double)(k - 1) / (double)k;

						P0 = t2 + t3 * (t2 - P_2);
					}

					/* Compute Legendre polynomial derivative at x0 */
					dpdx = ((x0 * P0 - P_1) * (double)n) / (x0 * x0 - 1.0);

					/* Newton step */
					x1 = x0 - P0 / dpdx;

					/* Weight computing */
					w1 = 2.0 / ((1.0 - x1 * x1) * dpdx * dpdx);

					/* Compute weight w0 on first iteration, needed for dw */
					if (j == 0) w0 = 2.0 / ((1.0 - x0 * x0) * dpdx * dpdx);

					dx = x0 - x1;
					dw = w0 - w1;
					x0 = x1;
					w0 = w1;

					j++;
				}
				while((fabs(dx) > eps || fabs(dw) > eps) && j < 100);
				x[(m-1)-(i-1)] = x1;
				w[(m-1)-(i-1)] = w1;
				std::cout << x1 << " " << w1 << std::endl;
			}
		}
	}
	
	const DomainType *const x = table->xw + 0;
	const DomainType *const w = table->xw + m;
	DomainType A, B, Ax, s;

	A = 0.5 * (b - a);
	B = 0.5 * (b + a);

	if(n&1){ // n odd
		s = w[0] * f(B, data);
		for(size_t i = 1; i < m; i++){
			Ax = A * x[i];
			s += w[i] * (f(B+Ax, data) + f(B-Ax, data));
		}
	}else{ // n even
		s = 0;
		for(size_t i = 0; i < m; i++){
			Ax = A * x[i];
			s += w[i] * (f(B+Ax, data) + f(B-Ax, data));
		}
	}
	*result = A*s;
}

}; // namespace Integration
}; // namespace RNP

#endif // RNP_INTEGRATION_HPP_INCLUDED
