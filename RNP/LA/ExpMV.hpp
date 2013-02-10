namespace RNP{
namespace LA{
namespace MatrixFunctions{

namespace Util{
namespace ExpMV{

template <typename T>
struct Constants{
	static const size_t ntheta = 0;
	static const T theta[] = {
		T(0)
	};
};

template <>
struct Constants<double>{
	static const size_t ntheta = 100;
	// Let tol = 2^-53
	// htilde[mp1_, x_] := Exp[x] - Sum[c[k] x^k, {k, 0, mp1 - 1}];
	// theta[m_]:= q /. FindRoot[htilde[m + 1, q] == tol*q, {q, m}];
	// except this won't work since we need to work at much higher precision
	static const double theta[] = {
		0., // First entry must be zero and is ignored
		2.2204460492503131e-016, // for m = 1
		2.5809568029946243e-008, //     m = 2
		1.3863478661191185e-005, //     etc.
		3.3971688399768305e-004,
		2.4008763578872742e-003,
		9.0656564075951018e-003,
		2.3844555325002736e-002,
		4.9912288711153226e-002,
		8.9577602032233430e-002,
		1.4418297616143780e-001,
		2.1423580684517107e-001,
		2.9961589138115802e-001,
		3.9977753363167950e-001,
		5.1391469361242936e-001,
		6.4108352330411988e-001,
		7.8028742566265763e-001,
		9.3053284607865683e-001,
		1.0908637192900361e+000,
		1.2603810606426387e+000,
		1.4382525968043369e+000,
		1.6237159502358214e+000,
		1.8160778162150852e+000,
		2.0147107809446161e+000,
		2.2190488693650896e+000,
		2.4285825244428265e+000,
		2.6428534574594353e+000,
		2.8614496339342641e+000,
		3.0840005449891619e+000,
		3.3101728398902708e+000,
		3.5396663487436895e+000,
		3.7722104956817510e+000,
		4.0075610861180397e+000,
		4.2454974425796959e+000,
		4.4858198594473686e+000,
		4.7283473457935390e+000,
		4.9729156261919814e+000,
		5.2193753710840580e+000,
		5.4675906305245441e+000,
		5.7174374475720127e+000,
		5.9688026300418491e+000,
		6.2215826616898910e+000,
		6.4756827360799845e+000,
		6.7310158983810240e+000,
		6.9875022821306301e+000,
		7.2450684295979526e+000,
		7.5036466857888637e+000,
		7.7631746573779870e+000,
		8.0235947289399796e+000,
		8.2848536298039175e+000,
		8.5469020456849325e+000,
		8.8096942699713221e+000,
		9.0731878901761451e+000,
		9.3373435056120133e+000,
		9.6021244728265565e+000,
		9.8674966757534008e+000,
		1.0133428317897478e+001,
		1.0399889734191031e+001,
		1.0666853220434106e+001,
		1.0934292878475777e+001,
		1.1202184475504579e+001,
		1.1470505316002537e+001,
		1.1739234125080184e+001,
		1.2008350942053168e+001,
		1.2277837023246892e+001,
		1.2547674753126438e+001,
		1.2817847562946627e+001,
		1.3088339856203294e+001,
		1.3359136940242903e+001,
		1.3630224963455026e+001,
		1.3901590857531859e+001,
		1.4173222284331819e+001,
		1.4445107586931254e+001,
		1.4717235744490083e+001,
		1.4989596330594328e+001,
		1.5262179474771679e+001,
		1.5534975826905702e+001,
		1.5807976524300871e+001,
		1.6081173161174046e+001,
		1.6354557760369328e+001,
		1.6628122747112073e+001,
		1.6901860924634942e+001,
		1.7175765451524093e+001,
		1.7449829820647437e+001,
		1.7724047839539214e+001,
		1.7998413612126303e+001,
		1.8272921521691810e+001,
		1.8547566214980510e+001,
		1.8822342587358953e+001,
		1.9097245768950550e+001,
		1.9372271111672617e+001,
		1.9647414177108576e+001,
		1.9922670725154251e+001,
		2.0198036703383082e+001,
		2.0473508237083141e+001,
		2.0749081619935929e+001,
		2.1024753305356054e+001,
		2.1300519898663481e+001,
		2.1576378150721375e+001,
		2.1852324954990110e+001,
		2.2128357353464594e+001
	};
};

template <typename T>
struct Am_functor_data{
	typename Traits<T>::real_type t, shift;
	void (*Aop)(const char *trans, size_t n, size_t nx, const T *x, void *data);
	void *data;
	T *work;
	size_t ldwork;
	size_t m;
};
template <typename T>
void Am_functor(
	const char *trans, size_t n, size_t nb, T *x, size_t ldx, void *data
){
	typedef typename Traits<T>::real_type real_type;
	Am_functor_data *d = reinterpret_cast<Am_functor_data*>(data);
	if(real_type(0) != d->shift){
		BLAS::Copy(n, nb, x, ldx, d->work, d->ldwork);
	}
	for(size_t i = 0; i < d->m; ++i){
		d->Aop(trans, n, nb, x, ldx, d->data);
	}
	for(size_t j = 0; j < nb; ++j){
		BLAS::Scale(n, d->t, &x[j*ldx], 1);
	}
	if(real_type(0) != d->shift){
		for(size_t j = 0; j < nb; ++j){
			BLAS::Axpy(n, -(d->shift), &(d->work[j*d->ldwork]), 1, &x[j*ldx], 1);
		}
	}
}

// 
template <typename T>
void SelectDegree(
	size_t n,
	size_t nb,
	const typename Traits<T>::real_type &t,
	void (*Aop)(const char *trans, size_t n, size_t nx, const T *x, void *data),
	void *data,
	const typename Traits<T>::real_type *M, // returned M matrix (size m_max by p_max-1)
	const typename Traits<T>::real_type *alpha, // returned alpha vector (length p_max-1)
	const typename Traits<T>::real_type *eta, // work (length p_max)
	const typename Traits<T>::real_type &shift = typename Traits<T>::real_type(0),
	const typename Traits<T>::real_type &norm1A = typename Traits<T>::real_type(0),
	size_t m_max = 55, size_t p_max = 8,
	T *work,
	typename Traits<T>::real_type *rwork,
	size_t *iwork
){
	typedef typename Traits<T>::real_type real_type;
	
	const real_type *theta = Util::ExpMVTraits<real_type>::theta;
	
	norm1A *= t;
	
	if(
		real_type(0) == norm1A &&
		norm1A <= 4*theta[m_max]*p_max*(p_max + 3)/(m_max*nb)
	){
		// We must estimate norm1(A)
		for(size_t i = 0; i < p_max; ++i){
			// Estimate A^{p+2}
			real_type normAp2;
			{
				T *x = work;
				T *xold = work+n*1;
				real_type *h = rwork;
				size_t *ind = iwork;
				size_t *indh = iwork+n;
				
				Util::ExpMV::Am_functor_data funcdata;
				funcdata.t = t;
				funcdata.shift = shift;
				funcdata.Aop = Aop;
				funcdata.data = data;
				funcdata.work = xold + n*1;
				funcdata.ldwork = n;
				funcdata.m = p+2;
				
				MatrixNorm1Estimate(
					n, 1,
					&Util::ExpMV::Am_functor, &funcdata,
					&normAp2, x, n, xold, n, h, ind, indh
				);
			}
			eta[p] = pow(normAp2, real_type(1)/real_type(p+2));
		}
		for(size_t i = 0; i+1 < p_max; ++i){
			alpha[i] = eta[p+1];
			if(eta[p] > alpha[i]){ alpha[i] = eta[p]; }
		}
	}else{
		for(size_t i = 0; i+1 < p_max; ++i){
			alpha[i] = norm1A;
		}
	}
	for(size_t p = 1; p < p_max; ++p){
		for(size_t m = p*(p+1)-2; m < m_max; ++m){
			M[m+(p-1)*m_max] = alpha[p-1] / theta[m];
		}
	}
}

} // namespace ExpMV
} // namespace Util

// Computes x <- expm(t*A)*b
// With shift, shift is assumed to be trace(a)/n
template <typename T>
void ExpMV(
	size_t n,
	size_t nb,
	const typename Traits<T>::real_type &t,
	void (*Aop)(const char *trans, size_t n, size_t nx, const T *x, void *data),
	void *data,
	const T *b, size_t ldb,
	const T *x, size_t ldx,
	const typename Traits<T>::real_type &shift = typename Traits<T>::real_type(0),
	const typename Traits<T>::real_type &normA = typename Traits<T>::real_type(0)
){
	typedef typename Traits<T>::real_type real_type;
	if(real_type(0) == t){ return; }
	
	static const bool full_term = false;
	
	size_t m_max = 55, p_max = 8;
	
	real_type *M = new real_type[m_max*p_max];
	real_type *alpha = new real_type[p_max];
	real_type *eta = new real_type[p_max];
	real_type *work = new T[(2+nb)*n];
	real_type *rwork = new real_type[n];
	size_t *iwork = new size_t[2*n];
	Util::ExpMV::SelectDegree(
		n, nb, t, Aop, data, M[0+1*m_max], alpha, eta, shift, normA,
		m_max, p_max, work, rwork, iwork
	);
	delete [] iwork;
	delete [] work;
	
	size_t s = 1;
	{ // Determine best s from cost matrix
		real_type at(Traits<real_type>::abs(t));
		for(size_t j = 1; j < p_max; ++j){
			M[i+0*m_max] = Traits<real_type>::max();
			for(size_t i = 0; i < m_max; ++i){
				M[i+j*m_max] = ceil(at*M[i+j*m_max]) * real_type(1+m);
				if(M[i+j*m_max] < M[i+0*m_max]){
					M[i+0*m_max] = M[i+j*m_max];
				}
			}
		}
		size_t m = 0;
		for(size_t i = 1; i < m_max; ++i){
			if(M[i] < M[m]){
				m = i;
			}
		}
		s = size_t(M[m]/m);
		if(1 > s){ s = 1; }
	}
	delete [] eta;
	delete [] alpha;
	delete [] M;
	
	real_type eta(1);
	if(real_type(0) != shift){
		eta = exp(t*shift/s);
	}
	BLAS::Copy(n, nb, b, ldb, f, n);
	for(size_t i = 0; i < s; ++i){
		real_type c1 = MatrixNorm("I", n, nb, b, ldb, work);
		for(size_t k = 0; k < m; ++k){
			Aop("N", n, nb, b, ldb, data);
			real_type scal = t/real_type(s*(k+1));
			for(size_t j = 0; j < nb; ++j){
				BLAS::Scale(n, scal, &b[j*ldb], ldb);
				BLAS::Axpy(n, T(1), &b[j*ldb], ldb, &f[j*n], n);
			}
			real_type c2 = MatrixNorm("I", n, nb, b, ldb, work);
			if(!full_term){
				real_type normIf = MatrixNorm("I", n, nb, f, n, work);
				if(c1+c2 < Traits<real_type>::eps()*normIf){
					break;
				}
				c1 = c2;
			}
		}
		if(real_type(1) != eta){
			for(size_t j = 0; j < nb; ++j){
				BLAS::Scale(n, eta, &f[j*n], n);
			}
		}
		BLAS::Copy(n, nb, f, n, b, ldb);
	}
	delete [] rwork;
}

template <typename T>
void ExpMV_sequence(
	size_t n,
	size_t nb,
	size_t nt,
	const typename Traits<T>::real_type &t0,
	const typename Traits<T>::real_type &tinc,
	void (*Aop)(const char *trans, size_t n, size_t nx, const T *x, void *data),
	void *data,
	const T *b, size_t ldb,
	const T *x, size_t ldx,
	const typename Traits<T>::real_type &shift = typename Traits<T>::real_type(0),
){
}

} // namespace MatrixFunctions
} // namespace LA
} // namespace RNP
