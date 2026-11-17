// Nemenman-Shafee-Bialek entropy estimate (bits), used only for the diagnostic line printed by -o.
//
// NOTE: the alphabet size is taken to be the number of states that were actually observed (p->n), not
// the full 2^n configurations. The estimate is therefore bounded by log2(number of observed states)
// and should be read as a diagnostic of the sample, not as an estimate of the entropy of the model.
// This is unchanged from the published code; no result in the paper depends on it.
//
// Nothing here needs an external library: the log-gamma function is the C library's, and the digamma and
// trigamma functions and the quadrature are below. (Earlier versions used the GNU Scientific Library.)
#include "mpf.h"

// digamma function, d/dx log Gamma(x), for x > 0: recurrence up to x >= 10, then the asymptotic series
// (the first neglected term is below 1e-16 there)
double digamma(double x) {
	double r=0, f;
	while (x < 10) { r -= 1/x; x += 1; }
	f=1/(x*x);
	return r+log(x)-0.5/x-f*(1.0/12-f*(1.0/120-f*(1.0/252-f*(1.0/240-f*(1.0/132-f*(691.0/32760-f/12))))));
}

// trigamma function, the derivative of digamma, for x > 0
double trigamma(double x) {
	double r=0, f;
	while (x < 10) { r += 1/(x*x); x += 1; }
	f=1/(x*x);
	return r+1/x+0.5*f+(f/x)*(1.0/6-f*(1.0/30-f*(1.0/42-f*(1.0/30-f*(5.0/66-f*(691.0/2730-f*7.0/6))))));
}

// 15-point Gauss-Kronrod rule on [a, b] with the error estimate of QUADPACK's qk15
static void gk15(double (*f)(double, void *), void *params, double a, double b, double *result, double *error) {
	static const double xgk[8]={0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
		0.864864423359769072789712788640926, 0.741531185599394439863864773280788, 0.586087235467691130294144838258730,
		0.405845151377397166906606412076961, 0.207784955007898467600689403773245, 0.0};
	static const double wgk[8]={0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
		0.104790010322250183839876322541518, 0.140653259715525918745189590510238, 0.169004726639267902826583426598550,
		0.190350578064785409913256402421014, 0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
	static const double wg[4]={0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
		0.381830050505118944950369775488975, 0.417959183673469387755102040816327};
	double c=0.5*(a+b), h=0.5*(b-a), fc=f(c, params), f1[7], f2[7], kron, gauss, mean, resabs, resasc, err;
	int j;
	kron=wgk[7]*fc; gauss=wg[3]*fc; resabs=wgk[7]*fabs(fc);
	for(j=0;j<7;j++) {
		f1[j]=f(c-h*xgk[j], params); f2[j]=f(c+h*xgk[j], params);
		kron += wgk[j]*(f1[j]+f2[j]);
		resabs += wgk[j]*(fabs(f1[j])+fabs(f2[j]));
		if (j % 2 == 1) gauss += wg[j/2]*(f1[j]+f2[j]); // the Gauss nodes are every second Kronrod node
	}
	mean=0.5*kron;
	resasc=wgk[7]*fabs(fc-mean);
	for(j=0;j<7;j++) resasc += wgk[j]*(fabs(f1[j]-mean)+fabs(f2[j]-mean));
	err=fabs((kron-gauss)*h); resabs *= fabs(h); resasc *= fabs(h);
	if (resasc != 0 && err != 0) { double q=pow(200*err/resasc, 1.5); err=(q < 1) ? resasc*q : resasc; }
	if (resabs > 2.2250738585072014e-308/(50*2.220446049250313e-16) && 50*2.220446049250313e-16*resabs > err) err=50*2.220446049250313e-16*resabs;
	*result=kron*h; *error=err;
}

// Adaptive quadrature of f over [a, b]: the interval with the largest error estimate is bisected until
// the total estimate is below epsrel times the integral. Returns 0 on success; non-zero if that was
// not reached with max_intervals intervals, the integrand was not finite, or max_intervals < 1.
int integrate(double (*f)(double, void *), void *params, double a, double b, double epsrel, int max_intervals, double *result) {
	double *lo, *hi, *val, *err, total, toterr, mid;
	int n=1, i, worst, status=1;

	if (max_intervals < 1) { *result=0; return 3; }
	lo=(double *)xmalloc(max_intervals*sizeof(double)); hi=(double *)xmalloc(max_intervals*sizeof(double));
	val=(double *)xmalloc(max_intervals*sizeof(double)); err=(double *)xmalloc(max_intervals*sizeof(double));
	lo[0]=a; hi[0]=b; gk15(f, params, a, b, &val[0], &err[0]);
	for(;;) {
		total=0; toterr=0; worst=0;
		for(i=0;i<n;i++) { total += val[i]; toterr += err[i]; if (err[i] > err[worst]) worst=i; }
		if (!isfinite(total) || !isfinite(toterr)) { status=2; break; }
		if (toterr <= epsrel*fabs(total)) { status=0; break; }
		mid=0.5*(lo[worst]+hi[worst]);
		if (n == max_intervals || mid <= lo[worst] || mid >= hi[worst]) break;
		lo[n]=mid; hi[n]=hi[worst]; hi[worst]=mid;
		gk15(f, params, lo[worst], hi[worst], &val[worst], &err[worst]);
		gk15(f, params, lo[n], hi[n], &val[n], &err[n]);
		n++;
	}
	*result=total;
	free(lo); free(hi); free(val); free(err);
	return status;
}

static void norm_prob(prob *a) {
	int i;
	if (a->norm < 0) {
		a->norm=0;
		for(i=0;i<a->n;i++) a->norm += a->p[i];
	}
}

static double delta_xi(double k, double beta) {
	return -trigamma(beta+1)+k*trigamma(beta*k+1);
}

// log of the evidence for the counts under a symmetric Dirichlet prior with parameter beta
static double log_evidence(double beta, const prob *p) {
	double v;
	int i;
	v=lgamma(p->n*beta)-lgamma(p->norm+p->n*beta);
	for(i=0;i<p->n;i++) v += lgamma(p->p[i]+beta)-lgamma(beta);
	return v;
}

static double rho(double beta, prob *p) {
	double ln_running=log_evidence(beta, p);
	ln_running -= p->holder;
	return exp(ln_running);
}

static double s(double beta, prob *p) {
	double running=0;
	int i;
	for(i=0;i<p->n;i++) running += (p->p[i]+beta)*(digamma(p->p[i]+1+beta)-digamma(p->norm+p->n*beta+1));
	running=running/(p->norm+p->n*beta);
	return -running/log(2);
}

// The two integrals over beta run from 1e-3 to 1e6; they are taken in u = log(beta), in which the
// integrands are smooth across those nine decades.
static double top_integrand(double u, void *params) {
	prob *p=(prob *)params;
	double beta=exp(u);
	return rho(beta, p)*s(beta, p)*delta_xi(p->n, beta)*beta;
}

static double bottom_integrand(double u, void *params) {
	prob *p=(prob *)params;
	double beta=exp(u);
	return rho(beta, p)*delta_xi(p->n, beta)*beta;
}

// returns -1 if the numerical integration fails
double entropy_nsb(prob *p) {
	double result_top=0, result_bottom=0;
	int test_top, test_bottom;

	if (p->n < 1) return -1;
	norm_prob(p);

	p->holder=log_evidence(1.0/(double)p->n, p);

	test_top=integrate(top_integrand, p, log(1e-3), log(1e6), 1e-10, 4000, &result_top);
	test_bottom=integrate(bottom_integrand, p, log(1e-3), log(1e6), 1e-10, 4000, &result_bottom);

	if (test_top != 0 || test_bottom != 0 || result_bottom == 0) return -1;
	return result_top/result_bottom;
}
