// Nemenman-Shafee-Bialek entropy estimate (bits), used only for the diagnostic line printed by -o.
//
// NOTE: the alphabet size is taken to be the number of states that were actually observed (p->n), not
// the full 2^n configurations. The estimate is therefore bounded by log2(number of observed states)
// and should be read as a diagnostic of the sample, not as an estimate of the entropy of the model.
// This is unchanged from the published code; no result in the paper depends on it.
#include "mpf.h"

static void norm_prob(prob *a) {
	int i;
	if (a->norm < 0) {
		a->norm=0;
		for(i=0;i<a->n;i++) a->norm += a->p[i];
	}
}

static double delta_xi(double k, double beta) {
	return -gsl_sf_psi_n(1, beta+1)+k*gsl_sf_psi_n(1, beta*k+1);
}

static double rho(double beta, prob *p) {
	double ln_running;
	int i;
	ln_running=gsl_sf_lngamma(p->n*beta)-gsl_sf_lngamma(p->norm+p->n*beta);
	for(i=0;i<p->n;i++) ln_running += gsl_sf_lngamma(p->p[i]+beta)-gsl_sf_lngamma(beta);
	ln_running -= p->holder;
	return exp(ln_running);
}

static double s(double beta, prob *p) {
	double running=0;
	int i;
	for(i=0;i<p->n;i++) running += (p->p[i]+beta)*(gsl_sf_psi_n(0, p->p[i]+1+beta)-gsl_sf_psi_n(0, p->norm+p->n*beta+1));
	running=running/(p->norm+p->n*beta);
	return -running/log(2);
}

static double top_integral(double beta, void *params) {
	prob *p=*(prob **)params;
	return rho(beta, p)*s(beta, p)*delta_xi(p->n, beta);
}

static double bottom_integral(double beta, void *params) {
	prob *p=*(prob **)params;
	return rho(beta, p)*delta_xi(p->n, beta);
}

// returns -1 if the numerical integration fails
double entropy_nsb(prob *p) {
	gsl_integration_workspace *w;
	gsl_function F_top, F_bottom;
	gsl_error_handler_t *old_handler;
	double result_top=0, error_top, result_bottom=0, error_bottom, beta;
	int test_top, test_bottom, i;

	if (p->n < 1) return -1;
	norm_prob(p);

	F_top.function=&top_integral; F_top.params=&p;
	F_bottom.function=&bottom_integral; F_bottom.params=&p;

	beta=1.0/(double)p->n;
	p->holder=gsl_sf_lngamma(p->n*beta)-gsl_sf_lngamma(p->norm+p->n*beta);
	for(i=0;i<p->n;i++) p->holder += gsl_sf_lngamma(p->p[i]+beta)-gsl_sf_lngamma(beta);

	w=gsl_integration_workspace_alloc(10000);
	old_handler=gsl_set_error_handler_off();
	test_top=gsl_integration_qag(&F_top, 1e-3, 1e6, -1, 1e-12, 10000, GSL_INTEG_GAUSS61, w, &result_top, &error_top);
	test_bottom=gsl_integration_qag(&F_bottom, 1e-3, 1e6, -1, 1e-12, 10000, GSL_INTEG_GAUSS61, w, &result_bottom, &error_bottom);
	gsl_set_error_handler(old_handler);
	gsl_integration_workspace_free(w);

	if (test_top != 0 || test_bottom != 0 || result_bottom == 0) return -1;
	return result_top/result_bottom;
}
