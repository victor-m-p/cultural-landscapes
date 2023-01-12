#include "mpf.h"

void norm_prob(prob *a) { // for speed, we just save the overall normalizations
	int i;
	
	if (a->norm < 0) {
		a->norm=0;
		for(i=0;i<a->n;i++) {
			a->norm += a->p[i];
		}
	}
}

double delta_xi(double k, double beta) {
	
	return -gsl_sf_psi_n(1, beta+1) + k*gsl_sf_psi_n(1, beta*k+1);
}

double rho(double beta, prob *p) {
	double ln_running;
	int i;
	
	ln_running=gsl_sf_lngamma(p->n*beta) - gsl_sf_lngamma(p->norm + p->n*beta);
	for(i=0;i<p->n;i++) {
		ln_running += gsl_sf_lngamma(p->p[i]+beta) - gsl_sf_lngamma(beta);
	}
	ln_running -= p->holder;
	
	return exp(ln_running);
}

double rho_mi(double beta, j_prob *p) {
	double ln_running;
	int i, j;
	
	ln_running=gsl_sf_lngamma(p->n*p->m*beta) - gsl_sf_lngamma(p->norm + p->n*p->m*beta);
	for(i=0;i<p->n;i++) {
		for(j=0;j<p->m;j++) {
			ln_running += gsl_sf_lngamma(p->p[i][j]+beta) - gsl_sf_lngamma(beta);			
		}
	}
	ln_running -= p->holder;
	
	return exp(ln_running);
}

double s(double beta, prob *p) {
	double running;
	int i;
	
	running=0;
	for(i=0;i<p->n;i++) {
		running += (p->p[i]+beta)*(gsl_sf_psi_n(0, p->p[i]+1+beta) - gsl_sf_psi_n(0, p->norm+p->n*beta+1));
	}
	running = running/(p->norm+p->n*beta);
	
	return -running/log(2);
}

double top_integral(double beta, void *params) {
	prob *p = *(prob **)params;
	
	return rho(beta, p)*s(beta, p)*delta_xi(p->n, beta);
}

double bottom_integral(double beta, void *params) {
	prob *p = *(prob **)params;
	
	return rho(beta, p)*delta_xi(p->n, beta);
}

double entropy_nsb(prob *p) {
	gsl_integration_workspace *w=gsl_integration_workspace_alloc(10000);
	gsl_function F_top;
	gsl_function F_bottom;
	double result_top, error_top, result_bottom, error_bottom, beta;
	gsl_error_handler_t *err_h;
	int test;
	int i;
		
	norm_prob(p);
	
    F_top.function = &top_integral;
    F_top.params = &p; // a pointer to a pointer!
	
    F_bottom.function = &bottom_integral;
    F_bottom.params = &p; // a pointer to a pointer!

	beta=1.0/(double)p->n;
	p->holder=gsl_sf_lngamma(p->n*beta) - gsl_sf_lngamma(p->norm + p->n*beta);
	for(i=0;i<p->n;i++) {
		p->holder += gsl_sf_lngamma(p->p[i]+beta) - gsl_sf_lngamma(beta);
	}
	
	gsl_set_error_handler_off(); // living on the edge
	
	test=gsl_integration_qag(&F_top, 1e-3, 1e6, -1, 1e-12, 10000, GSL_INTEG_GAUSS61, w, &result_top, &error_top);
	if (test != 0) {
		return -1;
	}
	test=gsl_integration_qag(&F_bottom, 1e-3, 1e6, -1, 1e-12, 10000, GSL_INTEG_GAUSS61, w, &result_bottom, &error_bottom);
	if (test != 0) {
		return -1;
	}
	gsl_set_error_handler(NULL);
	
	gsl_integration_workspace_free(w);
	
	return result_top/result_bottom;
}

