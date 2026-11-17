// Unit tests for the MPF code. Every numerical routine is compared with a slow, independent
// implementation written straight from the definitions (explicit bit-flips, energies from scratch).
// Build and run with `make test`.
#include "mpf.h"

static int n_checks=0, n_failed=0;
static void check(int ok, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
#include <stdarg.h>
static void check(int ok, const char *fmt, ...) {
	va_list ap;
	n_checks++;
	if (ok) return;
	n_failed++;
	va_start(ap, fmt);
	printf("  FAILED: ");
	vprintf(fmt, ap);
	printf("\n");
	va_end(ap);
}
static double reldiff(double a, double b) { return fabs(a-b)/(fabs(a)+fabs(b)+1e-300); }

// ---------- independent reference implementations ----------

static double ref_h(int n, const double *theta, uint64_t s) { // H of a packed state
	int i, j, cnt=0;
	double H=0;
	for(i=0;i<n;i++) {
		for(j=i+1;j<n;j++) { H += (((s >> i) & 1) ? 1 : -1)*(((s >> j) & 1) ? 1 : -1)*theta[cnt]; cnt++; }
	}
	for(i=0;i<n;i++) H += (((s >> i) & 1) ? 1 : -1)*theta[cnt+i];
	return H;
}

static double ref_lse(const double *v, int n) {
	double mx=v[0], z=0; int i;
	for(i=1;i<n;i++) if (v[i] > mx) mx=v[i];
	for(i=0;i<n;i++) z += exp(v[i]-mx);
	return mx+log(z);
}

// K for one objective, summing over the raw rows (no merging of duplicates), computed from log-sum-exps
// over completions so that it stays correct for extreme parameters. For OBJ_FROZEN the completion
// weights are computed at theta_w instead of theta.
static double ref_k(const dataset *d, const double *theta, const double *theta_w, int objective, double log_sparsity) {
	int n=d->n, np=n*(n+1)/2, i, j, k, nb, pos[MAX_NODES], c, nc;
	double K=0, total_w=0, *H, *Hw, *Hk, *t, z, sub, lw;
	uint64_t base, s;

	for(i=0;i<d->m;i++) {
		if (d->w[i] == 0) continue;
		total_w += d->w[i];
		base=0; nb=0;
		for(j=0;j<n;j++) {
			if (d->v[(size_t)i*n+j] > 0) base |= ((uint64_t)1 << j);
			if (d->v[(size_t)i*n+j] == 0) pos[nb++]=j;
		}
		nc=1 << nb;
		H=malloc(nc*sizeof(double)); Hw=malloc(nc*sizeof(double)); Hk=malloc(nc*sizeof(double)); t=malloc(nc*sizeof(double));
		for(c=0;c<nc;c++) {
			s=base;
			for(j=0;j<nb;j++) if ((c >> j) & 1) s |= ((uint64_t)1 << pos[j]);
			H[c]=ref_h(n, theta, s);
			Hw[c]=ref_h(n, objective == OBJ_FROZEN ? theta_w : theta, s);
		}
		lw=ref_lse(Hw, nc); // log normaliser of the completion weights
		sub=0;
		{ // every set A of 1..nn observed nodes, as a bit mask (single bits when nn == 1)
			uint64_t obsmask=0, A, limit;
			int nn=d->nn;
			for(k=0;k<n;k++) if (d->v[(size_t)i*n+k] != 0) obsmask |= ((uint64_t)1 << k);
			limit=(nn == 1) ? (uint64_t)n : ((uint64_t)1 << n);
			for(A=(nn == 1 ? 0 : 1);A<limit;A++) {
				uint64_t flip=(nn == 1) ? ((uint64_t)1 << A) : A;
				if (objective == OBJ_FROZEN && nn > 1) {
					if ((flip & obsmask) == 0) continue;                  // frozen, nn > 1: any set with an observed node
				} else if ((flip & ~obsmask) != 0) continue;              // otherwise flows only along observed nodes
				if (__builtin_popcountll(flip) > nn) continue;
				for(c=0;c<nc;c++) {
					s=base;
					for(j=0;j<nb;j++) if ((c >> j) & 1) s |= ((uint64_t)1 << pos[j]);
					Hk[c]=ref_h(n, theta, s ^ flip);
				}
				if (objective == OBJ_MARGINAL) {
					sub += exp(0.5*(ref_lse(Hk, nc)-ref_lse(H, nc)));           // sqrt( p(o^A) / p(o) )
				} else {
					for(c=0;c<nc;c++) t[c]=Hw[c]-lw+0.5*(Hk[c]-H[c]);           // log( w_c * flow )
					sub += exp(ref_lse(t, nc));
				}
			}
		}
		K += d->w[i]*sub;
		free(H); free(Hw); free(Hk); free(t);
	}
	z=0;
	for(i=0;i<np;i++) z += theta[i]*theta[i];
	{ // penalty: neighbours per complete state * lambda * total weight / number of parameters
		double n_prox=0, comb=1.0;
		for(j=1;j<=d->nn;j++) { comb=comb*(n-j+1)/j; n_prox += comb; }
		return K+n_prox*pow(10.0, log_sparsity)*total_w/np*z/2;
	}
}

static double ref_logz(int n, const double *theta) {
	uint64_t s, N=(uint64_t)1 << n;
	double mx=-INFINITY, z=0;
	for(s=0;s<N;s++) { double H=ref_h(n, theta, s); if (H > mx) mx=H; }
	for(s=0;s<N;s++) z += exp(ref_h(n, theta, s)-mx);
	return log(z)+mx;
}

static double ref_kl(int n, const double *truth, const double *inf) {
	uint64_t s, N=(uint64_t)1 << n;
	double lt=ref_logz(n, truth), li=ref_logz(n, inf), kl=0;
	for(s=0;s<N;s++) { double a=ref_h(n, truth, s)-lt, b=ref_h(n, inf, s)-li; kl += exp(a)*(a-b); }
	return kl;
}

// ---------- helpers ----------

static mpf_rng R_state, *R=&R_state;

static dataset *random_data(int m, int n, int max_blanks, int weighted, int duplicates) {
	dataset *d=malloc(sizeof(dataset));
	int i, j, k, nb;
	d->m=m; d->n=n; d->group=NULL; d->nn=1;
	d->v=malloc((size_t)m*n); d->w=malloc(m*sizeof(double)); d->n_blanks=malloc(m*sizeof(int));
	for(i=0;i<m;i++) {
		if (duplicates && i > 0 && rng_uniform(R) < 0.3) { // copy an earlier row
			k=(int)rng_uniform_int(R, i);
			for(j=0;j<n;j++) d->v[(size_t)i*n+j]=d->v[(size_t)k*n+j];
			d->n_blanks[i]=d->n_blanks[k];
		} else {
			for(j=0;j<n;j++) d->v[(size_t)i*n+j]=(rng_uniform(R) < 0.5) ? 1 : -1;
			nb=(max_blanks > 0 && rng_uniform(R) < 0.6) ? (int)rng_uniform_int(R, max_blanks+1) : 0;
			if (nb > n-1) nb=n-1;
			d->n_blanks[i]=0;
			while(d->n_blanks[i] < nb) {
				j=(int)rng_uniform_int(R, n);
				if (d->v[(size_t)i*n+j] != 0) { d->v[(size_t)i*n+j]=0; d->n_blanks[i]++; }
			}
		}
		d->w[i]=weighted ? 0.125*(1+rng_uniform_int(R, 16)) : 1.0;
	}
	return d;
}

static double *random_theta(int n, double scale) {
	int np=n*(n+1)/2, i;
	double *t=malloc(np*sizeof(double));
	for(i=0;i<np;i++) t[i]=rng_gaussian(R, scale);
	return t;
}

static const char *obj_name[3]={"marginal", "legacy", "frozen"};

// ---------- tests ----------

static void test_objective(void) {
	int sizes[]={2, 3, 5, 8, 13, 20, 33, 40, 62}, si, rep, obj, i, n, np, m;
	printf("objective and gradient against brute force (n up to 62, blanks, weights, duplicates)\n");
	for(si=0;si<9;si++) for(rep=0;rep<4;rep++) {
		n=sizes[si]; np=n*(n+1)/2; m=6+(int)rng_uniform_int(R, 25);
		dataset *d=random_data(m, n, rep == 0 ? 0 : (n > 20 ? 4 : 6), rep >= 2, rep == 3);
		double ls=-1.0+rep*0.7, scale=(n > 20) ? 0.3/sqrt(n) : 0.3;
		double *th=random_theta(n, scale), *th0=random_theta(n, scale), *g=malloc(np*sizeof(double)), *g2=malloc(np*sizeof(double));
		model *mo=build_model(d, NULL, ls);
		for(obj=0;obj<3;obj++) {
			double k, kr, kp, km, worst=0, gmax=0, ep=1e-5;
			if (obj == OBJ_FROZEN) compute_k(mo, th0, NULL, OBJ_FROZEN, 1); // freeze weights at th0
			k=compute_k(mo, th, g, obj, 0);
			kr=ref_k(d, th, th0, obj, ls);
			check(reldiff(k, kr) < 1e-11, "K %s n=%d rep=%d: fast %.15g reference %.15g", obj_name[obj], n, rep, k, kr);
			check(compute_k(mo, th, NULL, obj, 0) == k, "K %s n=%d: value differs with and without gradient", obj_name[obj], n);
			for(i=0;i<np;i++) if (fabs(g[i]) > gmax) gmax=fabs(g[i]);
			for(i=0;i<np;i++) {
				double save=th[i];
				th[i]=save+ep; kp=compute_k(mo, th, NULL, obj, 0);
				th[i]=save-ep; km=compute_k(mo, th, NULL, obj, 0);
				th[i]=save;
				if (fabs((kp-km)/(2*ep)-g[i]) > worst) worst=fabs((kp-km)/(2*ep)-g[i]);
			}
			check(worst < 2e-6*gmax, "gradient %s n=%d rep=%d: worst finite-difference error %.3g (max |g| %.3g)", obj_name[obj], n, rep, worst, gmax);
			// the gradient must not depend on earlier calls (scratch state)
			compute_k(mo, th0, g2, obj, 0); compute_k(mo, th, g2, obj, 0);
			for(i=0;i<np;i++) if (g2[i] != g[i]) break;
			check(i == np, "gradient %s n=%d: not reproducible across calls", obj_name[obj], n);
		}
		if (rep == 0) { // no missing data: the three objectives are the same function
			double a=compute_k(mo, th, NULL, OBJ_MARGINAL, 0), b=compute_k(mo, th, NULL, OBJ_LEGACY, 0), c=compute_k(mo, th, NULL, OBJ_FROZEN, 0);
			check(reldiff(a, b) < 1e-13 && reldiff(a, c) < 1e-13, "complete data n=%d: objectives differ (%.15g %.15g %.15g)", n, a, b, c);
		}
		free_model(mo); free_data(d); free(th); free(th0); free(g); free(g2);
	}
}

static void test_weights_and_duplicates(void) {
	int n=7, np=n*(n+1)/2, i, j, obj;
	printf("duplicate rows, weights, zero weights, row masks\n");
	dataset *a=random_data(12, n, 3, 0, 0), *b=malloc(sizeof(dataset));
	double *th=random_theta(n, 0.4);
	(void)np;
	// b = a with every row written three times, plus two zero-weight rows of junk
	b->m=3*a->m+2; b->n=n; b->group=NULL; b->nn=1;
	b->v=malloc((size_t)b->m*n); b->w=malloc(b->m*sizeof(double)); b->n_blanks=malloc(b->m*sizeof(int));
	for(i=0;i<3*a->m;i++) { for(j=0;j<n;j++) b->v[(size_t)i*n+j]=a->v[(size_t)(i%a->m)*n+j]; b->w[i]=1.0; b->n_blanks[i]=a->n_blanks[i%a->m]; }
	for(i=3*a->m;i<b->m;i++) { for(j=0;j<n;j++) b->v[(size_t)i*n+j]=1; b->w[i]=0.0; b->n_blanks[i]=0; }
	for(i=0;i<a->m;i++) a->w[i]=3.0;
	model *ma=build_model(a, NULL, 0.2), *mb=build_model(b, NULL, 0.2);
	check(ma->D == mb->D && ma->R == mb->R, "duplicates: model sizes differ (%d,%ld) vs (%d,%ld)", ma->D, ma->R, mb->D, mb->R);
	for(obj=0;obj<2;obj++) {
		double ka=compute_k(ma, th, NULL, obj, 0), kb=compute_k(mb, th, NULL, obj, 0);
		check(reldiff(ka, kb) < 1e-13, "duplicates %s: weight 3 gives %.15g, three copies give %.15g", obj_name[obj], ka, kb);
	}
	// a row mask is the same as deleting the rows
	char *mask=malloc(a->m);
	for(i=0;i<a->m;i++) mask[i]=(i%3 != 0);
	dataset *c=malloc(sizeof(dataset));
	c->n=n; c->group=NULL; c->nn=1; c->m=0; c->v=malloc((size_t)a->m*n); c->w=malloc(a->m*sizeof(double)); c->n_blanks=malloc(a->m*sizeof(int));
	for(i=0;i<a->m;i++) if (mask[i]) { for(j=0;j<n;j++) c->v[(size_t)c->m*n+j]=a->v[(size_t)i*n+j]; c->w[c->m]=a->w[i]; c->n_blanks[c->m]=a->n_blanks[i]; c->m++; }
	model *m1=build_model(a, mask, -0.3), *m2=build_model(c, NULL, -0.3);
	check(reldiff(compute_k(m1, th, NULL, OBJ_MARGINAL, 0), compute_k(m2, th, NULL, OBJ_MARGINAL, 0)) < 1e-13, "row mask differs from deleting the rows");
	free_model(ma); free_model(mb); free_model(m1); free_model(m2); free_data(a); free_data(b); free_data(c); free(mask); free(th);
}

static void test_enumeration(void) {
	int n, rep, np;
	printf("partition function, KL divergence, pseudo-likelihood against brute force\n");
	for(n=2;n<=16;n++) for(rep=0;rep<3;rep++) {
		double scale=(rep == 0) ? 0.2 : (rep == 1 ? 1.0 : 30.0); // 30: far outside the range where exp(H) is finite
		double *a=random_theta(n, scale), *b=random_theta(n, scale < 2 ? scale : 1.0);
		double lz=log_z(n, a), lr=ref_logz(n, a), kl=full_kl(n, a, b), kr=ref_kl(n, a, b);
		np=n*(n+1)/2; (void)np;
		check(isfinite(lz) && fabs(lz-lr) < 1e-9*(1+fabs(lr)), "log Z n=%d scale=%g: %.12g vs %.12g", n, scale, lz, lr);
		check(isfinite(kl) && fabs(kl-kr) < 1e-8*(1+fabs(kr)), "KL n=%d scale=%g: %.12g vs %.12g", n, scale, kl, kr);
		check(kl > -1e-9, "KL n=%d is negative: %g", n, kl);
		check(fabs(full_kl(n, a, a)) < 1e-9, "KL(p||p) n=%d is %g", n, full_kl(n, a, a));
		// pseudo-likelihood and H of a row
		int8_t v[MAX_NODES]; uint64_t s=0; int k; double pl=0, H;
		for(k=0;k<n;k++) { v[k]=(rng_uniform(R) < 0.5) ? 1 : -1; if (v[k] > 0) s |= ((uint64_t)1 << k); }
		H=ref_h(n, b, s);
		check(fabs(h_value(n, b, v)-H) < 1e-10*(1+fabs(H)), "h_value n=%d", n);
		for(k=0;k<n;k++) { double H2=ref_h(n, b, s ^ ((uint64_t)1 << k)), mx=fmax(H, H2); pl += H-(mx+log(exp(H-mx)+exp(H2-mx))); }
		check(fabs(pseudo_loglik(n, b, v)-pl) < 1e-9*(1+fabs(pl)), "pseudo-likelihood n=%d: %.12g vs %.12g", n, pseudo_loglik(n, b, v), pl);
		free(a); free(b);
	}
	{ // n = 21 crosses the threshold where enumeration is split across threads
		double *a=random_theta(21, 0.15), *b=random_theta(21, 0.15);
		check(fabs(log_z(21, a)-ref_logz(21, a)) < 1e-8, "log Z n=21 (threaded path)");
		check(fabs(full_kl(21, a, b)-ref_kl(21, a, b)) < 1e-8, "KL n=21 (threaded path)");
		free(a); free(b);
	}
}

static void test_probs_file(void) {
	int sizes[]={6, 21}, si; // 21 nodes spans two of the 2^20-state blocks that -z writes
	printf("probabilities of all configurations (-z)\n");
	for(si=0;si<2;si++) {
		int n=sizes[si], k, bad_order=0, bad_value=0;
		long i, N=1L << n, step=(n > 10) ? 9973 : 1;
		double *th=random_theta(n, n > 10 ? 0.15 : 0.5), lz=ref_logz(n, th), e, p, total=0;
		char line[128], name[]="/tmp/mpf_test_probs_XXXXXX";
		FILE *f;
		close(mkstemp(name));
		compute_probs(n, th, name);
		f=fopen(name, "r");
		for(i=0;i<N;i++) {
			if (fscanf(f, "%127s %le %le", line, &e, &p) != 3) { check(0, "probs file n=%d: short read at line %ld", n, i); break; }
			uint64_t s=0;
			for(k=0;k<n;k++) if (line[k] == '1') s |= ((uint64_t)1 << k);
			if ((long)s != i || (int)strlen(line) != n) bad_order++;
			if (i%step == 0 || i == N-1) if (fabs(e-ref_h(n, th, s)) > 1e-8 || fabs(p-exp(ref_h(n, th, s)-lz)) > 1e-9) bad_value++;
			total += p;
		}
		check(bad_order == 0, "probs file n=%d: %d lines out of order", n, bad_order);
		check(bad_value == 0, "probs file n=%d: %d wrong values", n, bad_value);
		check(fabs(total-1) < 1e-7, "probs file n=%d: probabilities sum to %.12g", n, total);
		fclose(f); unlink(name); free(th);
	}
}

static void test_samplers(void) {
	int n=4, N=1 << n, draws=200000, i;
	double *th=random_theta(n, 0.6), lz=ref_logz(n, th), chi2=0, expct;
	uint64_t *out=malloc(draws*sizeof(uint64_t));
	long *count=calloc(N, sizeof(long));
	printf("samplers against exact probabilities (chi-square)\n");
	sample_exact(n, th, draws, out, R);
	for(i=0;i<draws;i++) { check(out[i] < (uint64_t)N, "exact sampler: state out of range"); if (out[i] < (uint64_t)N) count[out[i]]++; }
	for(i=0;i<N;i++) { expct=draws*exp(ref_h(n, th, i)-lz); chi2 += (count[i]-expct)*(count[i]-expct)/expct; }
	check(chi2 < (N-1)+5*sqrt(2.0*(N-1)), "exact sampler: chi-square %.1f on %d dof", chi2, N-1);
	// Monte Carlo sampler: weak couplings, where it does equilibrate
	free(th); th=random_theta(n, 0.2); lz=ref_logz(n, th);
	memset(count, 0, N*sizeof(long)); chi2=0; draws=40000;
	for(i=0;i<draws;i++) { uint64_t c=rng_uniform_int(R, N); mcmc_sampler(&c, 200, n, th, R); count[c]++; }
	for(i=0;i<N;i++) { expct=draws*exp(ref_h(n, th, i)-lz); chi2 += (count[i]-expct)*(count[i]-expct)/expct; }
	check(chi2 < (N-1)+5*sqrt(2.0*(N-1)), "Monte Carlo sampler: chi-square %.1f on %d dof", chi2, N-1);
	free(th); free(out); free(count);
}

static void test_fit(void) {
	int n=8, np=n*(n+1)/2, i, obj;
	printf("minimiser: stationarity, independence of the starting point, optimality\n");
	dataset *d=random_data(60, n, 0, 1, 1);
	model *mo=build_model(d, NULL, 0.0);
	double *x=calloc(np, sizeof(double)), *y=random_theta(n, 1.0), *g=malloc(np*sizeof(double)), worst=0, k0, k1;
	fit_info a=fit_model(mo, x, OBJ_MARGINAL), b=fit_model(mo, y, OBJ_MARGINAL);
	check(a.status == FIT_CONVERGED && b.status == FIT_CONVERGED, "fit did not report convergence");
	for(i=0;i<np;i++) if (fabs(x[i]-y[i]) > worst) worst=fabs(x[i]-y[i]);
	check(worst < 1e-6, "convex fit depends on the starting point (max difference %.3g)", worst);
	k0=compute_k(mo, x, g, OBJ_MARGINAL, 0);
	worst=0; for(i=0;i<np;i++) if (fabs(g[i]) > worst) worst=fabs(g[i]);
	check(worst < 1e-6*mo->total_w*n, "gradient at the solution is %.3g", worst);
	for(i=0;i<20;i++) { // random perturbations never lower K
		double *p=random_theta(n, 1e-3); int j;
		for(j=0;j<np;j++) p[j] += x[j];
		k1=compute_k(mo, p, NULL, OBJ_MARGINAL, 0);
		check(k1 >= k0-1e-9*fabs(k0), "perturbation lowers K: %.12g < %.12g", k1, k0);
		free(p);
	}
	free_model(mo); free_data(d);
	// with missing data: each objective reaches a stationary point of its own function
	d=random_data(80, n, 3, 1, 0);
	mo=build_model(d, NULL, 0.0);
	for(obj=0;obj<3;obj++) {
		for(i=0;i<np;i++) x[i]=0;
		a=fit_model(mo, x, obj);
		check(a.status == FIT_CONVERGED, "%s fit with missing data did not converge", obj_name[obj]);
		if (obj == OBJ_FROZEN) compute_k(mo, x, NULL, OBJ_FROZEN, 1); // gradient with weights frozen at the solution
		compute_k(mo, x, g, obj, 0);
		worst=0; for(i=0;i<np;i++) if (fabs(g[i]) > worst) worst=fabs(g[i]);
		check(worst < 1e-6*mo->total_w*n, "%s fit with missing data: gradient at the solution is %.3g", obj_name[obj], worst);
	}
	free_model(mo); free_data(d); free(x); free(y); free(g);
}

// The central claim: with infinite data, MPF on the marginal (and the frozen-weight fixed point) recover the true
// parameters when entries are missing at random; the legacy Partial-MPF objective does not.
static void test_consistency(void) {
	int n=5, np=n*(n+1)/2, N=1 << n, a, b, s, j, row=0, obj, i;
	int n_masks=1+n*(n-1)/2;
	printf("infinite-data consistency of the three treatments of missing data\n");
	double *truth=random_theta(n, 0.4), lz=ref_logz(n, truth), err[3];
	dataset *d=malloc(sizeof(dataset));
	d->n=n; d->group=NULL; d->nn=1; d->m=n_masks*N;
	d->v=malloc((size_t)d->m*n); d->w=calloc(d->m, sizeof(double)); d->n_blanks=malloc(d->m*sizeof(int));
	// every (mask, full state) pair as a row, weighted by its exact probability: half the mass on complete
	// rows, half spread evenly over all ways of blanking two nodes. Duplicate observed parts are merged
	// by build_model, which sums their weights: exactly the marginal probability of the observed part.
	for(a=-1;a<n;a++) for(b=a+1;b<n;b++) {
		if (a == -1 && b > 0) continue; // a == -1: the complete pattern, once
		for(s=0;s<N;s++) {
			for(j=0;j<n;j++) d->v[(size_t)row*n+j]=((s >> j) & 1) ? 1 : -1;
			d->n_blanks[row]=0;
			d->w[row]=exp(ref_h(n, truth, s)-lz)*(a == -1 ? 0.5 : 0.5/(n_masks-1));
			if (a >= 0) { d->v[(size_t)row*n+a]=0; d->v[(size_t)row*n+b]=0; d->n_blanks[row]=2; }
			row++;
		}
	}
	check(row == d->m, "consistency test: built %d rows, expected %d", row, d->m);
	model *mo=build_model(d, NULL, -14.0); // effectively no regularisation
	double *x=malloc(np*sizeof(double));
	for(obj=0;obj<3;obj++) {
		for(i=0;i<np;i++) x[i]=0;
		fit_info fi=fit_model(mo, x, obj);
		check(fi.status == FIT_CONVERGED, "%s fit on population data: %s", obj_name[obj], fit_status_name(fi.status));
		err[obj]=0;
		for(i=0;i<np;i++) if (fabs(x[i]-truth[i]) > err[obj]) err[obj]=fabs(x[i]-truth[i]);
		printf("    %-8s max |fit - truth| = %.2e\n", obj_name[obj], err[obj]);
	}
	check(err[OBJ_MARGINAL] < 1e-5, "marginal objective does not recover the truth (error %.3g)", err[OBJ_MARGINAL]);
	check(err[OBJ_FROZEN] < 1e-4, "frozen-weight iteration does not recover the truth (error %.3g)", err[OBJ_FROZEN]);
	check(err[OBJ_LEGACY] > 1e-2, "legacy objective unexpectedly recovers the truth (error %.3g)", err[OBJ_LEGACY]);
	free_model(mo); free_data(d); free(truth); free(x);
}

// weighted = 1 is the program's default protocol: held-out rows count with their weights, and each fold
// fits its complete training rows first and then all its training rows from there (the same thing as a
// single fit when no row has a missing entry). weighted = 0 is the legacy protocol: equal weights, one fit.
static double manual_cv(const dataset *d, double ls, int objective, int weighted, int grouped) {
	int m=d->m, n=d->n, np=n*(n+1)/2, i, j;
	double total=0, tw=0, *x=malloc(np*sizeof(double));
	char *mask=malloc(m), *cmask=malloc(m), *done=calloc(m, 1);
	for(i=0;i<m;i++) {
		int ok=1;
		if (done[i] || d->w[i] == 0) continue;
		for(j=0;j<m;j++) {
			mask[j]=grouped ? (d->group[j] != d->group[i]) : (j != i);
			if (!mask[j] && d->n_blanks[j] > 0) ok=0;
		}
		if (!ok) continue;
		for(j=0;j<np;j++) x[j]=0;
		if (weighted) {
			for(j=0;j<m;j++) cmask[j]=mask[j] && d->n_blanks[j] == 0;
			model *m1=build_model(d, cmask, ls);
			fit_model(m1, x, objective);
			free_model(m1);
		}
		model *mo=build_model(d, mask, ls);
		fit_model(mo, x, objective);
		free_model(mo);
		double lz=ref_logz(n, x);
		for(j=0;j<m;j++) if (!mask[j] && d->w[j] > 0) {
			uint64_t s=0; int k;
			for(k=0;k<n;k++) if (d->v[(size_t)j*n+k] > 0) s |= ((uint64_t)1 << k);
			double wt=weighted ? d->w[j] : 1.0;
			total += wt*(ref_h(n, x, s)-lz); tw += wt; done[j]=1;
		}
	}
	free(x); free(mask); free(cmask); free(done);
	return total/tw;
}

static void test_cross_validation(void) {
	int n=6, i;
	options opt;
	printf("cross-validation against explicit refits (row-wise, weighted, grouped)\n");
	memset(&opt, 0, sizeof(opt));
	dataset *d=random_data(40, n, 0, 1, 1);
	for(i=0;i<d->m;i++) if (i%7 == 3) { d->v[(size_t)i*n]=0; d->n_blanks[i]=1; } // a few partial rows, never held out
	opt.objective=OBJ_MARGINAL; opt.legacy_cv=1;
	double a=cross(d, NULL, 0.3, NULL, &opt, NULL), b=manual_cv(d, 0.3, OBJ_MARGINAL, 0, 0);
	check(fabs(a-b) < 1e-6, "legacy (unweighted, row-wise) CV: %.10f vs explicit %.10f", a, b);
	opt.legacy_cv=0;
	a=cross(d, NULL, 0.3, NULL, &opt, NULL); b=manual_cv(d, 0.3, OBJ_MARGINAL, 1, 0);
	check(fabs(a-b) < 1e-6, "weighted CV: %.10f vs explicit %.10f", a, b);
	check(fabs(cross(d, NULL, 0.3, NULL, &opt, NULL)-a) < 1e-12, "CV is not reproducible between calls");
	d->group=malloc(d->m*sizeof(int));
	for(i=0;i<d->m;i++) d->group[i]=100+i/3; // triples; some contain a partial row and are never held out
	a=cross(d, NULL, 0.3, NULL, &opt, NULL); b=manual_cv(d, 0.3, OBJ_MARGINAL, 1, 1);
	check(fabs(a-b) < 1e-6, "grouped CV: %.10f vs explicit %.10f", a, b);
	opt.legacy_cv=1;
	check(fabs(cross(d, NULL, 0.3, NULL, &opt, NULL)-manual_cv(d, 0.3, OBJ_MARGINAL, 0, 0)) < 1e-6, "legacy CV must ignore groups");
	// a row mask restricts both the training rows and the held-out rows
	char *mask=malloc(d->m);
	dataset *c=malloc(sizeof(dataset));
	int j;
	c->n=n; c->group=NULL; c->nn=1; c->m=0; c->v=malloc((size_t)d->m*n); c->w=malloc(d->m*sizeof(double)); c->n_blanks=malloc(d->m*sizeof(int));
	for(i=0;i<d->m;i++) { mask[i]=(d->n_blanks[i] == 0); if (mask[i]) { for(j=0;j<n;j++) c->v[(size_t)c->m*n+j]=d->v[(size_t)i*n+j]; c->w[c->m]=d->w[i]; c->n_blanks[c->m]=0; c->m++; } }
	a=cross(d, mask, -0.2, NULL, &opt, NULL); b=cross(c, NULL, -0.2, NULL, &opt, NULL);
	check(fabs(a-b) < 1e-9, "CV with a row mask: %.10f vs %.10f on the reduced data", a, b);
	free(mask); free_data(c); free_data(d);
}


// build a dataset from strings such as "1X0" and weights
static dataset *make_data(int m, int n, const char **rows, const double *w) {
	dataset *d=malloc(sizeof(dataset));
	int i, j;
	d->m=m; d->n=n; d->group=NULL; d->nn=1;
	d->v=malloc((size_t)m*n); d->w=malloc(m*sizeof(double)); d->n_blanks=malloc(m*sizeof(int));
	for(i=0;i<m;i++) {
		d->n_blanks[i]=0;
		for(j=0;j<n;j++) {
			d->v[(size_t)i*n+j]=(rows[i][j] == '1') ? 1 : (rows[i][j] == '0' ? -1 : 0);
			if (rows[i][j] == 'X') d->n_blanks[i]++;
		}
		d->w[i]=w ? w[i] : 1.0;
	}
	return d;
}

// Parameters far outside the comfortable range: the objective must stay correct, and a failure must be
// reported as a failure. (Cases A2 of the external review.)
static void test_extreme_parameters(void) {
	int obj, i, rep;
	printf("extreme parameters: correctness in the log domain, and honest failure\n");
	{ // one complete row, huge fields: flows of exp(-400); the squared flow would underflow
		const char *rows[]={"11"};
		dataset *d=make_data(1, 2, rows, NULL);
		model *mo=build_model(d, NULL, 0.0);
		double th[3]={0, 400, 400}, g[3], x0[3]={0, 0, 0}, x1[3]={0, 400, 400}, k;
		for(obj=0;obj<3;obj++) {
			k=compute_k(mo, th, g, obj, 0);
			check(reldiff(k, ref_k(d, th, th, obj, 0.0)) < 1e-12, "huge fields, %s: K %.15g vs reference %.15g", obj_name[obj], k, ref_k(d, th, th, obj, 0.0));
			check(isfinite(g[0]) && isfinite(g[1]) && isfinite(g[2]) && fabs(g[1]) > 1, "huge fields, %s: gradient %g %g %g", obj_name[obj], g[0], g[1], g[2]);
		}
		fit_info a=fit_model(mo, x0, OBJ_MARGINAL), b=fit_model(mo, x1, OBJ_MARGINAL);
		check(a.status == FIT_CONVERGED && b.status == FIT_CONVERGED, "huge starting fields: status %s / %s", fit_status_name(a.status), fit_status_name(b.status));
		check(fabs(a.k-b.k) < 1e-9*a.k && fabs(x0[1]-x1[1]) < 1e-5, "huge starting fields: K %.12g from zero, %.12g from (0,400,400)", a.k, b.k);
		free_model(mo); free_data(d);
	}
	{ // one partial row, huge coupling: a completion weight of exp(-800) times a flow of exp(+400)
		const char *rows[]={"1X"};
		dataset *d=make_data(1, 2, rows, NULL);
		model *mo=build_model(d, NULL, -100.0);
		double th[3]={400, 0, 0}, g[3], k=compute_k(mo, th, g, OBJ_MARGINAL, 0);
		check(fabs(k-1.0) < 1e-12, "partial row with J=400: marginal K = %.15g, exact value is 1", k);
		check(isfinite(g[0]) && isfinite(g[1]) && isfinite(g[2]), "partial row with J=400: gradient %g %g %g", g[0], g[1], g[2]);
		for(obj=1;obj<3;obj++) {
			k=compute_k(mo, th, g, obj, obj == OBJ_FROZEN);
			check(reldiff(k, ref_k(d, th, th, obj, -100.0)) < 1e-12 && isfinite(g[0]), "partial row with J=400, %s: K %.15g vs %.15g", obj_name[obj], k, ref_k(d, th, th, obj, -100.0));
		}
		free_model(mo); free_data(d);
	}
	for(rep=0;rep<30;rep++) { // random data, parameters of size 5 to 60
		int n=3+(int)rng_uniform_int(R, 5), np=n*(n+1)/2;
		dataset *d=random_data(8, n, 3, 1, 0);
		double scale=5+rep*2, ls=-3.0, *th=random_theta(n, scale), *g=malloc(np*sizeof(double));
		model *mo=build_model(d, NULL, ls);
		for(obj=0;obj<3;obj++) {
			if (obj == OBJ_FROZEN) compute_k(mo, th, NULL, OBJ_FROZEN, 1);
			double k=compute_k(mo, th, g, obj, 0), kr=ref_k(d, th, th, obj, ls);
			if (isfinite(kr)) {
				check(reldiff(k, kr) < 1e-10, "large parameters (scale %g) %s: K %.15g vs %.15g", scale, obj_name[obj], k, kr);
				for(i=0;i<np;i++) if (!isfinite(g[i])) break;
				check(i == np, "large parameters (scale %g) %s: non-finite gradient although K is finite", scale, obj_name[obj]);
				// directional derivative along the gradient
				double gg=0, ep, *p=malloc(np*sizeof(double)), *q=malloc(np*sizeof(double));
				for(i=0;i<np;i++) gg += g[i]*g[i];
				if (isfinite(gg) && gg > 0) { // (the squared norm itself can overflow when the gradient is around 1e155)
					ep=1e-6/sqrt(gg);
					for(i=0;i<np;i++) { p[i]=th[i]+ep*g[i]; q[i]=th[i]-ep*g[i]; }
					if (obj == OBJ_FROZEN) compute_k(mo, th, NULL, OBJ_FROZEN, 1);
					double fd=(compute_k(mo, p, NULL, obj, 0)-compute_k(mo, q, NULL, obj, 0))/(2*ep);
					check(fabs(fd-gg) < 1e-3*gg, "large parameters (scale %g) %s: directional derivative %.6g vs %.6g", scale, obj_name[obj], fd, gg);
				}
				free(p); free(q);
			}
		}
		free_model(mo); free_data(d); free(th); free(g);
	}
	{ // a flow that genuinely overflows a double must be reported, never returned as a converged fit
		const char *rows[]={"11"};
		dataset *d=make_data(1, 2, rows, NULL);
		model *mo=build_model(d, NULL, 0.0);
		double x[3]={0, -800, -800};
		for(obj=0;obj<3;obj++) {
			x[0]=0; x[1]=-800; x[2]=-800;
			fit_info a=fit_model(mo, x, obj);
			check(a.status == FIT_NUMERICAL, "overflowing start, %s: reported '%s'", obj_name[obj], fit_status_name(a.status));
		}
		free_model(mo); free_data(d);
	}
	{ // separable data without regularisation: the minimum is at infinity, and that must not be called convergence
		const char *rows[]={"111", "111", "000"};
		dataset *d=make_data(3, 3, rows, NULL);
		model *mo=build_model(d, NULL, -100.0);
		double x[6]={0, 0, 0, 0, 0, 0};
		fit_info a=fit_model(mo, x, OBJ_MARGINAL);
		check(a.status != FIT_CONVERGED || a.gmax < 1e-7*mo->total_w*3, "separable data, no penalty: status '%s' with max |gradient| %.3g", fit_status_name(a.status), a.gmax);
		free_model(mo); free_data(d);
	}
}

// What each treatment of missing data assumes about WHY entries are missing, on exact population data.
//   MCAR: the pattern of missing entries does not depend on any answer.
//   MAR:  it may depend on answers that are observed in that row.
// The marginal objective is consistent under MCAR only; the fixed point of the frozen-weight iteration is
// consistent under MAR; the legacy objective under neither. (Case A1 of the external review.)
static void population_rows(int n, const double *truth, double (*p_hide)(int state), int hide_mask, dataset **out) {
	int N=1 << n, s, j, row=0;
	double lz=ref_logz(n, truth);
	dataset *d=malloc(sizeof(dataset));
	d->n=n; d->group=NULL; d->nn=1; d->m=2*N;
	d->v=malloc((size_t)d->m*n); d->w=malloc(d->m*sizeof(double)); d->n_blanks=malloc(d->m*sizeof(int));
	for(s=0;s<N;s++) {
		double p=exp(ref_h(n, truth, s)-lz), ph=p_hide(s);
		for(j=0;j<n;j++) d->v[(size_t)row*n+j]=((s >> j) & 1) ? 1 : -1;
		d->n_blanks[row]=0; d->w[row]=p*(1-ph); row++;
		for(j=0;j<n;j++) d->v[(size_t)row*n+j]=((hide_mask >> j) & 1) ? 0 : (((s >> j) & 1) ? 1 : -1);
		d->n_blanks[row]=__builtin_popcount(hide_mask); d->w[row]=p*ph; row++;
	}
	*out=d;
}
static double hide_mar2(int s) { return (s & 1) ? 0.8 : 0.2; }                 // depends on spin 0, always observed
static double hide_mcar(int s) { (void)s; return 0.5; }
static double hide_mar4(int s) { return (((s >> 0) & 1) == ((s >> 1) & 1)) ? 0.8 : 0.3; } // depends on spins 0 and 1
static void test_missingness_assumptions(void) {
	int obj, i, cas;
	printf("missingness assumptions (MCAR vs MAR) on exact population data\n");
	for(cas=0;cas<3;cas++) {
		int n=(cas == 2) ? 4 : 2, np=n*(n+1)/2, hide=(cas == 2) ? 12 : 2;
		double truth2[3]={0.7, 0.2, -0.4}, *truth=(cas == 2) ? random_theta(4, 0.4) : truth2, err[3], g[16], gmax=0;
		dataset *d;
		population_rows(n, truth, cas == 0 ? hide_mar2 : (cas == 1 ? hide_mcar : hide_mar4), hide, &d);
		model *mo=build_model(d, NULL, -14.0);
		compute_k(mo, truth, g, OBJ_MARGINAL, 0);
		for(i=0;i<np;i++) if (fabs(g[i]) > gmax) gmax=fabs(g[i]);
		for(obj=0;obj<3;obj++) {
			double x[16]={0};
			fit_model(mo, x, obj);
			err[obj]=0;
			for(i=0;i<np;i++) if (fabs(x[i]-truth[i]) > err[obj]) err[obj]=fabs(x[i]-truth[i]);
		}
		printf("    %-28s |grad marginal K| at truth %.1e;  max |fit - truth|: marginal %.1e, frozen %.1e, legacy %.1e\n",
			cas == 0 ? "MAR, 2 spins (review A1)" : (cas == 1 ? "MCAR, 2 spins" : "MAR, 4 spins"), gmax, err[OBJ_MARGINAL], err[OBJ_FROZEN], err[OBJ_LEGACY]);
		if (cas == 1) {
			check(gmax < 1e-9 && err[OBJ_MARGINAL] < 1e-5, "MCAR: marginal objective should be consistent (gradient %.3g, error %.3g)", gmax, err[OBJ_MARGINAL]);
			check(err[OBJ_FROZEN] < 1e-4, "MCAR: frozen-weight iteration should be consistent (error %.3g)", err[OBJ_FROZEN]);
		} else {
			check(gmax > 1e-3 && err[OBJ_MARGINAL] > 1e-2, "MAR: the marginal objective is expected to be inconsistent (gradient %.3g, error %.3g)", gmax, err[OBJ_MARGINAL]);
			check(err[OBJ_FROZEN] < 1e-4, "MAR: frozen-weight iteration should be consistent (error %.3g)", err[OBJ_FROZEN]);
		}
		check(err[OBJ_LEGACY] > 1e-2, "legacy objective unexpectedly consistent (error %.3g)", err[OBJ_LEGACY]);
		if (cas == 0) { // the numbers reported in the external review
			check(fabs(g[0]+0.016917885) < 1e-7 && fabs(g[1]+0.054395131) < 1e-7 && fabs(g[2]+0.163678253) < 1e-7, "review A1 gradient: %.9f %.9f %.9f", g[0], g[1], g[2]);
		}
		free_model(mo); free_data(d);
		if (cas == 2) free(truth);
	}
}

// Cross-validation with missing data must not let the held-out row influence its own fold: not through
// the starting point (Appendix B of the first external review), and not through a first-stage lambda
// chosen with its help (finding N1 and Appendix A of the second). The reference repeats the staged
// procedure by hand for every fold, from its training rows alone: fit the complete training rows at the
// lambda being scored, then all the training rows at the same lambda from there.
static double explicit_staged_cv(dataset *d, double ls, int obj) {
	int n=d->n, np=n*(n+1)/2, m=d->m, i, j, k, folds=0;
	double total=0, *x=(double *)xmalloc(np*sizeof(double));
	char *mask=(char *)xmalloc(m), *cmask=(char *)xmalloc(m);
	for(i=0;i<m;i++) {
		uint64_t st=0;
		if (d->n_blanks[i] > 0) continue;
		for(j=0;j<m;j++) { mask[j]=(j != i); cmask[j]=mask[j] && d->n_blanks[j] == 0; }
		for(k=0;k<np;k++) x[k]=0;
		model *m1=build_model(d, cmask, ls); fit_model(m1, x, obj); free_model(m1);
		model *m2=build_model(d, mask, ls); fit_model(m2, x, obj); free_model(m2);
		for(k=0;k<n;k++) if (d->v[(size_t)i*n+k] > 0) st |= ((uint64_t)1 << k);
		total += ref_h(n, x, st)-ref_logz(n, x);
		folds++;
	}
	free(x); free(mask); free(cmask);
	return total/folds;
}

static void test_cv_training_only(void) {
	const char *rows[]={"00000", "11110", "10001", "X0XX0", "XX0X1", "1XX0X", "X1X0X", "XXX10", "X0XX1", "XX0X1", "X0X1X", "XX1X1"};
	// the data set of the second review's Appendix A: six complete rows, ten partial ones
	const char *rows2[]={"00000", "11111", "10011", "11111", "01011", "11110", "11XXX", "1X0XX", "X1X1X", "0XX1X", "XX0X0", "1XX1X", "X1X0X", "X1XX1", "XX0X0", "X01XX"};
	int n=5, obj, set, li;
	double lambdas[3]={-2.0, 0.0, 1.0};
	options opt;
	printf("cross-validation with missing data: every fold repeats the staged procedure on its training rows only\n");
	memset(&opt, 0, sizeof(opt));
	for(set=0;set<2;set++) {
		dataset *d=set == 0 ? make_data(12, n, rows, NULL) : make_data(16, n, rows2, NULL);
		for(obj=0;obj<3;obj+=2) for(li=0;li<3;li++) { // marginal and frozen
			double ls=lambdas[li], total=explicit_staged_cv(d, ls, obj);
			// the same through cross(), handed two very different vectors as its optional starting point
			double *s1=random_theta(n, 0.01), *s2=random_theta(n, 2.0);
			int counts[2];
			opt.objective=obj;
			double a=cross(d, NULL, ls, s1, &opt, counts), b=cross(d, NULL, ls, s2, &opt, NULL);
			check(fabs(a-total) < 1e-5, "%s, data set %d, log-lambda %g: cross() gives %.8f, explicit staged refits give %.8f", obj_name[obj], set, ls, a, total);
			check(fabs(a-b) < 1e-9, "%s: cross() depends on the starting vector it is handed (%.8f vs %.8f)", obj_name[obj], a, b);
			check(counts[0] == (set == 0 ? 3 : 6), "cross() reports %d folds", counts[0]);
			free(s1); free(s2);
		}
		free_data(d);
	}
}

static double f_quad(double x, void *p) { double c=*(double *)p; return (x-c)*(x-c)+3; }

// Neighbourhoods larger than single flips (NN = 2, 3, 4): flows to every state within Hamming distance NN.
static void test_multi_flip(void) {
	int nn, rep, obj, i, n, np;
	printf("multi-flip neighbourhoods (NN = 2..4): objective and gradient against brute force\n");
	for(nn=2;nn<=4;nn++) for(rep=0;rep<6;rep++) {
		n=nn+1+(int)rng_uniform_int(R, 6); np=n*(n+1)/2;
		dataset *d=random_data(5+(int)rng_uniform_int(R, 12), n, rep < 2 ? 0 : 3, rep >= 3, rep == 5);
		d->nn=nn;
		double ls=-1.0+rep*0.5, *th=random_theta(n, 0.3), *th0=random_theta(n, 0.3), *g=malloc(np*sizeof(double));
		model *mo=build_model(d, NULL, ls);
		for(obj=0;obj<3;obj++) {
			double k, kr, kp, km, worst=0, gmax=0, ep=1e-5;
			if (obj == OBJ_FROZEN) compute_k(mo, th0, NULL, OBJ_FROZEN, 1);
			k=compute_k(mo, th, g, obj, 0);
			kr=ref_k(d, th, th0, obj, ls);
			check(reldiff(k, kr) < 1e-11, "NN=%d K %s n=%d rep=%d: fast %.15g reference %.15g", nn, obj_name[obj], n, rep, k, kr);
			check(compute_k(mo, th, NULL, obj, 0) == k, "NN=%d K %s: value differs with and without gradient", nn, obj_name[obj]);
			for(i=0;i<np;i++) if (fabs(g[i]) > gmax) gmax=fabs(g[i]);
			for(i=0;i<np;i++) {
				double save=th[i];
				th[i]=save+ep; kp=compute_k(mo, th, NULL, obj, 0);
				th[i]=save-ep; km=compute_k(mo, th, NULL, obj, 0);
				th[i]=save;
				if (fabs((kp-km)/(2*ep)-g[i]) > worst) worst=fabs((kp-km)/(2*ep)-g[i]);
			}
			check(worst < 2e-6*gmax, "NN=%d gradient %s n=%d rep=%d: worst finite-difference error %.3g (max |g| %.3g)", nn, obj_name[obj], n, rep, worst, gmax);
		}
		if (rep < 2) { // complete data: the three objectives are one function, and the fit is start-independent
			double a=compute_k(mo, th, NULL, OBJ_MARGINAL, 0), b=compute_k(mo, th, NULL, OBJ_LEGACY, 0), c=compute_k(mo, th, NULL, OBJ_FROZEN, 0);
			check(reldiff(a, b) < 1e-13 && reldiff(a, c) < 1e-13, "NN=%d complete data: objectives differ", nn);
			double *x=calloc(np, sizeof(double)), *y=random_theta(n, 1.0), w2=0;
			fit_info f1=fit_model(mo, x, OBJ_MARGINAL), f2=fit_model(mo, y, OBJ_MARGINAL);
			for(i=0;i<np;i++) if (fabs(x[i]-y[i]) > w2) w2=fabs(x[i]-y[i]);
			check(f1.status == FIT_CONVERGED && f2.status == FIT_CONVERGED && w2 < 1e-5, "NN=%d convex fit: status %s/%s, start dependence %.3g", nn, fit_status_name(f1.status), fit_status_name(f2.status), w2);
			free(x); free(y);
		}
		free_model(mo); free_data(d); free(th); free(th0); free(g);
	}
	{ // huge coupling with a partial row: the log-domain path for NN=2
		const char *rows[]={"1X1"};
		dataset *d=make_data(1, 3, rows, NULL); d->nn=2;
		model *mo=build_model(d, NULL, -100.0);
		double th[6]={400, 0.3, -0.2, 0.1, 0, 0.2}, g[6];
		for(obj=0;obj<3;obj++) {
			double k=compute_k(mo, th, g, obj, obj == OBJ_FROZEN), kr=ref_k(d, th, th, obj, -100.0);
			check(reldiff(k, kr) < 1e-11 && isfinite(g[0]) && isfinite(g[5]), "NN=2, J=400, %s: K %.15g vs %.15g", obj_name[obj], k, kr);
		}
		free_model(mo); free_data(d);
	}
	// population data: what each treatment of missing data needs when flows go to two-flip neighbours as well
	printf("multi-flip neighbourhoods: consistency on exact population data (NN = 2)\n");
	for(i=0;i<2;i++) {
		double *truth=random_theta(4, 0.4), err[3];
		dataset *d;
		population_rows(4, truth, i == 0 ? hide_mcar : hide_mar4, 12, &d);
		d->nn=2;
		model *mo=build_model(d, NULL, -14.0);
		for(obj=0;obj<3;obj++) {
			double x[10]={0}; int k;
			fit_model(mo, x, obj);
			err[obj]=0;
			for(k=0;k<10;k++) if (fabs(x[k]-truth[k]) > err[obj]) err[obj]=fabs(x[k]-truth[k]);
		}
		printf("    %-6s max |fit - truth|: marginal %.1e, frozen %.1e, legacy %.1e\n", i == 0 ? "MCAR" : "MAR", err[OBJ_MARGINAL], err[OBJ_FROZEN], err[OBJ_LEGACY]);
		if (i == 0) {
			check(err[OBJ_MARGINAL] < 1e-5 && err[OBJ_FROZEN] < 1e-4, "NN=2, MCAR: marginal (%.3g) and frozen (%.3g) should recover the truth", err[OBJ_MARGINAL], err[OBJ_FROZEN]);
			check(err[OBJ_LEGACY] > 1e-2, "NN=2, MCAR: legacy unexpectedly consistent (%.3g)", err[OBJ_LEGACY]);
		} else {
			// frozen weights with flows that also flip missing nodes stay consistent under MAR; the marginal objective does not
			check(err[OBJ_FROZEN] < 1e-4, "NN=2, MAR: frozen-weight iteration should recover the truth (error %.3g)", err[OBJ_FROZEN]);
			check(err[OBJ_MARGINAL] > 1e-2, "NN=2, MAR: the marginal objective is expected to be inconsistent (error %.3g)", err[OBJ_MARGINAL]);
		}
		free_model(mo); free_data(d); free(truth);
	}
	for(nn=2;nn<=3;nn++) { // the review's two-spin MAR example and a four-spin one, for NN = 2 and 3
		double truth2[3]={0.7, 0.2, -0.4}, *truth4=random_theta(4, 0.4);
		int cas;
		for(cas=0;cas<2;cas++) {
			int n=(cas == 0) ? 2 : 4, np=n*(n+1)/2, k;
			double *truth=(cas == 0) ? truth2 : truth4, x[10]={0}, e=0;
			dataset *d;
			if (nn > n) continue;
			population_rows(n, truth, cas == 0 ? hide_mar2 : hide_mar4, cas == 0 ? 2 : 12, &d);
			d->nn=nn;
			model *mo=build_model(d, NULL, -14.0);
			fit_info fi=fit_model(mo, x, OBJ_FROZEN);
			for(k=0;k<np;k++) if (fabs(x[k]-truth[k]) > e) e=fabs(x[k]-truth[k]);
			check(fi.status == FIT_CONVERGED && e < 1e-4, "NN=%d, MAR, %d spins: frozen-weight error %.3g (%s)", nn, n, e, fit_status_name(fi.status));
			free_model(mo); free_data(d);
		}
		free(truth4);
	}
	{ // cross-validation with NN=2 against explicit refits
		options opt; memset(&opt, 0, sizeof(opt));
		dataset *d=random_data(24, 5, 0, 1, 1); d->nn=2;
		double a=cross(d, NULL, 0.2, NULL, &opt, NULL), b=manual_cv(d, 0.2, OBJ_MARGINAL, 1, 0);
		check(fabs(a-b) < 1e-6, "NN=2 cross-validation: %.10f vs explicit %.10f", a, b);
		free_data(d);
	}
}

static double f_nan(double x, void *p) { (void)x; (void)p; return NAN; }
static double f_half_nan(double x, void *p) { double c=*(double *)p; return x < 0 ? NAN : (x-c)*(x-c); }
static double f_two(double x, void *p) { (void)p; return fmin((x+1.2)*(x+1.2)+0.5, 2*(x-2.3)*(x-2.3)); } // two minima; global at 2.3
static void test_search(void) {
	double c, x; int bd;
	printf("search over the regularisation strength\n");
	for(c=-1.9;c<3.95;c+=0.137) { x=search_lambda(f_quad, &c, 0, 0, &bd); check(fabs(x-c) < 2e-3 && !bd, "search: minimum at %.3f found at %.4f (boundary flag %d)", c, x, bd); }
	c=-5; x=search_lambda(f_quad, &c, 0, 0, &bd); check(fabs(x+2) < 2e-3 && bd, "search: minimum below the range gives %.4f, flag %d", x, bd);
	c=9; x=search_lambda(f_quad, &c, 0, 0, &bd); check(fabs(x-4) < 2e-3 && bd, "search: minimum above the range gives %.4f, flag %d", x, bd);
	x=search_lambda(f_two, NULL, 0, 0, &bd); check(fabs(x-2.3) < 2e-3, "search: global minimum of a two-well function found at %.4f", x);
	// with the downward extension the penalty has no floor at -2
	c=-5.3; x=search_lambda(f_quad, &c, 0, 1, &bd); check(fabs(x-c) < 2e-3 && !bd, "search (extended): minimum at -5.3 found at %.4f, flag %d", x, bd);
	c=-2.2; x=search_lambda(f_quad, &c, 0, 1, &bd); check(fabs(x-c) < 2e-3 && !bd, "search (extended): minimum at -2.2 found at %.4f, flag %d", x, bd);
	c=-1.8; x=search_lambda(f_quad, &c, 0, 1, &bd); check(fabs(x-c) < 2e-3 && !bd, "search (extended): minimum at -1.8 found at %.4f, flag %d", x, bd);
	c=-30; x=search_lambda(f_quad, &c, 0, 1, &bd); check(fabs(x+8) < 2e-3 && bd, "search (extended): minimum far below gives %.4f, flag %d", x, bd);
	c=1.234; x=search_lambda(f_quad, &c, 0, 1, &bd); check(fabs(x-c) < 2e-3 && !bd, "search (extended): interior minimum found at %.4f", x);
	// a function that is invalid everywhere must not produce a "best" value
	x=search_lambda(f_nan, NULL, 0, 0, &bd); check(isnan(x), "search: all-invalid function returned %.4f", x);
	x=search_lambda(f_nan, NULL, 0, 1, &bd); check(isnan(x), "search (extended): all-invalid function returned %.4f", x);
	// invalid at one end only: the valid minimum is still found
	c=2.9; x=search_lambda(f_half_nan, &c, 0, 1, &bd); check(fabs(x-c) < 2e-3, "search: partly-invalid function, minimum at 2.9 found at %.4f", x);
}

// ---- the code that replaced the GNU Scientific Library ----------------------------------------
// Known answers were recorded from GSL itself (validation/gsl_reference/check_against_gsl.c).
static double f_square(double x, void *p) { (void)p; return x*x; }
static double f_gauss(double x, void *p) { (void)p; return exp(-x*x); }
static double f_peak(double x, void *p) { (void)p; return 1.0/(1e-4+(x-0.3)*(x-0.3)); }
static double f_pole(double x, void *p) { (void)p; return 1.0/x; }

static void test_numerics(void) {
	const double euler=0.57721566490153286, pi=3.14159265358979324;
	unsigned long want[4]={62917, 848763, 486825, 537357};
	double c[8]={5, 3, 2, 1, 1, 1, 4, 2}, x, u, g1, g2, res, worst;
	mpf_rng r;
	prob p;
	int i, st;
	printf("random numbers, special functions, quadrature, NSB entropy\n");

	rng_seed(&r, 20230131);
	for(i=0;i<4;i++) { unsigned long k=rng_uniform_int(&r, 1000000); check(k == want[i], "random integer %d is %lu, expected %lu", i, k, want[i]); }
	u=rng_uniform(&r); g1=rng_gaussian(&r, 1.0); g2=rng_gaussian(&r, 1.0);
	check(u == 0.61667281831614673, "uniform draw %.17g", u);
	check(fabs(g1-0.083662098571901225) < 1e-13 && fabs(g2-0.24483154424005346) < 1e-13, "Gaussian draws %.17g %.17g", g1, g2);
	rng_seed(&r, 0); u=rng_uniform(&r); rng_seed(&r, 1); check(u == rng_uniform(&r), "seed 0 is not treated as seed 1");
	rng_seed(&r, 5); u=rng_uniform(&r); rng_seed(&r, 4294967296UL+5); check(u == rng_uniform(&r), "only the low 32 bits of the seed should matter");
	// seeds whose low 32 bits are zero used to leave the generator returning zero for ever
	rng_seed(&r, 1); u=rng_uniform(&r);
	rng_seed(&r, 4294967296UL); check(rng_uniform(&r) == u && r.s1 != 0, "seed 2^32 should behave as seed 1");
	rng_seed(&r, 2*4294967296UL); check(rng_uniform(&r) == u, "seed 2^33 should behave as seed 1");
	rng_seed(&r, 4294967296UL); x=NAN;
	if (r.s1 || r.s2 || r.s3) x=rng_gaussian(&r, 1.0); // (with an all-zero state this call would never return)
	check(isfinite(x), "Gaussian draw after seed 2^32");
	for(i=0, st=1;i<20000;i++) { unsigned long k=rng_uniform_int(&r, 7); x=rng_uniform(&r); if (k > 6 || x < 0 || x >= 1) st=0; }
	check(st, "random draws out of range");

	check(fabs(digamma(1.0)+euler) < 1e-15, "digamma(1) = %.17g", digamma(1.0));
	check(fabs(digamma(0.5)+euler+2*log(2.0)) < 2e-15, "digamma(1/2) = %.17g", digamma(0.5));
	check(fabs(trigamma(1.0)-pi*pi/6) < 1e-15, "trigamma(1) = %.17g", trigamma(1.0));
	check(fabs(trigamma(0.5)-pi*pi/2) < 4e-15, "trigamma(1/2) = %.17g", trigamma(0.5));
	for(worst=0, x=0.01;x<1e4;x*=1.37) { // recurrences, across the switch to the asymptotic series
		double a=fabs(digamma(x+1)-digamma(x)-1/x)*x, b=fabs(trigamma(x)-trigamma(x+1)-1/(x*x))*x*x;
		if (a > worst) worst=a;
		if (b > worst) worst=b;
	}
	check(worst < 1e-10, "digamma/trigamma recurrences violated by %.3g", worst); // (scaled by x and x^2; the differences cancel as x grows)

	st=integrate(f_square, NULL, 0, 1, 1e-12, 100, &res); check(st == 0 && fabs(res-1.0/3) < 1e-14, "integral of x^2: status %d, %.17g", st, res);
	st=integrate(f_gauss, NULL, -5, 5, 1e-12, 200, &res); check(st == 0 && fabs(res-sqrt(pi)*erf(5.0)) < 1e-12, "Gaussian integral: status %d, %.17g", st, res);
	st=integrate(f_peak, NULL, 0, 1, 1e-10, 400, &res); x=(atan(70.0)+atan(30.0))/0.01;
	check(st == 0 && fabs(res-x) < 1e-9*x, "peaked integrand: status %d, %.12g instead of %.12g", st, res, x);
	st=integrate(f_peak, NULL, 0, 1, 1e-10, 3, &res); check(st != 0, "integration reported success with too few intervals");
	st=integrate(f_pole, NULL, 0, 1, 1e-10, 400, &res); check(st != 0, "integration of 1/x from 0 reported success");
	st=integrate(f_square, NULL, 0, 1, 1e-10, 0, &res); check(st != 0, "integration with no intervals allowed reported success");
	st=integrate(f_square, NULL, 0, 1, 1e-10, -5, &res); check(st != 0, "integration with a negative interval limit reported success");

	p.n=8; p.norm=-1; p.p=c;
	x=entropy_nsb(&p);
	check(fabs(x-2.849845240970) < 1e-9, "NSB entropy %.12f, expected 2.849845240970", x);
	p.n=0; check(entropy_nsb(&p) == -1, "NSB entropy of nothing should fail");
}

int main(void) {
	rng_seed(R, 20230131);
	test_numerics();
	test_objective();
	test_weights_and_duplicates();
	test_enumeration();
	test_probs_file();
	test_samplers();
	test_fit();
	test_consistency();
	test_missingness_assumptions();
	test_extreme_parameters();
	test_cross_validation();
	test_cv_training_only();
	test_multi_flip();
	test_search();
	printf("%d checks, %d failed\n", n_checks, n_failed);
	return n_failed ? 1 : 0;
}
