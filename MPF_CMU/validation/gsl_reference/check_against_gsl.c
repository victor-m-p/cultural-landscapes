// Checks that the self-contained code which replaced the GNU Scientific Library gives the same numbers:
//   1. the random number generator and the three distributions drawn from it (must be identical),
//   2. the digamma and trigamma functions,
//   3. the NSB entropy estimate (special functions + quadrature).
// This is the only file here that needs GSL. From MPF_CMU/validation/gsl_reference, after `make` in ../../src:
//   gcc -O2 -I../../src check_against_gsl.c nsb_gsl.c ../../src/rng.o ../../src/nsb.o ../../src/mpf.o \
//       ../../src/mcmc.o ../../src/nlin.o -fopenmp -lm `gsl-config --cflags --libs` -o check_against_gsl
//   ./check_against_gsl > ../results/check_against_gsl.txt
#include "mpf.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf.h>

double entropy_nsb_gsl(prob *p);

int main(void) {
	unsigned long seeds[8]={1, 0, 2, 12345, 20230131, 4294967296UL+7, 0xFFFFFFFFUL, 0x8000000000000123UL};
	unsigned long ns[7]={1, 2, 20, 1000, 1UL << 24, 1UL << 31, 4294967295UL};
	long mismatches=0, draws=0;
	int si, i, k, fails_old=0, fails_new=0, fails_differ=0, cases=0;
	double x, worst_psi=0, worst_tri=0, worst_nsb=0;
	gsl_rng *g=gsl_rng_alloc(gsl_rng_taus);
	mpf_rng r;

	for(si=0;si<8;si++) {
		gsl_rng_set(g, seeds[si]); rng_seed(&r, seeds[si]);
		for(i=0;i<200000;i++) { draws++; if (gsl_rng_uniform(g) != rng_uniform(&r)) mismatches++; }
		for(k=0;k<7;k++) for(i=0;i<50000;i++) { draws++; if (gsl_rng_uniform_int(g, ns[k]) != rng_uniform_int(&r, ns[k])) mismatches++; }
		for(i=0;i<200000;i++) { draws++; if (gsl_ran_gaussian(g, 1.7) != rng_gaussian(&r, 1.7)) mismatches++; }
		// interleaved, as the program uses them
		for(i=0;i<100000;i++) {
			draws += 3;
			if (gsl_ran_gaussian(g, 0.3) != rng_gaussian(&r, 0.3)) mismatches++;
			if (gsl_rng_uniform_int(g, 37) != rng_uniform_int(&r, 37)) mismatches++;
			if (gsl_rng_uniform(g) != rng_uniform(&r)) mismatches++;
		}
	}
	printf("random numbers: %ld draws compared with GSL (8 seeds; uniform, uniform integer, Gaussian, interleaved): %ld differ\n", draws, mismatches);

	{ // drawn one at a time, from GSL: the order in which a compiler evaluates function arguments is not defined
		unsigned long k[4];
		double u, g1, g2;
		gsl_rng_set(g, 20230131);
		for(i=0;i<4;i++) k[i]=gsl_rng_uniform_int(g, 1000000);
		u=gsl_rng_uniform(g); g1=gsl_ran_gaussian(g, 1.0); g2=gsl_ran_gaussian(g, 1.0);
		printf("known answers for the unit test (from GSL), seed 20230131: uniform_int(1000000) = %lu %lu %lu %lu; then uniform = %.17g; then gaussian(1) = %.17g %.17g\n", k[0], k[1], k[2], k[3], u, g1, g2);
	}

	for(x=1e-3;x<1e8;x*=1.0173) {
		double ref=gsl_sf_psi_n(0, x), a=fabs(digamma(x)-ref)/(fabs(ref) > 1 ? fabs(ref) : 1), b=fabs(trigamma(x)-gsl_sf_psi_n(1, x))/fabs(gsl_sf_psi_n(1, x));
		if (a > worst_psi) worst_psi=a; // digamma has a zero near 1.4616, so the difference is relative to max(1, |digamma|)
		if (b > worst_tri) worst_tri=b;
	}
	printf("digamma: largest difference from GSL for x in [1e-3, 1e8], relative to max(1, |digamma|): %.3g\n", worst_psi);
	printf("trigamma: largest relative difference from GSL for x in [1e-3, 1e8]: %.3g\n", worst_tri);
	{ // a fixed case, recorded as a known answer in the unit tests
		double c[8]={5, 3, 2, 1, 1, 1, 4, 2}, c2[8]={5, 3, 2, 1, 1, 1, 4, 2};
		prob p, q;
		p.n=8; p.norm=-1; p.p=c; q.n=8; q.norm=-1; q.p=c2;
		printf("known answer for the unit test: counts {5,3,2,1,1,1,4,2}: GSL-based %.12f, self-contained %.12f\n", entropy_nsb_gsl(&p), entropy_nsb(&q));
	}

	rng_seed(&r, 99);
	for(cases=0;cases<400;cases++) {
		prob p, q;
		int K=2+(int)rng_uniform_int(&r, cases < 200 ? 40 : 400), maxc=1+(int)rng_uniform_int(&r, cases % 3 == 0 ? 2 : 60);
		double a, b;
		p.n=K; p.norm=-1; p.p=(double *)xmalloc(K*sizeof(double));
		q.n=K; q.norm=-1; q.p=(double *)xmalloc(K*sizeof(double));
		for(i=0;i<K;i++) { p.p[i]=1+rng_uniform_int(&r, maxc); if (cases % 5 == 0) p.p[i] *= 0.5; q.p[i]=p.p[i]; } // some fractional counts, as from weighted rows
		a=entropy_nsb_gsl(&p); b=entropy_nsb(&q);
		if (a == -1) fails_old++;
		if (b == -1) fails_new++;
		if ((a == -1) != (b == -1)) fails_differ++;
		if (a != -1 && b != -1 && fabs(a-b) > worst_nsb) worst_nsb=fabs(a-b);
		if (cases < 4) printf("  example %d: %d states, GSL-based %.12f, self-contained %.12f\n", cases, K, a, b);
		free(p.p); free(q.p);
	}
	printf("NSB entropy on %d random count vectors: integration failed (returns -1) in %d with the GSL-based code and %d with the self-contained code (%d cases where only one failed); "
		"largest absolute difference where both succeed: %.3g bits\n", cases, fails_old, fails_new, fails_differ, worst_nsb);
	return mismatches != 0;
}
