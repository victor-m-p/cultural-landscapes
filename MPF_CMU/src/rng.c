// Random numbers, with no external library.
//
// The generator is the combined Tausworthe generator of L'Ecuyer (Math. Comp. 65, 203 (1996)), period
// about 2^88. Earlier versions of this program took it, and the three derived distributions below, from
// the GNU Scientific Library (gsl_rng_taus, gsl_rng_uniform, gsl_rng_uniform_int, gsl_ran_gaussian). The
// code here follows the same algorithms step for step, so that a given seed produces exactly the same
// stream as before: `mpf -g --seed N` writes the same files with or without GSL. (The one exception is a
// seed that made GSL's generator degenerate; see rng_seed.)
#include "mpf.h"

static uint32_t taus_step(mpf_rng *r) {
	r->s1=((r->s1 & 4294967294U) << 12) ^ (((r->s1 << 13) ^ r->s1) >> 19);
	r->s2=((r->s2 & 4294967288U) << 4) ^ (((r->s2 << 2) ^ r->s2) >> 25);
	r->s3=((r->s3 & 4294967280U) << 17) ^ (((r->s3 << 3) ^ r->s3) >> 11);
	return r->s1 ^ r->s2 ^ r->s3;
}

// Only the low 32 bits of the seed are used. If they are zero the starting words would all be zero and
// the generator would return zero for ever (GSL has this flaw for seeds such as 2^32, and loops for ever
// when asked for a Gaussian); such a seed is treated as 1, as GSL already does for a seed of exactly 0.
void rng_seed(mpf_rng *r, unsigned long seed) {
	int i;
	uint32_t s=(uint32_t)(seed & 0xFFFFFFFFUL);
	if (s == 0) s=1;
	r->s1=69069U*s;                  // a linear congruential generator supplies the three starting words
	r->s2=69069U*r->s1;
	r->s3=69069U*r->s2;
	for(i=0;i<6;i++) taus_step(r);   // warm up
}

// uniform on [0, 1)
double rng_uniform(mpf_rng *r) {
	return taus_step(r)/4294967296.0;
}

// uniform on the integers 0 .. n-1; n must be between 1 and 2^32 - 1
unsigned long rng_uniform_int(mpf_rng *r, unsigned long n) {
	unsigned long scale, k;
	if (n == 0 || n > 4294967295UL) die("internal error: rng_uniform_int called with n = %lu", n);
	scale=4294967295UL/n;
	do { k=taus_step(r)/scale; } while (k >= n); // rejection keeps the result exactly uniform
	return k;
}

// normal with mean zero and standard deviation sigma (polar Box-Muller)
double rng_gaussian(mpf_rng *r, double sigma) {
	double x, y, r2;
	do {
		do { x=rng_uniform(r); } while (x == 0);
		do { y=rng_uniform(r); } while (y == 0);
		x=-1+2*x; y=-1+2*y;
		r2=x*x+y*y;
	} while (r2 > 1.0 || r2 == 0);
	return sigma*y*sqrt(-2.0*log(r2)/r2);
}
