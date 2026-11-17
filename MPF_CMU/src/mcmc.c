// Exact enumeration (partition function, KL divergence, probabilities of all states), exact sampling,
// and the single-spin-flip Monte Carlo sampler.
#include "mpf.h"

// The local fields phi of the spin vector s, and its H (returned).
static double init_fields(int n, const double *J, const double *h, const double *s, double *phi) {
	double H=0;
	int k, l;
	for(k=0;k<n;k++) {
		phi[k]=h[k];
		for(l=0;l<n;l++) phi[k] += J[k*n+l]*s[l];
		H += 0.5*s[k]*(phi[k]+h[k]);
	}
	return H;
}

// H of the 2^n_low states of nodes 0 .. n_low-1 (the other nodes stay as they are in s), visited in
// Gray-code order so that each costs O(n), stored by state index. On entry nodes 0 .. n_low-1 of s are -1.
static void gray_fill(int n, int n_low, const double *J, const double *h, double *s, double *phi, double *out) {
	uint64_t N=(uint64_t)1 << n_low, i, state=0;
	double H=init_fields(n, J, h, s, phi);
	int k, b;
	for(i=0;;) {
		out[state]=H;
		i++;
		if (i == N) break;
		b=__builtin_ctzll(i);
		H -= 2*s[b]*phi[b];
		s[b]=-s[b];
		state ^= ((uint64_t)1 << b);
		for(k=0;k<n;k++) phi[k] += 2*J[k*n+b]*s[b];
	}
}

// H for a fully-specified row
double h_value(int n, const double *theta, const int8_t *v) {
	int k, l, cnt=0;
	double H=0, sk;
	for(k=0;k<n;k++) {
		sk=(v[k] > 0) ? 1.0 : -1.0;
		for(l=k+1;l<n;l++) { H += sk*((v[l] > 0) ? 1.0 : -1.0)*theta[cnt]; cnt++; }
		H += sk*theta[n*(n-1)/2+k];
	}
	return H;
}

// log pseudo-likelihood of a fully-specified row: sum_k log p(s_k | all other nodes)
double pseudo_loglik(int n, const double *theta, const int8_t *v) {
	int k, l;
	double *J=(double *)xmalloc((size_t)n*n*sizeof(double)), phi, x, total=0;
	const double *h=theta+n*(n-1)/2;
	unpack_couplings(n, theta, J);
	for(k=0;k<n;k++) {
		phi=h[k];
		for(l=0;l<n;l++) phi += J[k*n+l]*((v[l] > 0) ? 1.0 : -1.0);
		x=2*((v[k] > 0) ? 1.0 : -1.0)*phi;
		total += (x > 0) ? -log1p(exp(-x)) : x-log1p(exp(x));
	}
	free(J);
	return total;
}

// Streaming log-sum-exp accumulator: value = sum * exp(top). Keeps exp() in range whatever the
// size of the parameters.
typedef struct { double top, sum, acc; } lse;

static inline void lse_add(lse *a, double H, double payload) {
	double e;
	if (H > a->top) {
		e=exp(a->top-H);
		a->sum=a->sum*e+1.0;
		a->acc=a->acc*e+payload;
		a->top=H;
	} else {
		e=exp(H-a->top);
		a->sum += e;
		a->acc += e*payload;
	}
}

static void lse_merge(lse *a, const lse *b) {
	double e;
	if (b->sum == 0) return;
	if (b->top > a->top) {
		e=exp(a->top-b->top);
		a->sum=a->sum*e+b->sum;
		a->acc=a->acc*e+b->acc;
		a->top=b->top;
	} else {
		e=exp(b->top-a->top);
		a->sum += e*b->sum;
		a->acc += e*b->acc;
	}
}

// Enumerate all 2^n states in Gray-code order (one spin flip, O(n) work, per state).
// Returns log Z for theta. If theta2 != NULL also returns log Z for theta2 and the mean of (H - H2)
// under the distribution defined by theta. The top bits are split across threads.
static void enumerate(int n, const double *theta, const double *theta2, double *logz, double *logz2, double *mean_diff) {
	int tbits, n_low, chunk, n_chunks;
	double *J, *J2=NULL;
	const double *h=theta+n*(n-1)/2, *h2=theta2 ? theta2+n*(n-1)/2 : NULL;
	lse tot1={-INFINITY, 0, 0}, tot2={-INFINITY, 0, 0}, *part1, *part2;

	if (n > ENUM_MAX_N) die("cannot enumerate 2^%d states (limit is n = %d)", n, ENUM_MAX_N);
	J=(double *)xmalloc((size_t)n*n*sizeof(double));
	unpack_couplings(n, theta, J);
	if (theta2) { J2=(double *)xmalloc((size_t)n*n*sizeof(double)); unpack_couplings(n, theta2, J2); }

	tbits=(n > 12) ? 6 : 0;
	n_low=n-tbits;
	n_chunks=1 << tbits;
	part1=(lse *)xmalloc(n_chunks*sizeof(lse)); // per-chunk results, merged in order so that the answer
	part2=(lse *)xmalloc(n_chunks*sizeof(lse)); // does not depend on the number of threads

#pragma omp parallel if(n >= 20)
	{
		double *s=(double *)xmalloc(n*sizeof(double)), *phi=(double *)xmalloc(n*sizeof(double)), *phi2=(double *)xmalloc(n*sizeof(double));
		lse loc1, loc2;
		int k, b;
		uint64_t i, N=(uint64_t)1 << n_low;
		double H, H2;

#pragma omp for schedule(static) nowait
		for(chunk=0;chunk<n_chunks;chunk++) {
			loc1.top=-INFINITY; loc1.sum=0; loc1.acc=0; loc2=loc1;
			for(k=0;k<n_low;k++) s[k]=-1.0;
			for(k=0;k<tbits;k++) s[n_low+k]=((chunk >> k) & 1) ? 1.0 : -1.0;
			H=init_fields(n, J, h, s, phi);
			H2=theta2 ? init_fields(n, J2, h2, s, phi2) : 0;
			for(i=0;;) {
				lse_add(&loc1, H, theta2 ? H-H2 : 0.0);
				if (theta2) lse_add(&loc2, H2, 0.0);
				i++;
				if (i == N) break;
				b=__builtin_ctzll(i);
				H -= 2*s[b]*phi[b];
				if (theta2) H2 -= 2*s[b]*phi2[b];
				s[b]=-s[b];
				for(k=0;k<n;k++) phi[k] += 2*J[k*n+b]*s[b];
				if (theta2) for(k=0;k<n;k++) phi2[k] += 2*J2[k*n+b]*s[b];
			}
			part1[chunk]=loc1; part2[chunk]=loc2;
		}
		free(s); free(phi); free(phi2);
	}
	for(chunk=0;chunk<n_chunks;chunk++) { lse_merge(&tot1, &part1[chunk]); lse_merge(&tot2, &part2[chunk]); }
	free(part1); free(part2);

	*logz=log(tot1.sum)+tot1.top;
	if (theta2) {
		*logz2=log(tot2.sum)+tot2.top;
		*mean_diff=tot1.acc/tot1.sum;
	}
	free(J); free(J2);
}

double log_z(int n, const double *theta) {
	double lz;
	enumerate(n, theta, NULL, &lz, NULL, NULL);
	return lz;
}

// KL(truth || inferred) in nats, by full enumeration
double full_kl(int n, const double *truth, const double *inferred) {
	double lz_t, lz_i, diff;
	enumerate(n, truth, inferred, &lz_t, &lz_i, &diff);
	return diff-lz_t+lz_i;
}

// Fill H[state] for all 2^n states, state index bit i = node i
static double *all_h(int n, const double *theta) {
	double *Hall=(double *)xmalloc(((size_t)1 << n)*sizeof(double)), *J=(double *)xmalloc((size_t)n*n*sizeof(double));
	double *s=(double *)xmalloc(n*sizeof(double)), *phi=(double *)xmalloc(n*sizeof(double));
	int k;
	unpack_couplings(n, theta, J);
	for(k=0;k<n;k++) s[k]=-1.0;
	gray_fill(n, n, J, theta+n*(n-1)/2, s, phi, Hall);
	free(J); free(s); free(phi);
	return Hall;
}

// Write every configuration, its H, and its probability (one line per state, node 0 first).
// The states are written in index order, in blocks of 2^20: within a block the low bits are enumerated
// in Gray-code order into a small buffer, so memory stays at a few MB whatever n is.
void compute_probs(int n, const double *theta, const char *filename) {
	int n_low=(n > 20) ? 20 : n, k;
	uint64_t n_blocks, block, i, N_low=(uint64_t)1 << n_low;
	double *Hblk, *J, *s, *phi, lz;
	char line[MAX_NODES+1];
	FILE *f;

	if (n > ENUM_MAX_N) die("cannot enumerate 2^%d states (limit is n = %d)", n, ENUM_MAX_N);
	n_blocks=(uint64_t)1 << (n-n_low);
	lz=log_z(n, theta);
	Hblk=(double *)xmalloc(N_low*sizeof(double));
	J=(double *)xmalloc((size_t)n*n*sizeof(double)); s=(double *)xmalloc(n*sizeof(double)); phi=(double *)xmalloc(n*sizeof(double));
	unpack_couplings(n, theta, J);
	f=fopen(filename, "w");
	if (f == NULL) die("cannot write '%s'", filename);
	line[n]='\0';
	for(block=0;block<n_blocks;block++) {
		for(k=0;k<n_low;k++) s[k]=-1.0;
		for(k=n_low;k<n;k++) s[k]=((block >> (k-n_low)) & 1) ? 1.0 : -1.0;
		gray_fill(n, n_low, J, theta+n*(n-1)/2, s, phi, Hblk);
		for(i=0;i<N_low;i++) {
			uint64_t full=(block << n_low) | i;
			for(k=0;k<n;k++) line[k]=((full >> k) & 1) ? '1' : '0';
			fprintf(f, "%s %.10e %.10e\n", line, Hblk[i], exp(Hblk[i]-lz));
		}
	}
	if (ferror(f) || fclose(f) != 0) die("error writing '%s'", filename);
	free(Hblk); free(J); free(s); free(phi);
}

// Draw n_obs independent states from the exact distribution (n <= EXACT_MAX_N)
void sample_exact(int n, const double *theta, int n_obs, uint64_t *out, mpf_rng *r) {
	uint64_t N=(uint64_t)1 << n, i, lo, hi, mid;
	double *c, mx, total, u;
	int j;

	if (n > EXACT_MAX_N) die("exact sampling needs n <= %d", EXACT_MAX_N);
	c=all_h(n, theta);
	mx=c[0];
	for(i=1;i<N;i++) if (c[i] > mx) mx=c[i];
	total=0;
	for(i=0;i<N;i++) { total += exp(c[i]-mx); c[i]=total; } // cumulative, unnormalised
	for(j=0;j<n_obs;j++) {
		u=rng_uniform(r)*total;
		lo=0; hi=N-1;
		while(lo < hi) { // first index with c[index] > u
			mid=lo+(hi-lo)/2;
			if (c[mid] > u) hi=mid; else lo=mid+1;
		}
		out[j]=lo;
	}
	free(c);
}

// Single-spin-flip heat-bath sampler: iter/2 sweeps annealing from infinite temperature, then iter/2
// sweeps at the target temperature. NOTE: for strong couplings this does not reach equilibrium in the
// 1000 sweeps used by -g (see CLAUDE_IMPROVEMENTS_SUMMARY.md); prefer sample_exact when n allows.
void mcmc_sampler(uint64_t *config, int iter, int n, const double *theta, mpf_rng *r) {
	int i, j, pos, phase, steps=n*iter/2;
	double *J=(double *)xmalloc((size_t)n*n*sizeof(double)), running, scale, sp, e;
	const double *h=theta+n*(n-1)/2;

	unpack_couplings(n, theta, J);
	for(phase=0;phase<2;phase++) {
		for(i=0;i<steps;i++) {
			scale=(phase == 0) ? i*1.0/steps : 1.0;
			pos=(int)rng_uniform_int(r, n);
			sp=((*config >> pos) & 1) ? 1.0 : -1.0;
			running=h[pos];
			for(j=0;j<n;j++) running += J[pos*n+j]*(((*config >> j) & 1) ? 1.0 : -1.0);
			running *= -2*sp*scale; // log of p(flipped) / p(current)
			e=(running > 0) ? 1.0/(1.0+exp(-running)) : exp(running)/(1.0+exp(running));
			if (rng_uniform(r) < e) *config ^= ((uint64_t)1 << pos);
		}
	}
	free(J);
}
