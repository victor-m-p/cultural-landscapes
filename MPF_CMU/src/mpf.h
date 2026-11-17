#ifndef MPF_H
#define MPF_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// state of the random number generator (rng.c)
typedef struct { uint32_t s1, s2, s3; } mpf_rng;

// ---- limits -------------------------------------------------------------------------------
#define MAX_NODES 62        // states are packed into a uint64_t
#define MAX_BLANKS 20       // an observation with b blanks is expanded into 2^b completions
#define MAX_NN 6            // largest neighbourhood: all states within this Hamming distance of a data state
#define EXACT_MAX_N 24      // largest n for which we enumerate all 2^n states inside loops (CV, sampling)
#define ENUM_MAX_N 30       // largest n for which we enumerate all 2^n states at all (KL, -z)

// ---- which Partial-MPF objective to use for rows with missing entries ----------------------
// Rows without blanks are treated identically by all three (ordinary MPF, convex).
// Population consistency (see CLAUDE_IMPROVEMENTS_SUMMARY.md): OBJ_MARGINAL requires missingness that does
// not depend on the answers (MCAR); the OBJ_FROZEN fixed point requires only missing-at-random; OBJ_LEGACY
// is biased under both.
#define OBJ_MARGINAL 0  // MPF applied to the marginal over the observed nodes (a single objective; the default)
#define OBJ_LEGACY   1  // the objective used for the published paper (weights differentiated through)
#define OBJ_FROZEN   2  // completion weights frozen while minimising, re-estimated, iterated to a fixed point

// ---- raw data -------------------------------------------------------------------------------
typedef struct {
	int m;            // number of rows in the file
	int n;            // number of nodes
	int8_t *v;        // m*n entries: +1, -1, or 0 (blank)
	double *w;        // m weights ("multiplicity")
	int *n_blanks;    // number of blanks in each row
	int *group;       // optional group id per row (rows of one entry share an id); NULL if none
	int nn;           // neighbourhood: flows go to every state within Hamming distance nn (1 = single flips)
} dataset;

// ---- everything the objective needs, built from a subset of the rows of a dataset ----------
typedef struct {
	int n;            // nodes
	int np;           // parameters: n(n-1)/2 couplings (row-major, i<j) then n fields
	int D;            // unique observations
	long R;           // total rows after expanding blanks
	int maxc;         // largest number of completions of any observation
	double *sig;      // R*n spins (+1/-1)
	double *omask;    // D*n: 1 if node is observed in that observation, 0 if blank
	double *mult;     // D weights
	long *start;      // D+1: first expanded row of each observation
	double *lwfrozen; // R frozen log completion weights (OBJ_FROZEN)
	double total_w;   // sum of weights
	int nn;           // neighbourhood size (see dataset)
	double n_prox;    // neighbours per fully-observed state: sum_{j=1..nn} C(n, j)
	double sparsity;  // penalty multiplier: n_prox * 10^log_sparsity * total_w / np
	// scratch
	double *J, *gJ, *X, *T, *H, *lw, *g, *u, *Rk, *LQ;
	// scratch for nn > 1: the subsets of observed nodes that are flipped together, and their flows
	int *sub_mem, *sub_size;
	double *E, *Rv, *LQs, *Ru, *Pu;
} model;

// how a fit ended
#define FIT_CONVERGED 0   // gradient (for OBJ_FROZEN: the fixed-point residual) below tolerance
#define FIT_STALLED   1   // no further progress (objective at machine precision, or the frozen-weight residual stuck) with the gradient not small
#define FIT_MAXIT     2   // iteration limit reached
#define FIT_NUMERICAL 3   // a non-finite objective or gradient was encountered

typedef struct {
	int iterations;
	long evaluations;
	double k;         // objective at the returned point
	double gmax;      // max |gradient component| at the returned point
	int status;       // FIT_*
} fit_info;
const char *fit_status_name(int status);

typedef struct {
	int objective;     // OBJ_*
	int legacy_cv;     // 1: cross-validate as the published code did: one complete row per fold whatever its weight or
	                   // group, every fold started from one fit to all the complete rows, lambda searched in [-2, 4] only
	int verbose;
	int force_mcmc;    // -g: use the Monte Carlo sampler even when exact sampling is possible
	unsigned long seed; int have_seed;
	char *groups_file;
} options;

// what the two functions handed to search_lambda need (cross_holder and kl_holder)
typedef struct {
	const dataset *data;
	const options *opt;
	const char *include;  // cross_holder: optional row mask (NULL = all rows)
	const double *start;  // cross_holder, --legacy only: the fit every fold starts from when rows have missing entries
	double best_value, best_x;        // cross_holder: the lowest value so far and where (set best_value to INFINITY before a search)
	int best_folds, best_unconverged; // ... and the number of folds there, and of fold fits that did not converge
	const double *truth;  // kl_holder: the true parameters
	fit_info last_fit;    // kl_holder: how its last fit ended
} cv_context;

// data
dataset *read_data(const char *filename);
void read_groups(dataset *d, const char *filename);
void free_data(dataset *d);
double *read_params(const char *filename, int np);
void write_params(const char *filename, const double *x, int np);
long parse_long(const char *text, const char *what, long lo, long hi);   // the whole text must be an integer in [lo, hi]
double parse_finite(const char *text, const char *what);                 // the whole text must be a finite number
void check_params(const double *theta, int np);                          // dies if the energies would not be representable
void unpack_couplings(int n, const double *theta, double *J);            // the n(n-1)/2 couplings as a symmetric n x n matrix

// model / objective / fitting
model *build_model(const dataset *d, const char *include, double log_sparsity);
void free_model(model *m);
double compute_k(model *m, const double *theta, double *grad, int objective, int set_frozen);
fit_info fit_model(model *m, double *x, int objective);
fit_info fit_rows(const dataset *d, const char *rows, double log_sparsity, double *x, int objective);
fit_info fit_staged(const dataset *d, const char *rows, double log_sparsity, double *x, int objective, int *first_stage);

// exact enumeration and sampling
double log_z(int n, const double *theta);
double full_kl(int n, const double *truth, const double *inferred);
double h_value(int n, const double *theta, const int8_t *v);
double pseudo_loglik(int n, const double *theta, const int8_t *v);
void compute_probs(int n, const double *theta, const char *filename);
void sample_exact(int n, const double *theta, int n_obs, uint64_t *out, mpf_rng *r);
void mcmc_sampler(uint64_t *config, int iter, int n, const double *theta, mpf_rng *r);

// cross-validation and the search over the regularisation strength
double cross(const dataset *d, const char *include, double log_sparsity, const double *start, const options *opt, int *counts);
double cross_holder(double log_sparsity, void *params);
double kl_holder(double log_sparsity, void *params);
double search_lambda(double (*f)(double, void *), void *params, int verbose, int extend_low, int *at_boundary);

// random numbers (rng.c)
void rng_seed(mpf_rng *r, unsigned long seed);
double rng_uniform(mpf_rng *r);                              // uniform on [0, 1)
unsigned long rng_uniform_int(mpf_rng *r, unsigned long n);  // uniform on 0 .. n-1
double rng_gaussian(mpf_rng *r, double sigma);

// entropy, and the special functions and quadrature it needs (nsb.c)
double digamma(double x);    // d/dx log Gamma(x), x > 0
double trigamma(double x);   // its derivative, x > 0
int integrate(double (*f)(double, void *), void *params, double a, double b, double epsrel, int max_intervals, double *result);
typedef struct {
	int n;         // number of observed states
	double *p;     // counts
	double norm;   // total count (set negative to have it recomputed)
	double holder;
} prob;
double entropy_nsb(prob *p);

void die(const char *fmt, ...);
void *xmalloc(size_t bytes);
void *xcalloc(size_t count, size_t size);

#endif
