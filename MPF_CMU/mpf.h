#ifndef MPF_H
#define MPF_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

// ---- limits -------------------------------------------------------------------------------
#define MAX_NODES 62        // states are packed into a uint64_t
#define MAX_BLANKS 20       // an observation with b blanks is expanded into 2^b completions
#define MAX_NN 6            // largest neighbourhood: all states within this Hamming distance of a data state
#ifndef EXACT_MAX_N
#define EXACT_MAX_N 24      // largest n for which we enumerate all 2^n states inside loops (CV, sampling)
#endif
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
	long max_sub;
	int *sub_mem, *sub_size;
	double *E, *Rv, *LQs, *Ru, *Pu;
} model;

// how a fit ended
#define FIT_CONVERGED 0   // gradient (for OBJ_FROZEN: the fixed-point residual) below tolerance
#define FIT_STALLED   1   // the objective stopped decreasing at machine precision but the gradient is not small
#define FIT_MAXIT     2   // iteration limit reached
#define FIT_NUMERICAL 3   // a non-finite objective or gradient was encountered

typedef struct {
	int iterations;
	long evaluations;
	double k;         // objective at the returned point
	double gmax;      // max |gradient component| at the returned point
	int status;       // FIT_*
	int converged;    // status == FIT_CONVERGED
} fit_info;
const char *fit_status_name(int status);

typedef struct {
	int objective;     // OBJ_*
	int legacy_cv;     // 1: score held-out rows without their weights, one row per fold (as published)
	int verbose;
	int force_mcmc;    // -g: use the Monte Carlo sampler even when exact sampling is possible
	unsigned long seed; int have_seed;
	char *groups_file;
} options;

typedef struct {
	const dataset *data;
	const char *include;  // optional row mask (NULL = all rows)
	const double *start;  // optional starting parameters for the full fit (NULL = zeros)
	const double *stage1; // log-sparsity of the complete-rows stage, if the fit is staged (NULL otherwise)
	const options *opt;
	const double *truth;  // for kl_holder
	double last_value;
	fit_info last_fit;    // set by kl_holder
} cv_context;

// data
dataset *read_data(const char *filename);
void read_groups(dataset *d, const char *filename);
void free_data(dataset *d);
double *read_params(const char *filename, int np);
void write_params(const char *filename, const double *x, int np);

// model / objective / fitting
model *build_model(const dataset *d, const char *include, double log_sparsity);
void free_model(model *m);
double compute_k(model *m, const double *theta, double *grad, int objective, int set_frozen);
fit_info fit_model(model *m, double *x, int objective);

// exact enumeration and sampling
double log_z(int n, const double *theta);
double full_kl(int n, const double *truth, const double *inferred);
double h_value(int n, const double *theta, const int8_t *v);
double pseudo_loglik(int n, const double *theta, const int8_t *v);
void compute_probs(int n, const double *theta, const char *filename);
void sample_exact(int n, const double *theta, int n_obs, uint64_t *out, gsl_rng *r);
void mcmc_sampler(uint64_t *config, int iter, int n, const double *theta, gsl_rng *r);

// cross-validation and the search over the regularisation strength
double cross(const dataset *d, const char *include, double log_sparsity, const double *start, const double *stage1, const options *opt);
double cross_holder(double log_sparsity, void *params);
double kl_holder(double log_sparsity, void *params);
double search_lambda(double (*f)(double, void *), void *params, int verbose, int extend_low, int *at_boundary);

// entropy
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
