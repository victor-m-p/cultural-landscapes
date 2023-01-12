#include <omp.h>

#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h> // we use this to get process ID and help randomness

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_statistics_double.h>

#include <gsl/gsl_sf.h>
//#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>

#define BIG 1e6
#define BIGI 1e-6
#define EPSILON 1e-16

typedef struct {
	char *filename;
	double *big_list_true;
	double kl_true;
	double *best_fit;
	int nn;
} cross_val;

typedef struct {
	unsigned long int config;
	int *data_prox; // points to a location in the data prox vector
} near_struct;

typedef struct {
	int *config_base; // base configuration (-1, +1, 0)
	double mult; // multiplicity of base config (including blanks)


	int n_blanks; // number of blanks
	int n_config; // 2^n_blanks
	int *blanks; // location of blanks

	// for the below, this is filled in *even* if there are no blanks
	
	unsigned long int *config; // all the configurations filled in and represented as an unsigned long for speed; number of configs is (1 << n_blanks) -- this is written once, then 
	double *mult_sim; // list of simulated conditional multiplicity configurations
	double **mult_sim_pointer; // pointers to the multi_sim in the all list
	int **prox;
} sample;

typedef struct {
	int m;	// number of samples
	int uniq; // number of unique samples
	int n; 	// number of nodes
	int max_config; // max number of blank configs 2^max_blanks
	int max_blanks; // max number of blanks
	
	int near_uniq; // total number of neighbours -- this will be constant for everyone
	int n_prox; // number of neighbours of any point -- this will be constant for everyone
	unsigned long int *near; // the list of neighbours, represented as unsigned long int for speed.
	near_struct **near_set;
		
	sample **obs_raw; // sample that is read in; this will be sorted and duplicate-filtered
	sample **obs; // the observations that we actually work with
	
	double log_sparsity;
	double sparsity;
	
	int n_all;
	
	int n_params;	
	double *big_list;
	double *old_list;
	
	double *best_fit;
		
	double *nei;
	double *ei;
	
	double k; 
	double *dk;
	
	int **ij;
	int h_offset;
	
	gsl_rng *r;
} all;

gsl_rng *rng();

all *new_data();
void delete_data(all *data);

void read_data(char *filename, all *data);
void process_obs_raw(all *data);
void init_params(all *data);
void create_near(all *data, int n_step);
void update_mult_sim(all *data);

void compute_k_general(all *data, int do_derivs);
void simple_minimizer(all *data);

void print_vec(unsigned long a);
unsigned long int convert(int *list);

void mcmc_sampler(unsigned long int *config, int iter, all *data);
double full_kl(all *data, double *inferred, double *truth);
double log_l(all *data, unsigned long int config, double *inferred, int do_approx);
void compute_probs(int n, double *big_list, char *filename);

double cross(char *filename, double log_sparsity, int nn, double *best_fit);
double minimize_kl(cross_val *cv, int fast_version);
void update_sparsity(all *data);

double kl_holder(double log_sparsity, void *params);
double minimize_kl_true(cross_val *cv);

typedef struct {
	int n;	// number of states
	double *p; // list of probabilities
	double norm; // overall normalization
	double holder; // useful holder
	double k; // total number of bins
	double kappa; // Eq 6 in NSB paper
	gsl_ran_discrete_t *pre_proc; // the pre-processor for fast number generation
} prob;

typedef struct {
	int n;	// number of states for process I
	int m; // number of states for process II
	
	double **p; // list of probabilities
	double holder; // useful holder
	double norm; // overall normalization
	gsl_ran_discrete_t *pre_proc; // the pre-processor for fast number generation
} j_prob;
double entropy_nsb(prob *p);

