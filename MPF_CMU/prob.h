#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>

typedef struct {
	int n;	// number of states
	double *p; // list of probabilities
	double norm; // overall normalization
	double holder; // useful holder
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

int d_compare(const void *elem1, const void *elem2);

gsl_rng *rng();

void norm_prob(prob *a);
void norm_jprob(j_prob *a);

prob *new_prob(int n);
j_prob *new_jprob(int n, int m);
void delete_prob(prob *a);
void delete_jprob(j_prob *a);

void print_prob(prob *a);
void print_jprob(j_prob *a);

void draw(prob *a, prob *samp, int k, gsl_rng *r);
void draw_joint(j_prob *pij, j_prob *samp, int k, gsl_rng *r);

void sample_prob(prob *a, double k, gsl_rng *r);
void sample_prob_dirichlet(prob *a, double k, gsl_rng *r);
void sample_prob_simple(prob *a, double k, gsl_rng *r);
void sample_jprob(j_prob *a, double k, gsl_rng *r);
struct nsb_params {
         double n, choice;
};
double cumulative_distribution(double beta, void *params);

prob *mixture(prob *a, prob *b, double alpha);

double entropy(prob *a);
double entropy_ww(prob *a);

double jsd(prob *a, prob *b, double alpha);

double mi(j_prob *a);
double mi_ww(j_prob *pij, double beta);

double *entropy_bs(prob *a, long n_samp, gsl_rng *r);
double *jsd_bs(prob *a, prob *b, double alpha, long n_samp, gsl_rng *r);
double *mi_bs(j_prob *a, long n_samp, gsl_rng *r);

void do_entropy(int n_states, int n_fact, char *filename);
void do_entropy_nsb(int n_states, int n_fact, char *filename);

void do_mi(int n_states, int m_states, int n_fact, char *filename);
void do_mi_nsb(int n_states, int m_states, int n_fact, char *filename);

void do_jsd(int n_states, double alpha, int n_fact, char *filename);

double entropy_nsb(prob *p);
double mi_nsb(j_prob *p);

prob* import_prob(char* file_name);
prob* prob_from_array(double* in_array, int n);
j_prob* jprob_from_array(double* j_in_array, int n1, int n2);
void return_array(double* array, int n);
void print_bs(double* bs, prob* a_prob);
