#include "mpf.h"
#include <errno.h>

static const char *usage_text=
"usage:\n"
"  mpf -l [filename] [logsparsity] [NN]           load data, fit at a fixed log10(lambda)\n"
"  mpf -c [filename] [NN]                         load data, pick lambda by cross-validation, fit\n"
"  mpf -g [filename] [n_nodes] [n_obs] [beta]     generate parameters and data (_params.dat, _data.dat)\n"
"  mpf -t [filename] [paramfile] [NN]             cross-validated fit, then KL divergence from the truth\n"
"  mpf -o [filename_prefix] [NN]                  compare the cross-validated lambda with the best possible one\n"
"  mpf -k [filename] [params_truth] [params_fit]  KL divergence of a fit from the truth\n"
"  mpf -z [paramfile] [n_nodes]                   write the probability of every configuration\n"
"\n"
"options (anywhere on the command line):\n"
"  --marginal      missing data: MPF on the marginal over the observed nodes (default)\n"
"  --legacy        reproduce the published fits: the original Partial-MPF objective; cross-validation\n"
"                  that scores every held-out row equally and starts every fold from one fit to all\n"
"                  the complete rows, at a lambda of its own; a lambda search confined to [-2, 4]\n"
"  --frozen        missing data: completion weights frozen, re-estimated, iterated to a fixed point (slower)\n"
"  --legacy-cv     only the cross-validation part of --legacy\n"
"  --groups FILE   one integer per data row; rows sharing an id are held out together\n"
"  --start FILE    (-l) starting parameters\n"
"  --seed N        (-g) seed for the random number generator\n"
"  --mcmc          (-g) use the Monte Carlo sampler even where exact sampling is possible\n"
"  --verbose       print every cross-validation evaluation\n"
"\n"
"NN is the neighbourhood: flows go from each data state to every state within Hamming distance NN\n"
"(1 = single flips, as in the paper's main results; 2 adds all pairs of flips, and so on). The cost per\n"
"state grows as the number of such neighbours: n, n + n(n-1)/2, ...\n";

static double now(void) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec+1e-9*t.tv_nsec;
}

static void usage(void) {
	fprintf(stderr, "%s", usage_text);
	exit(2);
}

// NN: flows go from each data state to every state within Hamming distance NN (1 = single flips)
static void set_nn(dataset *d, const char *arg) {
	int nn=(int)parse_long(arg, "NN", 1, MAX_NN);
	if (nn > d->n) die("NN (%d) cannot exceed the number of nodes (%d)", nn, d->n);
	d->nn=nn;
}

static void print_params(const double *x, int np) {
	int i;
	printf("params=[");
	for(i=0;i<np;i++) printf(i < np-1 ? "%.10e, " : "%.10e]\n", x[i]);
}

// A fit that failed numerically is an error; one that merely did not converge is reported and returned.
static void check_fit(const fit_info *info) {
	if (info->status == FIT_NUMERICAL) die("the fit failed: a non-finite objective or gradient was encountered "
		"(parameters or starting values too large for the flows to be representable)");
	if (info->status != FIT_CONVERGED) fprintf(stderr, "mpf: warning: the fit %s (max |gradient| = %.3g after %d iterations); "
		"without regularisation this is expected when the data are separable\n", fit_status_name(info->status), info->gmax, info->iterations);
}

static char *complete_mask(const dataset *d, int *n_partial) {
	char *mask=(char *)xmalloc(d->m);
	int i;
	*n_partial=0;
	for(i=0;i<d->m;i++) {
		mask[i]=(d->n_blanks[i] == 0);
		if (!mask[i]) (*n_partial)++;
	}
	return mask;
}

// Cross-validated fit; returns the chosen log-sparsity, and x receives the final fit.
//
// If some rows have missing entries the objective is not convex, and a fit is made in two stages: the
// complete rows first (strictly convex), then all rows starting from that solution.
//   default:   both stages use the same lambda (fit_staged). Cross-validation scores a lambda by making
//              that fit from each fold's training rows, and the final fit makes it from all rows at the
//              lambda chosen.
//   --legacy:  the published protocol. A lambda for the first stage is chosen by cross-validation on the
//              complete rows, the complete rows are fit at it, and that one solution is the starting
//              point of every later fit, in the folds and for the final fit.
static double cv_fit(const dataset *d, const options *opt, double *x, fit_info *info) {
	int np=d->n*(d->n+1)/2, n_partial, at_boundary, i, first_stage=FIT_CONVERGED;
	char *mask=complete_mask(d, &n_partial);
	double best, stage1, *start=NULL;
	cv_context cv;

	memset(&cv, 0, sizeof(cv));
	cv.data=d; cv.opt=opt;
	if (opt->legacy_cv && n_partial > 0) {
		cv.include=mask;
		stage1=search_lambda(cross_holder, &cv, opt->verbose, 0, &at_boundary);
		if (isnan(stage1)) die("cross-validation on the complete rows failed at every lambda");
		printf("Found a best sparsity without NaNs (%lf)\n", stage1);
		start=(double *)xcalloc(np, sizeof(double));
		*info=fit_rows(d, mask, stage1, start, opt->objective);
		check_fit(info);
		cv.include=NULL; cv.start=start;
	}
	cv.best_value=INFINITY;
	best=search_lambda(cross_holder, &cv, opt->verbose, !opt->legacy_cv, &at_boundary);
	if (isnan(best)) die("cross-validation failed at every lambda");
	if (cv.best_x != best) die("internal error: the search returned a value it did not score best");
	if (at_boundary) fprintf(stderr, "mpf: warning: the best log-sparsity (%g) is at the edge of the range searched\n", best);
	// (warnings printed during the search may concern other values of lambda)
	printf("Cross-validation at the chosen log-sparsity: %d folds, %d with a fit that did not converge\n", cv.best_folds, cv.best_unconverged);

	for(i=0;i<np;i++) x[i]=start ? start[i] : 0.0;
	if (opt->legacy_cv) *info=fit_rows(d, NULL, best, x, opt->objective);
	else *info=fit_staged(d, NULL, best, x, opt->objective, &first_stage);
	if (first_stage != FIT_CONVERGED) fprintf(stderr, "mpf: warning: the first stage of the fit (complete rows) %s\n", fit_status_name(first_stage));
	free(mask); free(start);
	return best;
}

static void make_rng(const options *opt, mpf_rng *r) {
	unsigned long seed;
	FILE *f;
	if (opt->have_seed) {
		seed=opt->seed;
	} else {
		f=fopen("/dev/urandom", "rb");
		if (f == NULL || fread(&seed, sizeof(unsigned long), 1, f) != 1) {
			fprintf(stderr, "mpf: warning: could not read /dev/urandom; seeding from the clock and process id\n");
			seed=(unsigned long)time(NULL)^((unsigned long)getpid() << 16);
		}
		if (f != NULL) fclose(f);
	}
	rng_seed(r, seed);
}

int main(int argc, char *argv[]) {
	options opt;
	char **arg=(char **)xmalloc((argc+1)*sizeof(char *)), *start_file=NULL, filename_sav[4096], mode;
	int nargs=0, i, j, n, np;
	double t0=now();
	dataset *d;
	fit_info info;

	opt.objective=OBJ_MARGINAL; opt.legacy_cv=0; opt.verbose=0; opt.force_mcmc=0; opt.have_seed=0; opt.seed=0; opt.groups_file=NULL;

	// separate options from positional arguments (negative numbers such as "-100" are positional)
	for(i=1;i<argc;i++) {
		if (strncmp(argv[i], "--", 2) != 0) { arg[nargs++]=argv[i]; continue; }
		if (strcmp(argv[i], "--marginal") == 0) opt.objective=OBJ_MARGINAL;
		else if (strcmp(argv[i], "--legacy") == 0) { opt.objective=OBJ_LEGACY; opt.legacy_cv=1; }
		else if (strcmp(argv[i], "--frozen") == 0) opt.objective=OBJ_FROZEN;
		else if (strcmp(argv[i], "--legacy-cv") == 0) opt.legacy_cv=1;
		else if (strcmp(argv[i], "--verbose") == 0) opt.verbose=1;
		else if (strcmp(argv[i], "--mcmc") == 0) opt.force_mcmc=1;
		else if (strcmp(argv[i], "--groups") == 0 && i+1 < argc) opt.groups_file=argv[++i];
		else if (strcmp(argv[i], "--start") == 0 && i+1 < argc) start_file=argv[++i];
		else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) {
			char *end, *text=argv[++i];
			errno=0;
			opt.seed=strtoul(text, &end, 10);
			if (text[0] < '0' || text[0] > '9' || *end != '\0' || errno != 0) die("--seed needs a non-negative integer (got '%s')", text);
			opt.have_seed=1;
		}
		else die("unknown or incomplete option '%s' (run mpf with no arguments for help)", argv[i]);
	}

	if (nargs == 0 || arg[0][0] != '-' || strlen(arg[0]) != 2) usage();
	mode=arg[0][1];

	switch(mode) {

	case 'l': { // fit at a fixed lambda
		model *mo;
		double *x, *st;
		if (nargs != 4) usage();
		d=read_data(arg[1]);
		set_nn(d, arg[3]);
		np=d->n*(d->n+1)/2;
		mo=build_model(d, NULL, parse_finite(arg[2], "logsparsity"));
		printf("%i unique observations; %ld rows after expanding missing entries; %.0f neighbours per complete state\n", mo->D, mo->R, mo->n_prox);
		x=(double *)xcalloc(np, sizeof(double));
		if (start_file != NULL) { st=read_params(start_file, np); for(i=0;i<np;i++) x[i]=st[i]; free(st); }
		info=fit_model(mo, x, opt.objective);
		check_fit(&info);
		printf("K=%.10f after %d iterations (%ld evaluations); %s, max |gradient| = %.3g\n", info.k, info.iterations, info.evaluations, fit_status_name(info.status), info.gmax);
		print_params(x, np);
		snprintf(filename_sav, sizeof(filename_sav), "%s_params.dat", arg[1]);
		write_params(filename_sav, x, np);
		free(x); free_model(mo); free_data(d);
		break;
	}

	case 'c': { // cross-validated fit
		double *x, best;
		if (nargs != 3) usage();
		d=read_data(arg[1]);
		set_nn(d, arg[2]);
		if (opt.groups_file != NULL) read_groups(d, opt.groups_file);
		np=d->n*(d->n+1)/2;
		x=(double *)xcalloc(np, sizeof(double));
		best=cv_fit(d, &opt, x, &info);
		check_fit(&info);
		printf("Best log_sparsity: %lf\n", best);
		printf("K=%.10f\n", info.k);
		print_params(x, np);
		snprintf(filename_sav, sizeof(filename_sav), "%s_params.dat", arg[1]);
		write_params(filename_sav, x, np);
		free(x); free_data(d);
		break;
	}

	case 'g': { // generate parameters and data
		int n_obs;
		double beta, *theta;
		uint64_t *draws;
		mpf_rng rng, *r=&rng;
		FILE *fp;
		if (nargs != 5) usage();
		n=(int)parse_long(arg[2], "n_nodes", 2, MAX_NODES); n_obs=(int)parse_long(arg[3], "n_obs", 1, 100000000L); beta=parse_finite(arg[4], "beta");
		if (beta < 0) die("beta must be non-negative");
		np=n*(n+1)/2;
		make_rng(&opt, r);
		theta=(double *)xmalloc(np*sizeof(double));
		for(i=0;i<np;i++) theta[i]=rng_gaussian(r, 1.0)*beta;
		check_params(theta, np);
		draws=(uint64_t *)xmalloc(n_obs*sizeof(uint64_t));
		if (n <= EXACT_MAX_N && !opt.force_mcmc) {
			sample_exact(n, theta, n_obs, draws, r);
		} else {
			if (n > EXACT_MAX_N) fprintf(stderr, "mpf: warning: n > %d, so data are drawn with the Monte Carlo sampler, "
				"which does not reach equilibrium for strong couplings\n", EXACT_MAX_N);
			for(j=0;j<n_obs;j++) {
				draws[j]=rng_uniform_int(r, 1UL << (n < 31 ? n : 31));
				if (n >= 31) draws[j] |= ((uint64_t)rng_uniform_int(r, 1UL << (n-31)) << 31);
				mcmc_sampler(&draws[j], 1000, n, theta, r);
			}
		}
		snprintf(filename_sav, sizeof(filename_sav), "%s_data.dat", arg[1]);
		fp=fopen(filename_sav, "w");
		if (fp == NULL) die("cannot write '%s'", filename_sav);
		fprintf(fp, "%i\n%i\n", n_obs, n);
		for(j=0;j<n_obs;j++) {
			for(i=0;i<n;i++) fputc(((draws[j] >> i) & 1) ? '1' : '0', fp);
			fprintf(fp, " 1.0\n");
		}
		if (ferror(fp) || fclose(fp) != 0) die("error writing '%s'", filename_sav);
		snprintf(filename_sav, sizeof(filename_sav), "%s_params.dat", arg[1]);
		write_params(filename_sav, theta, np);
		free(theta); free(draws);
		break;
	}

	case 't': { // cross-validated fit, then KL from the truth
		double *x, *truth, best;
		if (nargs != 4) usage();
		d=read_data(arg[1]);
		set_nn(d, arg[3]);
		if (opt.groups_file != NULL) read_groups(d, opt.groups_file);
		n=d->n; np=n*(n+1)/2;
		if (n > ENUM_MAX_N) die("-t needs n <= %d to compute the KL divergence", ENUM_MAX_N);
		truth=read_params(arg[2], np);
		x=(double *)xcalloc(np, sizeof(double));
		best=cv_fit(d, &opt, x, &info);
		check_fit(&info);
		printf("Best log_sparsity: %lf\n", best);
		printf("KL divergence from truth: %.10f\n", full_kl(n, truth, x));
		free(x); free(truth); free_data(d);
		break;
	}

	case 'o': { // optimal lambda (using the truth) against the cross-validated one
		double *x, *truth, best, kl_true, kl_true_sp, kl_cv, kl_unreg, ent=0, nsb, tot=0;
		cv_context cv;
		prob p;
		model *mo;
		if (nargs != 3) usage();
		snprintf(filename_sav, sizeof(filename_sav), "%s_data.dat", arg[1]);
		d=read_data(filename_sav);
		set_nn(d, arg[2]);
		if (opt.groups_file != NULL) read_groups(d, opt.groups_file);
		n=d->n; np=n*(n+1)/2;
		if (n > ENUM_MAX_N) die("-o needs n <= %d to compute the KL divergence", ENUM_MAX_N);
		snprintf(filename_sav, sizeof(filename_sav), "%s_params.dat", arg[1]);
		truth=read_params(filename_sav, np);

		// entropy of the sample of complete rows
		{
			int n_partial;
			char *mask=complete_mask(d, &n_partial);
			mo=build_model(d, mask, 0.0);
			p.n=mo->D; p.norm=-1; p.p=(double *)xmalloc(mo->D*sizeof(double));
			for(i=0;i<mo->D;i++) { p.p[i]=mo->mult[i]; tot += mo->mult[i]; }
			for(i=0;i<mo->D;i++) ent -= (mo->mult[i]/tot)*log(mo->mult[i]/tot)/log(2);
			free_model(mo); free(mask);
		}
		nsb=entropy_nsb(&p);
		printf("NSB entropy of data: %lf\n", nsb);
		printf("Naieve entropy of data: %lf\n", ent);

		memset(&cv, 0, sizeof(cv));
		cv.data=d; cv.opt=&opt; cv.truth=truth;
		kl_true_sp=search_lambda(kl_holder, &cv, opt.verbose, !opt.legacy_cv, NULL);
		if (isnan(kl_true_sp)) die("every fit failed while searching for the best lambda");
		kl_true=kl_holder(kl_true_sp, &cv);
		printf("Best log_sparsity: %lf\n", kl_true_sp);
		printf("KL at best log_sparsity: %lf\n", kl_true);

		x=(double *)xcalloc(np, sizeof(double));
		best=cv_fit(d, &opt, x, &info);
		check_fit(&info);
		kl_cv=full_kl(n, truth, x);
		printf("Best log_sparsity CV: %lf\n", best);
		printf("KL at CV'd log_sparsity: %lf\n", kl_cv);

		kl_unreg=kl_holder(-100, &cv); // no regularisation ("standard MPF")
		printf("Unregularised fit: %s (max |gradient| %.3g after %d iterations)\n", fit_status_name(cv.last_fit.status), cv.last_fit.gmax, cv.last_fit.iterations);
		printf("val=[%.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f]\n", ent, nsb, kl_true, kl_true_sp, kl_cv, best, kl_unreg, -1000.0);
		free(x); free(truth); free(p.p); free_data(d);
		break;
	}

	case 'k': { // KL(truth || inferred)
		double *truth, *inferred;
		if (nargs != 4) usage();
		d=read_data(arg[1]);
		n=d->n; np=n*(n+1)/2;
		if (n > ENUM_MAX_N) die("-k needs n <= %d", ENUM_MAX_N);
		truth=read_params(arg[2], np);
		inferred=read_params(arg[3], np);
		printf("KL: %.10f\n", full_kl(n, truth, inferred));
		free(truth); free(inferred); free_data(d);
		break;
	}

	case 'z': { // probabilities of all configurations
		double *theta;
		if (nargs != 3) usage();
		n=(int)parse_long(arg[2], "n_nodes (for -z)", 2, ENUM_MAX_N);
		theta=read_params(arg[1], n*(n+1)/2);
		snprintf(filename_sav, sizeof(filename_sav), "%s_probs.dat", arg[1]);
		compute_probs(n, theta, filename_sav);
		free(theta);
		break;
	}

	default:
		usage();
	}

	printf("Wall time: %.3f seconds.\n", now()-t0);
	free(arg);
	return 0;
}
