// Cross-validation and the one-dimensional search for the regularisation strength.
#include "mpf.h"

// Fit the rows selected by `rows` (NULL = all of them) at this log-sparsity. x holds the starting point
// on entry and the fit on exit.
fit_info fit_rows(const dataset *d, const char *rows, double log_sparsity, double *x, int objective) {
	model *mo=build_model(d, rows, log_sparsity);
	fit_info info=fit_model(mo, x, objective);
	free_model(mo);
	return info;
}

// The two-stage fit used when rows have missing entries. The objective is then not convex and the
// starting point can decide which local solution is found, so: fit the complete rows starting from x (a
// strictly convex problem: x affects its cost, not its answer), then all the rows starting from that
// solution, both at the same log-sparsity. If none of the rows has a missing entry this is a single fit
// from x; if none is complete, a single fit from zero. *first_stage receives how the complete-rows fit
// ended (FIT_CONVERGED if there was none).
fit_info fit_staged(const dataset *d, const char *rows, double log_sparsity, double *x, int objective, int *first_stage) {
	int m=d->m, np=d->n*(d->n+1)/2, i, n_complete=0, n_partial=0;
	char *complete;

	*first_stage=FIT_CONVERGED;
	for(i=0;i<m;i++) if ((rows == NULL || rows[i]) && d->w[i] > 0) { if (d->n_blanks[i] == 0) n_complete++; else n_partial++; }
	if (n_partial > 0 && n_complete > 0) {
		complete=(char *)xmalloc(m);
		for(i=0;i<m;i++) complete[i]=(rows == NULL || rows[i]) && d->n_blanks[i] == 0;
		*first_stage=fit_rows(d, complete, log_sparsity, x, objective).status;
		free(complete);
	} else if (n_partial > 0) {
		for(i=0;i<np;i++) x[i]=0.0;
	}
	return fit_rows(d, rows, log_sparsity, x, objective);
}

// Leave-one-out cross-validation: returns the mean held-out log-likelihood at this log-sparsity, or NaN
// if a fit failed numerically (so that the search treats this lambda as invalid). counts, if not NULL,
// receives the number of folds and the number whose fits did not converge.
//
// Only rows without missing entries are held out (as in the published code).
//
// Default:  rows that share a group id (if a groups file was given) are held out together, and each
//           held-out row is scored with its weight. Every fold is fit, by fit_staged, from its own
//           training rows and nothing else. (The folds are given a common warm start, the fit to all the
//           complete rows at this lambda; it is used only for the strictly convex part of each fold's
//           fit, where it changes the cost and not the answer.) Earlier versions started every fold
//           from one fit that had seen all the rows, and then from a first stage whose lambda had been
//           chosen using all the complete rows; both let a held-out row influence its own fold.
// Legacy:   every complete row is its own fold and counts equally, whatever its weight or group. If rows
//           have missing entries every fold is a single fit from `start`, the fit to all the complete
//           rows that the caller made at a lambda of its own, as in the published code (it has seen the
//           held-out row). This is what reproduces the published fits.
//
// For n <= EXACT_MAX_N the held-out score is the exact log-likelihood. For larger n the partition
// function cannot be enumerated and the log pseudo-likelihood of the held-out row is used instead
// (a different model-selection criterion; it agreed with exact CV in a small test at n=20).
double cross(const dataset *d, const char *include, double log_sparsity, const double *start, const options *opt, int *counts) {
	int m=d->m, n=d->n, np=n*(n+1)/2, i, j, n_folds=0, in, exact=(n <= EXACT_MAX_N), n_used=0, has_partial=0;
	int *fold_of=(int *)xmalloc(m*sizeof(int)); // fold index of each row, or -1 if the row is never held out
	double *warm=(double *)xcalloc(np, sizeof(double));
	double total=0, total_w=0, *fold_total, *fold_w;
	int *fold_status, n_unconverged=0, n_failed=0;

	for(i=0;i<m;i++) {
		fold_of[i]=-1;
		if ((include == NULL || include[i]) && d->w[i] > 0) {
			n_used++;
			if (d->n_blanks[i] > 0) has_partial=1;
		}
	}

	if (opt->legacy_cv || d->group == NULL) {
		for(i=0;i<m;i++) {
			if (include != NULL && !include[i]) continue;
			if (d->n_blanks[i] > 0 || d->w[i] == 0) continue;
			fold_of[i]=n_folds++;
		}
	} else {
		for(i=0;i<m;i++) {
			int ok=1, first=1;
			if (include != NULL && !include[i]) continue;
			if (d->w[i] == 0) continue;
			for(j=0;j<m;j++) {
				if (d->group[j] != d->group[i]) continue;
				if (include != NULL && !include[j]) continue;
				if (d->n_blanks[j] > 0) ok=0;       // only fully-observed groups are held out
				if (j < i && d->w[j] > 0) first=0;  // this group already has a fold
			}
			if (!ok) continue;
			if (first) fold_of[i]=n_folds++;
			else for(j=0;j<i;j++) if (d->group[j] == d->group[i] && fold_of[j] >= 0) { fold_of[i]=fold_of[j]; break; }
		}
	}
	if (n_folds == 0) die("cross-validation needs at least one row without missing entries");
	if (n_used < 2) die("cross-validation needs at least two rows");

	if (counts != NULL) { counts[0]=n_folds; counts[1]=0; }
	if (opt->legacy_cv && has_partial && start != NULL) {
		for(i=0;i<np;i++) warm[i]=start[i];
	} else {
		char *complete=(char *)xmalloc(m);
		int status;
		for(i=0;i<m;i++) complete[i]=(include == NULL || include[i]) && d->n_blanks[i] == 0;
		status=fit_rows(d, complete, log_sparsity, warm, opt->objective).status;
		free(complete);
		if (status == FIT_NUMERICAL) { free(fold_of); free(warm); return NAN; }
	}

	// per-fold results are stored and summed in order afterwards, so the answer does not depend on the
	// number of threads or on how the folds were scheduled
	fold_total=(double *)xcalloc(n_folds, sizeof(double));
	fold_w=(double *)xcalloc(n_folds, sizeof(double));
	fold_status=(int *)xcalloc(n_folds, sizeof(int));
#pragma omp parallel for schedule(dynamic)
	for(in=0;in<n_folds;in++) {
		char *mask=(char *)xmalloc(m);
		double *x=(double *)xmalloc(np*sizeof(double)), lz=0, ll, wt;
		fit_info fi;
		int a, left=0, first_stage=FIT_CONVERGED;

		for(a=0;a<m;a++) {
			mask[a]=(include == NULL || include[a]) && (fold_of[a] != in);
			if (mask[a] && d->w[a] > 0) left++;
		}
		fold_status[in]=FIT_CONVERGED;
		if (left > 0) {
			for(a=0;a<np;a++) x[a]=warm[a];
			if (opt->legacy_cv) fi=fit_rows(d, mask, log_sparsity, x, opt->objective);
			else fi=fit_staged(d, mask, log_sparsity, x, opt->objective, &first_stage);
			if (first_stage != FIT_CONVERGED) fold_status[in]=first_stage;
			if (fi.status != FIT_CONVERGED && fold_status[in] != FIT_NUMERICAL) fold_status[in]=fi.status;
			if (fold_status[in] != FIT_NUMERICAL) {
				if (exact) lz=log_z(n, x);
				for(a=0;a<m;a++) {
					if (fold_of[a] != in) continue;
					ll=exact ? h_value(n, x, d->v+(size_t)a*n)-lz : pseudo_loglik(n, x, d->v+(size_t)a*n);
					wt=opt->legacy_cv ? 1.0 : d->w[a];
					fold_total[in] += wt*ll;
					fold_w[in] += wt;
				}
			}
		}
		free(mask); free(x);
	}
	for(in=0;in<n_folds;in++) {
		total += fold_total[in]; total_w += fold_w[in];
		if (fold_status[in] == FIT_NUMERICAL) n_failed++;
		else if (fold_status[in] != FIT_CONVERGED) n_unconverged++;
	}
	free(fold_of); free(warm); free(fold_total); free(fold_w); free(fold_status);
	if (counts != NULL) counts[1]=n_unconverged+n_failed;
	if (n_failed > 0) {
		fprintf(stderr, "mpf: warning: at log-sparsity %g, %d of %d cross-validation fits failed numerically; this value is skipped\n", log_sparsity, n_failed, n_folds);
		return NAN;
	}
	if (n_unconverged > 0) fprintf(stderr, "mpf: warning: at log-sparsity %g, %d of %d cross-validation fits did not converge\n", log_sparsity, n_unconverged, n_folds);
	if (!(total_w > 0)) die("cross-validation could not score any held-out rows");
	if (opt->verbose) printf("For log-sparsity=%lf, Log-l of held-out data is: %lf\n", log_sparsity, total/total_w);
	return total/total_w;
}

// Negative held-out log-likelihood, for minimisation. Remembers the lowest value seen and how the folds
// ended there, which is what search_lambda will return.
double cross_holder(double log_sparsity, void *params) {
	cv_context *cv=(cv_context *)params;
	int counts[2]={0, 0};
	double v=-cross(cv->data, cv->include, log_sparsity, cv->start, cv->opt, counts);
	if (v < cv->best_value) { cv->best_value=v; cv->best_x=log_sparsity; cv->best_folds=counts[0]; cv->best_unconverged=counts[1]; }
	return v;
}

// KL divergence of the fit to all rows at this log-sparsity (a single fit from zero) from the known true
// parameters; NaN if the fit failed numerically
double kl_holder(double log_sparsity, void *params) {
	cv_context *cv=(cv_context *)params;
	int n=cv->data->n;
	double *x=(double *)xcalloc(n*(n+1)/2, sizeof(double)), v;

	cv->last_fit=fit_rows(cv->data, NULL, log_sparsity, x, cv->opt->objective);
	if (cv->last_fit.status != FIT_CONVERGED && cv->opt->verbose) fprintf(stderr, "mpf: warning: fit at log-sparsity %g %s (max |gradient| %.3g)\n", log_sparsity, fit_status_name(cv->last_fit.status), cv->last_fit.gmax);
	v=(cv->last_fit.status == FIT_NUMERICAL) ? NAN : full_kl(n, cv->truth, x);
	free(x);
	return v;
}

// Minimise f(log_sparsity).
//
// The published code handed GSL's golden-section minimiser the bracket (-2, 1, 4) and ignored the error
// it returns when f(1) is not below both end points (roughly a quarter of the time in our tests).
// Here a coarse grid on [-2, 4] is evaluated first, so that the bracket handed to the golden-section
// search is a real one. If extend_low is set and the best grid point is the lowest one, the grid is
// extended downwards (to -8 at most), so that the penalty is not held above a floor when the data would
// prefer less regularisation. The result is the best point found around the best grid point; like any
// such search it is not guaranteed to be the global minimum of a curve with several minima between
// grid points.
//
// Values that are not finite are treated as invalid. Returns NaN if f is invalid everywhere.
// *at_boundary is set if the result sits on the edge of the range that was searched.
#define SEARCH_LO -2.0
#define SEARCH_HI 4.0
#define SEARCH_FLOOR -8.0
#define SEARCH_STEP 0.5
#define SEARCH_TOL 1e-3

static double valid(double v) { return isfinite(v) ? v : INFINITY; }

double search_lambda(double (*f)(double, void *), void *params, int verbose, int extend_low, int *at_boundary) {
	const double gr=0.6180339887498949;
	int n_grid=(int)floor((SEARCH_HI-SEARCH_LO)/SEARCH_STEP+0.5)+1, i, best=0;
	double *val=(double *)xmalloc(n_grid*sizeof(double));
	double a, b, c, d2, fc, fd, xbest, fbest, lo=SEARCH_LO, vlow, below;

	for(i=0;i<n_grid;i++) {
		val[i]=valid(f(SEARCH_LO+i*SEARCH_STEP, params));
		if (verbose) printf("  log-sparsity %8.4f   objective %.6f\n", SEARCH_LO+i*SEARCH_STEP, val[i]);
		if (val[i] < val[best]) best=i;
	}
	xbest=SEARCH_LO+best*SEARCH_STEP; fbest=val[best];
	a=SEARCH_LO+(best > 0 ? best-1 : 0)*SEARCH_STEP;
	b=SEARCH_LO+(best < n_grid-1 ? best+1 : n_grid-1)*SEARCH_STEP;
	vlow=val[0];
	free(val);
	if (!isfinite(fbest)) {
		if (at_boundary != NULL) *at_boundary=1;
		return NAN;
	}

	if (extend_low && best == 0) {
		// walk downwards while the objective keeps improving
		while(lo-SEARCH_STEP >= SEARCH_FLOOR-1e-9) {
			below=valid(f(lo-SEARCH_STEP, params));
			if (verbose) printf("  log-sparsity %8.4f   objective %.6f\n", lo-SEARCH_STEP, below);
			if (!(below < vlow)) { a=lo-SEARCH_STEP; break; } // lo is a grid minimum with neighbours on both sides
			b=lo; lo -= SEARCH_STEP; vlow=below; xbest=lo; fbest=below; a=lo;
		}
	}

	c=b-gr*(b-a); d2=a+gr*(b-a);
	fc=valid(f(c, params)); fd=valid(f(d2, params));
	while((b-a) > SEARCH_TOL) {
		if (verbose) printf("  [%.5f, %.5f]\n", a, b);
		if (fc < fd) {
			if (fc < fbest) { fbest=fc; xbest=c; }
			b=d2; d2=c; fd=fc;
			c=b-gr*(b-a); fc=valid(f(c, params));
		} else {
			if (fd < fbest) { fbest=fd; xbest=d2; }
			a=c; c=d2; fc=fd;
			d2=a+gr*(b-a); fd=valid(f(d2, params));
		}
	}
	if (fc < fbest) { fbest=fc; xbest=c; }
	if (fd < fbest) { fbest=fd; xbest=d2; }

	if (at_boundary != NULL) *at_boundary=(xbest < (extend_low ? SEARCH_FLOOR : SEARCH_LO)+10*SEARCH_TOL) || (xbest > SEARCH_HI-10*SEARCH_TOL);
	return xbest;
}
