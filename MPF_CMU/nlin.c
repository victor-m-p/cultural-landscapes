// Cross-validation and the one-dimensional search for the regularisation strength.
#include "mpf.h"

// Leave-one-out cross-validation: returns the mean held-out log-likelihood at this log-sparsity, or NaN
// if a fit failed numerically (so that the search treats this lambda as invalid).
//
// Only rows without missing entries are held out (as in the published code).
//
// Default:  rows that share a group id (if a groups file was given) are held out together, and each
//           held-out row is scored with its weight. Every fold is fit from information in its own
//           training rows only:
//             - if no included row has missing entries the problem is strictly convex, the optimum is
//               unique, and the fold is warm-started from the full-data fit purely for speed;
//             - otherwise the objective is not convex and the starting point can decide which local
//               solution is found, so each fold repeats the staged procedure on its training rows:
//               fit the complete training rows at the stage-1 lambda (convex), then all training rows
//               at this lambda starting from that solution.
// Legacy:   every complete row is its own fold and counts equally, whatever its weight or group. With
//           missing entries every fold starts from the complete-rows fit of all the data (which has
//           seen the held-out row), as in the published code. This is what reproduces the published fits.
//
// For n <= EXACT_MAX_N the held-out score is the exact log-likelihood. For larger n the partition
// function cannot be enumerated and the log pseudo-likelihood of the held-out row is used instead
// (a different model-selection criterion; it agreed with exact CV in a small test at n=20).
double cross(const dataset *d, const char *include, double log_sparsity, const double *start, const double *stage1, const options *opt) {
	int m=d->m, n=d->n, np=n*(n+1)/2, i, j, n_folds=0, in, exact=(n <= EXACT_MAX_N), n_used=0, has_partial=0, train_only;
	int *fold_of=(int *)xmalloc(m*sizeof(int)); // fold index of each row, or -1 if the row is never held out
	double *full=(double *)xcalloc(np, sizeof(double));
	double total=0, total_w=0, *fold_total, *fold_w;
	int *fold_status, n_unconverged=0, n_failed=0;
	fit_info info;
	model *mo;

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

	train_only=(!opt->legacy_cv && has_partial);
	if (start != NULL) for(i=0;i<np;i++) full[i]=start[i];
	if (!train_only && !(opt->legacy_cv && has_partial && start != NULL)) {
		// No missing entries among these rows: strictly convex, so a full-data fit at this lambda is a
		// harmless warm start for every fold. (With missing entries, --legacy does what the published
		// code did and starts every fold from the complete-rows fit it was handed, with no extra fit.)
		mo=build_model(d, include, log_sparsity);
		info=fit_model(mo, full, opt->objective);
		free_model(mo);
		if (info.status == FIT_NUMERICAL) { free(fold_of); free(full); return NAN; }
	}

	// per-fold results are stored and summed in order afterwards, so the answer does not depend on the
	// number of threads or on how the folds were scheduled
	fold_total=(double *)xcalloc(n_folds, sizeof(double));
	fold_w=(double *)xcalloc(n_folds, sizeof(double));
	fold_status=(int *)xcalloc(n_folds, sizeof(int));
#pragma omp parallel for schedule(dynamic)
	for(in=0;in<n_folds;in++) {
		char *mask=(char *)xmalloc(m), *cmask=(char *)xmalloc(m);
		double *x=(double *)xmalloc(np*sizeof(double)), lz=0, ll, wt;
		fit_info fi;
		model *ml;
		int a, left=0, left_complete=0;

		for(a=0;a<m;a++) {
			mask[a]=(include == NULL || include[a]) && (fold_of[a] != in);
			cmask[a]=mask[a] && (d->n_blanks[a] == 0);
			if (mask[a] && d->w[a] > 0) left++;
			if (cmask[a] && d->w[a] > 0) left_complete++;
		}
		fold_status[in]=FIT_CONVERGED;
		if (left > 0) {
			if (!train_only) {
				for(a=0;a<np;a++) x[a]=full[a];
			} else if (stage1 != NULL && left_complete > 0) {
				// stage 1 on the complete training rows: strictly convex, so the warm start (the
				// complete-rows fit of the full data) changes the cost and not the answer
				for(a=0;a<np;a++) x[a]=full[a];
				ml=build_model(d, cmask, *stage1);
				fi=fit_model(ml, x, opt->objective);
				free_model(ml);
				if (fi.status != FIT_CONVERGED) fold_status[in]=fi.status;
			} else {
				for(a=0;a<np;a++) x[a]=0.0;
			}
			ml=build_model(d, mask, log_sparsity);
			fi=fit_model(ml, x, opt->objective);
			free_model(ml);
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
		free(mask); free(cmask); free(x);
	}
	for(in=0;in<n_folds;in++) {
		total += fold_total[in]; total_w += fold_w[in];
		if (fold_status[in] == FIT_NUMERICAL) n_failed++;
		else if (fold_status[in] != FIT_CONVERGED) n_unconverged++;
	}
	free(fold_of); free(full); free(fold_total); free(fold_w); free(fold_status);
	if (n_failed > 0) {
		fprintf(stderr, "mpf: warning: at log-sparsity %g, %d of %d cross-validation fits failed numerically; this value is skipped\n", log_sparsity, n_failed, n_folds);
		return NAN;
	}
	if (n_unconverged > 0) fprintf(stderr, "mpf: warning: at log-sparsity %g, %d of %d cross-validation fits did not converge\n", log_sparsity, n_unconverged, n_folds);
	if (!(total_w > 0)) die("cross-validation could not score any held-out rows");
	if (opt->verbose) printf("For log-sparsity=%lf, Log-l of held-out data is: %lf\n", log_sparsity, total/total_w);
	return total/total_w;
}

// negative held-out log-likelihood, for minimisation
double cross_holder(double log_sparsity, void *params) {
	cv_context *cv=(cv_context *)params;
	cv->last_value=-cross(cv->data, cv->include, log_sparsity, cv->start, cv->stage1, cv->opt);
	return cv->last_value;
}

// KL divergence of the fit at this log-sparsity from the known true parameters (NaN if the fit failed numerically)
double kl_holder(double log_sparsity, void *params) {
	cv_context *cv=(cv_context *)params;
	int n=cv->data->n, np=n*(n+1)/2, i;
	double *x=(double *)xcalloc(np, sizeof(double));
	model *mo=build_model(cv->data, cv->include, log_sparsity);
	fit_info info;

	if (cv->start != NULL) for(i=0;i<np;i++) x[i]=cv->start[i];
	info=fit_model(mo, x, cv->opt->objective);
	if (!info.converged && cv->opt->verbose) fprintf(stderr, "mpf: warning: fit at log-sparsity %g %s (max |gradient| %.3g)\n", log_sparsity, fit_status_name(info.status), info.gmax);
	free_model(mo);
	cv->last_value=(info.status == FIT_NUMERICAL) ? NAN : full_kl(n, cv->truth, x);
	cv->last_fit=info;
	free(x);
	return cv->last_value;
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
	} else {
		lo=SEARCH_LO;
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
