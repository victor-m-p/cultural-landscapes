// Data handling, the MPF objective (with three treatments of missing data), and the minimiser.
//
// Conventions (as in the paper): p(s) is proportional to exp(H(s)), with
//     H(s) = sum_{a<b} J_ab s_a s_b + sum_a h_a s_a,      s_a = +1 / -1.
// Parameters are stored as the n(n-1)/2 couplings in row-major order (0,1),(0,2),...,(1,2),... followed
// by the n fields.
//
// For a data state s, the flow into the neighbour with bit k flipped is
//     exp( (H(s^k) - H(s)) / 2 ) = exp( -s_k phi_k ),     phi_k = h_k + sum_j J_kj s_j,
// so all n neighbours of a state cost O(n^2) in total, and the gradient of each flow touches only the
// parameters that involve node k.
#include "mpf.h"
#include <stdarg.h>

void die(const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "mpf: ");
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, "\n");
	va_end(ap);
	exit(2);
}

void *xmalloc(size_t bytes) {
	void *p=malloc(bytes > 0 ? bytes : 1);
	if (p == NULL) die("out of memory (wanted %zu bytes)", bytes);
	return p;
}

void *xcalloc(size_t count, size_t size) {
	void *p=calloc(count > 0 ? count : 1, size > 0 ? size : 1);
	if (p == NULL) die("out of memory (wanted %zu x %zu bytes)", count, size);
	return p;
}

const char *fit_status_name(int status) {
	switch(status) {
		case FIT_CONVERGED: return "converged";
		case FIT_STALLED: return "stalled before the gradient became small";
		case FIT_MAXIT: return "reached the iteration limit";
		default: return "hit a non-finite objective or gradient";
	}
}

// ---------------------------------------------------------------------------------------------
// reading and writing
// ---------------------------------------------------------------------------------------------

// File format: number of rows, number of nodes, then one row per line: a string of n characters from
// {0, 1, X} (X = missing), whitespace, and a weight. If the file holds more rows than the header says,
// only the first m are used.
dataset *read_data(const char *filename) {
	FILE *f;
	dataset *d;
	char *buf, fmt[32];
	int i, j, m, n;
	size_t len;

	f=fopen(filename, "r");
	if (f == NULL) die("cannot open data file '%s'", filename);
	if (fscanf(f, "%d %d", &m, &n) != 2) die("'%s': expected the number of rows and the number of nodes on the first two lines", filename);
	if (m < 1) die("'%s': number of rows must be positive (got %d)", filename, m);
	if (n < 2 || n > MAX_NODES) die("'%s': number of nodes must be between 2 and %d (got %d)", filename, MAX_NODES, n);

	d=(dataset *)xmalloc(sizeof(dataset));
	d->m=m; d->n=n; d->group=NULL; d->nn=1;
	d->v=(int8_t *)xmalloc((size_t)m*n*sizeof(int8_t));
	d->w=(double *)xmalloc(m*sizeof(double));
	d->n_blanks=(int *)xmalloc(m*sizeof(int));

	buf=(char *)xmalloc(4096);
	snprintf(fmt, sizeof(fmt), "%%4095s");
	for(i=0;i<m;i++) {
		if (fscanf(f, fmt, buf) != 1) die("'%s': header promises %d rows but only %d were found", filename, m, i);
		len=strlen(buf);
		if ((int)len != n) die("'%s': row %d has %zu entries, expected %d", filename, i+1, len, n);
		d->n_blanks[i]=0;
		for(j=0;j<n;j++) {
			if (buf[j] == '1') d->v[(size_t)i*n+j]=1;
			else if (buf[j] == '0') d->v[(size_t)i*n+j]=-1;
			else if (buf[j] == 'X' || buf[j] == 'x') { d->v[(size_t)i*n+j]=0; d->n_blanks[i]++; }
			else die("'%s': row %d, entry %d is '%c'; expected 0, 1 or X", filename, i+1, j+1, buf[j]);
		}
		if (d->n_blanks[i] > MAX_BLANKS) die("'%s': row %d has %d missing entries; at most %d are supported", filename, i+1, d->n_blanks[i], MAX_BLANKS);
		if (d->n_blanks[i] == n) die("'%s': row %d has no observed entries", filename, i+1);
		if (fscanf(f, "%lf", &d->w[i]) != 1) die("'%s': row %d has no weight", filename, i+1);
		if (!(d->w[i] >= 0) || !isfinite(d->w[i])) die("'%s': row %d has an invalid weight", filename, i+1);
	}
	free(buf);
	fclose(f);
	return d;
}

// One integer per data row. Rows that share an id (e.g. the expansions of one inconsistently-coded
// entry) are held out together in cross-validation.
void read_groups(dataset *d, const char *filename) {
	FILE *f=fopen(filename, "r");
	int i;
	if (f == NULL) die("cannot open groups file '%s'", filename);
	d->group=(int *)xmalloc(d->m*sizeof(int));
	for(i=0;i<d->m;i++) {
		if (fscanf(f, "%d", &d->group[i]) != 1) die("'%s': expected %d group ids, found %d", filename, d->m, i);
	}
	fclose(f);
}

void free_data(dataset *d) {
	if (d == NULL) return;
	free(d->v); free(d->w); free(d->n_blanks); free(d->group); free(d);
}

double *read_params(const char *filename, int np) {
	FILE *f=fopen(filename, "r");
	double *x;
	int j;
	if (f == NULL) die("cannot open parameter file '%s'", filename);
	x=(double *)xmalloc(np*sizeof(double));
	for(j=0;j<np;j++) {
		if (fscanf(f, "%le", &x[j]) != 1) die("'%s': expected %d parameters, found %d", filename, np, j);
		if (!isfinite(x[j])) die("'%s': parameter %d is not a finite number", filename, j+1);
	}
	fclose(f);
	return x;
}

void write_params(const char *filename, const double *x, int np) {
	FILE *f=fopen(filename, "w");
	int j;
	if (f == NULL) die("cannot write parameter file '%s'", filename);
	for(j=0;j<np;j++) fprintf(f, "%.10e ", x[j]);
	if (ferror(f) || fclose(f) != 0) die("error writing parameter file '%s'", filename);
}

// ---------------------------------------------------------------------------------------------
// building the model: merge duplicate rows, expand blanks
// ---------------------------------------------------------------------------------------------

typedef struct { uint64_t blanks, bits; int row; } packed_row;

static int compare_packed(const void *a, const void *b) {
	const packed_row *x=(const packed_row *)a, *y=(const packed_row *)b;
	if (x->blanks != y->blanks) return x->blanks < y->blanks ? -1 : 1;
	if (x->bits != y->bits) return x->bits < y->bits ? -1 : 1;
	return x->row < y->row ? -1 : (x->row > y->row ? 1 : 0); // keeps the merge order deterministic
}

model *build_model(const dataset *d, const char *include, double log_sparsity) {
	model *mo;
	packed_row *pr;
	int n=d->n, i, j, k, used=0, D, nb, pos[MAX_NODES], c, nc;
	long r;
	size_t bytes;

	pr=(packed_row *)xmalloc(d->m*sizeof(packed_row));
	for(i=0;i<d->m;i++) {
		if (include != NULL && !include[i]) continue;
		if (d->w[i] == 0) continue; // zero-weight rows carry no information
		pr[used].blanks=0; pr[used].bits=0; pr[used].row=i;
		for(j=0;j<n;j++) {
			if (d->v[(size_t)i*n+j] == 0) pr[used].blanks |= ((uint64_t)1 << j);
			if (d->v[(size_t)i*n+j] > 0) pr[used].bits |= ((uint64_t)1 << j);
		}
		used++;
	}
	if (used == 0) die("no data rows to fit");
	qsort(pr, used, sizeof(packed_row), compare_packed);

	D=1;
	for(i=1;i<used;i++) if (pr[i].blanks != pr[i-1].blanks || pr[i].bits != pr[i-1].bits) D++;

	mo=(model *)xmalloc(sizeof(model));
	mo->n=n; mo->np=n*(n+1)/2; mo->D=D; mo->maxc=1; mo->total_w=0;
	mo->mult=(double *)xmalloc(D*sizeof(double));
	mo->omask=(double *)xmalloc((size_t)D*n*sizeof(double));
	mo->start=(long *)xmalloc((D+1)*sizeof(long));

	// first pass: weights, masks, and sizes
	k=-1; r=0;
	for(i=0;i<used;i++) {
		if (i == 0 || pr[i].blanks != pr[i-1].blanks || pr[i].bits != pr[i-1].bits) {
			k++;
			mo->mult[k]=0;
			nb=0;
			for(j=0;j<n;j++) {
				mo->omask[(size_t)k*n+j]=((pr[i].blanks >> j) & 1) ? 0.0 : 1.0;
				nb += (int)((pr[i].blanks >> j) & 1);
			}
			nc=1 << nb;
			if (nc > mo->maxc) mo->maxc=nc;
			mo->start[k]=r;
			r += nc;
		}
		mo->mult[k] += d->w[pr[i].row];
		mo->total_w += d->w[pr[i].row];
	}
	mo->start[D]=r;
	mo->R=r;

	bytes=(size_t)r*n*sizeof(double);
	if (bytes > ((size_t)1 << 35)) die("expanding the missing entries needs %.1f GB; reduce the number of blanks per row", bytes/1073741824.0);
	mo->sig=(double *)xmalloc(bytes);
	mo->lwfrozen=(double *)xmalloc(r*sizeof(double));

	// second pass: expand each observation into its 2^blanks completions (blank j gets bit j of c)
	k=-1;
	for(i=0;i<used;i++) {
		if (!(i == 0 || pr[i].blanks != pr[i-1].blanks || pr[i].bits != pr[i-1].bits)) continue;
		k++;
		nb=0;
		for(j=0;j<n;j++) if ((pr[i].blanks >> j) & 1) pos[nb++]=j;
		nc=1 << nb;
		for(c=0;c<nc;c++) {
			double *s=mo->sig+(size_t)(mo->start[k]+c)*n;
			for(j=0;j<n;j++) s[j]=((pr[i].bits >> j) & 1) ? 1.0 : -1.0;
			for(j=0;j<nb;j++) s[pos[j]]=((c >> j) & 1) ? 1.0 : -1.0;
			mo->lwfrozen[mo->start[k]+c]=-log((double)nc);
		}
	}
	free(pr);

	// neighbours per fully-observed state: every set of 1..nn nodes flipped together
	mo->nn=d->nn;
	if (mo->nn < 1 || mo->nn > MAX_NN || mo->nn > n) die("NN must be between 1 and %d (and at most the number of nodes); got %d", MAX_NN, mo->nn);
	mo->n_prox=0;
	{
		double comb=1.0;
		for(j=1;j<=mo->nn;j++) { comb=comb*(n-j+1)/j; mo->n_prox += comb; }
	}
	// Same scaling as the published code: lambda * |N| * |D| / N_params, with lambda = 10^log_sparsity
	mo->sparsity=mo->n_prox*exp(log_sparsity*log(10.0))*mo->total_w/mo->np;
	if (!isfinite(mo->sparsity)) die("log-sparsity %g is too large: the penalty overflows", log_sparsity);

	mo->J=(double *)xmalloc((size_t)n*n*sizeof(double));
	mo->gJ=(double *)xmalloc((size_t)n*n*sizeof(double));
	mo->X=(double *)xmalloc((size_t)mo->maxc*n*sizeof(double));
	mo->T=(double *)xmalloc((size_t)mo->maxc*n*sizeof(double));
	mo->H=(double *)xmalloc(mo->maxc*sizeof(double));
	mo->lw=(double *)xmalloc(mo->maxc*sizeof(double));
	mo->g=(double *)xmalloc(mo->maxc*sizeof(double));
	mo->u=(double *)xmalloc(n*sizeof(double));
	mo->Rk=(double *)xmalloc(n*sizeof(double));
	mo->LQ=(double *)xmalloc(n*sizeof(double));
	mo->max_sub=0; mo->sub_mem=NULL; mo->sub_size=NULL; mo->E=NULL; mo->Rv=NULL; mo->LQs=NULL; mo->Ru=NULL; mo->Pu=NULL;
	if (mo->nn > 1) {
		if (mo->n_prox*mo->maxc*sizeof(double) > 8e9) die("NN=%d with %d nodes needs %.0f flows per state; too large", mo->nn, n, mo->n_prox);
		mo->max_sub=(long)(mo->n_prox+0.5);
		mo->sub_mem=(int *)xmalloc((size_t)mo->max_sub*mo->nn*sizeof(int));
		mo->sub_size=(int *)xmalloc((size_t)mo->max_sub*sizeof(int));
		mo->E=(double *)xmalloc((size_t)mo->max_sub*mo->maxc*sizeof(double));
		mo->Rv=(double *)xmalloc((size_t)mo->max_sub*sizeof(double));
		mo->LQs=(double *)xmalloc((size_t)mo->max_sub*sizeof(double));
		mo->Ru=(double *)xmalloc(n*sizeof(double));
		mo->Pu=(double *)xmalloc((size_t)n*n*sizeof(double));
	}
	return mo;
}

void free_model(model *m) {
	if (m == NULL) return;
	free(m->sig); free(m->omask); free(m->mult); free(m->start); free(m->lwfrozen);
	free(m->J); free(m->gJ); free(m->X); free(m->T); free(m->H); free(m->lw); free(m->g); free(m->u); free(m->Rk); free(m->LQ);
	free(m->sub_mem); free(m->sub_size); free(m->E); free(m->Rv); free(m->LQs); free(m->Ru); free(m->Pu);
	free(m);
}

// ---------------------------------------------------------------------------------------------
// the objective
// ---------------------------------------------------------------------------------------------
//
// For an observation o with completions c (weights w_c = p_theta(c | o)), and T_ck = exp(-s_k phi_k)
// the flow out of completion c along observed node k (flows along missing nodes are not used):
//
//   OBJ_LEGACY    K_o = sum_c w_c sum_k T_ck                 gradient includes d w_c / d theta
//   OBJ_FROZEN    K_o = sum_c w_c sum_k T_ck                 w_c held fixed (re-estimated by fit_model)
//   OBJ_MARGINAL  K_o = sum_k sqrt( sum_c w_c T_ck^2 )       = sum_k sqrt( p_theta(o^k) / p_theta(o) )
//
// OBJ_MARGINAL is ordinary MPF for the marginal distribution of the observed nodes. With no missing
// entries the three are the same convex function. See mpf.h and CLAUDE_IMPROVEMENTS_SUMMARY.md for what
// each one assumes about how entries come to be missing.
//
// Everything that involves a completion weight is computed from log weights, so that a tiny weight
// multiplying a huge flow (or the reverse) keeps its true size; complete rows never square a flow.
//
// Returns K plus the penalty sparsity * |theta|^2 / 2. If grad != NULL it receives the gradient.
// If set_frozen != 0 the current completion weights are stored for OBJ_FROZEN.
// The result (or the gradient) can be non-finite if a flow itself overflows a double; callers check.
static double log_sum_exp(const double *v, int n) {
	double mx=v[0], z=0;
	int i;
	for(i=1;i<n;i++) if (v[i] > mx) mx=v[i];
	if (!isfinite(mx)) return mx;
	for(i=0;i<n;i++) z += exp(v[i]-mx);
	return mx+log(z);
}

// All subsets of 1..nn of the observed nodes, in a fixed order: mem[idx*nn+i] are the members (ascending).
static long build_subsets(int n_obs, const int *obs, int nn, int *mem, int *size) {
	long count=0;
	int idx[MAX_NN], i, j, k;
	for(j=1;j<=nn && j<=n_obs;j++) {
		for(i=0;i<j;i++) idx[i]=i;
		for(;;) {
			size[count]=j;
			for(i=0;i<j;i++) mem[count*nn+i]=obs[idx[i]];
			count++;
			for(i=j-1;i>=0 && idx[i] == n_obs-j+i;i--);
			if (i < 0) break;
			idx[i]++;
			for(k=i+1;k<j;k++) idx[k]=idx[k-1]+1;
		}
	}
	return count;
}

// The objective for nn > 1: flows from each data state to every state reached by flipping a set A of
// 1..nn observed nodes together,
//     exp( (H(s^A) - H(s)) / 2 ) = exp( - sum_{k in A} x_k + 2 sum_{k<l in A} J_kl s_k s_l ),   x_k = s_k phi_k.
// A flow depends on h_a only for a in A and on J_ab only when exactly one of a, b is in A, so with
//     Ru_a  = sum of (weighted) flows over the sets that contain a
//     Pu_ab = sum of (weighted) flows over the sets that contain both a and b
// the gradient is  dh_a = -s_a Ru_a,  dJ_ab = -s_a s_b (Ru_a + Ru_b - 2 Pu_ab).
// The three treatments of missing data are as for single flips, with "k" replaced by "A":
//     marginal:  K_o = sum_A sqrt( p(o^A) / p(o) ),  A a set of observed nodes;
//     legacy:    A a set of observed nodes;
//     frozen:    A any set with at least one observed node (see below), which is what keeps the
//                fixed point consistent under missing-at-random when nn > 1.
static double compute_k_multi(model *m, const double *theta, double *grad, int objective, int set_frozen) {
	int n=m->n, nn=m->nn, d, c, k, l, i, j, cnt, nc, n_obs, obs[MAX_NODES];
	const double *h=theta+n*(n-1)/2, *s, *om;
	double K=0, mult, x, phi, Hc, lse, total, a, U, v, coef, sumR, wc, e, t, *X, *E, *gh=NULL;
	long idx, n_sub;

	cnt=0;
	for(k=0;k<n;k++) {
		m->J[k*n+k]=0;
		for(l=k+1;l<n;l++) { m->J[k*n+l]=theta[cnt]; m->J[l*n+k]=theta[cnt]; cnt++; }
	}
	if (grad != NULL) {
		for(k=0;k<n*n;k++) m->gJ[k]=0;
		for(k=0;k<m->np;k++) grad[k]=0;
		gh=grad+n*(n-1)/2;
	}

	for(d=0;d<m->D;d++) {
		nc=(int)(m->start[d+1]-m->start[d]);
		mult=m->mult[d];
		om=m->omask+(size_t)d*n;
		if (objective == OBJ_FROZEN && nc > 1) {
			// Frozen weights: flows go to every neighbour within distance nn, including those that flip a
			// missing node together with an observed one. Summed over patterns of missing entries, the
			// observed parts with their model completions reproduce the full joint distribution when
			// entries are missing at random, so the fixed point solves the complete-data MPF equation
			// for any neighbourhood. Sets that lie entirely inside the missing nodes cancel identically
			// within a row and are dropped (for nn = 1 that leaves exactly the observed single flips).
			long kept=0;
			for(k=0;k<n;k++) obs[k]=k;
			n_sub=build_subsets(n, obs, nn, m->sub_mem, m->sub_size);
			for(idx=0;idx<n_sub;idx++) {
				int any=0;
				for(i=0;i<m->sub_size[idx];i++) if (om[m->sub_mem[idx*nn+i]] > 0) any=1;
				if (!any) continue;
				if (kept != idx) {
					m->sub_size[kept]=m->sub_size[idx];
					for(i=0;i<m->sub_size[idx];i++) m->sub_mem[kept*nn+i]=m->sub_mem[idx*nn+i];
				}
				kept++;
			}
			n_sub=kept;
		} else {
			n_obs=0;
			for(k=0;k<n;k++) if (om[k] > 0) obs[n_obs++]=k;
			n_sub=build_subsets(n_obs, obs, nn, m->sub_mem, m->sub_size);
		}

		// H and the log flow of every subset, for every completion
		for(c=0;c<nc;c++) {
			s=m->sig+(size_t)(m->start[d]+c)*n;
			X=m->X+(size_t)c*n;
			E=m->E+(size_t)c*n_sub;
			Hc=0;
			for(k=0;k<n;k++) {
				const double *Jk=m->J+(size_t)k*n;
				phi=h[k];
				for(l=0;l<n;l++) phi += Jk[l]*s[l];
				x=s[k]*phi;
				Hc += 0.5*(x+s[k]*h[k]);
				X[k]=x;
			}
			m->H[c]=Hc;
			for(idx=0;idx<n_sub;idx++) {
				const int *mem=m->sub_mem+idx*nn;
				int sz=m->sub_size[idx];
				e=0;
				for(i=0;i<sz;i++) {
					e -= X[mem[i]];
					for(j=i+1;j<sz;j++) e += 2*m->J[(size_t)mem[i]*n+mem[j]]*s[mem[i]]*s[mem[j]];
				}
				E[idx]=e;
			}
		}

		// log completion weights
		if (nc == 1) {
			m->lw[0]=0.0;
		} else if (objective == OBJ_FROZEN && !set_frozen) {
			for(c=0;c<nc;c++) m->lw[c]=m->lwfrozen[m->start[d]+c];
		} else {
			lse=log_sum_exp(m->H, nc);
			for(c=0;c<nc;c++) m->lw[c]=m->H[c]-lse;
		}
		if (set_frozen) for(c=0;c<nc;c++) m->lwfrozen[m->start[d]+c]=m->lw[c];

		if (objective != OBJ_MARGINAL || nc == 1) {
			// E becomes w_c * flow
			total=0;
			for(c=0;c<nc;c++) {
				E=m->E+(size_t)c*n_sub;
				v=0;
				for(idx=0;idx<n_sub;idx++) { E[idx]=exp(m->lw[c]+E[idx]); v += E[idx]; }
				m->g[c]=v;
				total += v;
			}
			K += mult*total;
			if (grad != NULL) for(c=0;c<nc;c++) {
				s=m->sig+(size_t)(m->start[d]+c)*n;
				E=m->E+(size_t)c*n_sub;
				for(k=0;k<n;k++) m->Ru[k]=0;
				for(k=0;k<n*n;k++) m->Pu[k]=0;
				for(idx=0;idx<n_sub;idx++) {
					const int *mem=m->sub_mem+idx*nn;
					int sz=m->sub_size[idx];
					t=mult*E[idx];
					for(i=0;i<sz;i++) {
						m->Ru[mem[i]] += t;
						for(j=i+1;j<sz;j++) m->Pu[(size_t)mem[i]*n+mem[j]] += t;
					}
				}
				a=(objective == OBJ_LEGACY && nc > 1) ? mult*(m->g[c]-exp(m->lw[c])*total) : 0.0;
				for(k=0;k<n;k++) {
					double sk=s[k], *gJk=m->gJ+(size_t)k*n, *Pk=m->Pu+(size_t)k*n;
					gh[k] += sk*(a-m->Ru[k]);
					for(l=k+1;l<n;l++) gJk[l] += sk*s[l]*(a-m->Ru[k]-m->Ru[l]+2*Pk[l]);
				}
			}
		} else {
			// log of p(o^A)/p(o) = log sum_c exp(log w_c + 2 * log flow)
			sumR=0;
			for(idx=0;idx<n_sub;idx++) {
				for(c=0;c<nc;c++) m->g[c]=m->lw[c]+2*m->E[(size_t)c*n_sub+idx];
				m->LQs[idx]=log_sum_exp(m->g, nc);
				m->Rv[idx]=exp(0.5*m->LQs[idx]);
				sumR += m->Rv[idx];
			}
			K += mult*sumR;
			if (grad != NULL) for(c=0;c<nc;c++) {
				s=m->sig+(size_t)(m->start[d]+c)*n;
				E=m->E+(size_t)c*n_sub;
				for(k=0;k<n;k++) m->Ru[k]=0;
				for(k=0;k<n*n;k++) m->Pu[k]=0;
				U=0;
				for(idx=0;idx<n_sub;idx++) {
					const int *mem=m->sub_mem+idx*nn;
					int sz=m->sub_size[idx];
					t=0.5*mult*m->Rv[idx]*exp(m->lw[c]+2*E[idx]-m->LQs[idx]);
					U += t;
					for(i=0;i<sz;i++) {
						m->Ru[mem[i]] += t;
						for(j=i+1;j<sz;j++) m->Pu[(size_t)mem[i]*n+mem[j]] += t;
					}
				}
				wc=exp(m->lw[c]);
				v=0.5*mult*wc*sumR;
				coef=U-v;
				for(k=0;k<n;k++) {
					double sk=s[k], *gJk=m->gJ+(size_t)k*n, *Pk=m->Pu+(size_t)k*n;
					gh[k] += sk*(coef-2*m->Ru[k]);
					for(l=k+1;l<n;l++) gJk[l] += sk*s[l]*(coef-2*m->Ru[k]-2*m->Ru[l]+4*Pk[l]);
				}
			}
		}
	}

	if (grad != NULL) {
		cnt=0;
		for(k=0;k<n;k++) for(l=k+1;l<n;l++) grad[cnt++]=m->gJ[k*n+l];
	}
	for(k=0;k<m->np;k++) K += m->sparsity*theta[k]*theta[k]/2;
	if (grad != NULL) for(k=0;k<m->np;k++) grad[k] += m->sparsity*theta[k];
	return K;
}

double compute_k(model *m, const double *theta, double *grad, int objective, int set_frozen) {
	int n=m->n, d, c, k, l, cnt, nc;
	const double *h=theta+n*(n-1)/2, *s, *om;
	double K=0, mult, x, phi, Hc, lse, total, a, U, v, coef, sumR, wc, *X, *T, *gh=NULL;

	if (m->nn > 1) return compute_k_multi(m, theta, grad, objective, set_frozen);

	cnt=0;
	for(k=0;k<n;k++) {
		m->J[k*n+k]=0;
		for(l=k+1;l<n;l++) { m->J[k*n+l]=theta[cnt]; m->J[l*n+k]=theta[cnt]; cnt++; }
	}
	if (grad != NULL) {
		for(k=0;k<n*n;k++) m->gJ[k]=0;
		for(k=0;k<m->np;k++) grad[k]=0;
		gh=grad+n*(n-1)/2;
	}

	for(d=0;d<m->D;d++) {
		nc=(int)(m->start[d+1]-m->start[d]);
		mult=m->mult[d];
		om=m->omask+(size_t)d*n;

		// H of every completion, and x_ck = s_k phi_k (the flow along k is exp(-x_ck))
		for(c=0;c<nc;c++) {
			s=m->sig+(size_t)(m->start[d]+c)*n;
			X=m->X+(size_t)c*n;
			Hc=0;
			for(k=0;k<n;k++) {
				const double *Jk=m->J+(size_t)k*n;
				phi=h[k];
				for(l=0;l<n;l++) phi += Jk[l]*s[l];
				x=s[k]*phi;
				Hc += 0.5*(x+s[k]*h[k]);
				X[k]=x;
			}
			m->H[c]=Hc;
		}

		if (nc == 1) { // a complete row: ordinary MPF, the same for all three objectives
			s=m->sig+(size_t)m->start[d]*n;
			total=0;
			for(k=0;k<n;k++) { m->T[k]=exp(-m->X[k]); total += m->T[k]; }
			K += mult*total;
			if (set_frozen) m->lwfrozen[m->start[d]]=0.0;
			if (grad != NULL) {
				for(k=0;k<n;k++) m->u[k]=mult*s[k]*m->T[k];
				for(k=0;k<n;k++) {
					double uk=m->u[k], sk=s[k], *gJk=m->gJ+(size_t)k*n;
					gh[k] -= uk;
					for(l=k+1;l<n;l++) gJk[l] -= uk*s[l]+sk*m->u[l];
				}
			}
			continue;
		}

		// log completion weights
		if (objective == OBJ_FROZEN && !set_frozen) {
			for(c=0;c<nc;c++) m->lw[c]=m->lwfrozen[m->start[d]+c];
		} else {
			lse=log_sum_exp(m->H, nc);
			for(c=0;c<nc;c++) m->lw[c]=m->H[c]-lse;
		}
		if (set_frozen) for(c=0;c<nc;c++) m->lwfrozen[m->start[d]+c]=m->lw[c];

		if (objective != OBJ_MARGINAL) {
			// T_ck here holds w_c * (flow along k) = exp(log w_c - x_ck)
			total=0;
			for(c=0;c<nc;c++) {
				X=m->X+(size_t)c*n; T=m->T+(size_t)c*n;
				v=0;
				for(k=0;k<n;k++) { T[k]=(om[k] > 0) ? exp(m->lw[c]-X[k]) : 0.0; v += T[k]; }
				m->g[c]=v;
				total += v;
			}
			K += mult*total;
			if (grad != NULL) for(c=0;c<nc;c++) {
				s=m->sig+(size_t)(m->start[d]+c)*n;
				T=m->T+(size_t)c*n;
				// derivative of the weights, w_c (f(c) - <f>), contributes (w_c g_c - w_c K_o) f(c)
				a=(objective == OBJ_LEGACY) ? mult*(m->g[c]-exp(m->lw[c])*total) : 0.0;
				for(k=0;k<n;k++) m->u[k]=mult*s[k]*T[k];
				for(k=0;k<n;k++) {
					double ask=a*s[k], uk=m->u[k], sk=s[k], *gJk=m->gJ+(size_t)k*n;
					gh[k] += ask-uk;
					for(l=k+1;l<n;l++) gJk[l] += ask*s[l]-uk*s[l]-sk*m->u[l];
				}
			}
		} else {
			// log of p(o^k)/p(o) = log sum_c exp(log w_c - 2 x_ck)
			sumR=0;
			for(k=0;k<n;k++) {
				if (om[k] > 0) {
					for(c=0;c<nc;c++) m->g[c]=m->lw[c]-2*m->X[(size_t)c*n+k];
					m->LQ[k]=log_sum_exp(m->g, nc);
					m->Rk[k]=exp(0.5*m->LQ[k]);
					sumR += m->Rk[k];
				} else {
					m->LQ[k]=0; m->Rk[k]=0; // no flow along a missing node
				}
			}
			K += mult*sumR;
			if (grad != NULL) for(c=0;c<nc;c++) {
				// d log R_k = ( <f>_{c | o^k} - <f>_{c | o} ) / 2; the posterior over completions given
				// the flipped observation has weights exp(log w_c - 2 x_ck - LQ_k)
				s=m->sig+(size_t)(m->start[d]+c)*n;
				X=m->X+(size_t)c*n;
				wc=exp(m->lw[c]);
				v=0.5*mult*wc*sumR;
				U=0;
				for(k=0;k<n;k++) {
					m->u[k]=(om[k] > 0) ? 0.5*mult*m->Rk[k]*exp(m->lw[c]-2*X[k]-m->LQ[k]) : 0.0;
					U += m->u[k];
				}
				coef=U-v;
				for(k=0;k<n;k++) {
					double sk=s[k], ck=coef-2*m->u[k], *gJk=m->gJ+(size_t)k*n;
					gh[k] += sk*ck;
					for(l=k+1;l<n;l++) gJk[l] += sk*s[l]*(ck-2*m->u[l]);
				}
			}
		}
	}

	if (grad != NULL) {
		cnt=0;
		for(k=0;k<n;k++) for(l=k+1;l<n;l++) grad[cnt++]=m->gJ[k*n+l];
	}
	for(k=0;k<m->np;k++) K += m->sparsity*theta[k]*theta[k]/2;
	if (grad != NULL) for(k=0;k<m->np;k++) grad[k] += m->sparsity*theta[k];
	return K;
}

// ---------------------------------------------------------------------------------------------
// minimisation: L-BFGS with a weak-Wolfe bisection line search
// ---------------------------------------------------------------------------------------------

#define LBFGS_MEMORY 12
#define GTOL_TARGET 1e-9   // stop when max |gradient| falls below this times (total weight * neighbours per state)
#define GTOL_ACCEPT 1e-7   // a fit counts as converged only if max |gradient| is below this times the same scale

// max |g_i|, or infinity if any component is not finite
static double max_abs(const double *g, int np) {
	double gmax=0;
	int i;
	for(i=0;i<np;i++) {
		if (!isfinite(g[i])) return INFINITY;
		if (fabs(g[i]) > gmax) gmax=fabs(g[i]);
	}
	return gmax;
}

static fit_info lbfgs(model *m, double *x, int objective, int maxit) {
	int np=m->np, M=LBFGS_MEMORY, i, j, it, hist=0, ls, stall=0, ended=0;
	double scale=m->total_w*m->n_prox; // the size of K: total weight times the number of flows per state
	double *g=(double *)xmalloc(np*sizeof(double)), *d=(double *)xmalloc(np*sizeof(double));
	double *xn=(double *)xmalloc(np*sizeof(double)), *gn=(double *)xmalloc(np*sizeof(double));
	double *Sm=(double *)xmalloc((size_t)M*np*sizeof(double)), *Ym=(double *)xmalloc((size_t)M*np*sizeof(double));
	double rho[LBFGS_MEMORY], al[LBFGS_MEMORY];
	double f, fn=0, gd, gnd, t, lo, hi, sy, yy, gmax, b, nn;
	fit_info info;

	info.evaluations=0; info.status=FIT_MAXIT;
	f=compute_k(m, x, g, objective, 0); info.evaluations++;
	gmax=max_abs(g, np);
	if (!isfinite(f) || !isfinite(gmax)) { info.status=FIT_NUMERICAL; ended=1; }

	for(it=0;!ended && it<maxit;it++) {
		if (gmax < GTOL_TARGET*scale) { ended=1; break; }

		// two-loop recursion
		for(i=0;i<np;i++) d[i]=-g[i];
		for(j=hist-1;j>=0;j--) {
			double *sj=Sm+(size_t)j*np, *yj=Ym+(size_t)j*np;
			al[j]=0;
			for(i=0;i<np;i++) al[j] += sj[i]*d[i];
			al[j] *= rho[j];
			for(i=0;i<np;i++) d[i] -= al[j]*yj[i];
		}
		if (hist > 0) {
			double *sj=Sm+(size_t)(hist-1)*np, *yj=Ym+(size_t)(hist-1)*np;
			sy=0; yy=0;
			for(i=0;i<np;i++) { sy += sj[i]*yj[i]; yy += yj[i]*yj[i]; }
			for(i=0;i<np;i++) d[i] *= sy/yy;
		}
		for(j=0;j<hist;j++) {
			double *sj=Sm+(size_t)j*np, *yj=Ym+(size_t)j*np;
			b=0;
			for(i=0;i<np;i++) b += yj[i]*d[i];
			b *= rho[j];
			for(i=0;i<np;i++) d[i] += sj[i]*(al[j]-b);
		}
		gd=0;
		for(i=0;i<np;i++) gd += g[i]*d[i];
		if (!(gd < 0)) { // not a descent direction: restart from steepest descent
			hist=0;
			for(i=0;i<np;i++) d[i]=-g[i];
			gd=0;
			for(i=0;i<np;i++) gd += g[i]*d[i];
		}
		t=1.0;
		if (hist == 0) {
			nn=0;
			for(i=0;i<np;i++) nn += g[i]*g[i];
			t=fmin(1.0, 1.0/sqrt(nn));
		}

		// line search: sufficient decrease and curvature conditions, by bisection / doubling.
		// A trial point with a non-finite objective or gradient is treated as a step that is too long.
		lo=0; hi=-1;
		for(ls=0;ls<60;ls++) {
			for(i=0;i<np;i++) xn[i]=x[i]+t*d[i];
			fn=compute_k(m, xn, gn, objective, 0); info.evaluations++;
			if (!isfinite(fn) || !isfinite(max_abs(gn, np))) { hi=t; t=0.5*(lo+hi); continue; }
			gnd=0;
			for(i=0;i<np;i++) gnd += gn[i]*d[i];
			if (!(fn <= f+1e-4*t*gd)) { hi=t; t=0.5*(lo+hi); }
			else if (gnd < 0.9*gd) { lo=t; t=(hi < 0) ? 2*t : 0.5*(lo+hi); }
			else break;
		}
		if (ls == 60 && (!isfinite(fn) || !isfinite(max_abs(gn, np)) || !(fn <= f))) { ended=1; break; } // no acceptable step exists: keep x

		if (hist == M) {
			memmove(Sm, Sm+np, (size_t)(M-1)*np*sizeof(double));
			memmove(Ym, Ym+np, (size_t)(M-1)*np*sizeof(double));
			memmove(rho, rho+1, (M-1)*sizeof(double));
			hist--;
		}
		sy=0;
		for(i=0;i<np;i++) {
			Sm[(size_t)hist*np+i]=xn[i]-x[i];
			Ym[(size_t)hist*np+i]=gn[i]-g[i];
			sy += (xn[i]-x[i])*(gn[i]-g[i]);
		}
		if (sy > 1e-300) { rho[hist]=1.0/sy; hist++; }

		if (fabs(f-fn) <= 1e-14*fabs(f)) stall++; else stall=0;
		for(i=0;i<np;i++) { x[i]=xn[i]; g[i]=gn[i]; }
		f=fn;
		gmax=max_abs(g, np);
		if (stall >= 3) { ended=1; it++; break; } // the objective cannot resolve further progress
	}

	// How it ended is decided by the gradient at the returned point, not by why the loop stopped: a
	// stalled line search is only called convergence if the gradient is in fact small.
	if (info.status != FIT_NUMERICAL) {
		if (gmax < GTOL_ACCEPT*scale) info.status=FIT_CONVERGED;
		else info.status=ended ? FIT_STALLED : FIT_MAXIT;
	}
	info.iterations=it; info.k=f; info.gmax=gmax; info.converged=(info.status == FIT_CONVERGED);
	free(g); free(d); free(xn); free(gn); free(Sm); free(Ym);
	return info;
}

// Fit the model; x holds the starting point on entry and the solution on exit.
// Without missing data the problem is convex and the result does not depend on the starting point.
// With missing data none of the three is convex: the result is a local solution that can depend on the start.
fit_info fit_model(model *m, double *x, int objective) {
	double scale=m->total_w*m->n_prox, *g;
	fit_info info, step;
	int outer;

	if (objective != OBJ_FROZEN || m->maxc == 1) {
		return lbfgs(m, x, objective == OBJ_FROZEN ? OBJ_MARGINAL : objective, 3000);
	}

	// Frozen-weight iteration: freeze the completion weights at the current parameters, take a few
	// L-BFGS steps on the (convex) frozen-weight flow objective, re-estimate the weights, and repeat.
	// A fixed point solves   sum_o sum_c p_theta(c|o) grad_theta [flows out of (o,c)] + penalty = 0,
	// i.e. the flow gradient with the weights treated as constants. This is an estimating equation, not
	// the gradient of an objective, and it is not likelihood EM: there is no monotonicity guarantee.
	// The residual of that equation at the returned point is reported as gmax and decides convergence.
	g=(double *)xmalloc(m->np*sizeof(double));
	info.iterations=0; info.evaluations=0; info.status=FIT_MAXIT; info.k=0; info.gmax=INFINITY;
	for(outer=0;outer<5000;outer++) {
		info.k=compute_k(m, x, g, OBJ_FROZEN, 1);            // freeze the weights at x; g = residual of the estimating equation
		info.evaluations++;
		info.gmax=max_abs(g, m->np);
		if (!isfinite(info.k) || !isfinite(info.gmax)) { info.status=FIT_NUMERICAL; break; }
		if (info.gmax < GTOL_ACCEPT*scale) { info.status=FIT_CONVERGED; break; }
		step=lbfgs(m, x, OBJ_FROZEN, 6);
		info.iterations += step.iterations; info.evaluations += step.evaluations;
		if (step.status == FIT_NUMERICAL) { info.status=FIT_NUMERICAL; break; }
	}
	if (info.status == FIT_MAXIT) {
		// the loop ran out: describe the point that is actually returned, not the one before the last step
		info.k=compute_k(m, x, g, OBJ_FROZEN, 1);
		info.evaluations++;
		info.gmax=max_abs(g, m->np);
		if (!isfinite(info.k) || !isfinite(info.gmax)) info.status=FIT_NUMERICAL;
		else if (info.gmax < GTOL_ACCEPT*scale) info.status=FIT_CONVERGED;
	}
	info.converged=(info.status == FIT_CONVERGED);
	free(g);
	return info;
}
