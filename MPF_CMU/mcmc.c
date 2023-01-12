#include "mpf.h"
#include "stdlib.h"
#define flip(X)  ((X) < 0 ? 1 : -1)
#define flip(X)  ((X) < 0 ? 1 : -1)
#define VAL(a, pos) ((a & (1 << pos)) ? 1.0 : -1.0)

void mcmc_sampler(unsigned long int *config, int iter, all *data) {
	int i, j, pos, count;
	double running, exp_running, rnd, scale;
		
	// quick simulated annealing...
	for(i=0;i<data->n*iter/2;i++) {
		scale=i*1.0/(data->n*iter/2);
		pos=(int)gsl_rng_uniform_int(data->r, data->n); // pick a point randomly
		
		running=0;
		// change in energy function from the proposed flip
		for(j=0;j<data->n;j++) {
			if (pos != j) {
				running -= (VAL((*config),pos) - VAL(((*config) ^ (1 << pos)),pos))*VAL((*config),j)*data->big_list[data->ij[pos][j]]*scale;
			}
		}
		running -= (VAL((*config),pos) - VAL(((*config) ^ (1 << pos)),pos))*data->big_list[data->h_offset+pos]*scale;
				
		exp_running=exp(running);
		rnd=gsl_rng_uniform(data->r);
		if (rnd < exp_running/(1+exp_running)) {
			(*config) = ((*config) ^ (1 << pos));
		}
	}

	// quick simulated annealing...
	for(i=0;i<data->n*iter/2;i++) {
		pos=(int)gsl_rng_uniform_int(data->r, data->n); // pick a point randomly
		
		running=0;
		// change in energy function from the proposed flip
		for(j=0;j<data->n;j++) {
			if (pos != j) {
				running -= (VAL((*config),pos) - VAL(((*config) ^ (1 << pos)),pos))*VAL((*config),j)*data->big_list[data->ij[pos][j]];
			}
		}
		running -= (VAL((*config),pos) - VAL(((*config) ^ (1 << pos)),pos))*data->big_list[data->h_offset+pos];
				
		exp_running=exp(running);
		rnd=gsl_rng_uniform(data->r);
		if (rnd < exp_running/(1+exp_running)) {
			(*config) = ((*config) ^ (1 << pos));
		}
	}
	
}

double log_l_parallel(all *data, unsigned long int config, double *inferred, int do_approx) {
	int i, n, ip, jp, sig_ip, sig_jp, hits, count=0, mc_iter=1000000;
	double z_inferred=0;
	double e_inferred, e_loc;
	unsigned long int config_sample;
	double t0;

	n=data->n;

	if (do_approx == 0) {
		t0=clock();
		
		for(i=0;i<(1 << n);i++) {
		
			e_inferred=0;
			count=0;
			for(ip=0;ip<n;ip++) {
				e_inferred += VAL(i, ip)*inferred[data->h_offset+ip];
				for(jp=(ip+1);jp<n;jp++) {
					e_inferred += VAL(i, ip)*VAL(i, jp)*inferred[count]; // data->ij[ip][jp] -- for super-speed, we'll live on the edge
					count++;
				}
			}
			z_inferred += exp(e_inferred);	
		}
	
		e_loc=0;
		count=0;
		for(ip=0;ip<n;ip++) {
			e_loc += VAL(config, ip)*inferred[data->h_offset+ip];
			for(jp=(ip+1);jp<n;jp++) {
				e_loc += VAL(config, ip)*VAL(config, jp)*inferred[count]; // data->ij[ip][jp] -- for super-speed, we'll live on the edge
				count++;
			}
		}
		// printf("Clock time Exact Computation: %14.12lf seconds.\n", (clock() - t0)/CLOCKS_PER_SEC);
		return e_loc-log(z_inferred);	
			
	} else { // do MCMC sampling
		t0=clock();
		hits=0;
		for(i=0;i<mc_iter;i++) {
			config_sample=gsl_rng_uniform_int(data->r, (1 << data->n));
			mcmc_sampler(&config_sample, 6, data);
			if (config_sample == config) {
				hits++;
			}
		}
		if (hits == 0) { // desparately seeking something...
			while((i<100*mc_iter) & (hits == 0)) {
				config_sample=gsl_rng_uniform_int(data->r, (1 << data->n));
				mcmc_sampler(&config_sample, 6, data);
				i++;
			}
		}
		printf("Clock time MCMC Sampling: %14.12lf seconds.\n", (clock() - t0)/CLOCKS_PER_SEC);
		return log(hits+1.0/((double)i))-log((double)i);
	}
}

void compute_probs(int n, double *big_list, char *filename) {
	int i, ip, jp, sig_ip, sig_jp, count, h_offset;
	double e_inferred, z_inferred=0;
	FILE *fn;

	h_offset=n*(n-1)/2;
	for(i=0;i<(1 << n);i++) {	
		e_inferred=0;
		count=0;
		for(ip=0;ip<n;ip++) {
			e_inferred += VAL(i, ip)*big_list[h_offset+ip];
			for(jp=(ip+1);jp<n;jp++) {
				e_inferred += VAL(i, ip)*VAL(i, jp)*big_list[count]; // data->ij[ip][jp] -- for super-speed, we'll live on the edge
				count++;
			}
		}
		z_inferred += exp(e_inferred);	
	}
	z_inferred=log(z_inferred);
	
    fn = fopen(filename, "w+");
	
	for(i=0;i<(1 << n);i++) {
		e_inferred=0;
		count=0;
		for(ip=0;ip<n;ip++) {
			e_inferred += VAL(i, ip)*big_list[h_offset+ip];
			for(jp=(ip+1);jp<n;jp++) {
				e_inferred += VAL(i, ip)*VAL(i, jp)*big_list[count]; // data->ij[ip][jp] -- for super-speed, we'll live on the edge
				count++;
			}
		}
		
		for(ip=0;ip<n;ip++) {
			if (i & (1 << ip)) {
				fprintf(fn, "1");
			} else {
				fprintf(fn, "0");
			}
		}
		
		fprintf(fn, " %.10e %.10e\n", e_inferred, exp(e_inferred-z_inferred));	
	}
	
    fclose(fn);
}

double log_l(all *data, unsigned long int config, double *inferred, int do_approx) {
	int i, n, ip, jp, sig_ip, sig_jp, hits, count=0, mc_iter=1000000;
	double z_inferred=0;
	double e_inferred, e_loc;
	unsigned long int config_sample;
	double t0;

	n=data->n;

	if (do_approx == 0) {
		t0=clock();
		
		for(i=0;i<(1 << n);i++) {
		
			e_inferred=0;
			count=0;
			for(ip=0;ip<n;ip++) {
				e_inferred += VAL(i, ip)*inferred[data->h_offset+ip];
				for(jp=(ip+1);jp<n;jp++) {
					e_inferred += VAL(i, ip)*VAL(i, jp)*inferred[count]; // data->ij[ip][jp] -- for super-speed, we'll live on the edge
					count++;
				}
			}
			z_inferred += exp(e_inferred);	
		}
	
		e_loc=0;
		count=0;
		for(ip=0;ip<n;ip++) {
			e_loc += VAL(config, ip)*inferred[data->h_offset+ip];
			for(jp=(ip+1);jp<n;jp++) {
				e_loc += VAL(config, ip)*VAL(config, jp)*inferred[count]; // data->ij[ip][jp] -- for super-speed, we'll live on the edge
				count++;
			}
		}
		// printf("Clock time Exact Computation: %14.12lf seconds.\n", (clock() - t0)/CLOCKS_PER_SEC);
		return e_loc-log(z_inferred);	
			
	} else { // do MCMC sampling
		t0=clock();
		hits=0;
		for(i=0;i<mc_iter;i++) {
			config_sample=gsl_rng_uniform_int(data->r, (1 << data->n));
			mcmc_sampler(&config_sample, 6, data);
			if (config_sample == config) {
				hits++;
			}
		}
		if (hits == 0) { // desparately seeking something...
			while((i<10*mc_iter) & (hits == 0)) {
				config_sample=gsl_rng_uniform_int(data->r, (1 << data->n));
				mcmc_sampler(&config_sample, 6, data);
				i++;
			}
		}
		printf("Clock time MCMC Sampling: %14.12lf seconds.\n", (clock() - t0)/CLOCKS_PER_SEC);
		return log(hits+1.0/((double)i))-log((double)i);
	}
}

double full_kl(all *data, double *inferred, double *truth) { // intense, full-enumeration kl calculation... explodes exponentially
	int i, n, ip, jp, sig_ip, sig_jp, count=0;
	double z_inferred=0, z_truth=0, kl=0, max_catch=0;
	double e_inferred, e_truth;
	double t0;

	t0=clock();
	
	n=data->n;
	// first compute the partition function -- we could actually save all the values to memory but it's faster not to; we have to do two loops; one calculates the two partition functions (normalizations) -- the second uses that normalization to compute the probabilities. Beware we are NOT doing checks for underflows/overflows in the exp calculation
	for(i=0;i<(1 << n);i++) {
		
		e_inferred=0;
		e_truth=0;
		count=0;
		for(ip=0;ip<n;ip++) {
			sig_ip=(i & (1 << ip));
			if (sig_ip) {
				sig_ip=1;
			} else {
				sig_ip=-1;
			}
			e_inferred += sig_ip*inferred[data->h_offset+ip];
			e_truth += sig_ip*truth[data->h_offset+ip];
			for(jp=(ip+1);jp<n;jp++) {
				sig_jp=(i & (1 << jp));
				if (sig_jp) {
					sig_jp=1;
				} else {
					sig_jp=-1;
				}
				e_inferred += sig_ip*sig_jp*inferred[count]; // data->ij[ip][jp] -- for super-speed, we'll live on the edge
				e_truth += sig_ip*sig_jp*truth[count];
				count++;
			}
		}
		// if (i == 0) { // check to see if we're exploding over our range
		// 	if ((fabs(e_inferred) > 30) || (fabs(e_truth) > 30)) {
		// 		max_catch=-(fabs(e_inferred)/e_inferred)*MAX(e_inferred, e_truth);
		// 	}
		// }
		z_inferred += exp(e_inferred);
		z_truth += exp(e_truth);
	}
	z_inferred=log(z_inferred);
	z_truth=log(z_truth);
	
	// then compute the kl function
	for(i=0;i<(1 << n);i++) {
		
		e_inferred=0;
		e_truth=0;
		count=0;
		for(ip=0;ip<n;ip++) {
			sig_ip=(i & (1 << ip));
			if (sig_ip) {
				sig_ip=1;
			} else {
				sig_ip=-1;
			}
			e_inferred += sig_ip*inferred[data->h_offset+ip];
			e_truth += sig_ip*truth[data->h_offset+ip];
			for(jp=(ip+1);jp<n;jp++) {
				sig_jp=(i & (1 << jp));
				if (sig_jp) {
					sig_jp=1;
				} else {
					sig_jp=-1;
				}
				e_inferred += sig_ip*sig_jp*inferred[count];
				e_truth += sig_ip*sig_jp*truth[count];
				count++;
			}
		}
		// printf("%i %lf %lf %lf\n", i, kl, e_truth, e_inferred);
		kl += exp(e_truth-z_truth)*(e_truth-z_truth-e_inferred+z_inferred);
			// log(exp(e_truth+max_catch-z_truth)/exp(e_inferred+max_catch-z_inferred));
	}
	// printf("Clock time KL: %14.12lf seconds.\n", (clock() - t0)/CLOCKS_PER_SEC);
	
	return kl;
}
