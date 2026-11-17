
unsigned long convert(samples *data, int *config) {
	int i;
	unsigned long out=0;
	
	for(i=0;i<data->n;i++) {
		if (config[i] == 1) {
			out += (1 << i);
		}
	}
	
	return out;
}

void mcmc_sampler_partial(samples *data, int loc, int iter, double *big_list) {
	int i, j, pos, count, *config;
	double running, exp_running;
	
	config=data->obs[loc];
	
	if (data->n_blanks[loc] == 1) { // if there's literally just one blank, it's very easy
		running=0;
		for(j=0;j<data->n;j++) {
			if (pos != j) {
				running += (config[pos] - flip(config[pos]))*config[j]*data->big_list[data->ij[pos][j]];
			}
		}
		running += (config[pos] - flip(config[pos]))*data->big_list[data->h_offset+pos];
		running = -1*running;
	
		exp_running=exp(running);
		if (gsl_rng_uniform(data->r) < exp_running/(1+exp_running)) {
			config[pos]=flip(config[pos]);
		}		
	} else {
		for(i=0;i<data->n_blanks[loc]*iter;i++) {
			pos=data->blanks[loc][(int)gsl_rng_uniform_int(data->r, data->n_blanks[loc])]; 
			// pick a point in the blank set randomly -- these are the only ones we consider for flips
			// we could speed this up by computing the fields for all the non-blank data
			// but this would require we keep track of a bunch of additional fields -- one for each blank, and memory access costs can be high
		
			running=0;
			// change in energy function from the proposed flip
			for(j=0;j<data->n;j++) {
				if (pos != j) {
					running += (config[pos] - flip(config[pos]))*config[j]*data->big_list[data->ij[pos][j]];
				}
			}
			running += (config[pos] - flip(config[pos]))*data->big_list[data->h_offset+pos];
		
			running = -1*running; // oops, i meant to get the other ratio; log P(xnew)/P(x)
		
			exp_running=exp(running);
			if (gsl_rng_uniform(data->r) < exp_running/(1+exp_running)) {
				config[pos]=flip(config[pos]);
			}
		
			// if (running > 0) { // if flipping increases the energy... go for it
			// 	config[pos]=flip(config[pos]);
			// } else { // if it decreases the energy, you still might accept
			// 	exp_running=exp(running);
			// 	if (gsl_rng_uniform(data->r) < exp_running) {
			// 		config[pos]=flip(config[pos]);
			// 	}
			// }
		}		
	}
	
}

void pair_ps_file(samples *data, double *inferred, double *truth, char *filename) { // intense, full-enumeration kl calculation... explodes exponentially
	int i, n, ip, jp, sig_ip, sig_jp, count=0;
	double z_inferred=0, z_truth=0, kl=0, max_catch=0, temp1, temp2;
	double e_inferred, e_truth;
	double **pset;
	double t0;
	FILE *fp;
	
	t0=clock();
	
	fp = fopen(filename, "w+");
	fprintf(fp, "%i\n", (1 << data->n));
	n=data->n;

	z_inferred=0;
	z_truth=0;

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
		}
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

		temp1=exp(e_inferred+max_catch);
		temp2=exp(e_truth+max_catch);
		
		z_inferred += temp1;
		z_truth += temp2;

		fprintf(fp, "%.10e  %.10e\n", temp1, temp2);
	}
	
	fprintf(fp, "%.10e  %.10e\n", z_inferred, z_truth);
	fclose(fp);
}

double **pair_ps(samples *data, double *inferred, double *truth) { // intense, full-enumeration kl calculation... explodes exponentially
	int i, n, ip, jp, sig_ip, sig_jp, count=0;
	double z_inferred=0, z_truth=0, kl=0, max_catch=0;
	double e_inferred, e_truth;
	double **pset;
	double t0;

	t0=clock();
	
	pset=(double **)malloc(2*sizeof(double *));
	pset[0]=(double *)malloc((1 << data->n)*sizeof(double));
	pset[1]=(double *)malloc((1 << data->n)*sizeof(double));		

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
		}
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

		pset[0][i] = exp(e_inferred+max_catch);
		pset[1][i] = exp(e_truth+max_catch);
	}
	
	z_inferred=0;
	z_truth=0;
	for(i=0;i<(1 << n);i++) {
		z_inferred += pset[0][i];
		z_truth += pset[1][i];
	}	

	for(i=0;i<(1 << n);i++) {
		pset[0][i] /= z_inferred;
		pset[1][i] /= z_truth;
	}	
	
	return pset;
}


double full_kl(samples *data, double *inferred, double *truth) { // intense, full-enumeration kl calculation... explodes exponentially
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
		}
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
		z_inferred += exp(e_inferred+max_catch);
		z_truth += exp(e_truth+max_catch);
	}
	
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
		kl += (exp(e_truth+max_catch)/z_truth)*log((exp(e_truth+max_catch)/z_truth)/(exp(e_inferred+max_catch)/z_inferred));
	}
	// printf("Clock time KL: %14.12lf seconds.\n", (clock() - t0)/CLOCKS_PER_SEC);
	
	return -kl;
}

double logl_full(samples *data, double *inferred) { // intense, full-enumeration kl calculation... explodes exponentially
	int i, n, ip, jp, sig_ip, sig_jp, count=0;
	double z_inferred=0, kl=0, max_catch=0;
	double e_inferred;
	double t0;

	t0=clock();
	
	n=data->n;
	// first compute the partition function -- we could actually save all the values to memory but it's faster not to; we have to do two loops; one calculates the two partition functions (normalizations) -- the second uses that normalization to compute the probabilities. Beware we are NOT doing checks for underflows/overflows in the exp calculation
	for(i=0;i<(1 << n);i++) {
		
		e_inferred=0;
		count=0;
		for(ip=0;ip<n;ip++) {
			sig_ip=(i & (1 << ip));
			if (sig_ip) {
				sig_ip=1;
			} else {
				sig_ip=-1;
			}
		}
		for(ip=0;ip<n;ip++) {
			sig_ip=(i & (1 << ip));
			if (sig_ip) {
				sig_ip=1;
			} else {
				sig_ip=-1;
			}
			e_inferred += sig_ip*inferred[data->h_offset+ip];
			for(jp=(ip+1);jp<n;jp++) {
				sig_jp=(i & (1 << jp));
				if (sig_jp) {
					sig_jp=1;
				} else {
					sig_jp=-1;
				}
				e_inferred += sig_ip*sig_jp*inferred[count]; // data->ij[ip][jp] -- for super-speed, we'll live on the edge
				count++;
			}
		}
		z_inferred += exp(e_inferred+max_catch);
	}
	
	// then compute the kl function
	for(i=0;i<(1 << n);i++) {
		
		e_inferred=0;
		count=0;
		for(ip=0;ip<n;ip++) {
			sig_ip=(i & (1 << ip));
			if (sig_ip) {
				sig_ip=1;
			} else {
				sig_ip=-1;
			}
			e_inferred += sig_ip*inferred[data->h_offset+ip];
			for(jp=(ip+1);jp<n;jp++) {
				sig_jp=(i & (1 << jp));
				if (sig_jp) {
					sig_jp=1;
				} else {
					sig_jp=-1;
				}
				e_inferred += sig_ip*sig_jp*inferred[count];
				count++;
			}
		}
		// printf("%i %lf %lf %lf\n", i, kl, e_truth, e_inferred);
		kl += (exp(e_inferred+max_catch)/z_inferred)*log((exp(e_inferred+max_catch)/z_inferred));
	}
	// printf("Clock time KL: %14.12lf seconds.\n", (clock() - t0)/CLOCKS_PER_SEC);
	
	return -kl;
}

double logl_sample(samples *data, double *inferred, int loc) { // intense, full-enumeration kl calculation... explodes exponentially
	int i, n, ip, jp, sig_ip, sig_jp, count=0;
	double z_inferred=0, kl=0, max_catch=0;
	double e_inferred;
	double t0;

	t0=clock();
	
	n=data->n;
	// first compute the partition function -- we could actually save all the values to memory but it's faster not to; we have to do two loops; one calculates the two partition functions (normalizations) -- the second uses that normalization to compute the probabilities. Beware we are NOT doing checks for underflows/overflows in the exp calculation
	for(i=0;i<(1 << n);i++) {
		
		e_inferred=0;
		count=0;
		for(ip=0;ip<n;ip++) {
			sig_ip=(i & (1 << ip));
			if (sig_ip) {
				sig_ip=1;
			} else {
				sig_ip=-1;
			}
		}
		for(ip=0;ip<n;ip++) {
			sig_ip=(i & (1 << ip));
			if (sig_ip) {
				sig_ip=1;
			} else {
				sig_ip=-1;
			}
			e_inferred += sig_ip*inferred[data->h_offset+ip];
			for(jp=(ip+1);jp<n;jp++) {
				sig_jp=(i & (1 << jp));
				if (sig_jp) {
					sig_jp=1;
				} else {
					sig_jp=-1;
				}
				e_inferred += sig_ip*sig_jp*inferred[count]; // data->ij[ip][jp] -- for super-speed, we'll live on the edge
				count++;
			}
		}
		z_inferred += exp(e_inferred+max_catch);
	}

	e_inferred=0;
	count=0;
	for(ip=0;ip<n;ip++) {
		sig_ip=(loc & (1 << ip));
		if (sig_ip) {
			sig_ip=1;
		} else {
			sig_ip=-1;
		}
		e_inferred += sig_ip*inferred[data->h_offset+ip];
		for(jp=(ip+1);jp<n;jp++) {
			sig_jp=(loc & (1 << jp));
			if (sig_jp) {
				sig_jp=1;
			} else {
				sig_jp=-1;
			}
			e_inferred += sig_ip*sig_jp*inferred[count];
			count++;
		}
	}
	
	return log(exp(e_inferred)/z_inferred);
}

double sample_states(samples *data, int iter, int n_samp, double *big_list) { // compare to data
	int i, j, hit, n_found=0;
	int *config, *sav;
	unsigned long **set, curr;
	double logl;
	
	config=(int *)malloc(data->n*sizeof(int));
	
	set=(unsigned long **)malloc(sizeof(unsigned long *));
	
	sav=data->obs[0];
	data->obs[0]=config;
	
	for(i=0;i<n_samp;i++) {
		for(j=0;j<data->n;j++) {
			data->obs[0][j]=(2*gsl_rng_uniform_int(data->r, 2)-1);
		}
		mcmc_sampler(data, 0, iter, big_list);
		curr=convert(data, data->obs[0]);
		if (n_found == 0) {
			set[0]=(unsigned long *)malloc(2*sizeof(unsigned long));
			set[0][0]=curr;
			set[0][1]=1;
			n_found++;
		} else {
			hit=0;
			for(j=0;j<n_found;j++) {
				if (set[j][0] == curr) {
					set[j][1]++;
					hit=1;
					break;
				}
			}
			if (hit == 0) {
				set=(unsigned long **)realloc(set, (n_found+1)*sizeof(unsigned long *));
				set[n_found]=(unsigned long *)malloc(2*sizeof(unsigned long));
				set[n_found][0]=curr;
				set[n_found][1]=1;
				n_found++;			
			}
		}
	}
	
	data->obs[0]=sav;
		
	logl=0;
	if (data->uniq == 0) {
		
		for(i=0;i<data->m;i++) {
			curr=convert(data, data->obs[i]);
			hit=0;
			for(j=0;j<n_found;j++) {
				if (curr == set[j][0]) {
					logl += log((double)set[j][1]/n_samp);
					hit=1;
					break;
				}
			}
			if (hit == 0) {
				logl += log(1.0/n_samp) - data->n*log(2.0);
				// printf("Missing data in simulation -- %li\n", curr);
			}
		}
		
	} else {
		
		for(i=0;i<data->uniq;i++) {
			curr=convert(data, data->obs[i]);
			hit=0;
			for(j=0;j<n_found;j++) {
				if (curr == set[j][0]) {
					logl += data->mult[i]*log((double)set[j][1]/n_samp);
					hit=1;
					break;
				}
			}
			if (hit == 0) {
				// printf("Missing data in simulation -- %i %li\n", data->mult[i], curr);
				logl += data->mult[i]*(log(1.0/n_samp) - data->n*log(2.0));
			}
		}		
		
	}
	
	return logl/data->m;
}

double compare_jij(samples *data, int iter, int n_samp, double *big_list, double *big_list_2) { // big_list vs big_list_compare
	int i, j, k, hit, n_found=0, n_found_2=0;
	int *config, *sav;
	unsigned long **set, **set2, curr;
	double logl, diff;
	int count;
	
	config=(int *)malloc(data->n*sizeof(int));
	sav=data->obs[0];
	data->obs[0]=config;
	
	set=(unsigned long **)malloc(sizeof(unsigned long *));	
	for(i=0;i<n_samp;i++) {
		for(j=0;j<data->n;j++) {
			data->obs[0][j]=(2*gsl_rng_uniform_int(data->r, 2)-1); // (2*(rand() % 2)-1); //
		}
		mcmc_sampler(data, 0, iter, big_list);
		curr=convert(data, data->obs[0]);
		if (n_found == 0) {
			set[0]=(unsigned long *)malloc(2*sizeof(unsigned long));
			set[0][0]=curr;
			set[0][1]=1;
			n_found++;
		} else {
			hit=0;
			for(j=0;j<n_found;j++) {
				if (set[j][0] == curr) {
					set[j][1]++;
					hit=1;
					break;
				}
			}
			if (hit == 0) {
				set=(unsigned long **)realloc(set, (n_found+1)*sizeof(unsigned long *));
				set[n_found]=(unsigned long *)malloc(2*sizeof(unsigned long));
				set[n_found][0]=curr;
				set[n_found][1]=1;
				n_found++;			
			}
		}
	}
	
	logl=0;
	for(i=0;i<n_found;i++) {
		diff=0;
		count=0;
		curr=set[i][0];
		for(j=0;j<data->n;j++) {
			if (curr & (1 << j)) {
				config[j]=1;
			} else {
				config[j]=-1;
			}
		}
		for(j=0;j<(data->n-1);j++) {
			for(k=(j+1);k<data->n;k++) {
				diff += (big_list[count]-big_list_2[count])*config[j]*config[k];
				count++;
			}
		}
		for(j=0;j<data->n;j++) {
			diff += (big_list[count]-big_list_2[count])*config[j];
			count++;
		}
		logl += ((double)set[i][1]/(double)n_samp)*diff;
	}
	
	for(i=0;i<n_found;i++) {
		free(set[i]);
	}
	free(set);
	free(config);
	
	data->obs[0]=sav;
		
	return logl;
}
