#include "mpf.h"
// mpf -l [filename] [logsparsity] [NN] // load in data, fit
// mpf -c [filename] [NN] // load in data, fit, using cross-validation to pick best sparsity
// mpf -g [filename] [n_nodes] [n_obs] [beta] // generate data, save both parameters and data to files
// mpf -t [filename] [paramfile] [NN] // load in test data, fit, get KL divergence from truth
// mpf -o [filename_prefix] [NN] // load in data (_data.dat suffix), find best lambda using _params.dat to determine KL
// mpf -k [filename] [paramfile_truth] [paramfile_inferred] // load data, compare truth to inferred
// mpf -z [paramfile] [n_nodes]  // print out probabilities of all configurations under paramfile

int main (int argc, char *argv[]) {
	double t0, beta, *big_list, *truth, *inferred, logl_ans, glob_nloops, best_log_sparsity, kl_cv, kl_cv_sp, kl_true, kl_true_sp, ent, *best_fit;
	all *data;
	int i, n, thread_id, last_pos, in, j, count, pos, n_obs, n_nodes, kfold, num_no_na, tot_uniq, has_nans;
	sample *sav, **sav_list;
	cross_val *cv;
	unsigned long int config;
	char filename_sav[1000];
    FILE *fp;
	prob *p;
	
	t0=clock();

	if ((argc == 1) || (argv[1][0] != '-')) {

		printf("Greetings, Professor Falken. Please specify a command-line option.\n");
		
	} else {
		
		if (argv[1][1] == 'l') {
			data=new_data();
			read_data(argv[2], data);
			process_obs_raw(data);
						
			init_params(data);
			data->log_sparsity=atof(argv[3]);
			create_near(data, atoi(argv[4]));
			
			printf("%i data vectors; %i total; %i NNs\n", data->uniq, data->n_all, data->near_uniq);
									
			simple_minimizer(data);
			
			printf("\n\nparams=[");
			for(i=0;i<data->n_params;i++) {
				if (i < (data->n_params-1)) {
					printf("%.10e, ", data->big_list[i]);
				} else {
					printf("%.10e]\n", data->big_list[i]);
				}
			}

			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_params.dat");
		    fp = fopen(filename_sav, "w+");
			for(j=0;j<data->n_params;j++) {
				fprintf(fp, "%.10e ", data->big_list[j]);
			}
		    fclose(fp);
						
		}

		if (argv[1][1] == 'c') { // cross validation
			// new idea : first find minimum without the NANs, then save that location, and "polish"
			
			data=new_data();
			read_data(argv[2], data);
			best_fit=NULL;
			
			has_nans=0;
			for(i=0;i<data->m;i++) {
				if (data->obs_raw[i]->n_blanks > 0) {
					has_nans++;
				}
			}
			if (has_nans > 0) {
				// first, clean out the blanks...
				sav_list=(sample **)malloc((data->m-has_nans)*sizeof(sample *));
				count=0;
				for(i=0;i<data->m;i++) {
					if (data->obs_raw[i]->n_blanks == 0) {
						sav_list[count]=(sample *)malloc(sizeof(sample));
						sav_list[count]->config_base=(int *)malloc(data->n*sizeof(int));
						sav_list[count]->n_blanks=0;
						sav_list[count]->blanks=NULL;
						for(j=0;j<data->n;j++) {
							sav_list[count]->config_base[j]=data->obs_raw[i]->config_base[j];
						}
						sav_list[count]->mult=data->obs_raw[i]->mult;
						count++;
					}
					free(data->obs_raw[i]->config_base);
					if (data->obs_raw[i]->blanks != NULL) {
						free(data->obs_raw[i]->blanks);
					}
				}
				free(data->obs_raw);
				data->obs_raw=sav_list;
				data->m=data->m-has_nans;
				// then, write the new thing to a temporary file
								
				strcpy(filename_sav, argv[2]);
				strcat(filename_sav, "_noNA.dat");
			    fp = fopen(filename_sav, "w+");
				fprintf(fp, "%i\n%i\n", data->m, data->n);
				for(i=0;i<data->m;i++) {
					for(j=0;j<data->n;j++) {
						if (data->obs_raw[i]->config_base[j] == 0) {
							fprintf(fp, "X");
						}
						if (data->obs_raw[i]->config_base[j] == 1) {
							fprintf(fp, "1");							
						}
						if (data->obs_raw[i]->config_base[j] == -1) {
							fprintf(fp, "0");														
						}
					}
					fprintf(fp, " %lf\n", data->obs_raw[i]->mult);
				}
			    fclose(fp);

				cv=(cross_val *)malloc(sizeof(cross_val));
				cv->filename=filename_sav;
				cv->nn=atoi(argv[3]);
				cv->best_fit=NULL;
				best_log_sparsity=minimize_kl(cv, 0);
				printf("Found a best sparsity without NaNs (%lf)\n", best_log_sparsity);

				data=new_data();
				read_data(filename_sav, data);
				process_obs_raw(data);
						
				init_params(data);
				data->log_sparsity=best_log_sparsity;
				create_near(data, cv->nn);
												
				simple_minimizer(data);
				
				// save the best fit for no-Nans
				printf("Found a best-fit solution without NaNs (%i; %lf): ", data->m, data->k);
				best_fit=(double *)malloc(data->n_params*sizeof(double));
				for(i=0;i<data->n_params;i++) {
					best_fit[i]=data->big_list[i];
					printf("%lf ", best_fit[i]);
				}
				printf("\n");
				
				// now need to find best sparsity for Nans, using the best fit minimum...
				cv=(cross_val *)malloc(sizeof(cross_val));
				cv->filename=argv[2];
				cv->nn=atoi(argv[3]);
				cv->best_fit=best_fit;
				best_log_sparsity=minimize_kl(cv, 0); // don't use fast version, just for safety

				printf("Found a best sparsity with NaNs (%lf): ", best_log_sparsity);
			} else {
				cv=(cross_val *)malloc(sizeof(cross_val));
				cv->filename=argv[2];
				cv->nn=atoi(argv[3]);
				cv->best_fit=best_fit;
				best_log_sparsity=minimize_kl(cv, 0); // don't use fast version, just for safety
			}
						
			printf("Best log_sparsity: %lf\n", best_log_sparsity);
						
			data=new_data();
			read_data(argv[2], data);
			data->best_fit=best_fit;
			process_obs_raw(data);
						
			init_params(data);
			data->log_sparsity=best_log_sparsity;
			create_near(data, atoi(argv[3]));
			
			printf("Now doing %i\n", data->m);			
			simple_minimizer(data);
			printf("%lf\n", data->k);
			printf("params=[");
			for(i=0;i<data->n_params;i++) {
				if (i < (data->n_params-1)) {
					printf("%.10e, ", data->big_list[i]);
				} else {
					printf("%.10e]\n", data->big_list[i]);
				}
			}
			
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_params.dat");
		    fp = fopen(filename_sav, "w+");
			for(j=0;j<data->n_params;j++) {
				fprintf(fp, "%.10e ", data->big_list[j]);
			}
		    fclose(fp);	
			
		}

		if (argv[1][1] == 'o') { // optimal lambda -- to be written
			
			cv=(cross_val *)malloc(sizeof(cross_val));
			
			// set up the data correctly...
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_data.dat");
			cv->filename=filename_sav;

			cv->nn=atoi(argv[3]);
			
			// read in the true parameters correctly
			data=new_data();
			read_data(filename_sav, data);
			process_obs_raw(data);						
			init_params(data);
			cv->big_list_true=(double *)malloc(data->n_params*sizeof(double));
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_params.dat");
		    fp = fopen(filename_sav, "r");
			for(j=0;j<data->n_params;j++) {
				fscanf(fp, "%le ", &(cv->big_list_true[j]));
			}
		    fclose(fp);
			
			p=(prob *)malloc(sizeof(prob));
			p->n=0;
			tot_uniq=0;
			for(i=0;i<data->uniq;i++) {
				if (data->obs[i]->n_blanks == 0) {
					p->n++;
					tot_uniq += data->obs[i]->mult;
				}
			}
			p->norm=-1;
			p->p=(double *)malloc(p->n*sizeof(double));
			ent=0;
			count=0;
			for(i=0;i<data->uniq;i++) {
				if (data->obs[i]->n_blanks == 0) {
					p->p[count]=data->obs[i]->mult;
					ent -= (data->obs[i]->mult*1.0/tot_uniq)*log((data->obs[i]->mult*1.0/tot_uniq))/log(2);
					count++;
				}
			}
			printf("NSB entropy of data: %lf\n", entropy_nsb(p));
			printf("Naieve entropy of data: %lf\n", ent);
			
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_data.dat");
			cv->filename=filename_sav;
			
			best_log_sparsity=minimize_kl_true(cv);

			cv->kl_true=kl_holder(best_log_sparsity, (void *)cv);
			
			printf("Best log_sparsity: %lf\n", best_log_sparsity);
			kl_true=cv->kl_true;
			kl_true_sp=best_log_sparsity;
			printf("KL at best log_sparsity: %lf\n", cv->kl_true);
			
			// now do CV
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_data.dat");
			cv->filename=filename_sav;
			best_log_sparsity=minimize_kl(cv, 0); // use fast version
			
			printf("Best log_sparsity CV: %lf\n", best_log_sparsity);
			
			data=new_data();
			read_data(cv->filename, data);
			process_obs_raw(data);
						
			init_params(data);
			data->log_sparsity=best_log_sparsity;
			create_near(data, cv->nn);
												
			simple_minimizer(data);
			
			cv->kl_true=kl_holder(best_log_sparsity, (void *)cv);
			kl_cv=cv->kl_true;
			kl_cv_sp=best_log_sparsity;
			printf("KL at CV'd log_sparsity: %lf\n", cv->kl_true);
			
			printf("val=[%.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f]\n", ent, entropy_nsb(p), kl_true, kl_true_sp, kl_cv, kl_cv_sp, kl_holder(-100, (void *)cv), -1000.0);
		}
		
		if (argv[1][1] == 'g') {
			n_nodes=atoi(argv[3]);
			n_obs=atoi(argv[4]);
			beta=atof(argv[5]);
	
			data=new_data();
			data->n=n_nodes;
			data->m=n_obs;
			init_params(data);
			
			for(i=0;i<data->n_params;i++) {
				data->big_list[i]=gsl_ran_gaussian(data->r, 1.0)*beta;
			}
			
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_data.dat");
		    fp = fopen(filename_sav, "w+");
		    fprintf(fp, "%i\n%i\n", data->m, data->n);
			for(j=0;j<data->m;j++) {
				config=gsl_rng_uniform_int(data->r, (1 << data->n));
				mcmc_sampler(&config, 1000, data);
				for(i=0;i<data->n;i++) {
					if (config & (1 << i)) {
						fprintf(fp, "1");
					} else {
						fprintf(fp, "0");
					}
				}
				fprintf(fp, " 1.0\n");
			}
		    fclose(fp);

			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_params.dat");
		    fp = fopen(filename_sav, "w+");
			for(j=0;j<data->n_params;j++) {
				fprintf(fp, "%.10e ", data->big_list[j]);
			}
		    fclose(fp);
		}
		if (argv[1][1] == 't') {
			
			cv=(cross_val *)malloc(sizeof(cross_val));
			cv->filename=argv[2];
			cv->nn=atoi(argv[4]);
			best_log_sparsity=minimize_kl(cv, 0); // use fast version
			
			printf("Best log_sparsity: %lf\n", best_log_sparsity);
			
			data=new_data();
			read_data(filename_sav, data);
			process_obs_raw(data);
						
			init_params(data);
			data->log_sparsity=best_log_sparsity;
			create_near(data, cv->nn);
												
			simple_minimizer(data);
			
			truth=(double *)malloc(data->n_params*sizeof(double));
		    fp = fopen(argv[3], "r");
			for(j=0;j<data->n_params;j++) {
				fscanf(fp, "%le ", &(truth[j]));
			}
		    fclose(fp);
			
			printf("KL divergence from truth: %.10f\n", full_kl(data, data->big_list, truth));
			// now compare to the true distribution
		}
		if (argv[1][1] == 'k') {
			data=new_data();
			read_data(argv[2], data);
			process_obs_raw(data);
			init_params(data);
			
			truth=(double *)malloc(data->n_params*sizeof(double));
		    fp = fopen(argv[3], "r");
			for(j=0;j<data->n_params;j++) {
				fscanf(fp, "%le ", &(truth[j]));
			}
		    fclose(fp);

			inferred=(double *)malloc(data->n_params*sizeof(double));
		    fp = fopen(argv[4], "r");
			for(j=0;j<data->n_params;j++) {
				fscanf(fp, "%le ", &(inferred[j]));
			}
		    fclose(fp);
			
			printf("KL: %lf\n", full_kl(data, inferred, truth));
			// n=atoi(argv[3]);
			// truth=(double *)malloc((n*(n+1)/2)*sizeof(double));
			// 		    fp = fopen(argv[2], "r");
			// for(j=0;j<(n*(n+1)/2);j++) {
			// 	fscanf(fp, "%le ", &(truth[j]));
			// }
			// 		    fclose(fp);
			//
			// strcpy(filename_sav, argv[2]);
			// strcat(filename_sav, "_probs.dat");
			// compute_probs(n, truth, filename_sav);
			// now compare to the true distribution
		}
		if (argv[1][1] == 'z') {
			n=atoi(argv[3]);
			truth=(double *)malloc((n*(n+1)/2)*sizeof(double));
		    fp = fopen(argv[2], "r");
			for(j=0;j<n*(n+1)/2;j++) {
				fscanf(fp, "%le ", &(truth[j]));
			}
		    fclose(fp);
			
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_probs.dat");
			
			compute_probs(n, truth, filename_sav);
		}
	}
	printf("Clock time: %14.12lf seconds.\n", (clock() - t0)/CLOCKS_PER_SEC);
	exit(1);
}