#include "mpf.h"
#include "stdlib.h"
#define flip(X)  ((X) < 0 ? 1 : -1)
#define MIN(a,b) ((a)<(b)?(a):(b))
#define VAL(a, pos) ((a & (1 << pos)) ? 1.0 : -1.0)
// macros for extracting the pos^th bit from unsigned long int a, and turning it into +1 or -1
int global_length;

unsigned long int convert(int *list) {
	int i;
	unsigned long int ans=0;
	
	for(i=0;i<global_length;i++) {
		if (list[i] > 0) {
			ans += (1 << i);	
		}
	}
	
	return ans;
}

void print_vec(unsigned long a) {
	int i;
	
	for(i=0;i<global_length;i++) {
		if (a & (1 << i)) {
			printf("1");
		} else {
			printf("0");
		}
	}
}

int hamming(unsigned long a, unsigned long b) {
	int i, dist=0;
	
	for(i=0;i<global_length;i++) {
		if ((a & (1 << i)) ^ (b & (1 << i))) {
			dist++;
		}
	}
	return dist;
}

int compare_states(const void* va, const void* vb) {

	  unsigned long int a = *(unsigned long int *)va; 
	  unsigned long int b = *(unsigned long int *) vb;
	  return a < b ? -1 : a > b ? +1 : 0;	
}

int compare_states_near(const void* a, const void* b) {
	unsigned long int va, vb;
	sample **arg1, **arg2;
 	
	va=((near_struct **)a)[0]->config;
 	vb=((near_struct **)b)[0]->config;
	
	return va < vb ? -1 : va > vb ? +1 : 0;	
}

int compare_states_base(const void* a, const void* b) {
	int i;
	sample **arg1, **arg2;
 	
	arg1=(sample **)a;
 	arg2=(sample **)b;
	
 	for(i=0;i<global_length;i++) {
 		if (arg1[0]->config_base[i] > arg2[0]->config_base[i]) return -1;
 		if (arg1[0]->config_base[i] < arg2[0]->config_base[i]) return 1;		
 	}
    return 0;
}

all *new_data() {
	FILE *fn; 
	unsigned long r_seed;
	all *data;
	
	data=(all *)malloc(sizeof(all));
	
	data->m=-1;
	data->n=-1;
	data->n_prox=-1;
	data->near_uniq=-1;
	data->obs_raw=NULL;
	data->obs=NULL;
	data->near=NULL;
	
	data->n_params=-1;
	data->big_list=NULL;
	data->ei=NULL;
	data->nei=NULL;
	
	data->k=-1;
	data->dk=NULL;
	data->ij=NULL;
	data->h_offset=-1;
	
	data->best_fit=NULL;
	
	gsl_rng *r;
	
	
	fn = fopen("/dev/urandom", "rb");
	
	if (fread(&r_seed, sizeof(unsigned long), 1, fn) != 1) {
		/* Failed!--use time instead; beware, could sync with other instances */
		printf("Warning: urandom read fail; using system clock\n");
		r_seed=(unsigned long)time(NULL);
	}
	fclose(fn);
	
	data->r=gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(data->r, r_seed);
	
	return data;
}

void delete_data(all *data) {
	int i, j;
	
	if (data->near != NULL) {
		free(data->near);
	}
	
	if (data->obs != NULL) {
		if (data->uniq > 0) {
			for(i=0;i<data->uniq;i++) {
				free(data->obs[i]->config_base);
				if (data->obs[i]->n_blanks > 0) {
					free(data->obs[i]->blanks); // location of blanks					
				}
				free(data->obs[i]->config);
				free(data->obs[i]->mult_sim);
				for(j=0;j<data->obs[i]->n_config;j++) {
					free(data->obs[i]->prox[j]);
				}
				free(data->obs[i]->prox);
				free(data->obs[i]);
			}		
		}
		free(data->obs);		
	}
	
	if (data->obs_raw != NULL) {
		for(i=0;i<data->m;i++) {
			free(data->obs_raw[i]);
		}
		free(data->obs_raw);
	}
	
	if (data->big_list != NULL) {
		free(data->big_list);
		free(data->ei);
		free(data->nei);
		free(data->dk);
		for(i=0;i<data->n;i++) {
			free(data->ij[i]);
		}
		free(data->ij);
	}
}

void read_data(char *filename, all *data) {
	int i, j, k, count, m, n, pos, count_uniq=0;
	double running;
	sample *current;
	FILE *f_in;
	char c;
	
	f_in=fopen(filename, "r"); // /home/simon/MPF_CMU/test.txt
	fscanf(f_in, "%i\n", &m); // number of samples
	fscanf(f_in, "%i\n", &n); // number of nodes
	data->m=m;
	data->n=n;
	global_length=data->n;
	
	data->obs_raw=(sample **)malloc(data->m*sizeof(sample *));
	
	// first, read in the raw data
	for(i=0;i<data->m;i++) {
		data->obs_raw[i]=(sample *)malloc(sizeof(sample));
		data->obs_raw[i]->config_base=(int *)malloc(data->n*sizeof(int)); // base configuration (-1, +1, 0)
		data->obs_raw[i]->n_blanks=0;
		data->obs_raw[i]->blanks=NULL;
		
		for(j=0;j<n;j++) {
			data->obs_raw[i]->config_base[j]=-100;
			fscanf(f_in, "%c", &c);
			if (c == '0') {
				data->obs_raw[i]->config_base[j]=-1;
			}
			if (c == '1') {
				data->obs_raw[i]->config_base[j]=1;
			}
			if (c == 'X') {
				data->obs_raw[i]->config_base[j]=0;
				data->obs_raw[i]->blanks = (int *)realloc(data->obs_raw[i]->blanks, (data->obs_raw[i]->n_blanks+1)*sizeof(int));
				data->obs_raw[i]->blanks[data->obs_raw[i]->n_blanks] = j;
				data->obs_raw[i]->n_blanks++;
			}
			if (data->obs_raw[i]->config_base[j] == -100) {
				printf("Bad entry for node %i of obs %i; entry was %c\n", j, i, c);
			}
		}
		fscanf(f_in, "%lf", &(data->obs_raw[i]->mult)); // read in multiplicity
		// data->obs_raw[i]->mult *= (data->n - data->obs_raw[i]->n_blanks)*1.0/data->n; // experiment -- test if downweighting X configs helps
		fscanf(f_in, "%c", &c);
		if ((c != '\n') && (i < (m-1))) {
			printf("Expected an end of line, didn't get one\n");
		}
	}
	fclose(f_in);
	
}

void process_obs_raw(all *data) {
	int i, j, k, count, m, n, pos, count_uniq=0;
	double running;
	sample *current;

	// assumed we have obs_raw all read in
	
	// now, clean this for duplicates...
	// we are going to sort the obs_raw data, on the basis of the "config_base"
	qsort(data->obs_raw, data->m, sizeof(sample *), compare_states_base); 

	count_uniq=1;
	i=1;
	while(i<(data->m)) {
		if (compare_states_base(&(data->obs_raw[i]), &(data->obs_raw[i-1])) != 0) {
			count_uniq++;
		}
		i++;
	}
	data->uniq=count_uniq;

	// now make the big set...
	data->obs=(sample **)malloc(data->uniq*sizeof(sample *));
	for(i=0;i<data->uniq;i++) {
		data->obs[i]=(sample *)malloc(sizeof(sample));
		data->obs[i]->config_base=(int *)malloc(data->n*sizeof(int));
		data->obs[i]->blanks=NULL;
	}
	
	// now we create the new "obs" list -- this is the one we work with...
	i=0;
	pos=1;
	for(j=0;j<data->n;j++) {
		data->obs[0]->config_base[j]=data->obs_raw[0]->config_base[j];
	}
	data->obs[0]->mult = data->obs_raw[0]->mult;
	data->obs[0]->n_blanks = data->obs_raw[0]->n_blanks;
	data->obs[0]->blanks=(int *)realloc(data->obs[0]->blanks, data->obs[0]->n_blanks*sizeof(int));
	for(j=0;j<data->obs[i]->n_blanks;j++) {
		data->obs[0]->blanks[j]=data->obs_raw[0]->blanks[j];
	}
	
	while(pos<data->m) {		
		if (compare_states_base(&(data->obs_raw[pos]), &(data->obs_raw[pos-1])) != 0) { // if the current one is different from the previous one then...
			i++; // increment the counter...
			for(j=0;j<data->n;j++) {
				data->obs[i]->config_base[j]=data->obs_raw[pos]->config_base[j];
			}
			data->obs[i]->mult = data->obs_raw[pos]->mult;
			data->obs[i]->n_blanks = data->obs_raw[pos]->n_blanks;
			data->obs[i]->blanks=NULL;
			data->obs[i]->blanks=(int *)realloc(data->obs[i]->blanks, data->obs[i]->n_blanks*sizeof(int));
			for(j=0;j<data->obs[i]->n_blanks;j++) {
				data->obs[i]->blanks[j]=data->obs_raw[pos]->blanks[j];
			}
		} else {
			data->obs[i]->mult += data->obs_raw[pos]->mult;
		}
		pos++;
	}
		
	// now we have to go through for each of the observations in the full list and construct the remainder of the data
	data->max_config=1;
	data->n_all=0;
	for(i=0;i<data->uniq;i++) {
		current=data->obs[i];
		current->n_config=(1 << current->n_blanks);
		if (current->n_config > data->max_config) {
			data->max_config=current->n_config;
			data->max_blanks=current->n_blanks;
		}
		current->config=(unsigned long int *)malloc(current->n_config*sizeof(unsigned long int));
		current->mult_sim=(double *)malloc(current->n_config*sizeof(double));
		current->mult_sim_pointer=(double **)malloc(current->n_config*sizeof(double *));
		if (current->n_blanks == 0) {
			current->config[0]=convert(current->config_base);
			current->mult_sim[0]=1;
		} else {
			for(j=0;j<current->n_config;j++) {
				for(k=0;k<current->n_blanks;k++) {
					current->config_base[current->blanks[k]] = VAL(j, k);
				}
				current->config[j]=convert(current->config_base);
				current->mult_sim[j]=1.0/(current->n_config);
				current->mult_sim_pointer[j]=NULL;
			}
		}

		for(k=0;k<current->n_blanks;k++) { // just to be careful, we'll reset the config-base
			current->config_base[current->blanks[k]] = 0;
		}
		data->n_all += current->n_config;
	}

	data->ei=(double *)malloc(data->n_all*sizeof(double));	

	data->h_offset=data->n*(data->n-1)/2;
	data->n_params=data->n*(data->n-1)/2+data->n;

}

void clear_data(all *data) {
	int i, j;
	
	free(data->big_list);
	for(i=0;i<data->uniq;i++) {
		for(j=0;j<data->obs[i]->n_config;j++) {
			
		}
		free(data->obs[i]);	
	}
	for(i=0;i<data->uniq;i++) {
		free(data->obs[i]);	
	}
	
}

void init_params(all *data) {
	int i, j, d, count;
	double running;
	gsl_rng *r;
	
	if (data->n_params < 0) {
		data->n_params=data->n*(data->n+1)/2;
		data->h_offset=data->n*(data->n-1)/2;
	}
	
	if (data->big_list == NULL) {
		data->big_list=(double *)malloc(data->n_params*sizeof(double));
		data->old_list=(double *)malloc(data->n_params*sizeof(double));
		data->dk=(double *)malloc(data->n_params*sizeof(double));
		r=data->r;
		
		data->ij=(int **)malloc(data->n*sizeof(int *));
		for(i=0;i<data->n;i++) {
			data->ij[i]=(int *)malloc(data->n*sizeof(int));
		}
		count=0;
		for(i=0;i<(data->n-1);i++) {
			for(j=(i+1);j<data->n;j++) {
				data->ij[i][j]=count;
				data->ij[j][i]=count;
				count++;
			}
		}
	}

	if (data->best_fit == NULL) {
		for(i=0;i<data->n_params;i++) {
			data->big_list[i]=gsl_ran_gaussian(data->r, 1.0)/100.0; // initiatize for tests
			data->old_list[i]=0; // initiatize for tests
		}		
	} else {
		for(i=0;i<data->n_params;i++) {
			data->big_list[i]=data->best_fit[i]; // initiatize for tests
			data->old_list[i]=0; // initiatize for tests
		}
	}

	global_length=data->n;
}

void create_near(all *data, int n_step) { // creates nearest neighbours, removes duplicates
	int i, j, temp_k, k, kp, kpp, count, found, pos, num_near, count_uniq, dist, t_count;
	double running;
	unsigned long int *near_temp;
	
	data->near=NULL;	
	data->near_set=NULL;
	
	data->n_prox=0;
	if (n_step >= 1) {	
		data->n_prox += data->n;
	}
	// if (n_step >= 2) {
	// 	data->n_prox += data->n*(data->n-1)/2;
	// }
	// if (n_step >= 3) {
	// 	data->n_prox += data->n*(data->n-1)*(data->n-2)/6;
	// }
	
	for(i=0;i<data->uniq;i++) {
		data->obs[i]->prox=(int **)malloc(data->obs[i]->n_config*sizeof(int *));
		for(j=0;j<data->obs[i]->n_config;j++) {
			data->obs[i]->prox[j]=(int *)malloc((data->n_prox-data->obs[i]->n_blanks)*sizeof(int));
		}
	}
	
	count=0;
	if (n_step >= 1) {	
			
		for(i=0;i<data->uniq;i++) {
			
			for(j=0;j<data->obs[i]->n_config;j++) { // do the total number of configurations here... now if we excluded "neighbours" in the simulated bits, what happens?

				// data->near_set=(near_struct **)realloc(data->near_set, (count+(data->n)*sizeof(near_struct *));
				data->near_set=(near_struct **)realloc(data->near_set, (count+(data->n-data->obs[i]->n_blanks))*sizeof(near_struct *));
				temp_k=0;
				for(k=0;k<data->n;k++) {
					// little trick -- don't flow into something that you are simulating... let's see what happens to fit quality
					// hmm -- makes it much worse, it seems
					found=0;
					for(kpp=0;kpp<data->obs[i]->n_blanks;kpp++) {
						if (k == data->obs[i]->blanks[kpp]) {
							found=1;
							break;
						}
					}
					if (found == 0) {
						data->near_set[count]=(near_struct *)malloc(sizeof(near_struct));
						data->near_set[count]->config=(data->obs[i]->config[j] ^ (1 << k)); // XOR at that bit to flip it
						data->near_set[count]->data_prox=&(data->obs[i]->prox[j][temp_k]); // pointer to the proximate location
						count++;						
						temp_k++;
					} else {
						// data->near_set[count]->config=gsl_rng_uniform_int(data->r, (1 << data->n)); // choose randomly!
					}
				}				
			}
			
		}

	}
	
	// if (n_step >= 2) {
	//
	// 	for(i=0;i<data->uniq;i++) {
	// 		// printf("Datapoint %i has %i configs.\n", i, data->obs[i]->n_config);
	//
	// 		for(j=0;j<data->obs[i]->n_config;j++) {
	//
	// 			data->near_set=(near_struct **)realloc(data->near_set, (count+data->n*(data->n-1)/2)*sizeof(near_struct *));
	// 			t_count=0;
	// 			for(k=0;k<(data->n-1);k++) {
	// 				for(kp=(k+1);kp<(data->n);kp++) {
	// 					data->near_set[count]=(near_struct *)malloc(sizeof(near_struct));
	// 					data->near_set[count]->config=((data->obs[i]->config[j] ^ (1 << k)) ^ (1 << kp)); // XOR at that bit to flip it
	// 					// printf("Setting %i pointer to %i %i %i\n", count, i, j, data->n+t_count);
	// 					data->near_set[count]->data_prox=&(data->obs[i]->prox[j][data->n+t_count]); // pointer to the proximate location
	// 					count++;
	// 					t_count++;
	// 				}
	// 			}
	// 		}
	//
	// 	}
	//
	// }
	//
	// if (n_step >= 3) {
	//
	// 	for(i=0;i<data->uniq;i++) {
	// 		// printf("Datapoint %i has %i configs.\n", i, data->obs[i]->n_config);
	//
	// 		for(j=0;j<data->obs[i]->n_config;j++) {
	//
	// 			data->near_set=(near_struct **)realloc(data->near_set, (count+data->n*(data->n-1)*(data->n-2)/6)*sizeof(near_struct *));
	// 			t_count=0;
	// 			for(k=0;k<(data->n-2);k++) {
	// 				for(kp=(k+1);kp<(data->n-1);kp++) {
	// 					for(kpp=(kp+1);kpp<(data->n);kpp++) {
	// 						data->near_set[count]=(near_struct *)malloc(sizeof(near_struct));
	// 						data->near_set[count]->config=(((data->obs[i]->config[j] ^ (1 << k)) ^ (1 << kp))) ^ (1 << kpp); // XOR at that bit to flip it
	// 						data->near_set[count]->data_prox=&(data->obs[i]->prox[j][data->n+data->n*(data->n-1)/2+t_count]); // pointer to the proximate location
	// 						count++;
	// 						t_count++;
	// 					}
	// 				}
	// 			}
	// 		}
	//
	// 	}
	//
	// }
	num_near=count;
	qsort(data->near_set, num_near, sizeof(near_struct *), compare_states_near);
	
	count_uniq=1;
	i=1;
	while(i<num_near) {
		if (compare_states_near(&(data->near_set[i]), &(data->near_set[i-1])) != 0) {
			count_uniq++;
		}
		i++;
	}
	near_temp=(unsigned long int *)malloc(count_uniq*sizeof(unsigned long int));
	data->near_uniq=count_uniq;
	
	i=0;
	pos=1;
	near_temp[0]=data->near_set[0]->config;
	(*data->near_set[0]->data_prox)=0;
	while(pos<num_near) {
		if (compare_states_near(&(data->near_set[pos]), &(data->near_set[pos-1])) != 0) { // if the current one is different from the previous one then...
			i++; // increment the counter...
			near_temp[i]=data->near_set[pos]->config; // save the new one...
		}
		(*data->near_set[pos]->data_prox)=i; // whatever the current position in near, assign it to the current value of near_temp
		pos++;
	}

	for(i=0;i<num_near;i++) {
		free(data->near_set[i]);
	}
	free(data->near_set);
	data->near=near_temp;
	
	// for(i=0;i<data->uniq;i++) {
	// 	for(j=0;j<data->obs[i]->n_config;j++) {
	// 		print_vec(data->obs[i]->config[j]);
	// 		printf(" has neighbours... (%i)\n", data->obs[i]->n_blanks);
	//
	// 		for(k=0;k<data->n_prox-data->obs[i]->n_blanks;k++) {
	// 			printf("%i %i\n", k, data->obs[i]->prox[j][k]);
	// 			print_vec(data->near[data->obs[i]->prox[j][k]]);
	// 			printf("\n");
	// 		}
	// 	}
	// }
	
	// compute the sparsity
	update_sparsity(data);

	data->nei=(double *)malloc(data->near_uniq*sizeof(double));
}

void update_sparsity(all *data) {
	int i;
	double running;
	
	data->sparsity=data->n_prox*exp(data->log_sparsity*log(10));
	running=0;
	for(i=0;i<data->uniq;i++) {
		running += data->obs[i]->mult; //*(data->n-data->obs_raw[i]->n_blanks)*(data->n-data->obs_raw[i]->n_blanks+1)*1.0/(data->n*(data->n+1));
	}
	data->sparsity *= running;
	data->sparsity *= (1.0/data->n_params);
}

void update_mult_sim(all *data) {
	int i, ip, jp, loc, loc_true, f, fixed, comb;
	double *fields, *z, z_tot, max_val=0;
	
	fields=(double *)alloca(data->n*sizeof(double));
	z=(double *)alloca(data->max_config*sizeof(double));
		
	for(i=0;i<data->uniq;i++) {
		if (data->obs[i]->n_config > 1) {
			
			// first, we're going to compute the influence of the fixed variables on the blanked out data
			for(loc=0;loc<data->obs[i]->n_blanks;loc++) { // go through all the blank positions...
				fields[loc]=0;
				loc_true=data->obs[i]->blanks[loc]; // actual node position
				
				for(ip=0;ip<data->n;ip++) { // compute the off diagonal terms... all the nodes...
					if (ip != loc_true) { // except the current blank one...
						
						fixed=1;
						for(f=0;f<data->obs[i]->n_blanks;f++) {
							if (data->obs[i]->blanks[f] == ip) { // and except the other blank ones...
								fixed=0;
								break;
							}
						}
						if (fixed) {
							fields[loc] += data->big_list[data->ij[ip][loc_true]]*VAL(data->obs[i]->config[0], ip); // contribute to the field on the loc_true node
						}
					}
				}
				fields[loc] += data->big_list[data->h_offset+loc_true];
			}
			
			// then, we are going to compute the rest of the energy... we are going to cycle through each of the combinations...
			max_val=-1e20;
			for(comb=0;comb<data->obs[i]->n_config;comb++) { // cycle through all the configurations of the blanks...
				z[comb]=0;
				// VAL(comb,ip) **should** be equal to VAL(data->obs[i]->config[comb],data->obs[i]->blanks[ip])
				// for(ip=0;ip<data->obs[i]->n_blanks;ip++) {
				// 	printf("(%lf == %lf) ", VAL(comb,ip), VAL(data->obs[i]->config[comb],data->obs[i]->blanks[ip]));
				// }
				// printf("\n");
				for(ip=0;ip<data->obs[i]->n_blanks;ip++) { // for all the blank nodes...
					z[comb] += fields[ip]*VAL(comb,ip); // first, add in all the external forcings...
					for(jp=(ip+1);jp<data->obs[i]->n_blanks;jp++) { // then you have to do the cross terms between the blank nodes...
						z[comb] += VAL(comb,ip)*VAL(comb,jp)*data->big_list[data->ij[data->obs[i]->blanks[ip]][data->obs[i]->blanks[jp]]];
					}
				}
				if (z[comb] > max_val) {
					max_val=z[comb];
				}
			}
			z_tot=0;
			for(comb=0;comb<data->obs[i]->n_config;comb++) {
				z[comb]=exp(z[comb]-max_val);
				z_tot += z[comb];
			}			
			for(comb=0;comb<data->obs[i]->n_config;comb++) {
				data->obs[i]->mult_sim[comb] = z[comb]/z_tot;
			}
							
		} else {
			data->obs[i]->mult_sim[0]=1.0;
		}
	}
	
}

double cross(char *filename, double log_sparsity, int nn, double *best_fit) {
	all *data;
	double glob_nloops, logl_ans;
	int i, thread_id, last_pos, in, j, count, pos, n_obs, n_nodes, kfold, num_no_na;
	unsigned long int config;
	sample *sav;
	double t0;
	
	data=new_data();
	read_data(filename, data);
	
	// here's what we'll do -- we'll cycle through a bunch of samples where we leave out one data point
	num_no_na=0;
	for(i=0;i<data->m;i++) {
		if (data->obs_raw[i]->n_blanks == 0) {
			num_no_na++;
		}
	}
	printf("%i observations can be cross-validated.\n", num_no_na);
	// if (num_no_na > 128) { // for the Pittsburgh Supercomputer Center, each node has a max of 128 cores, so let's restrict to this for the most efficient use of computer time
	// 	num_no_na=128;
	// }
	
	glob_nloops=0;
	t0=clock();
#pragma omp parallel private(data, pos, last_pos, count, sav, config, logl_ans, thread_id) reduction(+:glob_nloops)
	{

		// parallelize this for loop
#pragma omp for
		for(in=0;in<num_no_na;in++) {
			data=new_data();
			read_data(filename, data);
			data->best_fit=best_fit; // will either be NULL or a best guess
			
			data->m = data->m-1; // remove one data point

			pos=0;
			last_pos=0;
			count=0;
			while(count < (in+1)) {
				if(data->obs_raw[pos]->n_blanks == 0) { // if you see a good one, count it
					count++;
					last_pos=pos;
				}
				pos++; // move forward one unit
			}
			pos=last_pos;

			sav=data->obs_raw[pos]; // the pointer to the data we'll leave out
			data->obs_raw[pos]=data->obs_raw[data->m]; //
			data->obs_raw[data->m]=sav;
			
			process_obs_raw(data);				
			init_params(data);
			data->log_sparsity=log_sparsity;
			create_near(data, nn);

			simple_minimizer(data);

			config=0;
			for(i=0;i<data->n;i++) {
				if (data->obs_raw[data->m]->config_base[i] > 0) {
					config += (1 << i);
				}
			}
			if (data->n <= 20) {
				logl_ans=log_l(data, config, data->big_list, 0);
			} else {
				logl_ans=log_l(data, config, data->big_list, 1);
			}

			glob_nloops += logl_ans;
			thread_id = omp_get_thread_num();
			// printf("LogL of left-out point %i, computed in thread %i: %lf\n", in, thread_id, logl_ans);
		}
		
	}
	printf("For log-sparsity=%lf, Log-l of held-out data is: %lf\n", log_sparsity, glob_nloops*1.0/num_no_na);
	printf("Clock time for one iteration: %14.12lf seconds.\n", (clock() - t0)/CLOCKS_PER_SEC);
	
	return glob_nloops*1.0/num_no_na;
}

void compute_k_general(all *data, int do_derivs) {
	int d, dp, k, a, i, j, f, ip, jp, loc_p, n, count, term, loc, d_count, fixed, changed;
	int **ij;
	unsigned long int config1, config2;
	double **obs, **cross_terms;
	double max_val, min_val, *ei, energy, running, big_running, *big_running_k, *running_k, multiplier;
		
	ij=data->ij; // save typing
	if (do_derivs == 1) {
		for(i=0;i<data->n_params;i++) {
			data->dk[i]=0;
		}
		cross_terms=(double **)alloca(data->n*sizeof(double *));
		for(i=0;i<data->n;i++) {
			cross_terms[i]=(double *)alloca(data->n*sizeof(double));
		}
		running_k=(double *)alloca(data->n_params*sizeof(double));
		big_running_k=(double *)alloca(data->n_params*sizeof(double));
	}	
	data->k=0;
		
	// in these cases, we'll need to fill in missing data
	if (data->max_config > 1) { 
		changed=0;
		for(i=0;i<data->n_params;i++) {
			if (data->big_list[i] != data->old_list[i]) {
				changed=1;
				break;
			}
		}
		if (changed) {
			update_mult_sim(data);
		}
		for(i=0;i<data->n_params;i++) {
			data->old_list[i]=data->big_list[i];
		}
	}
	
	d_count=0;
	for(d=0;d<data->uniq;d++) { // for each unique datapoint...
		for(dp=0;dp<data->obs[d]->n_config;dp++) { // for each of the configurations
			data->ei[d_count]=0;
			
			count=0;
			for(i=0;i<data->n;i++) {
				for(j=(i+1);j<data->n;j++) {
					data->ei[d_count] += VAL(data->obs[d]->config[dp],i)*VAL(data->obs[d]->config[dp],j)*data->big_list[count];
					count++;
				}
				data->ei[d_count] += VAL(data->obs[d]->config[dp],i)*data->big_list[data->h_offset+i]; // local fields
			}
				
			data->ei[d_count] *= -1; // defined as the negative value in Jascha paper
			d_count++;
		}
	}
	
	for(d=0;d<data->near_uniq;d++) { // for each unique nearest neighbour...
		data->nei[d]=0;
		count=0;
		for(i=0;i<data->n;i++) {
			for(j=(i+1);j<data->n;j++) {
				data->nei[d] += VAL(data->near[d],i)*VAL(data->near[d],j)*data->big_list[count];
				count++;
			}
			data->nei[d] += VAL(data->near[d],i)*data->big_list[data->h_offset+i]; // local fields
		}
		data->nei[d] *= -1; // defined as the negative value in Jascha paper
	}
	
	d_count=0;
	for(d=0;d<data->uniq;d++) {

		if ((do_derivs == 1) && (data->obs[d]->n_config > 1)) {
			// compute the expectation value of the pair
			for(i=0;i<data->n;i++) {
				for(j=(i+1);j<data->n;j++) {
					cross_terms[i][j]=0;
					cross_terms[j][i]=0;
					for(k=0;k<data->obs[d]->n_config;k++) {
						cross_terms[i][j] += VAL(data->obs[d]->config[k],i)*VAL(data->obs[d]->config[k],j)*data->obs[d]->mult_sim[k];
						cross_terms[j][i] += VAL(data->obs[d]->config[k],i)*VAL(data->obs[d]->config[k],j)*data->obs[d]->mult_sim[k];
					}						
				}
				cross_terms[i][i]=0;
				for(k=0;k<data->obs[d]->n_config;k++) {
					cross_terms[i][i] += VAL(data->obs[d]->config[k],i)*data->obs[d]->mult_sim[k];
				}
				
			}
						
		}
		
		big_running=0;
		if (do_derivs == 1) {
			for(i=0;i<data->n_params;i++) {
				big_running_k[i]=0;
			}				
		}
		
		for(dp=0;dp<data->obs[d]->n_config;dp++) {
			config1=data->obs[d]->config[dp];
			
			max_val=-1e300;
			for(n=0;n<data->n_prox-data->obs[d]->n_blanks;n++) { // TKTK
				if ((data->ei[d_count]-data->nei[data->obs[d]->prox[dp][n]]) > max_val) {
					max_val=(data->ei[d_count]-data->nei[data->obs[d]->prox[dp][n]]);
				}
			}	
			max_val=max_val/2.0;
			running=0;
			if (do_derivs == 1) {
				for(i=0;i<data->n_params;i++) {
					running_k[i]=0;
				}				
			}
			for(n=0;n<data->n_prox-data->obs[d]->n_blanks;n++) { // TKTK
				loc=data->obs[d]->prox[dp][n];
				config2=data->near[loc];
				multiplier=data->obs[d]->mult*data->obs[d]->mult_sim[dp]*exp(0.5*(data->ei[d_count]-data->nei[loc])-max_val); 
				running += multiplier;

				// hard part -- do the derivatives
				if (do_derivs == 1) {
					count=0;
					for(ip=0;ip<(data->n);ip++) {
						for(jp=(ip+1);jp<data->n;jp++) {
							running_k[count] += -1*(VAL(config1,ip)*VAL(config1,jp)-VAL(config2, ip)*VAL(config2, jp))*multiplier/2.0;  
							// defined as the negative value in Jascha paper -- BUT: note that Jascha convention is different by a factor of 1/2, Eddie fixed it
							count++;
						}
					}
					for(ip=0;ip<(data->n);ip++) {
						running_k[data->h_offset+ip] += -1*(VAL(config1,ip)-VAL(config2,ip))*multiplier/2.0;  // defined as the negative value in Jascha paper
					}
					
					for(ip=0;ip<data->obs[d]->n_blanks;ip++) {
						loc_p=data->obs[d]->blanks[ip];
						running_k[data->h_offset+loc_p] += multiplier*(VAL(config1,loc_p) - cross_terms[loc_p][loc_p]);
						for(jp=0;jp<data->n;jp++) {
							if (loc_p != jp) {
								
								fixed=1;
								for(f=0;f<data->obs[d]->n_blanks;f++) {
									if (data->obs[d]->blanks[f] == jp) { 
										fixed=0;
										break;
									}
								}
								if (fixed) {
									running_k[ij[loc_p][jp]] += multiplier*(VAL(config1,loc_p)*VAL(config1,jp) - cross_terms[loc_p][jp]); 
								} else {
									if (loc_p > jp) { // if they are both in the blank set, don't double-count OMG THIS TOOK ME A WEEK TO FIND
										running_k[ij[loc_p][jp]] += multiplier*(VAL(config1,loc_p)*VAL(config1,jp) - cross_terms[loc_p][jp]); 
									}
								}
							}
						}
					}
				}	
			}
			big_running += running*exp(max_val);
			if (do_derivs == 1) {
				for(i=0;i<data->n_params;i++) {
					big_running_k[i] += running_k[i]*exp(max_val);
				}				
			}
			d_count++;
		}
		data->k += big_running;
		if (do_derivs == 1) {
			for(i=0;i<data->n_params;i++) {
				data->dk[i] += big_running_k[i];
			}				
		}
		
	}

	for(i=0;i<data->n_params;i++) {
		data->k += data->sparsity*data->big_list[i]*data->big_list[i]/2; // put in a sparse prior...
	}
	
	if (do_derivs == 1) {
		for(i=0;i<data->n_params;i++) {
			data->dk[i] += data->sparsity*data->big_list[i]; // don't forget that this also impacts the derivatives!
		}
	}
	
}

// GSL FUNCTIONS DEFINED
double k_function(const gsl_vector *v, void *params) {
	int i;
	all *data;
	
	data=(all *)params;
	for(i=0;i<data->n_params;i++) {
		data->big_list[i]=gsl_vector_get(v, i);
	}
	compute_k_general(data, 0);
	return data->k;
}

void dk_function(const gsl_vector *v, void *params, gsl_vector *df) {
	int i;
	double ep=1e-3, old;
	all *data;
	
	data=(all *)params;
	for(i=0;i<data->n_params;i++) {
		data->big_list[i]=gsl_vector_get(v, i);
	}
	compute_k_general(data, 1);
	
	for(i=0;i<data->n_params;i++) {
		gsl_vector_set(df, i, data->dk[i]);
	}
}
	
void kdk_function(const gsl_vector *x, void *params, double *f, gsl_vector *df) {
    *f = k_function(x, params);
    dk_function(x, params, df);
}

void simple_minimizer(all *data) {
    size_t iter = 0;
	double prev, num;
    int i, status;

    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;

	gsl_vector *x;
	gsl_multimin_function_fdf k_func;

	// set up the system
	k_func.n = data->n_params;  /* number of function components */
	k_func.f = &k_function;
	k_func.df = &dk_function;
	k_func.fdf = &kdk_function;
	k_func.params = (void *)data;
	
	x = gsl_vector_alloc(data->n_params);
	for(i=0;i<data->n_params;i++) {
		gsl_vector_set(x, i, data->big_list[i]); //
	}
	T = gsl_multimin_fdfminimizer_conjugate_fr; // 5867.659379
	// T = gsl_multimin_fdfminimizer_conjugate_pr; // 5866.193340 5865.871289 5866.563172 5868.561687
	// T = gsl_multimin_fdfminimizer_vector_bfgs2; // 6118.521483
	s = gsl_multimin_fdfminimizer_alloc(T, data->n_params);
	
	compute_k_general(data, 1);

	gsl_multimin_fdfminimizer_set(s, &k_func, x, 0.01, 1e-6);
	
	prev=1e300;
	do {
		iter++;
		status = gsl_multimin_fdfminimizer_iterate(s);

		status = gsl_multimin_test_gradient(s->gradient, 1e-6);
		
		// if (data->best_fit != NULL) {
			// printf ("%i %li (%lf) : ", status, iter, s->f);
			// for(i=0;i<data->n_params;i++) {
			// 	printf("%.10le ", gsl_vector_get (s->x, i));
			// }
			// printf("\n");
		// }
		// printf("Derivs: ");
		// for(i=0;i<data->n_params;i++) {
		// 	printf("%lf ", data->dk[i]);
		// }
		// printf("\n");
		
		num=0;
		for(i=0;i<data->n_params;i++) {
			num += fabs(data->big_list[i]);
		}
		
		if ((fabs(prev-num) < 1e-16) && ((iter % 20) == 19)) {
			break;
		}
		if ((iter % 20) == 0) {
			prev=num;
		}
	} while (status == GSL_CONTINUE && iter < 3000);

	// compute_k_general(data, 1);
	// printf("FINAL params: ");
	// for(i=0;i<data->n_params;i++) {
	// 	printf("%.10le ", gsl_vector_get (s->x, i));
	// }
	// printf("\n");

	for(i=0;i<data->n_params;i++) {
		data->big_list[i]=gsl_vector_get(s->x, i);
	}
	
	gsl_multimin_fdfminimizer_free(s);
	gsl_vector_free(x);
}
