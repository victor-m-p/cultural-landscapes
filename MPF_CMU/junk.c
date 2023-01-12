// heroic derivative testing code
			// testing fix here...
			// data->sparsity=0;
			// for(i=0;i<data->n_params;i++) {
			// 	data->big_list[i]=0.2;
			// }
			//
			// compute_k_general(data, 1);
			// for(i=0;i<data->n_params;i++) {
			// 	printf("%lf ", data->dk[i]);
			// }
			// printf("\n");
			//
			// old=data->k;
			// acc=0;
			// ep=atof(argv[5]);
			// for(i=0;i<data->n_params;i++) {
			// 	data->big_list[i] += ep;
			// 	compute_k_general(data, 0);
			// 	old=data->k;
			// 	data->big_list[i] -= 2*ep;
			// 	compute_k_general(data, 0);
			// 	printf("%.5le ", (data->dk[i]-((old-data->k)/(2*ep)))/fabs(data->dk[i]));
			// 	acc += fabs(data->dk[i]-((old-data->k)/(2*ep)))/fabs(data->dk[i]);
			// 	data->big_list[i] += ep;
			// }
			// printf("\n");
			// printf("MEAN ACC: %le\n", acc/data->n_params);
