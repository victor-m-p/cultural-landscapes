#include "mpf.h"
#include <stdio.h>
#include <gsl/gsl_multifit.h>

#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>

double cross_holder(double log_sparsity, void *params) {
	cross_val *cv;
	
	cv=(cross_val *)params;
	return -cross(cv->filename, log_sparsity, cv->nn, cv->best_fit);
}

double kl_holder(double log_sparsity, void *params) {
	cross_val *cv;
	all *data;
	
	cv=(cross_val *)params;
	
	// read in the data...
	data=new_data();
	read_data(cv->filename, data);
	process_obs_raw(data);				
	init_params(data);
	data->log_sparsity=log_sparsity;
	create_near(data, cv->nn);
					
	simple_minimizer(data);
	
	cv->kl_true=full_kl(data, data->big_list, cv->big_list_true);
	return cv->kl_true;
}

double minimize_kl_true(cross_val *cv) {
  int status;
  int iter = 0, max_iter = 20;
  const gsl_min_fminimizer_type *T;
  gsl_min_fminimizer *s;
  
  double m = 1.0;
  double a = -2.0, b = 4.0;
  gsl_function F;

  F.function = &kl_holder;
  F.params = cv;

  gsl_set_error_handler_off(); // living on the edge

  T = gsl_min_fminimizer_brent;
  s = gsl_min_fminimizer_alloc (T);
  gsl_min_fminimizer_set(s, &F, m, a, b);

  printf ("using %s method\n", gsl_min_fminimizer_name (s));

  printf ("%5s [%9s, %9s] %9s %9s\n",
          "iter", "lower", "upper", "min", "err(est)");

  printf ("%5d [%.7f, %.7f] %.7f %.7f\n",
          iter, a, b,
          m, b - a);


  do {
      iter++;
      status = gsl_min_fminimizer_iterate (s);

      m = gsl_min_fminimizer_x_minimum (s);
      a = gsl_min_fminimizer_x_lower (s);
      b = gsl_min_fminimizer_x_upper (s);

      status
        = gsl_min_test_interval (a, b, 0.001, 0.0);

      if (status == GSL_SUCCESS)
        printf ("Converged:\n");

      printf ("%5d [%.7f, %.7f] "
              "%.7f %.7f\n",
              iter, a, b,
              m, b - a);
    }
  while (status == GSL_CONTINUE && iter < max_iter);

	gsl_set_error_handler(NULL);

  gsl_min_fminimizer_free(s);
    
  return m;
}

double minimize_kl(cross_val *cv, int fast_version) {
  int status;
  int iter = 0, max_iter = 20;
  const gsl_min_fminimizer_type *T;
  gsl_min_fminimizer *s;
  
  double m = 1.0;
  double a = -2.0, b = 4.0;
  gsl_function F;

  F.function = &cross_holder;
  F.params = cv;

  gsl_set_error_handler_off(); // living on the edge

  if (fast_version == 1) {
	  T=gsl_min_fminimizer_brent; 
  } else {
	  T=gsl_min_fminimizer_goldensection;  	
  }
  s = gsl_min_fminimizer_alloc (T);
  gsl_min_fminimizer_set(s, &F, m, a, b);

  printf ("using %s method\n", gsl_min_fminimizer_name (s));

  printf ("%5s [%9s, %9s] %9s %9s\n",
          "iter", "lower", "upper", "min", "err(est)");

  printf ("%5d [%.7f, %.7f] %.7f %.7f\n",
          iter, a, b,
          m, b - a);


  do {
      iter++;
      status = gsl_min_fminimizer_iterate (s);

      m = gsl_min_fminimizer_x_minimum (s);
      a = gsl_min_fminimizer_x_lower (s);
      b = gsl_min_fminimizer_x_upper (s);

      status
        = gsl_min_test_interval (a, b, 0.001, 0.0);

      if (status == GSL_SUCCESS)
        printf ("Converged:\n");

      printf ("%5d [%.7f, %.7f] "
              "%.7f %.7f\n",
              iter, a, b,
              m, b - a);
    }
  while (status == GSL_CONTINUE && iter < max_iter);

	gsl_set_error_handler(NULL);

  gsl_min_fminimizer_free(s);

  return m;
}
