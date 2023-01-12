## Overview

Pre-review code for "Inferring Cultural Landscapes with the Inverse Ising Model"

Victor Møller Poulsen and Simon DeDeo

submitted to *Entropy*, 9 December 2022

This is *pre-review* code, not intended for general public circulation. We would be very grateful for comments, questions, and thoughts.

We are very happy to take requests. Our experience has been that iterating with potential users is the best way to make our code and data clear, reusable, and efficient to use. On re-submission and review, we will release general code, intended for consumer use, and a full guide.

## Components

### ```/data```
* ```data/mdl_final```: processed DRH data (post-MPF).
* ```data/clean```: processed DRH data (pre-MPF).
* ```data/reference```: reference files (i.e. ensuring links between questions and cultures). 
* ```data/analysis```: files used for the DRH analysis in ```/DRH```. 

Raw data from the DRH (i.e. pre-curation) is not provided given size limits. 
This might be available upon request. 

### ```/fig```
Figures for "Inferring Cultural Landscapes with the Inverse Ising Model". Currently only the figures for ```5. Results: The Database of Religious History``` (see ```/DRH``` for code). 


### ```/tables```
Tables for "Inferring Cultural Landscapes with the Inverse Ising Model". In particular, tables documenting the DRH dataset used in the article. See ```/DRH``` for code. 

### ```/DRH```
```DRH``` contains the ```Python``` (and ```Julia```) code to reproduce the DRH analysis as presented in "Inferring Cultural Landscapes with the Inverse Ising Model". It contains both the preprocessing, data-curation, creation of tables (for ```/tables```) and creation of figures (for ```/fig```). See ```/DRH``` for more details. 

### ```/MPF_CMU``` 

MPF_CMU contains the optimized C code to implement all of the extensions and modifications to MPF described in our paper. Many of the simulations were carried out on the Bridges2 Pittsburgh Supercomputing Center system, and for speed they are set up to use multiple cores with OpenMP. You will want to adjust the Makefile to compile on your local system. Note that the compiler that ships with the new Mac M1s does not support OpenMP; you will need to install a (non-Apple) clang compiler.

Some of this code is compute intensive; on a Mac M1 with 10 cores, for example, it takes a few seconds to fit n=20 with a few hundred observations, and a few minutes to do the same with Cross Validation.

MPF has seven different modes:

// mpf -l [filename] [logsparsity] [NN] // load in data, fit

Given a data file, a log-sparsity choice, and a specification of \mathcal{N}_i, produces a parameter fit file. e.g.,

./mpf -l test.dat 1.0 1

will fit using a lambda equal to 10^1.0, and the \mathcal{N}_1 strategy.

// mpf -c [filename] [NN] // load in data, fit, using cross-validation to pick best sparsity

The same as above, but does cross-validation to find the best value of lambda.

// mpf -g [filename] [n_nodes] [n_obs] [beta] // generate data, save both parameters and data to files

Generates a simulated dataset, with n_nodes nodes, and n_obs observations, with J and h drawn from a Gaussian with RMS equal to beta.

// mpf -t [filename] [paramfile] [NN] // load in test data, fit, get KL divergence from truth

If you know the true parameters (e.g., because you previously ran mpf -g), this will fit data, and tell you how well you did.

// mpf -o [filename_prefix] [NN] // load in data (_data.dat suffix), find best lambda using _params.dat to determine KL

If you know the true parameters, this will tell you the secret best value of lambda.

// mpf -k [filename] [paramfile_truth] [paramfile_inferred] // load data, compare truth to inferred

If you have previously fit some data, but you also know the true parameters, this will tell you how well you did.

// mpf -z [paramfile] [n_nodes]  // print out probabilities of all configurations under paramfile

If you have a parameter file, this will print out all the configurations, the energy for each config, and the probability.

<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Simon DeDeo (for ```/MPF_CMU``` questions):
* Twitter: [@LaboratoryMinds](https://twitter.com/LaboratoryMinds)
* Github: [@simon-dedeo](https://github.com/simon-dedeo)
* Mail: sdedeo@andrew.cmu.edu

Victor Poulsen (for other questions): 
* Twitter: [@vic_moeller](https://twitter.com/vic_moeller) 
* GitHub: [@victor-m-p](https://github.com/victor-m-p)
* Mail: victormoeller@gmail.com


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
We are particularly grateful to the creaters of ```conIII``` and the maintainers and creators of the ```Database of Religious History (DRH)```.

* [ConIII](https://github.com/eltrompetero/coniii)
* [Database of Religious History (DRH)](https://religiondatabase.org/landing/)

## FUNDING

This work used the Extreme Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number ACI-1548562. Specifically, it used the Bridges-2 system, which is supported by NSF award number ACI-1928147, at the Pittsburgh Supercomputing Center (PSC), under grant HUM220003. This work was supported in part by the Survival and Flourishing Fund.
