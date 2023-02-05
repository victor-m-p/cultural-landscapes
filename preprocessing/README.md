<!-- ABOUT THE PROJECT -->
## Overview
* ```preprocessing.py``` converts ```.json``` obtained from the DRH to ```.csv```.
* ```curation.py``` runs data subset curation (relies on ```curation_functions.py```. See also ```run_curation.sh```).
* ```prep_early.py``` creates central dataframes used across the analysis (as well as some of the tables). 
* ```expand_data.jl``` expands the entries with missing or inconsistent data. 
* ```prep_late.py``` combines information from ```prep_early.py``` and ```expand_data.jl```.

## Description
### ```preprocessing.py```
Preprocessing of the raw DRH data. The generated file is not included in this GitHub due to size limitations.
If a full reproduction is desired please contact Victor Poulsen (victormoeller@gmail.com).

### ```curation.py```
Runs data curation on the file created in the previous step (```preprocessing.py```). Uses functions from ```/curation_functions```.
Several datasets are created, but in the final analysis we only use the subset with
20 questions, and 5 unknown values allowed. Data from this step is included in the GitHub.

### ```prep_early.py```
Prepares key files, recodes questions to short-hands. Saves to ```../data/analysis```.

### ```expand_data.jl```
Expands combinations (e.g. when one religion has inconsistent codings or missing values). 
Creates an overview of all "possible" configurations that each religion could have.
Saves to ```../data/analysis```.

### ```prep_late.py```
Final preprocessing of documents. Saves to ```../data/analysis```

## Contact
Victor M. Poulsen 
* Twitter: [@vic_moeller](https://twitter.com/vic_moeller) 
* GitHub: [@victor-m-p](https://github.com/victor-m-p)
* Mail: victormoeller@gmail.com


