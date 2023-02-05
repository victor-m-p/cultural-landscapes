<!-- TABLE OF CONTENTS -->
## Overview
This folder documents the the analysis of DRH data reported in "Inferring Cultural Landscapes with the Inverse Ising Model" (mainly Section 5). Code-base currently being cleaned and made reproducible. 

* Table creation. 
* Figures 3 and 4. 


<!-- ABOUT THE PROJECT -->
## Files

* ```preprocessing.py``` converts ```.json``` obtained from the DRH to ```.csv```.
* ```curation.py``` runs data curation before ```MPF``` (relies on ```curation_functions.py```. See also ```run_curation.sh```).
* ```prep_early.py``` creates central dataframes used across the analysis (as well as some of the tables). 
* ```expand_data.jl``` expands the entries with missing or inconsistent data. 
* ```prep_late.py``` combines information from ```prep_early.py``` and ```expand_data.jl```.
* ```plot_parameters.py``` creates figure 3A and figure 3B (currently broken because of ongoing rework). 
* ```plot_landscape.py``` creates figure 4A and several tables (relies on ```prep_landscape.py```. 
* ```seed_methodist.py``` creates figure 4B (currently broken because of ongoing rework).
* ```seed_roman.py``` creates figure 4C (currently broken becaues of ongoing rework). 

<!-- GETTING STARTED -->
## Getting Started

Environments tested on ubuntu version 22.04 LTS. 

### Requirements 

Working installation of ```Python``` (tested with v3.10.6) and ```Julia``` (tested with vXXX).

### Installation


1. Clone the repo (if not already done). Here shown for ```ssh``` but ```https``` also fine:
    ```sh
    git clone git@github.com:victor-m-p/humanities-glass.git
    ```

2. Install the ```Python``` environment (```glassenv```):
    ```sh
    bash create_venv.sh
    bash add_venv.sh
    ```

3. Install the ```Julia``` environment  
TODO: figure out how to make Julia environment, and path management in ```Julia```. 

## Main part missing here 

<!-- CONTACT -->
## Contact

Victor M. Poulsen 
* Twitter: [@vic_moeller](https://twitter.com/vic_moeller) 
* GitHub: [@victor-m-p](https://github.com/victor-m-p)
* Mail: victormoeller@gmail.com
