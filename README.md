## Overview

![](fig/svg/landscape_dendrogram_labels.svg)

Code for "Inferring Cultural Landscapes with the Inverse Ising Model" (https://www.mdpi.com/1099-4300/25/2/264) accepted in *Entropy* on the 25th of January 2023.
We would be very grateful for comments, questions, and thoughts. Parts of the analysis are easily reproducible, but a full reproduction is currently not possible since the raw dataset exceeds GitHub size limits. If a full reproduction is desired, please contact the authors. 

```
@article{poulsen2023inferring,
  title={Inferring Cultural Landscapes with the Inverse Ising Model},
  author={Poulsen, Victor M{\o}ller and DeDeo, Simon},
  journal={Entropy},
  volume={25},
  number={2},
  pages={264},
  year={2023},
  publisher={MDPI}
}
```

## 2026 Claude/Codex Analysis

In November 2026 the code and methods of the paper were re-examined by Claude (Anthropic), with an independent review by Codex (OpenAI). This repository now holds the revised code; the code as published is in the git history (commit `23dce37` and earlier), and `mpf --legacy` reproduces the published fit. In brief:

* **The published results reproduce**: the parameter file to 2e-6, and all six tables byte for byte.
* **Partial-MPF, the paper's treatment of missing answers, can be improved** and the "slight tension" noted by Section 2.3.5 can be resolved. Two corrected versions are provided (`--marginal`, `--frozen`). At the paper's sample sizes the three agree closely, so the paper's simulation results and DRH parameters are not materially affected.
* **Blanks in the DRH are related to other answers** (e.g. entries with a blank report scriptures far less often). Of the three versions only `--frozen` has a justification that allows for this, and only if blanks depend on answers that are observed.
* **Two- and three-flip neighbourhoods (N₂, N₃) are implemented again.** N₂ lowers the KL divergence by 8% and 20% in the two strongly coupled simulation regimes, contrary to the paper's remark that it "does not improve significantly"; it makes no difference in the weakly coupled regimes, where the DRH sits.
* **The DRH landscape is stable, its division into three groups is not.** Refits with `--frozen` at N₁, N₂ and N₃ keep the 15 strongest couplings, 130–136 of the 150 most probable configurations, and the local landscapes of the Free Methodist Church and the Roman Imperial Cult. The clustering behind Figure 4A and Tables 5, A2–A4 changes as much when 149 or 151 configurations are clustered instead of 150 as it does under a refit.
* **Software**: defects fixed (none changes a number in the paper that could be checked), fits 50–150 times faster, a test suite, and one command to regenerate every figure and table from a parameter file. The MPF program is now a self-contained directory, [MPF_CMU/src](MPF_CMU/src), with no external libraries.

The full account, including the errors found, which of them reach the paper, the evidence for each statement, and the limits of this analysis, is in [CLAUDE_IMPROVEMENTS_SUMMARY.md](CLAUDE_IMPROVEMENTS_SUMMARY.md). The regenerated analyses are in [regenerated/](regenerated/).

## Comparison with bgms (2026)

[bgms](https://cran.r-project.org/package=bgms) is an R package from Maarten Marsman's group, first released in April 2023, three months after our paper. For yes/no data it fits the same Ising model as `mpf`, by a different route: it is Bayesian, it uses a pseudo-likelihood sampled by MCMC instead of minimising probability flow, and it regularises with a fixed prior instead of a penalty chosen by cross-validation. It can also estimate, for each coupling, the probability that it is present at all ("edge selection", on by default), and it treats missing answers either by dropping incomplete rows (its default) or by imputing them while it samples.

We compared bgms 0.2.0.0 with the revised `mpf` on identical data, giving each 4 cores. On simulated data the measure is the KL divergence from the true distribution (lower is better); for bgms it is computed for the posterior predictive distribution, which is the more favourable of the two ways of scoring it.

**Simulated data, 20 questions** (16 landscapes per regime; the DRH sits in the weakly coupled regimes on the left):

| | dispersed | ordered | near-critical | critical |
|---|---|---|---|---|
| *128 complete rows* | | | | |
| `mpf`, cross-validated | **0.21** | **0.56** | **0.60** | 0.59 |
| bgms, defaults | 0.64 | 0.91 | 0.73 | **0.39** |
| bgms, no edge selection | 1.09 | 1.08 | 1.03 | 0.40 |
| bgms, prior set to the true scale of the couplings (an oracle choice) | 0.29 | 0.65 | 0.72 | 0.50 |
| *128 complete rows + 128 rows with 5 of 20 answers missing* | | | | |
| `mpf --marginal`, cross-validated | **0.21** | **0.39** | **0.47** | 0.47 |
| bgms, imputing | 0.39 | 0.56 | 0.53 | **0.32** |
| bgms, default (drops incomplete rows) | 0.69 | 0.92 | 0.82 | 0.46 |

**Data shaped like the DRH** (40 datasets of 407 rows drawn from a fitted DRH model, with the real patterns of missing answers):

| | blanks at random | blanks depend on another answer |
|---|---|---|
| `mpf --frozen`, cross-validated | **0.21** | **0.21** |
| `mpf --marginal`, cross-validated | **0.21** | **0.21** |
| bgms, imputing | 0.31 | 0.30 |
| bgms, imputing, no edge selection | 0.41 | 0.40 |
| bgms, default (drops incomplete rows) | 0.58 | 0.75 |

**The real DRH data** (407 entries, one row each, an inconsistently coded answer treated as missing because bgms has no row weights). There is no truth to compare with, so entries were held out in a 10-fold cross-validation and scored by the log-probability of their observed answers (higher is better):

| | held-out log-probability per entry | difference from `mpf --frozen` | time for one fit, 4 cores |
|---|---|---|---|
| `mpf --frozen` | −6.85 | — | 6 min |
| `mpf --marginal` | −6.85 | 0.00 ± 0.01 | 2 min |
| bgms, imputing | **−6.79** | +0.06 ± 0.02 | 50 s |
| bgms, imputing, no edge selection | −6.87 | −0.02 ± 0.02 | 50 s |
| bgms, default (drops incomplete rows) | −7.33 | −0.48 ± 0.06 | 17 s |

What we take from this:

* **Neither method dominates.** With few observations and weak couplings, how strongly the fit is regularised matters more than how it is computed. `mpf` chooses that from the data and is clearly better on the simulations (in every one of the 32 weakly coupled landscapes); bgms's fixed prior is too loose there, and setting it to the right scale closes most of the gap. With strong couplings bgms is better.
* **On the real DRH data bgms predicts held-out entries slightly better than `mpf`**, provided edge selection and imputation are both on, and it is several times faster when answers are missing. (On complete data the two take the same time: about 15 s for 128 rows.) The simulated DRH-like data point the other way, but their "truth" is a dense model fitted by `mpf`, which favours `mpf`; the real data appear to be sparser than that. bgms puts 73 of the 190 couplings above an inclusion probability of one half, and 34 above 0.9.
* **bgms must be told to impute.** Its default of dropping incomplete rows discards more than half of the DRH entries and is much the worst option on every test.
* **bgms reports uncertainty, which `mpf` does not, but its intervals are too narrow.** Its 95% intervals contained the true coupling 71–85% of the time on the complete simulated data and 82% of the time on the DRH-like data (86–94% for the fields).
* **The two landscapes of the DRH differ more than any two `mpf` variants do.** The couplings from bgms and from `mpf --frozen` correlate 0.84 (against 0.99 between `mpf` variants) and share 9 of the 15 strongest couplings and 129 of the 150 most probable configurations, although `mpf`'s estimate lies inside bgms's interval for 188 of the 190 couplings. Statements about individual couplings are therefore less secure than the agreement among `mpf` variants suggests.

Limits: one version of bgms, with its default settings apart from the variations shown; KL and held-out probability are not the only things one might care about; the comparison does not include exact maximum likelihood. Scripts and results are in [MPF_CMU/validation/bgms](MPF_CMU/validation/bgms) and [MPF_CMU/validation/results/bgms](MPF_CMU/validation/results/bgms) (`summary.txt` has every number above with its standard error).

## Components
### ```/data```
* ```data/mdl_experiments```: processed DRH data (post-MPF).
* ```data/clean```: processed DRH data (pre-MPF).
* ```data/reference```: reference files (i.e. ensuring links between questions and cultures). 
* ```data/preprocessing```: files created from ```/preprocessing``` (used in ```/analysis```)
* ```data/analysis```: files created from ```/analysis```. 

Raw data from the DRH (i.e. pre-curation) is not provided given size limits. 
This might be available upon request. Everything after subset curation is provided.

### ```/fig```
Figures for "Inferring Cultural Landscapes with the Inverse Ising Model" (```5. Results: The Database of Religious History```)

### ```/tables```
Tables for "Inferring Cultural Landscapes with the Inverse Ising Model". (```5. Results: The Database of Religious History```)

### ```/preprocessing```
Preprocessing and data curation for the DRH (prior to ```/analysis```). Relies on ```Python``` (and optionally ```Julia```).
Some of the preprocessing (e.g. ```/preprocessing/preprocessing.py``` and ```/preprocessing/curation.py```) happens prior to the inference step, while the rest happens after. The raw data is not included here since it exceeds GitHub size limits, but the analysis can be reproduced from the inference step. For more, see the ```README``` in ```/preprocessing```.

### ```/analysis```
Code to reproduce the analysis, figures (```/fig```), and tables (```/tables```) related to the ```DRH``` case study. 
Where natural, one script reproduces a particular figure or table (e.g. ```/analysis/fig_3_parameters.py``` creates the two figures that make up *Figure 3* in the article). Some interdependencies between documents exist, e.g. ```/analysis/fig_4_landscape.py``` relies on ```/analysis/prep_fig_4_landscape.py``` and ```/analysis/fig_A1_dendrogram.py```. For more, see the ```README``` in ```/analysis```.

### ```/MPF_CMU``` 
MPF_CMU contains the optimized C code to implement all of the extensions and modifications to MPF described in our paper, as revised in 2026 (see *2026 Claude/Codex Analysis* above, and ```CLAUDE_IMPROVEMENTS_SUMMARY.md```). The program itself is in ```MPF_CMU/src```: a self-contained directory of C sources, Makefile and tests that needs only a C compiler (no external libraries; OpenMP is used if the compiler has it), so it can be copied and used for other work. ```cd MPF_CMU/src && make && make test```. The published fits are reproduced with the ```--legacy``` option. ```MPF_CMU/validation``` holds the experiments behind the 2026 analysis, and ```MPF_CMU/paper_runs``` the scripts and logs of the original runs for the paper, many of which were carried out on the Bridges2 Pittsburgh Supercomputing Center system. For details, see the READMEs in ```MPF_CMU```.

### ```/regenerated```
The whole DRH analysis (data products, figures, tables) regenerated by ```analysis/regenerate.py``` from the published parameters and from the 2026 refits, with comparisons. The published outputs in ```/data```, ```/fig``` and ```/tables``` are unchanged.

## Getting Started (DRH analysis)

Environments tested on ubuntu version 22.04 LTS. 

### Requirements 

Working installation of ```Python``` (tested with v3.10.6). ```Julia``` (tested with v1.8.5) is optional: ```/preprocessing/expand_data.py``` does the same job as ```/preprocessing/expand_data.jl```.

### Installation

1. Clone the repo (if not already done). Here shown for ```ssh``` but ```https``` also fine:
    ```sh
    git clone git@github.com:victor-m-p/humanities-glass.git
    ```

2. Install the ```Python``` environment (```glassenv```):
    ```sh
    bash venv_create.sh
    bash venv_add.sh
    ```

3. (Optional) Install the ```Julia``` packages used in ```/preprocessing/expand_data.jl``` with the Julia REPL.

4. To regenerate the whole analysis from a parameter file into a new folder (the published outputs are not touched):
    ```sh
    python analysis/regenerate.py data/mdl_experiments/matrix_questions_20_maxna_5_nrows_455_entries_407.txt.mpf_params.dat regenerated/NAME
    ```

<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE.md` for more information.

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
