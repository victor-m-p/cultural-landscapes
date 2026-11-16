# The DRH analysis regenerated from other parameter fits

Everything here is produced by `analysis/regenerate.py`, which runs the whole documented pipeline
(preprocessing, then every figure and table script) in a temporary copy, from a given parameter file. The
committed `data/`, `fig/` and `tables/` — the published outputs — are not touched.

| folder | parameters | purpose |
|---|---|---|
| `published/` | `data/mdl_experiments/..._params.dat`, the fit used in the paper | reproducibility check, and the baseline for comparisons |
| `frozen_nn1_entrycv/` | `MPF_CMU/validation/results/drh_refits/drh_frozen_entrycv.mpf_params.dat` | the revised fit: frozen-weight treatment of missing answers, single-flip neighbours, leave-one-entry-out cross-validation |
| `frozen_nn2_entrycv/` | `.../drh_frozen_entrycv_nn2.mpf_params.dat` | the same with two-flip neighbours |
| `frozen_nn3_entrycv/` | `.../drh_frozen_entrycv_nn3.mpf_params.dat` | the same with three-flip neighbours |

Each folder has `params.dat`, `run_log.txt` (which scripts ran, with which library versions),
`data/analysis`, the small files of `data/preprocessing`, `fig/` and `tables/`.

The `comparison_A_vs_B.txt` files are the output of
`python analysis/compare_regenerated.py regenerated/A regenerated/B`: parameters, the 150 most probable
configurations, the three groups (adjusted Rand index, contingency table, where each labelled religion
ends up), the tables, the Spartan example, and the local landscapes of the Free Methodist Church and the
Roman Imperial Cult. There is one for the published fit against each refit, and one for each pair of refits.

## What the comparisons show

The parameters, the most probable configurations and the local landscapes are stable across all four fits;
the division into three groups is not, and it is no more stable to clustering 149 or 151 configurations of
the *published* fit instead of 150 (`MPF_CMU/validation/results/clustering_sensitivity.txt`). See Section 5
of `CLAUDE_IMPROVEMENTS_SUMMARY.md`.

## Environment

All four folders were produced with the versions pinned in `requirements.txt` (Python 3.10, numpy 1.21.3,
pandas 1.5.2, scikit-learn 1.2.0, matplotlib 3.6.2, networkx 2.8.8; scipy 1.10.1), on Linux; `published/`
and `frozen_nn1_entrycv/` were also produced on macOS with the same tables as a result. In that
environment `published/` reproduces the committed outputs: all six tables are byte-identical, as are the
dendrogram clusters, the annotations and the group labels; other data products differ only in float
formatting and in the layout coordinates, which come from Graphviz. The clustering depends on tie-breaking
inside scikit-learn/scipy: with current versions of those libraries the published parameters give a
different dendrogram.

## How to read the figures for the refit

The scripts were written around the published fit. They now take the groups from the dendrogram itself (its
three top-level branches, numbered by probability mass) instead of from hand-written lists, and they place
labels by configuration rather than by rank, so they run for any fit. But the label offsets, the choice of
which configurations to label, and the colour names ("Red", "Blue", "Yellow") in the tables were chosen by
hand for the published layout and groups. In the `frozen_*` folders the labels may overlap, and "Group 2"
there is not the same set of religions as "Group 2" in the paper: use the comparison files.

## To regenerate

```
python analysis/regenerate.py PARAMS_FILE regenerated/NAME
python analysis/compare_regenerated.py regenerated/published regenerated/NAME > regenerated/comparison_published_vs_NAME.txt
```
