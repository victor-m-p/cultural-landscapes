'''
COGSCI23
'''

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

# plot setup
large_text = 18
small_text = 12

# usual setup
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')

# generate all states
n_nodes = 20
from fun import bin_states 
configurations = bin_states(n_nodes) 

# read the curated data 
d_edgelist = pd.read_csv('../data/COGSCI23/evo_clean/overview.csv')

### their data ###
# what we need ([x, y, z])
# what we need: 
d_edgelist_10 = d_edgelist[d_edgelist['t_to'] == 10]
d_edgelist_10 = d_edgelist_10.sort_values('config_to') 

# now flesh out these configurations 
config_to = d_edgelist_10['config_to'].tolist()
configs = configurations[config_to]

# (1) without normalizing 
# https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


param_grid = {
    "n_components": range(1, 7),
    "covariance_type": ["spherical", "tied", "diag", "full"],
}
grid_search = GridSearchCV(
    GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
)
grid_search.fit(configs)

# now check how it looks 
df = pd.DataFrame(grid_search.cv_results_)[
    ["param_n_components", "param_covariance_type", "mean_test_score"]
]
df["mean_test_score"] = -df["mean_test_score"]
df = df.rename(
    columns={
        "param_n_components": "Number of components",
        "param_covariance_type": "Type of covariance",
        "mean_test_score": "BIC score",
    }
)
df.sort_values(by="BIC score").head()

import seaborn as sns

sns.catplot(
    data=df,
    kind="bar",
    x="Number of components",
    y="BIC score",
    hue="Type of covariance",
)
plt.show()

cluster_by_sample = grid_search.predict(configs)
d_cluster_sample = pd.DataFrame(cluster_by_sample, 
                                columns = ['cluster'])
d_edgelist_10['sample_to'] = d_cluster_sample['cluster'].values
d_edgelist_10.to_csv('../data/COGSCI23/evo_GMM/t_10_att_to.csv', index = False)

# OK looks crazy: other approach
# (1) we need the full grid 
import itertools 
def create_grid(df, c1, c2):
    # create lists for both variables
    l_c1 = df[c1].drop_duplicates().to_list()
    l_c2 = df[c2].drop_duplicates().to_list()
    # create grid and to dataframe 
    l_comb = list(itertools.product(l_c1, l_c2))
    d_comb = pd.DataFrame(l_comb, columns=[c1, c2])
    return d_comb

d_edgelist_10 = d_edgelist_10.drop(columns = 't_to')
from_to_grid = create_grid(d_edgelist_10, 'config_from', 'config_to')
d_left = from_to_grid.merge(d_edgelist_10, 
                            on = ['config_from', 'config_to'],
                            how = 'left',
                            indicator = True)
d_left = d_left.rename(columns = {'_merge': 'n'})
d_left = d_left.replace({'n': {'both': 1, 'left_only': 0}})
d_left['n'] = [int(x) for x in d_left['n']]
## to probability 
d_left = d_left.groupby(['config_from', 'config_to'])['n'].sum().reset_index(name = 'total')
d_left['prob'] = d_left['total']/100
## make 100% sure that it is sorted correctly 
d_left = d_left.sort_values(['config_from', 'config_to'],
                            ascending = [True, True])
## now to array
d_probs = d_left.drop(columns = 'total')
d_wide = d_probs.pivot(index = 'config_from',
                       columns = 'config_to',
                       values = 'prob')
prob_array = d_wide.to_numpy()

## try to run it 
param_grid = {
    "n_components": range(1, 7),
    "covariance_type": ["spherical", "tied", "diag", "full"],
}
grid_search = GridSearchCV(
    GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
)
grid_search.fit(prob_array)

# now check how it looks 
df = pd.DataFrame(grid_search.cv_results_)[
    ["param_n_components", "param_covariance_type", "mean_test_score"]
]
df["mean_test_score"] = -df["mean_test_score"]
df = df.rename(
    columns={
        "param_n_components": "Number of components",
        "param_covariance_type": "Type of covariance",
        "mean_test_score": "BIC score",
    }
)
df.sort_values(by="BIC score").head()

# still looks crazy 
sns.catplot(
    data=df,
    kind="bar",
    x="Number of components",
    y="BIC score",
    hue="Type of covariance",
)
plt.show()

cluster_by_init = grid_search.predict(prob_array) 
## save this 
d_config_from = d_left[['config_from']].drop_duplicates()
d_config_from['cluster'] = cluster_by_init
d_config_from.to_csv('../data/COGSCI23/evo_GMM/t_10_att_from.csv', index = False)

# 260 array with (0, 1)
# ah, because we get 2 components 
# we obviously cannot plot it currently 
# not sure how we would be transform it 
## NB: we might want to try to run the top 500 
## configurations --- or 500 random ones ---
## to get a different sense of the landscape 
## i.e. there might be many more attractors
## but just only 2 in the subspace of obs. 

# (3) NORMALIZING: last new thing to try
# does not work because we get -infinity
# which is "too large for dtype('float64')". 
def lognorm(x): 
    if x != 1: 
        return np.log(x/(1-x))
    else: return np.inf 
   
d_probs = d_probs.assign(prob_norm = lambda x: np.log(x['prob']/(1-x['prob'])))
d_wide = d_probs.pivot(index = 'config_from',
                       columns = 'config_to',
                       values = 'prob_norm')
prob_array = d_wide.to_numpy()

