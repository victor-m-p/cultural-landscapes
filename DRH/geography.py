import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ptitprince as pt 
import seaborn as sns
import geopandas as gpd
pd.set_option('display.max_colwidth', None)

data_raw = pd.read_csv('../data/raw/drh_20221019.csv')
entry_reference = pd.read_csv('../data/analysis/entry_reference.csv')

data_geography = data_raw[['entry_name', 'entry_id', 'region_id', 'region_tags', 'region_desc']]
data_geography.tail(10) # ??

# low-resolution earth
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.plot()
