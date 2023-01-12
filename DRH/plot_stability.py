'''
VMP 2023-01-08: 
updated and run on new parameter file.
produces scatterplot of log(p(configuration)) x p(remain)
'''

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

# plotting setup
small_text = 12
large_text = 18

# import 
stability = pd.read_csv('../data/COGSCI23/evo_stability/maxlik_evo_stability.csv')

## to log 
stability['log_config_prob'] = np.log(stability['config_prob'])

# annotations 
## above trendline 
aztec = stability[(stability['remain_prob'] > 0.85) & 
                  (stability['log_config_prob'] < -6.5) &
                  (stability['log_config_prob'] > -7.5)]
pauline = stability[(stability['remain_prob'] > 0.89) &
                    (stability['log_config_prob'] > -4.5)]
top_1 = stability[(stability['remain_prob'] > 0.75) &
                  (stability['log_config_prob'] < -11)]
top_2 = stability[(stability['remain_prob'] > 0.79) &
                  (stability['log_config_prob'] > -10) &
                  (stability['log_config_prob'] < -9.5)]
top_3 = stability[(stability['remain_prob'] > 0.83) &
                  (stability['log_config_prob'] > -8.5) &
                  (stability['log_config_prob'] < -8)]
## below trendline
sadducees = stability[(stability['remain_prob'] < 0.69) &
                  (stability['log_config_prob'] < -13)]
bot_1 = stability[(stability['remain_prob'] < 0.69) & 
                  (stability['log_config_prob'] > -12) & 
                  (stability['log_config_prob'] < -10.5)]
bot_2 = stability[(stability['remain_prob'] > 0.72) & 
                  (stability['remain_prob'] < 0.73) & 
                  (stability['log_config_prob'] < -8) &
                  (stability['log_config_prob'] > -8.5)]
bot_3 = stability[(stability['remain_prob'] > 0.74) & 
                  (stability['remain_prob'] < 0.76) & 
                  (stability['log_config_prob'] < -7) &
                  (stability['log_config_prob'] > -7.5)]
bot_4 = stability[(stability['remain_prob'] > 0.8) & 
                  (stability['remain_prob'] < 0.84) & 
                  (stability['log_config_prob'] < -4.5) & 
                  (stability['log_config_prob'] > -5.2)]

## gather all of them 
annotations = pd.concat([aztec, pauline, top_1, top_2, top_3,
                         sadducees, bot_1, bot_2, bot_3, bot_4])
annotations = annotations.drop_duplicates()

## now find the corresponding religions 
pd.set_option('display.max_colwidth', None)
entry_configuration = pd.read_csv('../data/analysis/entry_configuration_master.csv')
entry_configuration = entry_configuration[['config_id', 'entry_drh']].drop_duplicates()
entry_configuration = entry_configuration.groupby('config_id')['entry_drh'].unique().reset_index(name = 'entry_drh')
annotations = entry_configuration.merge(annotations, on = 'config_id', how = 'inner')
annotations = annotations.sort_values('config_id')
annotations
## short names for the entries 
entries = pd.DataFrame({
    'config_id': [362374, 
                  370630,
                  372610, 501638, #525313,
                  #652162, 
                  769975, 774016,
                  #894854, 
                  896898, 913282,
                  929282, 978831, 995207,
                  #1016839
                  ],
    'entry_short': ['Pauline', #'Marcionites'
                    'Muslim Students US/CA',
                    'Yolngu', 'Donatism', #'Soviet Atheism',
                    #'Santal', 
                    'Aztec', 'Pagans under Julian',
                    #'Circumcellions', 
                    'Iban', 'Rwala Bedouin',
                    'Sadducees', 'Tang Tantrism', 'Samaritans',
                    #'Muridiyya Senegal'
                    ]})
## merge back in 
annotations = annotations.merge(entries, on = 'config_id', how = 'inner')
annotations = annotations.drop(columns = {'entry_drh'})

## prepare colors 
annotations_id = annotations[['config_id']]
stability = stability.merge(annotations_id, 
                            on = 'config_id',
                            how = 'left',
                            indicator = True)
stability = stability.rename(columns = {'_merge': 'color'})
stability = stability.replace({'color': {'left_only': 'tab:blue',
                                         'both': 'tab:orange'}})
stability = stability.sort_values('color')


# plot
fig, ax = plt.subplots(dpi = 300)

## the scatter
sns.scatterplot(data = stability, 
                x = 'log_config_prob',
                y = 'remain_prob',
                c = stability['color'].values)
sns.regplot(data = stability,
           x = 'log_config_prob',
           y = 'remain_prob',
           scatter = False, 
           color = 'tab:red')
## the annotations 
for _, row in annotations.iterrows(): 
    x = row['log_config_prob']
    y = row['remain_prob']
    label = row['entry_short']
    # Cistercians break the plot currently
    if label in ['Pauline', 'Astec', 'Samaritans',
                 'Muslim Students US/CA', 'Muridiyya Senegal',
                 'Tang Tantrism', 'Soviet Atheism']: 
        ax.annotate(label, xy = (x-0.1, y),
                    horizontalalignment = 'right',
                    verticalalignment = 'center')
    else: 
        ax.annotate(label, xy = (x+0.1, y),
                    horizontalalignment = 'left',
                    verticalalignment = 'center')
plt.xlabel('log(p(configuration))', size = small_text)
plt.ylabel('p(remain)', size = small_text)
plt.xlim(-14, -3.6)
## save figure 
plt.savefig('../fig/COGSCI23/overview/config_remain.pdf')