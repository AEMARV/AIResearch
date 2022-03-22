import torch
import sys
sys.path.append('..')
sys.path.append('../..')
from src.data import *
from src.optimizers import *
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# sns.set(font_scale=1.5, rc={'text.usetex' : True})
import numpy as np
from random import shuffle

if __name__ == '__main__':
    path_result_root = os.path.join('..','..','Results')
    expname= 'AlphaExpRefined'
    dataset= sys.argv[1]
    path_root_summary = os.path.join(path_result_root,expname,'stats','robustness')
    path_summary = os.path.join(path_root_summary,'robustness_'+dataset+'.pkl')
    df = pd.read_pickle(path_summary)
    print(df.shape)
    opt_maps = {'Joint_Probabilistic': 'Joint-Intersection',
                'Joint_Cross': 'Joint-Cross Entropy',
                'Conditional_Cross': 'Conditional-Cross Entropy'}

    select_data = df[
        (df['optimizer'] == ('Joint_Probabilistic')) | (df['optimizer'] == 'Joint_Cross') | (
                    df['optimizer'] == 'Conditional_Cross')]

    for key in opt_maps.keys():
        select_data.loc[select_data['optimizer']==key,'optimizer'] = opt_maps[key]
    select_data.rename(columns={'optimizer': 'Objective Function'}, inplace=True)
    select_data['Log AOC'] = np.log((1-select_data['AUC']))
    select_data = select_data.fillna(value=0)# type:pd.DataFrame
    select_data = select_data.sort_values(by='Objective Function')
    plot = sns.lineplot(data=select_data,x='alpha',y='AUC',hue='Objective Function')
    plot.set(ylabel='AUC Test Accuracy',xlabel='alpha')
    plot = plot.get_figure()
    plot.savefig(os.path.join(path_root_summary,dataset+'_AUC.png'))
    plt.clf()
    plot = sns.lineplot(data=select_data, x='alpha', y='Log AOC', hue='Objective Function')
    plot.set(ylabel='LogAOC Test Accuracy', xlabel='alpha')
    plot = plot.get_figure()
    plot.savefig(os.path.join(path_root_summary, dataset + '_LogAOC.png'))
    # plt.show()

