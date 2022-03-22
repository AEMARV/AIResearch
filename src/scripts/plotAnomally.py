import sys
sys.path.append('..')
sys.path.append('../..')
from src.data import *
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import seaborn as sns
sns.set_style(style='whitegrid')


if __name__ == '__main__':
    dataset= 'cifar10'
    score= sys.argv[1]
    exp_root =os.path.join('..','..','Results','AlphaExpRefined','stats','OOD')
    summary_df = pd.read_pickle(os.path.join(exp_root,'OOD_{}.pkl'.format(dataset)))
    print(summary_df)
    energy = summary_df['auroc','energy','mean']
    max_energy = summary_df['auroc', 'max_energy','mean']
    softmax_score = summary_df['auroc', 'softmax_score','mean']
    print(softmax_score)
    # summary_df.columns = summary_df.columns.to_flat_index()
    summary_df.columns = ["_".join(a) for a in summary_df.columns.to_flat_index()]

    print(summary_df.columns)
    summary_df.rename(columns={'optimizer__': 'Objective Function'}, inplace=True)
    summary_df = summary_df.sort_values(by='Objective Function')
    opt_maps = {'Joint_Probabilistic': 'Joint-Intersection',
                'Joint_Cross': 'Joint-Cross Entropy',
                'Conditional_Cross': 'Conditional-Cross Entropy'}

    select_data = summary_df[
        (summary_df['Objective Function'] == ('Joint_Probabilistic')) | (summary_df['Objective Function'] == 'Joint_Cross') | (
                    summary_df['Objective Function'] == 'Conditional_Cross')]
    for key in opt_maps.keys():
        select_data.loc[select_data['Objective Function'] == key, 'Objective Function'] = opt_maps[key]
    summary_df = select_data

    # sns.violinplot(split=True,data= summary_df,x=('alpha','',''),y=('auroc','max_energy','mean'),size=('augment_rate','',''),hue=('optimizer','',''))
    for score in ['energy','max_energy','softmax_score']:
        plt.clf()
        plot = sns.lineplot(data=summary_df, x='alpha__', y='auroc_{}_mean'.format(score),
                         hue='Objective Function',lw=1)
        plot.set(xlabel='alpha',ylabel='AUROC')
        plt.legend(loc='lower right',title='Objective Function')
        plot.get_figure().savefig(os.path.join(exp_root,'{}_AUROC_{}.png'.format(score,dataset)))

    plt.clf()
    # pd.melt(fame=summary_df,value_vars=['auroc_energy_mean','auroc_max_energy_mean','auroc_'])
    sns.set_palette('husl')
    for score in ['energy', 'max_energy', 'softmax_score']:
        plot.set(xlabel='alpha', ylabel='AUROC')
        plot = sns.lineplot(data=summary_df, x='alpha__', y='auroc_{}_mean'.format(score),
                              lw=1)
    plt.legend(['energy', 'max_energy', 'softmax_score'],title='Score',loc='lower right')




    plot.get_figure().savefig(os.path.join(exp_root, 'Score_Compare_AUROC_{}.png'.format(dataset)))
    # plot = sns.lineplot(data=summary_df, x=('alpha', '', ''), y=('auroc', 'energy', 'mean'),
    #                     hue=('optimizer', '', ''))
    # plot.get_figure().savefig('lineplot_AUROC_energy.png')
    #
    # plot = sns.lineplot(data=summary_df, x=('alpha', '', ''), y=('auroc', 'softmax_score', 'mean'),
    #                     hue=('optimizer', '', ''))
    # plot.get_figure().savefig('lineplot_AUROC_softmax_score.png')

    plt.show()


