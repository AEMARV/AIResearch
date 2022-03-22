import sys
import os
sys.path.append('..')
sys.path.append('../..')
sys.path.append(os.path.join('..','..','..'))
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import torch
import os
import seaborn as sns
sns.set_style(style='whitegrid')
# sns.set(font_scale=1.5, rc={'text.usetex' : True})
from torch.utils.tensorboard import SummaryWriter
def summary_to_table(summary:pd.DataFrame):
    summary = summary.fillna(value=0)
    # summary = summary[summary['alpha']>1]
    summary.columns = summary.columns.swaplevel(1, 2)
    print(summary.columns)

    dfgrouped= summary.groupby(['optimizer'])
    # dfgrouped = dfgrouped['auroc']
    mean_df = dfgrouped.mean()
    mean_df = mean_df['auroc']['mean']
    print(mean_df.columns)
    mean_df.columns = [(c,'mean') for c in mean_df.columns]

    std_df = dfgrouped.std()
    std_df = std_df['auroc']['mean']
    print(std_df.columns)
    std_df.columns = [(c, 'std') for c in std_df.columns]
    max_df = dfgrouped.max()
    max_df = max_df['auroc']['mean']
    max_df.columns = [(c, 'max') for c in max_df.columns]
    total_df = mean_df.join([std_df,max_df])

    total_df.columns = pd.MultiIndex.from_tuples(total_df.columns)
    total_df.columns = total_df.columns.swaplevel(0, 1)
    # total_df = total_df[['auroc']]#type:pd.DataFrame
    print(total_df.columns)
    total_df.to_latex()
    print( total_df.to_latex()
           )
    return total_df

def summaries_to_table(summaries, names):
    total_df = summaries[0]
    total_df.columns = pd.MultiIndex.from_tuples(total_df.columns)
    total_df.columns = [(names[0],)+ c for c in total_df.columns]
    for i, temp in enumerate(summaries[1:]):
        # temp =summary_to_table(summary)
        temp.columns = [(names[i+1],)+c for c in temp.columns]
        total_df = total_df.join(temp)
    total_df.columns = pd.MultiIndex.from_tuples(total_df.columns)
    # total_df =total_df.groupby(level=0)
    print(total_df.to_latex(float_format="%.3f",multicolumn_format='c'))



if __name__ == '__main__':
    datasets =['cifar10']
    summary_list =[]
    for dataset in  datasets:
        print(dataset)
        expdir = os.path.join('..', '..','..', 'Results', 'AlphaExpRefined')
        models_dir = os.path.join(expdir, dataset)
        stat_path = os.path.join(expdir, 'stats', 'OOD')
        summary_df = pd.read_pickle(os.path.join(stat_path,'OOD_{}.pkl'.format(dataset)))
        summary_list = summary_list + [summary_to_table(summary_df)]

    summaries_to_table(summary_list,datasets)



