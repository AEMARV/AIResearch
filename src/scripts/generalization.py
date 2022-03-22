import sys
sys.path.append('..')
sys.path.append('../..')
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import torch
import os
import seaborn as sns
sns.set_style(style='whitegrid')
# sns.set(font_scale=1.5, rc={'text.usetex' : True})
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    for dataset in ['nih_pancreas']:
        expdir =os.path.join('..','..','Results','PancSegmentBottled')
        models_dir = os.path.join(expdir,dataset)
        stat_path = os.path.join(expdir,'stats','generalization')
        os.makedirs(stat_path,exist_ok=True)
        summary_path = os.path.join(stat_path,dataset+'_generalization.pkl')
        conclusion_dict = {'joint':{},'conditional':{}}
        total_data = pd.DataFrame()
        rows = []
        for exp in os.listdir(models_dir):
            model_dir = os.path.join(models_dir, exp )
            print(exp)
            dict_path = os.path.join(model_dir, 'result.dict')
            if not os.path.exists(dict_path):
                continue

            result_dict = torch.load(dict_path)
            scalars = result_dict['scalars']
            setup = result_dict['setup']
            last_epoch_vals = []
            temp = setup

            for key in scalars.keys():
                last_epoch_vals = float(scalars[key][-1])
                max_epoch_vals = float(max(scalars[key]))
                # temp.update({key:max_epoch_vals})
                temp.update({key: last_epoch_vals})
            print(temp)
            rows.append(temp)
        total_data = pd.DataFrame.from_dict(rows,orient='columns')
        print(total_data.columns)
        print(total_data.iloc[total_data['acc_val  '].idxmax()])
        opt_maps = {'Joint_Probabilistic':'Joint-Intersection',
                    'Joint_Cross': 'Joint-Cross Entropy',
                    'Conditional_Cross': 'Conditional-Cross Entropy'}

        select_data = total_data#[(total_data['optimizer'] == ('Joint_Probabilistic'))|(total_data['optimizer']=='Joint_Cross') | (total_data['optimizer']=='Conditional_Cross')]
        for key in opt_maps.keys():
            select_data.loc[select_data['optimizer']==key,'optimizer'] = opt_maps[key]
        select_data.rename(columns={'optimizer':'Objective Function'},inplace=True)
        select_data.to_pickle(os.path.join(stat_path,'{}_gen_summary.pkl'.format(dataset)))
        select_data= select_data.fillna(value=0)
        select_data = select_data.sort_values(by='Objective Function')
        print(select_data)
        plot = sns.lineplot(data=select_data,x='alpha',y='dice_val  ',hue='Objective Function',linewidth=2)
        plot.set(ylabel='Test Accuracy',xlabel='alpha',title='Test Accuracy For Varying Loss Functions')
        plot.get_figure().savefig(os.path.join(stat_path,dataset+'_val_acc.png'))


