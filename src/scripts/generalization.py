import sys
sys.path.append('..')
sys.path.append('../..')
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import torch
import os
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
sns.set_style(style='whitegrid')
# sns.set(font_scale=1.5, rc={'text.usetex' : True})
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please enter experiment name as arguments")
        raise Exception()
    expname = sys.argv[1]
    expdir = os.path.join('..', '..', 'Results', expname)
    datasets = ['cifar10']
    print(datasets)
    for dataset in datasets:

        models_dir = os.path.join(expdir,dataset)
        stat_path = os.path.join(expdir,'stats','generalization')
        os.makedirs(stat_path,exist_ok=True)
        summary_writer = SummaryWriter(stat_path)
        summary_path = os.path.join(stat_path,dataset+'_generalization.pkl')
        conclusion_dict = {'joint':{},'conditional':{}}
        for epoch in range(200):
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
                    last_epoch_vals = float(scalars[key][epoch])
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
            select_data.rename(columns={'loss':'Objective Function'},inplace=True)
            select_data.to_pickle(os.path.join(stat_path,'{}_gen_summary.pkl'.format(dataset)))
            select_data= select_data.fillna(value=0)
            select_data = select_data.sort_values(by='Objective Function')
            print(select_data)
            def plot_this_that(df,x_name,y_name,hue_name='Objective Function'):
                plot = sns.lineplot(data=df, x=x_name, y=y_name, hue=hue_name, linewidth=2)
                tag = x_name + '_vs_' + y_name
                tag = tag.strip(' ')
                summary_writer.add_figure(tag=tag,figure=plot.get_figure(),global_step=epoch,walltime=epoch)
                plot.get_figure().savefig(os.path.join(stat_path, dataset + '_'+ tag + '.png'))
                return plot
            xs = ['alpha','filter_scale','init_coef']
            ys = ['acc_val  ','acc_train']
            for x in xs:
                for y in ys:
                    plot_this_that(select_data,x,y)
            # plot = sns.lineplot(data=select_data,x='scale',y='acc_val  ',hue='Objective Function',linewidth=2)

            # plot.set(ylabel='Test Accuracy',xlabel='alpha',title='Test Accuracy For Varying Loss Functions')
            # plot.get_figure().savefig(os.path.join(stat_path,dataset+'_val_acc.png'))


