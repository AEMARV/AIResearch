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

def energy_stats(model,dataset,dataset_name,setup,max_data=10000000):
    train_ldr = dataset
    total_energy = pd.DataFrame()
    # train_ldr, test_ldr = dataset(10,isshuffle=False)
    hasnan = False
    total_data= 0
    with torch.no_grad():
        for input,label in train_ldr:
            total_data += input.shape[0]
            input=input.to('cuda:0')
            output = model(input)
            output_conditional=  output.log_softmax(dim=1)
            softmax_score = output_conditional.max(dim=1,keepdim=True)[0].squeeze()
            max_energy = output.max(dim=1,keepdim=True)[0].squeeze()
            energy = output.logsumexp(dim=1,keepdim=False)
            hasnan = (energy[0] != energy[0]).item()
            energy = energy.squeeze().detach().to('cpu').numpy()
            temp_dict = {'energy':energy,
                         'softmax_score':softmax_score.to('cpu').numpy(),
                         'max_energy':max_energy.to('cpu').numpy(),
                         'dataset':dataset_name}
            temp_dict.update(setup)
            total_energy = total_energy.append(pd.DataFrame(temp_dict),ignore_index=True)
            if total_data>max_data:
                break

            if hasnan:
                break


    return total_energy

def calc_auroc(df1:pd.DataFrame,df2:pd.DataFrame,score):
    score1 = df1[score].to_numpy()
    label1 = np.ones_like(score1)
    score2 = df2[score].to_numpy()
    label2 = np.zeros_like(score2)
    all_scores = np.concatenate([score1,score2],axis=0)
    all_scores[all_scores!=all_scores]=0
    all_labels = np.concatenate([label1,label2],axis=0)
    auroc = roc_auc_score(all_labels,all_scores)
    negauroc = roc_auc_score(all_labels,-all_scores)
    auroc_final = auroc

    return auroc_final

def calc_auroc_per_dataset(model, in_dataset, out_dataset_list, setup): #-> pd.DataFrame
    in_score_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    in_ds_name = in_dataset['name']
    score_types = ['energy', 'max_energy','softmax_score']

    for in_ldr in in_dataset['loader_list']:
        temp_df = energy_stats(model, in_ldr, in_ds_name, setup)
        in_score_df = in_score_df.append(temp_df, ignore_index=True)

    # in_score_df is a dataframe each row representing a sample, and columns represent scores
    for out_ds in out_dataset_list:
        out_name = out_ds['name']
        print("running ",out_name)
        out_ldrs = out_ds['loader_list']
        out_score = pd.DataFrame()
        for out_ldr in out_ldrs:
            temp_df = energy_stats(model, out_ldr, out_name, setup)
            out_score = out_score.append(temp_df, ignore_index=True)

        for score_type in score_types:
            auroc = calc_auroc(in_score_df,out_score,score_type)
            # temp_dict=  dict(auroc=auroc,score_type=score_type, in_dataset=in_ds_name, out_dataset=out_name)
            # temp_dict.update(setup)
            summary_df['auroc',score_type,out_name] = [auroc]

            # summary_df = summary_df.append(temp_dict,ignore_index=True)
        summary_df.columns = pd.MultiIndex.from_tuples(summary_df.columns)
        for score_type in score_types:
            m = summary_df['auroc',score_type].mean(1)
            summary_df['auroc',score_type,'mean'] = m

        summary_df = summary_df.assign(in_dataset=in_ds_name)
        summary_df = summary_df.assign(**setup)

        print(summary_df)

    return summary_df

if __name__ == '__main__':
    in_dataset_name = 'cifar10'
    exp_folder = os.path.join('..','..', 'Results','AlphaExpRefined')
    result_folder = os.path.join('..','..', 'Results','AlphaExpRefined', in_dataset_name)
    folder = os.listdir(result_folder)
    total_energy = pd.DataFrame()
    in_dataset = dict(loader_list=[locals()[in_dataset_name](128)[1]],name=in_dataset_name) ## only the test loader
    out_datasets=[]
    out_datasets = out_datasets +[dict(name='cifar100',loader_list=cifar100(512))]
    out_datasets = out_datasets + [dict(name='mnist', loader_list=mnist(512))]
    out_datasets = out_datasets + [dict(name='svhn', loader_list=svhn(512))]
    out_datasets = out_datasets + [dict(name='fmnist', loader_list=fashionmnist(512))]
    out_datasets = out_datasets + [dict(name='stl10', loader_list=stl10(128))]
    out_datasets = out_datasets + [dict(name='kmnist', loader_list=kmnist(128))]

    # print(out_datasets)

    summary_path = os.path.join(exp_folder,'stats','OOD')
    os.makedirs(summary_path,exist_ok=True)
    summary_path = os.path.join(summary_path,'OOD_{}.pkl'.format(in_dataset_name))
    if os.path.exists(summary_path):
        # pd.json_normalize()
        pass
        # summary_df= pd.read_pickle(summary_path)
        # print(summary_df.columns)
        # summary_df.columns = pd.MultiIndex.from_tuples(summary_df.columns)
        # print(summary_df.columns)
    else:
        pass
    summary_df = pd.DataFrame()
    for i,models_folder_name in enumerate(folder):
        # break
        print("model ", str(i),' out of ', str(folder.__len__()), ' folders')
        models_folder_path =  os.path.join(result_folder,models_folder_name)
        model_path = os.path.join(models_folder_path,'model.pth')
        result_dict_path = os.path.join(models_folder_path,'result.dict')

        result_dict = torch.load(result_dict_path)
        setup = result_dict["setup"]
        model = torch.load(model_path)
        model = model.to('cuda:0')

        summary_temp = calc_auroc_per_dataset(model,in_dataset,out_datasets,setup)
        summary_df = summary_df.append(summary_temp,ignore_index=True)
        # print(summary_df.keys())
        # plt.clf()
        # summary_plot = sns.scatterplot(data=summary_df, x=('alpha','',''), y=('auroc','max_energy','cifar100'),style='optimizer',size='augment_rate')
        # summary_plot = summary_plot.get_figure()
        # summary_plot.savefig('summaryplot.svg')
        # plt.show(block=False)
        # plt.pause(0.01)
        model.to('cpu')

        summary_df.to_pickle(summary_path)

    # plt.clf()
    # summary_plot = sns.scatterplot(data=summary_df, x='alpha', y='auroc', style='optimizer', hue='score_type',
    #                                size='augment_rate')
    # summary_plot = summary_plot.get_figure()
    # summary_plot.savefig('summaryplot_alpha.eps')
    # summary_plot.savefig('summaryplot_alpha.png')
    #
    # plt.clf()
    # summary_plot = sns.scatterplot(data=summary_df, x='augment_rate', y='auroc', style='optimizer', hue='score_type',
    #                                size='alpha')
    # summary_plot = summary_plot.get_figure()
    # summary_plot.savefig('summaryplot_aug.eps')
    # summary_plot.savefig('summaryplot_aug.png')
    #
    # plt.clf()
    # summary_plot = sns.scatterplot(data=summary_df, x='score_type', y='auroc', style='optimizer', hue='alpha',
    #                                size='augment_rate')
    # summary_plot = summary_plot.get_figure()
    # summary_plot.savefig('summaryplot_score.eps')
    # summary_plot.savefig('summaryplot_score.png')
    #
    # plt.clf()
    # summary_plot = sns.scatterplot(data=summary_df, x='optimizer', y='auroc', style='score_type', hue='alpha',
    #                                size='augment_rate')
    # summary_plot = summary_plot.get_figure()
    # summary_plot.savefig('summaryplot_optimizer.eps')
    # summary_plot.savefig('summaryplot_optimizer.png')
    # # sns.catplot(data=total_energy,
    # #             y='energy',
    # #             x='alpha',
    # #             col='optimizer',
    # #             hue='dataset',
    # #             kind='violin',
    # #             split=True,ax=axes[0])
    # plt.show(block=False)
    # plt.pause(1)
    # input()









