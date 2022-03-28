import torch
import sys
sys.path.append('..')
sys.path.append('../..')
from src.data import *
from src.losses import *
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import numpy as np
from random import shuffle

def test( model,
                optimizer: Optimizer,
                testloader,
                noise_scale_per_prior= 0,
                prior_min = -1,
                prior_max =1
                ) -> dict:
    ''' The function returns a dict of scalars, the statistics of the epoch
    def train_epoch(self, model,
                optimizer:Optimizer,
                trainloader,
                testloader,
                prefix_text,
                path) -> dict:

    '''
    totalsamples = 0
    ISNAN = False
    avg_train_result = None
    avg_val_result = None
    noise_range = (prior_max-prior_min)*noise_scale_per_prior


    model.eval()
    totalsamples = 0
    for batch_n, data in enumerate(testloader):
        with torch.set_grad_enabled(False):
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            noise = torch.rand_like(inputs)*noise_range - noise_range/2
            inputs = inputs+noise
            inputs = inputs.clamp(prior_min,prior_max)
            this_samp_num = inputs.shape[0]
            temp_result = optimizer.calc_grad(model, inputs, labels)
            if avg_val_result is None:
                avg_val_result = temp_result
            else:
                avg_val_result = dict_lambda(avg_val_result, temp_result,
                                             f=lambda x, y: (x * totalsamples + y * this_samp_num) / (
                                                     totalsamples + this_samp_num))

            totalsamples = totalsamples + this_samp_num

        if ISNAN:
            break


    return avg_val_result


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    dataset= globals()[dataset_name]
    print(dataset)
    # dataset = cifar100
    exp_dir = os.path.join('..','..','Results','AlphaExpRefined',)
    models_dir = os.path.join(exp_dir, dataset.__name__)
    stat_dir = os.path.join(exp_dir,'stats','robustness')
    os.makedirs(stat_dir,exist_ok=True)
    summary_file_path = os.path.join(stat_dir,'robustness_'+dataset.__name__+'.pkl')

    df = pd.DataFrame()
    test_ldr = dataset(128, isshuffle=False)[1]

    temp = os.listdir(models_dir)
    shuffle(temp)
    for model in temp:

        model_path = os.path.join(models_dir,model,'model.pth')
        optimizer_path = os.path.join(models_dir,model,'optimizer.pth')
        result_dict_path = os.path.join(models_dir,model,'result.dict')
        res_dict = torch.load(result_dict_path)
        setup_dict = res_dict['setup']
        model = torch.load(model_path)
        optimizer = torch.load(optimizer_path)#type:Optimizer
        notNaN = True
        AUC = 0
        num_try= 0
        for noise_lvl in np.arange(0,1,0.1):
            noise_lvl = noise_lvl
            if notNaN:
                temp_res_dict = test(model,optimizer,test_ldr,noise_lvl)
                temp_res_dict = dict(test_accuracy=temp_res_dict['acc'],noise_lvl=noise_lvl)
                notNaN = temp_res_dict['test_accuracy']==temp_res_dict['test_accuracy']
            else:
                temp_res_dict = dict(test_accuracy=np.nan, noise_lvl=noise_lvl)
            AUC = AUC + temp_res_dict['test_accuracy']
            num_try= num_try +1


            print(temp_res_dict)
        mean_res_dict = dict(AUC=AUC/num_try)
        mean_res_dict.update(setup_dict)
        df = df.append(mean_res_dict,ignore_index=True)
        df_nonan = df.fillna(value=0.1)
        # plt.clf()
        # sns.lineplot(data=df_nonan,x='alpha',hue='optimizer',y='AUC')
        # plt.show(block=False)
        # plt.pause(0.01)

        pd.to_pickle(df,summary_file_path)
        print(df.shape)
    pd.to_pickle(df,summary_file_path)

