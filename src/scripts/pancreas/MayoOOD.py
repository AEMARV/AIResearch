import torch
import sys
import os
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
root_path = os.path.abspath(os.path.join('..','..','..'))
from src.data import mayo_pancreas
from src.data import nih_pancreas
from os.path import join as join
import pandas as pd
from pandas import DataFrame as DataFrame
from sklearn.metrics import roc_auc_score
import numpy as np
import monai.data.nifti_writer as nifti_writer
def odin_score(input:torch.Tensor,model,epsilon=0.0001):
    input.requires_grad = True
    output = model(input).logsumexp(dim=1).mean()
    output.backward()
    input2 = (input + input.grad.data * epsilon).detach()
    print(input.grad.data.abs().sum())
    with torch.no_grad():
        output = model(input2).logsumexp(dim=1)
    return output.detach()
def softmax_score(energy):
    with torch.no_grad():
        energy = energy.log_softmax(dim=1)
        output = energy.max(dim=1, keepdim=True)[0]
    return output.detach()
def max_energy_score(energy):
    with torch.no_grad():
        output = energy.max(dim=1,keepdim=True)[0]
    return output.detach()

def calc_energy_score(energy):
    with torch.no_grad():
        output = energy.logsumexp(dim=1,keepdim=True)
    return output

def visualize(model,dataloader,modelname):
    i= 0
    if modelname =='UNet':
        return
    for input,label,_ in dataloader:
        # print(input.shape)
        input = input.to('cuda:0')
        print(input.max(),input.min())
        energy = model(input)
        input = input.squeeze().cpu().numpy()
        prediction = (energy == energy.max(dim=1,keepdim=True)[0])
        prediction = prediction/prediction.sum(dim=1,keepdim=True)
        prediction = prediction[0:,0:1,0:].squeeze()
        nifti_path = os.path.join('.','pred_{}'.format(modelname)+'_%d'%i + '.nii.gz')
        nifti_path_orig = os.path.join('.', 'img_{}'.format(modelname) + '_%d' % i + '.nii.gz')

        # print(prediction.sum()/(prediction*0+1).sum())
        nifti_writer.write_nifti(prediction.cpu().numpy(),nifti_path)
        nifti_writer.write_nifti(input, nifti_path_orig)
        i = i + 1
        if i > 3:
            break

    return

def eval_model(model,dataloader,modelname):
    df = DataFrame()
    i= 0
    for input,label,_ in dataloader:
        # print(input.shape)
        input = input.to('cuda:0')
        energy = model(input)
        prediction = (energy == energy.max(dim=1,keepdim=True)[0])[0:,0:1].float()
        odinscore = odin_score(input,model,epsilon=100)
        max_energy = max_energy_score(energy)
        energy_score = calc_energy_score(energy)
        softmax = softmax_score(energy)
        softmax = (prediction*softmax).mean()
        odinscore = (prediction*odinscore).mean()
        max_energy = (prediction * max_energy).mean()
        energy_score = (prediction * energy_score).mean()

        # godin =godin_score(energy)
        temp_dict= dict(energy=energy_score.squeeze().cpu().item(),
                        max_energy= max_energy.squeeze().cpu().item(),
                        odin=odinscore.squeeze().cpu().item(),
                        softmax=softmax.cpu().numpy(),
                        label=label.squeeze().cpu().item())
        # print(temp_dict)
        df = df.append(temp_dict,ignore_index=True)
        i = i+1

        # if i > 10:
        #     summ = create_summary_entry(df)
        #     print(i, dataloader.__len__(), 'auroc ->' , summ )
        if i > 3:
            break

    return df

def calc_auroc(df1:pd.DataFrame,score):
    # df1 is a dataframe containing all scores in columns, rows correspond to data
    # df1 should contain a column callec labels
    # score is the name of the ood score used
    # returns a dict containing the auroc corresponding to all scores

    all_scores = df1[score].to_numpy()
    all_labels = df1['label'].to_numpy()
    # print(df1['label'])
    # print(all_scores)
    auroc = roc_auc_score(all_labels,all_scores)
    # negauroc = roc_auc_score(all_labels,-all_scores)
    auroc_final = auroc

    return auroc_final

def create_summary_entry(df:DataFrame)->dict:
    # The dataframe contains all energy scores and labels
    output_dict = dict()
    for score in ['max_energy','odin','energy','softmax']:
        auroc = calc_auroc(df,score)
        output_dict['auroc_'+score] = auroc
    return output_dict

if __name__ == '__main__':
    if sys.argv.__len__()<2:
        expname = 'SortedValueExp'
    else:
        expname = sys.argv[1]
    # print("Evaluating " + expname + " on Mayo")
    dataloader_mayo = mayo_pancreas(1, isshuffle=False, root_dir=root_path)
    print(dataloader_mayo)
    models_path = os.path.join(root_path,'Results',expname,'nih_pancreas')
    summary_df = DataFrame()
    for model_dir in os.listdir(models_path):
        if model_dir.split('.')[-1] == 'txt':
            continue
        model_path = join(models_path,model_dir,'model.pth')
        try:
            result_dict = torch.load(join(models_path,model_dir,'result.dict')) #type:dict
        except Exception:
            continue
        setup_dict= result_dict['setup']
        # print(result_dict.keys())
        scalar_dict = result_dict['scalars']
        print(setup_dict)
        compact_res = dict(acc = scalar_dict['acc_val  '][-1],dice= scalar_dict['dice_val  '][-1],IOU=scalar_dict['IOU_val  '][-1])
        model = torch.load(model_path)# type:torch.Module
        model.to('cuda:0')
        df = visualize(model,dataloader_mayo,setup_dict['model_name'])
        # df = eval_model(model,dataloader_mayo,setup_dict['model_name']) # df contains per model energy scores and labels

        # calculate AUROC
        # summary_dict = create_summary_entry(df)
        # print(summary_dict, setup_dict,compact_res)
        # summary_dict = summary_dict.update(summary_dict)
        # summary_dict = summary_dict.update(setup_dict)
        # summary_df.append(DataFrame(summary_dict),ignore_index=True)
        # summary_df.to_pickle()








        model.to('cpu')