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

def odin_score(energy):
    pass
def godin_score(energy):
    pass
def max_energy_score(energy):
    pass

def energy_score(energy):
    pass

def eval_model(model,dataloader):
    for input, label in dataloader:
        energy = model(input)
        odinscore = odin_score(energy)
        max_energy = max_energy_score(energy)
        energy_score = energy_score(energy)
        godin =godin_score(energy)
        temp_dict= dict(energy=energy
                        ,max_energy= max_energy,
                        odinscore=odinscore,
                        godin_score=godin)



if __name__ == '__main__':
    expname = sys.argv[1]
    print("Evaluating " + expname + " on Mayo")
    models_path = os.path.join(root_path,'Results',expname)
    dataloader_mayo = mayo_pancreas(1,isshuffle=False)
    dataloader_nih = nih_pancreas(1,isshuffle=False)
    for model_dir in os.listdir(models_path):
        model_path = join(model_dir,'model.pth')
        result_dict = torch.load(join(model_dir,'result.dict')) #type:dict
        model = torch.load(model_path)# type:torch.Module
        model.to('cuda:0')


        model.to('cpu')