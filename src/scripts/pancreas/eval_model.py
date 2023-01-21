import sys
sys.path.append('..')
sys.path.append('../..')
import torch
from src.data import nih_pancreas
import os

if __name__ == '__main__':
    Result_dir_path = os.path.join(*(3*['..'] + ['Results']))
    model_name = ''
    expname= ''
    model_dir_path= os.path.join(Result_dir_path,expname,'pancreas_nih',model_name)
    model_path = os.path.join(model_dir_path,'model.pth')
    model = torch.load(model_path)
    data_train,data_test = nih_pancreas(isshuffle=False,return_original_too=True)
    for vol, label, original_label in data_test:


    # Load Dataset
