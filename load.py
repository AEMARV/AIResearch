import pydicom
import torch
import glob
import os
import numpy as np
from pydicom.pixel_data_handlers import apply_voi_lut
from typing import *
from torch import Tensor
import SimpleITK as sitk
from src.data import nih_pancreas
from src.data import NIH_Pancreas
from src.data import mayo_pancreas
from src.data import Mayo_Pancreas
import nibabel
import pandas
import monai
# class NIH_Pancreas(Dataset):
#     def __init__(self,root_dir=None):
#         if not root_dir:
#             self.root_dir = os.path.join('.','data','NIH_pancreas')
#             self.scan_dir = os.path.join(root_dir,'Pancreas_CT','PANCREAS_{}','*','*')
#             label_dir= os.path.join(self.root_dir,'Label')
#         pass
#     def __getitem__(self, idx):
#         scan_dir =self.scan_dir.format(idx)
#         pathlist = glob(scan_dir)
#         pydicom.read_file(pathlist)


if __name__ == '__main__':
    root_dir = os.path.join('.', 'data', 'Mayo_pancreas')
    scan_dir = os.path.join(root_dir, 'images', '*.img')
    label_dir = os.path.join(root_dir, 'labels.xlsx')
    import matplotlib.pyplot as plt
    plt.show()
    for idx in range(87):
        file_path = glob.glob(scan_dir)[idx]
        dl = mayo_pancreas(1)
        dataset = Mayo_Pancreas()
        img,label,path = dataset[idx]
        for x in dl:
            img, label,path = x
            print(path, label.item())
        # this_dir = scan_dir.format('%04d'%(idx+1))
        # panc_dataset = NIH_Pancreas()
        # img,label = panc_dataset[idx]
        # label = label.unsqueeze(dim=0)
        print("Path",os.path.split(path)[-1])
        print('label',label)
        # print(label.shape)
        # plt.imshow((img).mean(axis=4).squeeze())
        # plt.show()
        # input()

        # print(img.max(),img.min())
        # path_list = glob.glob(this_dir,recursive=True)
        # print(path_list)
