import torch
import torchvision as tv
from torchvision import transforms
from src.globals import PATH_DATA
import ssl
from src.layers import *
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import pydicom
from glob import glob
import os
import pydicom
from typing import *
from pydicom.pixel_data_handlers import apply_voi_lut
from torch import Tensor
import SimpleITK as sitk
from torch.nn import functional as F
from pathlib import Path
import monai
import nibabel
import pandas

ssl._create_default_https_context = ssl._create_unverified_context
def cifar10(batchsz, isshuffle=True):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    # Construct loaders
    trainset = tv.datasets.CIFAR10(PATH_DATA, train=True, download=True, transform=transform)
    testset = tv.datasets.CIFAR10(PATH_DATA, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def cifar100(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.CIFAR100(PATH_DATA, train=True, download=True, transform=transform)
    testset = tv.datasets.CIFAR100(PATH_DATA, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def svhn(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.SVHN(PATH_DATA,split='train', download=True, transform=transform)
    testset = tv.datasets.SVHN(PATH_DATA, split='test', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def mnist(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),GrayToRGB(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.MNIST(PATH_DATA, train=True, download=True, transform=transform)
    testset = tv.datasets.MNIST(PATH_DATA, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def stl10(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),GrayToRGB(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),transforms.Resize(32)] + transform)
    # Construct loaders
    trainset = tv.datasets.STL10(PATH_DATA, split='train', download=True, transform=transform)
    testset = tv.datasets.STL10(PATH_DATA, split='test', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def lsun(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),GrayToRGB(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.LSUN(PATH_DATA, classes='train', transform=transform)
    testset = tv.datasets.LSUN(PATH_DATA, classes='test', transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def kmnist(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),GrayToRGB(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.KMNIST(PATH_DATA, train=True, download=True, transform=transform)
    testset = tv.datasets.KMNIST(PATH_DATA, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def fashionmnist(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),GrayToRGB(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.FashionMNIST(PATH_DATA, train=True, download=True, transform=transform)
    testset = tv.datasets.FashionMNIST(PATH_DATA, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def places365(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.Places365(PATH_DATA, split='train', transform=transform)
    testset = tv.datasets.Places365(PATH_DATA, split='test', transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

class PancreasSet(Dataset):
    def __init__(self, ctRoot, labelRoot, phase="train", randomcrop=False, mode="none"):
        self.ctRoot = ctRoot + "/PANCREAS_{}/*/*/*.dcm"
        self.labelRoot = labelRoot + "/label{}.nii.gz"

        # train, val, test
        self.phase = phase
        self.randomcrop = randomcrop

        # {none, manual, yolo}
        self.mode = mode
        self.caseid = {"train": ["%04d" % i for i in range(1, 71)],
                       # "val": ["%04d" % i for i in range(61, 71)],
                       "test": ["%04d" % i for i in range(71, 83)]}[self.phase]
        self.anchorBox = np.array([150, 210, 150])
        self.normparam = [-240, 210]

    def __len__(self):
        return len(self.caseid)

    def __getitem__(self, index):
        pid = self.caseid[index]
        imgdir = self.ctRoot.format(pid)
        labeldir = self.labelRoot.format(pid)
        ct, label = self.readData(imgdir, labeldir)
        ct = self.normalization(ct, self.normparam[0], self.normparam[1])

        if self.mode == "none":
            return {"image": ct, "label": label}

        else:
            crop_ct, crop_mask = self.crop(ct, label)

            return {"image": crop_ct, "label": crop_mask}

    def readData(self, imgdir, labeldir):
        segMask = sitk.GetArrayFromImage(sitk.ReadImage(labeldir)).transpose(1, 2, 0).astype(np.uint8)
        ct = np.zeros_like(segMask, dtype=np.int16)

        for img in glob(imgdir, recursive=True):
            sliceid = img[-7:-4]
            ct[..., int(sliceid) - 1] = self.loadimg(img)

        return ct, segMask

    def normalization(self, img, low, high):
        img = np.clip(img, low, high)
        return (img - low) / (high - low)

    def loadimg(self, img):
        instance = pydicom.read_file(img)
        data = instance.pixel_array
        return data

    def crop(self, ct, label):
        if self.randomcrop == True:
            noise = np.array([random.randint(-20, 20), random.randint(-20, 20), random.randint(-20, 20)])
        else:
            noise = np.zeros(3, dtype=np.int)
        xmin = label.nonzero()[0].min();
        xmax = label.nonzero()[0].max()
        ymin = label.nonzero()[1].min();
        ymax = label.nonzero()[1].max()
        zmin = label.nonzero()[2].min();
        zmax = label.nonzero()[2].max()
        center = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2], dtype=np.int)
        width = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=np.int)
        center += noise

        crop_ct = ct[int(center[0] - self.anchorBox[0] / 2):int(center[0] + self.anchorBox[0] / 2),
                  int(center[1] - self.anchorBox[1] / 2):int(center[1] + self.anchorBox[1] / 2),
                  int(center[2] - self.anchorBox[2] / 2):int(center[2] + self.anchorBox[2] / 2)]
        crop_mask = label[int(center[0] - self.anchorBox[0] / 2):int(center[0] + self.anchorBox[0] / 2),
                    int(center[1] - self.anchorBox[1] / 2):int(center[1] + self.anchorBox[1] / 2),
                    int(center[2] - self.anchorBox[2] / 2):int(center[2] + self.anchorBox[2] / 2)]
        return crop_ct, crop_mask


def panc_nih(batchsz,isshuffle=True,transform=[],num_worker=1):
    root = os.path.join('..','data','NIH_pancreas')
    ct_root = os.path.join(root,'Pancreas-CT')
    label_root = os.path.join(root, 'Label')
    trainset = PancreasSet(ctRoot=ct_root,
                            labelRoot=label_root,
                            phase="train",
                            randomcrop=False,
                            mode="manual")
    trainloader = DataLoader(dataset=trainset,
                                batch_size=batchsz,
                                shuffle=False,
                                num_workers=num_worker)
    test_set = PancreasSet(ctRoot=ct_root,
                           labelRoot=label_root,
                           phase="test",
                           randomcrop=False,
                           mode="manual")
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batchsz,
                             shuffle=False,
                             num_workers=num_worker)
    return trainloader,test_loader

class NIH_Pancreas(Dataset):
    def __init__(self,root_dir=None,subset='train'):
        self.subset = subset
        if not root_dir:
            root_dir = '.'

        self.root_dir = os.path.join(root_dir,'data','NIH_pancreas')
        self.scan_dir = os.path.join(self.root_dir,'Pancreas-CT','PANCREAS_{}','*','*')
        label_dir= os.path.join(self.root_dir,'Label')
        self.label_files_template = os.path.join(label_dir,'label{}.nii.gz')
        self.id_list = {'train':["%04d"%i for i in range(1,65)],
                   'test':["%04d"%i for i in range(65,83)]}[self.subset]

        pass

    def load_dicom(self,path_file: str) -> Optional[np.ndarray]:
        dicom = pydicom.dcmread(path_file)
        # TODO: adjust spacing in particular dimension according DICOM meta
        try:
            img = apply_voi_lut(dicom.pixel_array, dicom).astype(np.float32)
        except RuntimeError as err:
            print(err)
            return None
        return img

    def load_volume(self,path_volume: str, percentile: Optional[float] = 0.01) -> Tensor:
        glob_path_slices = os.path.join(path_volume, '*.dcm')
        path_slices = glob(glob_path_slices)
        path_slices = sorted(path_slices, )
        vol = []
        for p_slice in path_slices:
            img = self.load_dicom(p_slice)
            if img is None:
                continue
            vol.append(img.T)
        vol = np.stack(vol,axis=0)
        volume = torch.tensor(vol, dtype=torch.float32)
        if percentile is not None:
            # get extreme values
            p_low = np.quantile(volume, percentile) if percentile else volume.min()
            p_high = np.quantile(volume, 1 - percentile) if percentile else volume.max()
            # normalize
            volume = (volume - p_low) / (p_high - p_low)
        return volume.T
    def crop_volume(self,x,cut_val=0.1):
        means = x.mean()
        crop_index = []
        x = x.squeeze()
        for dim in range(3):
            x = x.transpose(0,dim)
            y = x
            y = (y-means).abs().sum(dim=(1,2))
            y = y.cumsum(dim=0)
            end_val = y[-1]
            index_end = y<(1-cut_val)*end_val
            index_begin = y > cut_val*end_val
            index = index_begin & index_end
            indices = np.where(index)
            indices = indices[0]
            indices = (int(indices[0]),int(indices[-1]))
            crop_index = crop_index + [(dim,)+indices]
            x= x[indices[0]:indices[-1],:]
            x= x.transpose(0,dim)

        x = x.unsqueeze(0)
        return x,crop_index

    def __getitem__(self, idx):
        idx = self.id_list[idx]
        scan_dir =self.scan_dir.format(idx)
        vol = self.load_volume(path_volume=scan_dir,percentile=0)
        vol = vol.unsqueeze(dim=0)
        label = sitk.GetArrayFromImage(sitk.ReadImage(self.label_files_template.format(idx))).transpose(1,2,0).astype(np.uint8)
        label= torch.tensor(label,dtype=torch.uint8)
        # vol, indices = self.crop_volume(vol,cut_val=0.2)
        # for index_tuple in indices:
        #     dim = index_tuple[0]
        #     indices = index_tuple[1:]
        #     print(index_tuple)
        #     label = label.transpose(0,dim)
        #     label = label[indices[0]:indices[1]]
        #     label = label.transpose(0,dim)
        vol = vol.unsqueeze(0)
        label = label.unsqueeze(0).unsqueeze(0)
        scale = 6
        vol = F.interpolate(vol,[int(x/scale) for x in vol.shape[2:]])
        label = F.interpolate(label,[int(x/scale) for x in label.shape[2:]])
        vol = vol.squeeze(dim=0)
        label = label.squeeze(dim=0)
        label = torch.cat([label,1-label],dim=0)
        vol = vol*2 -1

        return vol,label

    def __len__(self):
        return self.id_list.__len__()

class Mayo_Pancreas(Dataset):
    def __init__(self,root_dir=None,subset='train'):
        self.subset = subset
        self.labels_df= None
        if not root_dir:
            root_dir = '.'

        self.root_dir = os.path.join(root_dir,'data','Mayo_pancreas')
        self.scan_dir = os.path.join(self.root_dir,'images','*.img')
        file_list = glob(self.scan_dir)

        label_doc= os.path.join(self.root_dir,'labels.xlsx')
        self.labels_df = pandas.read_excel(label_doc,sheet_name='Sheet3')
        self.labels_df['ID'] = self.labels_df['ID'].astype('string')

        self.id_list = {'train':[x for x in file_list if x.__contains__('control')],
                   'test':[x for x in file_list if x.__contains__('case')],
                        'all':file_list}[self.subset]

        pass

    def load_img(self,path_file:str):
        data = nibabel.load(path_file).get_fdata()
        vol = torch.tensor(np.array(data),dtype=torch.float32)
        MIN = vol.min()
        MAX = vol.max()
        vol = (vol - MIN)/(MAX-MIN)
        return vol
    def load_label(self,id):
        label = self.labels_df[self.labels_df['ID']==id]['Label']
        T = torch.tensor(label.to_numpy(),dtype=torch.float32)
        return T

    def load_dicom(self,path_file: str) -> Optional[np.ndarray]:
        dicom = pydicom.dcmread(path_file)
        # TODO: adjust spacing in particular dimension according DICOM meta
        try:
            img = apply_voi_lut(dicom.pixel_array, dicom).astype(np.float32)
        except RuntimeError as err:
            print(err)
            return None
        return img

    def load_volume(self,path_volume: str, percentile: Optional[float] = 0.01) -> Tensor:
        glob_path_slices = os.path.join(path_volume, '*.dcm')
        path_slices = glob(glob_path_slices)
        path_slices = sorted(path_slices, )
        vol = []
        for p_slice in path_slices:
            img = self.load_dicom(p_slice)
            if img is None:
                continue
            vol.append(img.T)
        vol = np.stack(vol,axis=0)
        volume = torch.tensor(vol, dtype=torch.float32)
        if percentile is not None:
            # get extreme values
            p_low = np.quantile(volume, percentile) if percentile else volume.min()
            p_high = np.quantile(volume, 1 - percentile) if percentile else volume.max()
            # normalize
            volume = (volume - p_low) / (p_high - p_low)
        return volume.T

    def __getitem__(self, idx):
        img_path = self.id_list[idx]
        # scan_dir =self.scan_dir.format(idx)
        id = os.path.split(img_path)[-1].split('.')[-2].replace('control','').replace('case','')
        vol = self.load_img(img_path)
        label = (self.load_label(id)>0).int()
        vol = vol.unsqueeze(dim=0).unsqueeze(dim=0)
        scale = 6
        vol = F.interpolate(vol,[int(x/scale) for x in vol.shape[2:]])
        # vol = vol.squeeze(dim=0)
        # label = torch.cat([label,1-label],dim=0)
        vol = vol*2 -1

        return vol,label,img_path

    def __len__(self):
        return self.id_list.__len__()

def nih_pancreas(batchsz,isshuffle=True,root_dir=None,transform=[],num_worker=1):
    panc_train_ds =  NIH_Pancreas(subset='train',root_dir=root_dir)
    train_loader = DataLoader(dataset=panc_train_ds,
                              batch_size= batchsz,
                              shuffle=isshuffle,
                              num_workers=num_worker)
    panc_test_ds = NIH_Pancreas(subset='test',root_dir=root_dir)
    test_loader = DataLoader(dataset=panc_test_ds,
                              batch_size=batchsz,
                              shuffle=isshuffle,
                              num_workers=num_worker)
    return train_loader,test_loader


def mayo_pancreas(batchsz,isshuffle=True,root_dir='.',transform=[],num_worker=1,subset='all'):
    panc_train_ds = Mayo_Pancreas(subset=subset, root_dir=root_dir)
    train_loader = DataLoader(dataset=panc_train_ds,
                              batch_size=batchsz,
                              shuffle=isshuffle,
                              num_workers=num_worker)
    return train_loader