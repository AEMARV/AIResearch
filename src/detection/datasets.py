import torch
from torch.utils.data.dataloader import DataLoader
import torchvision as tv
from src.globals import PATH_DATA
import torchvision.transforms.transforms as transforms
import fiftyone as fo
import fiftyone.zoo as foz
def coco(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.CocoDetection(PATH_DATA, train=True, download=True, transform=transform)
    testset = tv.datasets.CocoDetection(PATH_DATA, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

if __name__ == '__main__':
    dataset = foz.load_zoo_dataset("quickstart")
    session = fo.zoo.load_zoo_dataset("coco-2017", label_types=['detection'], dataset_dir="D:/CocoDataset")