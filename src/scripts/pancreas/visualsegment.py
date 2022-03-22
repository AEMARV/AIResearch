import torch
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
from src.utils import hasnan
from src.data import nih_pancreas
import os
import monai.data.nifti_writer as nifti_writer

def print_results(path,model,dataset,mode:str):
    path = os.path.join('.',path)
    path = os.path.abspath(path)
    print(os.path.exists(path))

    filepath_true = os.path.join(path,mode +'_%d_true.nii.gz')
    filepath_pred = os.path.join(path,mode +'_%d_pred.nii.gz')
    filepath_vol = os.path.join(path,mode +'_%d_img.nii.gz')
    for i,data in enumerate(dataset):
        with torch.no_grad():
            if i > 10:
                break
            vol , label = data
            vol = vol.to('cuda:0')
            label = label.to('cuda:0')
            label = label[0:, 0:1, 0:]
            label_predicted = model(vol)
            if hasnan(label_predicted):
                break
            label_predicted= label_predicted == label_predicted.max(dim=1,keepdim=True)[0]
            label_predicted = label_predicted/label_predicted.sum(dim=1,keepdim=True)
            # label_predicted = label_predicted.softmax(dim=1)
            label_predicted = label_predicted[0:,0:1,0:].squeeze() # type:torch.Tensor


            # label_predicted = label_predicted - label_predicted.min()
            # label_predicted = label_predicted / label_predicted.max()
            # Convert labels
            nifti_writer.write_nifti(label_predicted.cpu().numpy(),filepath_pred%i)
            nifti_writer.write_nifti(label.squeeze().cpu().numpy(), filepath_true % i)
            nifti_writer.write_nifti(vol.squeeze().cpu().numpy(), filepath_vol % i)
            # label_predicted = label_predicted.to('cpu').numpy()
            # label = label.to('cpu').numpy()
            print('file %d'%i)
            # Write to File



if __name__ == '__main__':
    code_root = os.path.join('..','..','..')
    root_path = os.path.join('..','..','..','Results','Selected','nih_pancreas')
    visual_dir = 'visualized'
    data_train,data_test = nih_pancreas(1,isshuffle=False,root_dir=code_root)
    for model_dir in os.listdir(root_path):
        ### MAKE PATHs
        model_path = os.path.join(root_path, model_dir)
        visual_path = os.path.join(model_path,visual_dir)
        visual_path_train = os.path.join(visual_path,'train')
        visual_path_test = os.path.join(visual_path,'test')
        ### Create Folders
        os.makedirs(visual_path_train,exist_ok=True)
        os.makedirs(visual_path_test, exist_ok=True)
        ### Load model
        model = torch.load(os.path.join(model_path,'model.pth'))
        model.to('cuda:0')

        print_results(visual_path_train,model,data_train,'train')
        print_results(visual_path_test,model,data_test,'test')
        model.to('cpu')
