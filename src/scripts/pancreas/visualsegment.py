import torch
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
import torchvision
from src.utils import hasnan
from src.data import nih_pancreas
from src.data import mayo_pancreas
import os
import monai.data.nifti_writer as nifti_writer

def print_results_mayo(path,model,dataset,mode:str):
    segment_label = True
    path = os.path.join('.',path)
    path = os.path.abspath(path)
    print(os.path.exists(path))

    filepath_true = os.path.join(path,mode +'_%d_true.nii.gz')
    filepath_pred = os.path.join(path,mode +'_%d_pred.nii.gz')
    filepath_pred_2d = os.path.join(path, mode + '_%d_pred2d.png')
    filepath_vol = os.path.join(path,mode +'_%d_img.nii.gz')
    for i,data in enumerate(dataset):
        # with torch:
        if i > 2:
            break
        vol , label,_ = data
        vol = vol.to('cuda:0')#type: torch.Tensor
        if label.ndim!=2:
            segment_label =False
            label = label.to('cuda:0')
            label = label[0:, 0:1, 0:]
        init_vol = vol.clone().detach()
        vol.requires_grad = True
        vol.retain_grad()

        label_predicted = model(vol)
        init_label = label_predicted
        for j in range(20):

            energy= label_predicted.logsumexp(dim=1).mean()
            # energy = label_predicted.log_softmax(dim=1).max(dim=1)[0].mean()
            # energy = label_predicted.logsumexp(dim=1).logsumexp(dim=[0,1,2,3])
            norm = ((init_vol-vol)**2).sum()/100
            loss = energy - norm
            (loss).backward()
            print('loss:', loss.item(),'energy: ',energy.item(),'norm:', norm.item())
            vol.data = vol.data + vol.grad.data*20
            vol.grad.data = vol.grad.data*0
            # vol = vol.detach()
            # vol.requires_grad=True

            label_predicted = label_predicted == label_predicted.max(dim=1, keepdim=True)[0]
            label_predicted = label_predicted / label_predicted.sum(dim=1, keepdim=True)
            # label_predicted = label_predicted.softmax(dim=1)
            label_predicted = label_predicted[0:, 0:1, 0:].squeeze()  # type:torch.Tensor
            print('panc_vol: ',label_predicted.sum())
            label_predicted = model(vol)
        vol= vol.detach()
        if hasnan(label_predicted):
            break


        label_predicted= label_predicted == label_predicted.max(dim=1,keepdim=True)[0]
        label_predicted = label_predicted/label_predicted.sum(dim=1,keepdim=True)
        # label_predicted = label_predicted.softmax(dim=1)
        label_predicted = label_predicted[0:,0:1,0:].squeeze() # type:torch.Tensor

        label_predicted_2d= label_predicted.mean(dim=0)
        label_predicted_2d = label_predicted_2d- label_predicted_2d.min()
        label_predicted_2d = label_predicted_2d/ label_predicted_2d.max()
        vol = init_vol
        vol_2d = vol.squeeze().mean(dim=0).unsqueeze(dim=0)
        vol_2d = torch.cat([vol_2d,vol_2d,vol_2d],dim=0)
        vol_2d[0,0:] = vol_2d[0,0:] + label_predicted_2d
        # label_predicted = label_predicted - label_predicted.min()
        # label_predicted = label_predicted / label_predicted.max()
        # Convert labels
        nifti_writer.write_nifti(label_predicted.cpu().numpy(),filepath_pred%i)
        torchvision.utils.save_image(vol_2d.cpu(),filepath_pred_2d%i)
        if segment_label:
            nifti_writer.write_nifti(label.squeeze().cpu().numpy(), filepath_true % i)
        nifti_writer.write_nifti(vol.squeeze().cpu().numpy(), filepath_vol % i)
        # label_predicted = label_predicted.to('cpu').numpy()
        # label = label.to('cpu').numpy()
        print('file %d'%i)
        # Write to File
def print_results_nih(path,model,dataset,mode:str):
    segment_label = True
    path = os.path.join('.',path)
    path = os.path.abspath(path)
    print(os.path.exists(path))

    filepath_true = os.path.join(path,mode +'_%d_true.nii.gz')
    filepath_pred = os.path.join(path,mode +'_%d_pred.nii.gz')
    filepath_pred_2d = os.path.join(path, mode + '_%d_pred2d.png')
    filepath_vol = os.path.join(path,mode +'_%d_img.nii.gz')
    for i,data in enumerate(dataset):
        # with torch:
        if i > 2:
            break
        vol , label = data
        vol = vol.to('cuda:0')#type: torch.Tensor
        if label.ndim!=2:
            segment_label =True
            label = label.to('cuda:0')
            label = label[0:, 0:1, 0:]
        init_vol = vol.clone().detach()
        vol.requires_grad = True
        vol.retain_grad()

        label_predicted = model(vol)
        if hasnan(label_predicted):
            break


        label_predicted= label_predicted == label_predicted.max(dim=1,keepdim=True)[0]
        label_predicted = label_predicted/label_predicted.sum(dim=1,keepdim=True)
        # label_predicted = label_predicted.softmax(dim=1)
        label_predicted = label_predicted[0:,0:1,0:].squeeze() # type:torch.Tensor

        label_predicted_2d= label_predicted.mean(dim=0)
        label_predicted_2d = label_predicted_2d- label_predicted_2d.min()
        label_predicted_2d = label_predicted_2d/ label_predicted_2d.max()
        vol = init_vol
        vol_2d = vol.squeeze().mean(dim=0).unsqueeze(dim=0)
        vol_2d = torch.cat([vol_2d,vol_2d,vol_2d],dim=0)
        vol_2d[0,0:] = vol_2d[0,0:] + label_predicted_2d
        # label_predicted = label_predicted - label_predicted.min()
        # label_predicted = label_predicted / label_predicted.max()
        # Convert labels
        nifti_writer.write_nifti(label_predicted.cpu().numpy(),filepath_pred%i)
        torchvision.utils.save_image(vol_2d.cpu(),filepath_pred_2d%i)
        if segment_label:
            nifti_writer.write_nifti(label.squeeze().cpu().numpy(), filepath_true % i)
        nifti_writer.write_nifti(vol.squeeze().cpu().numpy(), filepath_vol % i)
        # label_predicted = label_predicted.to('cpu').numpy()
        # label = label.to('cpu').numpy()
        print('file %d'%i)
        # Write to File


if __name__ == '__main__':
    code_root = os.path.join('..','..','..')
    root_path = os.path.join('..','..','..','Results','SOTA','nih_pancreas')
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
        try:
            setup = torch.load(os.path.join(model_path,'result.dict'))['setup']
        except Exception:
            continue
        print(setup)
        model.to('cuda:0')

        print_results_nih(visual_path_train,model,data_test,'test')
        # print_results(visual_path_test,model,data_test,'test')
        model.to('cpu')
