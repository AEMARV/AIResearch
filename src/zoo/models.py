import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet,BasicBlock
from collections import OrderedDict
import torch.nn.functional as F

from src.layers import Fold
from src.layers import Sampler
import math
from src.utils import softmax
from monai.networks.nets import UNet


class ProbModel(nn.Module):
    def __init__(self, *args, input_shape=(1,3,1,1), **kwargs):
        super(ProbModel, self).__init__(*args, **kwargs)
        """ Noise should be broadcastable to input"""



        

class FCSimpleCIFAR10(nn.Module):

    def __init__(self,layers=12,classnum=-1,num_filter=64,filter_scale=1):
        super().__init__()
        num_filter = num_filter * filter_scale
        layer = torch.nn.Conv2d(3, num_filter, (3, 3), stride=(1, 1), padding=0)
        self.convs = [layer]
        self.fcs= []
        self.add_module('conv_'+'1',layer)
        for i in range(layers-1):
            layer = torch.nn.Conv2d(num_filter*2, num_filter, (3,3), stride=(1,1), padding=0)
            self.add_module('conv_'+str(i+2),layer)
            self.convs = self.convs +[layer]


        layer = torch.nn.Conv2d(num_filter*2, num_filter, (1, 1), stride=(1, 1), padding=0)
        self.add_module('fc_' + str(1), layer)
        self.fcs = self.fcs + [layer]

        layer = torch.nn.Conv2d(num_filter*2, classnum, (1, 1), stride=(1, 1), padding=0)
        self.add_module('fc_' + str(2), layer)
        self.fcs = self.fcs + [layer]


    def forward(self,x):

        nl = Fold()
        for i in range(self.convs.__len__()-1):
            x = self.convs[i](x)
            x = nl(x)
        x= self.convs[-1](x)
        x = F.max_pool2d(x,x.shape[2],stride=1)
        for i in range(self.fcs.__len__()-1):
            x= self.fcs[i](x)
            x= nl(x)
        x= self.fcs[-1](x)

        return x


class SimpleCIFAR10(nn.Module):

    def __init__(self,layers=12,classnum=-1,num_filter=128,filter_scale=1,init_coef=1):
        super().__init__()
        import math
        num_filter = math.ceil(num_filter * filter_scale)
        layer = torch.nn.Conv2d(3, num_filter, (3, 3), stride=(1, 1), padding=0)
        self.convs = [layer]
        self.add_module('conv_'+'1',layer)
        for i in range(layers-2):
            layer = torch.nn.Conv2d(num_filter*2, num_filter, (3,3), stride=(1,1), padding=0)
            self.add_module('conv_'+str(i+2),layer)
            self.convs = self.convs +[layer]

        layer = torch.nn.Conv2d(num_filter*2, classnum, (3,3), stride=(1,1), padding=0)
        self.add_module('conv_' + str(layers), layer)
        self.convs = self.convs +[layer]


        # self.add_module('conv1',self.conv1)
        # self.add_module('conv2', self.conv2)
        # self.add_module('conv3', self.conv3)
        # self.add_module('conv4', self.conv4)

    def forward(self,x):
        nl = Fold()
        for i in range(self.convs.__len__()-1):
            x = self.convs[i](x)
            x = nl(x)
        x= self.convs[-1](x)
        x = F.max_pool2d(x,x.shape[2],stride=1)
        return x


class BottleNet(nn.Module):
    def __init__(self, layers=12, classnum=10, num_filter=64, filter_scale=1,init_coef=1):
        num_filter = math.ceil(num_filter * filter_scale)
        filternums = (layers-1)*(num_filter,) + (classnum,)
        super().__init__()
        self.init_coef =init_coef

        k_sh = 5
        dilation =1


        self.blocks = []
        in_channel = 3
        # dilations = [x+1 for x in range(int(layers/2))] + [int((layers+1)/2)-x for x in range(int((layers+1)/2))]
        dilations = [1 for x in range(int(layers / 2))] + [1 for x in
                                                               range(int((layers + 1) / 2))]
        k_sh_list = 14*[3,] + 10*[1,]
        for i in range(layers):
            dilation = dilations[i]
            padding = int(dilation * (k_sh - 1) / 2)
            filternum = filternums[i]
            k_sh = k_sh_list[i]
            if i == layers-1:
                layer = torch.nn.Identity()
            else:
                layer = torch.nn.Conv2d(in_channel, filternum, (k_sh,k_sh),
                                        stride= (1, 1),
                                        dilation= (dilation,dilation),
                                        padding=(0,0)
                                        )
                layer.weight.data = layer.weight.data *init_coef
                self.add_module('conv_' + '%d' % i, layer)

            layer2 = torch.nn.Conv2d(in_channel, classnum, (1, 1),
                                    stride=(1, 1),
                                    padding=(0, 0)
                                    )
            layer2.weight.data = layer2.weight.data*init_coef
            self.add_module('skipconv_' + '%d'%i, layer2)

            block =(layer,layer2)
            self.blocks = self.blocks+ [block]
            in_channel = num_filter*2

    def forward(self, x):
        sh = x.shape
        nl = Fold()
        beta = 2
        # print("Shape of input", x.shape)
        y_min = -torch.zeros(1,device=x.device)#*torch.inf
        y_max = -torch.zeros(1,device=x.device)#*torch.inf
        for i in range(self.blocks.__len__()):
            y_temp = self.blocks[i][1](x) # skip layer'
            y_temp_max = ( beta*y_temp).logsumexp(dim=(2,3),keepdim=False)
            y_temp_min = (-beta*y_temp).logsumexp(dim=(2,3),keepdim=False)
            y_max = softmax(y_max, y_temp_max)
            y_min = softmax(y_min, y_temp_min)
            x = self.blocks[i][0](x) #type:torch.Tensor
            # x = torch.nn.functional.interpolate(x,sh[2:],mode='trilinear')
            # print("passed")
            x = nl(x)
        y = -y_min/beta - y_max/beta
        # y = y_max/beta
        return -y


class BottleNetMax(nn.Module):
    def __init__(self, layers=12, classnum=10, num_filter=64, filter_scale=1,init_coef=1):
        num_filter = math.ceil(num_filter * filter_scale)
        filternums = (layers-1)*(num_filter,) + (classnum,)
        super().__init__()
        self.init_coef =init_coef

        k_sh = 5
        dilation =1


        self.blocks = []
        in_channel = 3
        # dilations = [x+1 for x in range(int(layers/2))] + [int((layers+1)/2)-x for x in range(int((layers+1)/2))]
        dilations = [1 for x in range(int(layers / 2))] + [1 for x in
                                                               range(int((layers + 1) / 2))]
        self.input_noise_log_precision = torch.ones(1,3,1,1,requires_grad=True,dtype=torch.float)
        self.register_parameter('noise_precision',self.input_noise_log_precision)
        k_sh_list = 14*[3,] + 10*[1,]
        for i in range(layers):
            dilation = dilations[i]
            padding = int(dilation * (k_sh - 1) / 2)
            filternum = filternums[i]
            k_sh = k_sh_list[i]
            if i == layers-1:
                layer = torch.nn.Identity()
            else:
                layer = torch.nn.Conv2d(in_channel, filternum, (k_sh,k_sh),
                                        stride= (1, 1),
                                        dilation= (dilation,dilation),
                                        padding=(0,0)
                                        )
                layer.weight.data = layer.weight.data *init_coef
                self.add_module('conv_' + '%d' % i, layer)

            layer2 = torch.nn.Conv2d(in_channel, classnum, (1, 1),
                                    stride=(1, 1),
                                    padding=(0, 0)
                                    )
            layer2.weight.data = layer2.weight.data*init_coef
            self.add_module('skipconv_' + '%d'%i, layer2)

            block =(layer,layer2)
            self.blocks = self.blocks+ [block]
            in_channel = num_filter*2
    def add_noise(self,x):
        randoms= torch.randn_like(x)
        precision = self.input_noise_log_precision.exp()
        randoms = (precision*randoms).detach()
        energy = -((randoms/precision)**2)/2
        return randoms+ x, energy
    def forward(self, x,alpha=1):
        x,energy = self.add_noise(x)
        sh = x.shape
        nl = Fold()
        beta = 2
        # print("Shape of input", x.shape)
        y = -torch.zeros(1,device=x.device)#*torch.inf
        y_max = None
        y_min = None
        for i in range(self.blocks.__len__()):
            y_temp = self.blocks[i][1](x) # skip layer'
            y_temp_max = (alpha*y_temp).logsumexp(dim=(2,3),keepdim=False)
            y_temp_min = (-alpha*y_temp).logsumexp(dim=(2,3),keepdim=False)
            y_max = y_temp_max if y_max is None else softmax(y_max,y_temp_max)
            y_min = y_temp_min if y_min is None else softmax(y_min, y_temp_min)
            x = self.blocks[i][0](x) #type:torch.Tensor
            # x = torch.nn.functional.interpolate(x,sh[2:],mode='trilinear')
            # print("passed")
            x = nl(x)
        # y = y_max/beta
        return y_max/alpha,energy


''' Digital Models'''
class DPN_CIFAR10FC(nn.Module):

    def __init__(self,layers=12,classnum=-1,num_filter=64,filter_scale=1):
        super().__init__()
        num_filter = num_filter * filter_scale
        layer = torch.nn.Conv2d(3, num_filter, (3, 3), stride=(1, 1), padding=0)
        self.convs = [layer]
        self.fcs= []
        self.add_module('conv_'+'1',layer)
        for i in range(layers-1):
            layer = torch.nn.Conv2d(num_filter, num_filter, (3,3), stride=(1,1), padding=0)
            self.add_module('conv_'+str(i+2),layer)
            self.convs = self.convs +[layer]


        layer = torch.nn.Conv2d(num_filter, num_filter, (1, 1), stride=(1, 1), padding=0)
        self.add_module('fc_' + str(1), layer)
        self.fcs = self.fcs + [layer]

        layer = torch.nn.Conv2d(num_filter, classnum, (1, 1), stride=(1, 1), padding=0)
        self.add_module('fc_' + str(2), layer)
        self.fcs = self.fcs + [layer]


    def forward(self,x):

        nl = Sampler()
        energy = 0
        for i in range(self.convs.__len__()-1):
            x = self.convs[i](x)
            x,energy_temp = nl(x)
            energy = energy+energy_temp
        x= self.convs[-1](x)
        x = F.avg_pool2d(x,x.shape[2],stride=1)
        for i in range(self.fcs.__len__()-1):
            x= self.fcs[i](x)
            x, energy_temp = nl(x)
            energy = energy + energy_temp
        x= self.fcs[-1](x)

        return x,energy


class DPN_SimpleCIFAR10(nn.Module):

    def __init__(self,layers=12,classnum=-1,num_filter=64,filter_scale=1):
        super().__init__()
        import math
        num_filter = math.ceil(num_filter * filter_scale)
        layer = torch.nn.Conv2d(3, num_filter, (3, 3), stride=(1, 1), padding=0)
        self.convs = [layer]
        self.add_module('conv_'+'1',layer)
        for i in range(layers-2):
            layer = torch.nn.Conv2d(num_filter, num_filter, (3,3), stride=(1,1), padding=0)
            self.add_module('conv_'+str(i+2),layer)
            self.convs = self.convs +[layer]

        layer = torch.nn.Conv2d(num_filter, classnum, (3,3), stride=(1,1), padding=0)
        self.add_module('conv_' + str(layers), layer)
        self.convs = self.convs +[layer]


        # self.add_module('conv1',self.conv1)
        # self.add_module('conv2', self.conv2)
        # self.add_module('conv3', self.conv3)
        # self.add_module('conv4', self.conv4)

    def forward(self,x):
        nl = Sampler()
        energy=0
        for i in range(self.convs.__len__()-1):
            x = self.convs[i](x)
            x, energy_temp = nl(x)
            energy = energy + energy_temp
        x = self.convs[-1](x)
        x = F.max_pool2d(x,x.shape[2],stride=1)
        return x , energy


''' Medical Segmentation Models'''


class Panc_Segmentation(nn.Module):
    def __init__(self, layers=12, classnum=2, num_filter=64, filter_scale=1):
        num_filter = math.ceil(num_filter * filter_scale)
        filternums = (layers-1)*(num_filter,) + (classnum,)
        super().__init__()

        k_sh = 5
        dilation =2


        self.blocks = []
        in_channel = 1
        for i in range(layers):
            dilation = i+1
            padding = int(dilation * (k_sh - 1) / 2)
            filternum = filternums[i]
            if i == layers-1:
                layer = torch.nn.Identity()
            else:
                layer = torch.nn.Conv3d(in_channel, filternum, (k_sh,k_sh,k_sh),
                                        stride= (1, 1, 1),
                                        dilation= (dilation,dilation,dilation),
                                        padding=(padding,padding,padding)
                                        )
                layer.weight.data = layer.weight.data *0.001
                self.add_module('conv_' + '%d' % i, layer)

            layer2 = torch.nn.Conv3d(in_channel, classnum, (1, 1, 1),
                                    stride=(1, 1, 1),
                                    padding=(0, 0, 0)
                                    )
            layer2.weight.data = layer2.weight.data*0.001
            self.add_module('skipconv_' + '%d'%i, layer2)

            block =(layer,layer2)
            self.blocks = self.blocks+ [block]
            in_channel = num_filter*2



    def forward(self, x):
        sh = x.shape
        nl = Fold()
        # print("Shape of input", x.shape)
        y = -torch.ones(1,device=x.device)*torch.inf

        for i in range(self.blocks.__len__()):
            y_temp = self.blocks[i][1](x) # skip layer'

            y = softmax(y,y_temp)
            x = self.blocks[i][0](x) #type:torch.Tensor
            # x = torch.nn.functional.interpolate(x,sh[2:],mode='trilinear')
            # print("passed")
            x = nl(x)
        return y


class Panc_Seg_Bottled(nn.Module):
    def __init__(self, layers=12, classnum=2, num_filter=64, filter_scale=1,init_coef=1):
        num_filter = math.ceil(num_filter * filter_scale)
        filternums = (layers-1)*(num_filter,) + (classnum,)
        super().__init__()
        self.init_coef =init_coef

        k_sh = 5
        dilation =2


        self.blocks = []
        in_channel = 1
        dilations = [x+1 for x in range(int(layers/2))] + [int((layers+1)/2)-x for x in range(int((layers+1)/2))]
        for i in range(layers):
            dilation = dilations[i]
            padding = int(dilation * (k_sh - 1) / 2)
            filternum = filternums[i]
            if i == layers-1:
                layer = torch.nn.Identity()
            else:
                layer = torch.nn.Conv3d(in_channel, filternum, (k_sh,k_sh,k_sh),
                                        stride= (1, 1, 1),
                                        dilation= (dilation,dilation,dilation),
                                        padding=(padding,padding,padding)
                                        )

                layer.weight.data = layer.weight.data * init_coef
                self.add_module('conv_' + '%d' % i, layer)

            layer2 = torch.nn.Conv3d(in_channel, classnum, (1, 1, 1),
                                    stride=(1, 1, 1),
                                    padding=(0, 0, 0)
                                    )
            layer2.weight.data = layer2.weight.data
            self.add_module('skipconv_' + '%d'%i, layer2)

            block =(layer,layer2)
            self.blocks = self.blocks+ [block]
            in_channel = filternum*2



    def forward(self, x):
        sh = x.shape
        nl = Fold()
        # print("Shape of input", x.shape)
        y = torch.zeros(1, device=x.device)
        y_max = torch.zeros(1, device=x.device)
        y_min = torch.zeros(1, device=x.device)
        max_index= -1
        for i in range(self.blocks.__len__()):
            y_temp = self.blocks[i][1](x) # skip layer'
            y_temp_total = y_temp.logsumexp(dim=[x for x in range(y_temp.ndim)])
            y_total = y.logsumexp(dim=[x for x in range(y.ndim)])
            if y_total.item()<y_temp_total.item():
                max_index = i
            beta = 2
            y_max = softmax(y_max,beta*y_temp)
            y_min = softmax(y_min,-beta*y_temp)
            y = softmax(y,y_temp)
            x = self.blocks[i][0](x) #type:torch.Tensor
            # x = torch.nn.functional.interpolate(x,sh[2:],mode='trilinear')
            # print("passed")
            x = nl(x)
        print("most influencial:", max_index)
        return y_min/(-beta) - y_max/(beta)


class Panc_Seg_Bottled_FullScale_2D(nn.Module):
    def __init__(self, layers=12, classnum=2, num_filter=128, filter_scale=1,init_coef=1):
        num_filter = math.ceil(num_filter * filter_scale)
        filternums = (layers-1)*(num_filter,) + (classnum,)
        super().__init__()
        self.init_coef =init_coef

        k_sh = 5
        dilation =2


        self.blocks = []
        in_channel = 1
        dilations = [x+1 for x in range(int(layers/2))] + [int((layers+1)/2)-x for x in range(int((layers+1)/2))]
        for i in range(layers):
            dilation = dilations[i]
            padding = int(dilation * (k_sh - 1) / 2)
            filternum = filternums[i]
            if i == layers-1:
                layer = torch.nn.Identity()
            else:
                layer = torch.nn.Conv2d(in_channel, filternum, (k_sh,k_sh),
                                        stride= (1, 1),
                                        dilation= (dilation,dilation),
                                        padding=(padding,padding)
                                        )

                layer.weight.data = layer.weight.data * init_coef
                self.add_module('conv_' + '%d' % i, layer)

            layer2 = torch.nn.Conv2d(in_channel, classnum, (1, 1),
                                    stride=(1, 1),
                                    padding=(0, 0)
                                    )
            layer2.weight.data = layer2.weight.data
            self.add_module('skipconv_' + '%d'%i, layer2)

            block =(layer,layer2)
            self.blocks = self.blocks+ [block]
            in_channel = filternum*2



    def forward(self, x):
        beta = 2
        nl = Fold()
        # print("Shape of input", x.shape)
        y = torch.zeros(1, device=x.device)
        y_max = torch.zeros(1, device=x.device)
        y_min = torch.zeros(1, device=x.device)
        max_index= -1
        min_index= -1
        for i in range(self.blocks.__len__()):
            # print(x.shape)
            y_temp = self.blocks[i][1](x) # skip layer'
            # print(y_temp.shape)

            y_max = softmax(y_max,beta*y_temp)
            y_min = softmax(y_min,-beta*y_temp)
            # print(y_temp.mean(),y_max.mean(),y_min.mean())
            x = self.blocks[i][0](x) #type:torch.Tensor
            # x = torch.nn.functional.interpolate(x,sh[2:],mode='trilinear')
            # print("passed")
            x = nl(x)
        # print("max layer:, min layer", max_index, min_index)
        return y_min/(-beta) - y_max/(beta)

class Panc_Seg(nn.Module):
    def __init__(self, layers=12, classnum=2, num_filter=64, filter_scale=1,init_coef=1):
        num_filter = math.ceil(num_filter * filter_scale)
        filternums = (layers-1)*(num_filter,) + (classnum,)
        super().__init__()
        self.init_coef =init_coef

        k_sh = 5
        dilation =2


        self.blocks = []
        in_channel = 1
        dilations = [x+1 for x in range(int(layers/2))] + [int((layers+1)/2)-x for x in range(int((layers+1)/2))]
        for i in range(layers):
            dilation = 1
            padding = int(dilation * (k_sh - 1) / 2)
            filternum = filternums[i]
            if i == layers-1:
                layer = torch.nn.Identity()
            else:
                layer = torch.nn.Conv3d(in_channel, filternum, (k_sh,k_sh,k_sh),
                                        stride= (1, 1, 1),
                                        dilation= (dilation,dilation,dilation),
                                        padding=(padding,padding,padding)
                                        )
                layer.weight.data = layer.weight.data *init_coef
                self.add_module('conv_' + '%d' % i, layer)

            layer2 = torch.nn.Conv3d(in_channel, classnum, (1, 1, 1),
                                    stride=(1, 1, 1),
                                    padding=(0, 0, 0)
                                    )
            layer2.weight.data = layer2.weight.data*init_coef
            self.add_module('skipconv_' + '%d'%i, layer2)

            block =(layer,layer2)
            self.blocks = self.blocks+ [block]
            in_channel = filternum*2



    def forward(self, x):
        sh = x.shape
        nl = Fold()
        # print("Shape of input", x.shape)
        y = -torch.ones(1,device=x.device)*torch.inf

        for i in range(self.blocks.__len__()):
            y_temp = self.blocks[i][1](x) # skip layer'

            y = softmax(y,y_temp)
            x = self.blocks[i][0](x) #type:torch.Tensor
            # x = torch.nn.functional.interpolate(x,sh[2:],mode='trilinear')
            # print("passed")
            x = nl(x)
        return y
class Panc_Seg_Bottled_Sum(nn.Module):
    def __init__(self, layers=12, classnum=2, num_filter=64, filter_scale=1,init_coef=1):
        num_filter = math.ceil(num_filter * filter_scale)
        filternums = (layers-1)*(filter_scale*num_filter,) + (classnum,)
        super().__init__()
        self.init_coef =init_coef

        k_sh = 5
        dilation =2


        self.blocks = []
        in_channel = 1
        dilations = [x+1 for x in range(int(layers/2))] + [int((layers+1)/2)-x for x in range(int((layers+1)/2))]
        for i in range(layers):
            dilation = dilations[i]
            padding = int(dilation * (k_sh - 1) / 2)
            filternum = filternums[i]
            if i == layers-1:
                layer = torch.nn.Identity()
            else:
                layer = torch.nn.Conv3d(in_channel, filternum, (k_sh,k_sh,k_sh),
                                        stride= (1, 1, 1),
                                        dilation= (dilation,dilation,dilation),
                                        padding=(padding,padding,padding)
                                        )
                layer.weight.data = layer.weight.data *init_coef
                self.add_module('conv_' + '%d' % i, layer)

            layer2 = torch.nn.Conv3d(in_channel, classnum, (1, 1, 1),
                                    stride=(1, 1, 1),
                                    padding=(0, 0, 0)
                                    )
            layer2.weight.data = layer2.weight.data*init_coef
            self.add_module('skipconv_' + '%d'%i, layer2)

            block =(layer,layer2)
            self.blocks = self.blocks+ [block]
            in_channel = filternum*2



    def forward(self, x):
        sh = x.shape
        nl = Fold()
        # print("Shape of input", x.shape)
        y = -torch.zeros(1,device=x.device)

        for i in range(self.blocks.__len__()):
            y_temp = self.blocks[i][1](x) # skip layer'

            y = y/2+y_temp/2
            x = self.blocks[i][0](x) #type:torch.Tensor
            # x = torch.nn.functional.interpolate(x,sh[2:],mode='trilinear')
            # print("passed")
            x = nl(x)
        return y


def resnet_cifar_nmnist(layers=12, classnum=10, num_filter=64, filter_scale=1,init_coef=1,**kwargs):
    module = ResNet(BasicBlock,[layers//3,]*3 + [(layers%3)+1],num_classes=classnum)
    for params in module.parameters(recurse=True):
        params.data = params.data*init_coef
    return module


def unet(*args,**kwargs):
    network = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(8, 16, 32, 32, 16),
        strides=(1, 2, 2, 3),
        kernel_size=5,
        up_kernel_size=3,
        num_res_units=5,
        norm="batch"
    )
    return network