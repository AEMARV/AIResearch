import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models.resnet as ResNet
import torch.nn.functional as F
from src.layers import Fold
from src.layers import Sampler
import math
from src.utils import softmax
import monai.networks.nets.unet as UNET
class PancModel(nn.Module):

    def __init__(self,layers=12,classnum=-1,num_filter=64,filter_scale=1):
        super().__init__()
        import math
        num_filter = math.ceil(num_filter * filter_scale)
        layer = torch.nn.Conv3d(1, num_filter, (5, 5), stride=(1, 1), padding=0)
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


def unet():
    UNET()