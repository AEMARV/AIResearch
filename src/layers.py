import torch
from torch.nn import Module

class Fold(Module):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)

    def forward(self,x):
        xp = x.relu()
        xn = (-x).relu()
        x= torch.cat([xp,xn],dim=1)
        return x

class GrayToRGB(Module):
    ''' Dataloader module'''
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        if x.shape[0]==1:
            x= torch.cat([x,x,x],dim=0)
        return x

class Sampler(Module):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def sample(self,energy:torch.Tensor ,dim):
        energy = energy.transpose(0,dim)
        prob = energy.softmax(dim=0).detach()
        cumprob = prob.cumsum(dim=0)
        randoms = torch.rand_like(prob[0:1,0:])
        sample = (cumprob > randoms).float()
        sample = (sample.cumsum(dim=0)==1).float()

        energy = sample * energy
        energy[energy != energy] = 0
        energy = energy.sum(dim=0,keepdim=True)
        energy = energy.transpose(0,dim)
        return sample, energy
    def sigmoid_sample(self,energy):
        prob = energy.sigmoid().detach()
        samples = prob> torch.rand_like(prob)
        energy = energy * (samples-0.5)
        samples = torch.cat([samples])
    def forward(self,x,numstates=2):
        init_shape = x.shape
        x = x.reshape([x.shape[0],numstates,-1,x.shape[2],x.shape[3]])
        sample, energy = self.sample(x,1)
        sample = sample.reshape(init_shape)
        energy = energy.squeeze(1)
        energy = energy.sum(dim=[1,2,3],keepdim=True)
        return sample, energy


