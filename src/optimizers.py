import torch
import src.utils
from torch.nn import Module


class SGD(Module):
    def __init__(self,lr=1,momentum=0.9,trained_epoch=0,l1=0,l2=0):
        self.trained_epochs = trained_epoch
        self.max_epoch= 150
        self.momentum= momentum
        self.l1= l1
        self.l2= l2
        self.lr = lr
    def inc_epoch(self):
        self.trained_epochs = self.trained_epochs+1

    def get_lr(self):
        return self.lr

    def step(self, model: torch.nn.Module):
        params = model.parameters(recurse=True)
        for param in params:
            if param.grad is not None:
                grad  = param.grad.data - param.data.sign()*self.l1 - param.data*self.l2
                param.data = param.data + grad.data * self.get_lr()
                param.grad.data = param.grad.data * self.momentum

class Stochastic_Optimizer(SGD):
    ''' Optimizes the model by sampling from the paramter space and evaluating the samples'''

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.input_params = None
        self.model_params = None
        self.current_loss= None

    def init_var_model(self,model):
        params = model.parameters(recurse=True)
        self.model_params=[]
        for param in params:
            self.model_params = self.model_params + [torch.ones_like(param)]



class Normalized_SGD(Module):
    def __init__(self,lr=1,momentum=0.9,trained_epoch=0,l1=0,l2=0):
        super().__init__()
        self.trained_epochs = trained_epoch
        self.max_epoch= 150
        self.momentum= momentum
        self.l1= l1
        self.l2= l2
        self.lr = lr
        self.distance= None
    def inc_epoch(self):
        self.trained_epochs = self.trained_epochs+1

    def get_lr(self):
        return self.lr

    def init_distance(self,params):
        self.distance = []
        for param in params:
            self.distance =  self.distance + [torch.ones_like(param)]
        # self.register_parameter('distance',self.distance)
    def step(self, model: torch.nn.Module):
        params = model.parameters(recurse=True)
        if self.distance is None:
            self.init_distance(params)
        for i, param in enumerate(params):
            if param.grad is not None:
                grad  = param.grad.data - param.data.sign()*self.l1 - param.data*self.l2
                grad.data = grad.data/(grad.data**2 + 1).sqrt()
                grad = grad.data * self.get_lr()#/self.distance[i].data
                # self.distance[i].data = self.distance[i].data + grad.data.abs()
                param.data = param.data + grad.data
                param.grad.data = param.grad.data * self.momentum

class Stochastic_Optimizer(Module):
    def __init__(self,lr=1,momentum=0.9,trained_epoch=0,l1=0,l2=0):
        super().__init__()
        self.trained_epochs = trained_epoch
        self.max_epoch= 150
        self.momentum= momentum
        self.l1= l1
        self.l2= l2
        self.lr = lr
        self.distance= None
    def inc_epoch(self):
        self.trained_epochs = self.trained_epochs+1

    def get_lr(self):
        return self.lr

    def init_distance(self,params):
        self.distance = []
        for param in params:
            self.distance = self.distance + [torch.ones_like(param)]
        # self.register_parameter('distance',self.distance)
    def step(self, model: torch.nn.Module):
        params = model.parameters(recurse=True)
        if self.distance is None:
            self.init_distance(params)
        for i, param in enumerate(params):
            if param.grad is not None:
                grad  = param.grad.data - param.data.sign()*self.l1 - param.data*self.l2
                grad.data = grad.data/(grad.data**2 + 1).sqrt()
                grad = grad.data * self.get_lr()#/self.distance[i].data
                # self.distance[i].data = self.distance[i].data + grad.data.abs()
                param.data = param.data + grad.data
                param.grad.data = param.grad.data * self.momentum
