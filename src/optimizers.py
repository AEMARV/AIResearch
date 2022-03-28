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

