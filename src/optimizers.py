from torch import Tensor
import math
from src.utils import *
class Optimizer(torch.nn.Module):

    def __init__(self,*args, l1=0, l2=0,**kwargs):
        super().__init__(*args,**kwargs)
        self.loss = None
        self.trained_epochs=0
        self.l1 = l1
        self.l2 = l2

    def augment_data(self,augment_rate,inputs,labels,num_class):
        augment_mask = torch.rand((inputs.shape[0],1),device=inputs.device)>augment_rate
        augment_mask = augment_mask.float()
        augment_mask_input = augment_mask.unsqueeze(2).unsqueeze(3)
        augment_mask_label = augment_mask.squeeze()
        augment_label = torch.randint(num_class,labels.size(),device=labels.device)
        augment_input = (torch.rand(inputs.shape,device=inputs.device)-0.5)*2
        augmented_input = inputs*augment_mask_input + augment_input*(1-augment_mask_input)
        augmented_label = labels*augment_mask_label + augment_label*(1-augment_mask_label)
        augmented_label = augmented_label.type_as(labels)
        return augmented_input,augmented_label

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

    def backward(self,loss):
        ''' Backward Wrapper. Disables backward calculation when in test mode'''
        if torch.is_grad_enabled():
            loss.backward()
        return

    def calc_grad(self,model,inputs,labels):
        ''' Calculates Grad if it is in training mode. Returns statistics as a dictionary'''
        raise Warning("Not implemented")
        pass

    def output_stats(self,model,inputs,labels,energy=None, logprob=None):
        if energy is None:
            energy,logprob = prob_wrapper(model(inputs))[0:2]
        label_logit = energy.log_softmax(dim=1)
        one_hot = self.label_to_onehot(energy, labels)
        prediction = energy == energy.max(dim=1,keepdim=True)[0]
        prediction = prediction/prediction.sum(dim=1)

        # Caculate the statistics

        acc = prediction == one_hot
        acc = acc.mean(dim=0).sum()

        label_likelihood = label_logit*one_hot
        label_likelihood[label_likelihood!= label_likelihood]=0
        label_likelihood.mean(dim=0).sum()

        stats = dict(label_likelihood=label_likelihood.item(),
                     acc = acc.item())
        return stats

    def label_to_onehot(self, output: Tensor, label):
        onehot = output.new_zeros(output.size())
        label = label.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        label = label.transpose(0, 1)
        onehot.scatter_(1, label, 1)
        return onehot.float()

    def hyper_normalize(self, model, inputs,logprob_prev, min, max, alpha=1):
        sample = (inputs * 0) + torch.rand_like(inputs) * (max - min) + min
        # inputs = torch.cat([sample,inputs],dim=0)
        output, logprob = prob_wrapper(model(inputs))[0:2]
        output = output + logprob
        output =torch.cat([output,logprob_prev],dim=0)
        output = output * alpha
        free_energy = output.logsumexp(dim=(0,1,2,3))/alpha - math.log(inputs.shape[0]*inputs.shape[1])
        return free_energy


class OptimizerV2(Optimizer):
    ''' Simillar Optimizer except hyper normalize does not include the dataset input'''

    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.loss = None
        self.trained_epochs=0

    def hyper_normalize(self, model, inputs, min, max, alpha=1):
        inputs= torch.cat([inputs,inputs],dim=0)
        sample = (inputs * 0) + torch.rand_like(inputs) * (max - min) + min
        output, logprob = prob_wrapper(model(sample))[0:2]
        output = output * alpha
        free_energy = output.logsumexp(dim=(0,1,2,3))/alpha - math.log(inputs.shape[0]*inputs.shape[1])
        return free_energy


class Joint_Likelihood_SGD(Optimizer):
    '''alpha,*args,lr=1,momentum=0.9,**kwargs
     Treats the network as an energy model.

     '''
    def __init__(self,alpha,*args,lr=1,momentum=0.9,**kwargs):
        # super(Joint_Likelihood_SGD,self).__init__(*args,**kwargs)
        super().__init__(*args,**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.alpha= alpha

    def get_lr(self):
        return self.lr

    def calc_grad(self,model:torch.nn.Module,inputs,labels):
        alpha = self.alpha
        stats = self.joint_likelihood(model,inputs,labels,alpha=alpha)
        return stats


class Likelihood_SGD(Optimizer):

    '''alpha,*args,lr=1,momentum=0.9,**kwargs
     Treats the network as an energy model.

     '''
    def __init__(self,alpha,*args,lr=1,momentum=0.9,**kwargs):
        # super(Joint_Likelihood_SGD,self).__init__(*args,**kwargs)
        super().__init__(*args,**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.alpha= alpha

    def get_lr(self):
        return self.lr

    def calc_grad(self,model:torch.nn.Module,inputs,labels):
        alpha = self.alpha
        stats = self.conditional_likelihood(model,inputs,labels,alpha=alpha)
        return stats


class Conditional_Cross(Optimizer):
    '''alpha,*args,lr=1,momentum=0.9,**kwargs
      Treats the network as an energy model.

      '''

    def conditional_cross(self, model, inputs, labels, alpha=1):  # ->type:dict
        output, logprob = prob_wrapper(model(inputs))[0:2]  # quick fix if the model is not probabilistic
        conditional_output = output.log_softmax(dim=1)
        free_energy = (alpha*output).logsumexp(dim=(1),keepdim=True)/alpha
        conditional_model_likelhood = output - free_energy
        prediction = output == output.max(dim=1, keepdim=True)[0]
        prediction = prediction / prediction.sum(dim=1, keepdim=True)
        one_hot = self.label_to_onehot(output, labels)
        energy = one_hot * output
        model_likelihood= one_hot*conditional_model_likelhood

        energy[energy != energy] = 0
        model_likelihood[model_likelihood != model_likelihood] = 0
        energy = energy.sum(dim=1, keepdim=True).mean(dim=0)
        model_likelihood = model_likelihood.sum(dim=1,keepdim=True).mean(dim=0)
        self.backward(model_likelihood)
        # Gather Stats
        label_likelihood = one_hot * conditional_output
        label_likelihood[label_likelihood != label_likelihood] = 0
        label_likelihood = label_likelihood.sum(dim=1).mean().cpu().item()
        energy = energy.cpu().item()
        free_energy = free_energy.logsumexp(dim=0) - math.log(inputs.shape[0])
        free_energy = free_energy.squeeze().cpu().item()
        model_likelihood = model_likelihood.cpu().item()
        acc = ((prediction * one_hot).mean(dim=0)).sum().cpu().item()
        stats = dict(acc=acc,
                     energy=energy,
                     free_energy=free_energy,
                     model_likelihood=model_likelihood,
                     label_likelihood=label_likelihood)
        return stats

    def __init__(self, alpha, classnum,augmentrate, *args, lr=1, momentum=0.9, **kwargs):
        # super(Joint_Likelihood_SGD,self).__init__(*args,**kwargs)
        '''

        :param alpha:
        :param classnum:
        :param augmentrate:
        :param args:
        :param lr:
        :param momentum:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.momentum = momentum
        self.alpha = alpha
        self.classnum = classnum
        self.augment_rate = augmentrate
    def get_lr(self):
        return self.lr

    def calc_grad(self, model: torch.nn.Module, inputs, labels):
        alpha = self.alpha
        if model.training:
            inputs, labels = self.augment_data(self.augment_rate, inputs, labels, self.classnum)
        stats = self.conditional_cross(model, inputs, labels, alpha=alpha)
        return stats




class Conditional_Probabilistic(Optimizer):
    '''alpha,*args,lr=1,momentum=0.9,**kwargs
      Treats the network as an energy model.

      '''

    def __init__(self, alpha, classnum,augmentrate, *args, lr=1, momentum=0.9, **kwargs):
        # super(Joint_Likelihood_SGD,self).__init__(*args,**kwargs)
        '''

        :param alpha:
        :param classnum:
        :param augmentrate:
        :param args:
        :param lr:
        :param momentum:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.momentum = momentum
        self.alpha = alpha
        self.classnum = classnum
        self.augment_rate = augmentrate

    def conditional_probabilistic(self, model, inputs, labels, alpha=1,prior_min=-1,prior_max=1):
        output, logprob = prob_wrapper(model(inputs))[0:2] # quick fix if the model is not probabilistic
        free_energy = self.hyper_normalize(model, inputs,prior_min,prior_max ,alpha=alpha)
        conditional_output = output.log_softmax(dim=1)
        prediction = output == output.max(dim=1, keepdim=True)[0]
        prediction = prediction / prediction.sum(dim=1,keepdim=True)
        one_hot = self.label_to_onehot(output, labels)
        energy = one_hot * output
        label_likelihood = one_hot * conditional_output
        label_likelihood[label_likelihood != label_likelihood] = 0
        label_likelihood = label_likelihood.sum(dim=1,keepdim=True)
        energy[energy != energy] = 0
        energy = energy.sum(dim=1,keepdim=True)
        model_likelihood = (label_likelihood).logsumexp(dim=0,keepdim=True)\
                           - (output*alpha).logsumexp(dim=(0,1),keepdim=True)/alpha
        self.backward(model_likelihood)
        # Gather Stats
        label_likelihood = one_hot * conditional_output
        label_likelihood[label_likelihood != label_likelihood] = 0
        label_likelihood = label_likelihood.sum(dim=1).mean().cpu().item()
        energy = energy.logsumexp(dim=0).squeeze().cpu().item()
        free_energy = free_energy.cpu().item()
        model_likelihood = model_likelihood.cpu().item()
        acc = ((prediction*one_hot).mean(dim=0)).sum().cpu().item()
        stats = dict(acc=acc,
                     energy=energy,
                     free_energy=free_energy,
                     model_likelihood=model_likelihood,
                     label_likelihood=label_likelihood)
        return stats

    def get_lr(self):
        return self.lr

    def calc_grad(self, model: torch.nn.Module, inputs, labels):
        alpha = self.alpha
        if model.training:
            inputs, labels = self.augment_data(self.augment_rate, inputs, labels, self.classnum)
        stats = self.conditional_probabilistic(model, inputs, labels, alpha=alpha)
        return stats


class Joint_Probabilistic(Optimizer):
    '''alpha,*args,lr=1,momentum=0.9,**kwargs
     Treats the network as an energy model.
     The validation set is included as prior
     The number of validation set is fixed.
     The upgrade of this optimizer is to optimize the size of validation.
     Cross Entropy is not used, Meaning that the gradients of samples are weighted.

     '''
    def __init__(self,alpha,classnum,augment_rate,*args,lr=1,momentum=0.9,**kwargs):
        '''

        :param alpha:
        :param classnum:
        :param augment_rate: a float between 0:1
        :param args:
        :param lr:
        :param momentum:
        :param kwargs:
        '''
        # super(Joint_Likelihood_SGD,self).__init__(*args,**kwargs)
        super().__init__(*args,**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.alpha= alpha
        self.classnum=classnum
        self.augment_rate= augment_rate

    def get_lr(self):
        return self.lr

    def joint_probabilistic(self, model, inputs, labels, alpha=1, prior_min=-1, prior_max=1):
        output, logprob = prob_wrapper(model(inputs))[0:2]  # quick fix if the model is not probabilistic
        # print(logprob.shape)
        output = output+logprob
        free_energy = self.hyper_normalize(model, inputs, output, prior_min, prior_max, alpha=alpha)
        conditional_output = output.log_softmax(dim=1)
        prediction = output == output.max(dim=1, keepdim=True)[0]
        prediction = prediction / prediction.sum(dim=1, keepdim=True)
        one_hot = self.label_to_onehot(output, labels)
        energy = one_hot * output
        energy[energy != energy] = 0
        energy = energy.sum(dim=1, keepdim=True).logsumexp(dim=0)
        model_likelihood = energy - free_energy
        self.backward(model_likelihood)
        # Gather Stats
        label_likelihood = one_hot * conditional_output
        label_likelihood[label_likelihood != label_likelihood] = 0
        label_likelihood = label_likelihood.sum(dim=1).mean().cpu().item()
        energy = energy.cpu().item()
        free_energy = free_energy.cpu().item()
        model_likelihood = model_likelihood.cpu().item()
        acc = ((prediction * one_hot).mean(dim=0)).sum().cpu().item()
        stats = dict(acc=acc,
                     energy=energy,
                     free_energy=free_energy,
                     model_likelihood=model_likelihood,
                     label_likelihood=label_likelihood)
        return stats


    def augment_data(self,augment_rate,inputs,labels,num_class):
        augment_mask = torch.rand((inputs.shape[0],1),device=inputs.device)>augment_rate
        augment_mask = augment_mask.float()
        augment_mask_input = augment_mask.unsqueeze(2).unsqueeze(3)
        augment_mask_label = augment_mask.squeeze()
        augment_label = torch.randint(num_class,labels.size(),device=labels.device)
        augment_input = (torch.rand(inputs.shape,device=inputs.device)-0.5)*2
        augmented_input = inputs*augment_mask_input + augment_input*(1-augment_mask_input)
        augmented_label = labels*augment_mask_label + augment_label*(1-augment_mask_label)
        augmented_label = augmented_label.type_as(labels)
        return augmented_input,augmented_label

    def calc_grad(self,model:torch.nn.Module,inputs,labels):
        alpha = self.alpha
        if model.training:
            inputs,labels = self.augment_data(self.augment_rate,inputs,labels,self.classnum)
        stats = self.joint_probabilistic(model,inputs,labels,alpha=alpha)
        return stats



class Joint_ProbabilisticV2(OptimizerV2):
    '''alpha,*args,lr=1,momentum=0.9,**kwargs
     Treats the network as an energy model.
     The validation set is included as prior
     The number of validation set is fixed.
     The upgrade of this optimizer is to optimize the size of validation.
     Cross Entropy is not used, Meaning that the gradients of samples are weighted.
     Simillar to V1 except the sampling for regularization does not include the input.
     '''
    def __init__(self,alpha,classnum,augment_rate,*args,lr=1,momentum=0.9,**kwargs):
        '''

        :param alpha:
        :param classnum:
        :param augment_rate: a float between 0:1
        :param args:
        :param lr:
        :param momentum:
        :param kwargs:
        '''
        # super(Joint_Likelihood_SGD,self).__init__(*args,**kwargs)
        super().__init__(*args,**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.alpha= alpha
        self.classnum=classnum
        self.augment_rate= augment_rate

    def get_lr(self):
        return self.lr

    def joint_probabilistic(self, model, inputs, labels, alpha=1, prior_min=-1, prior_max=1):
        output, logprob = prob_wrapper(model(inputs))[0:2]  # quick fix if the model is not probabilistic
        free_energy = self.hyper_normalize(model, inputs, prior_min, prior_max, alpha=alpha)
        conditional_output = output.log_softmax(dim=1)
        prediction = output == output.max(dim=1, keepdim=True)[0]
        prediction = prediction / prediction.sum(dim=1, keepdim=True)
        one_hot = self.label_to_onehot(output, labels)
        energy = one_hot * output
        energy[energy != energy] = 0
        energy = energy.sum(dim=1, keepdim=True).logsumexp(dim=0)
        model_likelihood = energy - free_energy
        self.backward(model_likelihood)
        # Gather Stats
        label_likelihood = one_hot * conditional_output
        label_likelihood[label_likelihood != label_likelihood] = 0
        label_likelihood = label_likelihood.sum(dim=1).mean().cpu().item()
        energy = energy.cpu().item()
        free_energy = free_energy.cpu().item()
        model_likelihood = model_likelihood.cpu().item()
        acc = ((prediction * one_hot).mean(dim=0)).sum().cpu().item()
        stats = dict(acc=acc,
                     energy=energy,
                     free_energy=free_energy,
                     model_likelihood=model_likelihood,
                     label_likelihood=label_likelihood)
        return stats


    def augment_data(self,augment_rate,inputs,labels,num_class):
        augment_mask = torch.rand((inputs.shape[0],1),device=inputs.device)>augment_rate
        augment_mask = augment_mask.float()
        augment_mask_input = augment_mask.unsqueeze(2).unsqueeze(3)
        augment_mask_label = augment_mask.squeeze()
        augment_label = torch.randint(num_class,labels.size(),device=labels.device)
        augment_input = (torch.rand(inputs.shape,device=inputs.device)-0.5)*2
        augmented_input = inputs*augment_mask_input + augment_input*(1-augment_mask_input)
        augmented_label = labels*augment_mask_label + augment_label*(1-augment_mask_label)
        augmented_label = augmented_label.type_as(labels)
        return augmented_input,augmented_label

    def calc_grad(self,model:torch.nn.Module,inputs,labels):
        alpha = self.alpha
        if model.training:
            inputs,labels = self.augment_data(self.augment_rate,inputs,labels,self.classnum)
        stats = self.joint_probabilistic(model,inputs,labels,alpha=alpha)
        return stats



class Joint_Variable_Augmentation(Optimizer):
    def __init__(self,alpha,beta,classnum,**kwargs):
        super().__init__(alpha,classnum,)
        self.augment_rate= torch.zeros(1,requires_grad=True)
        self.register_parameter('aug_rate',self.input_logit_rate)

    def augment_data(self,augment_rate,inputs,labels,num_class):

        augment_mask = torch.rand((inputs.shape[0],1),device=inputs.device)>augment_rate
        augment_mask = augment_mask.float()
        augment_mask_input = augment_mask.unsqueeze(2).unsqueeze(3)
        augment_mask_label = augment_mask.squeeze()
        augment_label = torch.randint(num_class,labels.size(),device=labels.device)
        augment_input = (torch.rand(inputs.shape,device=inputs.device)-0.5)*2
        augmented_input = inputs*augment_mask_input + augment_input*(1-augment_mask_input)
        augmented_label = labels*augment_mask_label + augment_label*(1-augment_mask_label)
        augmented_label = augmented_label.type_as(labels)
        return augmented_input,augmented_label,augment_mask

    def joint_augment(self,model,inputs,labels,classnum,alpha=2,prior_min=-1,prior_max=1):
        if torch.is_grad_enabled():
            augment_prob = torch.sigmoid(self.augment_rate)
            inputs,labels,augment_mask = self.augment_data(self.augment_prob,inputs,labels,classnum)
            # augment_logprob =
        output, logprob = prob_wrapper(model(inputs))[0:2] # quick fix if the model is not probabilistic
        free_energy = self.hyper_normalize(model, inputs,prior_min,prior_max ,alpha=alpha)
        conditional_output = output.log_softmax(dim=1)
        prediction = output == output.max(dim=1, keepdim=True)[0]
        prediction = prediction / prediction.sum(dim=1,keepdim=True)
        one_hot = self.label_to_onehot(output, labels)
        energy = one_hot * output
        energy[energy != energy] = 0
        energy = energy.sum(dim=1,keepdim=True).mean(dim=0)
        model_likelihood = energy - free_energy
        self.backward(model_likelihood)
        # Gather Stats
        label_likelihood = one_hot * conditional_output
        label_likelihood[label_likelihood!= label_likelihood] = 0
        label_likelihood = label_likelihood.sum(dim=1).mean().cpu().item()
        energy = energy.cpu().item()
        free_energy = free_energy.cpu().item()
        model_likelihood = model_likelihood.cpu().item()
        acc = ((prediction*one_hot).mean(dim=0)).sum().cpu().item()
        stats = dict(acc=acc,
                     energy=energy,
                     free_energy=free_energy,
                     model_likelihood=model_likelihood,
                     label_likelihood=label_likelihood)
        return stats

    def calc_grad(self,model:torch.nn.Module,inputs,labels):
        alpha = self.alpha
        if model.training:
            inputs, labels = self.augment_data(self.augment_rate, inputs, labels, self.classnum)

        stats = self.joint_likelihood(model, inputs, labels, alpha=alpha)
        return stats


class Joint_Subset(Optimizer):
    def __init__(self, alpha, classnum, augment_rate, *args, lr=1, momentum=0.9, **kwargs):
        '''

        :param alpha:
        :param classnum:
        :param augment_rate: a float between 0:1
        :param args:
        :param lr:
        :param momentum:
        :param kwargs:
        '''
        # super(Joint_Likelihood_SGD,self).__init__(*args,**kwargs)
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.momentum = momentum
        self.alpha = alpha
        self.classnum = classnum
        self.augment_rate = augment_rate

    def hyper_normalize(self, model, inputs, min, max, alpha=1):
        sample = (inputs * 0) + torch.rand_like(inputs) * (max - min) + min
        # inputs = torch.cat([sample,inputs],dim=0)
        output, logprob = prob_wrapper(model(inputs))[0:2]
        output = output * alpha
        free_energy = output.logsumexp(dim=(0,1,2,3))/alpha - math.log(inputs.shape[0]*inputs.shape[1])
        return free_energy

    def calc_grad(self,model:torch.nn.Module,inputs,labels):
        alpha = self.alpha
        if model.training:
            inputs,labels = self.augment_data(self.augment_rate,inputs,labels,self.classnum)
        stats = self.joint_subset(model,inputs,labels,alpha=alpha)
        return stats

    def joint_subset(self, model, inputs, labels, alpha=1, prior_min=-1, prior_max=1):
        output, logprob = prob_wrapper(model(inputs))[0:2]  # quick fix if the model is not probabilistic
        free_energy = self.hyper_normalize(model, inputs, prior_min, prior_max, alpha=alpha)
        conditional_output = output.log_softmax(dim=1)
        prediction = output == output.max(dim=1, keepdim=True)[0]
        prediction = prediction / prediction.sum(dim=1, keepdim=True)
        one_hot = self.label_to_onehot(output, labels)
        energy =output - one_hot.log()
        # print(energy)
        # energy[energy != energy] = 0
        energy = (-alpha*energy).logsumexp(dim=1, keepdim=True).logsumexp(dim=0)/(-alpha)
        model_likelihood = energy - free_energy
        self.backward(model_likelihood)
        # Gather Stats
        label_likelihood = one_hot * conditional_output
        label_likelihood[label_likelihood != label_likelihood] = 0
        label_likelihood = label_likelihood.sum(dim=1).mean().cpu().item()
        energy = energy.cpu().item()
        free_energy = free_energy.cpu().item()
        model_likelihood = model_likelihood.cpu().item()
        acc = ((prediction * one_hot).mean(dim=0)).sum().cpu().item()
        stats = dict(acc=acc,
                     energy=energy,
                     free_energy=free_energy,
                     model_likelihood=model_likelihood,
                     label_likelihood=label_likelihood)
        return stats
    def joint_subset_v2(self, model, inputs, labels, alpha=1, prior_min=-1, prior_max=1):
        output, logprob = prob_wrapper(model(inputs))[0:2]  # quick fix if the model is not probabilistic
        free_energy = self.hyper_normalize(model, inputs, prior_min, prior_max, alpha=alpha)
        conditional_output = output.log_softmax(dim=1)
        prediction = output == output.max(dim=1, keepdim=True)[0]
        prediction = prediction / prediction.sum(dim=1, keepdim=True)
        one_hot = self.label_to_onehot(output, labels)
        energy =output - one_hot.log()
        beta = -alpha
        prob = (energy*(beta)) - (energy*(beta)).logsumexp(dim=0,keepdim=True).logsumexp(dim=1,keepdim=True)
        prob = prob.exp().detach()
        energy = (prob*energy)
        energy[energy!=energy] = 0
        energy = energy.sum(dim=1,keepdim=True).sum(dim=0)
        energy = energy.squeeze()
        # print(energy)
        # energy[energy != energy] = 0
        # energy = (-alpha*energy).logsumexp(dim=1, keepdim=True).logsumexp(dim=0)/(-alpha)
        model_likelihood = energy - free_energy
        self.backward(model_likelihood)
        # Gather Stats
        label_likelihood = one_hot * conditional_output
        label_likelihood[label_likelihood != label_likelihood] = 0
        label_likelihood = label_likelihood.sum(dim=1).mean().cpu().item()
        energy = energy.cpu().item()
        free_energy = free_energy.cpu().item()
        model_likelihood = model_likelihood.cpu().item()
        acc = ((prediction * one_hot).mean(dim=0)).sum().cpu().item()
        stats = dict(acc=acc,
                     energy=energy,
                     free_energy=free_energy,
                     model_likelihood=model_likelihood,
                     label_likelihood=label_likelihood)
        return stats

## Segmentations
class Subset_Seg(Optimizer):
    '''alpha,*args,lr=1,momentum=0.9,**kwargs
     Treats the network as an energy model.
     The validation set is included as prior
     The number of validation set is fixed.
     The upgrade of this optimizer is to optimize the size of validation.

     '''
    def __init__(self,alpha,beta,classnum,*args,lr=1,momentum=0.9,**kwargs):
        '''

        :param alpha:
        :param classnum:
        :param augment_rate: a float between 0:1
        :param args:
        :param lr:
        :param momentum:
        :param kwargs:
        '''
        # super(Joint_Likelihood_SGD,self).__init__(*args,**kwargs)
        super().__init__(*args,**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.alpha= alpha
        self.beta = beta
        self.classnum=classnum
        self.augment_rate= 0

    def hyper_normalize(self, model, inputs,logprob_prev, min, max, alpha=1):
        sample = torch.rand_like(inputs) * (max - min) + min
        output, logprob = prob_wrapper(model(sample))[0:2]
        free_energy_input = (logprob_prev*alpha).logsumexp(dim=list(range(logprob_prev.dim())))
        free_energy_sample = (output*alpha).logsumexp(dim=list(range(logprob_prev.dim())))
        free_energy = softmax(free_energy_input,free_energy_sample)/alpha
        return free_energy
    def scores(self,label_true,label_pred):
        label_true = label_true[0:,0:1,0:]
        label_pred = label_pred[0:,0:1,0:]
        intersect = (label_pred*label_true).sum()
        sum_measure = (label_true+label_pred).sum()
        dice = 2*intersect/sum_measure
        IOU = intersect/(sum_measure-intersect)
        return dice.detach(),IOU.detach()


    def joint_cross(self, model, inputs, labels, alpha=1,prior_min=-1,prior_max=1):
        output, logprob = prob_wrapper(model(inputs))[0:2] # quick fix if the model is not probabilistic
        free_energy = self.hyper_normalize(model, inputs, output, prior_min, prior_max, alpha=self.alpha)
        conditional_output = output.log_softmax(dim=1)
        label_prediction = labels == labels.max(dim=1,keepdim=True)[0]
        label_prediction = label_prediction/label_prediction.sum(dim=1,keepdim=True)
        prediction = output == output.max(dim=1, keepdim=True)[0]
        prediction = prediction / prediction.sum(dim=1,keepdim=True)
        # one_hot = self.label_to_onehot(output, labels)
        energy = output - labels.log()
        energy = -(-self.alpha *energy).logsumexp(list(range(energy.dim())))/self.alpha
        # energy [energy != energy] = 0
        model_likelihood = energy - free_energy
        self.backward(model_likelihood)
        # Gather Stats
        dice,IOU = self.scores(labels,prediction)
        label_likelihood = labels * conditional_output
        label_likelihood[label_likelihood!= label_likelihood] = 0
        label_likelihood = label_likelihood.sum(dim=1).mean().cpu().item()
        energy = energy.cpu().item()
        free_energy = free_energy.cpu().item()
        model_likelihood = model_likelihood.cpu().item()
        acc = ((prediction*label_prediction).sum(dim=1)).mean().cpu().item()
        dice = dice.cpu().item()
        IOU = IOU.cpu().item()
        stats = dict(acc=acc,
                     energy=energy,
                     free_energy=free_energy,
                     dice= dice,
                     IOU = IOU,
                     model_likelihood=model_likelihood,
                     label_likelihood=label_likelihood)
        return stats

    def get_lr(self):
        return self.lr

    def augment_data(self,augment_rate,inputs,labels,num_class):
        augment_mask = torch.rand((inputs.shape[0],1),device=inputs.device)>augment_rate
        augment_mask = augment_mask.float()
        augment_mask_input = augment_mask.unsqueeze(2).unsqueeze(3)
        augment_mask_label = augment_mask.squeeze()
        augment_label = torch.randint(num_class,labels.size(),device=labels.device)
        augment_input = (torch.rand(inputs.shape,device=inputs.device)-0.5)*2
        augmented_input = inputs*augment_mask_input + augment_input*(1-augment_mask_input)
        augmented_label = labels*augment_mask_label + augment_label*(1-augment_mask_label)
        augmented_label = augmented_label.type_as(labels)
        return augmented_input,augmented_label

    def calc_grad(self,model:torch.nn.Module,inputs,labels):
        alpha = self.alpha
        stats = self.joint_cross(model,inputs,labels,alpha=alpha)
        return stats

#TODO:Not Implemented
class Cross_Joint_Seg(Optimizer):
    '''alpha,*args,lr=1,momentum=0.9,**kwargs
     Treats the network as an energy model.
     The validation set is included as prior
     The number of validation set is fixed.
     The upgrade of this optimizer is to optimize the size of validation.

     '''
    def __init__(self,alpha,beta,classnum,*args,lr=1,momentum=0.9,**kwargs):
        '''

        :param alpha:
        :param classnum:
        :param augment_rate: a float between 0:1
        :param args:
        :param lr:
        :param momentum:
        :param kwargs:
        '''
        # super(Joint_Likelihood_SGD,self).__init__(*args,**kwargs)
        super().__init__(*args,**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.alpha= alpha
        self.beta = beta
        self.classnum=classnum
        self.augment_rate= 0

    def hyper_normalize(self, model, inputs,logprob_prev, min, max, alpha=1):
        sample = torch.rand_like(inputs) * (max - min) + min
        output, logprob = prob_wrapper(model(sample))[0:2]
        free_energy_input = (logprob_prev*alpha).logsumexp(dim=list(range(logprob_prev.dim())))
        free_energy_sample = (output*alpha).logsumexp(dim=list(range(logprob_prev.dim())))
        free_energy = softmax(free_energy_input,free_energy_sample)/alpha
        return free_energy
    def scores(self,label_true,label_pred):
        label_true = label_true[0:,0:1,0:]
        label_pred = label_pred[0:,0:1,0:]
        intersect = (label_pred*label_true).sum()
        sum_measure = (label_true+label_pred).sum()
        dice = 2*intersect/sum_measure
        IOU = intersect/(sum_measure-intersect)
        return dice.detach(),IOU.detach()


    def joint_cross(self, model, inputs, labels, alpha=1,prior_min=-1,prior_max=1):
        output, logprob = prob_wrapper(model(inputs))[0:2] # quick fix if the model is not probabilistic
        free_energy = self.hyper_normalize(model, inputs, output, prior_min, prior_max, alpha=self.alpha)
        conditional_output = output.log_softmax(dim=1)
        label_prediction = labels == labels.max(dim=1,keepdim=True)[0]
        label_prediction = label_prediction/label_prediction.sum(dim=1,keepdim=True)
        prediction = output == output.max(dim=1, keepdim=True)[0]
        prediction = prediction / prediction.sum(dim=1,keepdim=True)
        # one_hot = self.label_to_onehot(output, labels)
        energy = output - labels.log()
        energy = -(-self.alpha *energy).logsumexp(list(range(energy.dim())))/self.alpha
        # energy [energy != energy] = 0
        model_likelihood = energy - free_energy
        self.backward(model_likelihood)
        # Gather Stats
        dice,IOU = self.scores(labels,prediction)
        label_likelihood = labels * conditional_output
        label_likelihood[label_likelihood!= label_likelihood] = 0
        label_likelihood = label_likelihood.sum(dim=1).mean().cpu().item()
        energy = energy.cpu().item()
        free_energy = free_energy.cpu().item()
        model_likelihood = model_likelihood.cpu().item()
        acc = ((prediction*label_prediction).sum(dim=1)).mean().cpu().item()
        dice = dice.cpu().item()
        IOU = IOU.cpu().item()
        stats = dict(acc=acc,
                     energy=energy,
                     free_energy=free_energy,
                     dice= dice,
                     IOU = IOU,
                     model_likelihood=model_likelihood,
                     label_likelihood=label_likelihood)
        return stats

    def get_lr(self):
        return self.lr

    def augment_data(self,augment_rate,inputs,labels,num_class):
        augment_mask = torch.rand((inputs.shape[0],1),device=inputs.device)>augment_rate
        augment_mask = augment_mask.float()
        augment_mask_input = augment_mask.unsqueeze(2).unsqueeze(3)
        augment_mask_label = augment_mask.squeeze()
        augment_label = torch.randint(num_class,labels.size(),device=labels.device)
        augment_input = (torch.rand(inputs.shape,device=inputs.device)-0.5)*2
        augmented_input = inputs*augment_mask_input + augment_input*(1-augment_mask_input)
        augmented_label = labels*augment_mask_label + augment_label*(1-augment_mask_label)
        augmented_label = augmented_label.type_as(labels)
        return augmented_input,augmented_label

    def calc_grad(self,model:torch.nn.Module,inputs,labels):
        alpha = self.alpha
        stats = self.joint_cross(model,inputs,labels,alpha=alpha)
        return stats

class Cross_Cond_Seg(Optimizer):
    pass


class Subset_Seg_Balanced(Optimizer):
    '''alpha,*args,lr=1,momentum=0.9,**kwargs
     Treats the network as an energy model.
     The validation set is included as prior
     The number of validation set is fixed.
     The upgrade of this optimizer is to optimize the size of validation.

     '''
    def __init__(self,alpha,beta,classnum,*args,lr=1,momentum=0.9,**kwargs):
        '''

        :param alpha:
        :param classnum:
        :param augment_rate: a float between 0:1
        :param args:
        :param lr:
        :param momentum:
        :param kwargs:
        '''
        # super(Joint_Likelihood_SGD,self).__init__(*args,**kwargs)
        super().__init__(*args,**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.alpha= alpha
        self.beta = beta
        self.classnum=classnum
        self.augment_rate= 0

    def hyper_normalize(self, model, inputs,logprob_prev, min, max, alpha=1):
        sample = torch.rand_like(inputs) * (max - min) + min
        output, logprob = prob_wrapper(model(sample))[0:2]
        free_energy_input = (logprob_prev*alpha).logsumexp(dim=list(range(logprob_prev.dim())))
        free_energy_sample = (output*alpha).logsumexp(dim=list(range(logprob_prev.dim())))
        free_energy = softmax(free_energy_input,free_energy_sample)/alpha
        return free_energy
    def scores(self,label_true,label_pred):
        label_true = label_true[0:,0:1,0:]
        label_pred = label_pred[0:,0:1,0:]
        intersect = (label_pred*label_true).sum()
        sum_measure = (label_true+label_pred).sum()
        dice = 2*intersect/sum_measure
        IOU = intersect/(sum_measure-intersect)
        return dice.detach(),IOU.detach()


    def joint_cross(self, model, inputs, labels, alpha=1,prior_min=-1,prior_max=1):

        output, logprob = prob_wrapper(model(inputs))[0:2] # quick fix if the model is not probabilistic
        free_energy = self.hyper_normalize(model, inputs, output, prior_min, prior_max, alpha=self.alpha)
        conditional_output = output.log_softmax(dim=1)
        label_prediction = labels == labels.max(dim=1,keepdim=True)[0]
        label_prediction = label_prediction/label_prediction.sum(dim=1,keepdim=True)
        label_prediction[0:, 0:1] = label_prediction[0:, 0:1] * (label_prediction[0:, 1:].sum() / label_prediction[0:, 0:1].sum())
        label_prediction = label_prediction/label_prediction.sum()
        prediction = output == output.max(dim=1, keepdim=True)[0]
        prediction = prediction / prediction.sum(dim=1,keepdim=True)
        # one_hot = self.label_to_onehot(output, labels)
        energy = output - label_prediction.log()
        energy = -(-self.alpha *energy).logsumexp(list(range(energy.dim())))/self.alpha
        # energy [energy != energy] = 0
        model_likelihood = energy - free_energy
        self.backward(model_likelihood)
        # Gather Stats
        dice,IOU = self.scores(labels,prediction)
        label_likelihood = labels * conditional_output
        label_likelihood[label_likelihood!= label_likelihood] = 0
        label_likelihood = label_likelihood.sum(dim=1).mean().cpu().item()
        energy = energy.cpu().item()
        free_energy = free_energy.cpu().item()
        model_likelihood = model_likelihood.cpu().item()
        acc = ((prediction*label_prediction).sum(dim=1)).sum().cpu().item()
        dice = dice.cpu().item()
        IOU = IOU.cpu().item()
        stats = dict(acc=acc,
                     energy=energy,
                     free_energy=free_energy,
                     dice= dice,
                     IOU = IOU,
                     model_likelihood=model_likelihood,
                     label_likelihood=label_likelihood)
        return stats

    def get_lr(self):
        return self.lr

    def augment_data(self,augment_rate,inputs,labels,num_class):
        augment_mask = torch.rand((inputs.shape[0],1),device=inputs.device)>augment_rate
        augment_mask = augment_mask.float()
        augment_mask_input = augment_mask.unsqueeze(2).unsqueeze(3)
        augment_mask_label = augment_mask.squeeze()
        augment_label = torch.randint(num_class,labels.size(),device=labels.device)
        augment_input = (torch.rand(inputs.shape,device=inputs.device)-0.5)*2
        augmented_input = inputs*augment_mask_input + augment_input*(1-augment_mask_input)
        augmented_label = labels*augment_mask_label + augment_label*(1-augment_mask_label)
        augmented_label = augmented_label.type_as(labels)
        return augmented_input,augmented_label

    def calc_grad(self,model:torch.nn.Module,inputs,labels):
        alpha = self.alpha
        stats = self.joint_cross(model,inputs,labels,alpha=alpha)
        return stats

class Indpt_Seg_Balanced(Optimizer):
    '''alpha,*args,lr=1,momentum=0.9,**kwargs
     Treats the network as an energy model.
     The validation set is included as prior
     The number of validation set is fixed.
     The upgrade of this optimizer is to optimize the size of validation.

     '''
    def __init__(self,alpha,beta,classnum,*args,lr=1,momentum=0.9,**kwargs):
        '''

        :param alpha:
        :param classnum:
        :param augment_rate: a float between 0:1
        :param args:
        :param lr:
        :param momentum:
        :param kwargs:
        '''
        # super(Joint_Likelihood_SGD,self).__init__(*args,**kwargs)
        super().__init__(*args,**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.alpha= alpha
        self.beta = beta
        self.classnum=classnum
        self.augment_rate= 0

    def hyper_normalize(self, model, inputs,logprob_prev, min, max, alpha=1):
        sample = torch.rand_like(inputs) * (max - min) + min
        output, logprob = prob_wrapper(model(sample))[0:2]
        free_energy_input = (logprob_prev*alpha).logsumexp(dim=list(range(logprob_prev.dim())))
        free_energy_sample = (output*alpha).logsumexp(dim=list(range(logprob_prev.dim())))
        free_energy = softmax(free_energy_input,free_energy_sample)/alpha
        return free_energy
    def scores(self,label_true,label_pred):
        label_true = label_true[0:,0:1,0:]
        label_pred = label_pred[0:,0:1,0:]
        intersect = (label_pred*label_true).sum()
        sum_measure = (label_true+label_pred).sum()
        dice = 2*intersect/sum_measure
        IOU = intersect/(sum_measure-intersect)
        return dice.detach(),IOU.detach()


    def joint_cross(self, model, inputs, labels, alpha=1,prior_min=-1,prior_max=1):

        output, logprob = prob_wrapper(model(inputs))[0:2] # quick fix if the model is not probabilistic
        free_energy = self.hyper_normalize(model, inputs, output, prior_min, prior_max, alpha=self.alpha)
        conditional_output = output.log_softmax(dim=1)
        label_prediction = labels == labels.max(dim=1,keepdim=True)[0]
        label_prediction = label_prediction/label_prediction.sum(dim=1,keepdim=True)
        label_prediction[0:, 0:1] = label_prediction[0:, 0:1] * (label_prediction[0:, 1:].sum() / label_prediction[0:, 0:1].sum())
        label_prediction = label_prediction/label_prediction.sum()
        prediction = output == output.max(dim=1, keepdim=True)[0]
        prediction = prediction / prediction.sum(dim=1,keepdim=True)
        # one_hot = self.label_to_onehot(output, labels)
        energy = output + label_prediction.log()
        energy = (energy).logsumexp(list(range(energy.dim())))
        # energy [energy != energy] = 0
        model_likelihood = energy - free_energy
        self.backward(model_likelihood)
        # Gather Stats
        dice,IOU = self.scores(labels,prediction)
        label_likelihood = labels * conditional_output
        label_likelihood[label_likelihood!= label_likelihood] = 0
        label_likelihood = label_likelihood.sum(dim=1).mean().cpu().item()
        energy = energy.cpu().item()
        free_energy = free_energy.cpu().item()
        model_likelihood = model_likelihood.cpu().item()
        acc = ((prediction*label_prediction).sum(dim=1)).sum().cpu().item()
        dice = dice.cpu().item()
        IOU = IOU.cpu().item()
        stats = dict(acc=acc,
                     energy=energy,
                     free_energy=free_energy,
                     dice= dice,
                     IOU = IOU,
                     model_likelihood=model_likelihood,
                     label_likelihood=label_likelihood)
        return stats

    def get_lr(self):
        return self.lr

    def augment_data(self,augment_rate,inputs,labels,num_class):
        augment_mask = torch.rand((inputs.shape[0],1),device=inputs.device)>augment_rate
        augment_mask = augment_mask.float()
        augment_mask_input = augment_mask.unsqueeze(2).unsqueeze(3)
        augment_mask_label = augment_mask.squeeze()
        augment_label = torch.randint(num_class,labels.size(),device=labels.device)
        augment_input = (torch.rand(inputs.shape,device=inputs.device)-0.5)*2
        augmented_input = inputs*augment_mask_input + augment_input*(1-augment_mask_input)
        augmented_label = labels*augment_mask_label + augment_label*(1-augment_mask_label)
        augmented_label = augmented_label.type_as(labels)
        return augmented_input,augmented_label

    def calc_grad(self,model:torch.nn.Module,inputs,labels):
        alpha = self.alpha
        stats = self.joint_cross(model,inputs,labels,alpha=alpha)
        return stats
