import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
class Distribution:
    def sample(self,num_samples,alpha):

        return torch.zeros(1),torch.zeros(1)

class Normal(Distribution):
    """
    Normal distribution parameterized by mean and log of precision

    g(x) = e^(-p*x^2)
    """
    def __init__(self, mean:torch.Tensor,precision, dim=None):
        if dim:
            mean = torch.zeros(1,dim)
            precision = torch.zeros(1,dim)
        assert mean.numel() == precision.numel(), "size mismatch"
        assert mean.ndim==2, "the param vector needs to be a tensor with dimension of 2"
        assert mean.shape[0] == 1, "the param vector, dim=0, needs to be of size 1"
        self.__mean = mean
        self.__mean.requires_grad = True
        self.__precision = precision
        self.__precision.requires_grad = True
        self.__param_list = [self.__mean, self.__precision]
        self.dim = mean.shape[1]

    @property
    def mean(self):
        return self.__mean

    @property
    def precision(self):
        p = self.__precision.exp()
        return p

    def sample(self,num_samples,alpha):
        """

        :param num_samples:
        :param alpha:
        :return:
        """
        rands = torch.randn(num_samples,self.dim)/self.precision
        samples = (self.mean + rands).detach()
        energy = (rands**2)*self.precision
        energy = energy.sum(dim=1,keepdim=True)
        return samples, energy

    def step(self,lr, momentum=0):
        for param in self.__param_list:
            param.data = param.data + lr* param.grad.data
            param.grad.data = param.grad.data * momentum

    def __str__(self):
        mean = self.mean.squeeze().numpy()
        prec = self.precision.squeeze().numpy()
        var = (1/prec).numpy()
        return "Mean:{}\n Precision:{}\n Var:{}\n".format(str(mean),str(prec),str(var))

    def draw


def calc_grad(dist:Distribution, objective_function, alpha=1.0,beta=1,num_samples=1):
    """

    :param dist: the sampling distribution
    :param objective_function: The loss function
    :param alpha: regularization parameter, the more the more regularized
    :param beta: coefficient multiplied by the objective function. setting this param to negative numbers
    makes the optimization minimization
    :return:
    """
    samples, energies = dist.sample(num_samples,alpha=1)
    objective_energies = beta * objective_function(samples)
    regularized_objective= objective_energies.softmax(dim=0)*energies - (energies*(alpha-1)).softmax(dim=0).detach()*energies
    regularized_objective.mean().backward()



if __name__ == '__main__':
    def cost(x):
        return -(x-10)**2


