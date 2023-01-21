import random
import numpy as np
import scipy.special
import networkx
import torch
AND_TYPE=0
OR_TYPE=1
device='cuda'

def categorical_sampler(logprob,dim):
    """
    Samples from the logprobability tensor, where dim represents the dimension that probability dist of the random varialbe is stored along.
    logprob is any arbitrary tensor size.
    returns
    :param logprob:
    :param dim:
    :return: onehot, logprob
    onehot is a tensor of the same size as logprob
    logprob is a tensor same size as logprob except the size of the dim dimension is 1.
    """
    logprob = logprob.log_softmax(dim=dim)
    prob = logprob.exp().detach()
    cumprob = prob.cumsum(dim=dim)
    randoms = torch.rand_like(cumprob)
    randoms = (randoms.transpose(0,dim)[0:1,0:]).transpose(0,dim)
    onehot = ((cumprob>=randoms) & ((cumprob-prob)<randoms)).float()
    logprob = (onehot*logprob).sum(dim=dim, keepdim=True)
    return onehot, logprob

def rand_bool():
    return random.random() > 0.5

def ind_to_onehot(ind,whichdim,totalinds=2):
    ind = ind.squeeze()
    ind = ind.unsqueeze(1)
    onehot = torch.arange(0,totalinds).reshape([1,totalinds])
    onehot = (onehot ==ind).float()

    shape = [-1, 2, 1] if whichdim == 1 else [-1, 1, 2]
    onehot = onehot.reshape(shape)
    return onehot

class Edge:

    def __init__(self,node1,node2,coef=0,type=OR_TYPE):
        self.node_list = [node1,node2]
        self.energy = torch.rand(1,2,2,device=device)*coef
        self.energy.requires_grad = True
        (self.energy.sum()*0).backward()
        node1.add_edge(self)
        node2.add_edge(self)
        self.type=type
        self.__color= (1,0,0)

    def node_energy_vector(self,node):
        """

        :param node:
        :return:
        an energy tensor of size num_samples,2
        """
        if node == self.node_list[0]:
            state = self.node_list[1].state  # n by 2 onehot vector
            state_onehot = state.unsqueeze(2) # n,2,1
            energy_vec = (state_onehot*self.energy).sum(dim=2).squeeze()
        else:
            state = self.node_list[0].state  # n by 1 vector
            state_onehot = state.unsqueeze(2).transpose(1,2) # n,1,2
            energy_vec = (state_onehot * self.energy).sum(dim=1).squeeze()

        return energy_vec # n by 2

    def step(self, lr):
        self.energy.data = self.energy.grad.data*lr + self.energy.data
        self.energy.grad.data = self.energy.grad.data * 0

    @property
    def color(self):
        return self.__color


class Node:

    def __init__(self,id,num_samples=10):
        self.id=id
        self.num_samples = num_samples
        self.state = categorical_sampler(torch.zeros(num_samples,2,device=device),dim=1)[0]
        self.edge_list = []

    def add_edge(self,e):
        self.edge_list.append(e)

    def connect(self,node):
        e = Edge(self,node,type=int(rand_bool()))
        return e

    def reset(self):
        self.edge_list=[]

    def randomize_state(self):
        self.state = int(random.random() > 0.5)

    def sample(self,alpha=2):
        energy = torch.zeros(1,2,device='cuda')
        for e in self.edge_list: #type: Edge
            if e.type==0:
                energy = energy+ e.node_energy_vector(self) # vector of size n*2
            else:
                energy = torch.logaddexp(energy,e.node_energy_vector(self))
        log_probs = (energy) - (alpha*energy).logsumexp(dim=1,keepdim=True)/alpha
        probs = energy.softmax(dim=1)
        self.state = int((probs[0: 0:1] < torch.rand_like(probs[0:,0:1])).item())
        log_probs = log_probs[self.state]
        return log_probs

    @property
    def color(self):
        if self.state is None:
            return (0,0,0)
        elif self.state ==0:
            return (1, 0, 0)
        elif self.state==1:
            return (0,0,1)
        else:
            raise Exception("Color for the state is not defined")

    def __str__(self):
        #return 'Node= {} (id:{} , num_states:{})'.format(self.state,self.id,self.numstates)
        state = self.state
        if state is None:
            state = 'NAN'
        else:
            state = str(state)
        return state

class Graph:
    def __init__(self,num_nodes,connectivity_rate=0.5):
        self.nodes = [Node(id=i) for i in range(num_nodes)]
        self.edges = self.randomize_connection(connectivity_rate)
        self.logprob= None
        self.__state = self.calc_state()
        self.iteration_index = 0
        self.networkx_graph = None
        self.networkx_layout = None

    def reset(self):
        for node in self.nodes:
            node.reset()

    def randomize_connection(self, connectivity_rate):
        edges= []
        self.reset()
        for i,node1 in enumerate(self.nodes):
            remain_nodes = self.nodes[i+1:]
            for node2 in remain_nodes:
                if random.random()>connectivity_rate:
                    edges.append(node1.connect(node2))
        return edges

    def sample(self,alpha=2,iters=4):
        # self.randomize_state()
        logprob_total = torch.zeros(1)
        for k in range(iters):
            i= self.iteration_index
            node = self.nodes[i]
            logprob_temp = node.sample(alpha=alpha)
            self.__state[node.id]= node.state
            logprob_total = logprob_total + logprob_temp
            self.iteration_index= (self.iteration_index+1)%len(self.nodes)
        self.logprob = logprob_total
        return logprob_total

    def calc_state(self):

        return torch.from_numpy(np.array([node.state[0,1].cpu().item() for node in self.nodes]))

    @property
    def state(self):
        return self.__state

    def randomize_state(self):
        for n in self.nodes:
            n.randomize_state()

    def step(self,lr):
        for e in self.edges:
            e.step(lr)

    def to_network_x(self):
        graph = networkx.Graph()
        for n in self.nodes:
            graph.add_node(n, label=n.state)
        for e in self.edges:  # type:Edge
            graph.add_edge(e.node_list[0], e.node_list[1], color=e.color)
        return graph

    def draw(self,ax, redraw=False, layout=None):

        if redraw or self.networkx_graph is None:
            self.networkx_graph = self.to_network_x()
            self.networkx_layout = networkx.spring_layout(self.networkx_graph)

        self.networkx_graph.remove_edges_from(self.networkx_graph.edges)
        for e in self.edges: #type:Edge
            self.networkx_graph.add_edge(e.node_list[0], e.node_list[1], color=e.color)
        ax.clear()
        colors = [self.networkx_graph[u][v]['color'] for u, v in self.networkx_graph.edges]
        cvalmap = [node.color for node in self.networkx_graph.nodes]
        networkx.draw(self.networkx_graph,
                      self.networkx_layout,
                      ax,
                      # cmap=plt.get_cmap('viridis'),
                      with_labels=True,
                      node_color=cvalmap,
                      edge_color=colors,
                      font_color='white',
                      node_vmin=0,
                      node_vmax=1
                      )
        plt.draw()

def bits_to_num(bitlist:torch.Tensor):
    coeffs = 2.0**(-torch.arange(2,bitlist.numel()+2))
    num = 0.5+ ((2*bitlist.float()-1) * coeffs).sum()
    return num

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

if __name__ == '__main__':
    def cost(x):
        return -((x-2.3)**2)
    g = Graph(80,0.7)
    print("num edges", len(g.edges))
    from matplotlib import pyplot as plt
    line = np.arange(-10,10,0.01)
    cost_plot = cost(line)
    yline = line*0
    ys = []
    xs = []
    mcost = -torch.inf
    fig, axes = plt.subplots(3,1)
    # plt.show(block=False)
    import time
    while True:
        # ys = []
        # xs = []
        t1 = time.time()
        for i in range(1):
            logprob = g.sample(alpha=2,iters=100)
            state = g.state
            number = 2*bits_to_num(state[-20:])-1
            number = number*10#scipy.special.erfinv(number)*2
            c = cost(number)
            if c > mcost:
                mcost = c
            # c= c-mcost
            print('cost',c, 'mcost', mcost)
            (np.exp(c)*logprob).backward()
            xs.append(number.item())
            ys.append(logprob.item())
            # g.draw(axes[2])
        g.step(0.1)

        #plot
        if len(xs)>=1000:
            st = len(xs)-1000
            xs = xs[st:]
        # xs = xs[-1:]
        # plt.gcf().clear()
        axes[0].clear()
        axes[1].clear()
        axes[2].clear()
        axes[0].plot( np.arange(-len(xs),0),xs)
        xs_np = np.array(xs)
        ys_np = torch.from_numpy(np.array(ys)).softmax(dim=0).numpy()
        args = xs_np.argsort()
        xs_np = xs_np[args]
        ys_np = ys_np[args]

        axes[1].hist(xs,bins=100,density=True)
        axes[1].plot(line, 100*np.exp(cost_plot)/np.sum(np.exp(cost_plot)))
        axes[2].plot(xs_np,ys_np)
        # axes[2].plot(xs, ys)
        # axes[2].plot(line,yline)
        plt.show(block=False)
        plt.pause(0.001)
        print(number,logprob, time.time()-t1)