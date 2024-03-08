from torch.nn.parameter import Parameter
from torch import (
    linalg,
    nn,
    Tensor,
    stack,
    cat,
    transpose, 
    optim,
    zeros,
    diag,
    sign,
    norm,
    utils,
    no_grad
    )
from model.rbf_layer import RBFLayer
from icecream import ic
import model.torch_rbf as rbf
import numpy as np


#Training Loop


#Models
class RNN_Cell(nn.Module):
    def __init__(self, cell, **kwargs):
        super(RNN_Cell, self).__init__()
        self.cell = cell

    def forward(self, inputs, initial_state):
        seq_sz = len(inputs)
        state = []
        state.append(initial_state.cpu())
        for t in range(1, seq_sz): 
            input = inputs[t-1]
            state_t = self.cell.forward(input, state[t-1])
            state.append(state[t-1]+state_t)

        return stack((state),dim=1)
    
#INIT
class RungeKuttaIntegratorCell_(nn.Module):
    def __init__(self, dt, rbf_layer, **kwargs):
        super(RungeKuttaIntegratorCell_, self).__init__(**kwargs)
        self.state_size = 2
        self.A  = Tensor([0., 0.5, 0.5, 1.0])
        self.B  = Tensor([[1/6, 2/6, 2/6, 1/6]])
        self.dt = Tensor(dt)
        self.rbf_layer = rbf_layer


    def forward(self, inputs, states):

        rbf_layer = self.rbf_layer
        ic(states.requires_grad)
        ydot   = states.view(2,-1) #[q,qdot]
        ic(ydot)
        yddoti = self._fun(inputs, ydot, rbf_layer)
        #k1
        ydoti  = ydot + self.A[0] * yddoti * self.dt
        k1     = self._fun(inputs, ydoti, rbf_layer)
        #k2
        ydot2 = ydot + self.A[1] * yddoti * self.dt
        k2     = self._fun(inputs, ydot2, rbf_layer)
        #k3
        ydot3 = ydot + self.A[2] * yddoti * self.dt
        k3     = self._fun(inputs, ydot3, rbf_layer)
        #k4
        ydot4 = ydot + self.A[3] * yddoti * self.dt
        k4    = self._fun(inputs, ydot4, rbf_layer)
        ic(k4)

        st_n = self.dt/6.0*(k1+2*k2+2*k3+k4)
        return st_n.view(2)
    
    def _fun(self, u, ydot, rbf_layer):
        ala = rbf_layer(ydot[1])
        ydot = ydot[1].cuda()
        return Tensor([[ydot], [(u - ydot*214.9261-19.3607*sign(ydot)-rbf_layer(ydot)+3.2902)/95.45]])


class RungeKuttaIntegratorCell_cpu(nn.Module):
    def __init__(self, dt,rbf_layer, **kwargs):
        super(RungeKuttaIntegratorCell_cpu, self).__init__(**kwargs)

        self.state_size    = 2
        self.A  = Tensor([0., 0.5, 0.5, 1.0]).cpu()
        self.B  = Tensor([[1/6, 2/6, 2/6, 1/6]]).cpu()
        self.dt = Tensor(dt).cpu()
        self.rbf_layer = rbf_layer

        self.M = Tensor([1/95.4520]).cpu()
        self.offst = Tensor([-3.2902]).cpu()
        self.F_v = Tensor([214.9261]).cpu()
        self.F_c = Tensor([19.3607]).cpu()
        #Matrizes
        self.u_ = Tensor([[0],[1]]).cpu()
        self.y_ = Tensor([[0,1],[0,0]]).cpu()
        self.F_v_ = Tensor([[0,0],[0,self.F_v]]).cpu()
        self.F_f_ = Tensor([[0,0],[0,self.F_c]]).cpu()
        self.C_ = Tensor([[0],[1]]).cpu()
        
    def forward(self, inputs, states):
        ydot   = states.view(2,-1).cpu() #[q,qdot]
        yddoti = self._fun(inputs, ydot)
        #k1
        ydoti  = ydot + self.A[0] * yddoti * self.dt
        k1     = self._fun(inputs, ydoti)
        #k2
        ydot2 = ydot + self.A[1] * yddoti * self.dt
        k2     = self._fun(inputs, ydot2)
        #k3
        ydot3 = ydot + self.A[2] * yddoti * self.dt
        k3     = self._fun(inputs, ydot3)
        #k4
        ydot4 = ydot + self.A[3] * yddoti * self.dt
        k4    = self._fun(inputs, ydot4)

        st_n = self.dt/6.0*(k1+2*k2+2*k3+k4)
        return st_n.view(2)
    
    def _fun(self, u, ydot):
        return linalg.matmul(self.y_,ydot)+(self.u_*u - linalg.matmul(self.F_v_,ydot) - linalg.matmul(self.F_f_,ydot) - self.C_*(self.offst+self.rbf_layer(ydot[1])))*self.M
    

class RungeKuttaIntegratorCell_cuda(nn.Module):
    def __init__(self, dt,rbf_layer, **kwargs):
        super(RungeKuttaIntegratorCell_cuda, self).__init__(**kwargs)

        self.state_size    = 2
        self.A  = Tensor([0., 0.5, 0.5, 1.0]).cuda()
        self.B  = Tensor([[1/6, 2/6, 2/6, 1/6]]).cuda()
        self.dt = Tensor(dt).cuda()
        self.rbf_layer = rbf_layer

        self.M = Tensor([1/95.4520]).cuda()
        self.offst = Tensor([-3.2902]).cuda()
        self.F_v = Tensor([214.9261]).cuda()
        self.F_c = Tensor([19.3607]).cuda()
        #Matrizes
        self.u_ = Tensor([[0],[1]]).cuda()
        self.y_ = Tensor([[0,1],[0,0]]).cuda()
        self.F_v_ = Tensor([[0,0],[0,self.F_v]]).cuda()
        self.F_f_ = Tensor([[0,0],[0,self.F_c]]).cuda()
        self.C_ = Tensor([[0],[1]]).cuda()
        
    def forward(self, inputs, states):
        ydot   = states.view(2,-1).cuda() #[q,qdot]
        yddoti = self._fun(inputs, ydot)
        #k1
        ydoti  = ydot + self.A[0] * yddoti * self.dt
        k1     = self._fun(inputs, ydoti)
        #k2
        ydot2 = ydot + self.A[1] * yddoti * self.dt
        k2     = self._fun(inputs, ydot2)
        #k3
        ydot3 = ydot + self.A[2] * yddoti * self.dt
        k3     = self._fun(inputs, ydot3)
        #k4
        ydot4 = ydot + self.A[3] * yddoti * self.dt
        k4    = self._fun(inputs, ydot4)

        st_n = self.dt/6.0*(k1+2*k2+2*k3+k4)
        st_n = st_n.cuda()
        return st_n.view(2)
    
    def _fun(self, u, ydot):
        res = linalg.matmul(self.y_,ydot)+(self.u_*u - linalg.matmul(self.F_v_,ydot) - linalg.matmul(self.F_f_,ydot) - self.C_*(self.offst+self.rbf_layer(ydot[1])))*self.M
        return res.cuda()
    

class rbf_network_(nn.Module):
    def __init__(self, args):
        super(rbf_network_, self).__init__()
        self.rbf = RBFLayer(in_features_dim=args.IN_FEATURES,
                       num_kernels=args.NUM_KERNELS,
                       out_features_dim=args.OUT_FEATURES,
                       radial_function=self.rbf_gaussian,
                       norm_function=self.euclidean_norm,
                       normalization=False)
        
    def euclidean_norm(self,x):
        return norm(x, p=2, dim=-1)

    # Gaussian RBF
    def rbf_gaussian(self,x):
        return (-0.5*x.pow(2)).exp()
    
    def forward(self,x):
        return self.rbf(x)


class rbf_network(nn.Module):
    def __init__(self, layer_widths, layer_centres, basis_func):
        super(rbf_network, self).__init__()
        self.rbf_layers = nn.ModuleList()
        for i in range(len(layer_widths) - 1):
            self.rbf_layers.append(rbf.RBF(layer_widths[i], layer_centres[i], basis_func))
            #self.linear_layers.append(nn.Linear(layer_centres[i], layer_widths[i+1]))

    def forward(self, x):
        out = x
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](out)
        return out
    

class Dataset_loader(utils.data.Dataset):
    def __init__(self, input, output, args):
        self.input = input
        self.output = output
        self.args = args
    
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]
    

def validate(network, u_valid, y_valid, initial_state, loss_function):
  """
  This function validates convnet parameter optimizations
  """
  #  creating a list to hold loss per batch
  loss_per_batch = 0.0

  #  defining model state
  network.eval()

  #  defining dataloader

  print('validating...')
  #  preventing gradient calculations since we will not be optimizing
  with no_grad():
    #  iterating through batches
    
    #--------------------------------------
    #  sending images and labels to device
    #--------------------------------------
    outputs = network.forward(u_valid, initial_state)
    outputs = outputs[0,:]

    # calculate the loss
    output_data = y_valid[:,0]
    # calculate the loss
    loss_x1 = loss_function(outputs, output_data.float())

    #-----------------
    #  computing loss
    #-----------------
    loss = loss_x1
    loss_per_batch+= loss.item()

  return loss_per_batch 

class ValidationLossEarlyStopping:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience  # number of times to allow for no improvement before stopping the execution
        self.min_delta = min_delta  # the minimum change to be counted as improvement
        self.counter = 0  # count the number of times the validation accuracy not improving
        self.min_validation_loss = np.inf

    # return True when validation loss is not decreased by the `min_delta` for `patience` times 
    def early_stop_check(self, validation_loss):
        if ((validation_loss+self.min_delta) < self.min_validation_loss):
            self.min_validation_loss = validation_loss
            self.counter = 0  # reset the counter if validation loss decreased at least by min_delta
        elif ((validation_loss+self.min_delta) > self.min_validation_loss):
            self.counter += 1 # increase the counter if validation loss is not decreased by the min_delta
            if self.counter >= self.patience:
                return True
        return False