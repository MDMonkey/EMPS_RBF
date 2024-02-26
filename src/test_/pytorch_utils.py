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
    utils
    )
from model.rbf_layer import RBFLayer
from icecream import ic
import model.torch_rbf as rbf


#Training Loop

def training_loop(n_epochs, optimizer, model, loss_fn, train, label, initial_state):
    mae = nn.L1Loss()
    for epoch in range(1, n_epochs + 1):
        #Forward pass
        output_train = model(train, initial_state)
        loss_train = loss_fn(output_train[0,:], label)
        mae_train = mae(output_train[0,:], label)

        #Backward pass
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Training loss {loss_train.item():.4e}, mae {mae_train.item():.4e}")


#Models
class RNN_Cell(nn.Module):
    def __init__(self, cell, **kwargs):
        super(RNN_Cell, self).__init__()
        self.cell = cell

    def forward(self, inputs, initial_state):
        seq_sz = len(inputs)
        state = []
        state.append(initial_state)
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

#Forward
class RungeKuttaIntegratorCell__(nn.Module):
    def __init__(self, dt, **kwargs):
        super(RungeKuttaIntegratorCell__, self).__init__(**kwargs)
        #Parameters
        self.dt = Tensor(dt)
        self.A  = Tensor([0., 0.5, 0.5, 1.0]).float()
        self.B  = Tensor([[1/6, 2/6, 2/6, 1/6]]).float()
        
    def forward(self, rbf_layer, inputs, states):
        #C = stack((stack((self.c1+self.c2, -self.c2)), stack((-self.c2, self.c2+self.c3))))
        #States
        ydot   = states.view(2,-1) #[q,qdot]
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

        st_n = self.dt/6.0*(k1+2*k2+2*k3+k4)
        return st_n.view(2)
    
    def _fun(self, u, ydot, rbf_layer):
        ydot=ydot[1].cuda()
        #ic(ydot.device)
        #ic(u.device)
        return Tensor([[ydot], [(u - ydot*214.9261-19.3607*sign(ydot)-rbf_layer(ydot)+3.2902)/95.45]])

#SEM
class RungeKuttaIntegratorCell(nn.Module):
    def __init__(self, dt, **kwargs):
        super(RungeKuttaIntegratorCell, self).__init__(**kwargs)

        self.state_size    = 2
        self.A  = Tensor([0., 0.5, 0.5, 1.0])
        self.B  = Tensor([[1/6, 2/6, 2/6, 1/6]])
        self.dt = Tensor(dt)

        self.M = Tensor([95.45])
        self.offst = Tensor([3.2902])
        self.F_v = Tensor([214.9261])
        self.F_c = Tensor([19.3607])
        #Matrizes
        self.u_ = Tensor([[0],[1]])
        self.y_ = Tensor([[0,1],[0,0]])
        self.F_v_ = Tensor([[0,0],[0,self.F_v]])
        self.F_c_ = Tensor([[0,0],[0,self.F_c]])
        self.C_ = Tensor([[0],[1]])

        
    def forward(self, inputs, states):
        ydot   = states.cpu().view(2,-1) #[q,qdot]
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

        #ic(Tensor([[ydot[1]], [(u - ydot[1]*self.F_v - self.F_c*sign(ydot[1]) + self.offst)/self.M]]))
        #ic(((self.u_*u - linalg.matmul(self.F_v_,ydot) - linalg.matmul(self.F_c_,sign(ydot)) + self.C_*(self.offst))/self.M))

        return linalg.matmul(self.y_,ydot)+(self.u_*u-linalg.matmul(self.F_v_,ydot)-linalg.matmul(self.F_c_,sign(ydot))+self.C_*(self.offst))/self.M

    def _fun2(self, u, ydot):
        ic(ydot[1])
        ic((u-ydot[1]*self.F_v-self.F_c*sign(ydot[1])+self.offst)/self.M)
        ic((self.u_*u-linalg.matmul(self.F_v_,ydot)-linalg.matmul(self.F_c_,sign(ydot))+self.C_*(self.offst))/self.M)
        return Tensor([[ydot[1]], [(u - ydot[1]*self.F_v - self.F_c*sign(ydot[1]) + self.offst)/self.M]])
        
    def _fun2(self, u, ydot):

        return ((C*(u - self.offst)) + linalg.matmul(D,ydot) + linalg.matmul(re,self.F_c*sign(ydot)))/self.M[0]

        
    


class RungeKuttaIntegratorCell_grad(nn.Module):
    def __init__(self, dt,rbf_layer, **kwargs):
        super(RungeKuttaIntegratorCell_grad, self).__init__(**kwargs)

        self.state_size    = 2

        #self.C = 

        self.A  = Tensor([0., 0.5, 0.5, 1.0]).cuda()
        self.B  = Tensor([[1/6, 2/6, 2/6, 1/6]]).cuda()
        self.dt = Tensor(dt).cuda()
        self.rbf_layer = rbf_layer

        self.M = Tensor([1/95.45]).cuda()
        self.offst = Tensor([-3.2902]).cuda()
        self.F_v = Tensor([214.9261]).cuda()
        self.F_c = Tensor([19.3607]).cuda()
        #Matrizes
        self.u_ = Tensor([[0],[1]]).cuda()
        self.F_v_ = Tensor([[0,1],[0,self.F_v]]).cuda()
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
        return st_n.view(2)
    
    def _fun2(self, u, ydot):
        #ara = linalg.matmul(self.D,ydot)
        #pera = linalg.matmul(self.C,(u - self.rbf_layer(ydot[1])  - self.offst))
        re = Tensor([[0,0],[0,1]]).cuda()
        #regra = linalg.matmul(re,self.F_c*sign(ydot))
        #ic(regra)
        #ic(ara + pera + regra)
        #ic(linalg.matmul(self.M[0],ara + pera + regra))
        return self.M[0]*(linalg.matmul(self.C,(u - self.rbf_layer(ydot[1])  - self.offst)) + linalg.matmul(self.D,ydot) + linalg.matmul(re,self.F_c*sign(ydot)))

    def _fun(self, u, ydot):
        return (self.u_*u - linalg.matmul(self.F_v_,ydot) - linalg.matmul(self.F_f_,ydot) - self.C_*(self.offst+self.rbf_layer(ydot[1])))*self.M

    def _fun2(self, u, ydot):
        #ic(linalg.matmul(self.C,rbf_layer(ydot[1])))
        return Tensor([[ydot[1]], [(u - ydot[1]*self.F_v-self.F_f*sign(ydot[1])+self.offst)/self.M]]).cuda() - linalg.matmul(self.C,self.rbf_layer(ydot[1]))
        
    
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
        return len(self.input_fields)
    
    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]