from utils.read_utils import *
from utils.config_ import *
from icecream import ic

import torch as T
import matplotlib.pyplot as plt
#from model.pytorch_utils import *



#Save model
results_file = make_files_location()
ic(results_file)
exit()
exit()
#Load the arguments
args = load_args()

#args.PATH_DATA = r"C:\Users\totti\OneDrive\Documents\Coding\rk_function\EMPS_\Data\Organized\DATA_EMPS_TEST.mat"
t, x, y = import_data(args,True)

#Define the parameters
dt = t[32]-t[31]

#rkCell = RungeKuttaIntegratorCell3(dt)
#model = RNN_Cell(cell=rkCell)

#Initial States
initial_state = T.zeros(2)
initial_state[0] = Tensor(y[0])
initial_state[1] = Tensor([21.0])
initial_state = initial_state.view(2,-1)
#Values input and output
x_test = Tensor(x)
#y_test = y_test.reshape(2,-1)
M = Tensor([95.45])
F_f = Tensor([214.9261])
F_s = Tensor([19.3607])
offst = Tensor([3.29])


D = T.ones(2,1)
ic(D)
C = zeros(2,2)
C = Tensor([[0,1],[0,-F_f]])
ic(initial_state)
ic(C)
ic(transpose(C, 0, 1))
arpa = linalg.matmul(C,initial_state)
ic(arpa*2)
ic(sign(D))
ui = D*x_test[0]+linalg.matmul(C,initial_state)- D*F_s*sign(initial_state) + Tensor([[0],[offst]])
ic(ui)
exit()


#Predictions
yPred = model(x_test, initial_state)
yPred = yPred.detach().numpy()

# plotting prediction results
plt.plot(y, 'r', label='before training') 
plt.plot(yPred[0,:], 'b', label='after training')
plt.xlabel('t')
plt.ylabel('y')
plt.grid('on')
plt.legend()
plt.show()
