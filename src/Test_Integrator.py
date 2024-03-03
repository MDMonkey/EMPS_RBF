import numpy as np 
import matplotlib.pyplot as plt
import os
from icecream import ic
import torch as T
import matplotlib.pyplot as plt


from test_.pytorch_utils import *
from utils.read_utils import *
from utils.config_ import *



args = load_args()

args.PATH_DATA = r"C:\Users\totti\OneDrive\Documents\Coding\rk_function\EMPS_\Data\Organized\DATA_EMPS_TRAIN.mat"
""" 
print('eloo')
t, x_train, x_val, x_test, y_train, y_val, y_test  = import_data_(args)


def rk4singlestep(fun, dt,y0, input):
    f1 = fun(y0, input)
    f2 = fun(y0 + dt/2*f1, input)
    f3 = fun(y0 + dt/2*f2, input)
    f4 = fun(y0 + dt*f3, input)
    y1 = y0 + dt/6*(f1 + 2*f2 + 2*f3 + f4)
    return y1

def emps_f(ydot, input):
    return (input - ydot*214.9261 - 19.36*np.sign(ydot)+3.2902)/95.45

ini = 0.0825

dt =  0.004033884630899542

num_time_pts = len(x_test)
ic(num_time_pts)
t = np.linspace(0, 10, num_time_pts)

y = np.zeros(num_time_pts)
y[0] = ini

for i in range(num_time_pts-1):
    yout = rk4singlestep(emps_f, dt, y[i],x_test[i])
    y[i+1] = yout

ic(y_test.shape)




dt = t[32]-t[31]
ic(dt)

 """
t, u_test, y_test = import_data(args,True)
dt = t[32]-t[31]

rkCell = RungeKuttaIntegratorCell(dt)


initial_state = T.zeros(2)
initial_state[0] = Tensor(y_test[0])

x_test = Tensor(u_test)
model = RNN_Cell(cell=rkCell)

yPred = model(x_test, initial_state)
yPred = yPred.detach().numpy()

""" ic(yPred.shape)
ic(u_test)
ic(x_test) """

#plt.plot(y[2000:], label='rk4_numpy')
plt.plot(y_test,'r--', label = 'real')
plt.plot(yPred[0,:],'b--', label='pytorch')
plt.legend()
plt.show()

