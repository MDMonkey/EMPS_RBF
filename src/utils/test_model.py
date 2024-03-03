import sys
sys.path.append(r"C:\Users\totti\OneDrive\Documents\Coding\rk_function\EMPS_RBF\src")
from utils.read_utils import *
from utils.config_ import *
from model.utils_pytorch import *

import torch as T
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


if __name__ == "__main__":

    # Config of the model
    args = load_args()

    #Seed
    T.manual_seed(32)
    np.random.seed(32)

    #Device
    device = T.device('cpu')
    print(device)


    #Data
    #Train
    t_train, u_train, y_train = import_data(args,True)
    u_train, y_train= Tensor(u_train).to(device), Tensor(y_train).to(device)

    #Test
    t_test, u_test, y_test = import_data(args,False)
    u_test, y_test = Tensor(u_test).to(device), Tensor(y_test).to(device)

    #Initial 
    initial_state_test = T.zeros(2)
    initial_state_test[0] = Tensor(y_test[0]).to(device)
    y_test = y_test[1:]
    u_test = u_test[1:]

    initial_state = T.zeros(2)
    initial_state[0] = Tensor(y_train[0]).to(device)
    initial_state = initial_state.to(device)
    dt = t_train[22] - t_train[21]

    y_train = y_train[1:]
    u_train = u_train[1:]

    #OPTION 2  https://github.com/rssalessio/PytorchRBFLayer/tree/main
    rbf = rbf_network_(args)
    rbf.to(device)

    #################### Runge Kutta ###########################

    if args.CUDA:
        rk4 = RungeKuttaIntegratorCell_cuda(dt, rbf_layer=rbf)
    else:
       rk4 = RungeKuttaIntegratorCell_cpu(dt, rbf_layer=rbf)
    rk4.to(device)

    ###################### RNN - CELL #################
    model = RNN_Cell(rk4)
    model.to(device)

    model.load_state_dict(T.load(r"C:\Users\totti\OneDrive\Documents\Coding\rk_function\EMPS_RBF\other\EMPS_RBF-modified_rbf\Results_\Test_55\best_model.pth"))

    output_data = y_train[:,0]

    ######################## Testing #############	

    #prediction results after training
    
    yPred = model(u_test, initial_state_test)
    ic(yPred.shape)
    yPred = yPred.detach().cpu().numpy()[0,:]
    y_test = y_test.detach().cpu().numpy()

    print('\n TEST \n')
    print('R2 score: {} \n MAE: {} \n MSE: {} \t'.format(r2_score(y_test, yPred), mean_absolute_error(y_test, yPred), mean_squared_error(y_test, yPred)))


    # plotting prediction results
    plt.plot(t_test[1:], y_test, 'gray')
    plt.plot(t_test[1:], yPred, 'b', label='after training')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid('on')
    plt.legend()
    plt.show()

    

