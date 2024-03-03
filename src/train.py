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
    train_dataset = Dataset_loader(u_train, y_train, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=False, drop_last=True)

    #Test
    t_test, u_test, y_test = import_data(args,False)
    u_test, y_test = Tensor(u_test).to(device), Tensor(y_test).to(device)
    test_dataset = Dataset_loader(u_test, y_test, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False, drop_last=True)

    #Initial data
    initial_state = T.zeros(2)
    initial_state[0] = Tensor(y_train[0])
    dt = t_train[22] - t_train[21]
    
    #################### RBF - Layer ###########################
    #Option 1 - https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/tree/master
    #Features
    layer_widths = [8,1]
    layer_centres = [1]
    basis_func = rbf.gaussian

    #rbf = rbf_network(layer_widths, layer_centres, basis_func)
    #rbf.to(device)

    #OPTION 2  https://github.com/rssalessio/PytorchRBFLayer/tree/main
    rbf = rbf_network_(args)
    rbf.to(device)

    ic(rbf.rbf.kernels_centers)
    ic(rbf.rbf.weights)
    ic(rbf.rbf.log_shapes)
    #################### Runge Kutta ###########################

    rk4 = RungeKuttaIntegratorCell(dt, rbf_layer=rbf)
    rk4.to(device)

    ###################### RNN - CELL #################
    model = RNN_Cell(rk4)
    model.to(device)

    model.load_state_dict(T.load(r'C:\Users\totti\OneDrive\Documents\Coding\rk_function\EMPS_RBF\Results_\Test_20\model.pth'))

    ######################## Training #############
    #Loss and Optimizer
    loss_func = T.nn.MSELoss()
    optimizer = T.optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
    #step_lr_scheduler = T.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=7,gamma=0.1)
    best_loss = 10

    #Training
    for epoch in range(args.N_EPOCHS):
        train_loss = 0.0
        for data in tqdm(train_dataloader):
            input_data, output_data = data
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            model.train() 
            outputs  = model.forward(input_data, initial_state)
            outputs = outputs[0,:]

            # calculate the loss
            output_data = output_data[:,0]
            loss_x1 = loss_func(outputs, output_data)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss = loss_x1 
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()
        for i in model.parameters():
            print(i)
        
        # update the lr scheduler
        #step_lr_scheduler.step()

        train_loss = train_loss/len(train_dataloader)

        print('Epoch: {} \tTraining Loss: {:.6f} \t'.format(epoch, train_loss))

    #Save model
    results_folder = make_files_location()
    results_file = results_folder + r'\\model.pth'
    T.save(model.state_dict(), results_file)
    result_loss = results_folder + r'\\loss.txt'
    F = open(result_loss,"w")
 
    # \n is placed to indicate EOL (End of Line)
    F.write(results_folder+" \n")
    F.write('Epoch: {} \tTraining Loss: {:.6f} \t'.format(epoch, train_loss))

    ######################## Testing #############	

    #prediction results after training
    initial_state_test = T.zeros(2)
    initial_state_test[0] = Tensor(y_test[0]).to(device)

    yPred = model(u_test, initial_state_test)
    ic(yPred.shape)
    yPred = yPred.detach().cpu().numpy()[0,:]
    y_test = y_test.detach().cpu().numpy()
    
    F.write('\n TEST \n')
    F.write('R2 score: {} \n MAE: {} \n MSE: {} \t'.format(r2_score(y_test, yPred), mean_absolute_error(y_test, yPred), mean_squared_error(y_test, yPred)))
    F.close()

    
    # plotting prediction results
    plt.plot(t_test, y_test, 'gray')
    plt.plot(t_test, yPred, 'b', label='after training')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid('on')
    plt.legend()
    plt.show()
