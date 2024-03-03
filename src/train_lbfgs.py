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
    if args.CUDA:
       device = T.device('cuda' if T.cuda.is_available() else 'cpu')
       T.cuda.empty_cache()
    else:
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

    if args.CUDA:
        rk4 = RungeKuttaIntegratorCell_cuda(dt, rbf_layer=rbf)
    else:
       rk4 = RungeKuttaIntegratorCell_cpu(dt, rbf_layer=rbf)
    rk4.to(device)

    ###################### RNN - CELL #################
    model = RNN_Cell(rk4)
    model.to(device)

    #model.load_state_dict(T.load(r"C:\Users\totti\OneDrive\Documents\Coding\rk_function\EMPS_RBF\other\EMPS_RBF-modified_rbf\Results_\Test_55\best_model.pth"))

    ######################## Training #############
    #Loss and Optimizer
    loss_func = T.nn.MSELoss()
    #optimizer = T.optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
    optimizer = T.optim.LBFGS(model.parameters(), history_size=20, max_iter=10, lr=args.LEARNING_RATE)

    #step_lr_scheduler = T.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=7,gamma=0.1)
    best_loss = 10
    train_losses = []
    valid_losses = []
    #Make folderr
    results_folder = make_files_location()

    output_data = y_train[:,0]
    
    #Training
    for epoch in range(args.N_EPOCHS):
        train_loss = 0.0
        initial_state = initial_state.to(device)


        model.train() 
        def closure():
            if T.is_grad_enabled():
                optimizer.zero_grad()
            output = model(u_train, initial_state)
            y_pred = output[0,:]
            loss = loss_func(output_data, y_pred)
            if loss.requires_grad:
                loss.backward()
            return loss
        
        # perform a single optimization step (parameter update)
        optimizer.step(closure)
        loss = closure()

        # update running training loss
        train_loss += loss.item()
        for i in model.parameters():
            print(i)
        
        # update the lr scheduler
        #step_lr_scheduler.step()

        train_loss = train_loss/len(u_train)
        train_losses.append(train_loss)
        valid_loss = validate(model, u_test, y_test, initial_state_test, loss_func)

        valid_loss = valid_loss/len(u_test)
        if valid_loss < best_loss:
            best_loss = valid_loss
            T.save(model.state_dict(), results_folder+r'\\best_model.pth')
        valid_losses.append(valid_loss)
        print('Epoch: {} \tTraining Loss: {:.8f} \tValid Loss: {:.8f} '.format(epoch, train_loss, valid_loss))


    #Save model
    results_file = results_folder + r'\\model.pth'
    T.save(model.state_dict(), results_file)
    result_loss = results_folder + r'\\loss.txt'
    F = open(result_loss,"w")
 
    # \n is placed to indicate EOL (End of Line)
    F.write(results_folder+" \n")
    F.write('Epoch: {} \tTraining Loss: {:.6f} \t'.format(epoch, train_loss))
    F.write('Validation Loss: {} \t'.format(valid_loss))
    F.write('Best Loss: {} \t'.format(best_loss))
    F.write('Learning Rate: {} \n'.format(args.LEARNING_RATE))
    F.write('Batch Size: {} \t'.format(args.BATCH_SIZE))
    F.write('Number of Epochs: {} \t'.format(args.N_EPOCHS))
    F.write('Number of kernels: {} \t'.format(args.NUM_KERNELS))
    F.write('Optimizer: {} \t'.format(str(optim)))


    ######################## Testing #############	

    #prediction results after training
    

    yPred = model(u_test, initial_state_test)
    ic(yPred.shape)
    yPred = yPred.detach().cpu().numpy()[0,:]
    y_test = y_test.detach().cpu().numpy()
    
    F.write('\n TEST \n')
    F.write('R2 score: {} \n MAE: {} \n MSE: {} \t'.format(r2_score(y_test, yPred), mean_absolute_error(y_test, yPred), mean_squared_error(y_test, yPred)))
    F.close()

    
    # plotting prediction results
    plt.plot(t_test[1:], y_test, 'gray')
    plt.plot(t_test[1:], yPred, 'b', label='after training')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid('on')
    plt.legend()
    plt.savefig(results_folder + r'\\plot_test.png', bbox_inches='tight')
    plt.show()

    #Plot the loss
    epochs = (1,len(train_losses)-1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.show()
    plt.savefig(results_folder + r'\\loss.png', bbox_inches='tight')
