% based on sysid.m file from casadi example pack
% updated by Lucas Souza and Helon Ayala 06-2021
% Description:
% in this file we proceed the grey-box identification of state-space
% nonlinear models.
% it is possible to adapt it to different cases, by loading different data
% and changing the state equations as below

clc
clear all
close all

addpath(genpath('D:/OneDrive/PG/Doutorado/Controle MPC/CasADI'));
% addpath(genpath('C:/Users/coord/OneDrive/PG/Doutorado/Controle MPC/CasADI'));
import casadi.*


%%%%%%%%%%%% load data %%%%%%%%%%%%%%%%%%%%%
% File to load
load('DATA_EMPS.mat')
% Variables are:
% qm = motor position (measured through the motor encoder)
% qg = the reference position
% t = time
% vir = motor voltage (output of the controller)

% Construction of the vector measurements
Force1 = gtau*vir; % Motor force

decimate = 1; % see Forgione, Piga, EJC 59 (2021) 69-81
u_data = Force1(1:decimate:end);
y_data = qm(1:decimate:end);
t      = t(1:decimate:end); 

% plot(t,y_data)
%%
N  = length(t);  % Number of samples
Ts = round(t(2)-t(1),3);  % sampling time (seconds)
fs = 1/Ts;       % Sampling frequency [hz]

x0 = DM([qm(1),0]); % initial condition for simulation

% M = 95.1089;
% Fv = 203.5034;
% Fc = 20.3935;
% ofst = -3.1648;

% Complete data
M = 95.4520;
Fv = 214.9261;
Fc = 19.3607;
ofst = -3.2902;

% % Complete data
% M = 115.3942;
% Fv = 421.1115;
% ofst = -3.1592;

% M =  93.3523;
% Fv = 205.3014;
% Fc = 20.3510;
% ofst = -3.2205;

%%%%%%%%%%%% MODELING %%%%%%%%%%%%%%%%%%%%%
q  = MX.sym('q');
dq = MX.sym('dq');
u  = MX.sym('u');

states = [q;dq];
controls = u;

Nneu = 4; 

c =     MX.sym('c',Nneu,1);
d =     MX.sym('d',Nneu,1);
W =     MX.sym('W',Nneu,1);


c0 = -1 + 2*rand(Nneu,1);
d0 = rand(Nneu,1);
W0 = -100 + 200*rand(Nneu,1);

lbc = [ -1 + c0*0]; 
ubc = [ 1 + c0*0];
lbd = [0.01 + d0*0]; 
ubd = [  1 + d0*0];
lbW = [-100 + W0*0]; 
ubW = [ 100 + W0*0];

lbx = [lbc;lbd;lbW];
ubx = [ubc;ubd;ubW];

params = [c;d;W];
nparam = length(params);
param_guess = [c0;d0;W0];

Frbf = RBF_gauss(c,dq,d,W);

rhs = [dq; (u-Fv*dq-Fc*sign(dq)-Frbf-ofst)/M];
% rhs = [dq; (u-Fv*dq-Frbf-ofst)/M];
% rhs = [dq; (u-Frbf-ofst)/M];

% Form an ode function
ode = Function('ode',{states,controls,params},{rhs});

%%%%%%%%%%%% Creating a simulator %%%%%%%%%%
N_steps_per_sample = 40;
dt = 1/fs/N_steps_per_sample;

% Build an integrator for this system: Runge Kutta 4 integrator
k1 = ode(states,controls,params);
k2 = ode(states+dt/2.0*k1,controls,params);
k3 = ode(states+dt/2.0*k2,controls,params);
k4 = ode(states+dt*k3,controls,params);

states_final = states+dt/6.0*(k1+2*k2+2*k3+k4);

% Create a function that simulates one step propagation in a sample
one_step = Function('one_step',{states, controls, params},{states_final});

X = states;
for i=1:N_steps_per_sample
    X = one_step(X, controls, params);
end
%
% % Create a function that simulates all step propagation on a sample
one_sample = Function('one_sample',{states, controls, params}, {X});
%
% speedup trick: expand into scalar operations
one_sample = one_sample.expand();

%%%%%%%%%%%% Simulating the system %%%%%%%%%%

all_samples = one_sample.mapaccum('all_samples', N);

%%%%%%%%%%%% Identifying the simulated system %%%%%%%%%%
opts = struct;
% opts.ipopt.max_iter = 15;
% opts.ipopt.print_level = 3;%0,3
% opts.print_time = 1;
opts.ipopt.acceptable_tol = 1e-4;
opts.ipopt.acceptable_obj_change_tol = 1e-4;

%%%%%%%%%%%% multiple shooting strategy %%%%%%%%%%
% % All states become decision variables
X = MX.sym('X', 2, N);

res = one_sample.map(N, 'thread', 16);
Xn = res(X, u_data', repmat(params,1,N));

gaps = Xn(:,1:end-1)-X(:,2:end);

e = y_data-Xn(1,:)';

V = veccat(params, X);

J = 1/N*dot(e,e);

nlp = struct('x',V, 'f',J,'g',vec(gaps));

% Multipleshooting allows for careful initialization
yd = diff(y_data)*fs;
X_guess = [ y_data  [yd;yd(end)]]';

param_guess = [param_guess(:);X_guess(:)];

solver = nlpsol('solver','ipopt', nlp, opts);

% sol = solver('x0',param_guess,'lbg',0,'ubg',0);
sol = solver('x0',param_guess,'lbg',0,'ubg',0,'lbx',[lbx;-ones(length(V)-nparam,1)],'ubx',[ubx;ones(length(V)-nparam,1)]);
solx = sol.x.full;
paramhat = solx(1:nparam);


%% analisa resultado

% Mhat    = denorm(paramhat(1),parammax(1),parammin(1));
% ofsthat = denorm(paramhat(2),parammax(2),parammin(2));
% 
% disp('Parametros identificados:')
% [Mhat, Fvhat, Fchat, ofsthat]
% 
% disp('Parametros IDIM:')
% paramhatIDIM = [95.1089, 203.5034, 20.3935, -3.1648]';
% paramhatIDIM' 
% paramhatIDIM = normalize(paramhatIDIM,parammax,parammin); % simulation


%% compare both solutions train

Xhat = all_samples(x0, u_data, repmat(paramhat,1,N));
Xhat = Xhat.full;
yhatRBF = Xhat(1,:)';
ydothatRBF = Xhat(2,:)';

chat = paramhat(1:Nneu);
dhat = paramhat(Nneu+1:2*Nneu);
What = paramhat(2*Nneu+1:3*Nneu);

Frbfhat = RBF_gauss(chat,ydothatRBF,dhat,What);
Ffhat = Fv*ydothatRBF-Fc*sign(ydothatRBF);

figure
hold on
plot(t,y_data,'k-','linewidth',1.5)
plot(t,yhatRBF,'r--','linewidth',1.5)
% plot(t,yhatIDIM,'g--','linewidth',1.5)
grid on
xlabel('time')
% legend({'real','casadi','IDIM'},'location','best')

figure
hold on
plot(t,y_data-yhatRBF,'r-','linewidth',1.5)
% plot(t,y_data-yhatIDIM,'g--','linewidth',1.5)
grid on
xlabel('time')
ylabel('error')
% legend({'casadi','IDIM'},'location','best')

figure
hold on
plot(t,Frbfhat,'r-','linewidth',1.5)
plot(t,Ffhat,'b--','linewidth',1.5)
grid on
xlabel('time')
ylabel('error')
% legend({'casadi','IDIM'},'location','best')


%% output data train
% save outputFriction_RBF_32_neu_complete_train.mat yhatRBF ydothatRBF Frbfhat Ffhat paramhat chat dhat What

%% compare both solutions valid

load('DATA_EMPS_PULSES.mat')
decimate = 1; % see Forgione, Piga, EJC 59 (2021) 69-81
u_data = Force1(1:decimate:end);
y_data = qm(1:decimate:end);
t      = t(1:decimate:end);

N  = length(t);  % Number of samples
Ts = round(t(2)-t(1),3);  % sampling time (seconds)
fs = 1/Ts;       % Sampling frequency [hz]

x0 = DM([qm(1),0]); % initial condition for simulation

Xhat = all_samples(x0, u_data, repmat(paramhat,1,N));
Xhat = Xhat.full;
yhatRBF = Xhat(1,:)';
ydothatRBF = Xhat(2,:)';

Frbfhat = RBF_gauss(chat,ydothatRBF,dhat,What);
Ffhat = Fv*ydothatRBF-Fc*sign(ydothatRBF);

figure
hold on
plot(t,y_data,'k-','linewidth',1.5)
plot(t,yhatRBF,'r--','linewidth',1.5)
% plot(t,yhatIDIM,'g--','linewidth',1.5)
grid on
xlabel('time')
% legend({'real','casadi','IDIM'},'location','best')

figure
hold on
plot(t,y_data-yhatRBF,'r-','linewidth',1.5)
% plot(t,y_data-yhatIDIM,'g--','linewidth',1.5)
grid on
xlabel('time')
ylabel('error')
% legend({'casadi','IDIM'},'location','best')

figure
hold on
plot(t,Frbfhat,'r-','linewidth',1.5)
plot(t,Ffhat,'b--','linewidth',1.5)
grid on
xlabel('time')
ylabel('error')
% legend({'casadi','IDIM'},'location','best')

%% output data test
% save outputFriction_RBF_32_neu_complete_test.mat yhatRBF ydothatRBF Frbfhat Ffhat paramhat chat dhat What


%% load data for plotting derivatives (just a test)

% load outputFrictionPol.mat
load outputFrictionIDIM.mat
load outputFriction.mat

% load outputFrictionPol_train.mat
% load outputFrictionIDIM_train.mat
% load outputFriction_train.mat

ergry = y_data - yhat;
% erpol = y_data - yhatPol;
eIDIM = y_data - yhatIDIM;
eRBF  = y_data - yhatRBF;

e_RMS_gry  = sqrt((ergry'*ergry)/length(ergry))
% e_RMS_pol  = sqrt((erpol'*erpol)/length(erpol))
e_RMS_IDIM = sqrt((eIDIM'*eIDIM)/length(eIDIM))
e_RMS_RBF  = sqrt((eRBF'*eRBF)/length(eRBF))

% % 
% figure
% plot(t,abs(ergry), 'k',t, abs(erpol),'r',t, abs(eIDIM),'g',t,abs(eRBF),'b')
% grid on
% legend({'casadi grey','casadi pol','IDIM','RBF'},'location','best')
% 
% figure
% plot(t,y_data,'-k',t,yhat,'b',t,yhatRBF,'r')
% grid on
% legend({'data','casadi gray','RBF'},'location','best')

%% helper functions
function v = denorm(vn,vmax,vmin)
    v = vmin + (vmax-vmin)*vn;
end
function vn = normalize(v,vmax,vmin)
    vn = (v - vmin) ./ (vmax-vmin);
end

% function y = RBF_gauss (c,x,delta,W)
% 
% % out = gauss (centro, entradas, desvio)
% % out    - matriz com (mxn)
% % m      - numero de entradas
% % n      - numero de clusters
% % centro - matriz dos centros calculado pelo K-mean
% % datain - matriz com entradas
% % desvio - vetor com desvio padrao dos centros
% 
% [nc,p] = size(c);
% [N,p]  = size(x);
% 
% sig = sqrt(1./(2*delta));
% 
% % calculo da funcao gaussiana
% % HL = {,}; % hidden layer output
% % HL = MX.sym('HL',N,nc)
% for i=1:nc
%     if i == 1 
%         out = (exp(-0.5*( (( x-ones(N,1) * c(i,:) ) .^2) )*ones(p,1)/(sig(i)^2)));  
%     else
%         out = [out (exp(-0.5*( (( x-ones(N,1) * c(i,:) ) .^2) )*ones(p,1)/(sig(i)^2)))];  
%     end
% end
% 
% y = out * W;
% 
% end

