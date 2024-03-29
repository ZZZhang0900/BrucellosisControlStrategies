import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from tqdm.notebook import tqdm     # progress bar
from nnc.controllers.neural_network.nnc_controllers import\
     NeuralNetworkController, NNCDynamics
import nnc.controllers.baselines.ct_lti.dynamics as dynamics
from examples.directed.small_example_helpers import EluTimeControl, evaluate_trajectory, todf
import LiaoNingbrucellosis.Modelling4 as model
import torch
import math
import copy
from tqdm.notebook import tqdm
from torchdiffeq import odeint
import matplotlib.pyplot as plt


#%%
#device = 'cuda:0'
device = 'cpu'
dtype = torch.float
training_session = True
#training_session = False
n_nodes = 8
n_drivers = 1
parameters = torch.tensor([
    0.89, 0.24, 0.33, 15, 3.6, 0.5, 694e4, 6.28e-3, 0.4, 0.6, 0.00662,
    #0.8282,0.9012,9.664e-5,0.0002605
   0.6346,0.8040,0.0012, 0.0008
])
#b_s  theta/c   lamda  kappa delta  ntao  A   b_h   qgamma   nqgamma   d_h   11个


x0 = torch.tensor([[
        725.6e4,15.88e4,1.1668e+06,6e6,1421.9e4,1039,2288.6,1039
]])
data_year = torch.tensor([
    [606], [853],[1475],[2053],[2797],[2922],[2338],[1954],[2194],[2283],[2965]
])     #每日新增
x_target1 = torch.tensor([
    [606], [1459],[2934],[4987],[7784],[10706],[13044],[14998],[17192],[19475],[22440]
])
x_target=(torch.tensor(x_target1,dtype=torch.float))
#data_year=(torch.tensor(data_year1,dtype=torch.float))
T = 10#10
n_timesteps = 11#11
t = torch.linspace(0, T, n_timesteps)


#data_pre1 = torch.tensor([[606], [1], [2],[ 3], [4], [5], [6], [7], [8], [9], [10]])
#data_pre =(torch.tensor(data_pre1,dtype=torch.float))
######求解方程（1）train()
def train(nnc_dyn, epochs, lr,T,n_timesteps):  # simple training
    optimizer = torch.optim.Adam(nnc_dyn.parameters(), lr=lr)
    trajectories = [model.evaluate_trajectory(linear_dynamics,
                                        nnc_dyn.nnc,
                                        x0,
                                        T,
                                        n_timesteps,
                                        method='rk4',
                                        options=None
                                        )]  # keep trajectories for plots
    parameters = [deepcopy(dict(nnc_dyn.nnc.named_parameters()))]


    for i in tqdm(range(epochs)):
        optimizer.zero_grad()  # do not accumulate gradients over epochs

        x = x0.detach()
        x_nnc = odeint(nnc_dyn, x, t.detach(), method='dopri5')
        x_T = x_nnc[:,:,-1] # reached state at T

        #loss = ((data_year - data_pre) ** 2).sum()  # !No energy regularization
        loss = ((torch.log(x_T) - torch.log(x_target))** 2).sum()
        print('i=  ,loss = ',i,loss)

        loss.backward()
        optimizer.step()

        trajectory = model.evaluate_trajectory(linear_dynamics,
                                         nnc_dyn.nnc,
                                         x0,
                                         T,
                                         n_timesteps,
                                         method='rk4',
                                         options=None,
                                         )




        if i % 1000== 0:
            parameters.append(deepcopy(dict(nnc_dyn.nnc.named_parameters())))
            trajectories.append(trajectory)
            #print('x_nnc = ',x_nnc)
            #print('trajectory = ', trajectory)

            x_nnc = odeint(nnc_dyn, x0.detach(), t.detach(), method='dopri5')
            data = x_nnc[:, :, -1].squeeze(-1)
            data_pre = torch.tensor([606, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            for j in range(1, 11):
                data_pre[j] = data[j] - data[j - 1]
            #print(data_pre)
            plt.plot(t, data_year)  # 绘制x与y的图线
            plt.plot(t, data_pre)
            plt.text(0, 2800, i, size=15, alpha=0.2)
            plt.title('c=0.5,ntao=0.15*12,epochs=10002,lr=0.01,i=%d'%i)
            plt.show()  # 把绘制好的图形表示出来

            nncdata = model.evaluate_trajectory(
                linear_dynamics,
                nnc_model,
                x0,
                T,
                n_timesteps=11,
                method='rk4',
                options=None,
            )
            u_all1 = nncdata[:, :, -1]

            #print("u_all = ", u_all)
            plt.figure()
            plt.plot(t, u_all1.detach().numpy())

            plt.xlabel(r"t")
            plt.ylabel(r"u_all")
            plt.title('c=0.5,ntao=0.15*12,epochs=10002,lr=0.01,i=%d'%i)
            plt.show()

            torch.save(neural_net, '0114c_trained_epochs=3002,lr=0.01,i=%d.torch'%i)


    return parameters, torch.stack(trajectories).squeeze(-2)

######求解方程（2）5
linear_dynamics = model.brucellosis(parameters, dtype, device)
if training_session:
    torch.manual_seed(0)

    neural_net = model.EluTimeControl(n_nodes, n_drivers)
    nnc_model = model.NeuralNetworkController(neural_net)
    nnc_dyn = model.NNCDynamics(linear_dynamics, nnc_model)
    parameters = train(nnc_dyn, 10001, 0.01, t,n_timesteps)  # , 100 epochs, learning rate 0.01
    #df1 = todf(parameters, lr=0.01)
    #torch.save(neural_net, 'trained_elu_net_directed.torch')
    #alldf = pd.concat([df1], ignore_index=True)
    #alldf.to_csv('all_trajectories_directed.csv')
else:
    neural_net = torch.load('1213trained_epochs=10002,lr=0.01,i=9000.torch')
    #alldf = pd.read_csv('all_trajectories_directed.csv', index_col=0)
    nnc_model = model.NeuralNetworkController(neural_net)
    nnc_dyn = model.NNCDynamics(linear_dynamics, nnc_model)

#################画图#######
x_nnc = odeint(nnc_dyn, x0.detach(), t.detach(), method='dopri5')

'''plt.figure()  #打开一个窗口
plt.plot(t,x_target1) #绘制x与y的图线
plt.plot(t,x_nnc[:,:,-1].detach().numpy())
plt.show() #把绘制好的图形表示出来'''


data = x_nnc[:,:,-1].squeeze(-1)
data_pre=[606,1,2,3,4,5,6,7,8,9,10]
for i in range(1,11):
    data_pre[i]=data[i]-data[i-1]
print(data_pre)
plt.plot(t,data_year) #绘制x与y的图线
plt.plot(t,data_pre)
plt.show() #把绘制好的图形表示出来


print('x_target1 = ',x_target1)
print('x_nnc = ',x_nnc[:,:,-1])


#################画图#######
'''t = torch.linspace(0, T, 11)
x = x0.detach()
ld_controlled_lambda = lambda t, x_in: linear_dynamics(t, u=neural_net(t, x_in), x=x_in)
x_all_nn = odeint(ld_controlled_lambda, x0, t, method='rk4')
x_T = x_all_nn[-1, :]
print(str(x_T.flatten().detach().cpu().numpy()))
'''


####
nncdata = model.evaluate_trajectory(
    linear_dynamics,
    nnc_model,
    x0,
    T,
    n_timesteps=11,
    method='rk4',
    options=None,
)

#x_nnc = odeint(nnc_dyn, x0.detach(), t.detach(), method='dopri5')


u_all = nncdata[:, :, -1]
print("u_all = ",u_all)

'''uu = (u_all @ u_all.transpose(1,0))
uu = uu*(torch.ones_like(uu) - torch.ones(uu.shape[0]).diag())
print("uu = ",uu)'''
plt.figure()
plt.plot(t,u_all.detach().numpy())
plt.xlabel(r"t")
plt.ylabel(r"u_all")
plt.show()
'''################%%  U的轨迹图
trajectory = model.evaluate_trajectory(linear_dynamics,
                                 nnc_model,
                                 x0,
                                 T,
                                 n_timesteps,
                                 method='rk4',
                                 options=None
                                 )
nnc_trajectory = trajectory.squeeze(1).unsqueeze(0).detach().numpy()[0]
print("nnc_trajectory = ",nnc_trajectory)

#np.savetxt("nnc_trajectory_directed5.csv", np.c_[nnc_trajectory])

nnc_u = np.array([nnc_trajectory[i,3] for i in range(len(nnc_trajectory))])

energy_nnc = np.cumsum((nnc_u**2)*T/n_timesteps)

#np.savetxt("energies_directed5.csv", np.c_[time, energy_nnc])

plt.figure()
plt.plot(t,energy_nnc)
plt.plot(t,nnc_u)
plt.xlabel(r"t")
plt.ylabel(r"u(t)")
plt.show()'''
