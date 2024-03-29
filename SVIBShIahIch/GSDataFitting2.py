####估计 四个传播率 和 c theta，XiShu = [1, 1, 1, 1, 1e-2, 1e-3].
# 甘肃2010-2015年的拟合结果. 变量用2表示


import numpy as np
import pandas as pd
import torch
import os
from copy import deepcopy
from tqdm.notebook import tqdm     # progress bar
from nnc.controllers.neural_network.nnc_controllers import\
     NeuralNetworkController, NNCDynamics
import nnc.controllers.baselines.ct_lti.dynamics as dynamics
from examples.directed.small_example_helpers import EluTimeControl, evaluate_trajectory, todf
import LiaoNingbrucellosis.GSModelling2 as model
import torch
import math
import copy
from tqdm.notebook import tqdm
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# %%
device = 'cpu'
dtype = torch.float
# training_session = True
training_session = False
n_nodes = 8
n_drivers = 5
lr = 0.01    #学习率
iter = 279   #训练次数
ii = 50  #循环次数整除后画图
parameters = torch.tensor([
     1156e4, 0.58, 0.385, 31.36e4, 6.07*0.001, 0.33, 15, 3.2, 0.3*12
])
#       A       d_s   qgamma     B       d_h      lamda  kappa delta ntao

x0 = torch.tensor([[
    1749e4, 2.97e4*0.99, 7e3, 2560e4, 46, 46*0.385, 2.974e4, 46
]])

# #人急性
I_h_x_new = torch.tensor([
    [46], [50], [118], [499], [1404], [2308]
], dtype=torch.float)     # 每日新增
# #急性累计
I_h_x_cul = torch.tensor([
    [46], [96], [214], [713], [2117], [4425]
], dtype=torch.float)
# 羊的阳性数
I_s_cul = torch.tensor([
    [2.97e4], [4.88e4], [17.41e4], [40.41e4], [66.48e4], [113.80e4]
], dtype=torch.float)

T = 5
n_timesteps = 6
t = torch.linspace(0, T, n_timesteps)

# #####求解方程（1）train()
def train(nnc_dyn, epochs, lr, T, n_timesteps):  # simple training
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

    for i in range(epochs):
        optimizer.zero_grad()  # do not accumulate gradients over epochs

        x = x0.detach()
        x_nnc = odeint(nnc_dyn, x, t.detach(), method='dopri5')
        I_s_cul_X = x_nnc[:, :, -2]
        I_h_x_cul_X = x_nnc[:, :, -1]  # reached state at T

        loss1 = ((torch.log(I_h_x_cul_X) - torch.log(I_h_x_cul)) ** 2).sum()
        loss2 = ((torch.log(I_s_cul_X) - torch.log(I_s_cul)) ** 2).sum()
        loss = loss1 + loss2

        print('i=  ,loss = ', i, loss, loss1, loss2)

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

        # if i % ii == 0:
        if i % ii == 0 or loss<0.1004:
            print('i=  ,loss = ', i, loss)
            print("nncdata拟合变量",x_nnc)

            parameters.append(deepcopy(dict(nnc_dyn.nnc.named_parameters())))
            trajectories.append(trajectory)
            print("nncdata拟合系数", trajectory[:, :, -5:])

            x_nnc = odeint(nnc_dyn, x0.detach(), t.detach(), method='dopri5')
            data = x_nnc[:, :, -1].squeeze(-1)
            data_pre = torch.tensor([46, 1, 2, 3, 4, 5])
            for j in range(1, 6):
                data_pre[j] = data[j] - data[j - 1]
            plt.plot(t, I_h_x_new)  # 绘制x与y的图线
            plt.plot(t, data_pre)
            plt.text(0, 2800, loss, size=15, alpha=0.5)
            plt.title('GanSu,I_h_x_new,acute newly，i=%d'%i)
            plt.show()  # 把绘制好的图形表示出来

            #torch.save(neural_net, 'trained_epochs=12002,lr=0.01,i=%d.torch'%i)

    return parameters, torch.stack(trajectories).squeeze(-2)

# #####求解方程（2）5
linear_dynamics = model.brucellosis(parameters, dtype, device)
if training_session:
    torch.manual_seed(0)

    neural_net = model.EluTimeControl(n_nodes, n_drivers)
    nnc_model = model.NeuralNetworkController(neural_net)
    nnc_dyn = model.NNCDynamics(linear_dynamics, nnc_model)
    parameters = train(nnc_dyn, iter, lr, t, n_timesteps)  # , 100 epochs, learning rate 0.01
    torch.save(neural_net, 'GanSu2,trained_epochs=%d.torch' %iter)

else:
    neural_net = torch.load('GanSu2,trained_epochs=279.torch')
    #alldf = pd.read_csv('all_trajectories_directed.csv', index_col=0)
    nnc_model = model.NeuralNetworkController(neural_net)
    nnc_dyn = model.NNCDynamics(linear_dynamics, nnc_model)

# ################画图#######
x_nnc = odeint(nnc_dyn, x0.detach(), t.detach(), method='dopri5')
nncdata = model.evaluate_trajectory(
    linear_dynamics,
    nnc_model,
    x0,
    T,
    n_timesteps=11,
    method='rk4',
    options=None,
)

#需要的数据
ntao = parameters[8]
delta = parameters[7]
d = parameters[1]
kappa = parameters[6]

S = x_nnc[:,:,0].flatten()
I_x = x_nnc[:, :, -2].flatten()   # 羊阳性数累计
I = x_nnc[:,:,1].flatten()
N = (x_nnc[:,:,0]+x_nnc[:,:,1]).flatten()
I_h_x = x_nnc[:,:,-1].flatten() #急性人数

XiShu = [1e-1, 0, 1e-1, 1, 1e-3, 1e-3]
c = (nncdata[:, :, -5]*XiShu[0]).flatten()
beta_s = (nncdata[:, :, -4]*XiShu[2]).flatten()
beta_sw = (nncdata[:, :, -3]*XiShu[3]).flatten()
beta_hw = (nncdata[:, :, -1]*XiShu[4]).flatten()
beta_h = (nncdata[:, :, -2]*XiShu[5]).flatten()

#################计算Rt
#R_i = torch.tensor([0,1,2,3,4,5,6,7,8,9,10])
R_i= beta_s * S / (N * (c+d))
R_b=kappa * beta_sw * S /(N * (c+d)*(delta + ntao))
R_t = R_i+R_b

LHSmatrix = np.hstack((c.detach().numpy().reshape(6, 1), beta_s.detach().numpy().reshape(6,1),
                       beta_sw.detach().numpy().reshape(6, 1), beta_h.detach().numpy().reshape(6,1),
                       beta_hw.detach().numpy().reshape(6, 1), S.detach().numpy().reshape(6, 1),
                       I.detach().numpy().reshape(6, 1), N.detach().numpy().reshape(6, 1),
                       I_h_x.detach().numpy().reshape(6, 1), R_t.detach().numpy().reshape(6,1),
                       R_i.detach().numpy().reshape(6,1),R_b.detach().numpy().reshape(6,1),I_x.detach().numpy().reshape(6,1)))
LHSmatrixdf = pd.DataFrame(LHSmatrix, columns=['c', 'beta_s', 'beta_sw','beta_h', 'beta_hw','S', 'I', 'N','I_h_x',
                                               'R_t', 'R_i', 'R_b', 'I_x'])
LHSmatrixdf.to_csv(r"E:\文献阅读\AI\nnc-master\LiaoNingbrucellosis\data\pamaters_GS2.csv",
              header=['c', 'beta_s', 'beta_sw','beta_h', 'beta_hw', 'S', 'I','N','I_h_x','R_t','R_i','R_b','I_x'],index=0)