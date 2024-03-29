####估计 四个传播率 和 c theta，XiShu = [1, 1, 1, 1, 1e-2, 1e-3].
# 内蒙古2010-2020年的拟合结果--202311

import os
import numpy as np
import pandas as pd
from copy import deepcopy
from nnc.controllers.neural_network.nnc_controllers import\
     NeuralNetworkController, NNCDynamics
import nnc.controllers.baselines.ct_lti.dynamics as dynamics
from examples.directed.small_example_helpers import EluTimeControl, evaluate_trajectory, todf
import LiaoNingbrucellosis.Modelling as model
import torch
import math
import copy
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# %%
# device = 'cuda:0'
device = 'cpu'
dtype = torch.float
# training_session = True
training_session = False
n_nodes = 10
n_drivers = 6
lr = 0.01    #学习率
iter = 736    #训练次数
ii = 200  #循环次数整除后画图
parameters = torch.tensor([
     6400.0e4, 1.055, 0.385, 21.7e4,5.758*0.001, 0.33, 15, 3.2, 0.3*12
])
#       A       d_s   qgamma     B       d_h      lamda  kappa delta ntao

x0 = torch.tensor([[
    5277.2e4, 128.76e4, 2393e4, 6e6, 2470e4, 16604, 6393, 2393e4,128.76e4,16604
]])
'''x0 = torch.tensor([[
    5277.2e4, 128.76e4, 2393e4, 6e6, 2470e4, 16604, 6393, 128.76e4*0.01, 2393e4,128.76e4,16604
]])'''

# 人急性
I_h_x_new = torch.tensor([
    [16604], [20845], [12817], [9310], [10538], [7777], [6567], [7744], [10111], [14148], [16406]
], dtype=torch.float)     #每日新增
# 急性累计
I_h_x_cul = torch.tensor([
    [16604], [37449], [50266], [59576], [70114], [77891], [84458], [92202], [102313], [116461], [132867]
], dtype=torch.float)
# 人口
S_h_x_cul = torch.tensor([
    [2470.6e4], [2482e4], [2489.9e4], [2497.6e4], [2504.8e4], [2511.0e4], [2520.1e4], [2528.6e4], [2534e4], [2540e4], [2405e4]
], dtype=torch.float)
# 免疫
V_s_cul = torch.tensor([
    [2992e4], [6498.4e4], [9096.6e4], [12577.7e4], [16381.4e4], [20068e4], [23518.9e4], [27169e4], [29972.2e4], [32762e4], [35600e4]
], dtype=torch.float)
# 羊数
S_s_x_cul = torch.tensor([
    [5277.24e4], [5275.95e4], [5144e4], [5239.2e4], [5569.3e4], [5777.8e4], [5506.2e4], [6111.9e4], [6001.9e4], [5976e4], [6074.2e4]
], dtype=torch.float)
# S_s_x_cul1 = S_s_x_cul - V_s_cul1*0.8
# 羊的阳性数
I_s_cul = torch.tensor([
    [128.76e4], [234.81e4], [284.71e4], [330.81e4], [363.67e4], [390.83e4], [413.95e4], [437.79e4]
], dtype=torch.float)

T = 10
n_timesteps = 11
t = torch.linspace(0, T, n_timesteps)

# #####求解方程（1）train()
def train(nnc_dyn, epochs, lr,T,n_timesteps):   # simple training
    optimizer = torch.optim.Adam(nnc_dyn.parameters(), lr=lr)
    trajectories = [model.evaluate_trajectory(linear_dynamics, nnc_dyn.nnc, x0, T, n_timesteps, method='rk4', options=dict(step_size=T / n_timesteps))]  # keep trajectories for plots
    parameters = [deepcopy(dict(nnc_dyn.nnc.named_parameters()))]

    for i in range(epochs):
        optimizer.zero_grad()  # do not accumulate gradients over epochs

        x = x0.detach()
        x_nnc = odeint(nnc_dyn, x, t.detach(), method='dopri5')
        V_x_cul_X = x_nnc[:, :, -3]  # 羊免疫
        I_s_cul_X = x_nnc[0:8, :, -2]  # 羊阳性
        # I_cs_cul_X = x_nnc[:, :, -2]  #羊扑杀
        I_h_x_cul_X = x_nnc[:, :, -1]  # 人病例
        S_h_x_cul_X = x_nnc[:, :, 4]  # reached state at T
        S_s_x_cul_X = x_nnc[:, :, 0]
        # print("x_nnc", x_nnc)

        '''loss1 = ((torch.log(I_h_x_cul_X) - torch.log(I_h_x_cul))** 2).sum()\
               +((torch.log(S_h_x_cul_X) - torch.log(S_h_x_cul))** 2).sum()\
               +((torch.log(S_s_x_cul_X) - torch.log(S_s_x_cul))** 2).sum()\
               +((torch.log(V_x_cul_X) - torch.log(V_s_cul*0.8))** 2).sum()'''
        loss = ((torch.log(I_h_x_cul_X) - torch.log(I_h_x_cul)) ** 2).sum()\
               + ((torch.log(I_s_cul_X) - torch.log(I_s_cul)) ** 2).sum()\
               + ((torch.log(V_x_cul_X) - torch.log(V_s_cul * 0.8)) ** 2).sum()


        print('i=  ,loss = ',i,loss)

        loss.backward()
        optimizer.step()

        trajectory = model.evaluate_trajectory(linear_dynamics, nnc_dyn.nnc, x0, T, n_timesteps, method='rk4', options=dict(step_size=T / n_timesteps))

        if i % ii == 0:
        #if i % ii == 0 or loss<0.055:
            print('i=  ,loss = ', i, loss)
            parameters.append(deepcopy(dict(nnc_dyn.nnc.named_parameters())))
            trajectories.append(trajectory)
            print("nncdata拟合系数", trajectory[:, :, -6:])

            x_nnc = odeint(nnc_dyn, x0.detach(), t.detach(), method='dopri5')
            data = x_nnc[:, :, -1].squeeze(-1)
            data_pre = torch.tensor([16604, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            for j in range(1, 11):
                data_pre[j] = data[j] - data[j - 1]
            plt.plot(t, I_h_x_new)  # 绘制x与y的图线
            plt.plot(t, data_pre)
            plt.text(0, 2800, loss, size=15, alpha=0.2)
            plt.title('neimeng,I_h_x_new,acute newly，i=%d'%i)
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
    parameters, t2 = train(nnc_dyn, iter, lr, t, n_timesteps)  # , 100 epochs, learning rate 0.01
    torch.save(neural_net, 'NeiMeng,trained_epochs=%d.torch' %iter)

else:
    neural_net = torch.load('NeiMeng,trained_epochs=736.torch')
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

# 需要的数据
ntao = parameters[8]
delta = parameters[7]
d = parameters[1]
kappa = parameters[6]

S = x_nnc[:, :, 0].flatten()
V = x_nnc[:, :, 2].flatten()
N = (x_nnc[:, :, 0]+x_nnc[:, :, 1]+x_nnc[:, :, 2]).flatten()
I = x_nnc[:, :, 1].flatten()
I_s_x = x_nnc[:, :, -2].flatten()  # 病羊数

XiShu = [1, 1, 1, 1, 1e-2, 1e-3]
c = (nncdata[:, :, -6]*XiShu[0]).flatten()
theta = (nncdata[:, :, -5]*XiShu[1]).flatten()
beta_s = (nncdata[:, :, -4]*XiShu[2]).flatten()
beta_sw = (nncdata[:, :, -3]*XiShu[3]).flatten()
beta_h = (nncdata[:, :, -2]*XiShu[4]).flatten()
beta_hw = (nncdata[:, :, -1]*XiShu[5]).flatten()


print("beta_s",beta_s)
print("beta_sw",beta_sw)
print("beta_h",beta_h)
print("beta_hw",beta_hw)



LHSmatrix = np.hstack((c.detach().numpy().reshape(11, 1), theta.detach().numpy().reshape(11,1),beta_s.detach().numpy().reshape(11,1),
                       beta_sw.detach().numpy().reshape(11, 1), beta_hw.detach().numpy().reshape(11,1),beta_h.detach().numpy().reshape(11,1),
                       S.detach().numpy().reshape(11, 1), N.detach().numpy().reshape(11,1),I.detach().numpy().reshape(11,1),
                       V.detach().numpy().reshape(11,1)))
LHSmatrixdf = pd.DataFrame(LHSmatrix,columns=['c', 'theta', 'beta_s', 'beta_sw','beta_hw','beta_h', 'S', 'N','I','V'])
LHSmatrixdf.to_csv(r"E:\文献阅读\AI\nnc-master\LiaoNingbrucellosis\data\pamaters_NM.csv",
              header=['c', 'theta', 'beta_s', 'beta_sw', 'beta_hw', 'beta_h', 'S', 'N','I','V'])
LHSmatrixdf.to_csv(r"E:\文献阅读\AI\nnc-master\Matlab\pamaters_NM.csv",
              header=['c', 'theta', 'beta_s', 'beta_sw', 'beta_hw', 'beta_h', 'S', 'N','I','V'])


colors_use = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#bec1d4', '#bb7784', '#0000ff', '#111010', '#FFFF00',
              '#1f77b4', '#800080', '#959595', '#7d87b9', '#bec1d4', '#d6bcc0', '#bb7784', '#8e063b', '#4a6fe3',
              '#8595e1', '#b5bbe3', '#e6afb9', '#e07b91', '#d33f6a', '#11c638', '#8dd593', '#c6dec7', '#ead3c6',
              '#f0b98d', '#ef9708', '#0fcfc0', '#9cded6', '#d5eae7', '#f3e1eb', '#f6c4e1', '#f79cd4']
# ################计算Rt
R_i = beta_s * S / (N * (c+d))
R_b = kappa * beta_sw * S /(N * (c+d)*(delta + ntao))
R_t = R_i+R_b

for i in range(11):
    t2 = np.arange(i, i + 2, 1)
    Rt = np.expand_dims(R_t[i].detach().numpy(), 0)
    R_tt = np.concatenate((Rt, Rt))
    Ri = np.expand_dims(R_i[i].detach().numpy(), 0)
    R_ii = np.concatenate((Ri, Ri))
    Rb = np.expand_dims(R_b[i].detach().numpy(), 0)
    R_bb = np.concatenate((Rb, Rb))

    plt.plot(t2, R_tt, c='#f47e62', linewidth=2.5)   # 红色
    plt.plot(t2, R_ii, c='#6f80be', linewidth=2.5)    #蓝色
    plt.plot(t2, R_bb, c='#bd8ec0', linewidth=2.5)    # 紫色

    if R_t[i] < 1:
        plt.fill_between(t2, 0, 1.55, facecolor='#fff2df')
    else:
        plt.fill_between(t2, 0, 1.55, facecolor='#d6ecf0')
    plt.plot([i, i], [0, 1.55], c='b', alpha=0.25, linestyle='--')
plt.plot([11, 11], [0, 1.55], c='b', alpha=0.25, linestyle='--')
plt.plot([0, 11], [1, 1], c='r', alpha=0.85, linestyle='--')

# plt.fill_betweenx([0,1], 0, 11.5, facecolor='#fff2df')
# plt.fill_betweenx([1,1.55], 0, 11.5, facecolor='#d6ecf0')
plt.xlim(0, 11)
plt.ylim(0, 1.55)
# plt.show()
time = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
x_indexes = np.arange(len(time))
plt.xticks(ticks=x_indexes, labels=time, rotation=30, fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Time (Year)", fontsize=13, fontproperties="Times New Roman")
plt.ylabel(r"Average effective reproduction number ($R_t$(t))", fontsize=13, fontproperties="Times New Roman")

#plt.legend(["$R_{rt}$(t)", "$R_{rt}^s$(t)", "$R_{rt}^b$(t)"], ncol=3,
#           bbox_to_anchor=(0.75, -0.35), prop={'family': 'Times New Roman', 'size': 12})

foo_fig = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
foo_fig.savefig(os.path.join(figure_save_path, 'NM_Rt.eps'), dpi=600, format='eps', bbox_inches="tight")
plt.show()

# ################计算beta
# 计算beta的均值和方差
print("---NM--------------mean---------------------------------std")
print("beta_s",beta_s.mean(),beta_s.std())
print("beta_sw",beta_sw.mean(),beta_sw.std())
print("beta_h",beta_h.mean(),beta_h.std())
print("beta_hw",beta_hw.mean(),beta_hw.std())


plt.plot(t, beta_s.detach().numpy(), alpha=0.45, linewidth=2.5, c='g')   # 绿色
plt.plot(t, beta_sw.detach().numpy(), alpha=0.45, linewidth=2.5, c='r')    # 红色
plt.plot(t, beta_h.detach().numpy() * 10, alpha=0.45, linewidth=2.5, c='#FFDD44')   #黄色
plt.plot(t, beta_hw.detach().numpy() * 100, alpha=0.45, linewidth=2.5, c='purple')   #紫色

for i in range(11):
    plt.scatter(t, beta_s.detach().numpy(), marker='p', color='#66CD00', s=140)   # 绿色
    plt.scatter(t, beta_sw.detach().numpy(), marker='>', color='#e07b91', s=140)    # 粉红色
    plt.scatter(t, beta_h.detach().numpy()* 10, marker='*', color='#f47e62', s=140)
    plt.scatter(t, beta_hw.detach().numpy() * 100, marker='x', color='#7d87b9', s=140)

    tt = np.arange(i, i + 2, 1)
    if R_t[i] < 1:
        plt.fill_between(tt, 0, 1, facecolor='#fff2df')
    else:
        plt.fill_between(tt, 0, 1, facecolor='#d6ecf0')

plt.xlim(0, 10.1)
plt.ylim(0, 1)
plt.xticks(ticks=x_indexes[0:11], labels=time[0:11], rotation=30, fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Time (Year)", fontsize=14, fontproperties="Times New Roman")
plt.ylabel(r"Transmission rate ($\beta$(t))", fontsize=14, fontproperties="Times New Roman")
# 图例位置，大小
'''plt.legend([r"$\beta_s$", r"$\beta_{sb}$", r"$\beta_h$*10", r"$\beta_{hb}$*100"],
           ncol=4, bbox_to_anchor=(0.9, -0.3), prop={'family': 'Times New Roman', 'size': 12})'''

fig_b = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_b.savefig(os.path.join(figure_save_path, 'NM_betat.eps'), dpi=600, format='eps', bbox_inches="tight")
plt.show()
# ################人急性的拟合结果+短期预测
t_pre =  torch.linspace(0, 11, 12)
x_nnc_pre = odeint(nnc_dyn, x0.detach(), t_pre.detach(), method='dopri5')
I_h_x_new_hat = x_nnc_pre[:, :, -1].flatten()
data_pre = torch.tensor([16604, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
# 人急性+2021年
I_h_x_new_pre = torch.tensor([
    [16604], [20845], [12817], [9310], [10538], [7777], [6567], [7744], [10111], [14148], [16406],[21313]
], dtype=torch.float)
for j in range(1, 12):
    data_pre[j] = I_h_x_new_hat[j] - I_h_x_new_hat[j - 1]
plt.bar(t_pre, I_h_x_new_pre, color='#3D9140', alpha=0.45)   # 绘制x与y的图线
plt.plot(np.linspace(10,11,2), data_pre[10:12].detach().numpy(), alpha=0.4, color='#F08080',marker='*', markersize=20, linestyle='-.', linewidth=4)
plt.plot(t_pre[0:11], data_pre[0:11].detach().numpy(), alpha=0.4, color='red',marker='*', markersize=20, linestyle='-.', linewidth=4)
plt.xlim(-0.5, 11.5)
x_indexes = np.arange(len(time))
plt.xticks(ticks=x_indexes, labels=time, rotation=30, fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Time (Year)", fontsize=14, fontproperties="Times New Roman")
plt.ylabel("Number of newly infected population", fontsize=14, fontproperties="Times New Roman")
#plt.legend(["the number of estimated cases", "the number of predicted cases","the number of yearly newly confirmed cases",], ncol=3,
#        bbox_to_anchor=(0.95, -0.35), prop={'family': 'Times New Roman', 'size': 12})  #添加图例需要将第一个第二个plot交换位置


fig_h = plt.gcf()
figure_save_path = "picture"
# plt调用gcf函数取得当前绘制的figure并调用savefig函数
# 'get current figure'

if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_h.savefig(os.path.join(figure_save_path, 'NM_InfectedHuman_nihe.eps'), dpi=1000, format='eps', bbox_inches="tight")
plt.show()
# #####################PRCC
prcc = pd.read_csv(r'E:\文献阅读\AI\nnc-master\Matlab\PRCC_NM.csv')
# LHSmatrix = [beta_s_LHS beta_sb_LHS c_LHS kappa_LHS ntao_LHS ds_LHS delta_LHS S1_LHS]
prcc = prcc.to_csv("PRCC_NM2.csv", header=['beta_s', 'beta_sb', 'c', 'kappa', 'ntao', 'd_s', 'delta', 'S1'], index=0)
prcc = pd.read_csv(r'PRCC_NM2.csv')
# df = pd.read_csv("MappingAnalysis_Data.csv")

t=np.linspace(0,10,11)

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
markers = ['o','s','H','D','^','<','d']
labels = [r"$\beta_s$", r"$\beta_sb$", "c", r"$\kappa$", r"n$\tau$",r"$d_s$",r"$\delta$"]

plt.plot(t, prcc.iloc[:, [0]],
             marker=markers[0], markerfacecolor=colors_use[0], markersize=8, markeredgewidth=0.5,
             color="k", linewidth=0.9, linestyle="-", label=labels[0])
plt.plot(t, prcc.iloc[:, [1]],
             marker=markers[1], markerfacecolor=colors_use[1], markersize=7, markeredgewidth=0.5,
             color="k", linewidth=0.9, linestyle="-", label=labels[1])
plt.plot(t, prcc.iloc[:, [2]],
             marker=markers[2], markerfacecolor=colors_use[2], markersize=8, markeredgewidth=0.5,
             color="k", linewidth=0.9, linestyle=":", label=labels[2])
plt.plot(t, prcc.iloc[:, [3]],
             marker=markers[3], markerfacecolor=colors_use[3], markersize=7, markeredgewidth=0.5,
             color="k", linewidth=0.9, linestyle="-.", label=labels[3])
plt.plot(t, prcc.iloc[:, [4]],
             marker=markers[4], markerfacecolor=colors_use[4], markersize=7, markeredgewidth=0.5,
             color="k", linewidth=0.9, linestyle='solid', label=labels[4])
plt.plot(t, prcc.iloc[:, [5]],
             marker=markers[5], markerfacecolor=colors_use[8], markersize=7, markeredgewidth=0.5,
             color="k", linewidth=0.9, linestyle='dashdot', label=labels[5])
plt.plot(t, prcc.iloc[:, [6]],
             marker=markers[6], markerfacecolor=colors_use[7], markersize=7, markeredgewidth=0.5,
             color="k", linewidth=0.9, linestyle='dotted', label=labels[6])
plt.plot([0,11], [0, 0], c='b', alpha=0.25,linestyle='--')
plt.fill_between([-0.15,11],0.2,0.4, facecolor='#C0C0C0',alpha=0.7)
plt.fill_between([-0.15,11],-0.2,-0.4, facecolor='#C0C0C0',alpha=0.7)
plt.fill_between([-0.15,11],0.4,1.05, facecolor='#808A87',alpha=0.7)
plt.fill_between([-0.15,11],-0.4,-1.05, facecolor='#808A87',alpha=0.7)
#plt.yticks(ticks=[-1,-0.8,-0.4,-0.2,0,0.2,0.4,0.6,1],fontsize=10)
plt.xlim(-0.15, 10.5)
plt.ylim(-1.05, 1.05)
x_indexes = np.arange(len(time)-1)
plt.xticks(ticks=x_indexes, labels=time[0:11], rotation=30, fontsize=10)
plt.yticks(np.linspace(-1, 1, 11, endpoint=True), fontsize=10)
plt.xlabel("Time (Year)", fontsize=14, fontproperties="Times New Roman")
plt.ylabel("PRCC for $R_t$(t)", fontsize=14, fontproperties="Times New Roman")
'''# 图例位置，大小
labels = [r"$\beta_s$", r"$\beta_sb$", "c", r"$\kappa$", r"n$\tau$",r"$d_s$",r"$\delta$"]
plt.legend([r"$\beta_s$", r"$\beta_{sb}$", "c", r"$\kappa$", r"n$\tau$",r"$d_s$",r"$\delta$"], ncol=7,
            bbox_to_anchor=(0.95, -0.25), prop={'family': 'Times New Roman', 'size': 12})'''

fig_p = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_p.savefig(os.path.join(figure_save_path, 'NM_prcc.eps'), dpi=1000, format='eps', bbox_inches="tight")
plt.show()

# #####################扑杀
PuSha = c*I
t=np.linspace(0,10,11)

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
markers = ['o','s','H','D','^','<','d']
labels = [r"$\beta_s$", r"$\beta_sb$", "c", r"$\kappa$", r"n$\tau$",r"$d_s$",r"$\delta$"]

plt.plot(t, PuSha.detach().numpy(),
             marker=markers[6], markerfacecolor='r', markersize=7, markeredgewidth=0.5,
             color="k", linewidth=0.9, linestyle='dotted')
plt.xlim(-0.15, 10.5)

# x_indexes = np.arange(len(time))
plt.xticks(ticks=x_indexes, labels=time[0:11], rotation=30, fontsize=10)
plt.xlabel("Time (Year)", fontsize=14, fontproperties="Times New Roman")
plt.ylabel("The number of sheep killed", fontsize=14, fontproperties="Times New Roman")

fig_psN = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_psN.savefig(os.path.join(figure_save_path, 'NM_kill.eps'), dpi=1000, format='eps', bbox_inches="tight")
plt.show()

# #####################扑杀率c和拐点
from kneed import KneeLocator
def knee_test(df, cols):
        output_knees = []
        x, y = df,cols
        for curve in ['convex', 'concave']:
            for direction in ['increasing', 'decreasing']:
                model = KneeLocator(x=x, y=y, curve=curve, direction=direction, online=False)
                if model.knee != x[0] and model.knee != x[-1]:
                    output_knees.append((model.knee, model.knee_y, curve, direction))

        # 根据变量是否发现拐点，根据结果给予不同的提示及图形展示
        if output_knees.__len__() != 0:
            print('发现拐点！')
            return output_knees
        else:
            print('未发现拐点！')


t=np.linspace(0,10,11)
knee_info = knee_test(t,c.detach().numpy())
print("knee_info",knee_info)


#fig, axe = plt.subplots(figsize=[8, 6])
plt.plot([2, 2], [0, 1], c='g', alpha=0.25, linestyle='--')
plt.plot([6, 6], [0, 1], c='b', alpha=0.25, linestyle='--')
plt.plot(t, c.detach().numpy(), marker=markers[0], markerfacecolor='k', markersize=7, markeredgewidth=0.5,
             color="k", linewidth=0.9, linestyle='dotted')
for point in knee_info:
    plt.scatter(x=point[0], y=point[1], c='r', s=200, marker='o')
    plt.annotate(text=f'{point[2]} {point[3]}', xy=(point[0]+0.3, point[1]+0.05), fontsize=12)



plt.xlim(-0.15, 10.5)
plt.ylim(0, 1)
plt.xticks(ticks=x_indexes, labels=time[0:11], rotation=30, fontsize=10)
plt.xlabel("Time (Year)", fontsize=14, fontproperties="Times New Roman")
plt.ylabel("The culling rate c(t)", fontsize=14, fontproperties="Times New Roman")
# plt.legend(["NMLDPCP","NBPCP", "The culling rate c(t)","inflection point in learnt c(t)"], ncol=4,
#            bbox_to_anchor=(0.95, -0.25), prop={'family': 'Times New Roman', 'size': 12})


fig_pscG = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_pscG.savefig(os.path.join(figure_save_path, 'NM_killc.eps'), dpi=1000, format='eps', bbox_inches="tight")
plt.show()




'''# #####################传播风险因素  热图, 用免疫率、扑杀率计算的结果
risk = np.hstack((I_h_x_new.reshape(11, 1), beta_s.detach().numpy().reshape(11, 1),beta_sw.detach().numpy().reshape(11, 1),
                  beta_h.detach().numpy().reshape(11, 1),beta_hw.detach().numpy().reshape(11, 1),
                  theta.detach().numpy().reshape(11,1),c.detach().numpy().reshape(11,1)))

riskdf = pd.DataFrame(risk,columns=['Human','S-to-S','B-to-S','S-to-H','B-to-H','Immunization','Culling'])

# co = risk.corr()#皮尔逊相关系数
risk_sp = riskdf.corr(method='spearman')#斯皮尔曼相关系数
print(risk_sp)

vals = ['Human','S-to-S','B-to-S','S-to-H','B-to-H','Immunization','Culling']
risk_sp.insert(loc=0, column='vals', value=vals)
df_melt=pd.melt(risk_sp, id_vars=risk_sp.iloc[:,[0]],value_name='value',var_name='vals2')
df_melt = df_melt.drop([7,14,15,21,22,23,28,29,30,31,35,36,37,38,39,42,43,44,45,46,47])
df_melt.plot.scatter(x='vals', y='vals2', c='value', s=df_melt['value'].abs() * 1700, colormap='Oranges',colorbar=False)

plt.yticks(rotation=90, fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel("Control Measures", fontsize=14, fontproperties="Times New Roman")
plt.ylabel("Control Measures", fontsize=14, fontproperties="Times New Roman")
plt.xlim(-0.44, 6.44)
plt.ylim(-0.57, 6.57)

fig_hr = plt.gcf()
figure_save_path = "picture"
# plt调用gcf函数取得当前绘制的figure并调用savefig函数
# 'get current figure'

if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_hr.savefig(os.path.join(figure_save_path, 'NM_risk.eps'), dpi=1000, format='eps', bbox_inches="tight")
plt.show()'''



# #####################传播风险因素  热图, 用免疫数、扑杀数计算的结果
I_x = x_nnc[:, :, -2].flatten()   # 羊阳性数累计
V_x = x_nnc[:, :, -3].flatten()   # 羊免疫数累计


I_x_new = torch.tensor([1287600, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
V_x_new = torch.tensor([2393e4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

for j in range(1, 11):
    I_x_new[j] = I_x[j] - I_x[j - 1]
    V_x_new[j] = V_x[j] - V_x[j - 1]



# 下面是用免疫数、扑杀数计算的结果
risk = np.hstack((I_h_x_new.reshape(11, 1), I_x_new.detach().numpy().reshape(11, 1),
                  V_x_new.detach().numpy().reshape(11,1),PuSha.detach().numpy().reshape(11,1),
                       ))

riskdf = pd.DataFrame(risk,columns=['Human','Positive','Immunization','Culling'])

# co = risk.corr()#皮尔逊相关系数
risk_sp = riskdf.corr(method='spearman')#斯皮尔曼相关系数
print(risk_sp)

#  下面部分是普通热图 ， import 和 sns.heatmap，已不用了
# import seaborn as sns
# sns.heatmap(data=risk_sp,vmin=-1, vmax=1,cbar=False,cmap=plt.get_cmap('Oranges'),annot=True,fmt=".2f",annot_kws={'size':8,'weight':'normal', 'color':'#253D24'},
#            mask=np.triu(np.ones_like(risk_sp,dtype=np.bool))#显示对脚线下面部分图)

vals = ['Human','Positive','Immunization','Culling']
risk_sp.insert(loc=0, column='vals', value=vals)
df_melt=pd.melt(risk_sp, id_vars=risk_sp.iloc[:,[0]],value_name='value',var_name='vals2')
df_melt = df_melt.drop([4,8,9,12,13,14])


df_melt.plot.scatter(x='vals', y='vals2', c='value', s=df_melt['value'].abs() * 2600, colormap='Oranges',colorbar=False)

# ##下面三行用于设置colorbar， import norm1  cbar1，已不用了
#import matplotlib as mpl
#norm1 = mpl.colors.Normalize(vmin=-1, vmax=1)
# cbar1 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm1, cmap='Oranges'), orientation='horizontal',ticks=np.linspace(-1, 1, 5), label='value')

plt.yticks(rotation=90, fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel("Control Measures", fontsize=14, fontproperties="Times New Roman")
plt.ylabel("Control Measures", fontsize=14, fontproperties="Times New Roman")
plt.xlim(-0.36, 3.36)
plt.ylim(-0.36, 3.36)

fig_hr = plt.gcf()
figure_save_path = "picture"
# plt调用gcf函数取得当前绘制的figure并调用savefig函数
# 'get current figure'

if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_hr.savefig(os.path.join(figure_save_path, 'NM_risk.eps'), dpi=1000, format='eps', bbox_inches="tight")
plt.show()


'''#  柱状相关性
yinsu = ['Positive cases','Immunization','Culling','Human']
xr_indexes = np.arange(len(yinsu))*0.7
plt.bar(xr_indexes, risk_sp['human'][0:3],width=0.6, color=colors_use[15], alpha=0.25)
plt.plot([-0.42,2.5], [0, 0], c='black', alpha=0.45,linestyle='-')
# plt.show()
plt.xticks(ticks=xr_indexes, labels=yinsu, fontsize=10)# 绘制x与y的图线
plt.yticks(ticks=[-0.854545,0,0.554545,0.9,],fontsize=10)
plt.xlim(min(xr_indexes)-0.4,max(xr_indexes)+0.4)
plt.ylim(-1.1, 1.1)
plt.xlabel("Communication risk factors", fontsize=14, fontproperties="Times New Roman")
plt.ylabel("Correlation coefficient", fontsize=14, fontproperties="Times New Roman")
# plt.text(xr_indexes[0] +0.1, risk_sp['human'][0] + 0.3, risk_sp['human'][0], ha='center',fontproperties='Times New Roman', fontsize=10,  zorder=10) # 标数值
# plt.show()

fig_hr = plt.gcf()
figure_save_path = "picture"
# plt调用gcf函数取得当前绘制的figure并调用savefig函数
# 'get current figure'

if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_hr.savefig(os.path.join(figure_save_path, 'NM_risk.eps'), dpi=1000, format='eps', bbox_inches="tight")
plt.show()

####
I_s_cul_new = torch.tensor([1287600, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
for j in range(1, 8):
    I_s_cul_new[j] = I_s_cul[j] - I_s_cul[j - 1]
plt.bar(t, I_s_cul_new, color='g', alpha=0.45)   # 绘制x与y的图线
plt.plot(t, I_x_new.detach().numpy(), alpha=0.45, color='red',marker='*', markersize=20, linestyle='-.', linewidth=4)
plt.show()

V_s_cul_new = torch.tensor([2.9920e+07*0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
for j in range(1, 8):
    V_s_cul_new[j] = V_s_cul[j] - V_s_cul[j - 1]
plt.bar(t, V_s_cul_new*0.8, color='g', alpha=0.45)   # 绘制x与y的图线
plt.plot(t, V_x_new.detach().numpy(), alpha=0.45, color='red',marker='*', markersize=20, linestyle='-.', linewidth=4)
plt.show()'''
