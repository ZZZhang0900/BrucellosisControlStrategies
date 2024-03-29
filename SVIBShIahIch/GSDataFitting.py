####估计 四个传播率 和 c theta，XiShu = [1, 1, 1, 1, 1e-2, 1e-3].
# 甘肃2016-2020年的拟合结果，画图部分增加了2010-2015GSDataFitting的数据
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
import LiaoNingbrucellosis.GSModelling as model
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
#training_session = True
training_session = False
n_nodes = 10
n_drivers = 6
lr = 0.01    #学习率
iter = 1508   #训练次数
ii=500  #循环次数整除后画图
parameters = torch.tensor([
     1650e4, 0.78, 0.385, 29.66e4,6.80*0.001, 0.33, 15, 3.2, 0.3*12
])
#       A       d_s   qgamma     B       d_h      lamda  kappa delta ntao

x0 = torch.tensor([[
    1877.4e4-719.9e4*0.8, 17.65e4, 719.9e4*0.8, 2e5, 2609e4, 1745, 1745*0.385, 719.08e4*0.8,17.65e4,1745
]])

##人急性
I_h_x_new1 = torch.tensor([
    [1745],[1631],[1584],[1787],[3004]
],dtype=torch.float)     #每日新增
##急性累计
I_h_x_cul = torch.tensor([
    [1745], [3376],[4960],[6747],[9751]
],dtype=torch.float)
##人口
S_h_x_cul = torch.tensor([
    [2609.95e4],[2625.71e4],[2637.26e4],[2647.43e4],[2501.98e4]
],dtype=torch.float)
##免疫
V_s_cul = torch.tensor([
    [719.90e4], [1425.42e4], [2148.58e4], [2910.55e4],
],dtype=torch.float)
##羊数
S_s_x_cul = torch.tensor([
    [1877.4e4],[1839.4e4],[1885.9e4],[1987.1e4],[2191.8e4]
],dtype=torch.float)
#S_s_x_cul1 = S_s_x_cul - V_s_cul1*0.8
#羊的阳性数
I_s_cul = torch.tensor([
    [17.65e4],[32.00e4],[40.11e4],[65.74e4],[83.28e4]
],dtype=torch.float)

T = 4
n_timesteps = 5
t = torch.linspace(0, T, n_timesteps)

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

    for i in range(epochs):
        optimizer.zero_grad()  # do not accumulate gradients over epochs

        x = x0.detach()
        x_nnc = odeint(nnc_dyn, x, t.detach(), method='dopri5')
        V_x_cul_X = x_nnc[:4, :, -3]
        I_s_cul_X = x_nnc[:, :, -2]
        I_h_x_cul_X = x_nnc[:,:,-1] # reached state at T
        S_h_x_cul_X = x_nnc[:,:,4] # reached state at T
        S_s_x_cul_X = x_nnc[:,:,0]

        loss1 = ((torch.log(I_h_x_cul_X) - torch.log(I_h_x_cul)) ** 2).sum()
        loss2 = ((torch.log(I_s_cul_X) - torch.log(I_s_cul)) ** 2).sum()
        loss3 = ((torch.log(V_x_cul_X) - torch.log(V_s_cul * 0.8)) ** 2).sum()
        loss = loss1 + loss2+ loss3

        print('i=  ,loss012 = ', i, loss, loss1, loss2)
        print('     loss3 = ', loss3)

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

        #if i % ii == 0:
        if i % ii == 0 or loss<0.0130:
            print('i=  ,loss = ', i, loss)
            print("nncdata拟合变量",x_nnc)

            parameters.append(deepcopy(dict(nnc_dyn.nnc.named_parameters())))
            trajectories.append(trajectory)
            print("nncdata拟合系数", trajectory[:, :, -6:])

            x_nnc = odeint(nnc_dyn, x0.detach(), t.detach(), method='dopri5')
            data = x_nnc[:, :, -1].squeeze(-1)
            data_pre = torch.tensor([1745, 1, 2, 3, 4])
            for j in range(1, 5):
                data_pre[j] = data[j] - data[j - 1]
            plt.plot(t, I_h_x_new1)  # 绘制x与y的图线
            plt.plot(t, data_pre)
            plt.text(0, 2800, loss, size=15, alpha=0.5)
            plt.title('GanSu,I_h_x_new1,acute newly，i=%d'%i)
            plt.show()  # 把绘制好的图形表示出来

            #torch.save(neural_net, 'trained_epochs=12002,lr=0.01,i=%d.torch'%i)

    return parameters, torch.stack(trajectories).squeeze(-2)

######求解方程（2）5
linear_dynamics = model.brucellosis(parameters, dtype, device)
if training_session:
    torch.manual_seed(0)

    neural_net = model.EluTimeControl(n_nodes, n_drivers)
    nnc_model = model.NeuralNetworkController(neural_net)
    nnc_dyn = model.NNCDynamics(linear_dynamics, nnc_model)
    parameters = train(nnc_dyn, iter, lr, t,n_timesteps)  # , 100 epochs, learning rate 0.01
    torch.save(neural_net, 'GanSu,trained_epochs=%d.torch' %iter)

else:
    neural_net = torch.load('GanSu,trained_epochs=1508.torch')
    #alldf = pd.read_csv('all_trajectories_directed.csv', index_col=0)
    nnc_model = model.NeuralNetworkController(neural_net)
    nnc_dyn = model.NNCDynamics(linear_dynamics, nnc_model)

#################画图#######
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

# 需要的数据 2016-2020
ntao = parameters[8]
delta = parameters[7]
d = parameters[1]
kappa = parameters[6]

S1 = x_nnc[:,:,0].flatten()
I1 = x_nnc[:, :, 1].flatten()
I_x1 = x_nnc[:, :, -2].flatten()   # 羊阳性数累计
V_x1 = x_nnc[:, :, -3].flatten()   # 羊免疫数累计
N1 = (x_nnc[:,:,0]+x_nnc[:,:,1]+x_nnc[:,:,2]).flatten()
I_h_x1 = x_nnc[:,:,-1].flatten() #急性人数

XiShu = [1e-2, 1, 1e-1, 1e-1, 1e-3, 1e-3]
c1 = (nncdata[:, :, -6]*XiShu[0]).flatten()
theta1 = (nncdata[:, :, -5]*XiShu[1]).flatten()
beta_s1 = (nncdata[:, :, -4]*XiShu[2]).flatten()
beta_sw1 = (nncdata[:, :, -3]*XiShu[3]).flatten()
beta_hw1 = (nncdata[:, :, -1]*XiShu[4]).flatten()
beta_h1 = (nncdata[:, :, -2]*XiShu[5]).flatten()

# 需要的数据 2010-2015
pamaters2 = pd.read_csv(r'E:\文献阅读\AI\nnc-master\LiaoNingbrucellosis\data\pamaters_GS2.csv')
# LHSmatrix = [beta_s_LHS beta_sb_LHS c_LHS kappa_LHS ntao_LHS ds_LHS delta_LHS S1_LHS]
S2 = pamaters2['S']
I2 = pamaters2['I']
N2 = pamaters2['N']
I_x2= pamaters2['I_x']
I_h_x2= pamaters2['I_h_x']
c2= pamaters2['c']
beta_s2 = pamaters2['beta_s']
beta_sw2 = pamaters2['beta_sw']
beta_hw2 = pamaters2['beta_hw']
beta_h2 =pamaters2['beta_h']
R_t2 = pamaters2['R_t']
R_i2 = pamaters2['R_i']
R_b2 = pamaters2['R_b']


#  需要的数据 2010-2020合并
# R_t = np.concatenate((R_t2, R_t1.detach().numpy()))  # 207行
I_h_x = np.concatenate((c2, c1.detach().numpy()))
c = np.concatenate((c2, c1.detach().numpy()))
beta_s = np.concatenate((beta_s2, beta_s1.detach().numpy()))
beta_sw = np.concatenate((beta_sw2, beta_sw1.detach().numpy()))
beta_h = np.concatenate((beta_h2, beta_h1.detach().numpy()))
beta_hw = np.concatenate((beta_hw2, beta_hw1.detach().numpy()))
N = np.concatenate((N2, N1.detach().numpy()))
S = np.concatenate((S2, S1.detach().numpy()))
I_x = np.concatenate((I_x2, I_x1.detach().numpy()))
I = np.concatenate((I2, I1.detach().numpy()))


print("beta_s",beta_s)
print("beta_sw",beta_sw)
print("beta_h",beta_h)
print("beta_hw",beta_hw)
# 计算beta的均值和方差
print("---GS--------------mean---------------------------------std")
print("beta_s",beta_s.mean(),beta_s.std())
print("beta_sw",beta_sw.mean(),beta_sw.std())
print("beta_h",beta_h.mean(),beta_h.std())
print("beta_hw",beta_hw.mean(),beta_hw.std())

LHSmatrix = np.hstack((c.reshape(11, 1), beta_s.reshape(11,1), beta_sw.reshape(11, 1), beta_hw.reshape(11, 1),
                       beta_h.reshape(11,1),S.reshape(11, 1), N.reshape(11, 1)))
LHSmatrixdf = pd.DataFrame(LHSmatrix, columns=['c', 'beta_s', 'beta_sw','beta_hw','beta_h', 'S', 'N'])
LHSmatrixdf.to_csv(r"E:\文献阅读\AI\nnc-master\LiaoNingbrucellosis\data\pamaters_GS.csv",
              header=['c', 'beta_s', 'beta_sw', 'beta_hw', 'beta_h', 'S', 'N'])
LHSmatrixdf.to_csv(r"E:\文献阅读\AI\nnc-master\Matlab\pamaters_GS.csv",
              header=['c', 'beta_s', 'beta_sw', 'beta_hw', 'beta_h', 'S', 'N'])



I_h_x = np.concatenate((I_h_x2, I_h_x1.detach().numpy()))
T = torch.linspace(0, 10, 11)
# #################计算Rt
R_i1= beta_s1 * S1 / (N1 * (c1+d))
R_b1=kappa * beta_sw1 * S1 /(N1 * (c1+d)*(delta + ntao))
R_t1 = R_i1 + R_b1
R_t = np.concatenate((R_t2, R_t1.detach().numpy()))
R_i = np.concatenate((R_i2, R_i1.detach().numpy()))
R_b = np.concatenate((R_b2, R_b1.detach().numpy()))
for i in range(11):
    t2 = np.arange(i, i + 2, 1)
    Rt = np.expand_dims(R_t[i], 0)
    R_tt = np.concatenate((Rt, Rt))
    Ri = np.expand_dims(R_i[i], 0)
    R_ii = np.concatenate((Ri, Ri))
    Rb = np.expand_dims(R_b[i], 0)
    R_bb = np.concatenate((Rb, Rb))
    plt.plot([i, i], [0, 2.95], c='b', alpha=0.25, linestyle='--')
    plt.plot(t2, R_tt, c='#f47e62', linewidth=2.5)  # 红色
    plt.plot(t2, R_ii, c='#6f80be', linewidth=2.5)  # 蓝色
    plt.plot(t2, R_bb, c='#bd8ec0', linewidth=2.5)  # 紫色
    if R_t[i] < 1:
        plt.fill_between(t2, 0, 2.95, facecolor='#fff2df')
    else:
        plt.fill_between(t2, 0, 2.95, facecolor='#d6ecf0')

plt.plot([11, 11], [0, 2.95], c='b', alpha=0.25, linestyle='--')
plt.plot([0, 11], [1, 1], c='r', alpha=0.85, linestyle='--')
plt.xlim(0, 11)
plt.ylim(0, 2.95)
# plt.show()
time = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
x_indexes = np.arange(len(time))
plt.xticks(ticks=x_indexes, labels=time, rotation=30, fontsize=10)
plt.yticks(fontsize=10)


# font1 = {'family': 'Times New Roman', 'weight':'normal',}
plt.xlabel("Time (Year)", fontsize=13, fontproperties="Times New Roman")
plt.ylabel(r"Average effective reproduction number ($R_t$(t))", fontsize=13, fontproperties="Times New Roman")

foo_fig = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
foo_fig.savefig(os.path.join(figure_save_path, 'GS_Rt.eps'), format='eps', dpi=1000, bbox_inches='tight')
plt.show()

#################计算beta
colors_use=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#bec1d4', '#bb7784', '#0000ff', '#111010', '#FFFF00',   '#1f77b4', '#800080', '#959595',
 '#7d87b9', '#bec1d4', '#d6bcc0', '#bb7784', '#8e063b', '#4a6fe3', '#8595e1', '#b5bbe3', '#e6afb9', '#e07b91', '#d33f6a', '#11c638', '#8dd593', '#c6dec7', '#ead3c6', '#f0b98d', '#ef9708', '#0fcfc0', '#9cded6', '#d5eae7', '#f3e1eb', '#f6c4e1', '#f79cd4']
######分段，像求Rt
'''for i in range(11):
    t1 = np.arange(i, i + 2, 1)
    if i<6:
        beta_ss = np.expand_dims(beta_s2[i], 0)
        beta_ss2 = np.concatenate((beta_ss, beta_ss))
        beta_ssw = np.expand_dims(beta_sw2[i], 0)
        beta_ssw2 = np.concatenate((beta_ssw, beta_ssw))
        beta_hww = np.expand_dims(beta_hw2[i], 0)
        beta_hww2 = np.concatenate((beta_hww, beta_hww))
        beta_hh = np.expand_dims(beta_h2[i], 0)
        beta_hh2 = np.concatenate((beta_hh, beta_hh))
        plt.plot(t1, beta_ss2, alpha=0.45, linewidth=2.5, c='g')
        plt.plot(t1, beta_ssw2, alpha=0.45, linewidth=2.5, c='r')
        plt.plot(t1, beta_hh2 * 100, alpha=0.45, linewidth=2.5, c='#FFDD44')
        plt.plot(t1, beta_hww2 * 100, alpha=0.45, linewidth=2.5, c='purple')
    else:
        beta_ss = np.expand_dims(beta_s1[i-6].detach().numpy(), 0)
        beta_ss = np.concatenate((beta_ss, beta_ss))
        beta_ssw = np.expand_dims(beta_sw1[i-6].detach().numpy(), 0)
        beta_ssw = np.concatenate((beta_ssw, beta_ssw))
        beta_hww = np.expand_dims(beta_hw1[i-6].detach().numpy(), 0)
        beta_hww = np.concatenate((beta_hww, beta_hww))
        beta_hh = np.expand_dims(beta_h1[i-6].detach().numpy(), 0)
        beta_hh = np.concatenate((beta_hh, beta_hh))
        plt.plot(t1, beta_ss,alpha=0.45,linewidth=2.5,c='g')
        plt.plot(t1, beta_ssw, alpha=0.45, linewidth=2.5, c='r')
        plt.plot(t1, beta_hh * 100, alpha=0.45, linewidth=2.5, c='#FFDD44')
        plt.plot(t1, beta_hww*100, alpha=0.45, linewidth=2.5, c='purple')


    if R_t[i] < 1:
        plt.fill_between(t1, -0.05, 1.05, facecolor='#fff2df')
    else:
        plt.fill_between(t1, -0.05, 1.05, facecolor='#d6ecf0')

    plt.plot([i, i], [-0.05, 1.05], c='b', alpha=0.25, linestyle='--')
plt.plot([11, 11], [-0.05, 1.05], c='b', alpha=0.25, linestyle='--')
plt.xlim(0, 11)
plt.ylim(-0.05, 1.05)

plt.xticks(ticks=x_indexes, labels=time, rotation=30, fontsize=10)'''

t22=np.linspace(0,5,6)
t11=np.linspace(6,10,5)
plt.plot(t22, beta_s2, alpha=0.45, linewidth=2.5, c='g')
plt.plot(t22, beta_sw2, alpha=0.45, linewidth=2.5, c='r')
plt.plot(t22, beta_h2 * 100, alpha=0.45, linewidth=2.5, c='#FFDD44')
plt.plot(t22, beta_hw2 * 100, alpha=0.45, linewidth=2.5, c='purple')
plt.plot(t11, beta_s1.detach().numpy(), alpha=0.45, linewidth=2.5, c='g')
plt.plot(t11, beta_sw1.detach().numpy(), alpha=0.45, linewidth=2.5, c='r')
plt.plot(t11, beta_h1.detach().numpy() * 100, alpha=0.45, linewidth=2.5, c='#FFDD44')
plt.plot(t11, beta_hw1.detach().numpy() * 100, alpha=0.45, linewidth=2.5, c='purple')
beta_ss2 = np.expand_dims(beta_s2[5], 0)
beta_ss1 = np.expand_dims(beta_s1[0].detach().numpy(), 0)
beta_ss = np.concatenate((beta_ss2, beta_ss1))
beta_ssw = np.concatenate((np.expand_dims(beta_sw2[5], 0), np.expand_dims(beta_sw1[0].detach().numpy(), 0)))
beta_hh = np.concatenate((np.expand_dims(beta_h2[5], 0), np.expand_dims(beta_h1[0].detach().numpy(), 0)))
beta_hhw = np.concatenate((np.expand_dims(beta_hw2[5], 0), np.expand_dims(beta_hw1[0].detach().numpy(), 0)))

for i in range(11):
    tt = np.arange(i, i + 2, 1)
    plt.scatter(t22, beta_s2, marker='p', color='#66CD00', s=140)
    plt.scatter(t22, beta_sw2, marker='>', color='#e07b91', s=140)
    plt.scatter(t22, beta_h2 * 100, marker='*', color='#f47e62', s=140)
    plt.scatter(t22, beta_hw2 * 100, marker='x', color='#7d87b9', s=140)
    plt.scatter(t11, beta_s1.detach().numpy(), marker='p', color='#66CD00', s=140)
    plt.scatter(t11, beta_sw1.detach().numpy(), marker='>', color='#e07b91', s=140)
    plt.scatter(t11, beta_h1.detach().numpy() * 100, marker='*', color='#f47e62', s=140)
    plt.scatter(t11, beta_hw1.detach().numpy() * 100, marker='x', color='#7d87b9', s=140)

    if i==5:
        plt.plot(tt, beta_ss, alpha=0.45, linewidth=2.5, c='g')
        plt.plot(tt, beta_ssw, alpha=0.45, linewidth=2.5, c='r')
        plt.plot(tt, beta_hh * 100, alpha=0.45, linewidth=2.5, c='#FFDD44')
        plt.plot(tt, beta_hhw * 100, alpha=0.45, linewidth=2.5, c='purple')


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
'''plt.legend([r"$\beta_s$", r"$\beta_{sb}$", r"$\beta_h$*100", r"$\beta_{hb}$*100"],
           ncol=4, bbox_to_anchor=(0.9, -0.3), prop={'family': 'Times New Roman', 'size': 12})'''

GS_b = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
GS_b.savefig(os.path.join(figure_save_path, 'GS_betat.eps'), dpi=600, format='eps', bbox_inches="tight")

plt.show()

# ################人急性的拟合结果+短期预测
t_pre = torch.linspace(0, 5, 6)
x_nnc_pre = odeint(nnc_dyn, x0.detach(), t_pre.detach(), method='dopri5')
I_h_x_new_hat = x_nnc_pre[:, :, -1].flatten()
data_pre = torch.tensor([46, 1, 2, 3, 4, 5, 1745, 7, 8, 9, 10, 11])
# 人急性+2021年
I_h_x_new_pre = torch.tensor([
    [46], [50], [118], [499], [1404], [2308], [1745], [1631], [1584], [1787], [3004],[4562]
], dtype=torch.float)
for j in range(1, 6):
    if j< 6:
        data_pre[j] = I_h_x[j] - I_h_x[j - 1]
        data_pre[j+6] = I_h_x_new_hat[j] - I_h_x_new_hat[j- 1]
    else:
        data_pre[j] = I_h_x[j] - I_h_x[j - 1]

plt.bar(torch.linspace(0, 11, 12), I_h_x_new_pre, color='#3D9140', alpha=0.45)   # 绘制x与y的图线
plt.plot(np.linspace(10,11,2), data_pre[10:12].detach().numpy(), alpha=0.4, color='#F08080',marker='*', markersize=20, linestyle='-.', linewidth=4)
plt.plot(torch.linspace(0, 10, 11), data_pre[0:11].detach().numpy(), alpha=0.4, color='red',marker='*', markersize=20, linestyle='-.', linewidth=4)
plt.xlim(-0.5, 11.5)
x_indexes = np.arange(len(time))
plt.xticks(ticks=x_indexes, labels=time, rotation=30, fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Time (Year)", fontsize=14, fontproperties="Times New Roman")
plt.ylabel("Number of newly infected population", fontsize=14, fontproperties="Times New Roman")

fig_gh = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
fig_gh.savefig(os.path.join(figure_save_path, 'GS_InfectedHuman_nihe.eps'), dpi=1000, format='eps', bbox_inches="tight")
plt.show()

# #####################PRCC
# 在matlab中运行PRCC
prcc1 = pd.read_csv(r'E:\文献阅读\AI\nnc-master\Matlab\PRCC_GS.csv')
# LHSmatrix = [beta_s_LHS beta_sb_LHS c_LHS kappa_LHS ntao_LHS ds_LHS delta_LHS S1_LHS]
prcc = prcc1.to_csv("PRCC_GS2.csv", header=['beta_s', 'beta_sb', 'c', 'kappa', 'ntao', 'd_s', 'delta', 'S1'], index=0)
prcc = pd.read_csv(r'PRCC_GS2.csv')
# df = pd.read_csv("MappingAnalysis_Data.csv")

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
markers = ['o','s','H','D','^','<','d']
labels = [r"$\beta_s$", r"$\beta_sb$", "c", r"$\kappa$", r"n$\tau$", r"$d_s$", r"$\delta$"]

plt.plot(T, prcc.iloc[:, [0]], marker=markers[0], markerfacecolor=colors_use[0], markersize=8, markeredgewidth=0.5,
         color="k", linewidth=0.9, linestyle="-", label=labels[0])
plt.plot(T, prcc.iloc[:, [1]], marker=markers[1], markerfacecolor=colors_use[1], markersize=8, markeredgewidth=0.5,
         color="k", linewidth=0.9, linestyle="-", label=labels[1])
plt.plot(T, prcc.iloc[:, [2]], marker=markers[2], markerfacecolor=colors_use[2], markersize=8, markeredgewidth=0.5,
         color="k", linewidth=0.9, linestyle=":", label=labels[2])
plt.plot(T, prcc.iloc[:, [3]], marker=markers[3], markerfacecolor=colors_use[3], markersize=8, markeredgewidth=0.5,
         color="k", linewidth=0.9, linestyle="-.", label=labels[3])
plt.plot(T, prcc.iloc[:, [4]], marker=markers[4], markerfacecolor=colors_use[4], markersize=8, markeredgewidth=0.5,
         color="k", linewidth=0.9, linestyle='solid', label=labels[4])
plt.plot(T, prcc.iloc[:, [5]], marker=markers[5], markerfacecolor=colors_use[8], markersize=8, markeredgewidth=0.5,
         color="k", linewidth=0.9, linestyle='dashdot', label=labels[5])
plt.plot(T, prcc.iloc[:, [6]], marker=markers[6], markerfacecolor=colors_use[7], markersize=8, markeredgewidth=0.5,
         color="k", linewidth=0.9, linestyle='dotted', label=labels[6])
plt.plot([0,11], [0, 0], c='b', alpha=0.25,linestyle='--')
plt.fill_between([-0.15,11],0.2,0.4, facecolor='#C0C0C0',alpha=0.7)
plt.fill_between([-0.15,11],-0.2,-0.4, facecolor='#C0C0C0',alpha=0.7)
plt.fill_between([-0.15,11],0.4,1.05, facecolor='#808A87',alpha=0.7)
plt.fill_between([-0.15,11],-0.4,-1.05, facecolor='#808A87',alpha=0.7)
plt.xlim(-0.15, 10.5)
plt.ylim(-1.05, 1.05)
x_indexes = np.arange(len(time)-1)
plt.xticks(ticks=x_indexes, labels=time[0:11], rotation=30, fontsize=10)
plt.yticks(np.linspace(-1, 1, 11, endpoint=True), fontsize=10)
plt.xlabel("Time (Year)", fontsize=14, fontproperties="Times New Roman")
plt.ylabel(r"PRCC for $R_t$(t)", fontsize=14, fontproperties="Times New Roman")
# 图例位置，大小
# labels = [r"$\beta_s$", r"$\beta_sb$", "c", r"$\kappa$", r"n$\tau$",r"$d_s$",r"$\delta$"]
# plt.legend([r"$\beta_s$", r"$\beta_{sb}$", "c", r"$\kappa$", r"n$\tau$", r"$d_s$",r"$\delta$"], ncol=4,
#            bbox_to_anchor=(0.95, -0.25), prop={'family': 'Times New Roman', 'size': 12})

GS_p = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
GS_p.savefig(os.path.join(figure_save_path, 'GS_PRCC.eps'), dpi=1000, format='eps', bbox_inches="tight")
plt.show()

# #####################扑杀
PuSha = c*I
t=np.linspace(0,10,11)

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
markers = ['o','s','H','D','^','<','d']
labels = [r"$\beta_s$", r"$\beta_sb$", "c", r"$\kappa$", r"n$\tau$",r"$d_s$",r"$\delta$"]

plt.plot(T, PuSha, marker=markers[6], markerfacecolor='r', markersize=7, markeredgewidth=0.5,
             color="k", linewidth=0.9, linestyle='dotted')
plt.xlim(-0.15, 10.5)
# x_indexes = np.arange(len(time))
plt.xticks(ticks=x_indexes, labels=time[0:11], rotation=30, fontsize=10)
plt.xlabel("Time (Year)", fontsize=14, fontproperties="Times New Roman")
plt.ylabel("The number of sheep killed", fontsize=14, fontproperties="Times New Roman")

fig_psG = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_psG.savefig(os.path.join(figure_save_path, 'GS_kill.eps'), dpi=1000, format='eps', bbox_inches="tight")
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
knee_info = knee_test(t,c)
print("knee_info",knee_info)
print(c)

#fig, axe = plt.subplots(figsize=[8, 6])
plt.plot(t, c, marker=markers[0], markerfacecolor='k', markersize=7, markeredgewidth=0.5,
             color="k", linewidth=0.9, linestyle='dotted')
plt.plot([2, 2], [0, 1], c='g', alpha=0.25, linestyle='--')
plt.plot([6, 6], [0, 1], c='b', alpha=0.25, linestyle='--')
plt.xlim(-0.15, 10.5)
plt.ylim(0, 1)
plt.xticks(ticks=x_indexes, labels=time[0:11], rotation=30, fontsize=10)
plt.xlabel("Time (Year)", fontsize=14, fontproperties="Times New Roman")
plt.ylabel("The culling rate c(t)", fontsize=14, fontproperties="Times New Roman")

for point in knee_info:
    plt.scatter(x=point[0], y=point[1], c='r', s=200, marker='o')
    plt.annotate(s=f'{point[2]} {point[3]}', xy=(point[0]+0.3, point[1]+0.05), fontsize=12)
#plt.show()


fig_pscG = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_pscG.savefig(os.path.join(figure_save_path, 'GS_killc.eps'), dpi=1000, format='eps', bbox_inches="tight")
plt.show()


# #####################传播风险因素，用免疫数、扑杀数计算的结果
V_x = x_nnc[:, :, -3].flatten()   # 羊免疫数累计
# I_x = x_nnc[:, :, -2].flatten()   # 羊阳性数累计
# PuSha1 = c1.detach().numpy()*I1.detach().numpy()
I_x_new = torch.tensor([29740,1,2,3,4,5,17.65e4, 1, 2, 3, 4])
V_x_new = torch.tensor([0,0,0,0,0,0,5752640, 1, 2, 3, 4 ])
for j in range(1, 11):
    if i==6:
        I_x_new[j] = I_x[j]
    else:
        I_x_new[j] = I_x[j] - I_x[j - 1]


for j in range(1, 5):
    V_x_new[j+6] = V_x[j] - V_x[j - 1]

I_h_x_new = torch.tensor([
    [46], [50], [118], [499], [1404], [2308], [1745], [1631], [1584], [1787], [3004]
], dtype=torch.float)
risk = np.hstack((I_h_x_new.reshape(11, 1), I_x_new.detach().numpy().reshape(11, 1),
                    V_x_new.detach().numpy().reshape(11,1),PuSha.reshape(11,1)))
riskdf = pd.DataFrame(risk,columns=['Human','Positive','Immunization','Culling'])


# co = risk.corr()#皮尔逊相关系数
risk_sp = riskdf.corr(method='spearman')#斯皮尔曼相关系数
print(risk_sp)

vals = ['Human','Positive','Immunization','Culling']
risk_sp.insert(loc=0, column='vals', value=vals)
df_melt=pd.melt(risk_sp, id_vars=risk_sp.iloc[:,[0]],value_name='value',var_name='vals2')
df_melt = df_melt.drop([4,8,9,12,13,14])

df_melt.plot.scatter(x='vals', y='vals2', c='value', s=df_melt['value'].abs() * 2600, colormap='Oranges',colorbar=False)
plt.yticks(rotation=90, fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel("Control Measures", fontsize=14, fontproperties="Times New Roman")
plt.ylabel("Control Measures", fontsize=14, fontproperties="Times New Roman")
plt.xlim(-0.36, 3.36)
plt.ylim(-0.36, 3.36)



'''# #####################传播风险因素  热图, 用免疫率、扑杀率计算的结果
I_h_x_new = torch.tensor([
    [46], [50], [118], [499], [1404], [2308], [1745], [1631], [1584], [1787], [3004]
], dtype=torch.float)
th = torch.tensor([0,0,0,0,0,0])
theta=torch.cat([th,theta1],dim=0) # 在dim=0 处拼接
risk = np.hstack((I_h_x_new.reshape(11, 1), beta_s.reshape(11, 1),beta_sw.reshape(11, 1),
                  beta_h.reshape(11, 1),beta_hw.reshape(11, 1),
                  theta.detach().numpy().reshape(11,1),c.reshape(11,1)))

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
'''
fig_gr = plt.gcf()
figure_save_path = "picture"
# plt调用gcf函数取得当前绘制的figure并调用savefig函数
# 'get current figure'

if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_gr.savefig(os.path.join(figure_save_path, 'GS_risk.eps'), dpi=1000, format='eps', bbox_inches="tight")
plt.show()
