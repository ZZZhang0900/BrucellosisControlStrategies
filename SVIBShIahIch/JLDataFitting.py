#  吉林2016-2020年的拟合结果，画图部分增加了2010-2015GSDataFitting的数据
import numpy as np
import pandas as pd
import torch
import os
from copy import deepcopy
import LiaoNingbrucellosis.JLModelling as model
import torch
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
iter = 800   #训练次数
ii=50  #循环次数整除后画图
parameters = torch.tensor([
     358.9e4*1.1, 1.17, 0.385, 2559.6e4*4.7*0.001, 8.08*0.001, 0.33, 15, 3.2, 0.3*12
])
#       A       d_s   qgamma     B       d_h      lamda  kappa delta ntao

x0 = torch.tensor([[
    398.8e4, 49.2e2, 93.7e4*0.8, 5e3, 2559.6e4, 2905, 2905*0.385, 93.7e4*0.8, 49.2e2, 2905
]])

##人急性
I_h_x_new = torch.tensor([
    [2905],[2063],[2031],[2024],[1809],[1625],[1480],[1159],[1206],[1191],[1151]
],dtype=torch.float)     #每日新增
##急性累计
I_h_x_cul = torch.tensor([
    [2905], [4968],[6999],[9023],[10832],[12457],[13937],[15096],[16302],[17493],[18644]
],dtype=torch.float)

##免疫
V_s_cul = torch.tensor([
    [93.7e4], [220.5e4], [366.6e4], [510.5e4],[662.9e4],[838.6e4],[1030.9e4],[1197.8e4],[1353.6e4],[1533.1e4],[1760.9e4]
],dtype=torch.float)

#羊的阳性数
# 羊的阳性数
I_s_cul = torch.tensor([
    [49.2e2], [114.4e2], [205e2], [273.9e2], [330.6e2], [376.8e2], [405.8e2], [417.8e2], [439.5e2], [464.9e2], [527.7e2]
], dtype=torch.float)

'''I_s_cul = torch.tensor([
    [30.1e2],[59.1e2],[149.7e2],[218.6e2],[275.3e2],[321.5e2],[350.4e2],[362.4e2],[384.2e2],[409.6e2],[472.3e2]
],dtype=torch.float)
'''
T = 10
n_timesteps = 11
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
        V_x_cul_X = x_nnc[:, :, -3]
        I_s_cul_X = x_nnc[:, :, -2]
        I_h_x_cul_X = x_nnc[:,:,-1] # reached state at T

        loss1 = ((torch.log(I_h_x_cul_X) - torch.log(I_h_x_cul)) ** 2).sum()
        loss2 = ((torch.log(I_s_cul_X) - torch.log(I_s_cul)) ** 2).sum()
        loss3 = ((torch.log(V_x_cul_X) - torch.log(V_s_cul * 0.8)) ** 2).sum()
        loss = loss1 + loss2+ loss3

        print('i=  ,loss0-Ih-Is = ', i, loss, loss1, loss2)
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

        if i % ii == 0 or i==799:
            print("nncdata拟合变量", x_nnc)

            parameters.append(deepcopy(dict(nnc_dyn.nnc.named_parameters())))
            trajectories.append(trajectory)
            print("nncdata拟合系数", trajectory[:, :, -6:])

            data = x_nnc[:, :, -1].squeeze(-1)
            data_pre = torch.tensor([2905, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3000])
            for j in range(1, 11):
                data_pre[j] = data[j] - data[j - 1]
            plt.plot(t, I_h_x_new)  # 绘制x与y的图线
            plt.plot(t, data_pre)
            plt.text(0, 1000, loss, size=15, alpha=0.5)
            plt.title('JiLin,I_h_x_new,acute newly，i=%d'%i)
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
    Parameters = train(nnc_dyn, iter, lr, t,n_timesteps)  # , 100 epochs, learning rate 0.01
    torch.save(neural_net, 'JiLin,trained_epochs=%d.torch' %iter)

else:
    neural_net = torch.load('JiLin,trained_epochs=800.torch')
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
N = (x_nnc[:, :, 0]+x_nnc[:, :, 1]+x_nnc[:, :, 2]).flatten()
I = x_nnc[:, :, 1].flatten()
I_h_x = x_nnc[:, :, -2].flatten()  # 急性人数

XiShu = [1, 1, 1, 1, 1e-1, 1e-1]
c = (nncdata[:, :, -6]*XiShu[0]).flatten()
theta = (nncdata[:, :, -5]*XiShu[1]).flatten()
beta_s = (nncdata[:, :, -4]*XiShu[2]).flatten()
beta_sw = (nncdata[:, :, -3]*XiShu[3]).flatten()
beta_h = (nncdata[:, :, -2]*XiShu[4]).flatten()
beta_hw = (nncdata[:, :, -1]*XiShu[5]).flatten()
print("theta =",theta)
print("beta_s",beta_s)
print("beta_sw",beta_sw)
print("beta_h",beta_h)
print("beta_hw",beta_hw)

LHSmatrix = np.hstack((c.detach().numpy().reshape(11, 1), theta.detach().numpy().reshape(11,1),beta_s.detach().numpy().reshape(11,1),
                       beta_sw.detach().numpy().reshape(11, 1), beta_hw.detach().numpy().reshape(11,1),beta_h.detach().numpy().reshape(11,1),
                       S.detach().numpy().reshape(11, 1), N.detach().numpy().reshape(11,1)))
LHSmatrixdf = pd.DataFrame(LHSmatrix,columns=['c', 'theta', 'beta_s', 'beta_sw','beta_hw','beta_h', 'S', 'N'])
LHSmatrixdf.to_csv(r"E:\文献阅读\AI\nnc-master\LiaoNingbrucellosis\data\pamaters_JL.csv",
              header=['c', 'theta', 'beta_s', 'beta_sw', 'beta_hw', 'beta_h', 'S', 'N'])
LHSmatrixdf.to_csv(r"E:\文献阅读\AI\nnc-master\Matlab\pamaters_JL.csv",
              header=['c', 'theta', 'beta_s', 'beta_sw', 'beta_hw', 'beta_h', 'S', 'N'])


colors_use = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#bec1d4', '#bb7784', '#0000ff', '#111010', '#FFFF00',
              '#1f77b4', '#800080', '#959595', '#7d87b9', '#bec1d4', '#d6bcc0', '#bb7784', '#8e063b', '#4a6fe3',
              '#8595e1', '#b5bbe3', '#e6afb9', '#e07b91', '#d33f6a', '#11c638', '#8dd593', '#c6dec7', '#ead3c6',
              '#f0b98d', '#ef9708', '#0fcfc0', '#9cded6', '#d5eae7', '#f3e1eb', '#f6c4e1', '#f79cd4']
# ################计算Rt
R_i = beta_s * S / (N * (c+d))
R_b = kappa * beta_sw * S /(N * (c+d)*(delta + ntao))
R_t = R_i+R_b
print("R_t",R_t)

for i in range(11):
    t2 = np.arange(i, i + 2, 1)
    Rt = np.expand_dims(R_t[i].detach().numpy(), 0)
    R_tt = np.concatenate((Rt, Rt))
    Ri = np.expand_dims(R_i[i].detach().numpy(), 0)
    R_ii = np.concatenate((Ri, Ri))
    Rb = np.expand_dims(R_b[i].detach().numpy(), 0)
    R_bb = np.concatenate((Rb, Rb))
    plt.plot([i, i], [0, 1.34], c='b', alpha=0.25, linestyle='--')
    plt.plot(t2, R_tt, c='#f47e62', linewidth=2.5)   # 红色
    plt.plot(t2, R_ii, c='#6f80be', linewidth=2.5)    #蓝色
    plt.plot(t2, R_bb, c='#bd8ec0', linewidth=2.5)    # 紫色
    if R_t[i] < 1:
        plt.fill_between(t2, 0, 1.34, facecolor='#fff2df')
    else:
        plt.fill_between(t2, 0, 1.34, facecolor='#d6ecf0')

plt.plot([11, 11], [0, 1.34], c='b', alpha=0.25, linestyle='--')
plt.plot([0, 11], [1, 1], c='r', alpha=0.85, linestyle='--')
plt.xlim(0, 11)
plt.ylim(0, 1.34)
# plt.show()
time = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
x_indexes = np.arange(len(time))
plt.xticks(ticks=x_indexes, labels=time, rotation=30, fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Time (Year)", fontsize=13, fontproperties="Times New Roman")
plt.ylabel(r"Average effective reproduction number ($R_t$(t))", fontsize=13, fontproperties="Times New Roman")


foo_fig = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
foo_fig.savefig(os.path.join(figure_save_path, 'JL_Rt.eps'), dpi=600, format='eps', bbox_inches="tight")
plt.show()

# ################计算beta
######分段，像求Rt
'''for i in range(11):
    tt = np.arange(i, i + 2, 1)
    beta_ss = np.expand_dims(beta_s[i].detach().numpy(), 0)
    beta_ss = np.concatenate((beta_ss, beta_ss))
    beta_ssw = np.expand_dims(beta_sw[i].detach().numpy(), 0)
    beta_ssw = np.concatenate((beta_ssw, beta_ssw))
    beta_hh = np.expand_dims(beta_h[i].detach().numpy(), 0)
    beta_hh = np.concatenate((beta_hh, beta_hh))
    beta_hww = np.expand_dims(beta_hw[i].detach().numpy(), 0)
    beta_hww = np.concatenate((beta_hww, beta_hww))

    plt.plot(tt, beta_ss, alpha=0.45, linewidth=2.5, c='g')
    plt.plot(tt, beta_ssw, alpha=0.45, linewidth=2.5, c='r')
    plt.plot(tt, beta_hh, alpha=0.45, linewidth=2.5, c='#FFDD44')
    plt.plot(tt, beta_hww, alpha=0.45, linewidth=2.5, c='purple')

    if R_t[i] < 1:
        plt.fill_between(tt, -0.05, 1.05, facecolor='#fff2df')
    else:
        plt.fill_between(tt, -0.05, 1.05, facecolor='#d6ecf0')

    plt.plot([i, i], [-0.05, 1.05], c='b', alpha=0.25, linestyle='--')
plt.plot([11, 11], [-0.05, 1.05], c='b', alpha=0.25, linestyle='--')
plt.xlim(0, 11)
plt.ylim(-0.05, 1.05)
plt.xticks(ticks=x_indexes, labels=time, rotation=30, fontsize=10)'''

# 计算beta的均值和方差
print("---JL--------------mean---------------------------------std")
print("beta_s",beta_s.mean(),beta_s.std())
print("beta_sw",beta_sw.mean(),beta_sw.std())
print("beta_h",beta_h.mean(),beta_h.std())
print("beta_hw",beta_hw.mean(),beta_hw.std())

plt.plot(t, beta_s.detach().numpy(), alpha=0.45, linewidth=2.5, c='g')
plt.plot(t, beta_sw.detach().numpy(), alpha=0.45, linewidth=2.5, c='r')
plt.plot(t, beta_h.detach().numpy(), alpha=0.45, linewidth=2.5, c='#FFDD44')
plt.plot(t, beta_hw.detach().numpy(), alpha=0.45, linewidth=2.5, c='purple')
for i in range(11):
    tt = np.arange(i, i + 2, 1)

    plt.scatter(t, beta_s.detach().numpy(), marker='p', color='#66CD00', s=140)
    plt.scatter(t, beta_sw.detach().numpy(), marker='>', color='#e07b91', s=140)
    plt.scatter(t, beta_h.detach().numpy(), marker='*', color='#f47e62', s=140)
    plt.scatter(t, beta_hw.detach().numpy(), marker='x', color='#7d87b9', s=140)

    if R_t[i] < 1:
        plt.fill_between(tt, 0,1, facecolor='#fff2df')
    else:
        plt.fill_between(tt, 0,1, facecolor='#d6ecf0')

plt.xlim(0, 10.1)
plt.ylim(0,1)
plt.xticks(ticks=x_indexes[0:11], labels=time[0:11], rotation=30, fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Time (Year)", fontsize=14, fontproperties="Times New Roman")
plt.ylabel(r"Transmission rate ($\beta$(t))", fontsize=14, fontproperties="Times New Roman")
# 图例位置，大小
plt.legend([r"$\beta_s$", r"$\beta_{sb}$", r"$\beta_h$", r"$\beta_{hb}$"],
           ncol=4, bbox_to_anchor=(0.9, -0.3), prop={'family': 'Times New Roman', 'size': 12})

fig_b = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_b.savefig(os.path.join(figure_save_path, 'JL_betat.eps'), dpi=600, format='eps', bbox_inches="tight")
plt.show()


# ################人急性的拟合结果+短期预测
t_pre =  torch.linspace(0, 11, 12)
x_nnc_pre = odeint(nnc_dyn, x0.detach(), t_pre.detach(), method='dopri5')
I_h_x_new_hat = x_nnc_pre[:, :, -1].flatten()
data_pre = torch.tensor([2905, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
# 人急性+2021年
I_h_x_new_pre = torch.tensor([
    [2905],[2063],[2031],[2024],[1809],[1625],[1480],[1159],[1206],[1191],[1151],[1265]
],dtype=torch.float)

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

fig_h = plt.gcf()
figure_save_path = "picture"
# plt调用gcf函数取得当前绘制的figure并调用savefig函数
# 'get current figure'

if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_h.savefig(os.path.join(figure_save_path, 'JL_InfectedHuman_nihe.eps'), dpi=1000, format='eps', bbox_inches="tight")
plt.show()
# #####################PRCC
prcc = pd.read_csv(r'E:\文献阅读\AI\nnc-master\Matlab\PRCC_JL.csv')
# LHSmatrix = [beta_s_LHS beta_sb_LHS c_LHS kappa_LHS ntao_LHS ds_LHS delta_LHS S1_LHS]
prcc = prcc.to_csv("PRCC_JL2.csv", header=['beta_s', 'beta_sb', 'c', 'kappa', 'ntao', 'd_s', 'delta', 'S1'], index=0)
prcc = pd.read_csv(r'PRCC_JL2.csv')
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
plt.fill_between([-0.15,11],0.4,1.05, facecolor='#808A87',alpha=0.5)
plt.fill_between([-0.15,11],-0.4,-1.05, facecolor='#808A87',alpha=0.5)
plt.xlim(-0.15, 10.5)
plt.ylim(-1.05, 1.05)
x_indexes = np.arange(len(time)-1)
plt.xticks(ticks=x_indexes, labels=time[0:11], rotation=30, fontsize=10)
plt.yticks(np.linspace(-1, 1, 11, endpoint=True), fontsize=10)
plt.xlabel("Time (Year)", fontsize=14, fontproperties="Times New Roman")
plt.ylabel("PRCC for $R_t$(t)", fontsize=14, fontproperties="Times New Roman")
# 图例位置，大小
# labels = [r"$\beta_s$", r"$\beta_sb$", "c", r"$\kappa$", r"n$\tau$",r"$d_s$",r"$\delta$"]
# plt.legend([r"$\beta_s$", r"$\beta_{sb}$", "c", r"$\kappa$", r"n$\tau$",r"$d_s$",r"$\delta$"], ncol=4,
#            bbox_to_anchor=(-0.2, -0.25), prop={'family': 'Times New Roman', 'size': 12})

fig_pp = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_pp.savefig(os.path.join(figure_save_path, 'JL_PRCC.eps'), dpi=1000, format='eps', bbox_inches="tight")
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
fig_psN.savefig(os.path.join(figure_save_path, 'JL_kill.eps'), dpi=1000, format='eps', bbox_inches="tight")
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
print("c=",c)

#fig, axe = plt.subplots(figsize=[8, 6])
plt.plot(t, c.detach().numpy(), marker=markers[0], markerfacecolor='k', markersize=7, markeredgewidth=0.5,
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
    if point[0]==2.0:
        plt.annotate(s=f'{point[2]} {point[3]}', xy=(point[0]-0.3, point[1]+0.15), fontsize=12)
    else:
        plt.annotate(s=f'{point[2]} {point[3]}', xy=(point[0] + 0.3, point[1] + 0.05), fontsize=12)
#plt.show()


fig_pscG = plt.gcf()
figure_save_path = "picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_pscG.savefig(os.path.join(figure_save_path, 'JL_killc.eps'), dpi=1000, format='eps', bbox_inches="tight")
plt.show()




# #####################传播风险因素，用免疫数、扑杀数计算的结果
I_x = x_nnc[:, :, -2].flatten()   # 羊阳性数累计
V_x = x_nnc[:, :, -3].flatten()   # 羊免疫数累计

I_x_new = torch.tensor([4920, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
V_x_new = torch.tensor([749600, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

for j in range(1, 11):
    I_x_new[j] = I_x[j] - I_x[j - 1]
    V_x_new[j] = V_x[j] - V_x[j - 1]

risk = np.hstack((I_h_x_new.reshape(11, 1), I_x_new.detach().numpy().reshape(11, 1),
                  V_x_new.detach().numpy().reshape(11,1),PuSha.detach().numpy().reshape(11,1),
                       ))
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
plt.ylim(-0.57, 6.57)'''


fig_jr = plt.gcf()
figure_save_path = "picture"
# plt调用gcf函数取得当前绘制的figure并调用savefig函数
# 'get current figure'

if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
fig_jr.savefig(os.path.join(figure_save_path, 'JL_risk.eps'), dpi=1000, format='eps', bbox_inches="tight")
plt.show()

