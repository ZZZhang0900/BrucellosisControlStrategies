import sys
sys.path.append('../../')

# Custom modules path
from nnc.helpers.torch_utils.expm.expm import expm as torch_expm

import nnc.controllers.baselines.ct_lti.dynamics as dynamics
#import nnc.controllers.baselines.ct_lti.optimal_controllers as oc
from nnc.controllers.neural_network.nnc_controllers import NeuralNetworkController, NNCDynamics

# Computation and plot helpers for this example
from examples.directed.small_example_helpers import EluTimeControl, evaluate_trajectory, todf
from examples.directed.small_example_helpers import compare_trajectories, grand_animation

# Other libraries for computing
import torch
import numpy as np
import pandas as pd
from torchdiffeq import odeint

# progress bar
from tqdm.notebook import tqdm
###############################################方程初始化
# Basic setup for calculations, graph, number of nodes, etc.
dtype = torch.float32
#device = 'gpu'
training_session = True

# interaction matrix
A = torch.tensor([
    [0., 0., 1.],
    [1., 0., 0.],
    [0., 1., 0.]
])

# driver matrix
B = torch.tensor([
    [1.],
    [0.],
    [0.]
])

#interaction matrix dimensions denote how many nodes we have in the network
n_nodes = A.shape[-1]
# column dimension implies the number of driver nodes.
n_drivers = B.shape[-1]

# implementing the dynamics
linear_dynamics = dynamics.ContinuousTimeInvariantDynamics(A, B, dtype,)
print(linear_dynamics)
###############################################终止时间，初始状态，目标状态
T = 1
t = torch.linspace(0, T, 2)    #tensor([0,1])
x0 = torch.tensor([[
        1., 0., 0.

]])

# same applies for target state:
x_target = torch.tensor([[
    0., 0., 1.
]])


timesteps = torch.linspace(0, T, 2)

################################################神经网络
from copy import deepcopy


def train(nnc_dyn, epochs, lr, t, n_timesteps=40):  # simple training
    optimizer = torch.optim.Adam(nnc_dyn.parameters(), lr=lr)
    trajectories = [evaluate_trajectory(linear_dynamics,
                                        nnc_dyn.nnc,
                                        x0,
                                        T,
                                        n_timesteps,
                                        method='rk4',
                                        options=dict(step_size=T / n_timesteps)
                                        )]  # keep trajectories for plots
    parameters = [deepcopy(dict(nnc_dyn.nnc.named_parameters()))]  # dict字典，以键值对保存；deepcopy深度复制

    for i in tqdm(range(epochs)):
        optimizer.zero_grad()  # do not accumulate gradients over epochs

        x = x0.detach()
        x_nnc = odeint(nnc_dyn, x, t.detach(), method='dopri5')
        x_T = x_nnc[-1, :]  # reached state at T

        loss = ((x_target - x_T) ** 2).sum()  # !No energy regularization

        loss.backward()
        optimizer.step()

        trajectory = evaluate_trajectory(linear_dynamics,
                                         nnc_dyn.nnc,
                                         x0,
                                         T,
                                         n_timesteps,
                                         method='rk4',
                                         options=dict(step_size=T / n_timesteps)
                                         )

        if i % 1000 == 0:
            parameters.append(deepcopy(dict(nnc_dyn.nnc.named_parameters())))
            trajectories.append(trajectory)
    return torch.stack(trajectories).squeeze(-2), parameters


################

n_timesteps = 60  # relevant for plotting, ode solver is variable step
linear_dynamics = dynamics.ContinuousTimeInvariantDynamics(A, B, dtype,)

if training_session:
    torch.manual_seed(0)
    neural_net = EluTimeControl(n_nodes, n_drivers)
    nnc_model = NeuralNetworkController(neural_net)
    nnc_dyn = NNCDynamics(linear_dynamics, nnc_model)
    # time to train now:
    t = torch.linspace(0, T, n_timesteps)
    # t1 = train(nnc_dyn, 10, 0.1, t) # , 200 epochs, learning rate 0.1
    # df1 = todf(t1, lr=0.1)
    t2, parameters = train(nnc_dyn, 200, 0.0006, t)  # , 100 epochs, learning rate 0.01
    df2 = todf(t2, lr=0.0025)
    torch.save(neural_net, 'trained_elu_net_directed.torch')
    alldf = pd.concat([df2], ignore_index=True)

    alldf.to_csv('all_trajectories_directed.csv')
else:
    neural_net = torch.load('trained_elu_net_directed.torch')
    alldf = pd.read_csv('all_trajectories_directed.csv', index_col=0)
    nnc_model = NeuralNetworkController(neural_net)
    nnc_dyn = NNCDynamics(linear_dynamics, nnc_model)

########################3%%

t = torch.linspace(0, T, 2)
x = x0.detach()
ld_controlled_lambda = lambda t, x_in: linear_dynamics(t, u=neural_net(t, x_in), x=x_in)
x_all_nn = odeint(ld_controlled_lambda, x0, t, method='rk4',
                  options = dict(step_size=T/n_timesteps))
x_T = x_all_nn[-1, :]
print(str(x_T.flatten().detach().cpu().numpy()))

#%################################%

'''mport matplotlib.pyplot as plt


#np.savetxt("oc_trajectory_directed5.csv", np.c_[oc_trajectory])

trajectory = evaluate_trajectory(linear_dynamics,
                                 nnc_model,
                                 x0,
                                 T,
                                 n_timesteps,
                                 method='rk4',
                                 options=dict(step_size=T / n_timesteps)
                                 )
nnc_trajectory = trajectory.squeeze(1).unsqueeze(0).detach().numpy()[0]

#np.savetxt("nnc_trajectory_directed5.csv", np.c_[nnc_trajectory])

nnc_u = np.array([nnc_trajectory[i,3] for i in range(len(nnc_trajectory))])

energy_nnc = np.cumsum((nnc_u**2)*T/n_timesteps)
time = np.linspace(0,T,n_timesteps)

#np.savetxt("energies_directed5.csv", np.c_[time, energy_nnc, energy_oc])

plt.figure()
#plt.plot(time,energy_oc)
#plt.plot(time,energy_nnc)
plt.plot(time,nnc_u)
plt.xlabel(r"t")
plt.ylabel(r"u(t)")
plt.show()'''

#############################################cccccc

# %%

def parameter_norm_difference(n_prev, n_now):
    assert n_prev.keys() == n_now.keys()
    norm = 0

    for (k, v) in n_prev.items():
        # print(k)
        norm += ((v.detach() - n_now[k].detach()) ** 2).sum()
    # k = 'neural_net.neural_net.linear_final.weight'
    # norm = ((n_prev[k].detach() - n_now[k].detach())**2).sum()
    return (norm).cpu().detach().item()


def parameter_square_l2(param):
    norm = 0

    for (k, v) in param.items():
        norm += ((v.detach()) ** 2).sum()
    # k = 'neural_net.neural_net.linear_final.weight'
    # norm = ((n_prev[k].detach() - n_now[k].detach())**2).sum()
    return (norm).cpu().detach().item()


# %%

w_diff = [parameter_norm_difference(parameters[i], parameters[i + 1])
          for i in range(len(parameters) - 1)]

w_l2 = [parameter_square_l2(param) for param in parameters]

u_diff = []
en_all = []
n_points = 40
for sid in alldf['sample_id'].unique():
    if sid < alldf['sample_id'].max():
        dat_0 = alldf[alldf['sample_id'] == sid].sort_values('time')
        dat_1 = alldf[alldf['sample_id'] == sid + 1].sort_values('time')

        diff = ((dat_0['u'].values - dat_1['u'].values) ** 2).sum()
        u_diff.append(diff)
        en_all.append((dat_0['u'].values ** 2 * T / n_timesteps).sum())

print(len(w_diff))
w_diff = np.array(w_diff)
u_diff = np.array(u_diff)

np.savetxt("norm_diffs_directed.csv", np.c_[w_diff, u_diff])

# w_diff[w_diff > np.quantile(w_diff, 0.99)] = np.nan
# u_diff[u_diff > np.quantile(u_diff, 0.99)] = np.nan
# w_diff = w_diff/w_diff.max()
# u_diff = u_diff/u_diff.max()
dfff = pd.DataFrame(dict(u_d=u_diff, w_d=w_diff))
a = dfff.dropna()
a['u_d'] = (a['u_d'] - a['u_d'].min()) / (a['u_d'].max() - a['u_d'].min())
a['w_d'] = (a['w_d'] - a['w_d'].min()) / (a['w_d'].max() - a['w_d'].min())
from nnc.helpers.plot_helper import trendplot
from examples.directed.small_example_helpers import base_temp, axis_temp

'''fig_c = trendplot(a['u_d'].tolist(), a['w_d'].tolist(),
                  r'$||\Delta u||_2^2$', r'$||\Delta w||_2^2$')

fig_c.layout.annotations[0].x = 0.6
fig_c.layout.annotations[0].y = -0.1
fig_c.layout.annotations[0].xref = 'x'
fig_c.layout.annotations[0].yref = 'y'
fig_c.layout.annotations[0].align = "right"

fig_c.layout.width = 170
fig_c.layout.height = 150
fig_c.layout.margin.l = 35
fig_c.layout.margin.t = 0
fig_c.layout.margin.b = 25
fig_c.layout.margin.r = 0
fig_c.layout.font.size = 10
fig_c.update_layout(base_temp)
fig_c.update_xaxes(axis_temp)
fig_c.update_yaxes(axis_temp)
# fig_c.write_image('corr_plot_w_u.png')
fig_c
'''
# %%

wl2 = np.array(w_l2)
ul2 = np.array(en_all)

np.savetxt("norm_evolution_directed.csv", np.c_[wl2[:-1], ul2])



# %%

n_points

# %%

import plotly.express as px
from plotly.subplots import make_subplots

wl2dat = px.line(y=ul2).data[0]
wl2dat.name = r'$||w||_2^2$'
wl2dat.showlegend = True

wl2dat.yaxis = 'y2'
wl2dat.line.color = 'orange'

ul2dat = px.line(y=wl2).data[0]
ul2dat.name = r'$||u||_2^2$'
ul2dat.showlegend = True

# %%

from plotly import graph_objects as go

fig_d = make_subplots(specs=[[{"secondary_y": True}]])
fig_d.add_trace(ul2dat)
fig_d.add_trace(wl2dat, secondary_y=True)
fig_d.update_layout(base_temp)
fig_d.update_xaxes(axis_temp)
fig_d.update_yaxes(axis_temp)
fig_d.layout.xaxis.title = r'$\tau$'
fig_d.layout.yaxis1.title = r'$||u||_2^2$'
fig_d.layout.yaxis2.title = r'$||w||_2^2$'

fig_d.layout.width = 195
fig_d.layout.height = 150
fig_d.layout.margin.l = 35
fig_d.layout.margin.t = 0
fig_d.layout.margin.b = 0
fig_d.layout.margin.r = 35

fig_d.update_layout(legend=dict(
    orientation="v",
    yanchor="bottom",
    y=0.01,
    xanchor="right",
    x=0.96,
    font=dict(
        family="Courier",
        size=9,
        color="black"
    ),
    itemsizing='trace',
    bgcolor="rgba(255,255,255,0.0)",

))
#fig_d.write_image('norms.pdf')
fig_d.show()

# %%

'''ocdata = evaluate_trajectory(
    linear_dynamics,
    oc_baseline,
    x0,
    T,
    n_timesteps,
    method='rk4',
    options=None,
)

nncdata = evaluate_trajectory(
    linear_dynamics,
    nnc_model,
    x0,
    T,
    n_timesteps,
    method='rk4',
    options=None,
)

all_controls = pd.DataFrame(dict(oc=ocdata[:, 0, -1].detach().cpu().numpy(),
                                 nnc=nncdata[:, 0, -1].detach().cpu().numpy()))

fig_e = trendplot(all_controls['nnc'], all_controls['oc'],
                  'OC', 'NNC')

fig_e.layout.annotations[0].x = 2
fig_e.layout.annotations[0].y = -6
fig_e.layout.annotations[0].xref = 'x'
fig_e.layout.annotations[0].yref = 'y'
fig_e.layout.annotations[0].align = "right"

fig_e.layout.width = 170
fig_e.layout.height = 150
fig_e.layout.margin.l = 35
fig_e.layout.margin.t = 0
fig_e.layout.margin.b = 25
fig_e.layout.margin.r = 0
fig_e.layout.font.size = 12
fig_e.update_layout(base_temp)
fig_e.update_xaxes(axis_temp)
fig_e.update_yaxes(axis_temp)
# fig_e.write_image('corr_plot_controls.pdf')
fig_e

# %%

losses = (alldf[alldf['reached'] == True][['x1', 'x2']] ** 2).sum(1)

# %%

fig_g = px.line(y=losses)

fig_g.layout.width = 185
fig_g.layout.height = 150
fig_g.layout.margin.l = 35
fig_g.layout.margin.t = 0
fig_g.layout.margin.b = 0
fig_g.layout.margin.r = 0
fig_g.layout.font.size = 12
fig_g.update_layout(base_temp)
fig_g.update_xaxes(axis_temp)
fig_g.update_yaxes(axis_temp)
fig_g.layout.xaxis.title = r'$\tau$'
fig_g.layout.yaxis.title = r'MSE'
fig_g.layout.yaxis.type = 'log'
fig_g.layout.yaxis.exponentformat = 'power'
# fig_g.write_image('losses.pdf')
fig_g'''