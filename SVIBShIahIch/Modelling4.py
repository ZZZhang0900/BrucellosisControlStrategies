import torch
from typing import Union
from numbers import Number
from typing import Iterable
from nnc.controllers.base import ControlledDynamics, BaseController
from torchdiffeq import odeint
import numpy as np


class EluTimeControl(torch.nn.Module):
    """
    Very simple Elu architecture for control of linear systems
    """
    def __init__(self, n_nodes, n_drivers):
        super().__init__()
        self.linear = torch.nn.Linear(1, n_nodes+17)
        self.linear0 = torch.nn.Linear(n_nodes+17,n_nodes+17)
        self.linear_final = torch.nn.Linear(n_nodes+17, n_drivers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Control calculation via a fully connected NN.
        :param t: A scalar or a batch with scalars, shape: `[b, 1]` or '[1]'
        :param x: input_states for all nodes, shape `[b, m, n]`
        :return:
        """
        # sanity check to make sure we don't propagate through time
        t = t.detach() # we do not want to learn time :)
        # check for batch size and if t is scalar:
        if len(t.shape)  in {0 , 1} :
            if x is not None and len(list(x.shape)) > 1:
                t = t.repeat(x.shape[0], 1)
            else:
                # add single sample dimension if t is scalar or single dimension tensor
                # scalars are expected to have 0 dims, if i remember right?
                t = t.unsqueeze(0)



        u = self.linear(t)
        u = torch.nn.functional.elu(u)
        u = self.linear0(u)
        u = torch.nn.functional.elu(u)
        u = self.linear_final(u)
        '''if t>11:
        print('t = ',t)
        print('u = ',u)'''
        return u



class NeuralNetworkController(BaseController):

    def __init__(self, neural_net: torch.nn.Module):
        """
        Neural network wrapper for NNC.
        Provide the neural network as a submodule.
        """
        super().__init__()
        self.neural_net = neural_net

    def forward(self, t, x) -> torch.Tensor:
        """
        Wrapper method for the neural network.
        It is important that time and state tensors are provided to the neural network,
        and have the required dimensionality and values for control.
        :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
        across batch.
        :param x: A tensor containing state values, shape: `[b, m, n_nodes ]`
        :return: A tensor containing control values, shape: `[b, ?, ?]`
        """
        return self.neural_net(t, x)


class NNCDynamics(torch.nn.Module):
    def __init__(self,
                 underlying_dynamics: ControlledDynamics,
                 neural_network: torch.nn.Module,
                 ):
        """
        A constuctor that couples the controlled dynamics with the neural network.
        :param underlying_dynamics: A class implementing :class:`nnc.controllers.base.ControlledDynamics`
        :param neural_network: A neural network implementing a torch module, with inputs and
        outputs described in  :method:`nnc.controllers.base.NeuralNetworkController`
        """
        super().__init__()
        # assign nnc to the wrapper, may be considered redundant but for the sake of clarity
        self.nnc = NeuralNetworkController(neural_network)
        self.underlying_dynamics = underlying_dynamics
        # for ease of use, so that one can access the same pointer faster
        self.state_var_list = underlying_dynamics.state_var_list

    def forward(self, t, x):
        """
        Calculates the derivative or **amount of change** under neural network control for the
        given dynamics.
        Preserves gradient flows for training.
        :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
        across batch.
        :param x: A tensor containing state values, shape: `[b, m, n_nodes ]`
        :return: `dx` A tensor containing the derivative (**amount of change**) of `x`,
        shape: `[b, m, n_nodes]`
        """

        u = self.nnc(t, x)
        '''if t %0.01== 0:
            print('u=', u)
            print('t=', t)
        print('u=', u)
        print('t=', t)'''
        dx = self.underlying_dynamics(t=t, u=u, x=x)
        return dx


class brucellosis(ControlledDynamics):
    def __init__(self,
                 parameters,
                 n_drivers,
                 n_nodes,
                 dtype=torch.float32,
                 device=None
                 # k=0,
                 ):
        """
        Simplest dynamics of the form: dx/dt = AX + BU，构造方程
        :param adjacency_matrix: The matrix A
        :param driver_matrix: The matrix B
        :param dtype: Datatype of the calculations
        :param device: Device of the calculations, usuallu "cpu" or "cuda:0"
        """
        super().__init__(['concataned state of shape [8 x 1],' +
                          'with state vars in sequence: susceptible (S), infected (I), vaccinated (V), brucella (B), '
                          'susceptible_H (S_h), infected_h (I_h), chronic (I_ch), vaccinated_x (V_x),infected_h_x (I_h_x)' +
                          'containment (X)'])

        self.A = parameters[0]
        self.d_s = parameters[1]
        self.qgamma = parameters[2]
        self.B = parameters[3]
        self.d_h = parameters[4]
        self.lamda = parameters[5]
        #.theta = parameters[6]
        self.kappa = parameters[6]
        self.delta = parameters[7]
        self.ntao = parameters[8]
        '''self.beta_s = parameters[9]
        self.beta_sw = parameters[10]
        self.beta_h = parameters[11]
        self.beta_hw= parameters[12]'''


        self.label = 'SVIBShIhIch'
        self.dtype =  torch.float    #torch.float32
        self.device =  'cpu'    #self.device = device(type='cpu')

        self.n_nodes = n_nodes     #1024
        self.n_drivers = n_drivers        #512

        #driver_matrix是1024*512的矩阵，driver_matrix.sum(axis =-1或者1)按列行求和，axis =0为按列求和
        #self.k_0.size()为1024
        # self.k = k*self.driver_matrix.sum(-1) #use for closer to Brockman implmentations

    def forward2(self, t, x):
        """
        Alternative forward overlod with control concataned in the end of the state vector. Useful for some tests
        :param x: state vector with control appended
        :param t: time scalar, unused
        """
        return self.forward(x[:, :-self.drivers], t, x[:, -self.drivers:])

    def forward(self, t, x, u=None):
        """
        Dynamics forward overload with seperate control and batching.
        :param x: state vector
        :param t: time scalar, unused
        :param u: control vector
        """
        batch_size = x.shape[0]
        S = x[:,0]
        I = x[:,1]  # I
        V = x[:,2]
        B = x[:,3]
        S_h = x[:,4]
        I_h = x[:,5]
        I_ch = x[:,6]
        V_x = x[:, 7]
        I_h_x = x[:,8]

        # for sake of generality, treat no control as zeros.
        if u is None:
            u_hat = torch.zeros([batch_size, self.n_drivers], device=x.device, dtype=x.dtype)
        else:
            u_hat = u

        # calc derivatives
        #beta_s = u_hat[0,0].unsqueeze(-1).unsqueeze(-1)
        #beta_sw = u_hat[0,1].unsqueeze(-1).unsqueeze(-1)
        #beta_h = u_hat[0,0].unsqueeze(-1).unsqueeze(-1)
        #beta_hw = u_hat[0,0].unsqueeze(-1).unsqueeze(-1)
        c = u_hat[0,0].unsqueeze(-1).unsqueeze(-1)
        theta = u_hat[0,1].unsqueeze(-1).unsqueeze(-1)
        beta_s = u_hat[0, 2].unsqueeze(-1).unsqueeze(-1)
        beta_sw = u_hat[0,3].unsqueeze(-1).unsqueeze(-1)
        beta_h = u_hat[0, 4].unsqueeze(-1).unsqueeze(-1)
        beta_hw = u_hat[0, 5].unsqueeze(-1).unsqueeze(-1)

        XiShu = [1e-8, 1e-10, 1e-12, 1e-13]

        dS = self.A - beta_s * S * I *XiShu[0] - beta_sw * S * B  *XiShu[1]- theta * S + self.lamda * V - self.d_s * S
        dI = beta_s * S * I *XiShu[0]  + beta_sw * S * B *XiShu[1] - self.d_s * I - c * I
        dV = theta * S - self.lamda * V - self.d_s * V
        dB = self.kappa * I - self.delta * B - self.ntao * B
        dS_h = self.B - beta_h * S_h * I *XiShu[2] - beta_hw * S_h * B *XiShu[3] + (1-self.qgamma) * I_h - self.d_h * S_h
        dI_h = beta_h * S_h * I *XiShu[2] + beta_hw * S_h * B *XiShu[3] - I_h - self.d_h * I_h
        dI_ch = self.qgamma * I_h - self.d_h * I_ch
        dV_x = theta * S
        dI_h_x = beta_h * S_h * I *XiShu[2] + beta_hw * S_h * B *XiShu[3]

        # stack derivatives to state in received order
        dx = torch.cat([dS, dI, dV, dB.unsqueeze(-1),
                        dS_h, dI_h, dI_ch.unsqueeze(-1),
                        dV_x, dI_h_x], dim=-1)
        #print('u_hat = ',u_hat)
        return dx

def evaluate_trajectory(dynamics, controller, x0, total_time, n_timesteps, method='rk4',
                        options=None):
    all_controls = []
    all_control_times = []
    all_timesteps = torch.linspace(0, 10, 11)

    def apply_control(t, x):
        u = controller(t, x)
        #print(u.shape)
        all_control_times.append(t)
        all_controls.append(u)
        dx = dynamics(t=t, x=x, u=u)
        return dx

    trajectory = odeint(apply_control,
                        x0,
                        all_timesteps,
                        method=method,
                        options=None
                        )  # timesteps x n_nodes


    #print('trajectory(apply_control) = ',trajectory)

    all_controls = torch.stack(all_controls, 0)  # timesteps x n_nodes
    all_control_times = torch.stack(all_control_times)  # timesteps x 1


    # align ode_solver control timesteps with requested timesteps
    _, relevant_time_index = closest_previous_time(all_timesteps, all_control_times)
    relevant_controls = all_controls[relevant_time_index, :]

    return torch.cat([trajectory, relevant_controls], -1)

def closest_previous_time(requested_times, solver_times):
    requested_times = requested_times.unsqueeze(1)
    solver_times = solver_times.unsqueeze(0)
    difft = (requested_times - solver_times)
    difft = difft
    difft[difft < 0] = np.infty
    time_index = difft.argmin(1).flatten()
    return solver_times.squeeze()[time_index], time_index






