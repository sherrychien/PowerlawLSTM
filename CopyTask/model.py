import torch.nn as nn
import torch
# from scipy.stats import invgamma
import torch.nn.functional as F
# from util import hard_sigm
import math as m
from torch.autograd import Variable
import torch.autograd.profiler as profiler


class BaselineLSTMModel(nn.Module):
    def __init__(self, hidden_state_size, input_size=10, output_size=10, num_layers=1):
        super(BaselineLSTMModel, self).__init__()
        self.batch_first = True
        self.layers = num_layers
        self.embed = nn.Embedding.from_pretrained(torch.eye(input_size), freeze=True)  # one hot encoding
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_state_size, batch_first=self.batch_first,
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_state_size, output_size)


    def forward(self, input, hidden_state=None):
        e = self.embed(input)
        e_packed = torch.squeeze(e)
        out_packed, hidden = self.lstm(e_packed, hidden_state)
        output = self.linear(out_packed)
        log_probs = F.log_softmax(output, dim=2)
        return log_probs, hidden

from torch.nn import Parameter



import torch.jit as jit
from typing import List, Tuple


class PowerLawCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, p, learn_p, epsilon=1e-3, p_range=None, uniform_init=False, device='cpu'):
        super(PowerLawCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learn_p = learn_p
        self.p_value = p
        self.epsilon = epsilon
        self.device = device
        self.uniform_init = uniform_init
        self.weight_ih = Parameter(torch.empty(3 * self.hidden_size, self.input_size, device=device))
        self.weight_hh = Parameter(torch.empty(3 * self.hidden_size, self.hidden_size, device=device))
        self.bias = Parameter(torch.empty(3 * self.hidden_size, device=device))

        if self.learn_p:
            self.p = Parameter(torch.empty(hidden_size, device=device))
            self.low = p_range[0]
            self.high = p_range[1]
        else:
            self.p = torch.empty((hidden_size), device=device).fill_(p)

        self.init_weights()
        self.init_p()

    def init_weights(self):
        stdv = 1.0 / m.sqrt(self.hidden_size)
        param_list = [self.weight_ih, self.weight_hh, self.bias]
        for weight in param_list:
            weight.data.uniform_(-stdv, stdv)

    def init_p(self):
        if self.learn_p:
            self.p.data.uniform_(self.low, self.high)
        if self.uniform_init:
            self.p.data.uniform_(0, 1)
            self.p.data = torch.log(self.p.data/(1-self.p.data))

    def init_hidden(self):
        # Initialize hidden state (a_0 and c_0)
        return (torch.zeros((1, self.hidden_size)),
                torch.zeros((1, self.hidden_size)))

    @jit.script_method
    def forward(self, x, t, state):
        # type: (Tensor, int, Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]

        h, c, k = state

        if self.learn_p:
            p_t = torch.sigmoid(self.p).expand_as(h)
        else:
            p_t = self.p.expand_as(h)

        gates = torch.mm(x, self.weight_ih.t()) + torch.mm(h, self.weight_hh.t()) + self.bias
        rgate, outgate, cellgate = gates.chunk(3, 1)  # 1 is the dimension of batch?

        rgate = torch.sigmoid(rgate)
        # r_t = nn.Hardsigmoid()(rgate)
        outgate = torch.sigmoid(outgate)
        cellgate = torch.tanh(cellgate)

        k_new = rgate * (t - self.epsilon) + (1 - rgate) * k  # reset k_t (t0) to current time t
        forgetgate = torch.pow(((t - k_new + self.epsilon) / (t - k_new + 1.0)), p_t)
        c_new = forgetgate * c + (1 - forgetgate) * cellgate
        h_new = outgate * torch.tanh(c_new)


        return h_new, (h_new, c_new, k_new)


class PowerLawLayer(nn.Module):
    def __init__(self, hidden_size, input_size=10, p=0.2, learn_p=False, p_range=None, uniform_init=False, epsilon=1e-3,
                 batch_first=True, input_gate=False, device='cpu'):
        super(PowerLawLayer, self).__init__()
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device
        self.input_gate = input_gate
        self.p_range = p_range
        self.uniform_init = uniform_init
        self.p = p
        self.learn_p = learn_p
        self.layer1 = PowerLawCell(self.input_size, self.hidden_size, self.p, self.learn_p,
                                   epsilon=epsilon, p_range=self.p_range, uniform_init=uniform_init, device=device)

    def forward(self, inputs, hidden):
        if self.batch_first:
            time_steps = inputs.size(1)
            batch_size = inputs.size(0)
        else:
            time_steps = inputs.size(0)
            batch_size = inputs.size(1)

        if hidden is None:
            h_t1 = Variable(torch.zeros(batch_size, self.hidden_size, device=inputs.device), requires_grad=False)
            c_t1 = Variable(torch.zeros(batch_size, self.hidden_size, device=inputs.device), requires_grad=False)
            k_t1 = Variable(torch.zeros(batch_size, self.hidden_size, device=inputs.device)-1, requires_grad=False)
            state = (h_t1, c_t1, k_t1)
        else:
            state = hidden

        h_1 = []

        if self.batch_first:
            for t in range(time_steps):
                out, state = self.layer1(inputs[:, t, :], t, state)
                h_1 += [out]
        else:
            for t in range(time_steps):
                out, state = self.layer1(inputs[t, :, :], t, state)
                h_1 += [out]

        return torch.stack(h_1, dim=1), state


class PowerLawLayerAux(jit.ScriptModule):
    def __init__(self, hidden_size, input_size=10, p=0.2, batch_first=False, learn_p=False, p_range=None, uniform_init=False,
                 epsilon=1e-3, input_gate=False, device='cpu'):
        super(PowerLawLayerAux, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device
        self.batch_first = batch_first
        self.input_gate = input_gate
        self.p_range = p_range
        self.uniform_init = uniform_init
        self.p = p
        self.learn_p = learn_p
        self.layer1 = PowerLawCell(self.input_size, self.hidden_size, self.p, self.learn_p, epsilon=epsilon,
                                   p_range=self.p_range, uniform_init=self.uniform_init, device=device)

    @jit.script_method
    def forward(self, inputs, state):
        # type: (Tensor, Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]

        h_1 = torch.jit.annotate(List[torch.Tensor], [])
        for t in range(inputs.size(1)):
            out, state = self.layer1(inputs[:, t, :], t, state)
            h_1 += [out]

        return torch.stack(h_1, dim=1), state


class PowerLawLayerJIT(nn.Module):
    def __init__(self, hidden_size, *args, batch_first=True, **kwargs):
        super(PowerLawLayerJIT, self).__init__()
        self.layer_aux = PowerLawLayerAux(hidden_size, *args, batch_first=batch_first, **kwargs)
        self.hidden_size = hidden_size
        self.batch_first = batch_first

    def forward(self, inputs, hidden):
        if self.batch_first:
            batch_size = inputs.size(0)
        else:
            batch_size = inputs.size(1)
            inputs = inputs.movedim(1, 0)

        if hidden is None:
            h_t1 = Variable(torch.zeros(batch_size, self.hidden_size, device=inputs.device), requires_grad=False)
            c_t1 = Variable(torch.zeros(batch_size, self.hidden_size, device=inputs.device), requires_grad=False)
            k_t1 = Variable(torch.zeros(batch_size, self.hidden_size, device=inputs.device)-1, requires_grad=False)
            state = (h_t1, c_t1, k_t1)
        else:
            state = hidden

        return self.layer_aux(inputs, state)
        
class PowerLawLSTM(nn.Module):
    def __init__(self, hidden_size, input_size=10, output_size=10, p=0.2, learn_p=False, p_range=None, uniform_init=False, device='cuda'):
        super(PowerLawLSTM, self).__init__()
        self.batch_first = True
        self.p = p
        self.learn_p = learn_p
        self.device = device
        self.p_range = p_range
        self.uniform_init = uniform_init
        self.embed = nn.Embedding.from_pretrained(torch.eye(input_size), freeze=True)  # one hot encoding
        self.lstm = PowerLawLayerJIT(hidden_size, input_size,  p = self.p, learn_p = self.learn_p, p_range=self.p_range, uniform_init=self.uniform_init)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden=None):
        e = self.embed(input)
        e_packed = torch.squeeze(e)
        out_packed, hidden = self.lstm(e_packed, hidden)
        output = self.linear(out_packed)
        log_probs = F.log_softmax(output, dim=2)
        return log_probs, hidden