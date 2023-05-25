import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torchdiffeq import odeint_adjoint as odeint


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class odefunc2(nn.Module):
    def __init__(self, hid_size):
        super(odefunc2, self).__init__()

        self.hidden_dim = hid_size
        self.lin_hh = nn.Linear(hid_size, self.hidden_dim, bias=False)
        self.lin_hz = nn.Linear(hid_size, self.hidden_dim, bias=False)
        self.lin_hr = nn.Linear(hid_size, self.hidden_dim, bias=False)
        self.nfe = 0

    def forward(self, t, dz):
        self.nfe += 1

        x = torch.zeros_like(dz)
        r = F.sigmoid(x + self.lin_hr(dz))
        z = F.sigmoid(x + self.lin_hz(dz))
        u = F.tanh(x + self.lin_hh(r * dz))
        dh = (1 - z) * (u - dz)
        return dh



def get_array(cycle):
    output=np.zeros([cycle*2,cycle*2])
    for i in range(0,cycle):
        output[cycle-i-1,cycle-i-1:cycle+i]= np.arange(0, 2*i+1 )+(i*2)**2
    rot1=output.copy()
    for i in range(0,cycle):
        rot1[cycle-i-1,cycle-i-1:cycle+i]+= 2*(i+1)-1
    rot1=np.rot90(rot1,k=-1)
    rot2=output.copy()
    for i in range(0,cycle):
        rot2[cycle-i-1,cycle-i-1:cycle+i]+= 4*(i+1)-2
    rot2=np.rot90(rot2,k=-2)
    rot3=output.copy()
    for i in range(0,cycle):
        rot3[cycle-i-1,cycle-i-1:cycle+i]+= 6*(i+1)-3
    rot3=np.rot90(rot3,k=-3)
    return output+rot1+rot2+rot3

class ODE_Expander(nn.Module):
    def __init__(self, hidden_num, input_size=128, outout_size=192, patch_size=32):
        super().__init__()

        self.odefunc = odefunc2(hidden_num)
        self.noise_mlp = nn.Sequential(nn.Linear(hidden_num // 8, hidden_num // 4), nn.LayerNorm(hidden_num // 4),
                                       nn.ReLU(),
                                       nn.Linear(hidden_num // 4, hidden_num // 2), nn.LayerNorm(hidden_num // 2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_num // 2, hidden_num))
        self.hidden_num = hidden_num
        self.input_query_width = input_size // patch_size
        self.output_query_width = outout_size // patch_size
        self.inner_query_index, self.outer_query_index = self.get_index()
        mask_arr = get_array(cycle=3)
        self.mask = torch.from_numpy(mask_arr).long().cuda().view(-1)

    def get_index(self):
        mask = torch.ones(size=[self.output_query_width, self.output_query_width]).long()
        pad_width = (self.output_query_width - self.input_query_width) // 2
        mask[pad_width:-pad_width, pad_width:-pad_width] = 0
        mask = mask.view(-1)
        return mask == 0, mask == 1

    def forward(self, feature):
        ori_feature = feature

        noise = torch.randn(size=(feature.size(0), self.output_query_width ** 2, self.hidden_num // 8), dtype=torch.float32).to(feature.device)
        initial_query = self.noise_mlp(noise)

        initial_query[:, self.inner_query_index] = ori_feature

        for i in range(3):
            index_start=(2*i)**2
            index_end=(2*(i+1))**2
            li=index_end-index_start
            cycle=initial_query[:,self.mask==index_start,:]
            integration_time = torch.linspace(0, 1, li, device=feature.device)
            cycle = odeint(self.odefunc, cycle, integration_time)
            for j in range(li):
                initial_query[:,self.mask==(index_start+j),:]=cycle[j]

        initial_query[:, self.inner_query_index, :] = ori_feature
        return initial_query

