from reaction_diffusion import Assembler, reaction_diffusion_nature, reaction_diffusion_fast, multi_reaction_diffusion, reaction_diffusion_fast_ode, multi_reaction_diffusion_concat_out, second_layer_linear, viscoelastic
from model_GMAN import FC, temporalAttention

adjoint = True
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
from torchdiffeq import odeint_event


import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F


def softmax_temperature(input, t=1.0):
    # print("input", input)
    ex = torch.exp(input/t)
    # print("exp", ex)
    sum = torch.sum(ex, axis=-1)
    return ex / sum[:, :, :, None]

def sigmoid_temperature(input, t=1.0):
    ex = torch.exp(-input/t)
    return 1 / (1 + ex)

def binary(input):
    g = torch.sign(input)
    return (g*0.5) + 0.5

class ReactionDiffusionODE(nn.Module):
    def __init__(self, device, adj_path, num_node, w_o_reaction_diffusion, resolution, num_sequence=2, num_output_sequence=-1, enable_bias=False, num_rd_kernels=2, temperture=1):
        super(ReactionDiffusionODE, self).__init__()

        df = pd.read_csv(adj_path)
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = num_node
        self.num_output_sequence = num_output_sequence
        self.sigmoid = nn.Sigmoid()

        bn_decay = 0.1
        K = 8
        d = 8
        D = K * d
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=False)
        self.FC_1 = FC(input_dims=[3, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        # 2 for channel 1: 0, and channel 2, jump
        self.FC_2 = FC(input_dims=[D, D], units=[D, num_rd_kernels], activations=[F.relu, None],
                       bn_decay=bn_decay)
        
        self.softmax = nn.Softmax(dim = -1)
        self.temperture = temperture

        self.samp_ts = torch.from_numpy(np.linspace(0.0, 1.0, num=num_sequence)).to(device)
        self.samp_ts_gate = torch.from_numpy(np.linspace(0.0, 1.0, num=num_sequence-1)).to(device)
        
        self.odeFunc = multi_reaction_diffusion(device, adj_path, num_node, w_o_reaction_diffusion, resolution, enable_bias, num_sequence, self.samp_ts_gate, num_rd_kernels)    
        # self.odeFunc = reaction_diffusion_nature(device, adj_path, num_node, w_o_reaction_diffusion, resolution) 
        self.count = 0
        

    def forward(self, inputs):
        
        # inputs = inputs[:, :2, :, :]
        # temporal_input = self.FC_1(inputs.transpose(1, 3))

        temporal_input = self.FC_1(inputs.transpose(1, 3))
        temporal_output = self.temporalAttention(temporal_input)
        gate = self.FC_2(temporal_output)
        
        # self.odeFunc.l = sigmoid_temperature(gate, t=0.01)
        # self.odeFunc.l = binary(gate)
        # self.odeFunc.l = softmax_temperature(gate, self.temperture)
        self.odeFunc.l = self.softmax(gate)
        
        # self.odeFunc.l = self.sigmoid(gate)
        # gate = (gate > 0.8) * 0.4 + 0.4

        # np.save("temp-16-20/gate-sequence-" + str(self.count) + ".npy", self.odeFunc.l.cpu().numpy())
        self.count = self.count + 1
        

        input = inputs[:, 0, :, -1]
        output = odeint(self.odeFunc, input, self.samp_ts, gate)[self.num_output_sequence:]

        # output.shape = [12, 64, 207]
        return output

class JumpReactionDiffusionODE(nn.Module):
    def __init__(self, device, adj_path, num_node, w_o_reaction_diffusion, resolution, num_sequence=2, num_output_sequence=-1, enable_bias=False, num_rd_kernels=2, temperture=1):
        super(JumpReactionDiffusionODE, self).__init__()

        df = pd.read_csv(adj_path)
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = num_node
        self.num_output_sequence = num_output_sequence
        self.sigmoid = nn.Sigmoid()

        bn_decay = 0.1
        K = 8
        d = 8
        D = K * d
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=False)
        self.FC_1 = FC(input_dims=[3, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        # 2 for channel 1: 0, and channel 2, jump
        self.FC_2 = FC(input_dims=[D, D], units=[D, num_rd_kernels], activations=[F.relu, None],
                       bn_decay=bn_decay)
        
        self.temporalAttention_jump = temporalAttention(K, d, bn_decay, mask=False)
        self.FC_1_jump = FC(input_dims=[3, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        # 2 for channel 1: 0, and channel 2, jump
        self.FC_2_jump = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=bn_decay)
        

        self.FC_1.convs[0].conv.weight.register_hook(lambda x: print('gate temporal attention grad accumulated in temporalAttention'))
        self.FC_1_jump.convs[0].conv.weight.register_hook(lambda x: print('jump temporal attention grad accumulated in temporalAttention'))
        
        self.softmax = nn.Softmax(dim = -1)
        self.temperture = temperture

        self.samp_ts = torch.from_numpy(np.linspace(0.0, 1.0, num=num_sequence)).to(device)
        self.samp_ts_gate = torch.from_numpy(np.linspace(0.0, 1.0, num=num_sequence-1)).to(device)
        
        self.odeFunc = multi_reaction_diffusion(device, adj_path, num_node, w_o_reaction_diffusion, resolution, enable_bias, num_sequence, self.samp_ts_gate, num_rd_kernels)    
        # self.odeFunc = reaction_diffusion_nature(device, adj_path, num_node, w_o_reaction_diffusion, resolution) 
        self.count = 0
        

    def forward(self, inputs):
        
        # inputs = inputs[:, :2, :, :]
        # temporal_input = self.FC_1(inputs.transpose(1, 3))

        temporal_input = self.FC_1(inputs.transpose(1, 3))
        temporal_output = self.temporalAttention(temporal_input)
        gate = self.FC_2(temporal_output)

        temporal_input_jump = self.FC_1_jump(inputs.transpose(1, 3))
        temporal_output_jump = self.temporalAttention_jump(temporal_input_jump)
        jump = self.FC_2_jump(temporal_output_jump)        
        
        # self.odeFunc.l = sigmoid_temperature(gate, t=0.01)
        # self.odeFunc.l = binary(gate)
        # self.odeFunc.l = softmax_temperature(gate, self.temperture)
        self.odeFunc.l = self.softmax(gate)
        self.odeFunc.jump = jump
        # self.odeFunc.l = self.sigmoid(gate)
        # gate = (gate > 0.8) * 0.4 + 0.4

        # np.save("temp-16-20/gate-sequence-" + str(self.count) + ".npy", self.odeFunc.l.cpu().numpy())
        self.count = self.count + 1
        

        input = inputs[:, 0, :, -1]
        output = odeint(self.odeFunc, input, self.samp_ts)[self.num_output_sequence:]

        # output.shape = [12, 64, 207]
        return output

class JumpReactionDiffusionODEV2(nn.Module):
    def __init__(self, device, adj_path, num_node, w_o_reaction_diffusion, resolution, num_sequence=2, num_output_sequence=-1, enable_bias=False, num_rd_kernels=2, temperture=1):
        super(JumpReactionDiffusionODEV2, self).__init__()

        df = pd.read_csv(adj_path)
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = num_node
        self.num_output_sequence = num_output_sequence
        self.sigmoid = nn.Sigmoid()

        bn_decay = 0.1
        K = 8
        d = 8
        D = K * d
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=False)
        self.FC_1 = FC(input_dims=[3, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        # 2 for channel 1: 0, and channel 2, jump
        self.FC_2 = FC(input_dims=[D, D], units=[D, num_rd_kernels], activations=[F.relu, None],
                       bn_decay=bn_decay)
        
        self.temporalAttention_jump = temporalAttention(K, d, bn_decay, mask=False)
        self.FC_1_jump = FC(input_dims=[3, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        # 2 for channel 1: 0, and channel 2, jump
        self.FC_2_jump = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=bn_decay)
        

        # self.FC_1.convs[0].conv.weight.register_hook(lambda x: print('gate temporal attention grad accumulated in temporalAttention'))
        # self.FC_1_jump.convs[0].conv.weight.register_hook(lambda x: print('jump temporal attention grad accumulated in temporalAttention'))
        
        self.softmax = nn.Softmax(dim = -1)
        self.temperture = temperture

        self.samp_ts = torch.from_numpy(np.linspace(0.0, 1.0, num=num_sequence)).to(device)
        self.samp_ts_gate = torch.from_numpy(np.linspace(0.0, 1.0, num=num_sequence-1)).to(device)
        
        modulelist = []
        self.num_rd_kernels = num_rd_kernels
        for i in range(num_rd_kernels-1):
            modulelist.append(reaction_diffusion_fast_ode(device, adj_path, num_node, [True, True], resolution, enable_bias, 10/num_rd_kernels*i))
        self.modulelist = nn.ModuleList(modulelist)  
        # self.odeFunc = reaction_diffusion_nature(device, adj_path, num_node, w_o_reaction_diffusion, resolution) 
        self.count = 0
        

    def forward(self, inputs):
        
        # inputs = inputs[:, :2, :, :]
        # temporal_input = self.FC_1(inputs.transpose(1, 3))

        temporal_input = self.FC_1(inputs.transpose(1, 3))
        temporal_output = self.temporalAttention(temporal_input)
        gate = self.FC_2(temporal_output)
        gate = self.softmax(gate)
        gate = torch.transpose(gate, 0, 1)

        temporal_input_jump = self.FC_1_jump(inputs.transpose(1, 3))
        temporal_output_jump = self.temporalAttention_jump(temporal_input_jump)
        jump = self.FC_2_jump(temporal_output_jump)   
        jump = torch.transpose(jump, 0, 1)[..., -1]     

        input = inputs[:, 0, :, -1]
        outputs = []
        for i in range(self.num_rd_kernels-1):
            outputs.append(odeint(self.modulelist[i], input, self.samp_ts)[self.num_output_sequence:])
        outputs = torch.stack(outputs, dim=-1)
        
        # output.shape = [12, 64, 207]
        # return torch.sum(gate[..., :-1] * outputs, -1) + gate[..., -1] * jump
        return gate[..., -1] * jump

class Jump(nn.Module):
    def __init__(self, device, adj_path, num_node, w_o_reaction_diffusion, resolution, num_sequence=2, num_output_sequence=-1, enable_bias=False, num_rd_kernels=2, temperture=1):
        super(Jump, self).__init__()

        df = pd.read_csv(adj_path)
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = num_node
        self.num_output_sequence = num_output_sequence
        self.sigmoid = nn.Sigmoid()

        bn_decay = 0.1
        K = 8
        d = 8
        D = K * d
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=False)
        self.FC_1 = FC(input_dims=[3, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        # 2 for channel 1: 0, and channel 2, jump
        self.FC_2 = FC(input_dims=[D, D], units=[D, num_rd_kernels], activations=[F.relu, None],
                       bn_decay=bn_decay)
        
        self.softmax = nn.Softmax(dim = -1)
        self.temperture = temperture

    def forward(self, inputs, ind):
        
        # inputs = inputs[:, :2, :, :]
        # temporal_input = self.FC_1(inputs.transpose(1, 3))

        temporal_input = self.FC_1(inputs.transpose(1, 3))
        temporal_output = self.temporalAttention(temporal_input)
        gate = self.FC_2(temporal_output)
        
        # self.odeFunc.l = sigmoid_temperature(gate, t=0.01)
        # self.odeFunc.l = binary(gate)
        # self.odeFunc.l = softmax_temperature(gate, self.temperture)
        gate = gate[:, :, :, 0]
        return gate.transpose(0, 1)

class ReactionDiffusionODE_fix_gate(nn.Module):
    def __init__(self, device, adj_path, num_node, w_o_reaction_diffusion, resolution, num_sequence=2, num_output_sequence=-1, enable_bias=False, num_rd_kernels=2, temperture=1):
        super(ReactionDiffusionODE_fix_gate, self).__init__()

        df = pd.read_csv(adj_path)
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = num_node
        self.num_output_sequence = num_output_sequence
        self.sigmoid = nn.Sigmoid()
        
        self.softmax = nn.Softmax(dim = -1)
        self.temperture = temperture

        self.samp_ts = torch.from_numpy(np.linspace(0.0, 1.0, num=num_sequence)).to(device)
        self.samp_ts_gate = torch.from_numpy(np.linspace(0.0, 1.0, num=num_sequence-1)).to(device)
        
        self.odeFunc = multi_reaction_diffusion(device, adj_path, num_node, w_o_reaction_diffusion, resolution, enable_bias, num_sequence, self.samp_ts_gate, num_rd_kernels)    
        # self.odeFunc = reaction_diffusion_nature(device, adj_path, num_node, w_o_reaction_diffusion, resolution) 
        self.count = 0
        

    def forward(self, inputs, gate):
        
        # inputs = inputs[:, :2, :, :]
        # temporal_input = self.FC_1(inputs.transpose(1, 3)
        
        self.odeFunc.l = gate
        input = inputs[:, 0, :, -1]
        output = odeint(self.odeFunc, input, self.samp_ts)[self.num_output_sequence:]

        return output

class reaction_diffusion_nature_layer(nn.Module):
    def __init__(self, device, up_or_down, resolution, mat, node_size):
        super(reaction_diffusion_nature_layer, self).__init__()

        self.device = device
        self.resolution = resolution
        print("resolution: ", self.resolution)
        self.nconv = nconv()
        self.node_size = node_size

        # upper and lower can
        self.up = up_or_down[0]
        self.down = up_or_down[1]

        i, j = np.nonzero(mat)
        self.edge_size = len(i)
        self.index = [i, j]
        self.index_a = [j, i] 


        self.weight_react = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 10).to(device), requires_grad=True).to(device)
        self.weight_diff = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 1000).to(device), requires_grad=True).to(device)
        self.bias_reaction = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)
        self.bias_diffusion = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)            

        print("reaction and diffusion factors are same as nature")
        # self.w_d = nn.Parameter((torch.randn(1, 307, 1) / 10).to(device), requires_grad=True).to(device)


        self.tanh = nn.Tanh()

    def reac_diff_weight_construct(self, ind):
        ii = int(ind[0] / self.resolution)
    
        reaction_weight = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
        I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
        # reaction_weight =  I1 + reaction_weight
        reaction_weight =  I1 - reaction_weight

        diffusion_weight = torch.sparse_coo_tensor(self.index_a, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
        I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
        diffusion_weight = I2 - diffusion_weight

        for i in range(1, len(ind)):
            ii = int(ind[i] / self.resolution)
            reaction_element = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
            diffusion_element = torch.sparse_coo_tensor(self.index_a, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]         

            # create laplacian
            I1 = torch.diag(torch.sum(reaction_element, 1)[0, :])[None, :, :]
            # not laplacian
            # reaction_element = I1 + reaction_element
            reaction_element = I1 - reaction_element
            I2 = torch.diag(torch.sum(diffusion_element, 1)[0, :])[None, :, :]
            diffusion_element = I2 - diffusion_element

            reaction_weight = torch.cat((reaction_weight, reaction_element), 0)
            diffusion_weight = torch.cat((diffusion_weight, diffusion_element), 0)
        return reaction_weight.to(self.device), diffusion_weight.to(self.device)

    def reac_diff_bias_construct(self, ind):
        ind = [int(i/self.resolution) for i in ind]
        reaction_bias = self.bias_reaction[ind][:, :, None]
        diffusion_bias = self.bias_diffusion[ind][:, :, None]
        return reaction_bias.to(self.device), diffusion_bias.to(self.device)

    def forward(self, inputs, ind):
        
        input = inputs[:, 0, :, :]

        reaction_weight, diffusion_weight = self.reac_diff_weight_construct(ind)
        reaction_bias, diffusion_bias = self.reac_diff_bias_construct(ind)

        reaction = self.nconv(input, reaction_weight) + reaction_bias

        # heat diffusion kernel
        diffusion = self.nconv(input, diffusion_weight) + diffusion_bias

        # return (reaction) + self.tanh(diffusion) + input
        if(self.up and self.down):
            return self.tanh(reaction) + (diffusion) + input
        elif(self.up):
            return reaction + input
        elif(self.down):
            return diffusion + input
        else:
            print("bad input")

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nvl,nwv->nwl', (x, A))
        return x.contiguous()


class ReactionDiffusionODE_old(nn.Module):
    def __init__(self, device, resolution):
        super(ReactionDiffusionODE_old, self).__init__()

        df = pd.read_csv("data/metr-la/metrla-virtual-id-revised.csv")
        # df = pd.read_csv("data/seattle-loop/seattle_loop_virtual_link.csv")
        # df = pd.read_csv("data/pems-bay/pems-bay-virtual-id.csv")
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = 207
        # self.node_size = 323
        # self.node_size = 281
        A = np.zeros((self.node_size, self.node_size))

        for a in range(len(i)):
            A[i[a], j[a]] = 1
            A[j[a], i[a]] = 1

        A_pow = np.matmul(A, A)
        A_tri = np.matmul(A_pow, A)        

        self.ode_function_1_1 = reaction_diffusion_nature_layer(device, [True, True], resolution, A, self.node_size)
        self.ode_function_2_1 = reaction_diffusion_nature_layer(device, [True, True], resolution, A_pow, self.node_size)
        self.ode_function_3_1 = reaction_diffusion_nature_layer(device, [True, True], resolution, A_tri, self.node_size)
        self.assembler = Assembler(device, 3, 1, resolution)
        

    def forward(self, inputs, ind):
        input = inputs[:, :, :, -3][:, :, :, None]
        output1 = self.ode_function_3_1(input, ind-2)[:, None, :, :]
        output11 = self.ode_function_3_1(output1, ind-1)
        input = inputs[:, :, :, -2][:, :, :, None]
        output2 = self.ode_function_2_1(input, ind-1)
        input = inputs[:, :, :, -1][:, :, :, None]
        output3 = self.ode_function_1_1(input, ind)

        mid_output = torch.cat((output11, output2, output3), 2)
        output = self.assembler(mid_output, ind)

        return output
    

class ReactionDiffusionODE_one2one(nn.Module):
    def __init__(self, device, adj_path, num_node, w_o_reaction_diffusion, resolution, num_sequence=2, num_output_sequence=-1, enable_bias=False, num_rd_kernels=2, temperture=1):
        super(ReactionDiffusionODE_one2one, self).__init__()

        df = pd.read_csv(adj_path)
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = num_node
        self.num_output_sequence = num_output_sequence

        self.samp_ts = torch.from_numpy(np.linspace(0.0, 1.0, num=num_sequence)).to(device)
        self.samp_ts_gate = torch.from_numpy(np.linspace(0.0, 1.0, num=num_sequence-1)).to(device)
        
        self.odeFunc = reaction_diffusion_fast_ode(device, adj_path, num_node, w_o_reaction_diffusion, resolution, True, 1)  
        # self.odeFunc = reaction_diffusion_nature(device, adj_path, num_node, w_o_reaction_diffusion, resolution) 
        self.count = 0
        

    def forward(self, inputs, ind):

        input = inputs[:, 0, :, -1]
        output = odeint(self.odeFunc, input, self.samp_ts)[self.num_output_sequence:]

        return output
    
# This class consist of Neural ODE with just one RD equation.
class RDGODE_fixed_gate(nn.Module):
    def __init__(self, device, adj_path, num_node, w_o_reaction_diffusion, resolution, num_sequence=2, num_output_sequence=-1, enable_bias=False, num_rd_kernels=2, temperture=1):
        super(RDGODE_fixed_gate, self).__init__()

        df = pd.read_csv(adj_path)
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = num_node
        self.num_output_sequence = num_output_sequence

        self.samp_ts = torch.from_numpy(np.linspace(0.0, 1.0, num=num_sequence)).to(device)
        self.samp_ts_gate = torch.from_numpy(np.linspace(0.0, 1.0, num=num_sequence-1)).to(device)
        
        up_or_down = [True, True]
        self.odeFunc = reaction_diffusion_fast_ode(device, adj_path, num_node, up_or_down, resolution, enable_bias, 0.9) 
        # self.odeFunc = reaction_diffusion_nature(device, adj_path, num_node, w_o_reaction_diffusion, resolution) 
        self.count = 0
        

    def forward(self, inputs, ind):

        input = inputs[:, 0, :, -1]
        output = odeint(self.odeFunc, input, self.samp_ts)[self.num_output_sequence:]

        return output
    
    # This class consist of Neural ODE with just one RD equation.
class grey_local(nn.Module):
    def __init__(self, device, adj_path, num_node, w_o_reaction_diffusion, resolution, num_sequence=2, num_output_sequence=-1, enable_bias=False, num_rd_kernels=2, temperture=1):
        super(grey_local, self).__init__()

        df = pd.read_csv(adj_path)
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = num_node
        self.num_output_sequence = num_output_sequence
        self.device = device

        self.samp_ts = torch.from_numpy(np.linspace(0.0, 1.0, num=num_sequence)).to(device)
        self.samp_ts_gate = torch.from_numpy(np.linspace(0.0, 1.0, num=num_sequence-1)).to(device)
        
        up_or_down = [True, True]
        # self.odeFunc = second_layer_linear(device, adj_path, num_node, up_or_down, resolution, enable_bias, 0.9) 
        self.odeFunc = viscoelastic(device, adj_path, num_node, up_or_down, resolution, enable_bias, 0.9) 
        # self.odeFunc = reaction_diffusion_nature(device, adj_path, num_node, w_o_reaction_diffusion, resolution) 
        self.count = 0

        # We don't consider transient and equilibrium at this stage
        self.aggregator = nn.Parameter((torch.randn(self.node_size, 1)).to(device), requires_grad=True).to(device)
        self.nn = torch.arange(0, 14).to(device)

    
    def cal_aggregate_element(self, r, n):
        bottom = np.math.factorial(n)
        top = torch.prod(self.nn[:n-1] + r)
        return top/bottom

    def forward(self, inputs, ind):

        Ar = torch.diag(torch.ones(12))[None, :, :].repeat(self.node_size, 1, 1)
        col = 0
        for i in range(self.node_size):
            for j in range(12):
                for k in range(col):
                    Ar[i, j, k] = self.cal_aggregate_element(self.aggregator[i], j)
                col = col + 1
            col = 0
        input = inputs[:, 0, :, -1:]
        Ar = Ar[None, :, :, :].repeat(64, 1, 1, 1)
        # Ar = torch.abs(Ar)
        xr = torch.matmul(Ar.to(self.device), inputs[:, 0, :, :][:, :, :, None])[:, :, :, 0]

        total = (torch.sum((xr), axis = [0, 2]) / xr.shape[0])[:, None]
        total_mask = (total < 5) * 5

        self.odeFunc.M = total_mask + torch.abs(total)
        output = odeint(self.odeFunc, input, self.samp_ts)[self.num_output_sequence:]
        Ar_inv = torch.inverse(Ar)
        output = torch.matmul(Ar_inv.to(self.device), output.transpose(0, 1).transpose(1, 2))

        return output[:, :, :, 0]

    # def forward(self, inputs, ind):

    #     Ar = torch.diag(torch.ones(12))[None, :, :].repeat(self.node_size, 1, 1)
    #     col = 0
    #     for i in range(self.node_size):
    #         for j in range(12):
    #             for k in range(col):
    #                 Ar[i, j, k] = self.cal_aggregate_element(self.aggregator[i], j)
    #             col = col + 1
    #         col = 0
    #     input = inputs[:, 0, :, :]
    #     Ar = Ar[None, :, :, :].repeat(64, 1, 1, 1)
    #     xr = torch.matmul(Ar.to(self.device), input[:, :, :, None])[:, :, :, 0]
    #     output = odeint(self.odeFunc, xr, self.samp_ts)[self.num_output_sequence]
    #     Ar_inv = torch.inverse(Ar)
    #     output = torch.matmul(Ar_inv.to(self.device), output[:, :, :, None])
    #     return output[:, :, :, 0]