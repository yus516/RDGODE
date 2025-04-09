import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nvl,nwv->nwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gated_reaction_diffusion_gcn(nn.Module):
    def __init__(self, device, symetric, resolution, has_self_loop):
        super(gated_reaction_diffusion_gcn, self).__init__()

        self.device = device
        self.resolution = resolution
        print("resolution: ", self.resolution)
        self.nconv = nconv()

        self.symetric = symetric
        self.has_self_loop = has_self_loop

        df = pd.read_csv("data/metr-la/metrla-virtual-id-revised.csv")
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = 207

        # self.w_d = nn.Parameter((torch.randn(1, 307, 1) / 10).to(device), requires_grad=True).to(device)

        if symetric:
            self.index = [i+j, j+i] 
            self.weight_react = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 10).to(device), requires_grad=True).to(device)
            self.weight_diff = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 1000).to(device), requires_grad=True).to(device)

        else: 
            self.index = [i, j] 
            print("reaction and diffusion factors are not symetric")
            self.weight_react = nn.Parameter((torch.randn(int(288 * 2 / resolution), self.edge_size) / 10).to(device), requires_grad=True).to(device)
            self.weight_diff = nn.Parameter((torch.randn(int(288 * 2 / resolution), self.edge_size) / 1000).to(device), requires_grad=True).to(device)

        self.bias_reaction = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)
        self.bias_diffusion = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)

        self.gate_weight_edge = nn.Parameter((torch.randn(self.edge_size) / 1).to(device), requires_grad=True).to(device)
        self.gate_weight_node = nn.Parameter((torch.randn(self.node_size) / 1).to(device), requires_grad=True).to(device)

        if self.has_self_loop:
            self.weight_self_loop = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def gate_weight_construct(self):
        if self.symetric:
            gate_weight_not_diag = torch.sparse_coo_tensor(self.index, torch.cat((self.gate_weight_edge, self.gate_weight_edge), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
        else:
            gate_weight_not_diag = torch.sparse_coo_tensor(self.index, self.gate_weight_edge, (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
        gate_weight_diag = torch.diag(self.gate_weight_node)
        return gate_weight_not_diag + gate_weight_diag

    def reac_diff_weight_construct(self, ind):
        ii = int(ind[0] / self.resolution)

        if self.symetric:
            reaction_weight = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_react[ii], self.weight_react[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
            I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
            reaction_weight =  I1 + reaction_weight
            # reaction_weight =  I1 - reaction_weight

            diffusion_weight = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_diff[ii], self.weight_diff[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
            I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
            diffusion_weight = I2 - diffusion_weight
        
        else: 
            reaction_weight = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
            I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
            reaction_weight =  I1 + reaction_weight
            # reaction_weight =  I1 - reaction_weight

            diffusion_weight = torch.sparse_coo_tensor(self.index, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
            I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
            diffusion_weight = I2 - diffusion_weight

        for i in range(1, len(ind)):
            ii = int(ind[i] / self.resolution)
            if self.symetric:
                reaction_element = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_react[ii], self.weight_react[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
                diffusion_element = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_diff[ii], self.weight_diff[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
            else:
                reaction_element = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
                diffusion_element = torch.sparse_coo_tensor(self.index, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]                

            # create laplacian
            I1 = torch.diag(torch.sum(reaction_element, 1)[0, :])[None, :, :]
            # not laplacian
            reaction_element = I1 + reaction_element
            # reaction_element = I1 - reaction_element
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
        
        input = inputs.reshape([inputs.shape[0], inputs.shape[1] * inputs.shape[2], inputs.shape[3]])
        input = input.transpose(1,2)

        reaction_weight, diffusion_weight = self.reac_diff_weight_construct(ind)
        reaction_bias, diffusion_bias = self.reac_diff_bias_construct(ind)
        gate_weight = self.gate_weight_construct()
        gate = self.nconv(input, gate_weight)

        reaction = self.nconv(input, reaction_weight) + reaction_bias

        # heat diffusion kernel
        diffusion = self.nconv(input, diffusion_weight) + diffusion_bias

        if self.has_self_loop:
            ind = [int(i/self.resolution) for i in ind]
            self_loop_bias = torch.mul(self.weight_self_loop[ind][:, :, None], input)
            result = self.tanh(reaction) + diffusion + input + self_loop_bias
            return result.reshape(inputs.shape)
        else:
            # add the gate to the reaction
            gated_reaction = torch.mul(self.sigmoid(gate), reaction)
            gated_diffusion = torch.mul((1-self.sigmoid(gate)), diffusion)
            result = gated_reaction + gated_diffusion + input
            return result.reshape(inputs.shape)
        # return (reaction) + self.tanh(diffusion) + input
        
        # return self.tanh(reaction) + self.tanh(diffusion) + input
        # return diffusion + input

class gated_diffusion_diffusion_gcn(nn.Module):
    def __init__(self, device, symetric, resolution, has_self_loop):
        super(gated_diffusion_diffusion_gcn, self).__init__()

        self.device = device
        self.resolution = resolution
        print("resolution: ", self.resolution)
        self.nconv = nconv()

        self.symetric = symetric
        self.has_self_loop = has_self_loop

        df = pd.read_csv("data/metr-la/metrla-virtual-id-revised.csv")
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = 207

        # self.w_d = nn.Parameter((torch.randn(1, 307, 1) / 10).to(device), requires_grad=True).to(device)

        if symetric:
            self.index = [i+j, j+i] 
            self.weight_react = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 10).to(device), requires_grad=True).to(device)
            self.weight_diff = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 1000).to(device), requires_grad=True).to(device)

        else: 
            self.index = [i, j] 
            print("reaction and diffusion factors are not symetric")
            self.weight_react = nn.Parameter((torch.randn(int(288 * 2 / resolution), self.edge_size) / 10).to(device), requires_grad=True).to(device)
            self.weight_diff = nn.Parameter((torch.randn(int(288 * 2 / resolution), self.edge_size) / 1000).to(device), requires_grad=True).to(device)

        self.bias_reaction = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)
        self.bias_diffusion = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)

        self.gate_weight_edge = nn.Parameter((torch.randn(self.edge_size) / 1).to(device), requires_grad=True).to(device)
        self.gate_weight_node = nn.Parameter((torch.randn(self.node_size) / 1).to(device), requires_grad=True).to(device)

        if self.has_self_loop:
            self.weight_self_loop = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def gate_weight_construct(self):
        if self.symetric:
            gate_weight_not_diag = torch.sparse_coo_tensor(self.index, torch.cat((self.gate_weight_edge, self.gate_weight_edge), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
        else:
            gate_weight_not_diag = torch.sparse_coo_tensor(self.index, self.gate_weight_edge, (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
        gate_weight_diag = torch.diag(self.gate_weight_node)
        return gate_weight_not_diag + gate_weight_diag

    def reac_diff_weight_construct(self, ind):
        ii = int(ind[0] / self.resolution)

        if self.symetric:
            reaction_weight = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_react[ii], self.weight_react[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
            I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
            reaction_weight =  I1 - reaction_weight
            # reaction_weight =  I1 - reaction_weight

            diffusion_weight = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_diff[ii], self.weight_diff[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
            I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
            diffusion_weight = I2 - diffusion_weight
        
        else: 
            reaction_weight = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
            I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
            reaction_weight =  I1 - reaction_weight
            # reaction_weight =  I1 - reaction_weight

            diffusion_weight = torch.sparse_coo_tensor(self.index, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
            I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
            diffusion_weight = I2 - diffusion_weight

        for i in range(1, len(ind)):
            ii = int(ind[i] / self.resolution)
            if self.symetric:
                reaction_element = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_react[ii], self.weight_react[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
                diffusion_element = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_diff[ii], self.weight_diff[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
            else:
                reaction_element = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
                diffusion_element = torch.sparse_coo_tensor(self.index, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]                

            # create laplacian
            I1 = torch.diag(torch.sum(reaction_element, 1)[0, :])[None, :, :]
            # not laplacian
            reaction_element = I1 - reaction_element
            # reaction_element = I1 - reaction_element
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
        gate_weight = self.gate_weight_construct()
        gate = self.nconv(input, gate_weight)

        reaction = self.nconv(input, reaction_weight) + reaction_bias

        # heat diffusion kernel
        diffusion = self.nconv(input, diffusion_weight) + diffusion_bias

        if self.has_self_loop:
            ind = [int(i/self.resolution) for i in ind]
            self_loop_bias = torch.mul(self.weight_self_loop[ind][:, :, None], input)
            return self.tanh(reaction) + diffusion + input + self_loop_bias
        else:
            # add the gate to the reaction
            gated_reaction = torch.mul(self.tanh(gate), reaction)
            return self.sigmoid(gated_reaction) + diffusion + input
        # return (reaction) + self.tanh(diffusion) + input
        
        # return self.tanh(reaction) + self.tanh(diffusion) + input
        # return diffusion + input

class reaction_diffusion_gcn(nn.Module):
    # for stgcn 
    def __init__(self, symetric, resolution, has_self_loop):
        super(reaction_diffusion_gcn, self).__init__()

        print("reaction_diffusion_gcn: symetric is ", symetric, "resolution is ", resolution, "has self loop ", has_self_loop)

        self.resolution = resolution
        print("resolution: ", self.resolution)
        self.nconv = nconv()

        self.symetric = symetric
        self.has_self_loop = has_self_loop

        df = pd.read_csv("data/metr-la/metrla-virtual-id-revised.csv")
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = 207

        # self.w_d = nn.Parameter((torch.randn(1, 307, 1) / 10).to(device), requires_grad=True).to(device)

        if symetric:
            self.index = [i+j, j+i] 
            self.weight_react = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 10), requires_grad=True)
            self.weight_diff = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 1000), requires_grad=True)

        else: 
            self.index = [i, j] 
            print("reaction and diffusion factors are not symetric")
            self.weight_react = nn.Parameter((torch.randn(int(288 * 2 / resolution), self.edge_size) / 10), requires_grad=True)
            self.weight_diff = nn.Parameter((torch.randn(int(288 * 2 / resolution), self.edge_size) / 1000), requires_grad=True)

        self.bias_reaction = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10), requires_grad=True)
        self.bias_diffusion = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10), requires_grad=True)

        if self.has_self_loop:
            self.weight_self_loop = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10), requires_grad=True)

        self.device = "cuda:1"
        self.tanh = nn.Tanh()

    def reac_diff_weight_construct(self, ind):
        ii = int(ind[0] / self.resolution)

        if self.symetric:
            reaction_weight = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_react[ii], self.weight_react[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
            I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
            reaction_weight =  I1 + reaction_weight
            # reaction_weight =  I1 - reaction_weight

            diffusion_weight = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_diff[ii], self.weight_diff[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
            I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
            diffusion_weight = I2 - diffusion_weight
        
        else: 
            reaction_weight = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
            I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
            reaction_weight =  I1 + reaction_weight
            # reaction_weight =  I1 - reaction_weight

            diffusion_weight = torch.sparse_coo_tensor(self.index, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
            I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
            diffusion_weight = I2 - diffusion_weight

        for i in range(1, len(ind)):
            ii = int(ind[i] / self.resolution)
            if self.symetric:
                reaction_element = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_react[ii], self.weight_react[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
                diffusion_element = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_diff[ii], self.weight_diff[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
            else:
                reaction_element = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
                diffusion_element = torch.sparse_coo_tensor(self.index, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]                

            # create laplacian
            I1 = torch.diag(torch.sum(reaction_element, 1)[0, :])[None, :, :]
            # not laplacian
            reaction_element = I1 + reaction_element
            # reaction_element = I1 - reaction_element
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
        
        # input = inputs[:, 0, :, :]
        input = inputs.reshape([inputs.shape[0], inputs.shape[1] * inputs.shape[2], inputs.shape[3]])
        input = input.transpose(1,2)

        reaction_weight, diffusion_weight = self.reac_diff_weight_construct(ind)
        reaction_bias, diffusion_bias = self.reac_diff_bias_construct(ind)

        reaction = self.nconv(input, reaction_weight) + reaction_bias

        # heat diffusion kernel
        diffusion = self.nconv(input, diffusion_weight) + diffusion_bias

        # if self.has_self_loop:
        #     ind = [int(i/self.resolution) for i in ind]
        #     self_loop_bias = torch.mul(self.weight_self_loop[ind][:, :, None], input)
        #     return self.tanh(reaction) + diffusion + input + self_loop_bias
        # else:
        #     return self.tanh(reaction) + diffusion + input

        if self.has_self_loop:
            ind = [int(i/self.resolution) for i in ind]
            self_loop_bias = torch.mul(self.weight_self_loop[ind][:, :, None], input)
            result = self.tanh(reaction) + diffusion + input + self_loop_bias
            return result
        else:
            result = self.tanh(reaction) + diffusion + input
            return result.reshape(inputs.shape)

class diffusion_diffusion_gcn(nn.Module):
    def __init__(self, device, symetric, resolution, has_self_loop):
        super(diffusion_diffusion_gcn, self).__init__()

        self.device = device
        self.resolution = resolution
        print("resolution: ", self.resolution)
        self.nconv = nconv()

        self.symetric = symetric
        self.has_self_loop = has_self_loop

        df = pd.read_csv("data/metr-la/metrla-virtual-id-revised.csv")
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = 207

        # self.w_d = nn.Parameter((torch.randn(1, 307, 1) / 10).to(device), requires_grad=True).to(device)

        if symetric:
            self.index = [i+j, j+i] 
            self.weight_react = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 10).to(device), requires_grad=True).to(device)
            self.weight_diff = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 1000).to(device), requires_grad=True).to(device)

        else: 
            self.index = [i, j] 
            print("reaction and diffusion factors are not symetric")
            self.weight_react = nn.Parameter((torch.randn(int(288 * 2 / resolution), self.edge_size) / 10).to(device), requires_grad=True).to(device)
            self.weight_diff = nn.Parameter((torch.randn(int(288 * 2 / resolution), self.edge_size) / 1000).to(device), requires_grad=True).to(device)

        self.bias_reaction = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)
        self.bias_diffusion = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)

        if self.has_self_loop:
            self.weight_self_loop = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)


        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        

    def reac_diff_weight_construct(self, ind):
        ii = int(ind[0] / self.resolution)

        if self.symetric:
            reaction_weight = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_react[ii], self.weight_react[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
            I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
            reaction_weight =  I1 - reaction_weight
            # reaction_weight =  I1 - reaction_weight

            diffusion_weight = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_diff[ii], self.weight_diff[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
            I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
            diffusion_weight = I2 - diffusion_weight
        
        else: 
            reaction_weight = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
            I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
            reaction_weight =  I1 - reaction_weight
            # reaction_weight =  I1 - reaction_weight

            diffusion_weight = torch.sparse_coo_tensor(self.index, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
            I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
            diffusion_weight = I2 - diffusion_weight

        for i in range(1, len(ind)):
            ii = int(ind[i] / self.resolution)
            if self.symetric:
                reaction_element = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_react[ii], self.weight_react[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
                diffusion_element = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_diff[ii], self.weight_diff[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
            else:
                reaction_element = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
                diffusion_element = torch.sparse_coo_tensor(self.index, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]                

            # create laplacian
            I1 = torch.diag(torch.sum(reaction_element, 1)[0, :])[None, :, :]
            # not laplacian
            reaction_element = I1 - reaction_element
            # reaction_element = I1 - reaction_element
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

        if self.has_self_loop:
            ind = [int(i/self.resolution) for i in ind]
            self_loop_bias = torch.mul(self.weight_self_loop[ind][:, :, None], input)
            return self.tanh(reaction) + diffusion + input + self_loop_bias
        else:
            return self.tanh(reaction) + diffusion + input
        # return (reaction) + self.tanh(diffusion) + input
        
        # return self.tanh(reaction) + self.tanh(diffusion) + input
        # return diffusion + input

class diffusion_gcn(nn.Module):
    def __init__(self, device, symetric, resolution, has_self_loop):
        super(diffusion_gcn, self).__init__()

        self.device = device
        self.resolution = resolution
        print("resolution: ", self.resolution)
        self.nconv = nconv()

        self.symetric = symetric
        self.has_self_loop = has_self_loop

        df = pd.read_csv("data/metr-la/metrla-virtual-id-revised.csv")
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = 207

        

        # self.w_d = nn.Parameter((torch.randn(1, 307, 1) / 10).to(device), requires_grad=True).to(device)

        if symetric:
            self.index = [i+j, j+i] 
            # self.weight_react = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 10).to(device), requires_grad=True).to(device)
            self.weight_diff = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 1000).to(device), requires_grad=True).to(device)

        else: 
            self.index = [i, j] 
            print("reaction and diffusion factors are not symetric")
            # self.weight_react_a = nn.Parameter((torch.randn(int(288 * 2 / resolution), self.edge_size) / 10).to(device), requires_grad=True).to(device)
            self.weight_diff = nn.Parameter((torch.randn(int(288 * 2 / resolution), self.edge_size) / 1000).to(device), requires_grad=True).to(device)

        # self.bias_reaction = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)
        self.bias_diffusion = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)

        if self.has_self_loop:
            self.weight_self_loop = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)

        self.tanh = nn.Tanh()

    def diff_weight_construct(self, ind):
        ii = int(ind[0] / self.resolution)

        if self.symetric:
            # reaction_weight = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_react[ii], self.weight_react[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
            # I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
            # reaction_weight =  I1 + reaction_weight
            # reaction_weight =  I1 - reaction_weight

            diffusion_weight = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_diff[ii], self.weight_diff[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
            I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
            diffusion_weight = I2 - diffusion_weight
        
        else: 
            # reaction_weight = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
            # I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
            # reaction_weight =  I1 + reaction_weight
            # reaction_weight =  I1 - reaction_weight

            diffusion_weight = torch.sparse_coo_tensor(self.index, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
            I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
            diffusion_weight = I2 - diffusion_weight

        for i in range(1, len(ind)):
            ii = int(ind[i] / self.resolution)
            if self.symetric:
                # reaction_element = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_react[ii], self.weight_react[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
                diffusion_element = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_diff[ii], self.weight_diff[ii]), 0), (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
            else:
                # reaction_element = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
                diffusion_element = torch.sparse_coo_tensor(self.index, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]                

            # create laplacian
            # I1 = torch.diag(torch.sum(reaction_element, 1)[0, :])[None, :, :]
            # # not laplacian
            # reaction_element = I1 + reaction_element
            # reaction_element = I1 - reaction_element
            I2 = torch.diag(torch.sum(diffusion_element, 1)[0, :])[None, :, :]
            diffusion_element = I2 - diffusion_element

            # reaction_weight = torch.cat((reaction_weight, reaction_element), 0)
            diffusion_weight = torch.cat((diffusion_weight, diffusion_element), 0)
        return diffusion_weight.to(self.device)

    def diff_bias_construct(self, ind):
        ind = [int(i/self.resolution) for i in ind]
        # reaction_bias = self.bias_reaction[ind][:, :, None]
        diffusion_bias = self.bias_diffusion[ind][:, :, None]
        return diffusion_bias.to(self.device)

    def forward(self, inputs, ind):
        input = inputs[:, 0, :, :]
        # input = inputs.reshape([inputs.shape[0], inputs.shape[1] * inputs.shape[2], inputs.shape[3]])
        # input = input.transpose(1,2)

        diffusion_weight = self.diff_weight_construct(ind)
        diffusion_bias = self.diff_bias_construct(ind)

        # reaction = self.nconv(input, reaction_weight) + reaction_bias

        # heat diffusion kernel
        diffusion = self.nconv(input, diffusion_weight) + diffusion_bias

        # return (reaction) + self.tanh(diffusion) + input
        # return self.tanh(reaction) + diffusion + input
        # return self.tanh(reaction) + self.tanh(diffusion) + input
        if self.has_self_loop:
            ind = [int(i/self.resolution) for i in ind]
            self_loop_bias = torch.mul(self.weight_self_loop[ind][:, :, None], input)
            return diffusion + input + self_loop_bias
        else:
            result = diffusion + input
            # return result.reshape(inputs.shape)
            return result

class diffusion_gcn_distance_graph(nn.Module):
    def __init__(self, device, symetric, resolution):
        super(diffusion_gcn_distance_graph, self).__init__()

        self.device = device
        self.resolution = resolution
        print("resolution: ", self.resolution)
        self.nconv = nconv()

        self.symetric = symetric

        df = pd.read_csv("data/metr-la/adj_mat.csv")
        self.index = np.nonzero(df.values)

        self.edge_size = len(self.index[0])
        self.node_size = 207

        # self.w_d = nn.Parameter((torch.randn(1, 307, 1) / 10).to(device), requires_grad=True).to(device)


        # self.weight_react = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 10).to(device), requires_grad=True).to(device)
        self.weight_diff = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 1000).to(device), requires_grad=True).to(device)
        print("reaction and diffusion factors are not symetric")
        # self.weight_react_a = nn.Parameter((torch.randn(int(288 * 2 / resolution), self.edge_size) / 10).to(device), requires_grad=True).to(device)
        # self.weight_diff_a = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 1000).to(device), requires_grad=True).to(device)

        # self.bias_reaction = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)
        self.bias_diffusion = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)

        self.tanh = nn.Tanh()

    def diff_weight_construct(self, ind):
        ii = int(ind[0] / self.resolution)

        diffusion_weight = torch.sparse_coo_tensor(self.index, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
        I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
        diffusion_weight = I2 - diffusion_weight

        for i in range(1, len(ind)):
            ii = int(ind[i] / self.resolution)
            # reaction_element = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
            diffusion_element = torch.sparse_coo_tensor(self.index, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]                

            # create laplacian
            # I1 = torch.diag(torch.sum(reaction_element, 1)[0, :])[None, :, :]
            # # not laplacian
            # reaction_element = I1 + reaction_element
            # reaction_element = I1 - reaction_element
            I2 = torch.diag(torch.sum(diffusion_element, 1)[0, :])[None, :, :]
            diffusion_element = I2 - diffusion_element

            # reaction_weight = torch.cat((reaction_weight, reaction_element), 0)
            diffusion_weight = torch.cat((diffusion_weight, diffusion_element), 0)
        return diffusion_weight.to(self.device)

    def diff_bias_construct(self, ind):
        ind = [int(i/self.resolution) for i in ind]
        # reaction_bias = self.bias_reaction[ind][:, :, None]
        diffusion_bias = self.bias_diffusion[ind][:, :, None]
        return diffusion_bias.to(self.device)

    def forward(self, inputs, ind):
        
        input = inputs[:, 0, :, :]

        diffusion_weight = self.diff_weight_construct(ind)
        diffusion_bias = self.diff_bias_construct(ind)

        # reaction = self.nconv(input, reaction_weight) + reaction_bias

        # heat diffusion kernel
        diffusion = self.nconv(input, diffusion_weight) + diffusion_bias

        # return (reaction) + self.tanh(diffusion) + input
        # return self.tanh(reaction) + diffusion + input
        # return self.tanh(reaction) + self.tanh(diffusion) + input
        return diffusion + input

class single_GNN(nn.Module):
    def __init__(self, device, symetric=False):
        super(single_GNN, self).__init__()

        print("model is single layer GNN, Y=AWX, K=5")
        self.device = device
        # df = pd.read_csv("data/metr-la/metrla-virtual-id-revised.csv")
        df = pd.read_csv("data/pems-bay/pems-bay-virtual-id.csv")
        print("symetrix: ", symetric)
        i = df["from"].to_list()
        j = df["to"].to_list()
        # self.edge_size = 207
        self.edge_size = 281
        A = np.zeros((self.edge_size, self.edge_size))
        if symetric:
            for a in range(len(i)):
                A[i[a], j[a]] = 1
                A[j[a], i[a]] = 1
        else:
            for a in range(len(i)):
                A[i[a], j[a]] = 1
        self.A = A
        self.A_pow = np.matmul(A, A)
        K = 5
        for i in range(2, K):
            self.A_pow = np.matmul(self.A_pow, A)
        self.A_pow = torch.tensor(self.A_pow)        
        self.A_pow = torch.transpose(self.A_pow, 0, 1)
        self.A_pow = self.A_pow.repeat(64, 1, 1).to(device)
        print("Using A^T")

        self.theta = nn.Parameter((torch.randn(12, 1)).to(device), requires_grad=True).to(device)

        # df = pd.read_csv("data/metr-la/graph_sensor_locations.csv")
        df = pd.read_csv("data/pems-bay/filtered_location.csv")
        self.lat = df["latitude"].to_list()
        self.lng = df["longitude"].to_list()
        self.sensor_id = df["sensor_id"].to_list()

    def forward(self, inputs):
        inputs = inputs[:, 0, :, :]
        support = torch.bmm(inputs, self.theta.repeat(64, 1, 1))
        out = torch.bmm(self.A_pow, support.double())
        return out

class double_GNN(nn.Module):
    def __init__(self, device, symetric=False):
        super(double_GNN, self).__init__()

        self.device = device
        df = pd.read_csv("data/metr-la/metrla-virtual-id-revised.csv")
        i = df["from"].to_list()
        j = df["to"].to_list()
        self.edge_size = 207
        A = np.zeros((self.edge_size, self.edge_size))
        for a in range(len(j)):
            for b in range(len(i)):
                # A[i[a], j[b]] = 1
                A[j[a], i[b]] = 1
        self.A_pow = torch.tensor(np.matmul(A, A))
        self.A_pow = self.A_pow.repeat(64, 1, 1).to(device)
        self.theta1 = nn.Parameter((torch.randn(self.edge_size, self.edge_size)).to(device), requires_grad=True).to(device)
        self.theta2 = nn.Parameter((torch.randn(self.edge_size, self.edge_size)).to(device), requires_grad=True).to(device)

    def forward(self, inputs):
        inputs = inputs[:, 0, :, :]
        support = torch.mul(self.A_pow, self.theta1.repeat(64, 1, 1))
        # support = torch.bmm(inputs, self.theta1.repeat(64, 1, 1))
        out1 = torch.matmul(support.double(), inputs.double())

        support = torch.mul(self.A_pow, self.theta1.repeat(64, 1, 1))
        out2 = torch.matmul(support.double(), inputs.double())
        return out1 + out2 + inputs


class reaction_diffusion_nature_fml(nn.Module):
    def __init__(self, device, up_or_down, resolution):
        super(reaction_diffusion_nature_fml, self).__init__()

        self.device = device
        self.resolution = resolution
        print("resolution: ", self.resolution)
        self.nconv = nconv()

        # upper and lower can
        self.up = up_or_down[0]
        self.down = up_or_down[1]

        df = pd.read_csv("data/metr-la/metrla-virtual-id-revised.csv")
        # df = pd.read_csv("data/seattle-loop/seattle_loop_virtual_link.csv")
        # df = pd.read_csv("data/pems-bay/pems-bay-virtual-id.csv")
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = 207
        # self.node_size = 323
        # self.node_size = 281


        self.index = [i, j]
        self.index_a = [j, i] 

        self.weight_react = nn.Parameter((torch.randn(1, self.edge_size) / 10).to(device), requires_grad=True).to(device)
        self.weight_diff = nn.Parameter((torch.randn(1, self.edge_size) / 1000).to(device), requires_grad=True).to(device)
        self.bias_reaction = nn.Parameter((torch.randn(1, self.node_size) / 10).to(device), requires_grad=True).to(device)
        self.bias_diffusion = nn.Parameter((torch.randn(1, self.node_size) / 10).to(device), requires_grad=True).to(device)            

        print("reaction and diffusion factors are same as nature")
        # self.w_d = nn.Parameter((torch.randn(1, 307, 1) / 10).to(device), requires_grad=True).to(device)


        self.tanh = nn.Tanh()

    def reac_diff_weight_construct(self):
        ii = 0
    
        reaction_weight = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
        I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
        # reaction_weight =  I1 + reaction_weight
        reaction_weight =  I1 - reaction_weight

        diffusion_weight = torch.sparse_coo_tensor(self.index_a, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
        I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
        diffusion_weight = I2 - diffusion_weight

        return reaction_weight.repeat(64, 1, 1).to(self.device), diffusion_weight.repeat(64, 1, 1).to(self.device)

    def reac_diff_bias_construct(self):
        ind = [0]
        reaction_bias = self.bias_reaction[ind].T
        diffusion_bias = self.bias_diffusion[ind].T
        return reaction_bias.repeat(64, 1, 1).to(self.device), diffusion_bias.repeat(64, 1, 1).to(self.device)

    def forward(self, input):

        reaction_weight, diffusion_weight = self.reac_diff_weight_construct()
        reaction_bias, diffusion_bias = self.reac_diff_bias_construct()

        input = input[:, :, None]
        reaction = self.nconv(input, reaction_weight) + reaction_bias

        # heat diffusion kernel
        diffusion = self.nconv(input, diffusion_weight) + diffusion_bias

        # return (reaction) + self.tanh(diffusion) + input
        if(self.up and self.down):
            return (self.tanh(reaction) + (diffusion) + input)[:, :, 0]
        elif(self.up):
            return reaction + input
        elif(self.down):
            return diffusion + input
        else:
            print("bad input")
        # return self.tanh(reaction) + self.tanh(diffusion) + input
        # return diffusion + input

class reaction_diffusion_nature(nn.Module):
    def __init__(self, device, adj_path, num_node, up_or_down, resolution):
        super(reaction_diffusion_nature, self).__init__()

        self.device = device
        self.resolution = resolution
        print("resolution: ", self.resolution)
        self.nconv = nconv()

        # upper and lower can
        self.up = up_or_down[0]
        self.down = up_or_down[1]

        df = pd.read_csv(adj_path)
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = num_node
        # self.node_size = 323
        # self.node_size = 281


        self.index = [i, j]
        self.index_a = [j, i] 

        self.weight_react = nn.Parameter((torch.randn(1, self.edge_size) / 10).to(device), requires_grad=True).to(device)
        self.weight_diff = nn.Parameter((torch.randn(1, self.edge_size) / 1000).to(device), requires_grad=True).to(device)
        self.bias_reaction = nn.Parameter((torch.randn(1, self.node_size) / 10).to(device), requires_grad=True).to(device)
        self.bias_diffusion = nn.Parameter((torch.randn(1, self.node_size) / 10).to(device), requires_grad=True).to(device)            

        print("reaction and diffusion factors are same as nature")
        # self.w_d = nn.Parameter((torch.randn(1, 307, 1) / 10).to(device), requires_grad=True).to(device)


        self.tanh = nn.Tanh()

    def reac_diff_weight_construct(self):
        ii = 0
    
        reaction_weight = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
        I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
        # reaction_weight =  I1 + reaction_weight
        reaction_weight =  I1 - reaction_weight

        diffusion_weight = torch.sparse_coo_tensor(self.index_a, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
        I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
        diffusion_weight = I2 - diffusion_weight

        return reaction_weight.repeat(64, 1, 1).to(self.device), diffusion_weight.repeat(64, 1, 1).to(self.device)

    def reac_diff_bias_construct(self):
        ind = [0]
        reaction_bias = self.bias_reaction[ind].T
        diffusion_bias = self.bias_diffusion[ind].T
        return reaction_bias.repeat(64, 1, 1).to(self.device), diffusion_bias.repeat(64, 1, 1).to(self.device)

    # def reac_diff_weight_construct(self, ind):
    #     ii = int(ind[0] / self.resolution)
    
    #     reaction_weight = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
    #     I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
    #     # reaction_weight =  I1 + reaction_weight
    #     reaction_weight =  I1 - reaction_weight

    #     diffusion_weight = torch.sparse_coo_tensor(self.index_a, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
    #     I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
    #     diffusion_weight = I2 - diffusion_weight

    #     for i in range(1, len(ind)):
    #         ii = int(ind[i] / self.resolution)
    #         reaction_element = torch.sparse_coo_tensor(self.index, self.weight_react[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
    #         diffusion_element = torch.sparse_coo_tensor(self.index_a, self.weight_diff[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]         

    #         # create laplacian
    #         I1 = torch.diag(torch.sum(reaction_element, 1)[0, :])[None, :, :]
    #         # not laplacian
    #         # reaction_element = I1 + reaction_element
    #         reaction_element = I1 - reaction_element
    #         I2 = torch.diag(torch.sum(diffusion_element, 1)[0, :])[None, :, :]
    #         diffusion_element = I2 - diffusion_element

    #         reaction_weight = torch.cat((reaction_weight, reaction_element), 0)
    #         diffusion_weight = torch.cat((diffusion_weight, diffusion_element), 0)
    #     return reaction_weight.to(self.device), diffusion_weight.to(self.device)

    # def reac_diff_bias_construct(self, ind):
    #     ind = [int(i/self.resolution) for i in ind]
    #     reaction_bias = self.bias_reaction[ind][:, :, None]
    #     diffusion_bias = self.bias_diffusion[ind][:, :, None]
    #     return reaction_bias.to(self.device), diffusion_bias.to(self.device)

    def forward(self, t, input):

        t = t.to(self.device)

        reaction_weight, diffusion_weight = self.reac_diff_weight_construct()
        reaction_bias, diffusion_bias = self.reac_diff_bias_construct()

        input = input[:, :, None]
        reaction = self.nconv(input, reaction_weight) + reaction_bias

        # heat diffusion kernel
        diffusion = self.nconv(input, diffusion_weight) + diffusion_bias

        # return (reaction) + self.tanh(diffusion) + input
        if(self.up and self.down):
            return (self.tanh(reaction) + (diffusion) + input)[:, :, 0]
        elif(self.up):
            return reaction + input
        elif(self.down):
            return diffusion + input
        else:
            print("bad input")
        # return self.tanh(reaction) + self.tanh(diffusion) + input
        # return diffusion + input

class reaction_diffusion_nature_plus(nn.Module):
    def __init__(self, device, symetric, resolution):
        super(reaction_diffusion_nature_plus, self).__init__()

        self.device = device
        self.resolution = resolution
        print("resolution: ", self.resolution)
        self.nconv = nconv()

        self.symetric = symetric

        df = pd.read_csv("data/metr-la/metrla-virtual-id-revised.csv")
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = 207

        self.index = [i, j]
        self.index_a = [j, i] 

        print("reaction and diffusion factors are same as nature")
        # self.w_d = nn.Parameter((torch.randn(1, 307, 1) / 10).to(device), requires_grad=True).to(device)
        self.weight_react = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 10).to(device), requires_grad=True).to(device)
        self.weight_diff = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 1000).to(device), requires_grad=True).to(device)

        self.weight_react_a = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 10).to(device), requires_grad=True).to(device)
        self.weight_diff_a = nn.Parameter((torch.randn(int(288 / resolution), self.edge_size) / 1000).to(device), requires_grad=True).to(device)

        self.bias_reaction = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)
        self.bias_diffusion = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)

        self.bias_reaction_a = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)
        self.bias_diffusion_a = nn.Parameter((torch.randn(int(288 / resolution), self.node_size) / 10).to(device), requires_grad=True).to(device)

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

        reaction_weight_a = torch.sparse_coo_tensor(self.index, self.weight_react_a[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)
        I3 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
        # reaction_weight =  I1 + reaction_weight
        reaction_weight_a =  I3 + reaction_weight_a

        diffusion_weight_a = torch.sparse_coo_tensor(self.index_a, self.weight_diff_a[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :].to(self.device)       
        I4 = torch.diag(torch.sum(diffusion_weight_a, 1)[0, :])[None, :, :]
        diffusion_weight_a = I4 + diffusion_weight_a

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

            reaction_element_a = torch.sparse_coo_tensor(self.index, self.weight_react_a[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]
            diffusion_element_a = torch.sparse_coo_tensor(self.index_a, self.weight_diff_a[ii], (self.node_size, self.node_size), device = self.device).to_dense()[None, :]         

            # create laplacian
            I3 = torch.diag(torch.sum(reaction_element_a, 1)[0, :])[None, :, :]
            # not laplacian
            # reaction_element = I1 + reaction_element
            reaction_element_a = I3 + reaction_element_a
            I4 = torch.diag(torch.sum(diffusion_element_a, 1)[0, :])[None, :, :]
            diffusion_element_a = I4 + diffusion_element_a

            reaction_weight = torch.cat((reaction_weight, reaction_element), 0)
            diffusion_weight = torch.cat((diffusion_weight, diffusion_element), 0)
            reaction_weight_a = torch.cat((reaction_weight_a, reaction_element_a), 0)
            diffusion_weight_a = torch.cat((diffusion_weight_a, diffusion_element_a), 0)
        return reaction_weight.to(self.device), diffusion_weight.to(self.device), reaction_weight_a.to(self.device), diffusion_weight_a.to(self.device)

    def reac_diff_bias_construct(self, ind):
        ind = [int(i/self.resolution) for i in ind]
        reaction_bias = self.bias_reaction[ind][:, :, None]
        diffusion_bias = self.bias_diffusion[ind][:, :, None]
        reaction_bias_a = self.bias_reaction_a[ind][:, :, None]
        diffusion_bias_a = self.bias_diffusion_a[ind][:, :, None]
        return reaction_bias.to(self.device), diffusion_bias.to(self.device), reaction_bias_a.to(self.device), diffusion_bias_a.to(self.device)

    def forward(self, inputs, ind):
        
        input = inputs[:, 0, :, :]

        reaction_weight, diffusion_weight, reaction_weight_a, diffusion_weight_a = self.reac_diff_weight_construct(ind)
        reaction_bias, diffusion_bias, reaction_bias_a, diffusion_bias_a= self.reac_diff_bias_construct(ind)

        reaction = self.nconv(input, reaction_weight) + reaction_bias
        reaction_a = self.nconv(input, reaction_weight_a) + reaction_bias_a

        # heat diffusion kernel
        diffusion = self.nconv(input, diffusion_weight) + diffusion_bias
        diffusion_a = self.nconv(input, diffusion_weight_a) + diffusion_bias_a

        # return (reaction) + self.tanh(diffusion) + input
        return self.tanh(reaction) + self.tanh(reaction_a) + diffusion + diffusion_a + input
        # return self.tanh(reaction) + self.tanh(diffusion) + input
        # return diffusion + input


class multi_reaction_diffusion(nn.Module):
    def __init__(self, device, adj_path, num_node, up_or_down, resolution, enable_bias, num_sequence, time_samples, num_rd_kernels):
        super(multi_reaction_diffusion, self).__init__()

        self.device = device
        # self.time_samples = torch.cat((time_samples, torch.tensor([2]).to(self.device)), dim=0)
        self.time_samples = time_samples
        self.num_sequence = num_sequence
        self.num_rd_kernels = num_rd_kernels
        self.rdgcn_transient = reaction_diffusion_fast(device, adj_path, num_node, up_or_down, resolution, enable_bias, 0.9)
        self.rdgcn_equlibrium = reaction_diffusion_fast(device, adj_path, num_node, up_or_down, resolution, enable_bias, 0.1)
        # self.rdgcn_equlibrium.weight_diff.register_hook(lambda x: print('eq reac weight grad accumulated in fc1'))
        # self.rdgcn_transient.weight_diff.register_hook(lambda x: print('transient reac weight grad accumulated in fc1'))

        # self.rdgcn_zero = reaction_diffusion_fast(device, adj_path, num_node, up_or_down, resolution, enable_bias)
        # self.rdgcn_0 = reaction_diffusion_fast(device, adj_path, num_node, up_or_down, resolution, enable_bias)
        # self.rdgcn_1 = reaction_diffusion_fast(device, adj_path, num_node, up_or_down, resolution, enable_bias)

        modulelist = []
        if (num_rd_kernels - 2 > 0):
            self.num_ex_kernels = num_rd_kernels - 2
        else:
            self.num_ex_kernels = 0
        for i in range(self.num_ex_kernels):
            modulelist.append(reaction_diffusion_fast(device, adj_path, num_node, up_or_down, resolution, enable_bias, 10/num_rd_kernels*i))
        self.modulelist = nn.ModuleList(modulelist)
        self.l = 0
        self.jump = 0
    
    def forward(self, t, input):
        # print("input: ", input)
        # t = t % 1.9
        if (t > self.time_samples[-2]):
            k = 12
        else:
            k = next(x for x, val in enumerate(self.time_samples)
                                    if val > t)
        
        # z = self.l[:, (k-1), :, 0]
        # output = z*self.rdgcn_transient(input) + (1-z)*self.rdgcn_equlibrium(input)
        
        # z = self.l[:, int(t * (self.num_sequence-2))%11, :, 0]
        # if (k > 11):
        #     print(k, t)

        z = self.l[:, k-1, :, :]
        # z = self.l[:, int((k-1)/13*11), :, :]
        output = 0
        # for i in range(self.num_ex_kernels):
        #     output = output + z[..., i] * self.modulelist[i](input)
        output = output + z[..., -2] * self.rdgcn_transient(input) + z[..., -1] * self.rdgcn_equlibrium(input) + self.jump[:, k-1, :, 0]
        # output = z[..., 0]*self.rdgcn_transient(input) + z[..., 1]*self.rdgcn_equlibrium(input) + z[..., 2]*self.rdgcn_zero(input) + z[..., 3]*self.rdgcn_0(input)  + z[..., 4]*self.rdgcn_1(input)
        # output = z[..., 0]*self.rdgcn_transient(input) + (1-z[..., 0])*self.rdgcn_equlibrium(input)
        return output

class multi_reaction_diffusion_concat_out(nn.Module):
    def __init__(self, device, adj_path, num_node, up_or_down, resolution, enable_bias, num_sequence, time_samples, num_rd_kernels):
        super(multi_reaction_diffusion_concat_out, self).__init__()

        self.device = device
        # self.time_samples = torch.cat((time_samples, torch.tensor([2]).to(self.device)), dim=0)
        self.time_samples = time_samples
        self.num_sequence = num_sequence
        self.num_rd_kernels = num_rd_kernels
        self.rdgcn_transient = reaction_diffusion_fast(device, adj_path, num_node, up_or_down, resolution, enable_bias, 0.9)
        self.rdgcn_equlibrium = reaction_diffusion_fast(device, adj_path, num_node, up_or_down, resolution, enable_bias, 0.1)
        # self.rdgcn_zero = reaction_diffusion_fast(device, adj_path, num_node, up_or_down, resolution, enable_bias)
        # self.rdgcn_0 = reaction_diffusion_fast(device, adj_path, num_node, up_or_down, resolution, enable_bias)
        # self.rdgcn_1 = reaction_diffusion_fast(device, adj_path, num_node, up_or_down, resolution, enable_bias)

        modulelist = []
        if (num_rd_kernels - 2 > 0):
            self.num_ex_kernels = num_rd_kernels - 2
        else:
            self.num_ex_kernels = 0
        for i in range(self.num_ex_kernels):
            modulelist.append(reaction_diffusion_fast(device, adj_path, num_node, up_or_down, resolution, enable_bias, 10/num_rd_kernels*i))
        self.modulelist = nn.ModuleList(modulelist)
        self.l = 0
        self.jump = 0
    
    def forward(self, t, input):

        outputs = []
        for i in range(self.num_ex_kernels):
            outputs.append(self.modulelist[i](input)) 
        outputs.append(self.rdgcn_transient(input)) 
        outputs.append(self.rdgcn_transient(input))

        return torch.cat(outputs, dim=-1)

class reaction_diffusion_fast(nn.Module):
    def __init__(self, device, adj_path, num_node, up_or_down, resolution, enable_bias, scale):
        super(reaction_diffusion_fast, self).__init__()

        self.device = device
        self.resolution = resolution
        print("resolution: ", self.resolution)
        self.nconv = nconv() 

        # upper and lower can
        self.up = up_or_down[0]
        self.down = up_or_down[1]

        df = pd.read_csv(adj_path)
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = num_node
        # self.node_size = 323
        # self.node_size = 281


        self.index = [i, j]
        self.index_a = [j, i] 
        self.mask = np.zeros((self.node_size, self.node_size))
        for a in range(len(i)):
            self.mask[i[a], j[a]] = 1
        self.mask = torch.tensor(self.mask).to(device)
        self.mask_A = torch.transpose(self.mask, 0, 1).to(device)

        self.weight_react = nn.Parameter((scale * torch.randn(self.node_size, self.node_size) / 10).to(device), requires_grad=True).to(device)
        self.weight_diff = nn.Parameter((scale * torch.randn(self.node_size, self.node_size) / 10).to(device), requires_grad=True).to(device)
        if (enable_bias):
            self.bias_reaction = nn.Parameter((scale * torch.randn(self.node_size, 1) / 10).to(device), requires_grad=True).to(device)
            self.bias_diffusion = nn.Parameter((scale * torch.randn(self.node_size, 1) / 10).to(device), requires_grad=True).to(device)  
        else:
            self.bias_reaction = torch.zeros((self.node_size, 1))
            self.bias_diffusion = torch.zeros((self.node_size, 1))   

        print("reaction and diffusion factors are same as nature")
        # self.w_d = nn.Parameter((torch.randn(1, 307, 1) / 10).to(device), requires_grad=True).to(device)


        self.tanh = nn.Tanh()

    def forward(self, input):

        # t = t.to(self.device)

        reaction_element, diffusion_element = torch.mul(self.weight_react, self.mask), torch.mul(self.weight_diff, self.mask_A)

        I1 = torch.diag(torch.sum(reaction_element, 1))
        # not laplacian
        # reaction_element = I1 + reaction_element
        reaction_element = I1 - reaction_element
        I2 = torch.diag(torch.sum(diffusion_element, 1))
        diffusion_element = I2 - diffusion_element
        # reaction_weight = reaction_element[None, :, :].repeat(64, 1, 1).to(self.device) 
        # diffusion_weight = diffusion_element[None, :, :].repeat(64, 1, 1).to(self.device)
        # reaction_bias = self.bias_reaction[None, :, :].repeat(64, 1, 1).to(self.device)
        # diffusion_bias = self.bias_diffusion[None, :, :].repeat(64, 1, 1).to(self.device)
        # reaction_weight = reaction_weight.type(torch.cuda.FloatTensor)
        # diffusion_weight = diffusion_weight.type(torch.cuda.FloatTensor)

        # input = input[:, :, None]

        # reaction = self.nconv(input, reaction_weight) + reaction_bias
        # diffusion = self.nconv(input, diffusion_weight) + diffusion_bias

        input = input.type(torch.cuda.DoubleTensor)
        reaction = torch.matmul(reaction_element, input[:, :, None]) + self.bias_reaction
        diffusion = torch.matmul(diffusion_element, input[:, :, None]) + self.bias_diffusion
        if(self.up and self.down):
            # return 0
            return (self.tanh(reaction) + (diffusion))[:, :, 0]
        elif(self.up):
            return reaction
        elif(self.down):
            return diffusion
        else:
            print("bad input")
        # return self.tanh(reaction) + self.tanh(diffusion) + input
        # return diffusion + input


class reaction_diffusion_fast_ode(nn.Module):
    def __init__(self, device, adj_path, num_node, up_or_down, resolution, enable_bias, scale):
        super(reaction_diffusion_fast_ode, self).__init__()

        self.device = device
        self.resolution = resolution
        print("resolution: ", self.resolution)
        self.nconv = nconv() 

        # upper and lower can
        self.up = up_or_down[0]
        self.down = up_or_down[1]

        df = pd.read_csv(adj_path)
        i = df["from"].to_list()
        j = df["to"].to_list()

        self.edge_size = len(i)
        self.node_size = num_node
        # self.node_size = 323
        # self.node_size = 281


        self.index = [i, j]
        self.index_a = [j, i] 
        self.mask = np.zeros((self.node_size, self.node_size))
        for a in range(len(i)):
            self.mask[i[a], j[a]] = 1
        self.mask = torch.tensor(self.mask).to(device)
        self.mask_A = torch.transpose(self.mask, 0, 1).to(device)

        self.weight_react = nn.Parameter((scale * torch.randn(self.node_size, self.node_size) / 10).to(device), requires_grad=True).to(device)
        self.weight_diff = nn.Parameter((scale * torch.randn(self.node_size, self.node_size) / 10).to(device), requires_grad=True).to(device)
        if (enable_bias):
            self.bias_reaction = nn.Parameter((scale * torch.randn(self.node_size, 1) / 10).to(device), requires_grad=True).to(device)
            self.bias_diffusion = nn.Parameter((scale * torch.randn(self.node_size, 1) / 10).to(device), requires_grad=True).to(device)  
        else:
            self.bias_reaction = torch.zeros((self.node_size, 1))
            self.bias_diffusion = torch.zeros((self.node_size, 1))   

        print("reaction and diffusion factors are same as nature")
        # self.w_d = nn.Parameter((torch.randn(1, 307, 1) / 10).to(device), requires_grad=True).to(device)


        self.tanh = nn.Tanh()

    def forward(self, t, input):

        # t = t.to(self.device)

        reaction_element, diffusion_element = torch.mul(self.weight_react, self.mask), torch.mul(self.weight_diff, self.mask_A)

        I1 = torch.diag(torch.sum(reaction_element, 1))
        # not laplacian
        # reaction_element = I1 + reaction_element
        reaction_element = I1 - reaction_element
        I2 = torch.diag(torch.sum(diffusion_element, 1))
        diffusion_element = I2 - diffusion_element
        # reaction_weight = reaction_element[None, :, :].repeat(64, 1, 1).to(self.device) 
        # diffusion_weight = diffusion_element[None, :, :].repeat(64, 1, 1).to(self.device)
        # reaction_bias = self.bias_reaction[None, :, :].repeat(64, 1, 1).to(self.device)
        # diffusion_bias = self.bias_diffusion[None, :, :].repeat(64, 1, 1).to(self.device)
        # reaction_weight = reaction_weight.type(torch.cuda.FloatTensor)
        # diffusion_weight = diffusion_weight.type(torch.cuda.FloatTensor)

        # input = input[:, :, None]

        # reaction = self.nconv(input, reaction_weight) + reaction_bias
        # diffusion = self.nconv(input, diffusion_weight) + diffusion_bias

        input = input.type(torch.cuda.DoubleTensor)
        reaction = torch.matmul(reaction_element, input[:, :, None]) + self.bias_reaction
        diffusion = torch.matmul(diffusion_element, input[:, :, None]) + self.bias_diffusion
        if(self.up and self.down):
            # return 0
            return (self.tanh(reaction) + (diffusion))[:, :, 0]
        elif(self.up):
            return reaction
        elif(self.down):
            return diffusion
        else:
            print("bad input")
        # return self.tanh(reaction) + self.tanh(diffusion) + input
        # return diffusion + input

class viscoelastic(nn.Module):
    def __init__(self, device, adj_path, num_node, up_or_down, resolution, enable_bias, scale):
        super(viscoelastic, self).__init__()

        self.device = device
        self.node_size = num_node
        self.p = nn.Parameter((scale * torch.randn(self.node_size, 1)).to(device), requires_grad=True).to(device)
        self.q = nn.Parameter((scale * torch.randn(self.node_size, 1)).to(device), requires_grad=True).to(device)

        self.h = nn.Parameter((scale * torch.randn(self.node_size, 1)).to(device), requires_grad=True).to(device)
        self.b2 = nn.Parameter((scale * torch.randn(self.node_size, 1)).to(device), requires_grad=True).to(device)  
 

        print("reaction and diffusion factors are same as nature")
        # self.w_d = nn.Parameter((torch.randn(1, 307, 1) / 10).to(device), requires_grad=True).to(device)

        self.M = 12

        self.tanh = nn.Tanh()

    def forward(self, t, input):

        # t = t.to(self.device)

        input = input.type(torch.cuda.DoubleTensor)
        coef1 = torch.div(-self.q, self.M)
        coef2 = (self.q - self.p)
        coef3 = torch.mul(self.p, self.M)

        # r1 = torch.mul(input, input) * 100
        r2 = torch.mul(coef2, input) + coef3 + self.b2 + torch.mul(coef1, input)
        return r2 * (((t+0.1)) ** self.h) 
        # return self.tanh(reaction) + self.tanh(diffusion) + input
        # return diffusion + input 

class second_layer_linear(nn.Module):
    def __init__(self, device, adj_path, num_node, up_or_down, resolution, enable_bias, scale):
        super(second_layer_linear, self).__init__()

        self.device = device
        self.node_size = num_node
        self.w1 = nn.Parameter((scale * torch.randn(self.node_size, 1)).to(device), requires_grad=True).to(device)
        self.w2 = nn.Parameter((scale * torch.randn(self.node_size, 1)).to(device), requires_grad=True).to(device)

        self.b1 = nn.Parameter((scale * torch.randn(self.node_size, 1) / 10).to(device), requires_grad=True).to(device)
        self.b2 = nn.Parameter((scale * torch.randn(self.node_size, 1) / 10).to(device), requires_grad=True).to(device)  
 

        print("reaction and diffusion factors are same as nature")
        # self.w_d = nn.Parameter((torch.randn(1, 307, 1) / 10).to(device), requires_grad=True).to(device)


        self.tanh = nn.Tanh()

    def forward(self, t, input):

        # t = t.to(self.device)

        input = input.type(torch.cuda.DoubleTensor)
        r1 = torch.mul(self.w1, input) + self.b1
        r2 = torch.mul(self.w2, r1) + self.b2
        return r2
        # return self.tanh(reaction) + self.tanh(diffusion) + input
        # return diffusion + input viscoelastic

####################################################################################################################################
class reaction_gcn(nn.Module):
    def __init__(self, device, symetric, resolution):
        super(reaction_gcn, self).__init__()

        self.device = device
        self.resolution = resolution
        self.nconv = nconv()

        self.symetric = symetric

        df = pd.read_csv("data/PEMSD4/PEMS04.csv")
        i = df["from"].to_list()
        j = df["to"].to_list()
        self.index = [i+j, j+i]

        self.weight_react = nn.Parameter((torch.randn(int(288 / resolution), 340) / 10).to(device), requires_grad=True).to(device)
        # self.weight_diff = nn.Parameter((torch.randn(int(288 / resolution), 340) / 1000).to(device), requires_grad=True).to(device)

        if not symetric:
            print("reaction and diffusion factors are not symetric")
            self.weight_react_a = nn.Parameter((torch.randn(int(288/ resolution), 340) / 10).to(device), requires_grad=True).to(device)
            # self.weight_diff_a = nn.Parameter((torch.randn(int(288/ resolution), 340) / 1000).to(device), requires_grad=True).to(device)

        self.bias_reaction = nn.Parameter((torch.randn(int(288 / resolution), 307) / 10).to(device), requires_grad=True).to(device)
        # self.bias_diffusion = nn.Parameter((torch.randn(int(288 / resolution), 307) / 10).to(device), requires_grad=True).to(device)

        self.tanh = nn.Tanh()

    def reac_diff_weight_construct(self, ind):
        ii = int(ind[0] / self.resolution)

        if self.symetric:
            reaction_weight = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_react[ii], self.weight_react[ii]), 0), (307, 307), device = self.device).to_dense()[None, :].to(self.device)
            I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
            reaction_weight =  I1 - reaction_weight

            # diffusion_weight = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_diff[ii], self.weight_diff[ii]), 0), (307, 307), device = self.device).to_dense()[None, :].to(self.device)       
            # I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
            # diffusion_weight = I2 - diffusion_weight
        
        else: 
            reaction_weight = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_react[ii], self.weight_react_a[ii]), 0), (307, 307), device = self.device).to_dense()[None, :].to(self.device)
            I1 = torch.diag(torch.sum(reaction_weight, 1)[0, :])[None, :, :]
            reaction_weight =  I1 - reaction_weight

            # diffusion_weight = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_diff[ii], self.weight_diff_a[ii]), 0), (307, 307), device = self.device).to_dense()[None, :].to(self.device)       
            # I2 = torch.diag(torch.sum(diffusion_weight, 1)[0, :])[None, :, :]
            # diffusion_weight = I2 - diffusion_weight

        for i in range(1, len(ind)):
            ii = int(ind[i] / self.resolution)
            if self.symetric:
                reaction_element = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_react[ii], self.weight_react[ii]), 0), (307, 307), device = self.device).to_dense()[None, :]
                # diffusion_element = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_diff[ii], self.weight_diff[ii]), 0), (307, 307), device = self.device).to_dense()[None, :]
            else:
                reaction_element = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_react[ii], self.weight_react_a[ii]), 0), (307, 307), device = self.device).to_dense()[None, :]
                # diffusion_element = torch.sparse_coo_tensor(self.index, torch.cat((self.weight_diff[ii], self.weight_diff_a[ii]), 0), (307, 307), device = self.device).to_dense()[None, :]                

            # create laplacian
            I1 = torch.diag(torch.sum(reaction_element, 1)[0, :])[None, :, :]
            reaction_element = I1 - reaction_element
            # I2 = torch.diag(torch.sum(diffusion_element, 1)[0, :])[None, :, :]
            # diffusion_element = I2 - diffusion_element

            reaction_weight = torch.cat((reaction_weight, reaction_element), 0)
            # diffusion_weight = torch.cat((diffusion_weight, diffusion_element), 0)
        return reaction_weight.to(self.device)

    def reac_diff_bias_construct(self, ind):
        ind = [int(i/self.resolution) for i in ind]
        reaction_bias = self.bias_reaction[ind][:, :, None]
        return reaction_bias.to(self.device)

    def forward(self, inputs, ind):
        
        input = inputs[:, 0, :, :]

        reaction_weight = self.reac_diff_weight_construct(ind)
        reaction_bias = self.reac_diff_bias_construct(ind)

        reaction = self.nconv(input, reaction_weight) + reaction_bias

        return self.tanh(reaction) + input

class Assembler(nn.Module):
    def __init__(self, device, input_sequence=3, output_sequence=1, resolution=288):
        super(Assembler, self).__init__()
        self.device = device
        self.resolution = resolution
        self.weight = nn.Parameter(torch.randn(int(288/ resolution), input_sequence, output_sequence)).to(device)

    def forward(self, inputs, ind):
        ind = [int(i/self.resolution) for i in ind]
        temp_weight = self.weight[ind]
        return torch.matmul(inputs, temp_weight)



# class DMSTGCN(nn.Module):
#     def __init__(self, device, num_nodes, dropout=0.3,
#                  out_dim=12, residual_channels=16, dilation_channels=16, end_channels=512,
#                  kernel_size=2, blocks=4, layers=2, days=288, dims=40, order=2, in_dim=9, normalization="batch"):
#         super(DMSTGCN, self).__init__()
#         skip_channels = 8
#         self.dropout = dropout
#         self.blocks = blocks
#         self.layers = layers

#         self.filter_convs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#         self.residual_convs = nn.ModuleList()
#         self.skip_convs = nn.ModuleList()
#         self.normal = nn.ModuleList()
#         self.gconv = nn.ModuleList()

#         self.filter_convs_a = nn.ModuleList()
#         self.gate_convs_a = nn.ModuleList()
#         self.residual_convs_a = nn.ModuleList()
#         self.skip_convs_a = nn.ModuleList()
#         self.normal_a = nn.ModuleList()
#         self.gconv_a = nn.ModuleList()

#         self.gconv_a2p = nn.ModuleList()

#         self.start_conv_a = nn.Conv2d(in_channels=in_dim,
#                                       out_channels=residual_channels,
#                                       kernel_size=(1, 1))

#         self.start_conv = nn.Conv2d(in_channels=in_dim,
#                                     out_channels=residual_channels,
#                                     kernel_size=(1, 1))

#         receptive_field = 1

#         self.supports_len = 1
#         self.nodevec_p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
#         self.nodevec_p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
#         self.nodevec_p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
#         self.nodevec_pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
#         self.nodevec_a1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
#         self.nodevec_a2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
#         self.nodevec_a3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
#         self.nodevec_ak = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
#         self.nodevec_a2p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
#         self.nodevec_a2p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
#         self.nodevec_a2p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
#         self.nodevec_a2pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)

#         for b in range(blocks):
#             additional_scope = kernel_size - 1
#             new_dilation = 1
#             for i in range(layers):
#                 # dilated convolutions
#                 self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
#                                                    out_channels=dilation_channels,
#                                                    kernel_size=(1, kernel_size), dilation=new_dilation))

#                 self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
#                                                  out_channels=dilation_channels,
#                                                  kernel_size=(1, kernel_size), dilation=new_dilation))

#                 self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
#                                                      out_channels=residual_channels,
#                                                      kernel_size=(1, 1)))

#                 self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
#                                                  out_channels=skip_channels,
#                                                  kernel_size=(1, 1)))

#                 self.filter_convs_a.append(nn.Conv2d(in_channels=residual_channels,
#                                                      out_channels=dilation_channels,
#                                                      kernel_size=(1, kernel_size), dilation=new_dilation))

#                 self.gate_convs_a.append(nn.Conv1d(in_channels=residual_channels,
#                                                    out_channels=dilation_channels,
#                                                    kernel_size=(1, kernel_size), dilation=new_dilation))

#                 # 1x1 convolution for residual connection
#                 self.residual_convs_a.append(nn.Conv1d(in_channels=dilation_channels,
#                                                        out_channels=residual_channels,
#                                                        kernel_size=(1, 1)))
#                 if normalization == "batch":
#                     self.normal.append(nn.BatchNorm2d(residual_channels))
#                     self.normal_a.append(nn.BatchNorm2d(residual_channels))
#                 elif normalization == "layer":
#                     self.normal.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
#                     self.normal_a.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
#                 new_dilation *= 2
#                 receptive_field += additional_scope
#                 additional_scope *= 2
#                 self.gconv.append(
#                     gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
#                 self.gconv_a.append(
#                     gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
#                 self.gconv_a2p.append(
#                     gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))

#         self.relu = nn.ReLU(inplace=True)

#         self.end_conv_1 = nn.Conv2d(in_channels=skip_channels * (12 + 10 + 9 + 7 + 6 + 4 + 3 + 1),
#                                     out_channels=end_channels,
#                                     kernel_size=(1, 1),
#                                     bias=True)

#         self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
#                                     out_channels=out_dim,
#                                     kernel_size=(1, 1),
#                                     bias=True)

#         self.receptive_field = receptive_field

#     def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
#         adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
#         adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
#         adp = torch.einsum('ck, abk->abc', target_embedding, adp)
#         adp = F.softmax(F.relu(adp), dim=2)
#         return adp

#     def forward(self, inputs, ind):
#         """
#         input: (B, F, N, T)
#         """
#         in_len = inputs.size(3)
#         if in_len < self.receptive_field:
#             xo = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
#         else:
#             xo = inputs
#         x = self.start_conv(xo[:, [0]])
#         x_a = self.start_conv_a(xo[:, [1]])
#         skip = 0

#         # dynamic graph construction
#         adp = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
#         adp_a = self.dgconstruct(self.nodevec_a1[ind], self.nodevec_a2, self.nodevec_a3, self.nodevec_ak)
#         adp_a2p = self.dgconstruct(self.nodevec_a2p1[ind], self.nodevec_a2p2, self.nodevec_a2p3, self.nodevec_a2pk)

#         new_supports = [adp]
#         new_supports_a = [adp_a]
#         new_supports_a2p = [adp_a2p]

#         for i in range(self.blocks * self.layers):
#             # tcn for primary part
#             residual = x
#             filter = self.filter_convs[i](residual)
#             filter = torch.tanh(filter)
#             gate = self.gate_convs[i](residual)
#             gate = torch.sigmoid(gate)
#             x = filter * gate

#             # tcn for auxiliary part
#             residual_a = x_a
#             filter_a = self.filter_convs_a[i](residual_a)
#             filter_a = torch.tanh(filter_a)
#             gate_a = self.gate_convs_a[i](residual_a)
#             gate_a = torch.sigmoid(gate_a)
#             x_a = filter_a * gate_a

#             # skip connection
#             s = x
#             s = self.skip_convs[i](s)
#             if isinstance(skip, int):  # B F N T
#                 skip = s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]).contiguous()
#             else:
#                 skip = torch.cat([s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]), skip], dim=1).contiguous()

#             # dynamic graph convolutions
#             x = self.gconv[i](x, new_supports)
#             x_a = self.gconv_a[i](x_a, new_supports_a)

#             # multi-faceted fusion module
#             x_a2p = self.gconv_a2p[i](x_a, new_supports_a2p)
#             x = x_a2p + x

#             # residual and normalization
#             x_a = x_a + residual_a[:, :, :, -x_a.size(3):]
#             x = x + residual[:, :, :, -x.size(3):]
#             x = self.normal[i](x)
#             x_a = self.normal_a[i](x_a)

#         # output layer
#         x = F.relu(skip)
#         x = F.relu(self.end_conv_1(x))
#         x = self.end_conv_2(x)
#         return x
