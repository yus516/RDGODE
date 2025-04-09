from multi_layer_reaction_diffusion import ReactionDiffusionODE

import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import util
import torch

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, normalization, lrate, wdecay, device, days=288,
                 dims=40, order=2, resolution=12, time_point=-1, zero_=-3):
        # self.model = diffusion_gcn(device, False, resolution, False)
        # self.model = reaction_diffusion_nature(device, [True, True], resolution)
        self.model = ReactionDiffusionODE(device, resolution)
        # self.model = diffusion_gcn_distance_graph(device, False, resolution)
        # self.model = reaction_diffusion_nature(device, [True, True], resolution)
        # self.model = reaction_diffusion_gcn(device, True, resolution, True)
        # self.model = gated_reaction_diffusion_gcn(device, True, resolution, False)
        # self.model = gated_diffusion_diffusion_gcn(device, True, resolution, False)

        self.model.to(device)
        self.time_point = time_point
        self.predict_point = -1
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaeduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=10, threshold=1e-3,
                                                              min_lr=1e-5, verbose=True)
        print("predicting the traffic speed in " + str(5+5*self.time_point) + ' min.')
        if (self.time_point > -1 ):
            self.loss = util.masked_mae_one_point
        else:
            self.loss = util.masked_mae
        print("the loss function is masked mae")
        # self.loss = util.masked_mse
        # print("the loss function is masked mse")
        self.scaler = scaler
        self.clip = 5

        self.zero_ = zero_

    def train(self, input, real_val, ind):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input, (1, 0, 0, 0))
        # input = input[:, :, :, -1][:, :, :, None]

        # output = self.model(input)
        output = self.model(input, ind)
        output = output.transpose(0, 1).transpose(1,2)[:, None, :, :]
        # real = torch.unsqueeze(real_val, dim=1)
        real = real_val
        predict = self.scaler.inverse_transform(output)

        x_mask = (input[:, :, :, -1][:, :, :, None] >= self.zero_)
        x_mask = x_mask.float()
        x_mask /= torch.mean((x_mask))
        x_mask = torch.where(torch.isnan(x_mask), torch.zeros_like(x_mask), x_mask) 

        loss = self.loss(predict, real, 0.0, x_mask, predict_point=-1, label_point=self.time_point)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mae = self.loss(predict, real, 0.0, x_mask, predict_point=self.predict_point, label_point=self.time_point).item()
        mape = util.masked_mape(predict, real, 0.0, x_mask).item()
        rmse = util.masked_rmse(predict, real, 0.0, x_mask).item()
        return mae, mape, rmse

    # let the model diff 1 times
    def eval(self, input, real_val, ind):
        self.model.eval()
        # input = input[:, :, :, -1][:, :, :, None]
        output = self.model(input, ind)
        # output = self.model(input)
        outputs = output.transpose(0, 1).transpose(1,2)[:, None, :, :]

        # generate training mask
        x_mask = (input[:, :, :, -1][:, :, :, None] >= self.zero_)
        x_mask = x_mask.float()
        x_mask /= torch.mean((x_mask))
        x_mask = torch.where(torch.isnan(x_mask), torch.zeros_like(x_mask), x_mask)
        x_mask = x_mask[:, :, :, None]
        x_mask = torch.tile(x_mask, (1, 1, 1, 12))

        predict = self.scaler.inverse_transform(outputs)
        # real = torch.unsqueeze(real_val, dim=1)
        real = real_val
        # real = real[:, :, :, self.predict_point][:, :, :, None]
        mae = self.loss(predict, real, 0.0, x_mask, predict_point=self.predict_point, label_point=self.time_point).item()
        mape = util.masked_mape(predict, real, 0.0, x_mask).item()
        rmse = util.masked_rmse(predict, real, 0.0, x_mask).item()


        # predict = self.scaler.inverse_transform(outputs)
        # real = torch.unsqueeze(real_val, dim=1)
        # mae = util.masked_mae(predict, real, 0.0, zero_).item()
        # mape = util.masked_mape(predict, real, 0.0, zero_).item()
        # rmse = util.masked_rmse(predict, real, 0.0, zero_).item()
        return mae, mape, rmse