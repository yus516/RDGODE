import torch.optim as optim
import torch
import util
from multi_layer_reaction_diffusion import JumpReactionDiffusionODE, JumpReactionDiffusionODEV2
import numpy as np

# the max_speed function doesn't work for trained gate
class trainer():
    def __init__(self, scaler, adj_path, num_node, lrate, wdecay, device, resolution=12, num_sequence=2, num_output_sequence=-1, enable_bias=False, predict_point=0, zero_=-3,
                 num_rd_kernels=2, temperture=1, fixed_gate=[True, True], max_speed=0):
        self.model = JumpReactionDiffusionODEV2(device, adj_path, num_node, fixed_gate, resolution, num_sequence, num_output_sequence, enable_bias, num_rd_kernels, temperture)
        self.device = device
        self.model.to(device)
        self.predict_point = predict_point
        print("predicting the traffic speed in " + str(5+5*self.predict_point) + ' min.')
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaeduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=10, threshold=1e-3,
                                                              min_lr=1e-5, verbose=True)
        self.loss = util.masked_mae
        print("the loss function is masked mae")
        # self.loss = util.masked_mse
        # print("the loss function is masked mse")
        self.scaler = scaler
        self.clip = 5
        self.kl_loss  = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.step_size = 3
        self.task_level = 1
        self.seq_out_len = 12
        self.max_speed = max_speed

        self.zero_ = zero_

    def train(self, input, real_val, gate, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(input)
        output = output.transpose(0, 1)[:, :, :, None]
        if (self.predict_point < 0):
            real = real_val.transpose(1, 2)[:, :, :, None]
        else:
            real = torch.unsqueeze(real_val, dim=1)[:, :, :, self.predict_point][:, :, :, None]
        predict = self.scaler.inverse_transform(output)

        x_mask = util.generate_mask(input[:, :, :, -1:], self.zero_)

        if self.task_level<=self.seq_out_len:
            if epoch%self.step_size==0:
                self.task_level +=1
            loss = self.loss(predict[:, :self.task_level, :, :], real[:, :self.task_level, :, :], 0.0, x_mask)
        else:

            # diff = torch.mean(torch.abs(self.model.odeFunc.rdgcn_transient.weight_react - self.model.odeFunc.rdgcn_equlibrium.weight_react) + \
            #                 torch.abs(self.model.odeFunc.rdgcn_transient.weight_diff - self.model.odeFunc.rdgcn_equlibrium.weight_diff)) / 2
            loss = self.loss(predict, real, 0.0, x_mask) # + 1/(diff)
        # diff = (torch.mean(self.model.odeFunc.modulelist[0].weight_react)) + (torch.mean(self.model.odeFunc.modulelist[0].weight_diff)) 
                # self.kl_loss(self.model.odeFunc.rdgcn_transient.weight_react, self.model.odeFunc.rdgcn_equlibrium.weight_react) + \
                # self.kl_loss(self.model.odeFunc.rdgcn_transient.weight_diff, self.model.odeFunc.rdgcn_equlibrium.weight_diff)

         # + 1/(diff) #  diff
        # print("current loss: ", loss)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mae = util.masked_mae(predict, real, 0.0, x_mask).item()
        mape = util.masked_mape(predict, real, 0.0, x_mask).item()
        rmse = util.masked_rmse(predict, real, 0.0, x_mask).item()
        return mae, mape, rmse
    
    # def train_gate(self, input, real_val, gate, epoch):
    #     self.model.train()
    #     self.optimizer.zero_grad()

    #     output = self.model.FC_1(input)
    #     output = self.model.temporalAttention(output)
    #     output = self.model.FC_2(output)

    #     gate_truth = self.scaler.transform(real)
    #     real = real_val.transpose(1, 2)[:, :, :, None]

    #     output = output.transpose(0, 1)[:, :, :, None]
    #     if (self.predict_point < 0):
            
    #     else:
    #         real = torch.unsqueeze(real_val, dim=1)[:, :, :, self.predict_point][:, :, :, None]
    #     predict = self.scaler.inverse_transform(output)

    #     x_mask = util.generate_mask(input[:, :, :, -1:], self.zero_)

    #     if self.task_level<=self.seq_out_len:
    #         if epoch%self.step_size==0:
    #             self.task_level +=1
    #         loss = self.loss(predict[:, :self.task_level, :, :], real[:, :self.task_level, :, :], 0.0, x_mask)
    #     else:

    #         diff = torch.mean(torch.abs(self.model.odeFunc.rdgcn_transient.weight_react - self.model.odeFunc.rdgcn_equlibrium.weight_react) + \
    #                         torch.abs(self.model.odeFunc.rdgcn_transient.weight_diff - self.model.odeFunc.rdgcn_equlibrium.weight_diff)) / 2
    #         loss = self.loss(predict, real, 0.0, x_mask) + 1/(diff)
    #     # diff = (torch.mean(self.model.odeFunc.modulelist[0].weight_react)) + (torch.mean(self.model.odeFunc.modulelist[0].weight_diff)) 
    #             # self.kl_loss(self.model.odeFunc.rdgcn_transient.weight_react, self.model.odeFunc.rdgcn_equlibrium.weight_react) + \
    #             # self.kl_loss(self.model.odeFunc.rdgcn_transient.weight_diff, self.model.odeFunc.rdgcn_equlibrium.weight_diff)

    #      # + 1/(diff) #  diff
    #     # print("current loss: ", loss)
    #     loss.backward()
    #     if self.clip is not None:
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
    #     self.optimizer.step()

    #     mae = util.masked_mae(predict, real, 0.0, x_mask).item()
    #     mape = util.masked_mape(predict, real, 0.0, x_mask).item()
    #     rmse = util.masked_rmse(predict, real, 0.0, x_mask).item()
    #     return mae, mape, rmse

    # let the model diff 1 times
    def eval(self, input, real_val, gate):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(0, 1)[:, :, :, None]

        # generate training mask
        x_mask = util.generate_mask(input[:, :, :, -1:], self.zero_)
        predict = self.scaler.inverse_transform(output)
        if (self.predict_point < 0):
            real = real_val.transpose(1, 2)[:, :, :, None]
        else:
            real = torch.unsqueeze(real_val, dim=1)
            real = real[:, :, :, self.predict_point][:, :, :, None]
        mae = util.masked_mae(predict, real, 0.0, x_mask).item()
        mape = util.masked_mape(predict, real, 0.0, x_mask).item()
        rmse = util.masked_rmse(predict, real, 0.0, x_mask).item()


        return mae, mape, rmse


