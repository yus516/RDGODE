from math import sqrt
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torch.nn.modules.module import Module
import pandas as pd
import geopy.distance



"""
This is an adaption of the GNNExplainer of the PyTorch-Lightning library. 

The main similarity is the use of the methods _set_mask and _clear_mask to handle the mask. 
The main difference is the handling of different classification tasks. The original Geometric implementation only works for node 
classification. The implementation presented here also works for graph_classification datasets. 

link: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/gnn_explainer.html
"""


class GNNExplainer(Module):
    """
    A class encaptulating the GNNexplainer (https://arxiv.org/abs/1903.03894).
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph"
    :param epochs: amount of epochs to train our explainer
    :param lr: learning rate used in the training of the explainer
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    
    :function __set_masks__: utility; sets learnable mask on the graphs.
    :function __clear_masks__: utility; rmoves the learnable mask.
    :function _loss: calculates the loss of the explainer
    :function explain: trains the explainer to return the subgraph which explains the classification of the model-to-be-explained.
    """
    def __init__(self, device, model_to_explain, mat, 
                #  x_train, x_val, x_test,
                #  y_train, y_val, y_test,
                #  ypred_train, ypred_val, ypred_test, 
                trainloader, valloader, testloader,
                 lr=0.003, reg_coefs=(0.05, 1.0), name="full"):
        super(GNNExplainer, self).__init__()
        self.device = device
        self.model_to_explain = model_to_explain
        self.mat = mat
        self.name = name
        # self.df_location = pd.read_csv('data/metr-la/graph_sensor_locations.csv')
        # self.df_location = pd.read_csv('data/pems-bay/filtered_location.csv')
        self.df_location = pd.read_csv('data/seattle-loop/graph-sensor-location-seattle.csv')
        # self.x_train = x_train[0:64, :, :, 0][:, :, :, None].transpose(1, 3).transpose(2, 3).float()
        # self.x_val = x_val[0:64, :, :, 0][:, :, :, None].transpose(1, 3).transpose(2, 3).float()
        # self.x_test = x_test[0:64, :, :, 0][:, :, :, None].transpose(1, 3).transpose(2, 3).float()
        # self.y_train = y_train[0:64, :, :, 0][:, :, :, None].transpose(1, 3).transpose(2, 3).float()
        # self.y_val = y_val[0:64, :, :, 0][:, :, :, None].transpose(1, 3).transpose(2, 3).float()
        # self.y_test = y_test[0:64, :, :, 0][:, :, :, None].transpose(1, 3).transpose(2, 3).float()
        # self.ypred_train = ypred_train[0:64, :, :, 0][:, :, :, None].transpose(1, 3).transpose(2, 3).float()
        # self.ypred_val = ypred_val[0:64, :, :, 0][:, :, :, None].transpose(1, 3).transpose(2, 3).float()
        # self.ypred_test = ypred_test[0:64, :, :, 0][:, :, :, None].transpose(1, 3).transpose(2, 3).float()

        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        self.n_node = self.mat.shape[-1]
        # self.epochs = epochs
        self.lr = lr
        self.reg_coefs = reg_coefs
        
    def _set_masks(self, x):
        """
        Inject the explanation maks into the message passing modules.
        :param x: features
        :param edge_index: graph representation
        """
        # _, N = x.size()
        # N = 207
        # N=281
        N = 323

        # std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        # edge_mask = torch.nn.Parameter(torch.randn((N,N)) * std)
        edge_mask = torch.nn.Parameter(torch.randn((N,N), requires_grad=True, device=self.device))
        return edge_mask

    def mask_loss(self, mask, reg_coefs):
        # Regularization losses
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]
        mask = torch.sigmoid(mask)
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)
        mask_loss = size_loss + mask_ent_loss 
        return mask_loss / len(mask) 

    def process_mask(self, mask, df):
        _10th_col = mask[:, self.location]
        _10th_row = mask[self.location, :]
        _10th_col_top3_indices = np.argsort(_10th_col)[::-1][:3]
        _10th_row_top3_indices = np.argsort(_10th_row)[::-1][:3]
        print(_10th_col_top3_indices)
        print(_10th_row_top3_indices)
        coord_df_10 = (df.iloc[self.location]['latitude'], df.iloc[self.location]['longitude'])
        # df_10_lat = df.iloc[10]['latitude']
        # df_10_long = df.iloc[10]['longitude']
        col_dis = []
        row_dis = []
        for c in _10th_col_top3_indices:
            coord_c = (df.iloc[c]['latitude'], df.iloc[c]['longitude'])
            col_dis.append(geopy.distance.geodesic(coord_df_10, coord_c).km/1.6)
            # col_dis = col_dis + math.sqrt((df.iloc[c]['latitude'] - df_10_lat)**2 + (df.iloc[c]['longitude'] - df_10_long)**2)
        for r in _10th_row_top3_indices:
            coord_r = (df.iloc[r]['latitude'], df.iloc[r]['longitude'])
            row_dis.append(geopy.distance.geodesic(coord_df_10, coord_r).km/1.6)
            # row_dis = row_dis + math.sqrt((df.iloc[r]['latitude'] - df_10_lat)**2 + (df.iloc[r]['longitude'] - df_10_long)**2)
        # col_dis_avg = col_dis / 3
        # row_dis_avg = row_dis / 3
        return col_dis, row_dis

    def calculate_average_distance(self):
        return self.process_mask(self.mask_result, self.df_location)



    def mse_loss(self, masked_pred, original_pred):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        
        # Explanation loss
        mse_loss = torch.linalg.norm(masked_pred - original_pred, ord=2)
        return mse_loss

    def explain_single(self, location, epochs):
        """
        Main method to construct the explanation for a given sample. This is done by training a mask such that the masked graph still gives
        the same prediction as the original graph using an optimization approach
        :param index: index of the node/graph that we wish to explain
        :return: explanation graph and edge weights
        """
        self.mask = self._set_masks(self.trainloader.ys)
        # print(mask.shape, mask.type)
        optimizer = Adam([self.mask], lr=self.lr)
        self.location = location
        self.model_to_explain.eval()
        
        
        with torch.no_grad():
            for i in range(len(self.model_to_explain.st_blocks)):
                self.model_to_explain.st_blocks[i].graph_conv.gcnconv.gcnconv_matrix = self.mat

            outputs = []
            for itera, (x, y, ind) in enumerate(self.trainloader.get_iterator()):
                trainx = torch.Tensor(x).to(self.device)
                trainx = trainx.transpose(2, 3)
                trainx = trainx.transpose(1, 2)[:, 0, :, :][:, None, :, :]
                masked_train_pred_i = self.model_to_explain(trainx)
                outputs.append(masked_train_pred_i)
            ori_train_pred = torch.cat(outputs, dim=0)

            outputs = []
            for itera, (x, y, ind) in enumerate(self.valloader.get_iterator()):
                valx = torch.Tensor(x).to(self.device)
                valx = valx.transpose(2, 3)
                valx = valx.transpose(1, 2)[:, 0, :, :][:, None, :, :]
                masked_val_pred_i = self.model_to_explain(valx)
                outputs.append(masked_val_pred_i)
            ori_val_pred = torch.cat(outputs, dim=0)

            outputs = []
            for itera, (x, y, ind) in enumerate(self.testloader.get_iterator()):
                testx = torch.Tensor(x).to(self.device)
                testx = testx.transpose(2, 3)
                testx = testx.transpose(1, 2)[:, 0, :, :][:, None, :, :]
                masked_test_pred_i = self.model_to_explain(testx)
                outputs.append(masked_test_pred_i)
            ori_test_pred = torch.cat(outputs, dim=0)            
            # ori_train_pred = self.model_to_explain(self.x_train)
            # ori_val_pred = self.model_to_explain(self.x_val)
            # ori_test_pred = self.model_to_explain(self.x_test)

        best_val_mse = np.float('inf')
        # Start training loop
        # for e in tqdm(range(epochs)):
        for e in range(epochs):
            print("epochs, ", e)
            # print("self.mask: ", self.mask)
            optimizer.zero_grad()
            masked_mat = torch.sigmoid(self.mask) * self.mat
            for i in range(len(self.model_to_explain.st_blocks)):
                self.model_to_explain.st_blocks[i].graph_conv.gcnconv.gcnconv_matrix = masked_mat

            outputs = []
            for itera, (x, y, ind) in enumerate(self.trainloader.get_iterator()):
                trainx = torch.Tensor(x).to(self.device)
                trainx = trainx.transpose(2, 3)
                trainx = trainx.transpose(1, 2)[:, 0, :, :][:, None, :, :]
                masked_train_pred_i = self.model_to_explain(trainx)
                outputs.append(masked_train_pred_i)
            masked_train_pred = torch.cat(outputs, dim=0)

            mse_train_loss = self.mse_loss(torch.squeeze(masked_train_pred)[:, location], torch.squeeze(ori_train_pred)[:, location])
            mask_train_loss = self.mask_loss(self.mask, self.reg_coefs)
            loss = mse_train_loss + mask_train_loss
            print("loss is: ", loss, ", mse_train_loss is ", mse_train_loss, "mask_train_loss is ", mask_train_loss)
            loss.backward()
            optimizer.step()
    
            # eval
            for i in range(len(self.model_to_explain.st_blocks)):
                self.model_to_explain.st_blocks[i].graph_conv.gcnconv.gcnconv_matrix = masked_mat
            outputs = []
            for itera, (x, y, ind) in enumerate(self.valloader.get_iterator()):
                valx = torch.Tensor(x).to(self.device)
                valx = valx.transpose(2, 3)
                valx = valx.transpose(1, 2)[:, 0, :, :][:, None, :, :]
                masked_val_pred_i = self.model_to_explain(valx)
                outputs.append(masked_val_pred_i)
            masked_val_pred = torch.cat(outputs, dim=0)
            # print("masked_val_pred: ", torch.squeeze(masked_val_pred)[:, location])
            # print("graph: ", self.model_to_explain.st_blocks[i].graph_conv.gcnconv.gcnconv_matrix)
            mse_val_loss = self.mse_loss(torch.squeeze(masked_val_pred)[:, location], torch.squeeze(ori_val_pred)[:, location])
            print("mse_val_loss: ", mse_val_loss)
            if best_val_mse > mse_val_loss:
                best_val_mse = mse_val_loss
                best_e = e
                best_train_mse = mse_train_loss
                for i in range(len(self.model_to_explain.st_blocks)):
                    self.model_to_explain.st_blocks[i].graph_conv.gcnconv.gcnconv_matrix = masked_mat

                outputs = []
                for itera, (x, y, ind) in enumerate(self.testloader.get_iterator()):
                    testx = torch.Tensor(x).to(self.device)
                    testx = testx.transpose(2, 3)
                    testx = testx.transpose(1, 2)[:, 0, :, :][:, None, :, :]
                    masked_test_pred_i = self.model_to_explain(testx)
                    outputs.append(masked_test_pred_i)
                best_test_mse = torch.cat(outputs, dim=0)
                # best_test_mse = self.model_to_explain(self.x_test)
                
                masked_test_pred = best_test_mse
                m1train = self.eval_metric(self.trainloader.ys[:len(masked_train_pred), 0, :, 0], torch.squeeze(masked_train_pred), location)
                m1val = self.eval_metric(self.valloader.ys[:len(masked_val_pred), 0, :, 0], torch.squeeze(masked_val_pred), location)
                m1test = self.eval_metric(self.testloader.ys[:len(masked_test_pred), 0, :, 0], torch.squeeze(masked_test_pred), location)
            elif e - best_e >= 20:
                break
            print(m1train, m1val, m1test)
        # np.save("mask_"+str(location)+ self.name + ".npy", self.mask.detach().cpu().numpy())
        self.mask_result = self.mask.detach().cpu().numpy()
        return m1train, m1val, m1test
    
    
    def eval_metric(self, yts, fpred, location):
        yt = yts.transpose(1,0)
        ytp1 = yts.transpose(1,0)
        yti = yt[location, :]
        ytp1i = ytp1[location, :]
        fpred = torch.squeeze(fpred)[:, location]
        m1 =  np.abs( fpred.detach().cpu().numpy() - ytp1i / yti)
        # m1 = torch.abs(yti + fpred - ytp1i)
        m1 = np.mean(m1)
        return m1