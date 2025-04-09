import learn2learn as l2l
import numpy as np
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import sys
import copy
from pickletools import stringnl_noescape
import torch
import argparse
import util
import os
import json
import pandas as pd
from multi_layer_reaction_diffusion import RDGODE_fixed_gate as reaction_diffusion_nature
# from reaction_diffusion import reaction_diffusion_nature
import csv 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def check_exist(path):
    isExist = os.path.exists(path)
    if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

def load_weighted_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, header=None)
    return df.to_numpy()

# def plot_mae(filename):
#         df = pd.read_csv(filename)
#         for index, row in df.iterrows():
#             hrs = np.arange(24)
#             plt.plot(hrs, row, label='{}-{}'.format(str(index*4),str((index+1)*4)))
#         plt.ylabel('MAE')
#         plt.xlabel('Hour')
#         plt.ylim(0,6)
#         plt.title("MAE over time using various training interval")
#         plt.legend()
#         plt.show()

def get_dataset_name(dataset_name):
    if dataset_name == "pemsbayweekdayweekend":
        sub_folder = "pems_bay_200"
        data_set_name = "pems_bay"

    elif dataset_name == "metrlaweekdayweekend":
        sub_folder = "metr_la_200"
        data_set_name = "metr_la"

    elif dataset_name == "seattle-loop-weekday":
        sub_folder = "seattle_200"
        data_set_name = "seattle"

    else:
        print("Error: Dataset {} does not exist".format(dataset_name))

    return sub_folder, data_set_name

def STGCN_model(args):
    #STGCN model params
    Kt = 3
    Ks = 2
    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    device = torch.device(args.device)
    blocks = []
    stblock_num = 2
    n_his = 12
    
    
    gated_act_func = "glu"
    graph_conv_type = "gcnconv"
    drop_rate = 0.5
    Ko = n_his - (Kt - 1) * 2 * stblock_num
    blocks.append([1])
    for l in range(stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    #Pems-bay 281, metr-la 207, seattle 323
    if args.data == "pemsbayweekdayweekend":
        n_vertex = 281
        adj_mat_file ="data/pems-bay/adj_pems_bay.csv"

    elif args.data == "metrlaweekdayweekend":
        n_vertex = 207
        adj_mat_file = "data/metr-la/metrla-virtual-id-revised.csv"

    elif args.data == "seattle-loop-weekday":
        n_vertex = 323
        adj_mat_file = "data/seattle/adj_seattle.csv"
    else:
        print("Error: Dataset {} does not exist".format(args.data))

    adj_mat_file = "/home/yus516/data/code/explainable-graph/Physics-Informed-DMSTGCN/DMSTGCN-v1/" + adj_mat_file

    # adj_mat = load_weighted_adjacency_matrix(adj_mat_file)
    # mat_type = "hat_sym_normd_lap_mat"
    # mat = util.calculate_laplacian_matrix(adj_mat, mat_type)
    # gcnconv_matrix = torch.from_numpy(mat).float().to(device)
    model = reaction_diffusion_nature(device, adj_mat_file, n_vertex, [True, True], 288, 2, -1, True, 1)
    return model 



class STGCN_MAML():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        

    def fast_adapt(learner, predict,real, x_mask, adaptation_steps = 1):
        for _ in range(adaptation_steps):
            adaptation_loss = util.masked_mae(predict, real, 0.0, x_mask)
            learner.adapt(adaptation_loss,allow_unused=True)
    
    def generate_mask(self, data):
        x_mask = (data[:, :, -1, :] >= -3)
        x_mask = x_mask.float()[:, :, :, None]
        x_mask /= torch.mean((x_mask))
        x_mask = torch.where(torch.isnan(x_mask), torch.zeros_like(x_mask), x_mask) 
        return x_mask

    def loss(self, scaler, learner, x, y, ind):
        output = learner(x, ind)
        output = torch.squeeze(output)
        y = torch.unsqueeze(y, dim=1)[:, :, :, 0][:, :, :, None]
        pred = scaler.inverse_transform(output)
        x_mask = self.generate_mask(x)
        masked_loss = util.masked_mae(pred, y, 0.0, x_mask)
        return masked_loss

    def load_data(self, Is_train = True):
        args = self.args
        self.device = torch.device(args.device)
        self.adaptation_steps = args.adaptation_steps
        config = json.load(open("/home/yus516/data/code/explainable-graph/Physics-Informed-DMSTGCN/DMSTGCN-v1/data/config.json", "r"))[args.data]
        args.days = config["num_slots"]  # number of timeslots in a day which depends on the dataset
        args.num_nodes = config["num_nodes"]  # number of nodes
        args.normalization = config["normalization"]  # method of normalization which depends on the dataset
        args.data_dir = '/home/yus516/data/code/explainable-graph/Physics-Informed-DMSTGCN/DMSTGCN-v1/' + config["data_dir"]  # directory of data
        keep_order = False
        if Is_train:
            batch_size = args.train_batch_size
        else:
            batch_size = args.test_batch_size
        dataloader = util.load_dataset_weekday_weekend(args.data_dir, batch_size , batch_size , batch_size , days=args.days,
                                    sequence=args.seq_length, in_seq=args.in_len,
                                    filter=args.filter, start_point=args.start_point, lastinghours=args.lastinghours)
        return dataloader         

    def save_MAML_model(self, model, save_path, file_name):
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, file_name))

    def slice_every_hour(self, start_hour, num_hr):
        return np.arange(start_hour, num_hr, 24)   

    def split_support_and_query(self, batch_size, _seed):
        np.random.seed(_seed)
        x = np.arange(0, int(batch_size),2)
        np.random.shuffle(x)
        x1 = x[:int(len(x)/2)]
        x2 = x[int(len(x)/2):]
        support_idx, query_idx = [], []

        for i in range(int(len(x)/2)):
            support_idx.append(x1[i])
            support_idx.append(x1[i]+1)
            query_idx.append(x2[i])
            query_idx.append(x2[i]+1)
        return np.array(support_idx), np.array(query_idx)

    def test(self, load_path):
        dataloader = self.load_data(Is_train=False)    
        scaler = dataloader['scaler']
        maml = l2l.algorithms.MAML(self.model, lr=self.args.maml_learning_rate, first_order=False)
        maml.load_state_dict(torch.load(load_path))
        learner = maml.clone()
        outputs = []
        testx_mask = []

        realy = torch.Tensor(dataloader['test_loader'].ys).to(self.device)
        realy = realy.transpose(1, 3)[:, 0, :, :]

        for itera, (x, y, ind) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(self.device)
            testx = testx.transpose(1, 3)
            testx = testx[:, :, :, -1][:, :, :, None]
            with torch.no_grad():

                # preds = engine.model(testx, ind)[:, None, :, :]
                # preds = preds.transpose(1, 3)

                x_mask = (testx[:, :, :, -1][:, :, :, None] >= -3)
                x_mask = x_mask.float()
                x_mask /= torch.mean((x_mask))
                x_mask = torch.where(torch.isnan(x_mask), torch.zeros_like(x_mask), x_mask)  

                output = learner(testx, ind)
                output = torch.squeeze(output)
                preds = output

                # preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze())
            testx_mask.append(x_mask)

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        testx_mask = torch.cat(testx_mask, dim=0)
        testx_mask = testx_mask[:realy.size(0), ...]  
        testx_mask = testx_mask[:, 0, :, 0]

        amae = []
        amape = []
        armse = []
        for hr in range(24):

            i = self.args.predict_point
            pred = scaler.inverse_transform(yhat)[:, :, None]
            real = realy[:pred.shape[0], :, i][:, :, None]
            idx = self.slice_every_hour(hr, pred.shape[0])
            metrics = util.metric_strength(pred[idx], real[idx], testx_mask[idx])
            log =' Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])
        return amae, amape, armse

    def train(self, load_path, train_save_path, save_file_name):
        dataloader = self.load_data()    
        scaler = dataloader['scaler']
        num_support = self.args.train_support

        maml = l2l.algorithms.MAML(self.model, lr=self.args.maml_learning_rate, first_order=False)
        if self.args.resume:
            print("Load model from: " , load_path)
            maml.load_state_dict(torch.load(load_path))
        else:
            print("Train from the begining")
        # opt = torch.optim.SGD(maml.parameters(), lr=args.learning_rate)
        opt = torch.optim.Adam(maml.parameters(), lr=self.args.learning_rate)
        err_list = []
        for i in range(self.args.epochs):
            error = 0
            opt.zero_grad()
            #Task Iteration:
            num_batch = 0
            # dataloader['train_loader'].shuffle()
            #temp pattern, different time intervals 
            for itera, (x, y, ind) in enumerate(dataloader['train_loader'].get_iterator()):
                support_idx, query_idx = self.split_support_and_query(self.args.train_batch_size,int(100*i+itera))
                num_batch += 1
                learner = maml.clone() 
                #1. split into support and query set
                x = torch.Tensor(x).to(self.device)
                y = torch.Tensor(y).to(self.device)

                #Update here:
                #
                #
                x_support, y_support = x[support_idx], y[support_idx]
                x_query, y_query = x[query_idx], y[query_idx]  
                # x_support, y_support = x[:num_support], y[:num_support]
                # x_query, y_query = x[num_support:], y[num_support:]   
                # 
                #          
                #reshape
                x_support, y_support = x_support.transpose(1, 3)[:, :, :, -1][:, :, :, None], y_support.transpose(1, 3)[:, 0, :, :]             
                x_query, y_query = x_query.transpose(1, 3)[:, :, :, -1][:, :, :, None], y_query.transpose(1, 3)[:, 0, :, :]   

                #2. Adapt using support set
                #Takes a gradient step on the loss and updates the cloned parameters in place.
                adaptation_loss = self.loss(scaler, learner, x_support, y_support, ind)
                learner.adapt(adaptation_loss,allow_unused=True)
                #3. Meta-update the model parameters using query set
                validation_loss = self.loss(scaler, learner, x_query, y_query, ind)
                validation_loss.backward()

                error += validation_loss
            error = error.detach().cpu().numpy()/num_batch
            print("Epoch {} Training Validation Error: {}".format(i, error) )
            err_list.append(error)
            for p in maml.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / num_batch)

            opt.step()
            
        self.save_MAML_model(maml, train_save_path, save_file_name)
        return err_list

    def train_test(self, load_path):
        dataloader = self.load_data()    
        scaler = dataloader['scaler']


        maml = l2l.algorithms.MAML(self.model, lr=self.args.maml_learning_rate, first_order=False)
        maml.load_state_dict(torch.load(load_path))

        # opt = torch.optim.SGD(maml.parameters(), lr=args.learning_rate)
        opt = torch.optim.Adam(maml.parameters(), lr=self.args.learning_rate)
        err_list = []
        for i in range(args.epochs):
            error = 0
            opt.zero_grad()
            #Task Iteration:
            num_batch = 0
            dataloader['train_loader'].shuffle()
            #temp pattern, different time intervals 
            for itera, (x, y, ind) in enumerate(dataloader['train_loader'].get_iterator()):
                num_batch += 1
                learner = maml.clone() 
                #1. split into support and query set
                x = torch.Tensor(x).to(self.device)
                y = torch.Tensor(y).to(self.device)
  
                #reshape
                x, y= x.transpose(1, 3)[:, :, :, -1][:, :, :, None], y.transpose(1, 3)[:, 0, :, :]             

                #2. Adapt using support set
                #Takes a gradient step on the loss and updates the cloned parameters in place.
                adaptation_loss = self.loss(scaler, learner, x, y, ind)
                # learner.adapt(adaptation_loss,allow_unused=True)
                adaptation_loss.backward()

                error += adaptation_loss
            error = error.detach().cpu().numpy()/num_batch
            print("Epoch {} Training Loss: {}".format(i, error) )
            err_list.append(error)
            for p in maml.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / num_batch)

            opt.step()

        outputs = []
        testx_mask = []

        realy = torch.Tensor(dataloader['test_loader'].ys).to(self.device)
        realy = realy.transpose(1, 3)[:, 0, :, :]

        for itera, (x, y, ind) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(self.device)
            testx = testx.transpose(1, 3)
            testx = testx[:, :, :, -1][:, :, :, None]
            with torch.no_grad():

                x_mask = (testx[:, :, :, -1][:, :, :, None] >= -3)
                x_mask = x_mask.float()
                x_mask /= torch.mean((x_mask))
                x_mask = torch.where(torch.isnan(x_mask), torch.zeros_like(x_mask), x_mask)  

                # let it diffuse 12 times
                output = learner(testx, ind)
                output = torch.squeeze(output)
                preds = output

                # preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze())
            testx_mask.append(x_mask)

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        testx_mask = torch.cat(testx_mask, dim=0)
        testx_mask = testx_mask[:realy.size(0), ...]  
        testx_mask = testx_mask[:, 0, :, 0]

        amae = []
        amape = []
        armse = []
        for hr in range(24):

            i = self.args.predict_point
            pred = scaler.inverse_transform(yhat)[:, :, None]
            real = realy[:pred.shape[0], :, i][:, :, None]
            idx = self.slice_every_hour(hr, pred.shape[0])
            metrics = util.metric_strength(pred[idx], real[idx], testx_mask[idx])
            log =' Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])    
        # self.save_MAML_model(maml, train_save_path, save_file_name)

        return np.array(amae), np.array(amape), np.array(armse)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2, help='')
    parser.add_argument('--maml_learning_rate', type=float, default=0.00005, help='learning rate')    
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--train_support', type=int, default=32, help='size of support set during training')
    parser.add_argument('--test_support', type=int, default=2, help='size of support set during testing')  
    parser.add_argument('--adaptation_steps', type=int, default=1, help='number of adaptation steps during testing')  
    parser.add_argument('--resume', type=bool, default=False, help='')
    parser.add_argument('--predict_point', type=int, default=0, help='predict point')
    parser.add_argument('--start_point', type=int, default=16, help='start_point')

    parser.add_argument('--device', type=str, default='cuda:1', help='')

    # parser.add_argument('--data', type=str, default='pemsbayweekdayweekend', help='data path')
    parser.add_argument('--data', type=str, default='metrlaweekdayweekend', help='data path')
    # parser.add_argument('--data', type=str, default='seattle-loop-weekday', help='data path')
    parser.add_argument('--retrain', type=bool, default=True, help='')

    parser.add_argument('--seq_length', type=int, default=1, help='output length')
    parser.add_argument('--in_len', type=int, default=12, help='input length')
    parser.add_argument('--nhid', type=int, default=32, help='')
    parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--print_every', type=int, default=50, help='')
    parser.add_argument('--runs', type=int, default=1, help='number of experiments')
    parser.add_argument('--expid', type=int, default=1, help='experiment id')
    parser.add_argument('--filter', type=int, default=1, help='whether filter 1, do, 0, not')
    parser.add_argument('--lastinghours', type=int, default=4, help='how long does the period last')
    parser.add_argument('--iden', type=str, default='', help='identity')
    parser.add_argument('--dims', type=int, default=32, help='dimension of embeddings for dynamic graph')
    parser.add_argument('--order', type=int, default=2, help='order of graph convolution')
    parser.add_argument('--resolution', type=int, default=288, help='resolution')
    parser.add_argument('--horizon', type=int, default=8, help='order of graph convolution')



    valid_loss_list = []
    mae_list = []
    args = parser.parse_args()
    
    # Is_train = True
    # Is_Test = False
    # Is_plot = False

    Is_train = True
    Is_Test = True
    Is_plot = False
    

    Is_plot = False
    if Is_plot:
        Is_train = False
        Is_Test = False

    # Is_plot = True
    
    ####Transfer MAML weights#########
    # model = STGCN_model(args=args)
    # maml = l2l.algorithms.MAML(model, lr=args.maml_learning_rate, first_order=False)
    # maml.load_state_dict(torch.load("saved_maml_model/noshuffle200/maml_params_start0.pth"))
    # #Copy MAML initial params to STGCN
    # for k, v in maml.state_dict().items():
    #     model.state_dict()[k[7:]] = v

    # for k, v in model.state_dict().items():
    #     print(k)
    # os.makedirs("saved_maml_model/noshuffle200", exist_ok=True)
    # torch.save(model.state_dict(), os.path.join("saved_maml_model/noshuffle200", "stgcn_params_start0.pth"))

    data_set_dir, data_set_name = get_dataset_name(args.data)
    args.resume = False
    day_slice = "-24to-12"
    folder = "save_models/{0}/{1}".format(day_slice, data_set_dir)
    print("Data: ", args.data)

    if not Is_plot:
        print("!")
        for i in [0,4,8,12,16,20]:
            file_name = "maml_params_start{}.pth".format(i)
            args.start_point = i
            model = STGCN_model(args=args)
            mata_learner = STGCN_MAML(args=args, model=model)
            valid_loss = None
            if Is_train:
                if args.resume:
                    train_load_path = "saved_maml_model/epoch300/"+ file_name
                    train_save_path = "saved_maml_model/epoch200"
                    check_exist(train_save_path)
                    valid_loss = mata_learner.train(train_load_path, train_save_path, file_name)
                else:
                    train_save_path = folder 
                    # train_save_path = "saved_maml_model/seattle_200"
                    train_load_path = None
                    check_exist(train_save_path)
                    valid_loss = mata_learner.train(train_load_path, train_save_path ,file_name)

            if Is_Test:
                if args.retrain:
                    print("Training using MAML init params, and then TEST")
                else:
                    print("TEST using MAML init params")
                
                # test_load_path = "saved_maml_model/metr_la_200/"+ file_name
                # test_load_path = "saved_maml_model/pems_bay_200/"+ file_name
                # test_load_path = "saved_maml_model/seattle_200/"+ file_name

                test_load_path = "{}/{}".format(folder, file_name)
                if args.retrain:
                    mae, mape, rmse = mata_learner.train_test(load_path=test_load_path)
                else:
                    mae, mape, rmse = mata_learner.test(load_path= test_load_path)
                # print(np.mean(mae))
                mae_list.append(mae)



        # valid_loss_list.append(valid_loss)
    else:
        print("Error!!!!!!!!!!!!")

    if Is_Test:    
        if args.retrain: 
            csv_file = "{0}{1}.csv".format(data_set_name,"_train200")
        else:
            csv_file = "{0}{1}.csv".format(data_set_name,"")
        # csv_file = "{0}{1}.csv".format("metr_la","_train200")
        # csv_file = "{0}{1}.csv".format("metr_la","")

        # csv_file = "{0}{1}.csv".format("pems_bay","_train200")
        # csv_file = "{0}{1}.csv".format("pems_bay","")

        # csv_file = "{0}{1}.csv".format("seattle","_train200")
        # csv_file = "{0}{1}.csv".format("seattle","")

        df = pd.DataFrame(mae_list) 
        df.to_csv("{0}/{1}".format(folder, csv_file), index=False)



    




    # if not valid_loss:
    #     valid_loss = -1
    # optimizer = "Adam"
    # with open(f'./log.csv', 'a+') as log_file:
    #     csv_writer = csv.writer(log_file, delimiter=',')
    #     csv_writer.writerow([args.epochs, optimizer, args.maml_learning_rate, args.learning_rate, 
    #     args.train_batch_size, args.test_batch_size, mae, mape, rmse, valid_loss, args.train_support, args.test_support])
