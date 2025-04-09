import torch
import numpy as np
import argparse
import time
import util
from engine import trainer
import os
import json
import time

# test_load = []


def main(args, starting_hr):
    print("expid: ", args.expid)
    device = torch.device(args.device)
    root_path = "/home/yus516/data/code/explainable-graph/Physics-Informed-DMSTGCN/DMSTGCN-v1/"
    config = json.load(open(root_path + "data/config.json", "r"))[args.data]
    args.adj_path = root_path + config["adj_path"]
    args.days = config["num_slots"]  # number of timeslots in a day which depends on the dataset
    args.num_nodes = config["num_nodes"]  # number of nodes
    args.normalization = config["normalization"]  # method of normalization which depends on the dataset
    args.data_dir = config["data_dir"]  # directory of data

    args.start_point = starting_hr
    args.data_dir = root_path + args.data_dir
    dataloader = util.load_dataset_weekday_weekend(args.data_dir, args.batch_size, args.batch_size, args.batch_size, days=args.days,
                                   sequence=args.seq_length, in_seq=args.in_len, 
                                   filter=args.filter, start_point=args.start_point, lastinghours=args.lastinghours)
    scaler = dataloader['scaler']
    zero_ = scaler.transform(-1)


    print(args)
    start_epoch = 1
    engine = trainer(scaler, args.adj_path, args.num_nodes,
                     args.learning_rate, args.weight_decay, device, 
                     resolution=args.resolution, num_sequence=args.num_sequence, num_output_sequence=args.num_output_sequence, 
                     enable_bias=args.enable_bias, predict_point=args.predicting_point, zero_=zero_,
                     num_rd_kernels= args.num_rd_kernels, temperture = args.temperture)
    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    count = 0

    target_dataset = 'train_loader'
    print(target_dataset, ":", dataloader[target_dataset].size)

########################################################################################################

    # final test
    # engine.model.load_state_dict(torch.load(
    #     os.path.join("best_models/" + args.data, "exp2_best_4.07reaction-diffusion-nature-weekday[fast-sequence8-12-last-30-day-reaction-diffusion-pred--1-13]True.pth")))
    # engine.model.load_state_dict(torch.load(
    #     os.path.join("best_models/" + args.data, "exp1_best_3.35reaction-diffusion-nature-weekday[semi-continuous-sequence4-8-last-60-day-reaction-diffusion-pred--1-13]True.pth")))
    # engine.model.load_state_dict(torch.load(
    #     os.path.join("best_models/" + args.data, "exp0_best_1.12reaction-diffusion-nature-weekday[semi-continuous-sequence0-4-last-60-day-reaction-diffusion-pred--1-13]True.pth")))
    # all_files = os.listdir("best_models/" + args.data) 
    # f_name = "semi-continuous-sequence" + str(args.expid*4) + '-' + str(args.expid*4+4)
    # for f in all_files:
    #     if (f_name in f):
    #         engine.model.load_state_dict(torch.load(
    #             os.path.join("best_models/" + args.data, f)))
    #         print(f)
    #         break
    # engine.model.load_state_dict(torch.load(
    #     os.path.join("best_models/pemsbayweekdayweekend/binary-gate-first-12-weekday", "exp0_best_0.98reaction-diffusion-nature-weekday[pure_matmul_gate_binary-semi-continuous-sequence0-4-last-12-day-reaction-diffusion-pred--1-13]True.pth")))
    # engine.model.load_state_dict(torch.load(
    #     os.path.join("best_models/pemsbayweekdayweekend/binary-gate-first-12-weekday", "exp2_best_1.77reaction-diffusion-nature-weekday[pure_matmul_gate_binary-semi-continuous-sequence8-12-last-12-day-reaction-diffusion-pred--1-13]True.pth"))) 
    # engine.model.load_state_dict(torch.load(
    #     os.path.join("save_models/pemsbayweekdayweekend/binary-gate-first-12-weekday", "reaction-diffusion-nature-weekday[pure_matmul_gate_binary-semi-continuous-sequence8-12-last-12-day-reaction-diffusion-pred--1-13]True.pth"))) 
    # engine.model.load_state_dict(torch.load(
    #         os.path.join("best_models/pemsbayweekdayweekend", "exp0_best_1.41reaction-diffusion-nature-weekday[matmul_gate_2-mod-binary-high-low-penalty-semi-continuous-sequence0-4-last-12-day-reaction-diffusion-pred--1-13]True.pth"))) 
    # engine.model.load_state_dict(torch.load(
    #         os.path.join("best_models/pemsbayweekdayweekend", "exp2_best_2.61reaction-diffusion-nature-weekday[matmul_gate_2-mod-binary-high-low-penalty-semi-continuous-sequence8-12-last-12-day-reaction-diffusion-pred--1-13]True.pth"))) 
    engine.model.load_state_dict(torch.load(
            os.path.join("best_models/metrlaweekdayweekend", "exp2_best_4.54reaction-diffusion-nature-weekday[matmul_gate_5-mod-binary-high-low-penalty-semi-continuous-sequence8-12-from-0lasting12-day-pred--1-13]True.pth"))) 
 

    # engine.model.load_state_dict(torch.load(
    #     os.path.join("best_models/" + args.data, "exp2_best_2.32reaction-diffusion-nature-weekday[semi-continuous-sequence8-12-last-60-day-reaction-diffusion-pred--1-13]True.pth")))
    # engine.model.load_state_dict(torch.load(
    #     os.path.join("best_models/" + args.data, "exp2_best_2.32reaction-diffusion-nature-weekday[semi-continuous-sequence8-12-last-60-day-reaction-diffusion-pred--1-13]True.pth")))

    outputs = []
    # realy = torch.Tensor(dataloader['y_test']).to(device)
    # realy = torch.Tensor(dataloader['test_loader'].ys).to(device)
    realy = torch.Tensor(dataloader[target_dataset].ys).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    # test_load.append(dataloader['train_loader'])

    testx_mask = []
    inputs = []

    for itera, (x, y, ind) in enumerate(dataloader[target_dataset].get_iterator()):
    # for itera, (x, y, ind) in enumerate(dataloader['train_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        # testx = testx[:, :, :, None]
        testx = testx.transpose(1, 3)
        with torch.no_grad():

            # testx = testx[:, :, :, -1][:, :, :, None]
            x_mask = util.generate_mask(testx, zero_)
            testx_mask.append(x_mask)

            output = engine.model(testx, ind)
            output = output.transpose(0, 1).transpose(1, 2)
            preds = output
        outputs.append(preds)
        inputs.append(testx)
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    xhat = torch.cat(inputs, dim=0)
    xhat = xhat[:realy.size(0), ...]

    testx_mask = torch.cat(testx_mask, dim=0)
    testx_mask = testx_mask[:realy.size(0), ...]  
    testx_mask = testx_mask[:, 0, :, 0]

    amae = []
    amape = []
    armse = [] 

    pred = scaler.inverse_transform(yhat)
    base = scaler.inverse_transform(xhat)
    real = realy[:pred.shape[0], :, :]
    # a=np.array([pred.cpu().numpy(), real.cpu().numpy()])
    # np.save('pems-bay.npy', a)
    
    for i in range(yhat.shape[2]):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:pred.shape[0], :, i]
        metrics = util.metric_strength(pred, real, testx_mask)
        log = 'Evaluate best model on test data for horizon {:d},' \
              ' Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))


    time_period = '[fast-test' + str(starting_hr) + '-' + str(starting_hr+4) + '-last-12-day-reaction-diffusion-pred-' + str(args.predicting_point) + "-" + str(args.num_sequence) + ']' + str(args.enable_bias)
    print(time_period)
    for hr in range(24):
        for hrz in range(yhat.shape[2]):
            pred = scaler.inverse_transform(yhat[:, :, hrz])
            real = realy[:pred.shape[0], :, hrz]
            idx = util.slice_every_hour(hr, pred.shape[0])
            metrics = util.metric_strength(pred[idx], real[idx], testx_mask[idx])
            log ='Hour: {:.4f}, horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(hr, hrz, metrics[0], metrics[1], metrics[2]))

    return np.asarray(amae), np.asarray(amape), np.asarray(armse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--data', type=str, default='metrlaweekdayweekend', help='data path')
    parser.add_argument('--seq_length', type=int, default=12, help='output length')
    parser.add_argument('--in_len', type=int, default=12, help='input length')
    parser.add_argument('--nhid', type=int, default=32, help='')
    parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--epochs', type=int, default=1, help='')
    parser.add_argument('--print_every', type=int, default=50, help='')
    parser.add_argument('--start_runs', type=int, default=1, help='number of the starting experiments')
    parser.add_argument('--runs', type=int, default=1, help='number of experiments')
    parser.add_argument('--expid', type=int, default=1, help='experiment id')
    parser.add_argument('--filter', type=int, default=1, help='whether filter 1, do, 0, not')
    parser.add_argument('--start_point', type=int, default=0, help='start_point')
    parser.add_argument('--lastinghours', type=int, default=4, help='how long does the period last')
    parser.add_argument('--iden', type=str, default='', help='identity')
    parser.add_argument('--dims', type=int, default=32, help='dimension of embeddings for dynamic graph')
    parser.add_argument('--order', type=int, default=2, help='order of graph convolution')
    parser.add_argument('--resolution', type=int, default=288, help='resolution')
    parser.add_argument('--predicting_point', type=int, default=-1, help='predicting time point')
    parser.add_argument('--num_sequence', type=int, default=13, help='num of the predicting time sequence')
    parser.add_argument('--num_output_sequence', type=int, default=1, help='num of the output time sequence')
    parser.add_argument('--limited_test_batch_num', type=int, default=20, help='num of the predicting time sequence')
    parser.add_argument('--enable_bias', type=bool, default=True, help='num of the predicting time sequence')
    parser.add_argument('--num_weekday', type=int, default=60, help='num of the training weekdays')
    parser.add_argument('--start_weekday', type=int, default=0, help='starting point of the training weekdays')

    parser.add_argument('--temperture', type=float, default=0.5, help='num of the training weekdays')
    parser.add_argument('--num_rd_kernels', type=int, default=5, help='num of the training weekdays')


    

    # time.sleep(3 * 60 * 60)

    args = parser.parse_args()
    args.save = os.path.join('save_models/', os.path.basename(args.data) + args.iden)
    os.makedirs(args.save, exist_ok=True)
    t1 = time.time()
    metric = []

    # for training_size in [50, 100, 150]:

    for i in range(args.runs):
        args.expid = i
        metric.append(main(args, 8))
        t2 = time.time()
        print("Total time spent: {:.4f}".format(t2 - t1))
    metric = np.asarray(metric)
    print(metric)  # 5 3 12
    for i in range(0):
        print(f"mae for 5 {(i + 1)*5}: {np.mean(metric[:, 0, i])}±{np.std(metric[:, 0, i])}")
        print(f"mape for step{(i + 1)*5}: {np.mean(metric[:, 1, i])}±{np.std(metric[:, 1, i])}")
        print(f"rmse for step{(i + 1)*5}: {np.mean(metric[:, 2, i])}±{np.std(metric[:, 2, i])}")
    print(f"mean of best mae: {np.mean(metric[:, 0])}±{np.std(np.mean(metric[:, 0], axis=1))}")
    print(f"mean of best mape: {np.mean(metric[:, 1])}±{np.std(np.mean(metric[:, 1], axis=1))}")
    print(f"mean of best rmse: {np.mean(metric[:, 2])}±{np.std(np.mean(metric[:, 2], axis=1))}")
