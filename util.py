import pickle
import numpy as np
import os
import torch
from scipy.linalg import eigvalsh
from scipy.linalg import fractional_matrix_power
import pandas as pd
from statistics import mean


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, begin=0, days=288, pad_with_last_sample=True, add_ind=True, ind=0):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        if add_ind:
            self.ind = np.arange(begin, begin + self.size)
        else:
            self.ind = ind
        self.days = days
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            ind_padding = np.repeat(self.ind[-1:], num_padding, axis=0)
            self.xs = np.concatenate([xs, x_padding], axis=0)
            self.ys = np.concatenate([ys, y_padding], axis=0)
            self.ind = np.concatenate([self.ind, ind_padding], axis=0)
        self.size = len(self.xs)

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.ind = self.ind[permutation]
        self.xs = xs
        self.ys = ys

    def filter_by_slice(self, start_point, end_point):
        from_point = start_point * 12
        to_point = end_point * 12
        print("filtering samples from " + str(from_point) + "to " + str(to_point))
        mid = (from_point + to_point) / 2
        width = np.abs((from_point - to_point) / 2)
        good_index = np.where((np.abs(self.ind % 288 - mid) <= width))
        self.xs = self.xs[good_index[0]]
        self.ys = self.ys[good_index[0]]
        self.ind = self.ind[good_index[0]]
        self.size = len(self.ind)

    def filter_by_state(self, from_state, to_state, thresh):
        good_index_from = np.where((self.xs[:, -1, :, 0] > thresh) == from_state)
        good_index_to = np.where((self.ys[:, 0, :, 0] > thresh) == to_state)
        good_index = np.intersect1d(good_index_from, good_index_to)
        for k in range(len(good_index_from[0])):
            self.ys[good_index_from[0][k], 0, good_index_from[1][k], 0] = 0
        for k in range(len(good_index_to[0])):
            self.ys[good_index_to[0][k], 0, good_index_to[1][k], 0] = 0

        print('finish')

    def get_speed_thresh(self):
        num_node = self.ys[:, :, :, 0].shape[2]
        max_speed = np.amax(self.ys[:, :, :, 0].reshape((-1, num_node)), axis=0)

        return max_speed * 0.75

        # self.xs = self.xs[good_index]
        # self.ys = self.ys[good_index]
        # self.ind = self.ind[good_index]
        # self.size = len(self.ind)

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                i_i = self.ind[start_ind: end_ind, ...] % self.days
                # xi_i = np.tile(np.arange(x_i.shape[1]), [x_i.shape[0], x_i.shape[2], 1, 1]).transpose(
                #     [0, 3, 1, 2]) + self.ind[start_ind: end_ind, ...].reshape([-1, 1, 1, 1])
                # x_i = np.concatenate([x_i, xi_i % self.days / self.days, np.eye(7)[xi_i // self.days % 7].squeeze(-2)],
                #                      axis=-1)
                yield (x_i, y_i, i_i)
                self.current_ind += 1

        return _wrapper()

    def get_limited_iterator(self, limited_num):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < limited_num:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                i_i = self.ind[start_ind: end_ind, ...] % self.days
                # xi_i = np.tile(np.arange(x_i.shape[1]), [x_i.shape[0], x_i.shape[2], 1, 1]).transpose(
                #     [0, 3, 1, 2]) + self.ind[start_ind: end_ind, ...].reshape([-1, 1, 1, 1])
                # x_i = np.concatenate([x_i, xi_i % self.days / self.days, np.eye(7)[xi_i // self.days % 7].squeeze(-2)],
                #                      axis=-1)
                yield (x_i, y_i, i_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_dataset_weekday_weekend(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, days=288, sequence=12,
                 in_seq=12, filter=0, start_point=0, lastinghours=24, num_weekday=12, start_weekday=0, state=[True, True]):
    data = {}

    weekday_data = np.load(os.path.join(dataset_dir, 'train_weekday.npz'))
    weekend_data = np.load(os.path.join(dataset_dir, 'test_weekend.npz'))

    # add time index
    # cat_data = np.load(os.path.join("/home/yus516/data/code/explainable-graph/MTGNN/data/PEMS-BAY/", 'train.npz'))
    # valid_lengh_weekday = min(cat_data['x'].shape[0], weekday_data['x'].shape[0])
    # valid_lengh_weekend = min(cat_data['x'].shape[0], weekend_data['x'].shape[0])
    num_node = weekday_data['x'].shape[2]
    # time_index_weekday = cat_data['x'][:valid_lengh_weekday, -in_seq:, :num_node, 1]
    # time_index_weekend = cat_data['x'][:valid_lengh_weekend, -in_seq:, :num_node, 1]
    # since we only have 4 hours each day
    # time_index_weekday = ((time_index_weekday * 288) % 48) / 48
    # time_index_weekend = ((time_index_weekend * 288) % 48) / 48

    # train_val_data_x = np.concatenate((weekday_data['x'][:, -in_seq:, :, 0:2], np.expand_dims(time_index_weekday, axis=3)), axis=3)
    # test_data_x = np.concatenate((weekend_data['x'][:, -in_seq:, :, 0:2], np.expand_dims(time_index_weekend, axis=3)), axis=3)

    # train_val_data_x = weekday_data['x'][:, -in_seq:, :, 0:2]
    # train_val_data_y = weekday_data['y'][:valid_lengh_weekday, -in_seq:, :, 0:1]

    # test_data_x = weekend_data['x'][:, -in_seq:, :, 0:2]
    # test_data_y = weekend_data['y'][:valid_lengh_weekend, -in_seq:, :, 0:1]

    random_start = 288*start_weekday
    train_val_data = DataLoader(weekday_data['x'][random_start:random_start+288*num_weekday, ...], weekday_data['y'][random_start:random_start+288*num_weekday, ...], batch_size, days=days, begin=0, add_ind=True, ind=0)
    test_data = DataLoader(weekend_data['x'], weekend_data['y'], batch_size, days=days, begin=0, add_ind=True, ind=0)

    if ('pems' in dataset_dir):
        sensor_net = pd.read_csv("/home/yus516/data/code/explainable-graph/Physics-Informed-DMSTGCN/DMSTGCN-v1/data/pems-bay/pems-bay-virtual-id.csv")
        from_sensor = sensor_net['from'].values
        to_sensor = sensor_net['to'].values
        all_sensor = np.concatenate([from_sensor, to_sensor], axis=0)
        unique_sensor = np.unique(all_sensor)
        # rapid_data = rapid_data[:, :, :, unique_sensor]
        all_sensors = np.load(os.path.join(dataset_dir, 'matches.npy'))
        train_val_data.xs = train_val_data.xs[:, :, all_sensors, :]
        train_val_data.ys = train_val_data.ys[:, :, all_sensors, :]
        test_data.xs = test_data.xs[:, :, all_sensors, :]
        test_data.ys = test_data.ys[:, :, all_sensors, :]
        num_node = all_sensors.shape[0]

    # train_val_data.size = 288*num_weekday
    # print("training under ", num_weekday, ' weekdays')
    # random_start = 288*start_weekday
    # train_val_data.xs, train_val_data.ys = train_val_data.xs[random_start:random_start+train_val_data.size, ...], train_val_data.ys[random_start:random_start+train_val_data.size, ...]
    # train_val_data.ind = train_val_data.ind[0:train_val_data.size]

    max_speed = np.amax(train_val_data.xs[:, :, :, 0].reshape((-1, num_node)), axis=0)

    train_val_data.filter_by_slice(start_point, start_point + lastinghours)
    if (filter > 0):
        train_val_data.filter_by_state(state[0], state[1], max_speed * 0.75)


    train_val_data_x, train_val_data_y = train_val_data.xs, train_val_data.ys
    # test_data_x, test_data_y = test_data.xs[0:288*6, ...], test_data.ys[0:288*6, ...]
    test_data_x, test_data_y = test_data.xs, test_data.ys

    train_set_size = int(train_val_data.size / 4)

    permutation = np.random.RandomState(seed=42).permutation(train_val_data.size)
    train_val_data_x, train_val_data_y = train_val_data_x[permutation], train_val_data_y[permutation]

    train_x = train_val_data_x[:3*train_set_size, ...]
    train_y = train_val_data_y[:3*train_set_size, ...]
    val_x = train_val_data_x[-train_set_size:, ...]
    val_y = train_val_data_y[-train_set_size:, ...]

    data['scaler'] = StandardScaler(mean=train_x[..., 0].mean(), std=train_x[..., 0].std())
    train_x[..., 0] = data['scaler'].transform(train_x[..., 0])
    val_x[..., 0] = data['scaler'].transform(val_x[..., 0])
    test_data_x[..., 0] = data['scaler'].transform(test_data_x[..., 0])

    train_x[..., 1] = data['scaler'].transform(train_x[..., 1])
    val_x[..., 1] = data['scaler'].transform(val_x[..., 1])
    test_data_x[..., 1] = data['scaler'].transform(test_data_x[..., 1])

    train_x[..., 2] = (train_x[..., 2] % 48) / 48
    val_x[..., 2] = (val_x[..., 2] % 48) / 48
    test_data_x[..., 2] = (test_data_x[..., 2] % 48) / 48

    # train_x, val_x, test_x = train_x[:, :, :, None], val_x[:, :, :, None], test_x[:, :, :, None]

    data['train_loader'] = DataLoader(train_x, train_y, batch_size, days=days, begin=0, add_ind=True, ind=0)
    data['val_loader'] = DataLoader(val_x, val_y, valid_batch_size, days=days,
                                    begin=0, add_ind=True, ind=0)
    data['test_loader'] = DataLoader(test_data_x, test_data_y, test_batch_size, days=days,
                                    begin=0, add_ind=True, ind=0)


    return data

def load_dataset_weekday_weekend_back(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, days=288, sequence=12,
                 in_seq=12, keep_order=True, filter=0, start_point=0, lastinghours=4, num_weekday=12, start_weekday=0):
    data = {}

    weekday_data = np.load(os.path.join(dataset_dir, 'train_weekday.npz'))
    weekend_data = np.load(os.path.join(dataset_dir, 'test_weekend.npz'))
    # rapid_data = np.load(os.path.join("/home/yus516/data/code/explainable-graph/GTS/data/SEATTLE-LOOP", 'rapid.npz'))
    rapid_data = np.load(os.path.join("/home/yus516/data/code/explainable-graph/GTS/data/PEMS-BAY", 'rapid.npz'))
    # rapid_data = np.load(os.path.join("/home/yus516/data/code/explainable-graph/GTS/data/METR-LA", 'rapid.npz'))

    if (("PEMS-BAY" in dataset_dir)):
        sensor_net = pd.read_csv("/home/yus516/data/code/explainable-graph/Physics-Informed-DMSTGCN/DMSTGCN-v1/data/pems-bay/pems-bay-virtual-id.csv")
        from_sensor = sensor_net['from'].values
        to_sensor = sensor_net['to'].values
        all_sensor = np.concatenate([from_sensor, to_sensor], axis=0)
        unique_sensor = np.unique(all_sensor)
        rapid_data = rapid_data[:, :, :, unique_sensor]


        weekend_data_x = np.concatenate((weekend_data['x'], rapid_data['x'][:, :, unique_sensor, :]), axis=0)
        weekend_data_y = np.concatenate((weekend_data['y'], rapid_data['y'][:, :, unique_sensor, :]), axis=0)
    else:
        # weekend_data_x = np.concatenate((weekend_data['x'], rapid_data['x']), axis=0)
        # weekend_data_y = np.concatenate((weekend_data['y'], rapid_data['y']), axis=0)
        weekend_data_x = weekend_data['x']
        weekend_data_y = weekend_data['y']

    train_val_data_x = weekday_data['x'][:, -in_seq:, :, 0:2]
    train_val_data_y = weekday_data['y'][:, -in_seq:, :, 0:1]
    test_data_x = weekend_data_x[:, -in_seq:, :, 0:2]
    test_data_y = weekend_data_y[:, -in_seq:, :, 0:1]

    train_val_data = DataLoader(train_val_data_x, train_val_data_y, batch_size, days=days, begin=0, add_ind=True, ind=0)
    test_data = DataLoader(test_data_x, test_data_y, batch_size, days=days, begin=0, add_ind=True, ind=0)

    train_val_data.size = 288 * num_weekday
    random_start = 288*start_weekday
    train_val_data.xs, train_val_data.ys = train_val_data.xs[random_start:random_start+train_val_data.size, ...], train_val_data.ys[random_start:random_start+train_val_data.size, ...]
    train_val_data.ind = train_val_data.ind[0:train_val_data.size]

    train_val_data.filter_by_slice(start_point, start_point + lastinghours)

    train_val_data_x, train_val_data_y = train_val_data.xs, train_val_data.ys

    test_data_x, test_data_y = test_data.xs, test_data.ys


    # train_set_size = int(weekday_data['x'].shape[0] / 4)
    train_set_size = int(train_val_data.size / 4)

    permutation = np.random.RandomState(seed=42).permutation(train_val_data.size)
    train_val_data_x, train_val_data_y = train_val_data_x[permutation], train_val_data_y[permutation]

    train_x = train_val_data_x[:3*train_set_size, ...]
    train_y = train_val_data_y[:3*train_set_size, ...]
    val_x = train_val_data_x[-train_set_size:, ...]
    val_y = train_val_data_y[-train_set_size:, ...]

    data['scaler'] = StandardScaler(mean=train_x[..., 0].mean(), std=train_x[..., 0].std())
    train_x = data['scaler'].transform(train_x[..., 0])
    val_x = data['scaler'].transform(val_x[..., 0])
    test_x = data['scaler'].transform(test_data_x[..., 0])

    train_x, val_x, test_x = train_x[:, :, :, None], val_x[:, :, :, None], test_x[:, :, :, None]

    data['train_loader'] = DataLoader(train_x, train_y, batch_size, days=days, begin=0, add_ind=True, ind=0)
    data['val_loader'] = DataLoader(val_x, val_y, valid_batch_size, days=days,
                                    begin=0, add_ind=True, ind=0)
    data['test_loader'] = DataLoader(test_x, test_data_y, test_batch_size, days=days,
                                    begin=0, add_ind=True, ind=0)
    print("train set length is: ", data['train_loader'].xs.shape)
    print("val set length is: ", data['val_loader'].xs.shape)
    print("test set length is: ", data['test_loader'].xs.shape)

    return data

def load_dataset_rapid(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, days=288, sequence=12,
                 in_seq=12, filter=0, start_point=0, lastinghours=24, num_weekday=12):
    data = {}

    rapid_path = '/home/yus516/data/code/explainable-graph/GTS/'
    root_path = "/home/yus516/data/code/explainable-graph/Physics-Informed-DMSTGCN/DMSTGCN-v1/"
    dataset_dir_train = root_path + dataset_dir
    weekday_data = np.load(os.path.join(dataset_dir_train, 'train_weekday.npz'))
    
    if ('metr' in dataset_dir):
        dataset_dir = 'data/METR-LA'
        dataset_dir_rapid = rapid_path + dataset_dir
        weekend_data = np.load(os.path.join(dataset_dir_rapid, 'rapid.npz'))
        test_data_x = weekend_data['x'][:, -in_seq:, :, 0:2]
        test_data_y = weekend_data['y'][:, -in_seq:, :, 0:1]
    elif ('pems' in dataset_dir):
        dataset_dir = 'data/PEMS-BAY'
        dataset_dir_rapid = rapid_path + dataset_dir
        sensor_net = pd.read_csv("/home/yus516/data/code/explainable-graph/Physics-Informed-DMSTGCN/DMSTGCN-v1/data/pems-bay/pems-bay-virtual-id.csv")
        from_sensor = sensor_net['from'].values
        to_sensor = sensor_net['to'].values
        all_sensor = np.concatenate([from_sensor, to_sensor], axis=0)
        unique_sensor = np.unique(all_sensor)
        weekday_data = np.load(os.path.join(dataset_dir_train, 'train_weekday.npz'))
        weekend_data = np.load(os.path.join(dataset_dir_rapid, 'rapid.npz'))
        test_data_x = weekend_data['x'][:, -in_seq:, unique_sensor, 0:2]
        test_data_y = weekend_data['y'][:, -in_seq:, unique_sensor, 0:1]


    elif ('seattle' in dataset_dir):
        dataset_dir = 'data/SEATTLE-LOOP'
        dataset_dir_rapid = rapid_path + dataset_dir
        weekend_data = np.load(os.path.join(dataset_dir_rapid, 'rapid.npz'))
        test_data_x = weekend_data['x'][:, -in_seq:, :, 0:2]
        test_data_y = weekend_data['y'][:, -in_seq:, :, 0:1]


    train_val_data_x = weekday_data['x'][:, -in_seq:, :, 0:2]
    train_val_data_y = weekday_data['y'][:, -in_seq:, :, 0:1]
    # test_data_x = weekend_data['x'][:, -in_seq:, :, 0:2]
    # test_data_y = weekend_data['y'][:, -in_seq:, :, 0:1]

    train_val_data = DataLoader(train_val_data_x, train_val_data_y, batch_size, days=days, begin=0, add_ind=True, ind=0)
    test_data = DataLoader(test_data_x, test_data_y, batch_size, days=days, begin=0, add_ind=True, ind=0)

    train_val_data.size = 288*num_weekday
    print("training under ", num_weekday, ' weekdays')
    train_val_data.xs, train_val_data.ys = train_val_data.xs[-train_val_data.size:, ...], train_val_data.ys[-train_val_data.size:, ...]
    train_val_data.ind = train_val_data.ind[0:train_val_data.size]

    train_val_data.filter_by_slice(start_point, start_point + lastinghours)

    train_val_data_x, train_val_data_y = train_val_data.xs, train_val_data.ys
    test_data_x, test_data_y = test_data.xs, test_data.ys

    train_set_size = int(train_val_data.size / 4)

    permutation = np.random.RandomState(seed=42).permutation(train_val_data.size)
    train_val_data_x, train_val_data_y = train_val_data_x[permutation], train_val_data_y[permutation]

    train_x = train_val_data_x[:3*train_set_size, ...]

    data['scaler'] = StandardScaler(mean=train_x[..., 0].mean(), std=train_x[..., 0].std())
    test_x = data['scaler'].transform(test_data_x[..., 0])

    data['test_loader'] = DataLoader(test_x, test_data_y, test_batch_size, days=days,
                                    begin=0, add_ind=True, ind=0)


    return data


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, days=288, sequence=12,
                 in_seq=12, keep_order=True, filter=0, start_point=0, lastinghours=24, missingnode=-1, norm_y=False):
    
    if (missingnode >= 0):
        global missing_node
        missing_node = missingnode
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x'][:, -in_seq:, :, 0:2]  # B T N F speed flow
        data['y_' + category] = cat_data['y'][:, :sequence, :, 0:1]

        if category == "train":
            data['scaler'] = StandardScaler(mean=cat_data['x'][..., 0].mean(), std=cat_data['x'][..., 0].std())
    for si in range(0, data['x_' + category].shape[-1]):
        scaler_tmp = StandardScaler(mean=data['x_train'][..., si].mean(), std=data['x_train'][..., si].std())
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., si] = scaler_tmp.transform(data['x_' + category][..., si])
    if (norm_y):
        for category in ['train', 'val', 'test']:
            data['y_' + category] = data['scaler'].transform(data['y_' + category])

    if (keep_order):
        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, days=days, begin=0, add_ind=True, ind=0)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size, days=days,
                                        begin=data['x_train'].shape[0], add_ind=True, ind=0)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, days=days,
                                        begin=data['x_train'].shape[0] + data['x_val'].shape[0], add_ind=True, ind=0)    
    else:
        all_data_x = np.concatenate([data['x_train'], data['x_val'], data['x_test']], 0)
        all_data_y = np.concatenate([data['y_train'], data['y_val'], data['y_test']], 0)
        all_data = DataLoader(all_data_x, all_data_y, batch_size, days=days, begin=0, add_ind=True, ind=0)

        # do the filter
        if (filter > 0):
            all_data.filter_by_slice(start_point, start_point + lastinghours)
        
        all_data.shuffle()
        test_size = int(all_data.size * 0.2)
        break1 = all_data.size - 2 * test_size
        break2 = all_data.size - test_size
        data['train_loader'] = DataLoader(all_data.xs[:break1, ...], all_data.ys[:break1, ...], batch_size, days=days, begin=0, add_ind=False, ind=all_data.ind[:break1, ...])
        data['val_loader'] = DataLoader(all_data.xs[break1:break2, ...], all_data.ys[break1:break2, ...], batch_size, days=days, begin=0, add_ind=False, ind=all_data.ind[break1:break2, ...])
        data['test_loader'] = DataLoader(all_data.xs[:test_size, ...], all_data.ys[:test_size, ...], batch_size, days=days, begin=0, add_ind=False, ind=all_data.ind[:test_size, ...])
    return data

def masked_mse(preds, labels, null_val=np.nan, x_mask=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    # mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    # strength the mask by remove the prediction from 0  => real speed
    if (len(x_mask.shape)==4):
        mask = mask * x_mask[:, :, :, 0][:, :, :, None]
    else:
        if (len(mask.shape)==3):
            mask = mask.reshape(mask.shape[0], mask.shape[1])
        mask = mask * x_mask
    mask /= torch.mean((mask))
    loss = (preds - labels) ** 2
    if (len(loss.shape)==3):
        loss = mask.reshape(loss.shape[0], loss.shape[1])
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan, x_mask=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val, x_mask=x_mask))


def masked_mae(preds, labels, null_val=np.nan, x_mask=np.nan):

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)


    if (len(mask.shape)==3):
        mask = mask.reshape(mask.shape[0], mask.shape[1])
    mask = mask * x_mask

    mask = mask.float()
    # mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.squeeze(preds) - torch.squeeze(labels))
    # if (len(loss.shape) == 3):
    #     loss = loss.reshape(loss.shape[0], loss.shape[1])
    # mask = mask[:, 0, :, 0]
    # loss = loss * mask
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan, x_mask=np.nan):

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # if not np.isnan(x_mask):
    #     mask = mask * x_mask

    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # return torch.mean(loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse

def metric_unmasked(pred, real):
    mae = masked_mae(pred, real, 1000).item()
    mape = masked_mape(pred, real, 1000).item()
    rmse = masked_rmse(pred, real, 1000).item()
    return mae, mape, rmse

def metric_strength(pred, real, x_mask):
    mae = masked_mae(pred, real, 0.0, x_mask).item()
    mape = masked_mape(pred, real, 0.0, x_mask).item()
    rmse = masked_rmse(pred, real, 0.0, x_mask).item()
    return mae, mape, rmse

def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]
    id_mat = np.identity(n_vertex)

    # D_row
    deg_mat_row = np.diag(np.sum(adj_mat, axis=1))
    # D_com
    #deg_mat_col = np.diag(np.sum(adj_mat, axis=0))

    # D = D_row as default
    deg_mat = deg_mat_row

    # wid_A = A + I
    wid_adj_mat = adj_mat + id_mat
    # wid_D = D + I
    wid_deg_mat = deg_mat + id_mat

    # Combinatorial Laplacian
    # L_com = D - A
    com_lap_mat = deg_mat - adj_mat

    if mat_type == 'id_mat':
        return id_mat
    elif mat_type == 'com_lap_mat':
        return com_lap_mat

    if (mat_type == 'sym_normd_lap_mat') or (mat_type == 'wid_sym_normd_lap_mat') or (mat_type == 'hat_sym_normd_lap_mat'):
        deg_mat_inv_sqrt = fractional_matrix_power(deg_mat, -0.5)
        wid_deg_mat_inv_sqrt = fractional_matrix_power(wid_deg_mat, -0.5)

        # Symmetric normalized Laplacian
        # For SpectraConv
        # L_sym = D^{-0.5} * L_com * D^{-0.5} = I - D^{-0.5} * A * D^{-0.5}
        sym_normd_lap_mat = np.matmul(np.matmul(deg_mat_inv_sqrt, com_lap_mat), deg_mat_inv_sqrt)

        # For ChebConv
        # wid_L_sym = 2 * L_sym / lambda_max_sym - I
        ev_max_sym = max(eigvalsh(sym_normd_lap_mat))
        wid_sym_normd_lap_mat = 2 * sym_normd_lap_mat / ev_max_sym - id_mat

        # For GCNConv
        # hat_L_sym = wid_D^{-0.5} * wid_A * wid_D^{-0.5}
        hat_sym_normd_lap_mat = np.matmul(np.matmul(wid_deg_mat_inv_sqrt, wid_adj_mat), wid_deg_mat_inv_sqrt)

        if mat_type == 'sym_normd_lap_mat':
            return sym_normd_lap_mat
        elif mat_type == 'wid_sym_normd_lap_mat':
            return wid_sym_normd_lap_mat
        elif mat_type == 'hat_sym_normd_lap_mat':
            return hat_sym_normd_lap_mat

    elif (mat_type == 'rw_normd_lap_mat') or (mat_type == 'wid_rw_normd_lap_mat') or (mat_type == 'hat_rw_normd_lap_mat'):

        deg_mat_inv = fractional_matrix_power(deg_mat, -1)
        wid_deg_mat_inv = fractional_matrix_power(wid_deg_mat, -1)

        # Random Walk normalized Laplacian
        # For SpectraConv
        # L_rw = D^{-1} * L_com = I - D^{-1} * A
        rw_normd_lap_mat = np.matmul(deg_mat_inv, com_lap_mat)

        # For ChebConv
        # wid_L_rw = 2 * L_rw / lambda_max_rw - I
        ev_max_rw = max(eigvalsh(rw_normd_lap_mat))
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / ev_max_rw - id_mat

        # For GCNConv
        # hat_L_rw = wid_D^{-1} * wid_A
        hat_rw_normd_lap_mat = np.matmul(wid_deg_mat_inv, wid_adj_mat)

        if mat_type == 'rw_normd_lap_mat':
            return rw_normd_lap_mat
        elif mat_type == 'wid_rw_normd_lap_mat':
            return wid_rw_normd_lap_mat
        elif mat_type == 'hat_rw_normd_lap_mat':
            return hat_rw_normd_lap_mat

def evaluate_model(model, loss, data_iter, zscore):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y, zscore)
            # l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)

            # add mask
            mask = np.where(y >= 0.3)
            y = y[mask]
            y_pred = y_pred[mask]

            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        #return MAE, MAPE, RMSE
        return MAE, RMSE, WMAPE


def data_transform(data_raw, data_masked, n_his, n_pred, day_slot):
    # produce data slices for x_data and y_data

    n_vertex = data_raw.shape[1]
    len_record = len(data_raw)
    num = len_record - n_his - n_his
    
    x = np.zeros([num, n_his, n_vertex, 1])
    y = np.zeros([num, n_his, n_vertex, 1])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data_masked[head: tail].reshape(n_his, n_vertex, 1)
        y[i, :, :, :] = data_raw[tail: tail + 12].reshape(n_his, n_vertex, 1)

    return x, y

def generate_mask(testx, zero_):
    # 0 represent speed, 1 represent time index
    x_mask = (testx[:, 0, :, :] >= zero_)
    x_mask = x_mask.float()
    x_mask /= torch.mean((x_mask))
    x_mask = torch.where(torch.isnan(x_mask), torch.zeros_like(x_mask), x_mask)   
    # print("min mask value: ", x_mask[:, None, :, :].min())
    return x_mask[:, None, :, :]

def slice_every_hour(start_hour, num_hr):
    return np.arange(start_hour, num_hr, 24) 


def get_filter_state(x, y, thresh):
    good_index_from = ((x <= thresh))
    good_index_to = (y <= thresh)
    good_index00 = (good_index_from * good_index_to) * 1
    good_index_from = ((x <= thresh))
    good_index_to = (y > thresh)
    good_index01 = (good_index_from * good_index_to) * 1
    good_index_from = ((x > thresh))
    good_index_to = (y <= thresh)
    good_index10 = (good_index_from * good_index_to) * 1  
    good_index_from = ((x > thresh))
    good_index_to = (y > thresh)
    good_index11 = (good_index_from * good_index_to) * 1   

    return good_index00, good_index01, good_index10, good_index11
    # for k in range(len(good_index_from[0])):
    #     self.ys[good_index_from[0][k], 0, good_index_from[1][k], 0] = 0
    # for k in range(len(good_index_to[0])):
    #     self.ys[good_index_to[0][k], 0, good_index_to[1][k], 0] = 0


def get_group_filter_state(x, y, thresh):
    gate00, gate01, gate10, gate11 = get_filter_state(x[:, -1, :, 0], y[:, 0, :, 0], thresh)
    sgate00=[gate00]
    sgate01=[gate01]
    sgate10=[gate10]
    sgate11=[gate11]
    
    for i in range(1, 12):
        gate00, gate01, gate10, gate11 = get_filter_state(y[:, i-1, :, 0], y[:, i, :, 0], thresh)
        sgate00.append(gate00)
        sgate01.append(gate01)
        sgate10.append(gate10)
        sgate11.append(gate11)

    sgate00 = torch.stack(sgate00, dim=1)
    sgate01 = torch.stack(sgate01, dim=1)
    sgate10 = torch.stack(sgate10, dim=1)
    sgate11 = torch.stack(sgate11, dim=1)

    all_gate = torch.stack([sgate00, sgate01, sgate10, sgate11], dim=-1)

    return all_gate

def print_mae_loss(pred_data, real_data, base_data, predict_point, congestion):
    p = pred_data[:, :, predict_point]
    r = real_data[:, :, predict_point]
    b = base_data[:, :, -1]

    res = [[], [], [], []]

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if (p[i, j] >= 0):
                if (r[i, j] >= 0):
                    if (b[i, j] > congestion[j] and r[i, j] > congestion[j]):
                        res[0].append(np.abs(r[i, j] - p[i, j]))
                    elif (b[i, j] > congestion[j] and r[i, j] <= congestion[j]):
                        res[1].append(np.abs(r[i, j] - p[i, j]))
                    elif (b[i, j] <= congestion[j] and r[i, j] > congestion[j]):
                        res[2].append(np.abs(r[i, j] - p[i, j]))
                    else:
                        res[3].append(np.abs(r[i, j] - p[i, j]))

    data_size = len(res[0])  + len(res[1]) + len(res[2])  + len(res[3])               
    print("state 1: both at free flow: ", len(res[0]), len(res[0])/data_size)
    print("state 2: start at free flow and end with congestion: ", len(res[1]), len(res[1])/data_size)
    print("state 3: start at congestion and end with free flow: ", len(res[2]), len(res[2])/data_size)
    print("state 4: both at congestion: ", len(res[3]), len(res[3])/data_size)

    print('MAE of state 1', mean(res[0]))
    print('MAE of state 2', mean(res[1]))
    print('MAE of state 3', mean(res[2]))
    print('MAE of state 4', mean(res[3]))
    print(mean(res[0] + res[1] + res[2] + res[3]))


def domain_masked_mae(base, preds, labels, null_val=np.nan, x_mask=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss_above_above = (loss[(base > 52) & (labels > 52)])
    loss_above_above = torch.where(torch.isnan(loss_above_above), torch.zeros_like(loss_above_above), loss_above_above)
    loss_above_below = (loss[(base > 52) & (labels <= 52)])
    loss_above_below = torch.where(torch.isnan(loss_above_below), torch.zeros_like(loss_above_below), loss_above_below)
    loss_below_above = (loss[(base <= 52) & (labels > 52)])
    loss_below_above = torch.where(torch.isnan(loss_below_above), torch.zeros_like(loss_below_above), loss_below_above)
    loss_below_below = (loss[(base <= 52) & (labels <= 52)])
    loss_below_above = torch.where(torch.isnan(loss_below_below), torch.zeros_like(loss_below_below), loss_below_below)

    return torch.mean(loss), torch.mean(loss_above_above), torch.mean(loss_above_below), torch.mean(loss_below_above), torch.mean(loss_below_below)


def domain_masked_rmse(base, preds, labels, null_val=np.nan, x_mask=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    # mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    # strength the mask by remove the prediction from 0  => real speed
    # if (x_mask != np.nan):
    #     if (len(x_mask.shape)==4):
    #         mask = mask * x_mask[:, :, :, 0][:, :, :, None]
    #     else:
    #         mask = torch.squeeze(mask)
    #         x_mask = torch.squeeze(x_mask)
    #         mask = mask * x_mask
    mask /= torch.mean((mask))
    loss = (preds - labels) ** 2
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss_above_above = (loss[(base > 52) & (labels > 52)])
    loss_above_above = torch.where(torch.isnan(loss_above_above), torch.zeros_like(loss_above_above), loss_above_above)
    loss_above_below = (loss[(base > 52) & (labels <= 52)])
    loss_above_below = torch.where(torch.isnan(loss_above_below), torch.zeros_like(loss_above_below), loss_above_below)
    loss_below_above = (loss[(base <= 52) & (labels > 52)])
    loss_below_above = torch.where(torch.isnan(loss_below_above), torch.zeros_like(loss_below_above), loss_below_above)
    loss_below_below = (loss[(base <= 52) & (labels <= 52)])
    loss_below_above = torch.where(torch.isnan(loss_below_below), torch.zeros_like(loss_below_below), loss_below_below)

    return torch.sqrt(torch.mean(loss)), torch.sqrt(torch.mean(loss_above_above)), torch.sqrt(torch.mean(loss_above_below)), torch.sqrt(torch.mean(loss_below_above)), torch.sqrt(torch.mean(loss_below_below))


def domain_masked_mape(base, preds, labels, null_val=np.nan, x_mask=np.nan):

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss_above_above = (loss[(base > 52) & (labels > 52)])
    loss_above_above = torch.where(torch.isnan(loss_above_above), torch.zeros_like(loss_above_above), loss_above_above)
    loss_above_below = (loss[(base > 52) & (labels <= 52)])
    loss_above_below = torch.where(torch.isnan(loss_above_below), torch.zeros_like(loss_above_below), loss_above_below)
    loss_below_above = (loss[(base <= 52) & (labels > 52)])
    loss_below_above = torch.where(torch.isnan(loss_below_above), torch.zeros_like(loss_below_above), loss_below_above)
    loss_below_below = (loss[(base <= 52) & (labels <= 52)])
    loss_below_above = torch.where(torch.isnan(loss_below_below), torch.zeros_like(loss_below_below), loss_below_below)

    return torch.mean(loss), torch.mean(loss_above_above), torch.mean(loss_above_below), torch.mean(loss_below_above), torch.mean(loss_below_below)



def domain_metric(base, pred, real):
    # mae = masked_mae(pred,real,0.0).item()

    # used to correct the prediction in the missing data part.
    if(len(pred) != len(real)):
        min_length = min(len(pred), len(real))
        pred, real = pred[:min_length, ...], real[:min_length, ...]
    # mae = masked_mae_no_pred_from_0(pred,real,0.0,x_mask_all)
    overall_mae, mae_ff, mae_fc, mae_cf, mae_cc = domain_masked_mae(base, pred,real,0.0)
    overall_mape, mape_ff, mape_fc, mape_cf, mape_cc = domain_masked_mape(base, pred,real,0.0)
    overall_rmse, rmse_ff, rmse_fc, rmse_cf, rmse_cc = domain_masked_rmse(base, pred,real,0.0)
    return overall_mae, overall_mape, overall_rmse, mae_ff, mae_fc, mae_cf, mae_cc, mape_ff, mape_fc, mape_cf, mape_cc,  rmse_ff, rmse_fc, rmse_cf, rmse_cc
    