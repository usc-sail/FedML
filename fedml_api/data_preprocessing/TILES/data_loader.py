import json
import os
import pickle as pkl 

import torch
import numpy as np
motif_dict = {'other': 0, 'go_to_bed': 1, 'sleep': 2, 'wake_up': 3}

def read_data(train_data_path, test_data_path):
    motif_dict = {'other': 0, 'go_to_bed': 1, 'sleep': 2, 'wake_up': 3}
    train_data = pkl.load(open(train_data_path, 'rb'))
    test_data = pkl.load(open(test_data_path, 'rb'))
    train_clients = []
    test_clients = []
    train_data_out = {}
    test_data_out = {}
    # count =0 
    for client in train_data.keys():
        cdata = {}
        cdata['x'] = []
        cdata['y'] = []
        train_data_client = train_data[client]
        train_clients.append(client)
        for sample in train_data_client.keys():
            cdata['x'].append(train_data_client[sample]['data'])
            # try:
            cdata['y'].append(motif_dict[train_data_client[sample]['label']])
            # except:
            #     # print(train_data_client[sample]['label'])
            #     cdata['y'].append(4)
            #     count += 1
            #     continue
        train_data_out[client] = cdata
        # print(client)
    # print(count)
    # count = 0
    for client in test_data.keys():
        cdata = {}
        cdata['x'] = []
        cdata['y'] = []
        test_data_client = test_data[client]
        test_clients.append(client)
        for sample in test_data_client.keys():
            cdata['x'].append(test_data_client[sample]['data'])
            # try:
            cdata['y'].append(motif_dict[test_data_client[sample]['label']])
            # except:
            #     # print(test_data_client[sample]['label'])
            #     cdata['y'].append(4)
            #     count += 1
            #     continue
        test_data_out[client] = cdata
    # print(count)
    return train_clients, test_clients, train_data_out, test_data_out
        



def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data

def load_partition_data_tiles(batch_size,
                              train_path='/data/rash/tiles-motif/data/training_with_aug.pkl',
                              test_path='/data/rash/tiles-motif/data/validation.pkl'):
    train_clients, test_clients, train_data, test_data = read_data(train_path, test_path)
    # return read_data(train_path, test_path)
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    for u in train_clients:
        client_train_data_num = len(train_data[u]['x'])
        train_data_num += client_train_data_num
        train_data_local_num_dict[client_idx] = client_train_data_num
        train_batch = batch_data(train_data[u], batch_size)
        train_data_local_dict[client_idx] = train_batch
        train_data_global += train_batch
        client_idx += 1
    
    client_idx = 0
    for u in test_clients:
        client_test_data_num = len(test_data[u]['x'])
        test_data_num += client_test_data_num
        test_batch = batch_data(test_data[u], batch_size)
        test_data_local_dict[client_idx] = test_batch
        test_data_global += test_batch
        client_idx += 1
    class_num = 4

    return train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

def load_data_tiles(batch_size, data_path='/data/rash/tiles-motif/data/test.pkl'):
    
    # read data
    data = pkl.load(open(data_path, 'rb'))
    data_dict = {}
    data_dict['x'] = list()
    data_dict['y'] = list()
    for client in data.keys():
        for sample in data[client].keys():
            data_dict['x'].append(data[client][sample]['data'])
            data_dict['y'].append(motif_dict[data[client][sample]['label']])
    return batch_data(data_dict, batch_size)
