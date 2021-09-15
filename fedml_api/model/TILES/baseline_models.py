#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tiantian
"""
from re import T
import pandas as pd
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn import functional as F
import pdb
import logging
from torch.nn.modules import dropout
import itertools


class OneDCnnLstm(nn.Module):
    def __init__(self, input_channel, lstm_hidden_size=128, num_layers_lstm=2, pred='motif',
                 bidirectional=True, rnn_cell='lstm', num_pred=2):
        logging.info('creating tiles model')
        super(OneDCnnLstm, self).__init__()
        self.input_channel = input_channel
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        self.num_layers_lstm = num_layers_lstm
        self.dropout_p = 0.2
        self.num_pred = num_pred
        self.rnn_input_size = 32
        self.pred = pred

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),

            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p)
        )

        self.rnn = self.rnn_cell(input_size=self.rnn_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.num_layers_lstm, batch_first=True,
                                 dropout=self.dropout_p, bidirectional=self.bidirectional)
        
        # dense input should be calculate based on the input variable later, not hard coded
        self.dense1 = nn.Linear(1920*2, 128)
        self.dense_relu1 = nn.ReLU()
        self.pred_layer = nn.Linear(128, self.num_pred) 
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, input_var, global_feature=None):

        # (B, D, T)
        # logging.info('came to training with input shape: {}'.format(input_var.shape))

        x = input_var.permute(0, 2, 1) 
        x = self.conv(x.float())
        x = x.permute(0, 2, 1)
        # logging.info('shape after convolution: {}'.format(x.shape))
        x, h_state = self.rnn(x)
        # logging.info('shape after lstm: {}'.format(x.shape))
        z = x.reshape(-1, x.shape[1]*x.shape[2])
        # logging.info('shape after reshape: {}'.format(z.shape))
        z = self.dense1(z)
        z = self.dense_relu1(z)
        preds = self.pred_layer(z)

        return preds