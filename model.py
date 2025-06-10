from __future__ import division
from __future__ import print_function
import math
from math import sqrt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import kan
import torch.nn.init as init
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
import numpy as np
import pandas as pd
import numpy as np
import math
import pandas as pd
from typing import Text, Union
import copy
import tensorflow as tf
# from ...utils import get_or_create_path
# from ...log import get_module_logger
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# from .pytorch_utils import count_parameters
# from ...model.base import Model
# from ...data.dataset import DatasetH
# from ...data.dataset.handler import DataHandlerLP
from tcn.tcn import TCN
class AddNorm(nn.Module):
    def __init__(self, normalize_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(normalize_shape)

    def forward(self, X, Y):
        Y = self.dropout(Y) + X
        return self.normalize(Y)


class FFN(nn.Module):
    # def __init__(self, ffn_inputs, ffn_hiddens=1024, ffn_outputs=256):
    def __init__(self, ffn_inputs, ffn_hiddens=1024, ffn_outputs=256):
        super().__init__()
        self.dense1 = nn.Linear(ffn_inputs, ffn_hiddens)
        # self.dense3 = nn.Linear(ffn_hiddens, 1024)
        self.dense2 = nn.Linear(ffn_hiddens, ffn_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.dense2(self.relu(self.dense1(X)))
        return X


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout=0.0, desc='enc'):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.normalize = AddNorm(num_hiddens, dropout)
        self.desc = desc

        # define q,k,v linear layer
        self.Wq = nn.Linear(self.num_hiddens, self.num_hiddens)
        self.Wk = nn.Linear(self.num_hiddens, self.num_hiddens)
        self.Wv = nn.Linear(self.num_hiddens, self.num_hiddens)

        self.relu = nn.ReLU()
        self.Q = nn.Sequential(self.Wq, self.relu)
        self.K = nn.Sequential(self.Wk, self.relu)
        self.V = nn.Sequential(self.Wv, self.relu)

    def forward(self, queries, keys, values):
        # get matrices of q, k, v
        q, k, v = self.Q(queries), self.K(keys), self.V(values)
        q_split = q.unsqueeze(1).chunk(self.num_heads, dim=-1)
        k_split = k.unsqueeze(1).chunk(self.num_heads, dim=-1)
        v_split = v.unsqueeze(1).chunk(self.num_heads, dim=-1)

        q_stack = torch.stack(q_split, dim=1)
        k_stack = torch.stack(k_split, dim=1)
        v_stack = torch.stack(v_split, dim=1)

        score = torch.matmul(q_stack, k_stack.permute(0, 1, 3, 2))
        score = score / (k_stack.size()[-1] ** 0.5)
        score = F.softmax(score, dim=-1)
        a = torch.matmul(score, v_stack)
        a = torch.reshape(a.permute(0, 1, 3, 2), shape=(q.size(0), q.size(1)))
        a += queries
        return a
class MultiHeadAttention2(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout=0.0, desc='enc'):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.normalize = AddNorm(num_hiddens, dropout)
        self.desc = desc

        # define q,k,v linear layer
        self.Wq = nn.Linear(self.num_hiddens, self.num_hiddens)
        self.Wk = nn.Linear(self.num_hiddens, self.num_hiddens)
        self.Wv = nn.Linear(self.num_hiddens, self.num_hiddens)

        self.relu = nn.ReLU()
        self.Q = nn.Sequential(self.Wq, self.relu)
        self.K = nn.Sequential(self.Wk, self.relu)
        self.V = nn.Sequential(self.Wv, self.relu)

    def forward(self, queries, keys, values):
        # get matrices of q, k, v
        q, k, v = self.Q(queries), self.K(keys), self.V(values)
        q_split = q.unsqueeze(2)
        k_split = k.unsqueeze(2)
        v_split = v.unsqueeze(2)
        #
        # q_stack = torch.stack(q_split, dim=2)
        # k_stack = torch.stack(k_split, dim=2)
        # v_stack = torch.stack(v_split, dim=2)

        score = torch.matmul(q_split, k_split.permute(0, 2, 1))
        score = score / (k.size()[-1] ** 0.5)
        score = F.softmax(score, dim=-1)
        a = torch.matmul(score, v_split)
        a = a.squeeze(2)        # a = torch.reshape(a.permute(0, 1, 3, 2), shape=(q.size(0), q.size(1)))
        a += queries
        return a
class BridgeTowerBlock(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout_rate):
        super().__init__()
        self.self_attetion_1 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)
        self.self_attetion_2 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)

        self.cross_attetion_1 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)
        self.cross_attetion_2 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)

        self.bridge_layer_1 = nn.Linear(num_hiddens, num_hiddens)
        self.bridge_layer_2 = nn.Linear(num_hiddens, num_hiddens)

        self.ffn_1 = FFN(num_hiddens)
        self.ffn_2 = FFN(num_hiddens)

        self.AddNorm1 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm2 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm3 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm4 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm5 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm6 = AddNorm(num_hiddens, dropout=dropout_rate)

    def forward(self, modality_1, modality_2):
        # modality_1 = self.bridge_layer_1(modality_1)
        # modality_2 = self.bridge_layer_2(modality_2)
        #
        # output_attn_1 = self.self_attetion_1( modality_1,  modality_1,  modality_1)
        # output_attn_2 = self.self_attetion_2( modality_2,  modality_2,  modality_2)
        # #
        # modality_1 = self.AddNorm1(modality_1, output_attn_1)
        # modality_2 = self.AddNorm2(modality_2, output_attn_2)

        output_attn_1 = self.cross_attetion_1(modality_1, modality_2,modality_2)
        output_attn_2 = self.cross_attetion_2(modality_2,modality_2,modality_1)
        #
        modality_1 = self.AddNorm3(modality_1, output_attn_1)
        modality_2 = self.AddNorm4(modality_2, output_attn_2)

        output_attn_1 = self.self_attetion_1( modality_1,  modality_1,  modality_1)
        output_attn_2 = self.self_attetion_2( modality_2,  modality_2,  modality_2)

        # modality_1 = self.AddNorm1(modality_1, output_attn_1)
        # modality_2 = self.AddNorm2(modality_2, output_attn_2)
        output1 = self.ffn_1(output_attn_1)
        output2 = self.ffn_2(output_attn_2)
        #
        output1 = self.AddNorm5(output_attn_1, output1)
        output2 = self.AddNorm6(output_attn_2, output2)

        return  output1,  output2

class BridgeTowerBlock2(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout_rate):
        super().__init__()
        self.self_attetion_1 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)
        self.self_attetion_2 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)

        self.cross_attetion_1 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)
        self.cross_attetion_2 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)

        self.bridge_layer_1 = nn.Linear(num_hiddens, num_hiddens)
        self.bridge_layer_2 = nn.Linear(num_hiddens, num_hiddens)

        self.ffn_1 = FFN(num_hiddens)
        self.ffn_2 = FFN(num_hiddens)

        self.AddNorm1 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm2 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm3 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm4 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm5 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm6 = AddNorm(num_hiddens, dropout=dropout_rate)

    def forward(self, modality_1, modality_2):
        # modality_1 = self.bridge_layer_1(modality_1)
        # modality_2 = self.bridge_layer_2(modality_2)
        #
        # output_attn_1 = self.self_attetion_1( modality_1,  modality_1,  modality_1)
        # output_attn_2 = self.self_attetion_2( modality_2,  modality_2,  modality_2)
        # #
        # modality_1 = self.AddNorm1(modality_1, output_attn_1)
        # modality_2 = self.AddNorm2(modality_2, output_attn_2)

        output_attn_1 = self.cross_attetion_1(modality_1, modality_2,modality_2)
        output_attn_2 = self.cross_attetion_2(modality_2,modality_2,modality_1)
        #
        x_test_cross=torch.cat((output_attn_1,output_attn_2),dim=-1)
        np.save('layers_features/x_test_cross.npy', x_test_cross.cpu().numpy())
        # torch.save(output_attn_2, 'x_2_cross.numpy')
        modality_1 = self.AddNorm3(modality_1, output_attn_1)
        modality_2 = self.AddNorm4(modality_2, output_attn_2)
        x_test_cross_add=torch.cat((modality_1 ,modality_2),dim=-1)
        np.save('layers_features/x_test_cross_add.npy', x_test_cross_add.cpu().numpy())
        # torch.save(modality_1,'x_1_cross_add.numpy')
        # torch.save(modality_2, 'x_2_cross_add.numpy')
        output_attn_1 = self.self_attetion_1( modality_1,  modality_1,  modality_1)
        output_attn_2 = self.self_attetion_2( modality_2,  modality_2,  modality_2)
        x_test_cross_add_att=torch.cat((output_attn_1 ,output_attn_2),dim=-1)
        np.save('layers_features/x_test_cross_add_att.npy', x_test_cross_add_att.cpu().numpy())
        # torch.save(output_attn_1,'x_1_cross_add_self.numpy')
        # torch.save(output_attn_2, 'x_2_cross_add_self.numpy')
        # modality_1 = self.AddNorm1(modality_1, output_attn_1)
        # modality_2 = self.AddNorm2(modality_2, output_attn_2)
        output1 = self.ffn_1(output_attn_1)
        output2 = self.ffn_2(output_attn_2)
        x_test_cross_add_att_ffn=torch.cat((output1,output2),dim=-1)
        np.save('layers_features/x_test_cross_add_att_ffn.npy', x_test_cross_add_att_ffn.cpu().numpy())
        # torch.save(output1,'x_1_cross_add_self_ffn.numpy')
        # torch.save(output2, 'x_2_cross_add_self_add_ffn.numpy')
        #
        output1 = self.AddNorm5(output_attn_1, output1)
        output2 = self.AddNorm6(output_attn_2, output2)
        x_test_cross_add_att_ffn_add=torch.cat((output1,output2),dim=-1)
        np.save('layers_features/x_test_cross_add_att_ffn_add.npy', x_test_cross_add_att_ffn_add.cpu().numpy())
        # torch.save(output1,'x_1_cross_add_self_ffn_add.numpy')
        # torch.save(output2, 'x_2_cross_add_self_add_ffn_add.numpy')
        return  output1,  output2

d2_times = 2
class model_A(nn.Module):
    def __init__(self, input_shape_1d, input_shape_2d, input_shape_esm, lstm_hidden, d_another_h, d_model, dim_in,
                 output_dim, dropout):
        # def __init__(self,input_shape_2d, d_another_h,  output_dim):
        super(model_A, self).__init__()
        "****************cnn******************"
        self.convs_1 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=input_shape_2d,
                                    out_channels=1108,
                                    kernel_size=h),
                          nn.BatchNorm1d(num_features=int(1108)),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=100 - h + 1))
            for h in [3, 5, 7, 9, 11]
            # for h in [2,3,4, 5, 6]
        ])
        self.contraLinear1 = nn.Linear(int(input_shape_2d * 2), int(lstm_hidden*2))
        self.contraLinear2 = nn.Linear(int(lstm_hidden*2), int(lstm_hidden*4))
        # self.contraLinear_all = nn.Linear(int(1108+144), int(lstm_hidden * 2))
        self.contraLinear_all = nn.Linear(int(1108+144), int(lstm_hidden * 2))
        self.contraLinear_all_ = nn.Linear(int(144), int(lstm_hidden * 2))
        self.relu = nn.ReLU()
        self.maxpool_1 = nn.MaxPool1d(kernel_size=5)
        self.fc = nn.Linear(d_another_h, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.Layer_norm = nn.LayerNorm(lstm_hidden*2)
        self.Layer_norm2 = nn.LayerNorm(lstm_hidden * 2)

        "****************cnn******************"
        self.convs_esm = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=input_shape_esm,
                                    out_channels=d_another_h,
                                    kernel_size=h,
                                    padding='same'),
                          nn.BatchNorm1d(num_features=d_another_h),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=100 - h + 1))
            for h in [3]
        ])

        self.maxpool_1 = nn.MaxPool1d(kernel_size=5)
        self.fc = nn.Linear(d_another_h, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.Layer_norm_1d = nn.LayerNorm(144)
        self.Layer_norm = nn.LayerNorm(lstm_hidden * 2)
        self.Layer_norm2 = nn.LayerNorm(lstm_hidden * 2)
        self.batchnorm1d = nn.BatchNorm1d(int(1108))
        "****************cnn******************"

        "****************cnn+cnn******************"
        self.cov_1 = nn.Conv1d(in_channels=input_shape_esm,out_channels=int(input_shape_esm/2), kernel_size=3, padding='same')
        self.cov_2 = nn.Conv1d(in_channels=int(input_shape_esm/2), out_channels=int(input_shape_esm / 4), kernel_size=1, padding='same')
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.pooling = nn.MaxPool1d(kernel_size=1, stride=1)
        "****************cnn+cnn******************"

        self.conv1d_esm = nn.Conv1d(in_channels=input_shape_esm, out_channels=320, kernel_size=3,padding=1)
        "****************三个特征提取方式******************"
        self.lstm1 = nn.LSTM(input_size=input_shape_1d,  # 1141
                             hidden_size=int(72),
                             # hidden_size=int(64),
                             num_layers=1,
                             batch_first=True,
                             bidirectional=True)

        self.lstm2 = nn.LSTM(input_size=input_shape_2d,
                             hidden_size=128,
                             num_layers=1,
                             batch_first=True,
                             bidirectional=True)

        self.lstm_esm = nn.LSTM(input_size=input_shape_esm,  # 1141
                             hidden_size=lstm_hidden,  # 64,128,256,512,1024
                             num_layers=1,
                             batch_first=True,
                             bidirectional=True)

        "****************transformer******************"
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_shape_esm, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=3)
        "****************transformer******************"

        "****************rnn******************"
        self.rnn = nn.RNN(input_size=input_shape_1d,
                          hidden_size=d_model,
                          num_layers=1,
                          batch_first=True)
        "****************rnn******************"
        "****************rnn(esm)******************"
        self.rnn_esm = nn.RNN(input_size=input_shape_esm,
                          hidden_size=int(input_shape_esm/2),
                          num_layers=1,
                          batch_first=True)
        "****************rnn(esm)******************"

        "****************gru******************"
        self.gru_esm = nn.GRU(input_shape_esm, int(input_shape_esm / 8), batch_first=True, bidirectional=True)
        "****************gru******************"
        "****************gru******************"
        self.gru_1d = nn.GRU(input_shape_1d, int(d_model / 2), batch_first=True, bidirectional=True)
        "****************gru******************"
        "*************Attention*********************"
        self.self_attetion_1 = MultiHeadAttention(num_hiddens=int(input_shape_2d*2), num_heads=8)
        self.self_attetion_2 = MultiHeadAttention(num_hiddens=int(lstm_hidden*2), num_heads=8)
        self.self_attetion_all = MultiHeadAttention(num_hiddens=int(input_shape_2d * 2+input_shape_1d/8+lstm_hidden*2), num_heads=4)

        self.layer = nn.Linear(int(input_shape_esm / 4), 256)
        self.layer_1 = nn.Linear(int(input_shape_2d *2), int(input_shape_2d*2))
        self.layer_2 = nn.Linear(int(input_shape_2d *2), int(input_shape_2d*2))
        self.layer_3 = nn.Linear(int(lstm_hidden*2), int(lstm_hidden*2))
        self.layer_4 = nn.Linear(int(lstm_hidden*2), int(lstm_hidden*2))
        self.layer_5 = nn.Linear(int(input_shape_esm / 4), int(input_shape_esm / 4))
        self.layer_6 = nn.Linear(int(input_shape_esm / 4), int(input_shape_esm / 4))
        self.layer_all_1 = nn.Linear(int(input_shape_2d * 2+input_shape_1d/8+lstm_hidden*2), int(input_shape_2d * 2+input_shape_1d/8+lstm_hidden*2))
        self.layer_all_2 = nn.Linear(int(input_shape_2d * 2+input_shape_1d/8+lstm_hidden*2), int(input_shape_2d * 2+input_shape_1d/8+lstm_hidden*2))
        "*************Attention*********************"

        self.block3 = nn.Sequential(

            nn.Linear(input_shape_esm, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.1))
        "**********************************"
        self.block1 = nn.Sequential(
            # 64,128,256,512,1024
            nn.Linear(256, 1)
            )
        # nn.Linear(512, 64),
        # nn.ReLU(),
        # nn.Dropout(0.1))
        # nn.Linear(512, 256),
        # nn.ReLU(),
        # nn.Dropout(0.1),

        "*************bridge_tower*********************"
        self.bridge_tower = nn.ModuleList(BridgeTowerBlock(num_hiddens=256,
                                                           num_heads=8,
                                                           dropout_rate=dropout)  # 本来是0.1
                                          for _ in range(1))

        "*************bridge_tower*********************"



        # self.Wq = nn.Linear(d_model, d_model)  # 不改变形状的线性变换
        # self.Wk = nn.Linear(d_model, d_model)
        # self.Wv = nn.Linear(d_model, d_model)
        # self.att_weights = nn.Parameter(torch.Tensor(256, 1))
        # self.softmax = nn.Softmax(dim=1)
        # #
        # self.linear_q = nn.Linear(self.dim_in, self.dim_k, bias=False)
        # self.linear_k = nn.Linear(self.dim_in, self.dim_k, bias=False)
        # self.linear_v = nn.Linear(self.dim_in, self.dim_v, bias=False)
        # self._norm_fact = 1 / sqrt(self.dim_k // self.num_heads)
        #
        # self.dim_in = dim_in
        # self.num_heads = 8
        # self.dim_k = dim_in
        # self.dim_v = dim_in
        #
        # self.linear_q = nn.Linear(dim_in, dim_in)
        # self.linear_k = nn.Linear(dim_in,dim_in)
        # self.linear_v = nn.Linear(dim_in,dim_in)
        # self._norm_fact = 1.0 / (dim_in ** 0.5)  # Normalization factor for attention scores

        self.block = nn.Sequential(nn.Linear(896, 512),  # 2 * hidden_size for BiLSTM
                                   nn.BatchNorm1d(512),
                                   #nn.LayerNorm(1024),
                                   # nn.ReLU(),
                                   # nn.ELU(),
                                   # nn.PReLU(),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.1),
                                   # nn.Linear(512, 256),  # 2 * hidden_size for BiLSTM
                                   # nn.BatchNorm1d(256),
                                   # nn.LeakyReLU(),
                                   # nn.Dropout(0.1),
                                   nn.Linear(512, 256),
                                   # nn.LeakyReLU(),
                                   nn.Linear(256, 1),
                                   # nn.BatchNorm1d(64),
                                   )

        self.classifier = nn.Sequential(

            nn.Linear(64, 1),
            # nn.Linear(256, 64),
            # nn.ReLU(),
            # nn.ELU() ,
            # nn.LeakyReLU(),
            # nn.Dropout(0.1))
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(64, 1),
            # nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.fcn = nn.Sequential(nn.Linear(int(input_shape_esm / 4), int(input_shape_esm / 4)),
                  nn.ReLU(), nn.Linear(int(input_shape_esm / 4), 1))

        self.fc_trans = nn.Sequential(

            nn.Linear(input_shape_esm, 1024),
            # nn.ReLU(),
            nn.LeakyReLU(),
            #nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1))

        self.fcn_bilstm = nn.Sequential(nn.Linear(lstm_hidden*2, int(lstm_hidden)),
                                 nn.ReLU(), nn.Linear(int(lstm_hidden), 1))
        # self.fcn_bilstm_2d = nn.Sequential(nn.Linear(lstm_hidden * 2, int(lstm_hidden)),
        #                                 nn.ReLU(), nn.Linear(int(lstm_hidden), 1))
        self.fcn_gru = nn.Sequential(nn.Linear(int(input_shape_esm/4), int(input_shape_esm/8)),
                                        nn.ReLU(), nn.Linear(int(input_shape_esm/8), 1))
        self.fcn_rnn = nn.Sequential(nn.Linear(int(input_shape_esm/2), int(input_shape_esm / 2)),
                  nn.ReLU(), nn.Linear(int(input_shape_esm / 2), 1))
        self.fcn_multi_cnn = nn.Sequential(nn.Linear(int(input_shape_esm / 2), int(input_shape_esm / 2)),
                                     nn.ReLU(), nn.Linear(int(input_shape_esm / 2), 1))

        self.fcn_multi_cnn_2d = nn.Sequential(nn.Linear(int(input_shape_2d * d2_times), int(input_shape_2d *d2_times)),
                                       nn.ReLU(), nn.Linear(int(input_shape_2d * d2_times), 1))
        self.tcn = TCN(nb_filters=128, kernel_size=5, dropout_rate=0.3, nb_stacks=1, dilations=[1, 2, 4, 8],
                return_sequences=True, activation='relu', padding='same', use_skip_connections=True)

        self.fcn_block = nn.Sequential(nn.Linear(int(lstm_hidden * 2 + input_shape_2d * d2_times + input_shape_1d // 8),
                                                 int(lstm_hidden * 2 + input_shape_2d * d2_times + input_shape_1d // 8)),
                                       nn.ReLU(), nn.Linear(
                int((lstm_hidden * 2 + input_shape_2d * d2_times + input_shape_1d // 8)), 1))

        self.fcn_block2 = nn.Sequential(nn.Linear(int(lstm_hidden * 4),
                                             int(lstm_hidden *4)),
                                   nn.ReLU(),nn.Linear(
            int((lstm_hidden * 4)), 1))
        # self.kan = kan.KAN([28 * 28, 64, 10],device='cpu')

    #     self.apply(self.init_weights)
    #
    # def init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Conv1d):
    #         init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Conv2d):
    #         init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LSTM):
    #         for name, param in m.named_parameters():
    #             if 'weight' in name:
    #                 if 'ih' in name:
    #                     init.xavier_uniform_(param)
    #                 elif 'hh' in name:
    #                     init.orthogonal_(param)
    #             elif 'bias' in name:
    #                 init.constant_(param, 0)
    #     elif isinstance(m, nn.TransformerEncoderLayer):
    #         for param in m.parameters():
    #             if len(param.shape) >= 2:
    #                 init.xavier_uniform_(param)
    #             elif len(param.shape) == 1:
    #                 init.normal_(param, mean=0, std=0.01)
    #     elif isinstance(m, nn.GRU):
    #         for name, param in m.named_parameters():
    #             if 'weight' in name:
    #                 init.xavier_uniform_(param)
    #             elif 'bias' in name:
    #                 init.constant_(param, 0)


    def forward(self, x_1d, x_2d, x_esm):
        '--------------------------------1d------------------------------------'
        '*****1d+bilstm*****'
        # lstm_out, _ = self.lstm1(x_1d.float())
        # x_1d = self.Layer_norm_1d(lstm_out)
        # x_all = torch.cat((x_2d, x_1d_1), dim=1)
        #
        # lstm_out, _ = self.lstm3(x_esm.float())
        # x_1d = self.Layer_norm2(lstm_out)
        '*****1d+bilstm*****'

        '*****1d+cnn(320d)*****'
        # x = x_1d.permute(0, 2, 1)
        # x = self.conv1d(x)
        # x = self.relu(x)
        # x = self.pooling(x)
        # x = x.view(-1, x.size(1))
        '*****1d+cnn(320d)*****'

        '*****1d+transformer*****'
        # x_1d = self.transformer(x_1d)
        # x_1d = self.block3(x_1d)#64维度
        '*****1d+transformer*****'

        '*****1d+multi-cnn*****'
        # x_2d = x_2d.permute(0, 2, 1)
        # x_2d = x_2d.float()
        # out_2 = [conv(x_2d) for conv in self.convs_1]
        # out_2 = torch.cat(out_2, dim=2)
        # out_2 = self.maxpool_1(out_2)
        # x_2d = out_2.view(-1, out_2.size(1))#256维度
        # x_2d = self.batchnorm1d(x_2d)
        # nn.ReLU()
        # nn.Dropout(0.1)
        '*****1d+multi-cnn*****'
        '--------------------------------1d------------------------------------'

        '--------------------------------2d------------------------------------'
        '*****2d+bilstm*****'
        # lstm_out, _ = self.lstm2(x_2d)
        # # x= self.block(lstm_out)
        # forward = lstm_out[:, -1, :128]
        # backward = lstm_out[:, 0, 128:]
        # x_2d = torch.cat((forward, backward), dim=1)
        # x_2d = self.Layer_norm(x_2d)  # [batch_size, 128]
        '*****2d+bilstm*****'

        '*****2d+multi-cnn*****'
        # x_2d = x_2d.permute(0, 2, 1)
        # x_2d = x_2d.float()
        # out_2 = [conv(x_2d) for conv in self.convs_1]
        # out_2 = torch.cat(out_2, dim=2)
        # out_2 = self.maxpool_1(out_2)
        # x_2d = out_2.view(-1, out_2.size(1))#256维度
        # x_2d = self.batchnorm1d(x_2d)
        # x_2d_1 = self.layer_1(x_2d)
        # x_2d_2 = self.layer_2(x_2d)
        # x_2d = self.self_attetion_1(x_2d, x_2d_1, x_2d_2)
        # x_2d = self.contraLinear1(x_2d)
        # nn.ReLU()
        # nn.Dropout(0.1)
        '*****2d+multi-cnn*****'

        '*****2d+transformer*****'
        # x_esm = self.transformer(x_esm)
        # x_esm = self.block3(x_esm)#64维度
        '*****2d+cnn-cnn*****'
        # x_esm = x_esm.unsqueeze(2)
        # # x_esm = x_esm.permute(0, 2, 1)
        # x_esm = F.relu(self.cov_1(x_esm))
        # x_esm = F.relu(self.cov_2(x_esm))
        # x_esm = x_esm.reshape(x_esm.size(0), -1)
        # '*****esm+cnn-cnn*****'
        # x_esm_1 = self.layer_1(x_esm)
        # x_esm_2 = self.layer_2(x_esm)
        # x_esm = self.self_attetion_1(x_esm,x_esm_1,x_esm_2)
        # 64维度

        # x_esm_1 = self.layer_3(x_esm)
        # x_esm_2 = self.layer_4(x_esm)
        # x_esm = self.self_attetion_2(x_esm,x_esm_1,x_esm_2)
        '*****2d+gru*****'
        # gru_out, _ = self.gru_esm(x_esm.float())
        # # x_esm = self.Layer_norm2(lstm_out)
        # x_esm = gru_out
        # x_esm_1 = self.layer_5(x_esm)
        # x_esm_2 = self.layer_6(x_esm)
        # x_esm = self.self_attetion_1(x_esm, x_esm_1, x_esm_2)
        '*****2d+rnn*****'
        # rnn_out, _ = self.rnn_esm(x_esm.float())
        # # x_esm = self.Layer_norm2(lstm_out)
        # x_esm = rnn_out
        '*****2d+multi-cnn*****'
        # x_esm = x_esm.unsqueeze(2)
        # x_esm = x_esm.float()
        # out_2 = [conv(x_esm) for conv in self.convs_esm]
        # out_2 = torch.cat(out_2, dim=2)
        # out_2 = self.maxpool_1(out_2)
        # x_esm = out_2.view(-1, out_2.size(1))#256维度
        # x_esm = self.batchnorm1d(x_esm)
        # nn.ReLU()

        # nn.Dropout(0.1)
        # '*****esm+cnn(320d)*****'
        # x = x_esm.unsqueeze(2)
        # x = self.conv1d_esm(x)
        # x = self.relu(x)
        # x = self.pooling(x)
        # x_esm = x.view(-1, x.size(1))
        # x_esm_1 = self.layer_1(x_esm)
        # x_esm_2 = self.layer_2(x_esm)
        # x_esm = self.self_attetion_1(x_esm,x_esm_1,x_esm_2)
        '*****1d+cnn(320d)*****'

        '--------------------------------2d------------------------------------'

        '--------------------------------esm------------------------------------'
        '*****esm+transformer*****'
        # x_esm = self.transformer(x_esm)
        # x_esm = self.block3(x_esm)#64维度
        '*****esm+cnn-cnn*****'
        # x_esm = x_esm.unsqueeze(2)
        # # x_esm = x_esm.permute(0, 2, 1)
        # x_esm = F.relu(self.cov_1(x_esm))
        # x_esm = F.relu(self.cov_2(x_esm))
        # x_esm = x_esm.reshape(x_esm.size(0), -1)
        # '*****esm+cnn-cnn*****'
        # x_esm_1 = self.layer_1(x_esm)
        # x_esm_2 = self.layer_2(x_esm)
        # x_esm = self.self_attetion_1(x_esm,x_esm_1,x_esm_2)
        #64维度
        '*****esm+bilstm*****'
        lstm_out, _ = self.lstm_esm(x_esm.float())
        x_esm = self.Layer_norm2(lstm_out)
        # x_esm_1 = self.layer_3(x_esm)
        # x_esm_2 = self.layer_4(x_esm)
        # x_esm = self.self_attetion_2(x_esm,x_esm_1,x_esm_2)

        # tf_tensor_cpu = x_2d.numpy()
        # # 自动将其从 GPU 转到 CPU
        # x_2d = torch.from_numpy(tf_tensor_cpu)
        # x_2d = x_2d.view(x_2d.size(0), -1)

        x_all=torch.cat((x_2d,x_1d),dim=1)
        # x_all = x_1d
        # x_esm = self.contraLinear2(x_esm)
        x_all = self.contraLinear_all(x_all)
        # x_esm_1 = self.layer_3(x_esm)
        # x_esm_2 = self.layer_4(x_esm)
        # x_esm = self.self_attetion_2(x_esm,x_esm_1,x_esm_2)
        # x_esm = lstm_out
        # x_esm_1 = self.layer_3(x_esm)
        # x_esm_2 = self.layer_4(x_esm)
        # x_esm = self.self_attetion_2(x_esm,x_esm_1,x_esm_2)
        '*****esm+gru*****'
        # gru_out, _ = self.gru_esm(x_esm.float())
        # # x_esm = self.Layer_norm2(lstm_out)
        # x_esm = gru_out
        # x_esm_1 = self.layer_5(x_esm)
        # x_esm_2 = self.layer_6(x_esm)
        # x_esm=self.self_attetion_1(x_esm,x_esm_1,x_esm_2)
        '*****esm+rnn*****'
        # rnn_out, _ = self.rnn_esm(x_esm.float())
        # # x_esm = self.Layer_norm2(lstm_out)
        # x_esm = rnn_out
        '*****esm+multi-cnn*****'
        # x_esm = x_esm.unsqueeze(2)
        # x_esm = x_esm.float()
        # out_2 = [conv(x_esm) for conv in self.convs_esm]
        # out_2 = torch.cat(out_2, dim=2)
        # out_2 = self.maxpool_1(out_2)
        # x_esm = out_2.view(-1, out_2.size(1))#256维度
        # x_esm = self.batchnorm1d(x_esm)
        # nn.ReLU()

        # nn.Dropout(0.1)
        # '*****esm+cnn(320d)*****'
        # x = x_esm.unsqueeze(2)
        # x = self.conv1d_esm(x)
        # x = self.relu(x)
        # x = self.pooling(x)
        #  x_esm = x.view(-1, x.size(1))
        # x_esm_1 = self.layer_1(x_esm)
        # x_esm_2 = self.layer_2(x_esm)
        # x_esm = self.self_attetion_1(x_esm,x_esm_1,x_esm_2)
        '*****1d+cnn(320d)*****'


        '--------------------------------esm------------------------------------'

        '************************+1d+2d+bridge_tower**************'
        for layer in self.bridge_tower:
            x_all, x_esm = layer(x_all, x_esm)
        x = torch.cat((x_all, x_esm), dim=-1)

        # return x_1d, x_2d
        # return x_esm

        return x

    def trainModel(self, x_1d, x_2d=None, x_esm=None):
        # x_all, x_2d = self.forward(x_1d, x_2d, x_esm)
        # x = torch.cat((x_2d, x_all), dim=-1)
        # x = self.block(x)
        # # x = self.classifier(x)
        # # nn.ReLU()
        # # nn.Dropout(0.7)
        # x = self.sigmoid(x)
        # return x

        # x_1d,x_2d,x_esm= self.forward(x_1d, x_2d, x_esm)
        x= self.forward(x_1d, x_2d, x_esm)
        # for layer in self.bridge_tower:
        #     x_1d, x_2d = layer(x_esm, x_2d)
        # for layer in self.bridge_tower:
        #     x_2d,x_esm = layer(x_2d, x_esm)
        # x = torch.cat((x_all, x_esm), dim=-1)
        # x_1 = self.layer_all_1(x)
        # x_2 = self.layer_all_2(x)
        # x= self.self_attetion_all(x,x_1,x_2)

        'cnn-cnn(esm)'
        # x = self.fcn(x_all)
        'transofrmer(esm)'
        # x = self.fc_trans(x_all)
        'bilstm(esm)'
        # x = self.fcn_bilstm(x_all)
        'gru(esm)'
        #x = self.fcn_gru(x_all)
        'gru(esm)'
        # x = self.fcn_rnn(x_all)
        'multi-cnn(esm)'
        # x = self.fcn_multi_cnn(x_all)
        'multi-cnn(2d)'
        x = self.fcn_block2(x)
        return x

    def get_output(self, x_1d, x_2d, x_esm):
        x = self.forward(x_1d, x_2d, x_esm)
        x= self.fcn_block2[0](x)
        return x

# class model_A(nn.Module):
#     def __init__(self, input_shape_1d, input_shape_2d, input_shape_esm, lstm_hidden, d_another_h, d_model, dim_in,
#                  output_dim, dropout):
#         # def __init__(self,input_shape_2d, d_another_h,  output_dim):
#         super(model_A, self).__init__()
#         "****************cnn******************"
#         self.convs_1 = nn.ModuleList([
#             nn.Sequential(nn.Conv1d(in_channels=input_shape_2d,
#                                     out_channels=1108,
#                                     kernel_size=h),
#                           nn.BatchNorm1d(num_features=int(1108)),
#                           nn.ReLU(),
#                           nn.MaxPool1d(kernel_size=100 - h + 1))
#             for h in [3, 5, 7, 9, 11]
#             # for h in [2,3,4, 5, 6]
#         ])
#         self.contraLinear1 = nn.Linear(int(input_shape_2d * 2), int(lstm_hidden*2))
#         self.contraLinear2 = nn.Linear(int(lstm_hidden*2), int(lstm_hidden*4))
#         self.contraLinear_all = nn.Linear(int(1108), int(lstm_hidden * 2))
#         self.relu = nn.ReLU()
#         self.maxpool_1 = nn.MaxPool1d(kernel_size=5)
#         self.fc = nn.Linear(d_another_h, output_dim)
#         self.sigmoid = nn.Sigmoid()
#         self.Layer_norm = nn.LayerNorm(lstm_hidden*2)
#         self.Layer_norm2 = nn.LayerNorm(lstm_hidden * 2)
#
#         "****************cnn******************"
#         self.convs_esm = nn.ModuleList([
#             nn.Sequential(nn.Conv1d(in_channels=input_shape_esm,
#                                     out_channels=d_another_h,
#                                     kernel_size=h,
#                                     padding='same'),
#                           nn.BatchNorm1d(num_features=d_another_h),
#                           nn.ReLU(),
#                           nn.MaxPool1d(kernel_size=100 - h + 1))
#             for h in [3]
#         ])
#
#         self.maxpool_1 = nn.MaxPool1d(kernel_size=5)
#         self.fc = nn.Linear(d_another_h, output_dim)
#         self.sigmoid = nn.Sigmoid()
#         self.Layer_norm_1d = nn.LayerNorm(144)
#         self.Layer_norm = nn.LayerNorm(lstm_hidden * 2)
#         self.Layer_norm2 = nn.LayerNorm(lstm_hidden * 2)
#         self.batchnorm1d = nn.BatchNorm1d(int(1108))
#         "****************cnn******************"
#
#         "****************cnn+cnn******************"
#         self.cov_1 = nn.Conv1d(in_channels=input_shape_esm,out_channels=int(input_shape_esm/2), kernel_size=3, padding='same')
#         self.cov_2 = nn.Conv1d(in_channels=int(input_shape_esm/2), out_channels=int(input_shape_esm / 4), kernel_size=1, padding='same')
#         self.relu = nn.ReLU()
#         self.leakyrelu = nn.LeakyReLU()
#         self.pooling = nn.MaxPool1d(kernel_size=1, stride=1)
#         "****************cnn+cnn******************"
#
#         self.conv1d_esm = nn.Conv1d(in_channels=input_shape_esm, out_channels=320, kernel_size=3,padding=1)
#         "****************三个特征提取方式******************"
#         self.lstm1 = nn.LSTM(input_size=input_shape_1d,  # 1141
#                              hidden_size=int(72),
#                              # hidden_size=int(64),
#                              num_layers=1,
#                              batch_first=True,
#                              bidirectional=True)
#
#         self.lstm2 = nn.LSTM(input_size=input_shape_2d,
#                              hidden_size=128,
#                              num_layers=1,
#                              batch_first=True,
#                              bidirectional=True)
#
#         self.lstm_esm = nn.LSTM(input_size=input_shape_esm,  # 1141
#                              hidden_size=lstm_hidden,  # 64,128,256,512,1024
#                              num_layers=1,
#                              batch_first=True,
#                              bidirectional=True)
#
#         "****************transformer******************"
#         self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_shape_esm, nhead=8)
#         self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=3)
#         "****************transformer******************"
#
#         "****************rnn******************"
#         self.rnn = nn.RNN(input_size=input_shape_1d,
#                           hidden_size=d_model,
#                           num_layers=1,
#                           batch_first=True)
#         "****************rnn******************"
#         "****************rnn(esm)******************"
#         self.rnn_esm = nn.RNN(input_size=input_shape_esm,
#                           hidden_size=int(input_shape_esm/2),
#                           num_layers=1,
#                           batch_first=True)
#         "****************rnn(esm)******************"
#
#         "****************gru******************"
#         self.gru_esm = nn.GRU(input_shape_esm, int(input_shape_esm / 8), batch_first=True, bidirectional=True)
#         "****************gru******************"
#         "****************gru******************"
#         self.gru_1d = nn.GRU(input_shape_1d, int(d_model / 2), batch_first=True, bidirectional=True)
#         "****************gru******************"
#         "*************Attention*********************"
#         self.self_attetion_1 = MultiHeadAttention(num_hiddens=int(input_shape_2d*2), num_heads=8)
#         self.self_attetion_2 = MultiHeadAttention(num_hiddens=int(lstm_hidden*2), num_heads=8)
#         self.self_attetion_all = MultiHeadAttention(num_hiddens=int(input_shape_2d * 2+input_shape_1d/8+lstm_hidden*2), num_heads=4)
#
#         self.layer = nn.Linear(int(input_shape_esm / 4), 256)
#         self.layer_1 = nn.Linear(int(input_shape_2d *2), int(input_shape_2d*2))
#         self.layer_2 = nn.Linear(int(input_shape_2d *2), int(input_shape_2d*2))
#         self.layer_3 = nn.Linear(int(lstm_hidden*2), int(lstm_hidden*2))
#         self.layer_4 = nn.Linear(int(lstm_hidden*2), int(lstm_hidden*2))
#         self.layer_5 = nn.Linear(int(input_shape_esm / 4), int(input_shape_esm / 4))
#         self.layer_6 = nn.Linear(int(input_shape_esm / 4), int(input_shape_esm / 4))
#         self.layer_all_1 = nn.Linear(int(input_shape_2d * 2+input_shape_1d/8+lstm_hidden*2), int(input_shape_2d * 2+input_shape_1d/8+lstm_hidden*2))
#         self.layer_all_2 = nn.Linear(int(input_shape_2d * 2+input_shape_1d/8+lstm_hidden*2), int(input_shape_2d * 2+input_shape_1d/8+lstm_hidden*2))
#         "*************Attention*********************"
#
#         self.block3 = nn.Sequential(
#
#             nn.Linear(input_shape_esm, 512),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(512, 64),
#             nn.ReLU(),
#             nn.Dropout(0.1))
#         "**********************************"
#         self.block1 = nn.Sequential(
#             # 64,128,256,512,1024
#             nn.Linear(256, 1)
#             )
#
#         "*************bridge_tower*********************"
#         self.bridge_tower = nn.ModuleList(BridgeTowerBlock(num_hiddens=256,
#                                                            num_heads=8,
#                                                            dropout_rate=dropout)  # 本来是0.1
#                                           for _ in range(1))
#
#         "*************bridge_tower*********************"
#         self.block = nn.Sequential(nn.Linear(896, 512),  # 2 * hidden_size for BiLSTM
#                                    nn.BatchNorm1d(512),
#                                    nn.LeakyReLU(),
#                                    nn.Dropout(0.1),
#                                    nn.Linear(512, 256),
#                                    nn.Linear(256, 1),
#                                    )
#
#         self.classifier = nn.Sequential(
#
#             nn.Linear(64, 1),
#             nn.Dropout(0.1)
#         )
#
#         self.fcn = nn.Sequential(nn.Linear(int(input_shape_esm / 4), int(input_shape_esm / 4)),
#                   nn.ReLU(), nn.Linear(int(input_shape_esm / 4), 1))
#
#         self.fc_trans = nn.Sequential(
#
#             nn.Linear(input_shape_esm, 1024),
#             nn.LeakyReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(1024, 64),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(64, 1))
#
#         self.fcn_bilstm = nn.Sequential(nn.Linear(lstm_hidden*2, int(lstm_hidden)),
#                                  nn.ReLU(), nn.Linear(int(lstm_hidden), 1))
#         self.fcn_gru = nn.Sequential(nn.Linear(int(input_shape_esm/4), int(input_shape_esm/8)),
#                                         nn.ReLU(), nn.Linear(int(input_shape_esm/8), 1))
#         self.fcn_rnn = nn.Sequential(nn.Linear(int(input_shape_esm/2), int(input_shape_esm / 2)),
#                   nn.ReLU(), nn.Linear(int(input_shape_esm / 2), 1))
#         self.fcn_multi_cnn = nn.Sequential(nn.Linear(int(input_shape_esm / 2), int(input_shape_esm / 2)),
#                                      nn.ReLU(), nn.Linear(int(input_shape_esm / 2), 1))
#
#         self.fcn_multi_cnn_2d = nn.Sequential(nn.Linear(int(input_shape_2d * d2_times), int(input_shape_2d *d2_times)),
#                                        nn.ReLU(), nn.Linear(int(input_shape_2d * d2_times), 1))
#         self.tcn = TCN(nb_filters=128, kernel_size=5, dropout_rate=0.3, nb_stacks=1, dilations=[1, 2, 4, 8],
#                 return_sequences=True, activation='relu', padding='same', use_skip_connections=True)
#
#         self.fcn_block = nn.Sequential(nn.Linear(int(lstm_hidden * 2 + input_shape_2d * d2_times + input_shape_1d // 8),
#                                                  int(lstm_hidden * 2 + input_shape_2d * d2_times + input_shape_1d // 8)),
#                                        nn.ReLU(), nn.Linear(
#                 int((lstm_hidden * 2 + input_shape_2d * d2_times + input_shape_1d // 8)), 1))
#
#         self.fcn_block2 = nn.Sequential(nn.Linear(int(lstm_hidden * 4),
#                                              int(lstm_hidden *4)),
#                                    nn.ReLU(),nn.Linear(
#             int((lstm_hidden * 4)), 1))
#
#     def forward(self, x_1d, x_2d, x_esm):
#         '--------------------------------1d------------------------------------'
#         '*****1d+bilstm*****'
#         lstm_out, _ = self.lstm1(x_1d.float())
#         x_1d = self.Layer_norm_1d(lstm_out)
#
#         '*****1d+bilstm*****'
#
#
#         '*****2d+multi-cnn*****'
#         x_2d = x_2d.permute(0, 2, 1)
#         x_2d = x_2d.float()
#         out_2 = [conv(x_2d) for conv in self.convs_1]
#         out_2 = torch.cat(out_2, dim=2)
#         out_2 = self.maxpool_1(out_2)
#         x_2d = out_2.view(-1, out_2.size(1))#256维度
#         x_2d = self.batchnorm1d(x_2d)
#
#         '*****2d+multi-cnn*****'
#
#         '*****2d+transformer*****'
#
#         '*****esm+bilstm*****'
#         lstm_out, _ = self.lstm_esm(x_esm.float())
#         x_esm = self.Layer_norm2(lstm_out)
#         x_all = x_2d
#         x_all = self.contraLinear_all(x_all)
#
#         '--------------------------------esm------------------------------------'
#
#         '************************+1d+2d+bridge_tower**************'
#         for layer in self.bridge_tower:
#             x_all, x_esm = layer(x_all, x_esm)
#         x = torch.cat((x_all, x_esm), dim=-1)
#         return x
#
#     def trainModel(self, x_1d, x_2d=None, x_esm=None):
#         x= self.forward(x_1d, x_2d, x_esm)
#         x = self.fcn_block2(x)
#         return x
#
#     def get_output(self, x_1d, x_2d, x_esm):
#         x = self.forward(x_1d, x_2d, x_esm)
#         x= self.fcn_block2[0](x)
#         return x

class SequenceMultiCNNLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
#原来d_model=64
    def __init__(self, d_input, d_model,d_another_h=256,
                 dropout=0.405, lstm_dropout=0.9,
                 nlayers=1, bidirectional=True,  k_cnn=[2, 3, 4, 5, 6], d_output=1):
        super(SequenceMultiCNNLSTM, self).__init__()
        self.d_model = d_model
        self.num_heads = 8
        self.dim_k = 8
        self.dim_v = 8
        self.dim_in = 64
        self.lstm = nn.LSTM(input_size=d_input,
                            hidden_size=d_model // 2 if bidirectional else d_model,
                            num_layers=nlayers, dropout=lstm_dropout,
                            bidirectional=bidirectional)
        '*********************增加**********************'
        self.rnn = nn.RNN(input_size=d_model,
                          hidden_size=d_model,
                          num_layers=1,
                          batch_first=True)

        self.gru = nn.GRU(d_model, int(d_model/2), batch_first=True, bidirectional=True)
        self.Wq = nn.Linear(d_model, d_model)  # 不改变形状的线性变换
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.att_weights = nn.Parameter(torch.Tensor(256, 1))
        self.softmax = nn.Softmax(dim=1)

        self.linear_q = nn.Linear(self.dim_in, self.dim_k, bias=False)
        self.linear_k = nn.Linear(self.dim_in, self.dim_k, bias=False)
        self.linear_v = nn.Linear(self.dim_in, self.dim_v, bias=False)
        self._norm_fact = 1 / sqrt(self.dim_k // self.num_heads)

        '*********************增加**********************'

        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=d_model,
                                    out_channels=d_another_h,
                                    kernel_size=h),
                          #                              nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=100 - h + 1))
            for h in k_cnn
        ])
        self.drop = nn.Dropout(dropout)
        #二维的
        self.fc = nn.Linear(d_another_h * len(k_cnn), d_output)
        #一维的
        self.fc_1 = nn.Linear(int(64), d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x, _ = self.lstm(x.float())
        x = self.drop(x)
        x = self.fc_1(x)
        x = self.sigmoid(x)
        return x
#lstm+多种CNN


class PositionalEmbedding(nn.Module):
    '''
    Modified from Annotated Transformer
    http://nlp.seas.harvard.edu/2018/04/03/attention.html
    '''
    def __init__(self, d_model, max_len=1024):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros((max_len, d_model), requires_grad=False).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class InputPositionEmbedding(nn.Module):
    def __init__(self, vocab_size=None, embed_dim=None, dropout=0.1,
                init_weight=None, seq_len=None):
        super(InputPositionEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.position_embed = PositionalEmbedding(embed_dim, max_len=seq_len)
        self.reproject = nn.Identity()
        if init_weight is not None:
            self.embed = nn.Embedding.from_pretrained(init_weight)
            self.reproject = nn.Linear(init_weight.size(1), embed_dim)

    def forward(self, inputs):
        # print(inputs.size())
        x = self.embed(inputs)
        # print(x.size())
        x = x + self.position_embed(inputs)
        # print(x)
        x = self.reproject(x)
        x = self.dropout(x)
        return x
class TranformerModel(nn.Module):
    def __init__(self, vocab_size=24, hidden_dim=25, d_embed=554, max_length=100):
        super(TranformerModel, self).__init__()

        # self.embedding = (vocab_size, d_embed, padding_idx=0)
        # self.embed = InputPositionEmbedding(vocab_size=vocab_size,
        #                                     seq_len=max_length, embed_dim=d_embed)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_embed, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.gru = nn.GRU(d_embed, hidden_dim, num_layers=2,
                          bidirectional=True, dropout=0.2)

        self.block1 = nn.Sequential(nn.Linear(d_embed * max_length, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(1024, 256),
                                    )

        self.block2 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.embed(x)
        output = self.transformer_encoder(x.float())
        # print(output.size())
        # output,hn=self.gru(output)
        # print(output.size())
        # print(hn.size())
        # hn=hn.permute(1,0,2)

        output = output.reshape(output.shape[0], -1)
        # hn=hn.reshape(output.shape[0],-1)
        # print(output.size())
        # print(hn.size())
        # output=torch.cat([output,hn],1)
        # print(output.size())
        output = self.block1(output)
        output = self.block2(output)
        output = self.sigmoid(output)
        # print(output.size())
        return output
class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att


import torch
import torch.nn as nn



class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = 256
        self.Wq = nn.Linear(hidden_dim, hidden_dim)  # 不改变形状的线性变换
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)

        # self.att_weights = nn.Parameter(torch.Tensor(256, 1))
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        """
        lstm_output: (batch_size, seq_len, hidden_dim * 2 if bidirectional else hidden_dim)
        """
        q = self.Wq(lstm_output)
        k = self.Wk(lstm_output)
        v = self.Wv(lstm_output)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(float(self.hidden_dim)))
        # 计算向量权重，softmax归一化
        attention_weight = F.softmax(attention_scores, dim=-1)
        # 计算输出
        output = torch.matmul(attention_weight, v)
        return output




