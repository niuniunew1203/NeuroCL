from catboost import CatBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import StackingClassifier
from utils import read_and_extract
from features import combinefeature
from utils import prediction_results
from utils import prediction_results_
import pandas as pd
from sklearn.datasets import *
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score,GridSearchCV
import os
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
import joblib
import torchvision.transforms as transforms
import torch
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from tqdm import tqdm
import warnings
import torch.nn as nn
import torch.nn.init as init
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from model.model import SequenceMultiCNNLSTM
from model.model import TranformerModel,MultiHeadSelfAttention
from torch.utils.data import random_split
import math
import csv
import torch.nn.functional as F
from transformers import get_scheduler

warnings.filterwarnings("ignore", category=FutureWarning)
# 定义数据增强
transform = transforms.Compose([
    transforms.RandomRotation(10),  # 随机旋转 +/- 10 度
    transforms.ToTensor()
])
'train_features-所有训练特征'
'test_features-所有测试特征'
'train_labels_tensor-所有训练标签'
'test_labels_tensor-所有测试标签'
'all_TRAIN-训练特征, all_VAL-验证特征, Y_TRAIN-训练标签, Y_VAL-验证标签'
seed = 991
torch.manual_seed(seed)
dropouts = [0.1]
batch_sizes = [256]
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
# for batch_size in batch_sizes:
# 定义文件列表
train_list = ['neuro/pos_train_fasta.txt', 'neuro/neg_train_fasta.txt']
test_list = ['neuro/pos_test_fasta.txt','neuro/neg_test_fasta.txt']
'所有的序列为train_sequence,test_sequence'
train_sequences = []#3880
test_sequences=[]#970

'读取所有的序列'
# 遍历文件列表，读取并提取序列和标签
for filename in train_list:
    sequences = read_and_extract(filename)
    train_sequences.extend(sequences)

for filename in test_list:
    sequences = read_and_extract(filename)
    test_sequences.extend(sequences)


train_sequences_dict = {}
index = 0
for sequence in train_sequences:
    train_sequences_dict[index] = sequence
    index += 1

test_sequences_dict = {}
index = 0
for sequence in test_sequences:
    test_sequences_dict[index] = sequence
    index += 1

'生成标签的过程'
# 生成形状为 (3880,) 的数组，初始值为 0
train_label = np.zeros((3880,))
# 设置前 1940 个元素的值为 1
train_label[:1940] = 1
test_label = np.zeros((970,))
# 设置前 1940 个元素的值为 1
test_label[:485] = 1
train_labels_tensor = torch.tensor(train_label)
test_labels_tensor = torch.tensor(test_label)
feature_list_ml = [ 'OE', 'AAC','DPC', 'PSAAC','asdc','AAE','BLO2']
train_features_1d,test_features_1d,train_features_2d, test_features_2d = combinefeature(feature_list_ml, train_sequences, test_sequences)
features_name = '+'.join(feature_list_ml)
print(features_name)
print('1d形状',train_features_1d.shape)
print('2d形状',train_features_2d.shape)

train_features_esm = np.load('neuro/esm2_650M_train.npy')
test_features_esm = np.load('neuro/esm2_650M_test.npy')
print('esm形状',train_features_esm.shape)


'转化成tensor'


'标准化特征矩阵'
scaler = StandardScaler()
train_features_esm = scaler.fit_transform(train_features_esm)
test_features_esm = (scaler.transform
                 (test_features_esm))
scaler = StandardScaler()
train_features_1d = scaler.fit_transform(train_features_1d)
test_features_1d = (scaler.transform
                 (test_features_1d))
'转化成tensor'
train_features_esm_tensor = torch.tensor(train_features_esm, dtype=torch.float32)
test_features_esm_tensor = torch.tensor(test_features_esm, dtype=torch.float32)
train_features_1d_tensor = torch.tensor(train_features_1d, dtype=torch.float32)
test_features_1d_tensor = torch.tensor(test_features_1d, dtype=torch.float32)

n_samples, height, width = train_features_2d.shape

train_features_2d_ = train_features_2d.reshape(train_features_2d.shape[0], -1)
test_features_2d_ = test_features_2d.reshape(test_features_2d.shape[0], -1)
# Apply StandardScaler
scaler = StandardScaler()
train_features_2d_ = scaler.fit_transform(train_features_2d_)
test_features_2d_ = scaler.transform(test_features_2d_)
train_features_2d_tensor = train_features_2d_.reshape(train_features_2d.shape[0], train_features_2d.shape[1],train_features_2d.shape[2])
test_features_2d_tensor = test_features_2d_.reshape(test_features_2d.shape[0], test_features_2d.shape[1],test_features_2d.shape[2])
train_features_2d_tensor = torch.tensor(train_features_2d_tensor, dtype=torch.float32)
test_features_2d_tensor = torch.tensor(test_features_2d_tensor, dtype=torch.float32)

test_dataset = TensorDataset(test_features_1d_tensor,test_features_2d_tensor,test_features_esm_tensor, test_labels_tensor)
train_dataset = TensorDataset(train_features_1d_tensor,train_features_2d_tensor,train_features_esm_tensor, train_labels_tensor)

train_size = int(0.9 * len(train_dataset))  # 训练集占90%
val_size = len(train_dataset) - train_size  # 验证集占10%
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
batch_size =40

shuffle = True
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
class ContrastiveLoss_2(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss_2, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        # print('label.shape', label.shape)
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def ContrastiveLoss(x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        a1 = torch.einsum('ik,jk->ij', x1, x2)
        a2 = torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix_ = a1 / a2
        sim_matrix = torch.exp(sim_matrix_ / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        all_sim = sim_matrix.sum(dim=1)
        neg_sim = all_sim - pos_sim
        loss = pos_sim / neg_sim
        loss = - torch.log(loss).mean()
        return loss
def collate(batch):
    device = torch.device("cpu")
    x_1d_1s = []
    x_1d_2s = []
    x_2d_1s = []
    x_2d_2s = []
    x_esm_1s = []
    x_esm_2s = []
    label_1s = []
    label_2s = []
    labels = []
    batch_size = len(batch)
    # print(batch[1][1])
    # print(batch[:][1])
    # my_batch =np.array(batch[:][1])
    # print(my_batch.shape)

    for i in range(int(batch_size / 2)):
        x_1d_1, x_2d_1, x_esm_1,label_1 = batch[i][0], batch[i][1], batch[i][2],batch[i][3]
        x_1d_2, x_2d_2, x_esm_2,label_2 = batch[i + int(batch_size / 2)][0], batch[i + int(batch_size / 2)][1][:], batch[i + int(batch_size / 2)][2][:],batch[i + int(batch_size / 2)][3]
        x_1d_1s.append(x_1d_1.unsqueeze(0))
        x_1d_2s.append(x_1d_2.unsqueeze(0))
        x_2d_1s.append(x_2d_1.unsqueeze(0))
        x_2d_2s.append(x_2d_2.unsqueeze(0))
        label = (label_1.long() ^ label_2.long())
        x_esm_1s.append(x_esm_1.unsqueeze(0))
        x_esm_2s.append(x_esm_2.unsqueeze(0))
        label_1s.append(label_1.unsqueeze(0))
        label_2s.append(label_2.unsqueeze(0))
        labels.append(label.unsqueeze(0))
    x_1d_1s = torch.cat(x_1d_1s).to(device)
    x_1d_2s = torch.cat(x_1d_2s).to(device)
    x_esm_1s = torch.cat(x_esm_1s).to(device)
    x_esm_2s = torch.cat(x_esm_2s).to(device)
    x_2d_1s = torch.cat(x_2d_1s).to(device)
    x_2d_2s = torch.cat(x_2d_2s).to(device)
    label_1s = torch.cat(label_1s).to(device)
    label_2s = torch.cat(label_2s).to(device)
    labels = torch.cat(labels).to(device)

    return x_1d_1s,x_1d_2s, x_2d_1s, x_2d_2s,  x_esm_1s, x_esm_2s,label_1s,label_2s,labels


'------------------用dataloader加载并全部获取整个数据集的数据------------------'
from model.model import model_A
# 将数据集划分为 10 个折
epochs = 53
#三个模型的参数
lstm_hidden =128
d_another_h=256
d_model=820
#输出的参数
dim_in =128#64+128+64
output_dim = 1
save_path = 'D:/working/666.csv'
#三个输入的维度
input_shape_1d=train_features_1d_tensor.shape[1]


input_shape_2d=width
input_shape_esm=train_features_esm_tensor.shape[1]
test_accuracy = []
test_precision = []
test_recall = []
test_f1 = []
test_mcc = []
test_auroc = []
test_sn = []
test_sp = []












my_model = model_A(input_shape_1d, input_shape_2d, input_shape_esm, lstm_hidden, d_another_h, d_model, dim_in,output_dim, dropout=0.1)
criterion = nn.BCEWithLogitsLoss()
contrastive_loss_fn = ContrastiveLoss_2()
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.0008)
num_warmup_steps=1
num_training_steps = 950
scheduler = get_scheduler(
    name = 'cosine',
    optimizer= optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
#按批次加载
train_iter_cont = DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, collate_fn=collate)
for epoch in range(epochs):
    my_model.train()
    total_loss = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_mcc = 0
    total_auroc = 0
    total_sn = 0
    total_sp = 0
    # 分割特征
    for train_features_1d_1, train_features_1d_2, train_features_2d_1, train_features_2d_2, train_features_esm_1, train_features_esm_2, train_labels_1, train_labels_2, binary_labels in train_iter_cont:
        embedding_1 = my_model(train_features_1d_1, train_features_2d_1, train_features_esm_1)  # 32*384
        embedding_2 = my_model(train_features_1d_2, train_features_2d_2, train_features_esm_2)#32*384
        outputs1 = my_model.trainModel(train_features_1d_1, train_features_2d_1, train_features_esm_1)
        outputs2 = my_model.trainModel(train_features_1d_2, train_features_2d_2, train_features_esm_2)
        contrastive_loss = contrastive_loss_fn(embedding_1, embedding_2, binary_labels)
        loss1 = criterion(outputs1.float().view(-1), train_labels_1.float())
        loss2 = criterion(outputs2.float().view(-1), train_labels_2.float())
        losses = contrastive_loss + loss1 + loss2
        #losses = loss1 + loss2
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()
        outputs = torch.cat((outputs1, outputs2), dim=0)
        predictions = outputs > 0.5
        train_labels = torch.cat((train_labels_1, train_labels_2), dim=0)
        total_loss += losses.item()
        accuracy = accuracy_score(train_labels.cpu().detach().numpy(), predictions.cpu().detach().numpy())
        total_accuracy += accuracy
        precision = precision_score(train_labels.cpu().detach().numpy(), predictions.cpu().detach().numpy(),
                                    zero_division=1)
        total_precision += precision
        recall = recall_score(train_labels.cpu().detach().numpy(), predictions.cpu().detach().numpy())
        total_recall += recall
        f1 = f1_score(train_labels.cpu().detach().numpy(), predictions.cpu().detach().numpy())
        total_f1 += f1
        mcc = matthews_corrcoef(train_labels, predictions.cpu().detach().numpy())
        total_mcc += mcc
        probabilities = outputs.cpu().detach().numpy()
        auroc = roc_auc_score(train_labels, probabilities)
        total_auroc += auroc
        tn, fp, fn, tp = confusion_matrix(train_labels, predictions.cpu().detach().numpy()).ravel()
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)
        total_sn += sn
        total_sp += sp

    average_loss = total_loss / len(train_loader)
    average_accuracy = total_accuracy / len(train_loader)
    average_precision = total_precision / len(train_loader)
    average_recall = total_recall / len(train_loader)
    average_f1 = total_f1 / len(train_loader)
    average_mcc = total_mcc / len(train_loader)
    average_auroc = total_auroc / len(train_loader)
    average_sn = total_sn / len(train_loader)
    average_sp = total_sp / len(train_loader)
    print(
        f"Epoch {epoch + 1}/{epochs} , Loss: {average_loss}, Accuracy: {average_accuracy}, Precision: {average_precision}, Sn:{sn:.4f}, Sp:{sp:.4f}, Mcc:{average_mcc:.4f}, Auroc:{average_auroc:.4f}, Recall: {average_recall:.4f}, F1 Score: {average_f1:.4f}")
    my_model.eval()
    val_features_1d,val_features_2d,val_features_esm, val_labels = next(iter(val_loader))
    with torch.no_grad():
        # val_features_2d=val_features_2d.reshape(-1, height, width)
        val_outputs = my_model.trainModel(val_features_1d, val_features_2d, val_features_esm)
        predictions = val_outputs > 0.5
        auc_score = roc_auc_score(val_labels, val_outputs)
        accuracy = accuracy_score(val_labels.cpu().detach().numpy(), predictions.cpu().detach().numpy())
        prediction_results('val', predictions, val_labels, auc_score, lstm_hidden, save_path)

    # if epoch >25:
    #     torch.save(my_model.state_dict(), f'cp_ASDC_PSAAC_AAC_DPC_AAE_OE/model_{epoch+1}.pth')

my_model.eval()
test_features_1d,test_features_2d,test_features_esm, test_labels = next(iter(test_loader))
with torch.no_grad():
    # test_features_2d.reshape(-1, height, width)
    test_outputs = my_model.trainModel(test_features_1d, test_features_2d, test_features_esm)
    predictions = test_outputs > 0.5

    t_accuracy = accuracy_score(test_labels, predictions)
    t_precision = precision_score(test_labels, predictions, average='binary', zero_division=1)
    t_recall = recall_score(test_labels, predictions, average='binary')
    t_f1 = f1_score(test_labels, predictions, average='binary')
    t_mcc = matthews_corrcoef(test_labels, predictions)
    t_auc_score = roc_auc_score(test_labels, test_outputs)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    t_sn = tp / (tp + fn)  # 真正例率
    t_sp = tn / (tn + fp)  # 真负例率
    test_accuracy.append(t_accuracy)
    test_precision.append(t_precision)
    test_recall.append(t_recall)
    test_f1.append(t_f1)
    test_mcc.append(t_mcc)
    test_auroc.append(t_auc_score)
    test_sn.append(t_sn)
    test_sp.append(t_sp)
    csv_columns = [
        'methods', 'Accuracy', 'Precision', 'Sn', 'Sp',
        'Mcc', 'Auroc', 'Recall', 'F1_score'
    ]
    methods = f'test{seed}'
    results = {
        "methods": methods,
        "Accuracy": t_accuracy,
        "Precision": t_precision,
        "Sn": t_sn,
        "Sp": t_sp,
        "Mcc": t_mcc,
        "Auroc": t_auc_score,
        "Recall": t_recall,
        "F1_score": t_f1,
    }
    if save_path:
        with open(save_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            csvfile.seek(0, os.SEEK_END)  # 移动到文件末尾
            if csvfile.tell() == 0:
                writer.writeheader()  # 只有文件为空时才写入标题行

            writer.writerow(results)  # 写入数据行
    # 将结果列表转换为字符串，并用逗号连接每个元素
    results_str = f"Accuracy: {results['Accuracy']},Precision:{results['Precision']},Sn: {results['Sn']},Sp: {results['Sp']},Mcc: {results['Mcc']},Auroc: {results['Auroc']},Recall: {results['Recall']},F1_score: {results['F1_score']}"
    # 打印结果
    print(f"{methods} - {results_str}")








