'''
Conversion of "main_svae.py" to run from the terminal
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import gc
import sys
import time
import json
import pickle
import random
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime as dt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


## Utlity functions

LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor

is_cuda_available = torch.cuda.is_available()

if is_cuda_available: 
    print("Using CUDA...\n")
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor
    
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def save_obj_json(obj, name):
    with open(name + '.json', 'w') as f:
        json.dump(obj, f)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_obj_json(name):
    with open(name + '.json', 'r') as f:
        return json.load(f)

def file_write(log_file, s):
    print(s)
    f = open(log_file, 'a')
    f.write(s+'\n')
    f.close()

def clear_log_file(log_file):
    f = open(log_file, 'w')
    f.write('')
    f.close()

def pretty_print(h):
    print("{")
    for key in h:
        print(' ' * 4 + str(key) + ': ' + h[key])
    print('}\n')

## Hyper Parameters

'''
NOTES:

- Try having two different layers for mu and sigma
- Never using dropout
- Not using L2 Norm at input

'''

hyper_params = {
#     'data_base': '../saved_data/netflix/pro_sg/',
#     'project_name': 'svae_netflix',
    'data_base': '../saved_data/ml-1m/pro_sg/',
    'project_name': 'svae_ml1m',
#     'data_base': '../saved_data/netflix-full/pro_sg/',
#     'project_name': 'svae_netflix_full',
#     'data_base': '../saved_data/netflix-good-sample/pro_sg/',
#     'project_name': 'svae_netflix_good_sample',
    'model_file_name': '',
    'log_file': '',
    'history_split_test': [0.8, 0.2], # Part of test history to train on : Part of test history to test

    'learning_rate': 0.01, # learning rate is required only if optimizer is adagrad
    'optimizer': 'adam',
    'weight_decay': float(5e-3),

    'epochs': 25,
    'batch_size': 1,
    
    'item_embed_size': 256,
    'rnn_size': 200,
    'hidden_size': 150,
    'latent_size': 64,
    'loss_type': 'next_k', # [predict_next, same, prefix, postfix, exp_decay, next_k]
    'next_k': 4,

    'number_users_to_keep': 1000000000,
    'batch_log_interval': 40000,
    'train_cp_users': 200,
    'exploding_clip': 0.25,
}

if len(sys.argv) >= 2:
    print("Using new hyper-parameters..")
    json_file = sys.argv[1]
    if json_file[-5:] == '.json': json_file = json_file[:-5]

    update_hyper = load_obj_json(json_file)
    for key in update_hyper:
        print("Updating value for", key, "to", update_hyper[key])
        hyper_params[key] = update_hyper[key]

file_name = '_optimizer_' + str(hyper_params['optimizer'])
if hyper_params['optimizer'] == 'adagrad':
    file_name += '_lr_' + str(hyper_params['learning_rate'])
file_name += '_weight_decay_' + str(hyper_params['weight_decay'])
file_name += '_loss_type_' + str(hyper_params['loss_type'])
file_name += '_item_embed_size_' + str(hyper_params['item_embed_size'])
file_name += '_rnn_size_' + str(hyper_params['rnn_size'])
file_name += '_latent_size_' + str(hyper_params['latent_size'])

hyper_params['log_file'] = '../saved_logs/' + hyper_params['project_name'] + '_log' + file_name + '.txt'
hyper_params['model_file_name'] = '../saved_models/' + hyper_params['project_name'] + '_model' + file_name + '.pt'


## Data Parsing

def load_data(hyper_params):
    
    file_write(hyper_params['log_file'], "Started reading data file")
    
    f = open(hyper_params['data_base'] + 'train.csv')
    lines_train = f.readlines()[1:]
    
    f = open(hyper_params['data_base'] + 'test_tr.csv')
    lines_test_tr = f.readlines()[1:]
    
    f = open(hyper_params['data_base'] + 'test_te.csv')
    lines_test_te = f.readlines()[1:]
    
    unique_sid = list()
    with open(hyper_params['data_base'] + 'unique_sid.txt', 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    num_items = len(unique_sid)
    
    file_write(hyper_params['log_file'], "Data Files loaded!")

    train_reader = DataReader(hyper_params, lines_train, None, num_items, True)
    test_reader = DataReader(hyper_params, lines_test_tr, lines_test_te, num_items, False)

    return train_reader, test_reader, num_items

class DataReader:
    def __init__(self, hyper_params, a, b, num_items, is_training):
        self.hyper_params = hyper_params
        self.batch_size = hyper_params['batch_size']
        
        num_users = 0
        min_user = 1000000000000000000000000 # Infinity
        for line in a:
            line = line.strip().split(",")
            num_users = max(num_users, int(line[0]))
            min_user = min(min_user, int(line[0]))
        num_users = num_users - min_user + 1
        
        self.num_users = num_users
        self.min_user = min_user
        self.num_items = num_items
        
        self.data_train = a
        self.data_test = b
        self.is_training = is_training
        self.all_users = []
        
        self.prep()
        self.number()

    def prep(self):
        self.data = []
        for i in range(self.num_users): self.data.append([])
            
        for i in tqdm(range(len(self.data_train))):
            line = self.data_train[i]
            line = line.strip().split(",")
            self.data[int(line[0]) - self.min_user].append([ int(line[1]), 1 ])
        
        if self.is_training == False:
            self.data_te = []
            for i in range(self.num_users): self.data_te.append([])
                
            for i in tqdm(range(len(self.data_test))):
                line = self.data_test[i]
                line = line.strip().split(",")
                self.data_te[int(line[0]) - self.min_user].append([ int(line[1]), 1 ])
        
    def number(self):
        self.num_b = int(min(len(self.data), self.hyper_params['number_users_to_keep']) / self.batch_size)
    
    def iter(self):
        users_done = 0

        x_batch = []
        now_at = 0
        
        user_iterate_order = list(range(len(self.data)))
        # np.random.shuffle(user_iterate_order)
        
        for user in user_iterate_order:

            if users_done > self.hyper_params['number_users_to_keep']: break
            users_done += 1

            y_batch_s = torch.zeros(self.batch_size, len(self.data[user])-1, self.num_items).cuda()
            
            if self.hyper_params['loss_type'] == 'predict_next':
                for timestep in range(len(self.data[user]) - 1):
                    y_batch_s[len(x_batch), timestep, :].scatter_(
                        0, LongTensor([ i[0] for i in [ self.data[user][timestep + 1] ] ]), 1.0
                    )
                x_batch.append([ i[0] for i in self.data[user][:-1] ])
                
            elif self.hyper_params['loss_type'] == 'next_k':
                for timestep in range(len(self.data[user]) - 1):
                    y_batch_s[len(x_batch), timestep, :].scatter_(
                        0, LongTensor([ i[0] for i in self.data[user][timestep + 1:][:self.hyper_params['next_k']] ]), 1.0
                    )
                x_batch.append([ i[0] for i in self.data[user][:-1] ])

            elif self.hyper_params['loss_type'] == 'postfix':
                for timestep in range(len(self.data[user]) - 1):
                    y_batch_s[len(x_batch), timestep, :].scatter_(
                        0, LongTensor([ i[0] for i in self.data[user][timestep + 1:] ]), 1.0
                    )
                x_batch.append([ i[0] for i in self.data[user][:-1] ])
            
            now_at += 1
    
            if now_at == self.batch_size:

                yield Variable(LongTensor(x_batch)), Variable(y_batch_s, requires_grad=False)

                x_batch = []
                now_at = 0

    def iter_eval(self):

        x_batch = []
        y_batch_s = torch.zeros(self.batch_size, self.num_items).cuda()
        y_batch = []
        test_movies, test_movies_r = [], []
        now_at = 0
        users_done = 0
        
        for user in range(len(self.data)):
            
            users_done += 1
            if users_done > self.hyper_params['number_users_to_keep']: break
            
            if self.is_training == True: 
                split = float(self.hyper_params['history_split_test'][0])
                base_predictions_on = self.data[user][:int(split * len(self.data[user]))]
                heldout_movies = self.data[user][int(split * len(self.data[user])):]
            else:
                base_predictions_on = self.data[user]
                heldout_movies = self.data_te[user]
                
            y_batch_s = torch.zeros(self.batch_size, len(base_predictions_on)-1, self.num_items).cuda()
            if self.hyper_params['loss_type'] == 'predict_next':
                for timestep in range(len(base_predictions_on) - 1):
                    y_batch_s[len(x_batch), timestep, :].scatter_(
                        0, LongTensor([ i[0] for i in [ base_predictions_on[timestep + 1] ] ]), 1.0
                    )
                x_batch.append([ i[0] for i in base_predictions_on[:-1] ])
                
            elif self.hyper_params['loss_type'] == 'next_k':
                for timestep in range(len(base_predictions_on) - 1):
                    y_batch_s[len(x_batch), timestep, :].scatter_(
                        0, LongTensor([ i[0] for i in base_predictions_on[timestep + 1:][:self.hyper_params['next_k']] ]), 1.0
                    )
                x_batch.append([ i[0] for i in base_predictions_on[:-1] ])

            elif self.hyper_params['loss_type'] == 'postfix':
                for timestep in range(len(base_predictions_on) - 1):
                    y_batch_s[len(x_batch), timestep, :].scatter_(
                        0, LongTensor([ i[0] for i in base_predictions_on[timestep + 1:] ]), 1.0
                    )
                x_batch.append([ i[0] for i in base_predictions_on[:-1] ])
            
#             elif self.hyper_params['loss_type'] == 'same':
#                 x_batch.append([ i[0] for i in base_predictions_on[:] ])
#                 y_batch_s[now_at, :].scatter_(
#                     0, LongTensor([ i[0] for i in [ base_predictions_on[0] ] ]), 1.0
#                 )
#                 y_batch.append([ i[0] for i in base_predictions_on[:] ])
            
#             elif self.hyper_params['loss_type'] == 'prefix':
#                 x_batch.append([ i[0] for i in base_predictions_on[:] ])
#                 y_batch_s[now_at, :].scatter_(
#                     0, LongTensor([ i[0] for i in [ base_predictions_on[0] ] ]), 1.0
#                 )
#                 y_batch.append([ i[0] for i in base_predictions_on[:] ])
            
#             elif self.hyper_params['loss_type'] == 'postfix':
#                 x_batch.append([ i[0] for i in base_predictions_on[:] ])
#                 y_batch_s[now_at, :].scatter_(
#                     0, LongTensor([ i[0] for i in base_predictions_on[:] ]), 1.0
#                 )
#                 y_batch.append([ i[0] for i in base_predictions_on[:] ])
                
#             elif self.hyper_params['loss_type'] == 'exp_decay':
#                 x_batch.append([ i[0] for i in base_predictions_on[:] ])
#                 y_batch_s[now_at, :].scatter_(
#                     # 0, LongTensor([ i[0] for i in base_predictions_on[:] ]), FloatTensor([ np.e ** (-1.0 * i) for i in range(len(base_predictions_on[:])) ]), 
#                     0, LongTensor([ i[0] for i in base_predictions_on[:] ]), FloatTensor([ 1.0 / (i + 1.0) for i in range(len(base_predictions_on[:])) ]), 
#                     # 0, LongTensor([ i[0] for i in base_predictions_on[:] ]), FloatTensor([ 1.0 / np.log2(i + 2.0) for i in range(len(base_predictions_on[:])) ]), 
#                 )
#                 y_batch.append([ i[0] for i in base_predictions_on[:-1] ])
            
            now_at += 1
            
            test_movies.append([ i[0] for i in heldout_movies ])
            test_movies_r.append([ i[1] for i in heldout_movies ])
            
            if now_at == self.batch_size:
                
                yield Variable(LongTensor(x_batch)), Variable(y_batch_s, requires_grad=False),                 test_movies, test_movies_r
                
                x_batch = []
                y_batch_s = torch.zeros(self.batch_size, self.num_items).cuda()
                y_batch = []
                test_movies, test_movies_r = [], []
                now_at = 0


## Evaluation Code

# In[5]:

def evaluate(model, criterion, reader, hyper_params, is_train_set):
    model.eval()

    metrics = {}
    metrics['loss'] = 0.0
    Ks = [10, 100]
    for k in Ks: 
        metrics['NDCG@' + str(k)] = 0.0
        metrics['HR@' + str(k)] = 0.0
        metrics['Prec@' + str(k)] = 0.0

    batch = 0
    total_ndcg = 0.0
    
    len_to_ndcg_at_100_map = {}

    for x, y_s, test_movies, test_movies_r in reader.iter_eval():
        batch += 1
        if is_train_set == True and batch > hyper_params['train_cp_users']: break

        decoder_output, z_mean, z_log_sigma = model(x)
        
        metrics['loss'] += criterion(decoder_output, z_mean, z_log_sigma, y_s, 0.2).data[0]
        
        # decoder_output[X.nonzero()] = -np.inf
        decoder_output = decoder_output.data
        
        x_scattered = torch.zeros(decoder_output.shape[0], decoder_output.shape[2]).cuda()
        x_scattered[0, :].scatter_(0, x[0].data, 1.0)
        
        # If loss type is predict next, the last element in the train sequence is not included in x
        #### Should ideally be done, but it's alright :)
        # if hyper_params['loss_type'] == 'predict_next': x_scattered[0, y[0][-1]] = 1.0
        
        last_predictions = decoder_output[:, -1, :] -         (torch.abs(decoder_output[:, -1, :] * x_scattered) * 100000000)
        
        for batch_num in range(last_predictions.shape[0]):
            predicted_scores = last_predictions[batch_num]
            actual_movies_watched = test_movies[batch_num]
            actual_movies_ratings = test_movies_r[batch_num]
                    
            # Calculate NDCG
            _, argsorted = torch.sort(-1.0 * predicted_scores)
            for k in Ks:
                best = 0.0
                now_at = 0.0
                dcg = 0.0
                hr = 0.0
                
                rec_list = list(argsorted[:k].cpu().numpy())
                for m in range(len(actual_movies_watched)):
                    movie = actual_movies_watched[m]
                    now_at += 1.0
                    if now_at <= k: best += 1.0 / float(np.log2(now_at + 1))
                    
                    if movie not in rec_list: continue
                    hr += 1.0
                    dcg += 1.0 / float(np.log2(float(rec_list.index(movie) + 2)))
                
                metrics['NDCG@' + str(k)] += float(dcg) / float(best)
                metrics['HR@' + str(k)] += float(hr) / float(len(actual_movies_watched))
                metrics['Prec@' + str(k)] += float(hr) / float(k)
                
                if k == 100:
                    seq_len = int(len(actual_movies_watched)) + int(x[batch_num].shape[0]) + 1
                    if seq_len not in len_to_ndcg_at_100_map: len_to_ndcg_at_100_map[seq_len] = []
                    len_to_ndcg_at_100_map[seq_len].append(float(dcg) / float(best))
                
            total_ndcg += 1.0
    
    metrics['loss'] = float(metrics['loss']) / float(batch)
    metrics['loss'] = round(metrics['loss'], 4)
    
    for k in Ks:
        metrics['NDCG@' + str(k)] = round((100.0 * metrics['NDCG@' + str(k)]) / float(total_ndcg), 4)
        metrics['HR@' + str(k)] = round((100.0 * metrics['HR@' + str(k)]) / float(total_ndcg), 4)
        metrics['Prec@' + str(k)] = round((100.0 * metrics['Prec@' + str(k)]) / float(total_ndcg), 4)
        
    return metrics, len_to_ndcg_at_100_map


## Model

# In[6]:

class Encoder(nn.Module):
    def __init__(self, hyper_params):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(
            hyper_params['rnn_size'], hyper_params['hidden_size']
        )
        nn.init.xavier_normal(self.linear1.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hyper_params):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(hyper_params['latent_size'], hyper_params['hidden_size'])
        self.linear2 = nn.Linear(hyper_params['hidden_size'], hyper_params['total_items'])
        nn.init.xavier_normal(self.linear1.weight)
        nn.init.xavier_normal(self.linear2.weight)
        self.activation = nn.Tanh()
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        # x = self.dropout(x)
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Model(nn.Module):
    def __init__(self, hyper_params):
        super(Model, self).__init__()
        self.hyper_params = hyper_params
        
        self.encoder = Encoder(hyper_params)
        self.decoder = Decoder(hyper_params)
        
        # No +1 means can never pad, hence bsz has to be equal 1
        self.item_embed = nn.Embedding(hyper_params['total_items'], hyper_params['item_embed_size'])
        
        self.gru = nn.GRU(
            hyper_params['item_embed_size'], hyper_params['rnn_size'], 
            batch_first=True, num_layers=1
        )
        
        self.layer_temp = nn.Linear(hyper_params['hidden_size'], 2 * hyper_params['latent_size'])
        nn.init.xavier_normal(self.layer_temp.weight)
        
        self.tanh = nn.Tanh()
        
    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        temp_out = self.layer_temp(h_enc)
        
        mu = temp_out[:, :self.hyper_params['latent_size']]
        log_sigma = temp_out[:, self.hyper_params['latent_size']:]
        
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        if is_cuda_available: std_z = std_z.cuda()

        self.z_mean = mu
        self.z_log_sigma = log_sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, x):
        in_shape = x.shape
        x = x.view(-1)
        
        x = self.item_embed(x)
        x = x.view(in_shape[0], in_shape[1], -1)
        
        rnn_out, _ = self.gru(x)
        rnn_out = rnn_out.view(in_shape[0] * in_shape[1], -1)
        
        enc_out = self.encoder(rnn_out)
        sampled_z = self.sample_latent(enc_out)
        
#         if self.hyper_params['conditional'] == True:
#             mult_matrix = torch.zeros(in_shape[0], in_shape[1], in_shape[1]).cuda()
#             for b in range(in_shape[0]):
#                 for i in range(in_shape[1]):
#                     num_one = i + 1 - max(i-self.hyper_params['attention_context']+1, 0)
#                     to_put = 1.0 / float(num_one)
#                     for j in range(max(i-self.hyper_params['attention_context']+1, 0), i+1): mult_matrix[b, i, j] = to_put
#             mult_matrix = Variable(mult_matrix.view(in_shape[0]*in_shape[1], in_shape[1]))
            
#             print(mult_matrix.shape)
#             print(sampled_z.shape)
#             sampled_z = torch.mm(mult_matrix, sampled_z)
        
        dec_out = self.decoder(sampled_z)
        dec_out = dec_out.view(in_shape[0], in_shape[1], -1)
                              
        return dec_out, self.z_mean, self.z_log_sigma


## Custom loss

# In[7]:

class VAELoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super(VAELoss,self).__init__()
        self.hyper_params = hyper_params

    def forward(self, decoder_output, mu_q, logvar_q, y_true_s, anneal):
        # Calculate KL Divergence loss
        kld = torch.mean(torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q**2 - 1), -1))
    
        # decoder_output shape : [batch_size, seq_len, all_items]
        dec_shape = decoder_output.shape

        # Do you want to try negative sampling?
        decoder_output = F.log_softmax(decoder_output, -1)
        num_ones = float(torch.sum(y_true_s[0, 0]))
        
        likelihood = torch.sum(
            -1.0 * y_true_s.view(dec_shape[0] * dec_shape[1], -1) * \
            decoder_output.view(dec_shape[0] * dec_shape[1], -1)
        ) / (float(self.hyper_params['batch_size']) * num_ones)
        
        final = (anneal * kld) + (1.0 * likelihood)
        
        return final


## Training loop

# In[8]:

def train(reader):
    model.train()
    total_loss = 0
    start_time = time.time()
    batch = 0
    batch_limit = int(train_reader.num_b)
    total_anneal_steps = 200000
    anneal = 0.0
    update_count = 0.0
    anneal_cap = 0.2

    for x, y_s in reader.iter():
        # print(x[0])
        batch += 1
        
        model.zero_grad()
        optimizer.zero_grad()

        decoder_output, z_mean, z_log_sigma = model(x)
        
        loss = criterion(decoder_output, z_mean, z_log_sigma, y_s, anneal)
        loss.backward()
        
        # nn.utils.clip_grad_norm(model.parameters(), hyper_params['exploding_clip'])
        optimizer.step()

        total_loss += loss.data
        
        if total_anneal_steps > 0:
            anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
        else:
            anneal = anneal_cap
        update_count += 1.0

        if (batch % hyper_params['batch_log_interval'] == 0 and batch > 0) or batch == batch_limit:
            div = hyper_params['batch_log_interval']
            if batch == batch_limit: div = (batch_limit % hyper_params['batch_log_interval']) - 1
            if div <= 0: div = 1

            cur_loss = (total_loss[0] / div)
            elapsed = time.time() - start_time
            
            ss = '| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f}'.format(
                    epoch, batch, batch_limit, (elapsed * 1000) / div, cur_loss
            )
            
            file_write(hyper_params['log_file'], ss)

            total_loss = 0
            start_time = time.time()

train_reader, test_reader, total_items = load_data(hyper_params)
# print(train_reader.data[:10])
# print(test_reader.data[:10])
# print(test_reader.data_te[:10])
hyper_params['total_items'] = total_items
hyper_params['testing_batch_limit'] = test_reader.num_b

file_write(hyper_params['log_file'], "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n")
file_write(hyper_params['log_file'], "Data reading complete!")
file_write(hyper_params['log_file'], "Number of train batches: {:4d}".format(train_reader.num_b))
file_write(hyper_params['log_file'], "Number of test batches: {:4d}".format(test_reader.num_b))
# file_write(hyper_params['log_file'], "Total Users: " + str(total_users))
file_write(hyper_params['log_file'], "Total Items: " + str(total_items) + "\n")

model = Model(hyper_params)
if is_cuda_available: model.cuda()

criterion = VAELoss(hyper_params)

if hyper_params['optimizer'] == 'adagrad':
    optimizer = torch.optim.Adagrad(
        model.parameters(), weight_decay=hyper_params['weight_decay'], lr=hyper_params['learning_rate']
    )
elif hyper_params['optimizer'] == 'adadelta':
    optimizer = torch.optim.Adadelta(
        model.parameters(), weight_decay=hyper_params['weight_decay']
    )
elif hyper_params['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=hyper_params['weight_decay']#, lr=hyper_params['learning_rate']
    )
elif hyper_params['optimizer'] == 'rmsprop':
    optimizer = torch.optim.RMSprop(
        model.parameters(), weight_decay=hyper_params['weight_decay']#, lr=hyper_params['learning_rate']
    )

file_write(hyper_params['log_file'], str(model))
file_write(hyper_params['log_file'], "\nModel Built!\nStarting Training...\n")

best_val_loss = None

try:
    for epoch in range(1, hyper_params['epochs'] + 1):
        epoch_start_time = time.time()
        
        train(train_reader)
        
        # Calulating the metrics on the train set
        metrics, _ = evaluate(model, criterion, train_reader, hyper_params, True)
        string = ""
        for m in metrics: string += " | " + m + ' = ' + str(metrics[m])
        string += ' (TRAIN)'
    
        # Calulating the metrics on the test set
        metrics, len_to_ndcg_at_100_map = evaluate(model, criterion, test_reader, hyper_params, False)
        string2 = ""
        for m in metrics: string2 += " | " + m + ' = ' + str(metrics[m])
        string2 += ' (TEST)'

        ss  = '-' * 89
        ss += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
        ss += string
        ss += '\n'
        ss += '-' * 89
        ss += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
        ss += string2
        ss += '\n'
        ss += '-' * 89
        file_write(hyper_params['log_file'], ss)
        
        # Plot sequence length vs NDCG@100 graph
        # plot_len_vs_ndcg(len_to_ndcg_at_100_map)
        
        if not best_val_loss or metrics['loss'] <= best_val_loss:
            with open(hyper_params['model_file_name'], 'wb') as f: torch.save(model, f)
            best_val_loss = metrics['loss']

except KeyboardInterrupt: print('Exiting from training early')

# Checking metrics on best saved model
with open(hyper_params['model_file_name'], 'rb') as f: model = torch.load(f)
metrics, len_to_ndcg_at_100_map = evaluate(model, criterion, test_reader, hyper_params, False)

string = ""
for m in metrics: string += " | " + m + ' = ' + str(metrics[m])

ss  = '=' * 89
ss += '\n| End of training'
ss += string
ss += '\n'
ss += '=' * 89
file_write(hyper_params['log_file'], ss)

# Plot sequence length vs NDCG@100 graph
plot_len_vs_ndcg(len_to_ndcg_at_100_map)

# Plot Traning graph
f = open(model.hyper_params['log_file'])
lines = f.readlines()
lines.reverse()

train = []
test = []

for line in lines:
    if line[:10] == 'Simulation' and len(train) > 5: break
    elif line[:10] == 'Simulation' and len(train) <= 5: train, test = [], []
        
    if line[2:5] == 'end' and line[-6:-2] == 'TEST': test.append(line.strip().split("|"))
    elif line[2:5] == 'end' and line[-7:-2] == 'TRAIN': train.append(line.strip().split("|"))

train.reverse()
test.reverse()

train_cp, train_ndcg = [], []
test_cp, test_ndcg = [], []

for i in train:
    train_cp.append(float(i[3].split('=')[1].strip(' ')))
    train_ndcg.append(float(i[-3].split('=')[1].split(' ')[1]))
    
for i in test:
    test_cp.append(float(i[3].split('=')[1].strip(' ')))
    test_ndcg.append(float(i[-3].split('=')[1].split(' ')[1]))

plt.figure(figsize=(12, 5))
plt.plot(train_ndcg, label='Train set')
plt.plot(test_ndcg, label='Test set')
plt.ylabel("NDCG@100")
plt.xlabel("Epochs")

leg = plt.legend(loc='best', ncol=2)
pass

