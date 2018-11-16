
# coding: utf-8

# # VAE for ranking items

# ## Model Formalization
# 
# For each user $u \in U$, we have a set, $P_u$ = { $(m_1, m_2)$ | $rating_u^{m_1}$ > $rating_u^{m_2}$) } 
# 
# $P$ =  $\bigcup\limits_{\forall u \; \in \; U} P_u$
# 
# $\forall (u, m_1, m_2) \in P, $ we send two inputs, $x_1 = u \Vert m_1$ and $x_2 = u \Vert m_2$ to a VAE (with the same parameters).
# 
# We expect the VAE's encoder to produce $z_1$ (sampled from the distribution: $(\mu_1 , \Sigma_1$)) from $x_1$ ; and similarly $z_2$ from $x_2$ using the parameters $\theta$.
# 
# The decoder network is expected to learn a mapping function $f_{\phi}$ from $z_1$ to $m_1$.
# 
# We currently have 2 ideas for the decoder network:
# 1. Using two sets of network parameters, $\phi$ and $\psi$ for $z_1$ and $z_2$ respectively.
# 2. Using $\phi$ for both $z_1$ and $z_2$.
# 
# For ranking the pairs of movies, we have another network:
# 1. The input of the network is $z_1 \Vert z_2$, 
# 2. Is expected to learn a mapping, $f_{\delta}$ to a bernoulli distribution over True/False, modelling $rating_u^{m_1} > rating_u^{m_2}$.
# 
# ## Loss Function
# 
# $$Loss \; = \; KL( \, \phi(z_1 \vert x_1) \Vert {\rm I\!N(0, I)} \, ) \; + \; KL( \, \psi(z_2 \vert x_2) \Vert {\rm I\!N(0, I)} \, ) \; - \; \sum_{i} m_{1i} \, log( \, f_{\phi}(z_1)_i ) \; - \; \sum_{i} m_{2i} \, log( \, f_{\psi}(z_2)_i ) \; - \; f_{\delta}(z_1 \Vert z_2) $$

# # Imports

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import gc
import time
import json
import pickle
import random
import functools
import numpy as np
from tqdm import tqdm
import datetime as dt
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# # Utlity functions

# In[2]:


LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor

is_cuda_available = torch.cuda.is_available()
# is_cuda_available = False

if is_cuda_available: 
    print("Using CUDA...\n")
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor

def before_rnn(seq_tensor, seq_lengths):
    # print(seq_lengths)
    seq_lengths = LongTensor(seq_lengths)
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    _, reverse_perm_idx = perm_idx.sort(0)
    seq_tensor = seq_tensor[perm_idx]

    return pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy(), batch_first=True), reverse_perm_idx

def after_rnn(output, reverse_perm_idx):
    ret, _ = pad_packed_sequence(output, batch_first=True)
    return ret[reverse_perm_idx]
    
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


# # Hyper Parameters

# In[3]:


hyper_params = {
    'data_base': 'saved_data/',
    'project_name': 'ranking_vae_recurrent',
    'model_file_name': '',
    'log_file': '',
    'history_split_test': [0.8, 0.2], # Part of test history to train on : Part of test history to test

    'learning_rate': 0.01, # if optimizer is adadelta, learning rate is not required
    'optimizer': 'adam',
    'weight_decay': float(1e-4),

    'epochs': 200,
    'batch_size': 1,
    'dynamic_rnn': False,

    'item_embed_size': 128,
    'rnn_size': 100,
    'hidden_size': 64,
    'latent_size': 32,

    'number_users_to_keep': 100000000000,
    'batch_log_interval': 1000,
    'train_cp_users': 200,
}

file_name = '_optimizer_' + str(hyper_params['optimizer'])
if hyper_params['optimizer'] == 'adagrad':
    file_name += '_lr_' + str(hyper_params['learning_rate'])
file_name += '_item_embed_size_' + str(hyper_params['item_embed_size'])
file_name += '_weight_decay_' + str(hyper_params['weight_decay'])

hyper_params['log_file'] = 'saved_logs/' + hyper_params['project_name'] + '_log' + file_name + '.txt'
hyper_params['model_file_name'] = 'saved_models/' + hyper_params['project_name'] + '_model' + file_name + '.pt'


# # Data Parsing

# In[4]:


def load_data(hyper_params):
    
    file_write(hyper_params['log_file'], "Started reading data file")
    
    train = load_obj_json(hyper_params['data_base'] + 'train_' + hyper_params['project_name'])
    test = load_obj_json(hyper_params['data_base'] + 'test_' + hyper_params['project_name'])
    num_users = load_obj_json(hyper_params['data_base'] + 'num_users_' + hyper_params['project_name'])
    num_items = load_obj_json(hyper_params['data_base'] + 'num_items_' + hyper_params['project_name'])

    file_write(hyper_params['log_file'], "Data Files loaded!")

    train_reader = DataReader(hyper_params, train, num_users, num_items, True)
    test_reader = DataReader(hyper_params, test, num_users, num_items, False)

    return train_reader, test_reader, num_users, num_items

def scatter(x):
    max_len = max(list(map(len, x)))
    
    this = 0.0
    if type(x[0][0]) == int: this = 0

    ret = [[this] * max_len] * len(x)
    
    for i in range(len(x)):
        for j in range(len(x[i])):
            ret[i][j] = x[i][j]
    
    return ret

class DataReader:

    def __init__(self, hyper_params, data, num_users, num_items, is_training):
        self.hyper_params = hyper_params
        self.batch_size = hyper_params['batch_size']
        self.num_users = num_users
        self.num_items = num_items
        self.data = data
        self.is_training = is_training
        self.all_users = []

        self.number()

    def number(self):
        users_done = 0
        count = 0
        y_batch = []

        for user in range(len(self.data)):

            if users_done > self.hyper_params['number_users_to_keep']: break
            users_done += 1

            y_batch.append(0)

            if len(y_batch) == self.batch_size:
                y_batch = []
                count += 1

        self.num_b = count
        
    def iter(self):
        users_done = 0

        x_batch_item = []
        y_batch = []
        y_batch_s = torch.zeros(self.batch_size, self.hyper_params['total_items']+1).cuda()

        for user in range(len(self.data)):

            if users_done > self.hyper_params['number_users_to_keep']: break
            users_done += 1
            
            x_batch_item.append([ i[0] for i in self.data[user][:-1] ])
            y_batch.append([ i[0] for i in self.data[user][1:] ])
            y_batch_s[0, :].scatter_(0, LongTensor([ i[0] for i in self.data[user][1:] ]), 1.0)

            if len(y_batch) == self.batch_size:

                yield Variable(LongTensor(scatter(x_batch_item))),                 y_batch, Variable(y_batch_s, requires_grad=False),                 list(map(len, x_batch_item))

                x_batch_item = []
                y_batch = []
                y_batch_s = torch.zeros(self.batch_size, self.hyper_params['total_items']+1).cuda()

    def iter_eval(self):

        for user in range(len(self.data)):
            test_movies = []
            
            split_point = int(hyper_params['history_split_test'][0] * len(self.data[user]))
            
            x_batch_item = [ i[0] for i in self.data[user][:split_point] ]
            y_batch = [ i[0] for i in self.data[user][1:split_point+1] ]
            y_batch_s = torch.zeros(1, self.hyper_params['total_items']+1).cuda()
            y_batch_s[0, :].scatter_(0, LongTensor([ i[0] for i in self.data[user][1:split_point+1] ]), 1.0)
            test_movies = [ i[0] for i in self.data[user][split_point:] ]
            test_movies_r = [ i[1] for i in self.data[user][split_point:] ]
                
            yield Variable(LongTensor([ x_batch_item ])),             [ y_batch ], Variable(y_batch_s, requires_grad=False),             list(map(len, [ x_batch_item ])), [ test_movies ], [ test_movies_r ]


# # Evaluation Code

# In[5]:


def evaluate(model, criterion, reader, hyper_params, is_train_set):
    model.eval()

    metrics = {}
#     metrics['CP'] = 0.0
#     metrics['ZEROS'] = 0.0
    metrics['loss'] = 0.0
    Ks = [100]
    for k in Ks: 
        metrics['NDCG@' + str(k)] = 0.0
        metrics['HR@' + str(k)] = 0.0

    cp = 0
    ncp = 0
    zer = 0
    total = 0
    total_ndcg = 0
    batch = 0

    for x, y, y_s, seq_lengths, test_movies, test_movies_r in reader.iter_eval():
        batch += 1
        if is_train_set == True and batch > hyper_params['train_cp_users']: break

        decoder_output, z_mean, z_sigma = model(x, seq_lengths)
        metrics['loss'] += criterion(decoder_output, z_mean, z_sigma, y, y_s).data[0]
        
        last_predictions = decoder_output[:, -1, :]
        
        for batch_num in range(last_predictions.shape[0]):
            predicted_scores = last_predictions[batch_num]
            actual_movies_watched = test_movies[batch_num]
            actual_movies_ratings = test_movies_r[batch_num]
            
            # Calculate CP
#             for m1 in range(len(actual_movies_watched)):
#                 for m2 in range(m1+1, len(actual_movies_watched)):
#                     if actual_movies_ratings[m1] == actual_movies_ratings[m2]: continue
                    
#                     s1 = float(predicted_scores[actual_movies_watched[m1]].data)
#                     s2 = float(predicted_scores[actual_movies_watched[m2]].data)
#                     temp_product = (actual_movies_ratings[m1] - actual_movies_ratings[m2]) * (s1-s2)
                    
#                     if temp_product == 0.0: zer += 1.0
#                     elif temp_product > 0.0: cp += 1.0
#                     elif temp_product < 0.0: ncp += 1.0
#                     total += 1.0
                    
            # Calculate NDCG
            _, argsorted = torch.sort(predicted_scores, -1, True)
            total_ndcg += 1
            for k in Ks:
                best = 0.0
                now_at = 0.0
                dcg = 0.0
                hr = 0.0
                
                rec_list = list(argsorted[:k].data)
                for m in range(len(actual_movies_watched)):
                    movie = actual_movies_watched[m]
                    now_at += 1.0
                    if now_at <= k: best += 1.0 / np.log2(now_at + 1)
                    
                    if movie not in rec_list: continue
                    hr += 1.0
                    dcg += 1.0 / np.log2(float(rec_list.index(movie) + 2))
                metrics['NDCG@' + str(k)] += float(dcg) / float(best)
                metrics['HR@' + str(k)] += float(hr) / float(min(k, len(actual_movies_watched)))

#     assert cp + ncp + zer == total

#     metrics['CP'] = float(cp) / float(total)
#     metrics['CP'] *= 100.0
#     metrics['CP'] = round(metrics['CP'], 4)

#     metrics['ZEROS'] = float(zer) / float(total)
#     metrics['ZEROS'] *= 100.0
#     metrics['ZEROS'] = round(metrics['ZEROS'], 4)

    metrics['loss'] = float(metrics['loss']) / float(batch)
    metrics['loss'] = round(metrics['loss'], 4)
    
    for k in Ks:
        metrics['NDCG@' + str(k)] = round(100.0 * metrics['NDCG@' + str(k)] / float(total_ndcg), 4)
        metrics['HR@' + str(k)] = round(100.0 * metrics['HR@' + str(k)] / float(total_ndcg), 4)

    return metrics


# # Model

# In[6]:


class Encoder(nn.Module):
    def __init__(self, hyper_params):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(
            hyper_params['rnn_size'], hyper_params['hidden_size']
        )
        nn.init.xavier_normal(self.linear1.weight)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hyper_params):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(hyper_params['latent_size'], hyper_params['hidden_size'])
        self.linear2 = nn.Linear(hyper_params['hidden_size'], hyper_params['total_items']+1)
        nn.init.xavier_normal(self.linear1.weight)
        nn.init.xavier_normal(self.linear2.weight)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)
        
        x = self.linear2(x)
        x = self.dropout(x)
#         x = self.activation(x)
        return x

class Model(nn.Module):
    def __init__(self, hyper_params):
        super(Model, self).__init__()
        self.hyper_params = hyper_params
        
        self.encoder = Encoder(hyper_params)
        self.decoder = Decoder(hyper_params)
        
        self.gru = nn.GRU(hyper_params['item_embed_size'], hyper_params['rnn_size'], batch_first=True, num_layers=1, dropout=0.2)
        
        self._enc_mu = nn.Linear(hyper_params['hidden_size'], hyper_params['latent_size'])
        self._enc_log_sigma = nn.Linear(hyper_params['hidden_size'], hyper_params['latent_size'])
        nn.init.xavier_normal(self._enc_mu.weight)
        nn.init.xavier_normal(self._enc_log_sigma.weight)
        
        self.item_embed = nn.Embedding(hyper_params['total_items']+1, hyper_params['item_embed_size'])
        nn.init.normal(self.item_embed.weight.data, mean=0, std=0.01)

        self.dropout = nn.Dropout(0.2)
        
        # self.activation = nn.ReLU()
        # self.activation_last = nn.Tanh()
        # if self.hyper_params['loss_type'] == 'bce': self.activation_last = nn.Sigmoid()
        # xavier_uniform
        
        # self.dropout = nn.Dropout(0.2)
        
    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        mu = self.dropout(mu)
        log_sigma = self._enc_log_sigma(h_enc)
        log_sigma = self.dropout(log_sigma)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        if is_cuda_available: std_z = std_z.cuda()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, x, seq_lengths):
        x_shape = x.shape
        
        item = self.item_embed(x)
        
        self.gru.flatten_parameters()
        if self.hyper_params['dynamic_rnn'] == True: z_packed, reverse_perm = before_rnn(item, seq_lengths)
        else: z_packed = item
        
        output, _ = self.gru(z_packed)
        
        if self.hyper_params['dynamic_rnn'] == True: output = after_rnn(output, reverse_perm)
        output = output.contiguous().view(x_shape[0] * x_shape[1], -1)
        
        h_enc = self.encoder(output)
        z = self.sample_latent(h_enc)
        z = z.view(x_shape[0], x_shape[1], -1)
        
        dec = self.decoder(z)
                              
        return dec.view(x_shape[0], x_shape[1], -1), self.z_mean, self.z_sigma


# # Custom loss

# In[7]:


class VAELoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super(VAELoss,self).__init__()
        self.cce = nn.CrossEntropyLoss(size_average=True)

    def forward(self, decoder_output, z_mean, z_sigma, y_true, y_true_s):
        
        mean_sq = z_mean * z_mean
        stddev_sq = z_sigma * z_sigma
        kld = torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
        
        likelihood = None
        decoder_output = F.log_softmax(decoder_output, -1)
        div_l = 0.0
        for s1 in range(decoder_output.shape[1]):  
            temp = y_true_s * decoder_output[:, s1, :]

            if likelihood is None: 
                likelihood = -torch.sum(temp, -1) / float(decoder_output.shape[1] - s1)
            else: 
                likelihood += -torch.sum(temp, -1) / float(decoder_output.shape[1] - s1)
            div_l += 1.0
            
            temp = y_true_s.clone()
            temp[0, y_true[0][s1]] = 0.0
            y_true_s = temp
                    
        final = (0.1 * kld) + (1 * (likelihood / div_l))
        
        return final


# # Training loop

# In[ ]:


def train(reader):
    model.train()
    total_loss = 0
    start_time = time.time()
    batch = 0
    batch_limit = int(train_reader.num_b)

    for x, y, y_s, seq_lengths in reader.iter():
        batch += 1
        
        model.zero_grad()
        optimizer.zero_grad()

        decoder_output, z_mean, z_sigma = model(x, seq_lengths)
        
        loss = criterion(decoder_output, z_mean, z_sigma, y, y_s)
        loss.backward()

        optimizer.step()

        total_loss += loss.data

        if (batch % hyper_params['batch_log_interval'] == 0 and batch > 0) or batch == batch_limit:
            div = hyper_params['batch_log_interval']
            if batch == batch_limit: div = (batch_limit % hyper_params['batch_log_interval']) - 1
            if div <= 0: div = 1

            cur_loss = (total_loss[0] / div)
            elapsed = time.time() - start_time
            
            # print(x.shape)
            ss = '| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f}'.format(
                    epoch, batch, batch_limit, (elapsed * 1000) / div, cur_loss
            )
            
            file_write(hyper_params['log_file'], ss)

            total_loss = 0
            start_time = time.time()

train_reader, test_reader, total_users, total_items = load_data(hyper_params)
hyper_params['total_users'] = total_users
hyper_params['total_items'] = total_items
hyper_params['testing_batch_limit'] = test_reader.num_b

file_write(hyper_params['log_file'], "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n")
file_write(hyper_params['log_file'], "Data reading complete!")
file_write(hyper_params['log_file'], "Number of train batches: {:4d}".format(train_reader.num_b))
file_write(hyper_params['log_file'], "Number of test batches: {:4d}".format(test_reader.num_b))
file_write(hyper_params['log_file'], "Total Users: " + str(total_users))
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
        metrics = evaluate(model, criterion, train_reader, hyper_params, True)
        string = ""
        for m in metrics: string += " | " + m + ' = ' + str(metrics[m])
        string += ' (TRAIN)'
    
        # Calulating the metrics on the test set
        metrics = evaluate(model, criterion, test_reader, hyper_params, False)
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
        
        if not best_val_loss or metrics['loss'] <= best_val_loss:
            with open(hyper_params['model_file_name'], 'wb') as f: torch.save(model, f)
            best_val_loss = metrics['loss']

except KeyboardInterrupt: print('Exiting from training early')

# Checking metrics on best saved model
with open(hyper_params['model_file_name'], 'rb') as f: model = torch.load(f)
metrics = evaluate(model, criterion, test_reader, hyper_params, False)

string = ""
for m in metrics: string += " | " + m + ' = ' + str(metrics[m])

ss  = '=' * 89
ss += '\n| End of training'
ss += string
ss += '\n'
ss += '=' * 89
file_write(hyper_params['log_file'], ss)

# Plot Traning graph
f = open(model.hyper_params['log_file'])
lines = f.readlines()
lines.reverse()

train = []
test = []

for line in lines:
    if line[:10] == 'Simulation' and len(train) > 0: break
    if line[2:5] == 'end' and line[-6:-2] == 'TEST': test.append(line.strip().split("|"))
    elif line[2:5] == 'end' and line[-7:-2] == 'TRAIN': train.append(line.strip().split("|"))

train.reverse()
test.reverse()

train_cp, train_ndcg = [], []
test_cp, test_ndcg = [], []

for i in train:
    train_cp.append(float(i[3].split('=')[1].strip(' ')))
    train_ndcg.append(float(i[-2].split('=')[1].split(' ')[1]))
    
for i in test:
    test_cp.append(float(i[3].split('=')[1].strip(' ')))
    test_ndcg.append(float(i[-2].split('=')[1].split(' ')[1]))

#plt.figure(figsize=(12, 5))
#plt.plot(train_ndcg, label='Train')
#plt.plot(test_ndcg, label='Test')
#plt.ylabel("NDCG@100")
#plt.xlabel("Epochs")

#leg = plt.legend(loc='best', ncol=2)
#pass

