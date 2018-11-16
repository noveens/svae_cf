
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

import gc
import time
import json
import pickle
import random
import functools
import numpy as np
from tqdm import tqdm
import datetime as dt


# # Utlity functions

# In[2]:


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


# # Hyper Parameters

# In[3]:


hyper_params = {
    'data_base': 'saved_data/',
    'project_name': 'ranking_vae',
    'model_file_name': '',
    'log_file': '',
    'data_split': [0.8, 0.2], # Train : Test
    'max_user_hist': 100,
    'min_user_hist': 5,

    'learning_rate': 0.05, # if optimizer is adadelta, learning rate is not required
    'optimizer': 'adam',
    'loss_type': 'easy_hinge',
    'm_loss': float(1),
    'weight_decay': float(1e-3),

    'epochs': 50,
    'batch_size': 128,

    'user_embed_size': 100,
    'item_embed_size': 100,
    
    'hidden_size': 128,
    'latent_size': 100,

    'number_users_to_keep': 1000000000000,
    'batch_log_interval': 3000,
}

file_name = '_optimizer_' + str(hyper_params['optimizer'])
if hyper_params['optimizer'] != 'adadelta':
    file_name += '_lr_' + str(hyper_params['learning_rate'])
file_name += '_user_embed_size_' + str(hyper_params['user_embed_size'])
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
    user_hist = load_obj_json(hyper_params['data_base'] + 'user_hist_' + hyper_params['project_name'])
    item_hist = load_obj_json(hyper_params['data_base'] + 'item_hist_' + hyper_params['project_name'])

    file_write(hyper_params['log_file'], "Data Files loaded!")

    train_reader = DataReader(hyper_params, train, len(user_hist), item_hist, True)
    test_reader = DataReader(hyper_params, test, len(user_hist), item_hist, False)

    return train_reader, test_reader, len(user_hist), len(item_hist)

class DataReader:

    def __init__(self, hyper_params, data, num_users, item_hist, is_training):
        self.hyper_params = hyper_params
        self.batch_size = hyper_params['batch_size']
        self.item_hist = item_hist
        self.num_users = num_users
        self.num_items = len(item_hist)
        self.data = data
        self.is_training = is_training
        self.all_users = []

        self.number_users()
        self.number()

    def number(self):
        users_done = 0
        count = 0
        y_batch = []

        for user in self.data:

            if users_done > self.hyper_params['number_users_to_keep']: break
            users_done += 1

            for i in range(len(self.data[user])):
                for ii in range(i+1, len(self.data[user])):
                    if self.data[user][i][1] == self.data[user][ii][1]: continue

                    y_batch.append(0)

                    if len(y_batch) == self.batch_size:
                        y_batch = []
                        count += 1

        self.num_b = count

    def number_users(self):
        users_done = 0
        count = 0

        for user in self.data:

            if users_done > self.hyper_params['number_users_to_keep']: break
            users_done += 1

            y_batch = []

            for i in range(len(self.data[user])):
                for ii in range(i+1, len(self.data[user])):
                    if self.data[user][i][1] == self.data[user][ii][1]: continue

                    y_batch.append(0)
                    y_batch.append(0)

            if len(y_batch) > 0:
                count += 1
                self.all_users.append(user)

        self.num_u = count

    def iter(self):
        users_done = 0

        x_batch_user = []
        x_batch_item = []
        y_batch = []

        for user in self.data:

            if users_done > self.hyper_params['number_users_to_keep']: break
            users_done += 1

            for i in range(len(self.data[user])):
                for ii in range(i+1, len(self.data[user])):
                    # If rating is equal, fuck off
                    if self.data[user][i][1] == self.data[user][ii][1]: continue

                    first, second = i, ii
                    
                    # Uncomment the below line to always send the greater one first, i.e y = 1.0 always
                    # if self.data[user][i][1] < self.data[user][ii][1]: first, second = second, first

                    x_batch_user.append(int(user))
                    x_batch_user.append(int(user))

                    x_batch_item.append(self.data[user][first][0])
                    x_batch_item.append(self.data[user][second][0])
                        
                    y = 1.0
                    if float(self.data[user][first][1]) < float(self.data[user][second][1]): y = -1.0
                    if y < 0.0 and self.hyper_params['loss_type'] == 'bce': y = 0.0
                    y_batch.append(float(y))

                    if len(y_batch) == self.batch_size:

                        yield [
                            Variable(LongTensor(x_batch_user[::2])), 
                            Variable(LongTensor(x_batch_item[::2])),
                        ], [
                            Variable(LongTensor(x_batch_user[1::2])), 
                            Variable(LongTensor(x_batch_item[1::2])),
                        ], Variable(FloatTensor(y_batch))
                        
                        x_batch_user = []
                        x_batch_item = []
                        y_batch = []

    def iter_eval(self):

        num_active_users = len(self.all_users)
        num_active_users = min(num_active_users, self.hyper_params['number_users_to_keep'])

        for user_now in tqdm(range(num_active_users)):
            user = self.all_users[user_now]

            x_batch_user = []
            x_batch_item = []
            y_batch = []
            all_movies = []

            for i in range(len(self.data[user])):
                all_movies.append(self.data[user][i][0])

                for ii in range(i+1, len(self.data[user])):
                    # If rating is equal, fuck off
                    if self.data[user][i][1] == self.data[user][ii][1]: continue

                    first, second = i, ii
                    
                    # Uncomment the below line to always send the greater one first, i.e y = 1.0 always
                    # if self.data[user][i][1] < self.data[user][ii][1]: first, second = second, first

                    x_batch_user.append(int(user))
                    x_batch_user.append(int(user))

                    x_batch_item.append(self.data[user][first][0])
                    x_batch_item.append(self.data[user][second][0])

                    y_batch.append(float(self.data[user][first][1])) # Already divided by 5
                    y_batch.append(float(self.data[user][second][1])) # Already divided by 5

            if len(y_batch) > 0:
                # print(y_batch)
                yield [
                    Variable(LongTensor(x_batch_user[::2])),
                    Variable(LongTensor(x_batch_item[::2])),
                ], [
                    Variable(LongTensor(x_batch_user[1::2])), 
                    Variable(LongTensor(x_batch_item[1::2])),
                ], Variable(FloatTensor(y_batch[::2])), Variable(FloatTensor(y_batch[1::2])), all_movies


# # Evaluation Code (for BCE PLEAE CHANGE ME!)

# In[5]:


def map_int(a):
    if float(a.data) < 0.0: return -1
    if float(a.data) > 0.0: return 1
    return 0
#     if float(a.data) > 0.5:
#         return 1
#     if float(a.data) < 0.5:
#         return -1
#     return 0

def evaluate_ndcg(model, criterion, reader, hyper_params):
    model.eval()

    ret = 0.0
    
    NDCG = {}
    total_NDCG = {}

    Ks = [1, 5, 10, 15, 20]

    for k in Ks:
        NDCG[str(k)] = 0.0
        total_NDCG[str(k)] = 0.0

    total = 0
    user_done = 0

    for x1, x2, y1, y2, all_movies in reader.iter_eval():
        user_done += 1

        _, output = model(x1, x2)

        y_diff = torch.gt(y1, y2).float() - torch.gt(y1, y2).float()

        y_pair_map = {}
        out_pair_map = {}
        y_r = {}
        for i in range(x1[0].shape[0]):
            # if float(y_diff[i].data) == 0.0: continue
            
            m1 = int(x1[1][i])
            m2 = int(x2[1][i])

            y_r[str(m1)] = round(float(y1[i].data), 2)
            y_r[str(m2)] = round(float(y2[i].data), 2)

            if m1 not in y_pair_map: y_pair_map[m1] = {}
            if m2 not in y_pair_map: y_pair_map[m2] = {}
            if m1 not in out_pair_map: out_pair_map[m1] = {}
            if m2 not in out_pair_map: out_pair_map[m2] = {}

            y_pair_map[m1][m2] = int(y_diff[i])
            y_pair_map[m2][m1] = -1 * int(y_diff[i])

            out_pair_map[m1][m2] = map_int(output[i])
            out_pair_map[m2][m1] = -1 * map_int(output[i])

        def compare_out(item1, item2):
            if item2 not in out_pair_map[item1]: return 0
            return out_pair_map[item1][item2]

        all_movies = sorted(all_movies, key=functools.cmp_to_key(compare_out))
        all_movies.reverse()
        
        final = []
        final_sorted = []
        for i in all_movies: 
            final.append([i, y_r[str(i)]])
            final_sorted.append([i, y_r[str(i)]])
        final_sorted = sorted(final_sorted, key=lambda x: x[1])
        final_sorted.reverse()
        
        # Calculate NDCG
        for k in Ks:
            if k <= len(final):
                out = final[:k]
                out_sorted = final_sorted[:k]

                now = 0.0
                now_best = 0.0
                for i in range(k):
                    now += float(out[i][1]) / float(np.log2(i+2))
                    now_best += float(out_sorted[i][1]) / float(np.log2(i+2))

                NDCG[str(k)] += float(now) / float(now_best)
                total_NDCG[str(k)] += 1.0

    for k in NDCG:
        if total_NDCG[k] > 0: NDCG[k] = float(NDCG[k]) / float(total_NDCG[k])
        NDCG[k] *= 100.0
        NDCG[k] = round(NDCG[k], 4)

    return NDCG

def evaluate(model, criterion, reader, hyper_params, is_train_set):
    model.eval()

    metrics = {}
    metrics['CP'] = 0.0
    metrics['ZEROS'] = 0.0
    metrics['loss'] = 0.0

    correct = 0
    not_correct = 0
    zeros = 0
    total = 0
    batch = 0

    for x1, x2, y in reader.iter():
        batch += 1
        if is_train_set == True and batch > hyper_params['testing_batch_limit']: break

        o1, output = model(x1, x2)

        metrics['loss'] += criterion(o1, output, y, x1[1], x2[1]).data
        
        if hyper_params['loss_type'] != 'bce':
            temp_correct  = int(torch.sum(torch.lt(y, 0.0) * torch.lt(output, 0.0)))
            temp_correct += int(torch.sum(torch.gt(y, 0.0) * torch.gt(output, 0.0)))

            temp_not_correct  = int(torch.sum(torch.lt(y, 0.0) * torch.gt(output, 0.0)))
            temp_not_correct += int(torch.sum(torch.gt(y, 0.0) * torch.lt(output, 0.0)))

            temp_zeros = int(torch.sum(torch.eq(output, 0.0)))
        else:
            temp_correct  = int(torch.sum(torch.eq(y, 0.0) * torch.lt(output, 0.5)))
            temp_correct += int(torch.sum(torch.eq(y, 1.0) * torch.gt(output, 0.5)))

            temp_not_correct  = int(torch.sum(torch.eq(y, 0.0) * torch.gt(output, 0.5)))
            temp_not_correct += int(torch.sum(torch.eq(y, 1.0) * torch.lt(output, 0.5)))

            temp_zeros = int(torch.sum(torch.eq(output, 0.5)))

        correct += temp_correct
        not_correct += temp_not_correct
        zeros += temp_zeros
        total += int(y.shape[0])

        assert temp_correct + temp_not_correct + temp_zeros == int(y.shape[0])

    assert correct + not_correct + zeros == total

    metrics['CP'] = float(correct) / float(total)
    metrics['CP'] *= 100.0
    metrics['CP'] = round(metrics['CP'], 4)

    metrics['ZEROS'] = float(zeros) / float(total)
    metrics['ZEROS'] *= 100.0
    metrics['ZEROS'] = round(metrics['ZEROS'], 4)

    metrics['loss'] = float(metrics['loss'][0]) / float(batch)
    metrics['loss'] = round(metrics['loss'], 4)

    if is_train_set == False:
        ndcg = evaluate_ndcg(model, criterion, reader, hyper_params)
        for k in ndcg: metrics['NDCG@' + str(k)] = ndcg[k]

    return metrics


# # Model

# In[6]:


class Encoder(nn.Module):
    def __init__(self, hyper_params):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(
            hyper_params['user_embed_size'] + hyper_params['item_embed_size'], hyper_params['hidden_size']
        )
        nn.init.xavier_normal(self.linear1.weight)
        self.activation = nn.ReLU()

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
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        
        x = self.linear2(x)
        x = self.activation(x)
        return x

class Model(nn.Module):
    def __init__(self, hyper_params):
        super(Model, self).__init__()
        self.hyper_params = hyper_params
        
        self.encoder = Encoder(hyper_params)
        self.decoder = Decoder(hyper_params)
        
        self._enc_mu = nn.Linear(hyper_params['hidden_size'], hyper_params['latent_size'])
        self._enc_log_sigma = nn.Linear(hyper_params['hidden_size'], hyper_params['latent_size'])
        nn.init.xavier_normal(self._enc_mu.weight)
        nn.init.xavier_normal(self._enc_log_sigma.weight)
        
        self.user_embed = nn.Embedding(hyper_params['total_users'], hyper_params['user_embed_size'])
        self.item_embed = nn.Embedding(hyper_params['total_items'], hyper_params['item_embed_size'])
        nn.init.normal(self.user_embed.weight.data, mean=0, std=0.01)
        nn.init.normal(self.item_embed.weight.data, mean=0, std=0.01)

        self.activation_last = nn.Tanh()
        if self.hyper_params['loss_type'] == 'bce': self.activation_last = nn.Sigmoid()
        
        prev = hyper_params['latent_size'] * 2
        self.layer_hinge = nn.Linear(prev, 1)
        nn.init.xavier_normal(self.layer_hinge.weight)
        # xavier_uniform
        
        # self.dropout = nn.Dropout(0.2)
        
    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().cuda()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, x1, x2):
        user1 = self.user_embed(x1[0])
        item1 = self.item_embed(x1[1])
        
        h_enc1 = self.encoder(torch.cat([user1, item1], dim=-1))
        z1 = self.sample_latent(h_enc1)
        dec1 = self.decoder(z1)
                              
        zm1 = self.z_mean
        zs1 = self.z_sigma
        
        user2 = self.user_embed(x2[0])
        item2 = self.item_embed(x2[1])
                              
        h_enc2 = self.encoder(torch.cat([user2, item2], dim=-1))
        z2 = self.sample_latent(h_enc2)
        dec2 = self.decoder(z2)
                              
        output = self.layer_hinge(torch.cat([z1, z2], dim=-1))
        # output = self.dropout(output)
        if self.hyper_params['loss_type'] == 'easy_hinge': 
            output = 10.0 * self.activation_last(output / 10.0)
        else:
            output = self.activation_last(output)
                              
        return [dec1, dec2, zm1, zs1, self.z_mean, self.z_sigma], output.squeeze(-1)


# # Custom loss

# In[7]:


class VAELoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super(VAELoss,self).__init__()

        self.loss_type = hyper_params['loss_type']
        self.m_loss = hyper_params['m_loss']
        batch_size = hyper_params['batch_size']

        self.zeros_while_max = torch.zeros(int(batch_size)).float()
        if is_cuda_available: self.zeros_while_max = self.zeros_while_max.cuda()
        self.zeros_while_max = Variable(self.zeros_while_max)
        self.exp = Variable(FloatTensor([np.e]))
        self.cce = nn.CrossEntropyLoss(size_average=True)
        self.bce = nn.BCELoss(size_average=True)

    def forward(self, o1, o2, y, t1, t2):
        
        m1, m2, zm1, zs1, zm2, zs2 = o1
        
        mean_sq1 = zm1 * zm1
        stddev_sq1 = zs1 * zs1
        kld  = torch.mean(mean_sq1 + stddev_sq1 - torch.log(stddev_sq1) - 1)
        
        mean_sq2 = zm2 * zm2
        stddev_sq2 = zs2 * zs2
        kld += torch.mean(mean_sq2 + stddev_sq2 - torch.log(stddev_sq2) - 1)
        
        likelihood  = self.cce(m1, t1) # Try using a weight parameters, alpha when summing
        likelihood += self.cce(m2, t2) # Try using a weight parameters, alpha when summing
        
        # Reference: https://papers.nips.cc/paper/3708-ranking-measures-and-loss-functions-in-learning-to-rank.pdf
        if self.loss_type == 'hinge':
            pairwise_loss = self.m_loss - (y * o2)
            pairwise_loss = torch.mean(torch.max(self.zeros_while_max, pairwise_loss))
            
        elif self.loss_type == 'bce':
            pairwise_loss = self.bce(o2, y)
            
        elif self.loss_type == 'easy_hinge':
            # pairwise_loss = torch.log(2.0 - (y * o2))# / np.log(2) # torch.log is base "e"
            pairwise_loss = torch.log((o2*o2) - (10*y*o2) + 26) - 2
            pairwise_loss = torch.mean(torch.max(self.zeros_while_max, pairwise_loss))
        
        elif self.loss_type == 'saddle':
            pairwise_loss = torch.pow(y + o2, 2)

        elif self.loss_type == 'exp':
            pairwise_loss = torch.pow(self.exp, y * o2)

        elif self.loss_type == 'logistic':
            pairwise_loss = torch.log(self.m_loss + torch.pow(self.exp, -(y * o2)))
        
        final = (0.15 * kld) + (0.15 * likelihood) + (0.7 * pairwise_loss)
        
        return final


# # Training loop

# In[ ]:


def train(reader):
    model.train()
    total_loss = 0
    start_time = time.time()
    batch = 0
    batch_limit = int(train_reader.num_b)

    for x1, x2, y in reader.iter():
        batch += 1
        
        model.zero_grad()
        optimizer.zero_grad()

        o1, o2 = model(x1, x2)
        
        loss = criterion(o1, o2, y, x1[1], x2[1])
        loss.backward()

        optimizer.step()

        total_loss += loss.data

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

