
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
    'project_name': 'ranking_vae_recurrent',
    'model_file_name': '',
    'log_file': '',
    'data_split': [0.8, 0.2], # Train : Test
    'max_user_hist': 100,
    'min_user_hist': 5,

    'learning_rate': 0.05, # if optimizer is adadelta, learning rate is not required
    'optimizer': 'adam',
    'loss_type': 'hinge',
    'm_loss': float(1),
    'weight_decay': float(1e-4),

    'epochs': 500,
    'batch_size': 1,

    'user_embed_size': 50,
    'item_embed_size': 50,
    'rnn_size': 16,
    
    'hidden_size': 64,
    'latent_size': 32,

    'number_users_to_keep': 100000000000,
    'batch_log_interval': 500,
    'train_cp_users': 200,
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
    
    train = load_obj_json(hyper_params['data_base'] + 'train_ranking_vae')
    test = load_obj_json(hyper_params['data_base'] + 'test_ranking_vae')
    user_hist = load_obj_json(hyper_params['data_base'] + 'user_hist_ranking_vae')
    item_hist = load_obj_json(hyper_params['data_base'] + 'item_hist_ranking_vae')

    file_write(hyper_params['log_file'], "Data Files loaded!")

    train_reader = DataReader(hyper_params, train, len(user_hist), item_hist, True)
    test_reader = DataReader(hyper_params, test, len(user_hist), item_hist, False, train)

    return train_reader, test_reader, len(user_hist), len(item_hist)

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

    def __init__(self, hyper_params, data, num_users, item_hist, is_training, context=None):
        self.hyper_params = hyper_params
        self.batch_size = hyper_params['batch_size']
        self.item_hist = item_hist
        self.num_users = num_users
        self.num_items = len(item_hist)
        self.data = data
        self.is_training = is_training
        self.all_users = []
        self.context = context

        self.number_users()
        self.number()

    def number(self):
        users_done = 0
        count = 0
        y_batch = []

        for user in self.data:

            if users_done > self.hyper_params['number_users_to_keep']: break
            users_done += 1

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
            
            x_batch_user2 = []
            x_batch_item2 = []
            y_batch2 = []
            
            for i in range(len(self.data[user])):
                x_batch_user2.append(int(user))
                x_batch_item2.append(self.data[user][i][0])
                y_batch2.append(float(self.data[user][i][1]))
                
            x_batch_user.append(x_batch_user2)
            x_batch_item.append(x_batch_item2)
            y_batch.append(y_batch2)

            if len(y_batch) == self.batch_size:

                yield [
                    Variable(LongTensor(scatter(x_batch_user))), 
                    Variable(LongTensor(scatter(x_batch_item)))
                ], y_batch

                x_batch_user = []
                x_batch_item = []
                y_batch = []

    def iter_eval(self):

        num_active_users = len(self.all_users)
        num_active_users = min(num_active_users, self.hyper_params['number_users_to_keep'])

        for user_now in tqdm(range(num_active_users)):
            user = self.all_users[user_now]

            all_movies = []
            x_batch_user2 = []
            x_batch_item2 = []
            y_batch2 = []
            if self.context is None: ad = []
            else: ad = self.context[user]
            
            for i in range(len(self.data[user])):
                all_movies.append(self.data[user][i][0])

            for i in range(len(ad + self.data[user])):
                #all_movies.append(self.data[user][i][0])
                x_batch_user2.append(int(user))
                x_batch_item2.append((ad + self.data[user])[i][0])
                y_batch2.append(float((ad + self.data[user])[i][1]))
                
            yield [
                Variable(LongTensor([ x_batch_user2 ])), 
                Variable(LongTensor([ x_batch_item2 ]))
            ], [ y_batch2 ], all_movies


# # Evaluation Code

# In[5]:


def map_int(a):
    if float(a.data) < 0.0: return -1
    if float(a.data) > 0.0: return 1
    return 0

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

    for x, y, all_movies in reader.iter_eval():
        user_done += 1

        _, o = model(x)

        y_pair_map = {}
        out_pair_map = {}
        y_r = {}
        
        for user in range(len(y)):
            for i in range(len(y[user])):
                for j in range(i+1, len(y[user])):
                    m1 = int(x[1][user][i].data)
                    m2 = int(x[1][user][j].data)
                    
                    y_r[str(m1)] = round(float(y[user][i]), 2)
                    y_r[str(m2)] = round(float(y[user][j]), 2)

                    if y[user][i] == y[user][j]: continue
                    
                    result_y = 1
                    if y[user][i] < y[user][j]: result_y = -1
                    
                    result_out = 1
                    if float(o[user][i].data) < float(o[user][j].data): result_out = -1
                    
                    if m1 not in y_pair_map: y_pair_map[m1] = {}
                    if m2 not in y_pair_map: y_pair_map[m2] = {}
                    if m1 not in out_pair_map: out_pair_map[m1] = {}
                    if m2 not in out_pair_map: out_pair_map[m2] = {}

                    y_pair_map[m1][m2] = result_y
                    y_pair_map[m2][m1] = -1 * result_y

                    out_pair_map[m1][m2] = result_out
                    out_pair_map[m2][m1] = -1 * result_out

        def compare_out(item1, item2):
            if item1 not in out_pair_map: return 0
            if item2 not in out_pair_map[item1]: return 0
            return out_pair_map[item1][item2]

        all_movies = sorted(all_movies, key=functools.cmp_to_key(compare_out))
        all_movies.reverse()
        
        final = []
        final_sorted = []
        for i in all_movies: 
            final.append([i, y_r[str(i)]])
            final_sorted.append([i, y_r[str(i)]])
        final_sorted = sorted(final_sorted, key=lambda xx: xx[1])
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
    
    temp_zer = Variable(FloatTensor([0.0]))

    for x, y in reader.iter():
        batch += 1
        if is_train_set == True and batch > hyper_params['train_cp_users']: break

        o1, o = model(x)
        
        loss_now = 0.0
        total2 = 0.0
        temp_correct, temp_not_correct, temp_zeros = 0.0, 0.0, 0.0
        
        for user in range(len(y)):
            for i in range(len(y[user])):
                for j in range(i+1, len(y[user])):
                    if y[user][i] == y[user][j]: continue
                    
                    y_temp = 1.0
                    if y[user][i] < y[user][j]: y_temp = -1.0
                    loss_now += float(torch.max(temp_zer, hyper_params['m_loss'] - (y_temp * (o[user][i] - o[user][j]))).data)
                    total2 += 1.0
                    
                    result = float((y_temp * (o[user][i] - o[user][j])).data)
                    
                    if result > 0.0: temp_correct += 1
                    elif result < 0.0: temp_not_correct += 1
                    else: temp_zeros += 1
        
        assert temp_correct + temp_not_correct + temp_zeros == total2
        
        if total2 > 0.0: metrics['loss'] += loss_now / total2
        
        correct += temp_correct
        not_correct += temp_not_correct
        zeros += temp_zeros
        total += total2

    assert correct + not_correct + zeros == total

    metrics['CP'] = float(correct) / float(total)
    metrics['CP'] *= 100.0
    metrics['CP'] = round(metrics['CP'], 4)

    metrics['ZEROS'] = float(zeros) / float(total)
    metrics['ZEROS'] *= 100.0
    metrics['ZEROS'] = round(metrics['ZEROS'], 4)

    metrics['loss'] = float(metrics['loss']) / float(batch)
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
        
        self.gru = nn.GRU(hyper_params['latent_size'], hyper_params['rnn_size'], batch_first=True, num_layers=1)
        
        self._enc_mu = nn.Linear(hyper_params['hidden_size'], hyper_params['latent_size'])
        self._enc_log_sigma = nn.Linear(hyper_params['hidden_size'], hyper_params['latent_size'])
        nn.init.xavier_normal(self._enc_mu.weight)
        nn.init.xavier_normal(self._enc_log_sigma.weight)
        
        self.user_embed = nn.Embedding(hyper_params['total_users'], hyper_params['user_embed_size'])
        self.item_embed = nn.Embedding(hyper_params['total_items'], hyper_params['item_embed_size'])
        nn.init.normal(self.user_embed.weight.data, mean=0, std=0.01)
        nn.init.normal(self.item_embed.weight.data, mean=0, std=0.01)
        
        self.activation = nn.ReLU()
        self.activation_last = nn.Tanh()
        if self.hyper_params['loss_type'] == 'bce': self.activation_last = nn.Sigmoid()
        
        prev = hyper_params['rnn_size']
        self.layer_hinge1 = nn.Linear(prev, 1)
#         self.layer_hinge2 = nn.Linear(64, 1)
        nn.init.xavier_normal(self.layer_hinge1.weight)
#         nn.init.xavier_normal(self.layer_hinge2.weight)
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

    def forward(self, x):
        x_shape = x[0].shape
        
        user = self.user_embed(x[0]).view(x_shape[0] * x_shape[1], -1)
        item = self.item_embed(x[1]).view(x_shape[0] * x_shape[1], -1)
        
        h_enc = self.encoder(torch.cat([user, item], dim=-1))
        z = self.sample_latent(h_enc)
        dec = self.decoder(z)
        
        z = z.view(x_shape[0], x_shape[1], -1)
        
        output, _ = self.gru(z)
        
        output = output.contiguous().view(x_shape[0] * x_shape[1], -1)
        
        output = self.layer_hinge1(output)
#         output = self.activation(output)
#         output = self.layer_hinge2(output)
        output = self.activation_last(output)
                              
        return [
            dec, self.z_mean, self.z_sigma
        ], output.squeeze(-1).view(x_shape[0], x_shape[1])


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
        self.zer = Variable(FloatTensor([0.0]))
        self.exp = Variable(FloatTensor([np.e]))
        self.hundred = Variable(FloatTensor([100.0]))
        self.cce = nn.CrossEntropyLoss(size_average=True)
        self.bce = nn.BCELoss(size_average=True)

    def forward(self, o1, o, y, t):
        
        m, zm, zs = o1
        
        mean_sq = zm * zm
        stddev_sq = zs * zs
        kld = torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
        
#         print(m.shape, t.shape)
        likelihood = self.cce(m, t) # Try using a weight parameters, alpha when summing
        
        total = 0
        for user in range(len(y)):
            for i in range(len(y[user])):
                for j in range(i+1, len(y[user])):
                    if y[user][i] == y[user][j]: continue
                    total += 1
        
        if total == 0: total = 1
        pairwise = Variable(FloatTensor([0.0] * total))
        total = 0
        for user in range(len(y)):
            for i in range(len(y[user])):
                for j in range(i+1, len(y[user])):
                    if y[user][i] == y[user][j]: continue
                    
                    y_temp = 1.0
                    if y[user][i] < y[user][j]: y_temp = -1.0
#                     if pairwise == None: pairwise = torch.max(self.zer, self.m_loss - (y_temp * (o[user][i] - o[user][j])))
                    pairwise[total] = torch.max(self.zer, self.m_loss - (y_temp * (o[user][i] - o[user][j])))
                    total += 1
        
        final = (0.05 * kld) + (0.2 * likelihood) + (0.75 * torch.mean(pairwise))
        
        return final


# # Training loop

# In[ ]:


def train(reader):
    model.train()
    total_loss = 0
    start_time = time.time()
    batch = 0
    batch_limit = int(train_reader.num_b)

    for x, y in reader.iter():
        batch += 1
        
        model.zero_grad()
        optimizer.zero_grad()

        temp_o1, temp_o2 = model(x)
        
        loss = criterion(temp_o1, temp_o2, y, x[1].view(-1))
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

