# I copied this file from "saved_data/netflix-good-sample/" to "code/"
# Not sure if this is the verison we executed

import random
import math

import bz2

#f = bz2.BZ2File('netflix_all.csv.bz2')
f = open("ratings_full.csv")
lines = f.readlines()

out = open("ratings.csv", 'w')


user_hist = {}
# sampled_users = []

for line in lines[1:]:
    line = line.strip().split(',')
    if int(line[-1]) <= 3: continue
        
    if line[0] not in user_hist: user_hist[line[0]] = []
    user_hist[line[0]].append([ line[1], line[2], line[3] ])
    
sess_len = {}
for u in user_hist:
    if len(user_hist[u]) not in sess_len: sess_len[len(user_hist[u])] = []
    sess_len[len(user_hist[u])].append(u)
    
keep_prob = 0.1
save_str = "User,Movie,Timestamp,Rating\n"

out.write(save_str)


temp = list(sess_len.keys())
temp.sort()

len_list = len(temp)

step = (1-keep_prob)/float(len_list)

for l in temp:
    all_users = sess_len[l]
    random.shuffle(all_users)


    sample_size = int(math.ceil(keep_prob * float(len(all_users))))

    if sample_size == 0:
        sampled_users = [all_users[0]]
    else:
        sampled_users = all_users[:sample_size]
        #sampled_users += all_users[:int(keep_prob * float(len(all_users)))]
    for u in sampled_users:
        for m in user_hist[u]:
            save_str = str(u) + ',' + str(m[0]) + ',' + str(m[1]) + ',' + str(m[2]) + '\n'
            out.write(save_str)
    out.flush()

    keep_prob = keep_prob + step

# Sampling done
# for u in sampled_users:
#     for m in user_hist[u]:
#         save_str += str(u) + ',' + str(m) + ',1\n'

out.close()