import os
import random
import json

def save_obj_json(obj, name):
    with open(name + '.json', 'w') as f:
        json.dump(obj, f)

hyper_params = {
    'data_base': 'saved_data/',
    'project_name': 'ranking_vae',
    'data_split': [0.8, 0.2],
    'max_user_hist': 500,
    'min_user_hist': 5,
}

# Calculating min and max time for scaling later
f = open(hyper_params['data_base'] + 'ratings.dat')
lines = f.readlines()

max_time, min_time = 0, 1000000000000000000

for line in lines:
	temp = list(map(int, line.strip().split("::")))

	max_time = max(max_time, int(temp[3]))
	min_time = min(min_time, int(temp[3]))

# Calculating user and item histories
user_hist_old = {}
item_hist_old = {}

for line in lines:
	temp = list(map(int, line.strip().split("::")))

	if temp[0] not in user_hist_old: user_hist_old[temp[0]] = []
	if temp[1] not in item_hist_old: item_hist_old[temp[1]] = []
	
	user_hist_old[temp[0]].append([temp[1], float(temp[2]) / 5.0, float(temp[3] - min_time) / float(max_time - min_time)])
	item_hist_old[temp[1]].append([temp[0], float(temp[2]) / 5.0, float(temp[3] - min_time) / float(max_time - min_time)])

# Sorting user and item histories
for user in user_hist_old:
	user_hist_old[user].sort(key=lambda x: x[2])

for item in item_hist_old:
	item_hist_old[item].sort(key=lambda x: x[2])

# Number items according to train
item_map = {}
user_map = {}
count_now = 0 # Start from 0 because no padding
count_user = 0 # Start from 0 because no padding

for user in user_hist_old:

	if len(user_hist_old[user]) > hyper_params['max_user_hist']: continue
	if len(user_hist_old[user]) < hyper_params['min_user_hist']: continue

	if user not in user_map:
		user_map[user] = count_user
		count_user += 1

	split = int(hyper_params['data_split'][0] * len(user_hist_old[user]))

	for item in user_hist_old[user][:split]: # Train split NOT CONSIDERED
		if item[0] not in item_map:
			item_map[item[0]] = count_now
			count_now += 1

# Numbering histories
user_hist = {}
item_hist = {}

for item in item_hist_old:
	if item in item_map:
		item_hist[item_map[item]] = []
		for user in item_hist_old[item]:
			if user[0] in user_map:
				item_hist[item_map[item]].append([ user_map[user[0]], user[1], user[2] ])

for user in user_hist_old:
	if user in user_map:
		user_hist[user_map[user]] = []
		for item in user_hist_old[user]:
			if item[0] in item_map:
				user_hist[user_map[user]].append([ item_map[item[0]], item[1], item[2] ])

assert len(user_hist) == count_user
assert len(item_hist) == count_now

# Calculating all items for each user
user_hist_items = {}
for user in user_hist:

	if len(user_hist[user]) > hyper_params['max_user_hist']: continue
	if len(user_hist[user]) < hyper_params['min_user_hist']: continue
	
	user_hist_items[user] = set()

	for item in user_hist[user]:
		if item[0] not in user_hist_items[user]:
			user_hist_items[user].add(item[0])

# Calculating the train and test splits
train = {}
test = {}

users_kept = 0
all_items = list(item_hist.keys())

for user in user_hist:

	if len(user_hist[user]) > hyper_params['max_user_hist']: continue
	if len(user_hist[user]) < hyper_params['min_user_hist']: continue
	
	users_kept += 1

	split = int(hyper_params['data_split'][0] * len(user_hist[user]))

	train_split = user_hist[user][:split]
	test_split = user_hist[user][split:]

	train[user] = train_split
	test[user] = test_split

print("users_kept = " + str(users_kept))
assert users_kept == len(user_hist)

# Saving all objects
save_obj_json(train, hyper_params['data_base'] + "train_" + hyper_params['project_name'])
save_obj_json(test, hyper_params['data_base'] + "test_" + hyper_params['project_name'])
save_obj_json(item_hist, hyper_params['data_base'] + "item_hist_" + hyper_params['project_name'])
save_obj_json(user_hist, hyper_params['data_base'] + "user_hist_" + hyper_params['project_name'])
