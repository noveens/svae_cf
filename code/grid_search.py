import json
import os
import sys

def save_obj_json(obj, name):
    with open(name + '.json', 'w') as f:
        json.dump(obj, f)

if len(sys.argv) == 1:
    print("Usage: PS1$ python grid_search.py <RANGE>\n\nWhere RANGE specifies the number of configurations to run in this session.\nNote: The RANGE follows 0-based indexing, and is inclusive.\n\nEg. PS1$ python grid_search.py 0-2")
    exit(0)
        
to_search = {
    # 'project_name': ['ml-1m', 'netflix-full'], 
    'project_name': ['netflix-full'], 
    # 'weight_decay': [float(1e-2), float(5e-3), float(1e-3)], 
    'network': [ [128, 100, 75, 32], [256, 200, 150, 64], [512, 400, 300, 128] ],
    # 'loss_type': ['next_k', 'postfix'], 
    'loss_type': ['postfix'], 
    # 'next_k': [4, 8, 16, 32, 64]
}

all_settings = []

def run_with_setting(setting):
    # Correct the model file name and the log file name and the data root, NOT DONE
    hyper = {}
    for key in setting:
        if key != "network": hyper[key] = setting[key]
        else:
            hyper['item_embed_size'] = setting[key][0]
            hyper['rnn_size'] = setting[key][1]
            hyper['hidden_size'] = setting[key][2]
            hyper['latent_size'] = setting[key][3]
    
    hyper['data_base'] = '../saved_data/' + hyper['project_name'] + '/pro_sg/'
    hyper['project_name'] = 'svae_' + hyper['project_name']
    all_settings.append(hyper)

def grid_search(order, at, setting):
    if at == len(order):
        run_with_setting(setting)
        return
    
    for possible in to_search[order[at]]:
        setting[order[at]] = possible
        #if order[at] == "loss_type" and possible == "postfix":
            #run_with_setting(setting)
            #continue
        grid_search(order, at + 1, setting)

all_keys_order = list(to_search.keys())
grid_search(all_keys_order, 0, {})

print("Total settings possible:", len(all_settings))
print("Running", sys.argv[1].split("-")[0], "to", sys.argv[1].split("-")[1], "settings (0-based, inclusive)")

for index in range(int(sys.argv[1].split("-")[0]), min(int(sys.argv[1].split("-")[1]) + 1, len(all_settings))):
    hyper = all_settings[index]
    save_obj_json(hyper, "temp")
    os.system("python main_svae.py temp.json")
