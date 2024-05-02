import pickle
import numpy as np
import os
metadata_path = '/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement/test.pkl'

with open(metadata_path, 'rb') as file:
    loaded_data = pickle.load(file)
    train_data = loaded_data['train']
    print(loaded_data)

data_list = []
data = {}
for i in range(20):
    data_list.append({
        'image_path': 'xxxxxxxxxxxxxxxxxxxx',
        'w_t_c': np.array([1, 2, 3]),
        'c_q_w': 43 + i,
    })
data['train'] = data_list
with open(os.path.join(metadata_path), 'wb') as f:
    pickle.dump(data, f)

