import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate some random data
error_lower_path = '/home/juy220/PycharmProjects/PoseNet-Pytorch/test_Resnet34_Lower_Floor/2024-04-04-14:24:52/error.csv'
error_base_path = '/home/juy220/PycharmProjects/PoseNet-Pytorch/test_Resnet34_Basement/2024-04-10-23:13:11/error.csv'

lower_data = pd.read_csv(error_lower_path, header = None, names = ['pos', 'ori'])
base_data = pd.read_csv(error_base_path, header = None, names = ['pos', 'ori'])

# Sort the data
lower_pos_sorted = np.sort(lower_data['pos']) / 30.757
lower_ori_sorted = np.sort(lower_data['ori']) * 180 / np.pi

base_pos_sorted = np.sort(base_data['pos']) / 29.355
base_ori_sorted = np.sort(base_data['ori']) * 180 / np.pi

# Calculate the CDF values
lower_cdf = np.arange(1, len(lower_pos_sorted) + 1) / len(lower_pos_sorted)
base_cdf = np.arange(1, len(base_pos_sorted) + 1) / len(base_pos_sorted)

# Plotting the CDF

plt.figure(figsize=(8, 6))
plt.plot(lower_pos_sorted, lower_cdf, marker='.', markersize = 2, color='mediumseagreen', linestyle='none', label='Lower Level')
plt.plot(base_pos_sorted, base_cdf, marker='.', markersize = 2, color='darkorange', linestyle='none', label='Basement')  # New CDF
plt.xlim(0, 4)
plt.ylim(0, 1)
plt.title('CDF of Position Error', fontsize=18)
plt.xlabel('Position Error (m)', fontsize=14)
plt.ylabel('CDF', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)  # Add a legend to distinguish the datasets
plt.show()



plt.figure(figsize=(8, 6))
plt.plot(lower_ori_sorted, lower_cdf, marker='.', markersize = 2, color='mediumseagreen', linestyle='none', label='Lower Level')
plt.plot(base_ori_sorted, base_cdf, marker='.', markersize = 2, color='darkorange', linestyle='none', label='Basement')
plt.xlim(0, 30)
plt.ylim(0, 1)
plt.title('CDF of Orientation Error', fontsize = 18)
plt.xlabel('Orientation Error (\u00B0)', fontsize = 14)
plt.ylabel('CDF', fontsize = 14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()