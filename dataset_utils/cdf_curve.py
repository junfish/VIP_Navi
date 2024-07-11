import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Generate some random data

purple_rgb = (233/255, 133/255, 255/255) # lower
green_rgb = (177/255, 228/255, 149/255) # basement
orange_rgb = (235/255, 173/255, 133/255) # 2
blue_rgb = (66/255, 155/255, 215/255) # 1

error_level_2_path = '/home/juy220/PycharmProjects/VIP_Navi/test_Resnet34_Level_2/2024-05-10-13:45:25/error.csv'
error_level_1_path = '/home/juy220/PycharmProjects/VIP_Navi/test_Resnet34_Level_1/2024-05-10-13:42:05/error.csv'
error_lower_path = '/home/juy220/PycharmProjects/VIP_Navi/test_Resnet34_Lower_Level/2024-05-10-01:03:58/error.csv'
error_base_path = '/home/juy220/PycharmProjects/VIP_Navi/test_Resnet34_Basement/2024-05-10-00:57:23/error.csv'

level_2_data = pd.read_csv(error_level_2_path, header = None, names = ['pos', 'ori'])
level_1_data = pd.read_csv(error_level_1_path, header = None, names = ['pos', 'ori'])
lower_data = pd.read_csv(error_lower_path, header = None, names = ['pos', 'ori'])
base_data = pd.read_csv(error_base_path, header = None, names = ['pos', 'ori'])

# Sort the data
level_2_pos_sorted = np.sort(level_2_data['pos']) / 29.32
level_2_ori_sorted = np.sort(level_2_data['ori']) * 180 / np.pi

level_1_pos_sorted = np.sort(level_1_data['pos']) / 29.48
level_1_ori_sorted = np.sort(level_1_data['ori']) * 180 / np.pi

lower_pos_sorted = np.sort(lower_data['pos']) / 30.75
lower_ori_sorted = np.sort(lower_data['ori']) * 180 / np.pi

base_pos_sorted = np.sort(base_data['pos']) / 29.355
base_ori_sorted = np.sort(base_data['ori']) * 180 / np.pi

# Calculate the CDF values
level_2_cdf = np.arange(1, len(level_2_pos_sorted) + 1) / len(level_2_pos_sorted)
level_1_cdf = np.arange(1, len(level_1_pos_sorted) + 1) / len(level_1_pos_sorted)
lower_cdf = np.arange(1, len(lower_pos_sorted) + 1) / len(lower_pos_sorted)
base_cdf = np.arange(1, len(base_pos_sorted) + 1) / len(base_pos_sorted)


# Plotting the CDF
plt.figure(figsize=(12, 6))
plt.plot(level_2_pos_sorted, level_2_cdf, marker='.', markersize = 3, color=orange_rgb, linestyle='none', label='Level 2')
plt.plot(level_1_pos_sorted, level_1_cdf, marker='.', markersize = 3, color=blue_rgb, linestyle='none', label='Level 1')
plt.plot(lower_pos_sorted, lower_cdf, marker='.', markersize = 3, color=purple_rgb, linestyle='none', label='Lower Level')
plt.plot(base_pos_sorted, base_cdf, marker='.', markersize = 3, color=green_rgb, linestyle='none', label='Basement')  # New CDF
plt.xlim(0, 3)
plt.ylim(0, 1)


# plt.title('CDF of Position Error', fontsize=24)
plt.xlabel('Position Error (m)', fontsize=48)
plt.ylabel('CDF', fontsize=48)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)
plt.grid(True)
plt.legend(fontsize=36)  # Add a legend to distinguish the datasets
plt.legend(scatterpoints=12, loc='lower right', prop={'size': 24}, markerscale=10)
plt.savefig('pos.png', dpi=600, bbox_inches='tight')
plt.show()



plt.figure(figsize=(12, 6))
plt.plot(level_2_ori_sorted, level_2_cdf, marker='.', markersize = 3, color=orange_rgb, linestyle='none', label='Level 2')
plt.plot(level_1_ori_sorted, level_1_cdf, marker='.', markersize = 3, color=blue_rgb, linestyle='none', label='Level 1')
plt.plot(lower_ori_sorted, lower_cdf, marker='.', markersize = 3, color=purple_rgb, linestyle='none', label='Lower Level')
plt.plot(base_ori_sorted, base_cdf, marker='.', markersize = 3, color=green_rgb, linestyle='none', label='Basement')
plt.xlim(0, 30)
plt.ylim(0, 1)

# plt.title('CDF of Orientation Error', fontsize = 24)
plt.xlabel('Orientation Error (\u00B0)', fontsize = 48)
plt.ylabel('CDF', fontsize = 48)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)
plt.grid(True)
plt.legend(fontsize=36)
plt.legend(scatterpoints=12, loc='lower right', prop={'size': 24}, markerscale=10)
plt.savefig('ori.png', dpi=600, bbox_inches='tight')
plt.show()