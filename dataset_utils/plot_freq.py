import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from PIL import Image

base_folder_path = sorted(glob.glob("../../data/HST_video/Jun/Basement/20*"))
print("Basement: %s files: " % str(len(base_folder_path)))
lower_folder_path = sorted(glob.glob("../../data/HST_video/Jun/Lower_Floor/20*"))
print("Lower level: %s files: " % str(len(lower_folder_path)))
floor_1_folder_path = sorted(glob.glob("../../data/HST_video/Jun/Floor_2/20*"))
print("Floor 1: %s files: " % str(len(floor_1_folder_path)))
floor_2_folder_path = sorted(glob.glob("../../data/HST_video/Jun/Floor_3/20*"))
print("Floor 2: %s files: " % str(len(floor_2_folder_path)))

time_spread = np.zeros((4,24))
hand_stat = np.zeros((4,4))



dates = pd.date_range(start = "12/05/2023", end = "03/31/2024")
data = np.zeros(dates.size)

df = pd.DataFrame({'Date': dates, 'Freq': data})

for folder in base_folder_path:
    time = int(folder.split('/')[-1].split('_')[1][:2])
    time_spread[0, time] += 1
    date = folder.split('/')[-1].split('_')[0]
    date = date[:4] + "-" + date[4:]
    date = date[:-2] + "-" + date[-2:]
    df.loc[df['Date'] == date, 'Freq'] += 1
    proj_folder_path = sorted(glob.glob(folder+"/*20*"))
    img = glob.glob(proj_folder_path[0]+"/*")[0]
    width, height = Image.open(img).size
    hand_str = proj_folder_path[0].split("/")[-1].split("_")[0]
    if height/width > 1 and hand_str == "HAND":
        hand_stat[3, 0] += 1
    elif height/width < 1 and hand_str == "HAND":
        hand_stat[3, 1] += 1
    elif height/width > 1 and hand_str == "DJI":
        hand_stat[3, 2] += 1
    elif height/width < 1 and hand_str == "DJI":
        hand_stat[3, 3] += 1
    else:
        print("Error happens...")

for folder in lower_folder_path:
    time = int(folder.split('/')[-1].split('_')[1][:2])
    time_spread[1, time] += 1
    date = folder.split('/')[-1].split('_')[0]
    date = date[:4] + "-" + date[4:]
    date = date[:-2] + "-" + date[-2:]
    df.loc[df['Date'] == date, 'Freq'] += 1
    proj_folder_path = sorted(glob.glob(folder+"/*20*"))
    img = glob.glob(proj_folder_path[0]+"/*")[0]
    width, height = Image.open(img).size
    hand_str = proj_folder_path[0].split("/")[-1].split("_")[0]
    if height/width > 1 and hand_str == "HAND":
        hand_stat[2, 0] += 1
    elif height/width < 1 and hand_str == "HAND":
        hand_stat[2, 1] += 1
    elif height/width > 1 and hand_str == "DJI":
        hand_stat[2, 2] += 1
    elif height/width < 1 and hand_str == "DJI":
        hand_stat[2, 3] += 1
    else:
        print("Error happens...")

for folder in floor_1_folder_path:
    time = int(folder.split('/')[-1].split('_')[1][:2])
    time_spread[2, time] += 1
    date = folder.split('/')[-1].split('_')[0]
    date = date[:4] + "-" + date[4:]
    date = date[:-2] + "-" + date[-2:]
    df.loc[df['Date'] == date, 'Freq'] += 1
    proj_folder_path = sorted(glob.glob(folder+"/*20*"))
    img = glob.glob(proj_folder_path[0]+"/*")[0]
    width, height = Image.open(img).size
    hand_str = proj_folder_path[0].split("/")[-1].split("_")[0]
    if height/width > 1 and hand_str == "HAND":
        hand_stat[1, 0] += 1
    elif height/width < 1 and hand_str == "HAND":
        hand_stat[1, 1] += 1
    elif height/width > 1 and hand_str == "DJI":
        hand_stat[1, 2] += 1
    elif height/width < 1 and hand_str == "DJI":
        hand_stat[1, 3] += 1
    else:
        print("Error happens...")
for folder in floor_2_folder_path:
    time = int(folder.split('/')[-1].split('_')[1][:2])
    time_spread[3, time] += 1
    date = folder.split('/')[-1].split('_')[0]
    date = date[:4] + "-" + date[4:]
    date = date[:-2] + "-" + date[-2:]
    df.loc[df['Date'] == date, 'Freq'] += 1
    proj_folder_path = sorted(glob.glob(folder+"/*20*"))
    img = glob.glob(proj_folder_path[0]+"/*")[0]
    width, height = Image.open(img).size
    hand_str = proj_folder_path[0].split("/")[-1].split("_")[0]
    if height/width > 1 and hand_str == "HAND":
        hand_stat[0, 0] += 1
    elif height/width < 1 and hand_str == "HAND":
        hand_stat[0, 1] += 1
    elif height/width > 1 and hand_str == "DJI":
        hand_stat[0, 2] += 1
    elif height/width < 1 and hand_str == "DJI":
        hand_stat[0, 3] += 1
    else:
        print("Error happens...")

print(np.max(time_spread))
# time_spread = time_spread/np.max(time_spread)
# Plotting
plt.figure(figsize=(12, 6))
plt.imshow(time_spread, cmap='YlGn', aspect='auto')
cbar = plt.colorbar()
cbar.set_label(label='Contributions (#Video)', size=14)
cbar.ax.tick_params(labelsize=12)
plt.yticks(ticks=range(4), labels=['Basement', 'Lower Level', 'Level 1', 'Level 2'])
plt.ylabel('Floor', fontsize=14)
plt.xticks(ticks=range(24), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
plt.xlabel('Time of the Day', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Contribution Graph', fontsize=16)
plt.show()


# Define the start and end date for the data range
start_date = pd.to_datetime("2023-12-05")
end_date = pd.to_datetime("2024-03-31")


# Calculate the difference in days from the start date and derive the week index
df['DayDelta'] = (df['Date'] - start_date).dt.days
df['WeekIndex'] = df['DayDelta'] // 7
df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6

# Initialize a matrix for plotting
num_weeks = df['WeekIndex'].max() + 1
date_spread = np.zeros((num_weeks, 7))  # 7 days of the week

# Fill the matrix with actual contributions
for index, row in df.iterrows():
    date_spread[row['WeekIndex'], row['DayOfWeek']] = row['Freq']

# Plotting
plt.figure(figsize=(12, 6))
plt.imshow(date_spread.T, cmap='YlGn', aspect='auto', interpolation='nearest') # YlGn # Greens
cbar = plt.colorbar()
cbar.set_label(label='Contributions (#Video)', size=14)
cbar.ax.tick_params(labelsize=12)
plt.yticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.ylabel('Days of the Week', fontsize=14)
plt.xticks(ticks=range(num_weeks), labels=[f'W{w+1}' for w in range(num_weeks)])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Week Index from December 5, 2023, to March 31, 2024', fontsize=14)
plt.title('Contribution Graph', fontsize=16)
plt.tight_layout()
plt.show()


sections = ['Portrait by HAND', 'Landscape by HAND', 'Portrait by DJI Stabilizer', 'Landscape by DJI Stabilizer']

titles = ['Floor 2', 'Floor 1', 'Lower Level', 'Basement']  # Titles for each pie chart

# Colors: Use the YlGn colormap
colormap = plt.get_cmap('YlGn')
colors = colormap(np.linspace(0.3, 0.9, len(sections)))

# Setting up the figure for 4 subplots (2x2 grid)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()  # Flatten the array of axes for easy iteration

# Creating each pie chart
for i, floor in enumerate(hand_stat):
    wedges, texts, autotexts = axs[i].pie(floor, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 18})
    axs[i].set_title(titles[i], fontsize=20)


# fig.legend(wedges, sections, loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, fontsize=14, title='Sections')
legend = fig.legend(wedges, sections, loc="lower right", bbox_to_anchor=(1, 0), bbox_transform=plt.gcf().transFigure, fontsize=12, title='Capture Styles')
legend.get_title().set_fontsize(16)  # Setting a different fontsize for the title
# Adjust the layout to prevent overlap
plt.tight_layout()

plt.show()



