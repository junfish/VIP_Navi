import glob

dir = "../../data/HST_video/Eashan/Floor_3/"
base_folder_path = sorted(glob.glob(dir + "20*"))
file_path = dir + "proj_name.txt"
with open(file_path, 'w') as file:
    for proj in base_folder_path:
        file.write(proj.split("/")[-1]+"\n")