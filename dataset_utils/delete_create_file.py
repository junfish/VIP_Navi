import glob
import os
# ../../data/HST_video/Jun/test/Basement/20*
# ../../data/HST_video/Jun/Floor_3/20*
folder_path = sorted(glob.glob("../../data/HST_video/Jun/Floor_2/20*"))
for folder in folder_path:
    if os.path.exists(folder + "/sparse/0/project.ini"):
        os.remove(folder + "/sparse/0/project.ini")
    if not os.path.exists(folder + "/sparse/geo"):
        os.mkdir(folder + "/sparse/geo")
    else:
        print("Directory already exists!")
    try:
        with open(folder + "/geo_coord.txt", 'x') as file:
            file.write("")
    except FileExistsError:
        print(f"The file geo_coord.txt already exists.")


