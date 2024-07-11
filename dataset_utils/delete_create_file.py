import glob
import os

###########################
###### Change Floors ######
floor_name = "Level_1"  ###
###########################
###########################

folder_path = sorted(glob.glob("/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/test/" + floor_name + "/20*"))
for folder in folder_path:
    if os.path.exists(folder + "/sparse/0/project.ini"):
        os.remove(folder + "/sparse/0/project.ini")
    if not os.path.exists(folder + "/sparse/geo"):
        os.mkdir(folder + "/sparse/geo")
    else:
        print("Directory already exists!")
    if os.path.exists(folder + "/sparse/geo/project.ini"):
        os.remove(folder + "/sparse/geo/project.ini")
    try:
        with open(folder + "/geo_coord.txt", 'x') as file:
            file.write("")
    except FileExistsError:
        print(f"The file geo_coord.txt already exists.")


