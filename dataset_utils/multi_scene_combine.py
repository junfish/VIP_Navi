import os

basement_metadata_path = "/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement/image_train_all.txt"
lower_level_metadata_path = "/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level/image_train_all.txt"
level_1_metadata_path = "/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1/image_train_all.txt"
level_2_metadata_path = "/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2/image_train_all.txt"

all_scene_path = [basement_metadata_path, lower_level_metadata_path, level_1_metadata_path, level_2_metadata_path]

f = open(os.path.join("/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun", "multiscene_train_all.txt"), 'w')
f.write("Lehigh Health Science and Technology (HST) Building (https://www2.lehigh.edu/news/new-health-science-and-technology-building-a-hub-for-interdisciplinary-research).\n")
# f.write("The Lower-Level Floor features a two-tiered structure.\n")
f.write("SCENE_ID, IMG_PATH, QW, QX, QY, QZ, TX, TY, TZ\n")

for scene_path in all_scene_path:
    raw_lines = open(scene_path, 'r').readlines()
    data_lines = raw_lines[2:]
    for i, line in enumerate(data_lines):
        splits = line.split()
        poses = list(map(lambda x: float(x.strip().replace(",", "")), splits[2:]))
        filename = os.path.join(scene_path.split('/')[-2], splits[0].strip().replace(",", ""))
        IMG_ID, QW, QX, QY, QZ, TX, TY, TZ, NAME = line.split(',')
        img_id = int(IMG_ID)
        qw, qx, qy, qz, tx, ty, tz = float(QW), float(QX), float(QY), float(QZ), float(TX), float(TY), float(TZ)
        f.write('%-40s, %4.0f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f\n' % (
        NAME.strip(), img_id, qw, qx, qy, qz, tx, ty, tz))