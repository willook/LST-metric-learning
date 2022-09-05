import os
import numpy as np

file_list = sorted(os.listdir("../../nturgbd_raw/nturgb+d_skeletons120/"))
name_list = []

missings = np.loadtxt('ntu_rgb120_missings.txt', dtype=str)
skes_available_name_file = "skes_available_name.txt"
setup_file = "setup.txt"
camera_file = "camera.txt"
performer_file = "performer.txt"
replication_file = "replication.txt"
label_file = "label.txt"

skes_available_name_list = []
setup_list = []
camera_list = []
performer_list = []
replication_list = []
label_list = []

for filename in file_list:
    name = (filename.split('.')[0])
    if name in missings:
        continue
    skes_available_name_list.append(name)
    setup_list.append(str(int(name[1:4]))) # setup
    camera_list.append(str(int(name[5:8]))) # camera
    performer_list.append(str(int(name[9:12]))) # performer
    replication_list.append(str(int(name[13:16]))) # replication
    label_list.append(str(int(name[17:20]))) # label
    
with open(skes_available_name_file, 'w') as f:
    f.write("\n".join(skes_available_name_list))

with open(setup_file, 'w') as f:
    f.write("\n".join(setup_list))

with open(camera_file, 'w') as f:
    f.write("\n".join(camera_list))

with open(performer_file, 'w') as f:
    f.write("\n".join(performer_list))

with open(replication_file, 'w') as f:
    f.write("\n".join(replication_list))

with open(label_file, 'w') as f:
    f.write("\n".join(label_list))