import re
import os.path as osp
from glob import glob
import os

import pickle

root = "/mnt/lustre/share/cp/caodongliang/MPI-Sintel/"

for split in ['training']:
    for dstype in ['clean', 'final']:
        image_list = []
        flow_list = []
        extra_info_list = []
    
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        for scene in os.listdir(image_root):
            images = sorted(glob(osp.join(image_root, scene, '*.png')))
            flows = sorted(glob(osp.join(flow_root, scene, '*.flo')))

            for idx in range(len(images)):
                images[idx] = images[idx].replace("/mnt/lustre/share/cp/caodongliang/MPI-Sintel", "Sintel") + "\n"
            for idx in range(len(flows)):
                flows[idx] = flows[idx].replace("/mnt/lustre/share/cp/caodongliang/MPI-Sintel", "Sintel") + "\n"

            image_list.append(images)
            flow_list.append(flows)
            extra_info_list.append(scene)
        
        with open("sintel_training_"+dstype+"_png.pkl", 'wb') as f:
            pickle.dump(image_list, f)
        with open("sintel_training_"+dstype+"_flo.pkl", 'wb') as f:
            pickle.dump(flow_list, f)
        with open("sintel_training_scene.pkl", 'wb') as f:
            pickle.dump(extra_info_list, f)
        

