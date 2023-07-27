import re
import os.path as osp
from glob import glob
import os

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

            for i in range(len(images)-1):
                if i==0:
                    image_list.append(images[0])
                else:
                    image_list.append(images[i-1])
                
                image_list.append(images[i])
                image_list.append(images[i+1])

                flow_list.append(flows[i])
                extra_info_list.append(scene)
                extra_info_list.append(str(i))
        
        for idx in range(len(image_list)):
            image_list[idx] = image_list[idx].replace("/mnt/lustre/share/cp/caodongliang/MPI-Sintel", "Sintel") + "\n"
        for idx in range(len(flow_list)):
            flow_list[idx] = flow_list[idx].replace("/mnt/lustre/share/cp/caodongliang/MPI-Sintel", "Sintel") + "\n"
        for idx in range(len(extra_info_list)):
            extra_info_list[idx] = extra_info_list[idx] + "\n"

        with open(osp.join("sintel_three_frames", "Sintel_"+dstype+"_png.txt"), 'w') as f:
            f.writelines(image_list)
            print(len(image_list))
        with open(osp.join("sintel_three_frames", "Sintel_"+dstype+"_flo.txt"), 'w') as f:
            f.writelines(flow_list)
            print(len(flow_list))
        with open(osp.join("sintel_three_frames", "Sintel_"+dstype+"_extra_info.txt"), 'w') as f:
            f.writelines(extra_info_list)
            print(len(extra_info_list))