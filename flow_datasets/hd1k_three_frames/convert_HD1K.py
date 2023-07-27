import re
import os.path as osp
from glob import glob
import os

root = "/mnt/lustre/share/cp/caodongliang/HD1K/"

image_list = []
flow_list = []

seq_ix = 0

while 1:
    flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
    images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

    if len(flows) == 0:
        break

    print(seq_ix, len(flows), images[0], images[-1], "!!!!!!!!!!!!!!")

    for i in range(len(images)-1):
        if i==0:
            image_list.append(images[0])
        else:
            image_list.append(images[i-1])
        
        image_list.append(images[i])
        image_list.append(images[i+1])

        flow_list.append(flows[i])
    
    seq_ix += 1

for idx in range(len(image_list)):
    image_list[idx] = image_list[idx].replace("/mnt/lustre/share/cp/caodongliang/HD1K", "HD1K") + "\n"
for idx in range(len(flow_list)):
    flow_list[idx] = flow_list[idx].replace("/mnt/lustre/share/cp/caodongliang/HD1K", "HD1K") + "\n"

with open(osp.join("hd1k_three_frames", "hd1k"+"_image.txt"), 'w') as f:
    f.writelines(image_list)
    print(len(image_list))
with open(osp.join("hd1k_three_frames", "hd1k"+"_flo.txt"), 'w') as f:
    f.writelines(flow_list)
    print(len(flow_list))
