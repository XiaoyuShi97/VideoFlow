import os
import math
import random
from glob import glob
import os.path as osp

split = "testing"
root = "KITTI"


root = osp.join(root, split)
images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

extra_info = []
flow_list = []
image_list = []

for img1, img2 in zip(images1, images2):
    frame_id = img1.split('/')[-1]
    extra_info += [ frame_id+"\n" ]
    image_list += [ img1+"\n", img2+"\n" ]

if split == 'training':
    _flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
    flow_list = [s+"\n" for s in _flow_list]

print(len(image_list), len(flow_list), len(extra_info))

with open('KITTI_{}_image.txt'.format(split), 'w') as f:
	f.writelines(image_list)

with open('KITTI_{}_flow.txt'.format(split), 'w') as f:
	f.writelines(flow_list)

with open('KITTI_{}_extra_info.txt'.format(split), 'w') as f:
	f.writelines(extra_info)
