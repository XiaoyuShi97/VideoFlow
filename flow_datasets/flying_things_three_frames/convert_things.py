import re
import os.path as osp
from glob import glob

root = "/mnt/lustre/share/cp/caodongliang/FlyingThings3D/"

for dstype in ['frames_cleanpass', 'frames_finalpass']:
    image_list = []
    flow_list = []
    for cam in ['left']:
        image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
        image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

    flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
    flow_future_dirs = sorted([osp.join(f, 'into_future', cam) for f in flow_dirs])
    flow_past_dirs = sorted([osp.join(f, 'into_past', cam) for f in flow_dirs])
    
    for idir, fdir, pdir in zip(image_dirs, flow_future_dirs, flow_past_dirs):
        images = sorted(glob(osp.join(idir, '*.png')) )
        future_flows = sorted(glob(osp.join(fdir, '*.pfm')) )
        past_flows = sorted(glob(osp.join(pdir, '*.pfm')) )
        
        for i in range(1, len(images)-1):
            image_list.append(images[i-1])
            image_list.append(images[i])
            image_list.append(images[i+1])
            
            flow_list.append(future_flows[i])
            flow_list.append(past_flows[i])

    for idx in range(len(image_list)):
        image_list[idx] = image_list[idx].replace("/mnt/lustre/share/cp/caodongliang/FlyingThings3D", "flow_data") + "\n"
    for idx in range(len(flow_list)):
        flow_list[idx] = flow_list[idx].replace("/mnt/lustre/share/cp/caodongliang/FlyingThings3D", "flow_data") + "\n"
    

    with open(osp.join("flying_things_three_frames", "flyingthings_"+dstype+"_png.txt"), 'w') as f:
        f.writelines(image_list)
        print(len(image_list))
    with open(osp.join("flying_things_three_frames", "flyingthings_"+dstype+"_pfm.txt"), 'w') as f:
        f.writelines(flow_list)
        print(len(flow_list))
    
        
        
