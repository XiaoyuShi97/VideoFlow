import re
import os.path as osp
from glob import glob
import pickle

root = "/mnt/lustre/share/cp/caodongliang/FlyingThings3D/"

for dstype in ['frames_cleanpass', 'frames_finalpass']:
    image_list = []
    fflow_list = []
    pflow_list = []


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
        
        for idx in range(len(images)):
            images[idx] = images[idx].replace("/mnt/lustre/share/cp/caodongliang/FlyingThings3D", "flow_data") + "\n"
        for idx in range(len(future_flows)):
            future_flows[idx] = future_flows[idx].replace("/mnt/lustre/share/cp/caodongliang/FlyingThings3D", "flow_data") + "\n"
        for idx in range(len(past_flows)):
            past_flows[idx] = past_flows[idx].replace("/mnt/lustre/share/cp/caodongliang/FlyingThings3D", "flow_data") + "\n"
        
        image_list.append(images)
        fflow_list.append(future_flows)
        pflow_list.append(past_flows)
                  
    with open("flyingthings_"+dstype+"_png.pkl", 'wb') as f:
        pickle.dump(image_list, f)
    with open("flyingthings_"+dstype+"_future_pfm.pkl", 'wb') as f:
        pickle.dump(fflow_list, f)
    with open("flyingthings_"+dstype+"_past_pfm.pkl", 'wb') as f:
        pickle.dump(pflow_list, f)
