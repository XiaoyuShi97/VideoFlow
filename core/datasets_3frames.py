import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from .utils import frame_utils
from .utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from torchvision.utils import save_image

from .utils import flow_viz
import cv2
from .utils.utils import coords_grid, bilinear_sampler

import copy

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, oneside=False, reverse_rate=0.3):
        self.augmentor = None
        self.sparse = sparse
        self.oneside = oneside
        self.reverse_rate = reverse_rate
        print("[reverse_rate is {}]".format(self.reverse_rate))


        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        #print(self.flow_list[index])
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img3 = frame_utils.read_gen(self.image_list[index][2])

            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img3 = np.array(img3).astype(np.uint8)[..., :3]
            
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            img3 = torch.from_numpy(img3).permute(2, 0, 1).float()
            
            return torch.stack([img1, img2, img3]), self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid1 = valid2 = None

        if self.oneside:
            if self.sparse:
                flow1, valid1 = frame_utils.readFlowKITTI(self.flow_list[index])
                flow2 = copy.deepcopy(flow1)
                valid2 = copy.deepcopy(valid1) * 0
            else:
                flow1 = frame_utils.read_gen(self.flow_list[index])
                flow2 = copy.deepcopy(flow1) * 0 + 10000

        else:
            flow1 = frame_utils.read_gen(self.flow_list[index][0])
            flow2 = frame_utils.read_gen(self.flow_list[index][1])


        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img3 = frame_utils.read_gen(self.image_list[index][2])
        
        flow1 = np.array(flow1).astype(np.float32)
        flow2 = np.array(flow2).astype(np.float32)
        
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        img3 = np.array(img3).astype(np.uint8)
        
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
            img3 = np.tile(img3[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
            img3 = img3[..., :3]
            
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, img3, flow1, flow2, valid1, valid2 = self.augmentor(img1, img2, img3, flow1, flow2, valid1, valid2)
            else:
                img1, img2, img3, flow1, flow2 = self.augmentor(img1, img2, img3, flow1, flow2)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        img3 = torch.from_numpy(img3).permute(2, 0, 1).float()
        
        flow1 = torch.from_numpy(flow1).permute(2, 0, 1).float()
        flow2 = torch.from_numpy(flow2).permute(2, 0, 1).float()

        if valid1 is not None and valid2 is not None:
            valid1 = torch.from_numpy(valid1)
            valid2 = torch.from_numpy(valid2) * 0 # sparse must be oneside
        else:
            valid1 = (flow1[0].abs() < 1000) & (flow1[1].abs() < 1000)
            valid2 = (flow2[0].abs() < 1000) & (flow2[1].abs() < 1000)
        
        if np.random.rand() < self.reverse_rate:
            return torch.stack([img3, img2, img1]), torch.stack([flow2, flow1]), torch.stack([valid2.float(), valid1.float()])
        else:
            return torch.stack([img1, img2, img3]), torch.stack([flow1, flow2]), torch.stack([valid1.float(), valid2.float()])

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)

class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        self.image_list = []
        with open("./flow_datasets/flying_things_three_frames/flyingthings_"+dstype+"_png.txt") as f:
            images = f.readlines()
            for img1, img2, img3 in zip(images[0::3], images[1::3], images[2::3]):
                self.image_list.append([root+img1.strip(), root+img2.strip(), root+img3.strip()])
        self.flow_list = []
        with open("./flow_datasets/flying_things_three_frames/flyingthings_"+dstype+"_pfm.txt") as f:
            flows = f.readlines()
            for flow1, flow2 in zip(flows[0::2], flows[1::2]):
                self.flow_list.append([root+flow1.strip(), root+flow2.strip()])

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean', reverse_rate=0.3):
        super(MpiSintel, self).__init__(aug_params, oneside=True, reverse_rate=reverse_rate)

        self.image_list = []
        with open("./flow_datasets/sintel_three_frames/Sintel_"+dstype+"_png.txt") as f:
            images = f.readlines()
            for img1, img2, img3 in zip(images[0::3], images[1::3], images[2::3]):
                self.image_list.append([root+img1.strip(), root+img2.strip(), root+img3.strip()])
        
        self.flow_list = []
        with open("./flow_datasets/sintel_three_frames/Sintel_"+dstype+"_flo.txt") as f:
            flows = f.readlines()
            for flow in flows:
                self.flow_list.append(root+flow.strip())
        
        assert (len(self.image_list) == len(self.flow_list))

        self.extra_info = []
        with open("./flow_datasets/sintel_three_frames/Sintel_"+dstype+"_extra_info.txt") as f:
            info = f.readlines()
            for scene, id in zip(info[0::2], info[1::2]):
                self.extra_info.append((scene.strip(), int(id.strip())))

class MpiSintel_submission(FlowDataset):
    def __init__(self, aug_params=None, split='test', root='datasets/Sintel', dstype='clean', reverse_rate=-1):
        super(MpiSintel_submission, self).__init__(aug_params, oneside=True, reverse_rate=-1)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                if i==0:
                    self.image_list += [ [image_list[i], image_list[i], image_list[i+1]] ]
                else:
                    self.image_list += [ [image_list[i-1], image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True, oneside=True)

        self.image_list = []
        with open("./flow_datasets/hd1k_three_frames/hd1k_image.txt") as f:
            images = f.readlines()
            for img1, img2, img3 in zip(images[0::3], images[1::3], images[2::3]):
                self.image_list.append([root+img1.strip(), root+img2.strip(), root+img3.strip()])
        self.flow_list = []
        with open("./flow_datasets/hd1k_three_frames/hd1k_flo.txt") as f:
            flows = f.readlines()
            for flow in flows:
                self.flow_list.append(root+flow.strip())

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True, oneside=True)
        if split == 'testing':
            self.is_test = True

        self.image_list = []
        with open("./flow_datasets/KITTI/KITTI_{}_image.txt".format(split)) as f:
            images = f.readlines()
            for img1, img2 in zip(images[0::2], images[1::2]):
                self.image_list.append([root+img1.strip().replace("_10", "_09").replace("KITTI", "KITTI-full"), root+img1.strip(), root+img2.strip()])

        self.extra_info = []
        with open("./flow_datasets/KITTI/KITTI_{}_extra_info.txt".format(split)) as f:
            info = f.readlines()
            for id in info:
                self.extra_info.append([id.strip()])

        if split == "training":
            self.flow_list = []
            with open("./flow_datasets/KITTI/KITTI_{}_flow.txt".format(split)) as f:
                flow = f.readlines()
                for flo in flow:
                    self.flow_list.append(root+flo.strip())
        
        print(self.image_list[:10])
        print(self.flow_list[:10])
        

def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
   
    if args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset
    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        
        hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
        kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
        train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things
    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=args.batch_size*2, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

if __name__ == "__main__":
    return
