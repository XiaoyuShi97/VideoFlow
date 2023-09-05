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
from .utils.augmentor_multiframes import FlowAugmentor, SparseFlowAugmentor
from torchvision.utils import save_image

from .utils import flow_viz
import cv2
from .utils.utils import coords_grid, bilinear_sampler

import copy

import pickle

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, oneside=False, input_frames=5, reverse_rate=0.3):
        self.augmentor = None
        self.sparse = sparse
        self.oneside = oneside
        self.input_frames = input_frames
        print("[input frame number is {}]".format(self.input_frames))
        self.reverse_rate = reverse_rate
        print("[reverse_rate is {}]".format(self.reverse_rate))

        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.has_gt_list = []

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        valids = None

        if self.oneside and not self.sparse:
            # sintel
            flows = []
            for idx in range(len(self.has_gt_list[index])):
                if self.has_gt_list[index][idx]:
                    flow = frame_utils.read_gen(self.flow_list[index][idx])
                    flows.append(flow)
                else:
                    flows.append(copy.deepcopy(flows[-1]) * 0 + 10000) # so invalid
        elif self.oneside and self.sparse:
            flows = []
            valids = []
            for idx in range(len(self.has_gt_list[index])):
                if self.has_gt_list[index][idx]:
                    flow, valid = frame_utils.readFlowKITTI(self.flow_list[index][idx])
                    flows.append(flow)
                    valids.append(valid)
                else:
                    flow, valid = frame_utils.readFlowKITTI(self.flow_list[index][idx])
                    flows.append(flow)
                    valids.append(valid*0.0)
        else:
            flows = [frame_utils.read_gen(path) for path in self.flow_list[index]]

        imgs = [frame_utils.read_gen(path) for path in self.image_list[index]]

        flows = [np.array(flow).astype(np.float32) for flow in flows]

        imgs = [np.array(img).astype(np.uint8) for img in imgs]
        
        # grayscale images
        if len(imgs[0].shape) == 2:
            imgs = [np.tile(img[...,None], (1, 1, 3)) for img in imgs]
        else:
            imgs = [img[..., :3] for img in imgs]
            
        if self.augmentor is not None:
            if self.sparse:
                imgs, flows, valids = self.augmentor(imgs, flows, valids)
            else:
                imgs, flows = self.augmentor(imgs, flows)
        
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]
        
        if valids is None:
            valids = [((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float() for flow in flows]
        else:
            valids = [torch.from_numpy(valid).float() for valid in valids]

        if np.random.rand() < self.reverse_rate:
            return torch.stack(imgs[::-1]), torch.stack(flows[::-1]), torch.stack(valids[::-1])
        else:
            return torch.stack(imgs), torch.stack(flows), torch.stack(valids)

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.has_gt_list = v * self.has_gt_list
        return self
        
    def __len__(self):
        return len(self.image_list)


class FlowDatasetTest(data.Dataset):
    def __init__(self, input_frames=5, return_gt=True):
        self.input_frames = input_frames
        print("[input frame number is {}]".format(self.input_frames))

        self.return_gt = return_gt
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info_list = []

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        imgs = [frame_utils.read_gen(path) for path in self.image_list[index]]
        imgs = [np.array(img).astype(np.uint8) for img in imgs]
        # grayscale images
        if len(imgs[0].shape) == 2:
            imgs = [np.tile(img[...,None], (1, 1, 3)) for img in imgs]
        else:
            imgs = [img[..., :3] for img in imgs]
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]

        if self.return_gt:
            flows = [frame_utils.read_gen(path) for path in self.flow_list[index]]
            flows = [np.array(flow).astype(np.float32) for flow in flows]
            flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]

            return torch.stack(imgs), torch.stack(flows), self.extra_info_list[index]
        else:
            return torch.stack(imgs), self.extra_info_list[index]

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)

class MpiSintelTrain(FlowDatasetTest):
    def __init__(self, return_gt=True, input_frames=5, dstype='clean'):
        super(MpiSintelTrain, self).__init__(input_frames=input_frames, return_gt=return_gt)

        root = 'datasets/Sintel/'

        self.image_list = []
        self.flow_list = []
        self.extra_info_list = []

        with open("./flow_dataset_mf/sintel_training_"+dstype+"_png.pkl", "rb") as f:
            _image_list = pickle.load(f)
        with open("./flow_dataset_mf/sintel_training_"+dstype+"_flo.pkl", "rb") as f:
            _future_flow_list = pickle.load(f)
        with open("./flow_dataset_mf/sintel_training_scene.pkl", "rb") as f:
            extra_info_list = pickle.load(f)
        
        len_list = len(_image_list)

        for idx_list in range(len_list):
            image_num = 0
            flow_num = 0
            tested_flow_num = 0

            _images = _image_list[idx_list]
            _future_flows = _future_flow_list[idx_list]

            len_image = len(_images)

            for idx in range(len_image-1):
                _images[idx] = root+_images[idx].strip()
                _future_flows[idx] = root+_future_flows[idx].strip()
                image_num += 1
                flow_num += 1
            _images[-1] = root+_images[-1].strip()
            image_num += 1
            _images = [_images[0]] + _images # repeat the first frame
            len_image = len_image + 1 # repeat the first frame

            for idx_image in range(0, len_image-input_frames+1, input_frames-2):
                self.image_list.append(_images[idx_image:idx_image+input_frames])
                self.flow_list.append(_future_flows[idx_image:idx_image+input_frames-2])
                self.extra_info_list.append((extra_info_list[idx_list], list(range(idx_image+1, idx_image+input_frames-1))))
                tested_flow_num += (input_frames - 2)

            remainder = (len_image-2) % (input_frames-2)
            if remainder > 0:
                self.image_list.append(_images[-input_frames:])
                self.flow_list.append(_future_flows[-remainder:])
                self.extra_info_list.append((extra_info_list[idx_list], list(range(len_image-remainder-1, len_image-1))))
                tested_flow_num += (remainder)
            print(image_num, flow_num, tested_flow_num)


        print(self.image_list[:20])
        print(self.flow_list[:20])
        print(self.extra_info_list[:20])

class ThingsTEST(FlowDatasetTest):
    def __init__(self, return_gt=True, input_frames=5, dstype='frames_cleanpass'):
        super(ThingsTEST, self).__init__(input_frames=input_frames, return_gt=return_gt)

        root = 'datasets/FlyingThings3D/'

        self.image_list = []
        self.flow_list = []
        self.extra_info_list = []

        len_image = 10

        for subset in ["A", "B", "C"]:

            for dir_index in range(50):

                if (subset, dir_index) in [("A", 4), ("B", 31), ("C", 18), ("C", 43)]:
                    continue

                _images = [root+"flow_data/"+dstype+"/TEST/"+subset+"/{:04}".format(dir_index)+"/left/"+"{:04}.png".format(idx) for idx in range(6, 16)]
                _flows = [root+"flow_data/optical_flow/TEST/"+subset+"/{:04}".format(dir_index)+"/into_future/left/"+"OpticalFlowIntoFuture_{:04}_L.pfm".format(idx) for idx in range(7, 15)]

                for idx_image in range(0, len_image-input_frames+1, input_frames-2):
                    self.image_list.append(_images[idx_image:idx_image+input_frames])
                    self.flow_list.append(_flows[idx_image:idx_image+input_frames-2])
                    self.extra_info_list.append(["{}_{}".format(subset, dir_index), list(range(6+idx_image+1, 6+idx_image+input_frames-1))])
                
                remainder = (len_image-2) % (input_frames-2)
                if remainder > 0:
                    self.image_list.append(_images[-input_frames:])
                    self.flow_list.append(_flows[-remainder:])
                    self.extra_info_list.append(["{}_{}".format(subset, dir_index), list(range(6+len_image-remainder-1, 6+len_image-1))])

        print(self.image_list[:20])
        print(self.flow_list[:20])
        print(self.extra_info_list[:20])


class MpiSintel_submission(FlowDatasetTest):
    def __init__(self, return_gt=False, root='datasets/Sintel', dstype='clean', input_frames=6):
        super(MpiSintel_submission, self).__init__(return_gt=return_gt, input_frames=input_frames)

        split = "test"
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            image_list = [image_list[0]] + image_list
            len_image = len(image_list)
            _images = image_list
            for idx_image in range(0, len_image-input_frames+1, input_frames-2):
                self.image_list.append(_images[idx_image:idx_image+input_frames])
                self.extra_info_list.append((list(range(idx_image+1, idx_image+input_frames-1)), scene))
                
            remainder = (len_image-2) % (input_frames-2)
            if remainder > 0:
                self.image_list.append(_images[-input_frames:])
                self.extra_info_list.append((list(range(len_image-remainder-1, len_image-1)), scene))
        
        for i in range(20):
            print("~~~~~~~~~~~~~~")
            print(self.image_list[i])
            print(self.extra_info_list[i])


class MpiSintel_submission_stride1(FlowDatasetTest):
    def __init__(self, return_gt=False, root='datasets/Sintel', dstype='clean', input_frames=6):
        super(MpiSintel_submission_stride1, self).__init__(return_gt=return_gt, input_frames=input_frames)

        split = "test"
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            image_list = [image_list[0]] + image_list
            len_image = len(image_list)
            _images = image_list
            for idx_image in range(0, len_image-input_frames+1, 1):
                self.image_list.append(_images[idx_image:idx_image+input_frames])
                if idx_image == 0:
                    self.extra_info_list.append(([1, 2], scene))
                elif idx_image == len_image-input_frames:
                    self.extra_info_list.append(([idx_image+2, idx_image+3], scene))
                else:
                    self.extra_info_list.append(([idx_image+2], scene))

        
        for i in range(70):
            print(self.image_list[i])
            print(self.extra_info_list[i])

class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, input_frames=5, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params=aug_params, input_frames=input_frames, oneside=False, sparse=False)

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []
        with open("./flow_dataset_mf/flyingthings_"+dstype+"_png.pkl", "rb") as f:
            _image_list = pickle.load(f)
        with open("./flow_dataset_mf/flyingthings_"+dstype+"_future_pfm.pkl", "rb") as f:
            _future_flow_list = pickle.load(f)
        with open("./flow_dataset_mf/flyingthings_"+dstype+"_past_pfm.pkl", "rb") as f:
            _past_flow_list = pickle.load(f)
        
        len_list = len(_image_list)
        print(len(_image_list), len(_future_flow_list), len(_past_flow_list))

        for idx_list in range(len_list):
            _images = _image_list[idx_list]
            _future_flows = _future_flow_list[idx_list]
            _past_flows = _past_flow_list[idx_list]
            
            len_image = len(_images)

            for idx in range(len_image):
                _images[idx] = root+_images[idx].strip()
                _future_flows[idx] = root+_future_flows[idx].strip()
                _past_flows[idx] = root+_past_flows[idx].strip()

            for idx_image in range(0, len_image-input_frames+1):
                self.image_list.append(_images[idx_image:idx_image+input_frames])
                self.flow_list.append(_future_flows[idx_image+1:idx_image+input_frames-1]+_past_flows[idx_image+1:idx_image+input_frames-1])
                self.has_gt_list.append([True]*(input_frames-2)*2)
       
        print(self.image_list[:10])
        print(self.flow_list[:10])
        print(self.has_gt_list[:10])


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, input_frames=5, dstype='clean'):
        super(MpiSintel, self).__init__(aug_params=aug_params, input_frames=input_frames, oneside=True, sparse=False)

        root = 'datasets/Sintel/'

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []

        with open("./flow_dataset_mf/sintel_training_"+dstype+"_png.pkl", "rb") as f:
            _image_list = pickle.load(f)
        with open("./flow_dataset_mf/sintel_training_"+dstype+"_flo.pkl", "rb") as f:
            _future_flow_list = pickle.load(f)
        
        len_list = len(_image_list)
        print(len(_image_list), len(_future_flow_list))

        for idx_list in range(len_list):
            _images = _image_list[idx_list]
            _future_flows = _future_flow_list[idx_list]
            
            len_image = len(_images)

            for idx in range(len_image-1):
                _images[idx] = root+_images[idx].strip()
                _future_flows[idx] = root+_future_flows[idx].strip()
            _images[-1] = root+_images[-1].strip()

            for idx_image in range(-1, len_image-input_frames+1):
                if idx_image == -1:
                    self.image_list.append([_images[0]]+_images[0:input_frames-1])
                    self.flow_list.append(_future_flows[idx_image+1:idx_image+input_frames-1]*2)
                    self.has_gt_list.append([True]*(input_frames-2)+[False]*(input_frames-2))
                else:
                    self.image_list.append(_images[idx_image:idx_image+input_frames])
                    self.flow_list.append(_future_flows[idx_image+1:idx_image+input_frames-1]*2)
                    self.has_gt_list.append([True]*(input_frames-2)+[False]*(input_frames-2))

class HD1K(FlowDataset):
    def __init__(self, aug_params=None, input_frames=5):
        super(HD1K, self).__init__(aug_params=aug_params, input_frames=input_frames, oneside=True, sparse=True)

        root = 'datasets/KITTI/'

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []

        with open("./flow_dataset_mf/hd1k_png.pkl", "rb") as f:
            _image_list = pickle.load(f)
        with open("./flow_dataset_mf/hd1k_flo.pkl", "rb") as f:
            _future_flow_list = pickle.load(f)
        
        len_list = len(_image_list)
        print(len(_image_list), len(_future_flow_list))

        for idx_list in range(len_list):
            _images = _image_list[idx_list]
            _future_flows = _future_flow_list[idx_list]
            
            len_image = len(_images)

            for idx in range(len_image-1):
                _images[idx] = root+_images[idx].strip()
                _future_flows[idx] = root+_future_flows[idx].strip()
            _images[-1] = root+_images[-1].strip()

            for idx_image in range(-1, len_image-input_frames+1):
                if idx_image == -1:
                    self.image_list.append([_images[0]]+_images[0:input_frames-1])
                    self.flow_list.append(_future_flows[idx_image+1:idx_image+input_frames-1]*2)
                    self.has_gt_list.append([True]*(input_frames-2)+[False]*(input_frames-2))
                else:
                    self.image_list.append(_images[idx_image:idx_image+input_frames])
                    self.flow_list.append(_future_flows[idx_image+1:idx_image+input_frames-1]*2)
                    self.has_gt_list.append([True]*(input_frames-2)+[False]*(input_frames-2))

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, input_frames=5, reverse_rate=0.3):
        super(KITTI, self).__init__(aug_params=aug_params, input_frames=input_frames, oneside=True, sparse=True, reverse_rate=reverse_rate)

        root = 'datasets/KITTI/'

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []

        for idx_list in range(200):
            for idx_image in range(1, input_frames-1):
                self.image_list.append([(root+"KITTI-full/training/image_2/000{:03}_{:02}.png".format(idx_list, i-idx_image+10)) for i in range(input_frames)])
                self.flow_list.append([root+"KITTI/training/flow_occ/000{:03}_10.png".format(idx_list)]*2*(input_frames-2))
                self.has_gt_list.append([False]*(idx_image-1)+[True]+[False]*((input_frames-2)*2-idx_image))

class KITTITest(FlowDataset):
    def __init__(self, aug_params=None, input_frames=5, reverse_rate=0.3):
        super(KITTITest, self).__init__(aug_params=aug_params, input_frames=input_frames, oneside=True, sparse=True, reverse_rate=reverse_rate)

        root = 'datasets/KITTI/'

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []

        for idx_list in range(200):
            for idx_image in [(input_frames-1)//2]:
                self.image_list.append([(root+"KITTI-full/training/image_2/000{:03}_{:02}.png".format(idx_list, i-idx_image+10)) for i in range(input_frames)])
                self.flow_list.append([root+"KITTI/training/flow_occ/000{:03}_10.png".format(idx_list)]*2*(input_frames-2))
                self.has_gt_list.append([False]*(idx_image-1)+[True]+[False]*((input_frames-2)*2-idx_image))

        print(self.image_list[:10])
        print(self.flow_list[:10])
        print(self.has_gt_list[:10])

class KITTISubmission(FlowDatasetTest):
    def __init__(self, return_gt=False, input_frames=5):
        super(KITTISubmission, self).__init__(input_frames=input_frames, return_gt=return_gt)

        root = 'data_scene_flow_multiview/testing/'

        self.image_list = []
        self.extra_info_list = []

        for idx_list in range(200):
            for idx_image in [(input_frames-1)//2]:
                self.image_list.append([(root+"image_2/000{:03}_{:02}.png".format(idx_list, i-idx_image+10)) for i in range(input_frames)])
                self.extra_info_list.append("000{:03}_10.png".format(idx_list))

        print(self.image_list[:10])
        print(self.extra_info_list[:10])
    
class KITTISubmission_online(FlowDatasetTest):
    def __init__(self, return_gt=False, input_frames=5):
        super(KITTISubmission_online, self).__init__(input_frames=input_frames, return_gt=return_gt)

        root = 'data_scene_flow_multiview/testing/'

        self.image_list = []
        self.extra_info_list = []

        for idx_list in range(200):
            for idx_image in [3]:
                self.image_list.append([(root+"image_2/000{:03}_{:02}.png".format(idx_list, i-idx_image+10)) for i in range(input_frames)])
                self.extra_info_list.append("000{:03}_10.png".format(idx_list))

        print(self.image_list[:10])
        print(self.extra_info_list[:10])

class KITTICenter(FlowDataset):
    def __init__(self, aug_params=None, input_frames=5):
        super(KITTICenter, self).__init__(aug_params=aug_params, input_frames=input_frames, oneside=True, sparse=True, kitti=True)

        root = 'datasets/KITTI/'

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []

        for idx_list in range(200):
            for idx_image in [(input_frames-1)//2]:
                self.image_list.append([(root+"KITTI-full/training/image_2/000{:03}_{:02}.png".format(idx_list, i-idx_image+10)) for i in range(input_frames)])
                self.flow_list.append([root+"KITTI/training/flow_occ/000{:03}_10.png".format(idx_list)]*2*(input_frames-2))
                self.has_gt_list.append([False]*(idx_image-1)+[True]+[False]*((input_frames-2)*2-idx_image))
        
        print(self.image_list[:10])
        print(self.flow_list[:10])
        print(self.has_gt_list[:10])

def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
   
    if args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, input_frames=args.input_frames, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, input_frames=args.input_frames, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset
    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass', input_frames=args.input_frames)
        sintel_clean = MpiSintel(aug_params, dstype='clean', input_frames=args.input_frames)
        sintel_final = MpiSintel(aug_params, dstype='final', input_frames=args.input_frames)        
        hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True}, input_frames=args.input_frames)
        kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}, input_frames=args.input_frames)

        print("[dataset len: ]", len(things), len(sintel_clean), len(hd1k), len(kitti))

        train_dataset = 100*sintel_clean + 100*sintel_final + 50*kitti + 5*hd1k + things
    
    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTICenter(aug_params, input_frames=args.input_frames)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=args.batch_size*2, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

if __name__ == "__main__":
    print("hi!!!!!!!!!!!!!!!")

