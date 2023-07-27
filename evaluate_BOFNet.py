import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.sintel_submission import get_cfg
from core.utils.misc import process_cfg
from utils import flow_viz
import core.datasets_3frames as datasets
from core import datasets_multiframes

from core.Networks import build_network

from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate
import itertools


@torch.no_grad()
def create_sintel_submission(model, output_path='output'):
    """ Create submission for the Sintel leaderboard """
    print("no warm start")
    results = {}
    model.eval()

    for dstype in ['final', 'clean']:
        test_dataset = datasets.MpiSintel_submission(split='test', aug_params=None, dstype=dstype, root="Sintel-test", reverse_rate=-1)

        for test_id in range(len(test_dataset)):
            if (test_id+1) % 100 == 0:
                print(f"{test_id} / {len(test_dataset)}")

            images, (sequence, frame) = test_dataset[test_id]
            images = images[None].cuda() 

            padder = InputPadder(images.shape)
            images = padder.pad(images)

            flow_pre, _ = model(images, {})

            flow = padder.unpad(flow_pre[0][0]).permute(1, 2, 0).cpu().numpy()

            # flow_img = flow_viz.flow_to_image(flow)
            # image = Image.fromarray(flow_img)
            # if not os.path.exists(f'vis_sintel_3frames_f'):
            #     os.makedirs(f'vis_sintel_3frames_f/flow')
                
            # image.save(f'vis_sintel_3frames_f/flow/{sequence}_{frame}_forward.png')

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)

    return results

@torch.no_grad()
def validate_sintel(model):
    """ Peform validation using the Sintel (train) split """

    model.eval()
    results = {}

    records = []

    boundary_index = [0, 19, 68, 117, 166, 215, 264, 313, 352, 401, 421, 470, 519, 568, 617, 666, 715, 764, 813, 862, 911, 943, 992]

    for dstype in ['final', "clean"]:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, reverse_rate=-1)
        
        epe_list = []
        epe_list_no_boundary = []

        for val_id in range(len(val_dataset)):
            if val_id % 50 == 0:
                print(val_id)
                        
            images, flows, valids = val_dataset[val_id]

            images = images[None].cuda()

            padder = InputPadder(images.shape)
            images = padder.pad(images)

            flow_pre, _ = model(images, {})
            
            flow = padder.unpad(flow_pre[0][0]).cpu()
            flow_gt = flows[0]
            
            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()

            records.append("{}\n".format(torch.mean(epe)))
            
            epe_list.append(epe.view(-1).numpy())

            if val_id not in boundary_index:
                epe_list_no_boundary.append(epe.view(-1).numpy())
            else:
                print("skip~", val_id)
        
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

        epe_all = np.concatenate(epe_list_no_boundary)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) no boundary EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))

    return results

@torch.no_grad()
def validate_things(model):
    """ Peform validation using the Sintel (train) split """

    model.eval()
    results = {}

    for dstype in ['frames_cleanpass', "frames_finalpass"]:
        val_dataset = datasets_multiframes.ThingsTEST(dstype=dstype, input_frames=3, return_gt=True)
        
        epe_list = []
        epe_list_no_boundary = []

        records = []
        import pickle

        for val_id in range(len(val_dataset)):
            if val_id % 50 == 0:
                print(val_id)
                        
            images, flows, extra_info = val_dataset[val_id]

            images = images[None].cuda()
            # images = torch.flip(images, dims=[1])

            padder = InputPadder(images.shape)
            images = padder.pad(images)

            flow_pre, _ = model(images, {})
            
            flow_pre = padder.unpad(flow_pre[0]).cpu()

            flow_pre = flow_pre[:flow_pre.shape[0]//2, ...][-flows.shape[0]:, ...]
            # flow_pre = flow_pre[1:, ...]
            
            epe = torch.sum((flow_pre - flows)**2, dim=1).sqrt()
            valid = torch.sum(flows**2, dim=1).sqrt() < 400
            this_error = epe.view(-1)[valid.view(-1)].mean().item()
            #records.append(this_error)
            epe_list.append(epe.view(-1)[valid.view(-1)].numpy())

            records.append(extra_info)

            flow_pre = flow_pre[0].permute(1, 2, 0).numpy()
            flow_gt = flows[0].permute(1, 2, 0).numpy()

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = epe
    return results

@torch.no_grad()
def validate_kitti(model):
    """ Peform validation using the Sintel (train) split """

    model.eval()
    results = {}

    val_dataset = datasets_multiframes.KITTITest(input_frames=3, aug_params=None, reverse_rate=0)
    
    epe_list = []
    out_list = []

    for val_id in range(len(val_dataset)):
        if val_id % 50 == 0:
            print(val_id)
                    
        images, flows, valids = val_dataset[val_id]

        images = images[None].cuda()

        padder = InputPadder(images.shape)
        images = padder.pad(images)

        flow_pre, _ = model(images, {})
        
        flow_pre = padder.unpad(flow_pre[0]).cpu()

        flow_pre = flow_pre[0]
        valids = valids[0]
        flows = flows[0]

        epe = torch.sum((flow_pre - flows)**2, dim=0).sqrt()
        mag = torch.sum(flows**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valids.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return 

@torch.no_grad()
def create_kitti_submission(model, output_path):
    """ Peform validation using the Sintel (train) split """

    model.eval()
    results = {}

    val_dataset = datasets_multiframes.KITTISubmission(input_frames=3, return_gt=False)
    
    epe_list = []
    out_list = []

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for val_id in range(len(val_dataset)):
                    
        images, frame_id = val_dataset[val_id]

        print(frame_id, images.shape)

        images = images[None].cuda()

        padder = InputPadder(images.shape)
        images = padder.pad(images)

        flow_pre, _ = model(images, {})
        
        flow_pre = padder.unpad(flow_pre[0]).cpu()

        flow_pre = flow_pre[0].permute(1, 2, 0).numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow_pre)

        flow_img = flow_viz.flow_to_image(flow_pre)
        image = Image.fromarray(flow_img)

        if not os.path.exists(f'vis_kitti'):
            os.makedirs(f'vis_kitti/flow')

        image.save(f'vis_kitti/flow/{frame_id}.png')

    return 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    cfg = get_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_network(cfg))
    
    if cfg.model is not None:
        model.load_state_dict(torch.load(cfg.model))
    else:
        print("[Not loading pretrained checkpoint]")

    model.cuda()
    model.eval()

    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))
    print(args.dataset)
    with torch.no_grad():
        if args.dataset == 'sintel':
            validate_sintel(model.module)
        elif args.dataset == 'things':
            validate_things(model.module)
        elif args.dataset == 'kitti':
            validate_kitti(model.module)
        elif args.dataset == 'kitti_submission':
            create_kitti_submission(model.module, output_path="flow")
        elif args.dataset == 'sintel_submission':
            create_sintel_submission(model.module, output_path="output")
        

    print(cfg.model)


