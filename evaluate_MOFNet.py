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
from configs.multiframes_sintel_submission import get_cfg
from core.utils.misc import process_cfg
from utils import flow_viz
import core.datasets_multiframes as datasets

from core.Networks import build_network

from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate
import itertools
import imageio


@torch.no_grad()
def validate_sintel(model, cfg):
    """ Peform validation using the Sintel (train) split """

    model.eval()
    results = {}

    records = []

    for dstype in ['final', "clean"]:
        val_dataset = datasets.MpiSintelTrain(dstype=dstype, input_frames=cfg.input_frames, return_gt=True)
        
        epe_list = []
        epe_list_no_boundary = []

        for val_id in range(len(val_dataset)):
            if val_id % 50 == 0:
                print(val_id)
                        
            images, flows, extra_info = val_dataset[val_id]

            images = images[None].cuda()

            padder = InputPadder(images.shape)
            images = padder.pad(images)

            flow_pre, _ = model(images, {})
            
            flow_pre = padder.unpad(flow_pre[0]).cpu()

            flow_pre = flow_pre[:flow_pre.shape[0]//2, ...][-flows.shape[0]:, ...]

            epe = torch.sum((flow_pre - flows)**2, dim=1).sqrt()
            epe_list.append(epe.view(-1).numpy())

            if extra_info[1][0] == 1:
                print("[Skip {}]".format(extra_info))
            else:
                epe_list_no_boundary.append(epe)

        epe_all = np.concatenate(epe_list_no_boundary)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) no boundary EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = epe

    return results

@torch.no_grad()
def validate_things(model, cfg):
    """ Peform validation using the Sintel (train) split """

    model.eval()
    results = {}

    records = []

    for dstype in ['frames_finalpass', "frames_cleanpass"]:
        val_dataset = datasets.ThingsTEST(dstype=dstype, input_frames=cfg.input_frames, return_gt=True)
        
        epe_list = []
        epe_list_no_boundary = []

        import pickle

        for val_id in range(len(val_dataset)):
            if val_id % 50 == 0:
                print(val_id)
                        
            images, flows, extra_info = val_dataset[val_id]

            images = images[None].cuda()

            padder = InputPadder(images.shape)
            images = padder.pad(images)

            flow_pre, _ = model(images, {})
            
            flow_pre = padder.unpad(flow_pre[0]).cpu()

            flow_pre = flow_pre[:flow_pre.shape[0]//2, ...][-flows.shape[0]:, ...]

            epe = torch.sum((flow_pre - flows)**2, dim=1).sqrt()
            valid = torch.sum(flows**2, dim=1).sqrt() < 400
            this_error = epe.view(-1)[valid.view(-1)].mean().item()

            for idx in range(epe.shape[0]):
                epe_list.append(epe[idx].view(-1)[valid[idx].view(-1)].numpy())
      

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = epe
    return results

@torch.no_grad()
def validate_kitti(model, cfg):
    """ Peform validation using the Sintel (train) split """

    model.eval()
    results = {}

    val_dataset = datasets.KITTITest(input_frames=cfg.input_frames, aug_params=None, reverse_rate=0)
    
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

        gt_index = (cfg.input_frames-3) // 2

        flow_pre = flow_pre[gt_index]
        valids = valids[gt_index]
        flows = flows[gt_index]

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
def create_sintel_submission(model, cfg, output_path='output'):
    """ Create submission for the Sintel leaderboard """
    print("no warm start")
    results = {}
    model.eval()

    for dstype in ['final','clean']:
        test_dataset = datasets.MpiSintel_submission(dstype=dstype, root="Sintel-test", input_frames=cfg.input_frames, return_gt=False)
        epe_list = []

        for test_id in range(len(test_dataset)):
            if (test_id+1) % 50 == 0:
                print(f"{test_id} / {len(test_dataset)}")

            images, (frame, sequence) = test_dataset[test_id]
            images = images[None].cuda() 

            padder = InputPadder(images.shape)
            images = padder.pad(images)

            flow_pre, _ = model(images, {})

            flow_pre = padder.unpad(flow_pre[0]).cpu()
            flow_pre = flow_pre[:flow_pre.shape[0]//2, ...][-len(frame):, ...]

            for idx in range(flow_pre.shape[0]):
                
                _flow = flow_pre[idx].permute(1, 2, 0).numpy()


                #flow_img = flow_viz.flow_to_image(_flow)
                #image = Image.fromarray(flow_img)
                #if not os.path.exists(f'vis_sintel'):
                #    os.makedirs(f'vis_sintel/flow')
                #    os.makedirs(f'vis_sintel/image')
                    
                #image.save(f'vis_sintel/flow/{sequence}_{frame[idx]}.png')
                #imageio.imwrite(f'vis_sintel/image/{test_id}_0.png', image1[0].cpu().permute(1, 2, 0).numpy())
                #imageio.imwrite(f'vis_sintel/image/{test_id}_1.png', image2[0].cpu().permute(1, 2, 0).numpy())

                output_dir = os.path.join(output_path, dstype, sequence)
                output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame[idx]))

                if not os.path.exists(output_dir):
                   os.makedirs(output_dir)

                frame_utils.writeFlow(output_file, _flow)


    return results


@torch.no_grad()
def create_sintel_submission_stride1(model, cfg, output_path='sintel_submission_multi8_768'):
    """ Create submission for the Sintel leaderboard """
    print("no warm start")
    print("[stride1]")
    results = {}
    model.eval()

    for dstype in ['final', 'clean']:
        test_dataset = datasets.MpiSintel_submission_stride1(dstype=dstype, root="/mnt/lustre/shixiaoyu1/data/Sintel-test", input_frames=cfg.input_frames, return_gt=False)
        epe_list = []

        for test_id in range(len(test_dataset)):
            if (test_id+1) % 50 == 0:
                print(f"{test_id} / {len(test_dataset)}")

            images, (frame, sequence) = test_dataset[test_id]
            images = images[None].cuda() 

            padder = InputPadder(images.shape)
            images = padder.pad(images)

            flow_pre, _ = model(images, {})

            flow_pre = padder.unpad(flow_pre[0]).cpu()
            
            if frame[0] == 1:
                flow_pre_back = flow_pre[flow_pre.shape[0]//2:, ...][:len(frame), ...]
                flow_pre = flow_pre[:flow_pre.shape[0]//2, ...][:len(frame), ...]
            elif len(frame) == 1:
                flow_pre_back = flow_pre[flow_pre.shape[0]//2:, ...][(cfg.input_frames-2)//2:(cfg.input_frames-2)//2+1, ...]
                flow_pre = flow_pre[:flow_pre.shape[0]//2, ...][(cfg.input_frames-2)//2:(cfg.input_frames-2)//2+1, ...]
            else:
                assert len(frame) == ((cfg.input_frames-2)+1)//2
                flow_pre_back = flow_pre[flow_pre.shape[0]//2:, ...][-len(frame):, ...]
                flow_pre = flow_pre[:flow_pre.shape[0]//2, ...][-len(frame):, ...]
            print(flow_pre.shape, frame)
            for idx in range(flow_pre.shape[0]):
                
                _flow = flow_pre[idx].permute(1, 2, 0).numpy()

                output_dir = os.path.join(output_path, dstype, sequence)
                output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame[idx]))

                if not os.path.exists(output_dir):
                   os.makedirs(output_dir)

                frame_utils.writeFlow(output_file, _flow)

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))

    return results

@torch.no_grad()
def create_kitti_submission(model, cfg):

    model.eval()
    results = {}

    val_dataset = datasets.KITTISubmission(input_frames=cfg.input_frames, return_gt=False)
    
    epe_list = []
    out_list = []

    output_path = "flow"

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

        flow_pre = flow_pre[(cfg.input_frames-3)//2].permute(1, 2, 0).numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow_pre)

        flow_img = flow_viz.flow_to_image(flow_pre)
        image = Image.fromarray(flow_img)

        if not os.path.exists(f'vis_kitti'):
            os.makedirs(f'vis_kitti/flow')

        image.save(f'vis_kitti/flow/{frame_id}.png')

    return 

@torch.no_grad()
def create_kitti_submission_online(model, cfg):
    """ Peform validation using the Sintel (train) split """

    model.eval()
    results = {}

    val_dataset = datasets.KITTISubmission_online(input_frames=cfg.input_frames, return_gt=False)
    
    epe_list = []
    out_list = []

    output_path = "flow"

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

        flow_pre = flow_pre[2].permute(1, 2, 0).numpy()

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
    args = parser.parse_args()
    cfg = get_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_network(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))
    print(args.dataset)
    print("[Input frame number is: {}]".format(cfg.input_frames))
    
    with torch.no_grad():
        if args.dataset == 'sintel':
            validate_sintel(model.module, cfg)
        elif args.dataset == 'things':
            validate_things(model.module, cfg)
        elif args.dataset == 'kitti':
            validate_kitti(model.module, cfg)
        elif args.dataset == 'sintel_submission':
            create_sintel_submission(model.module, cfg, output_path="output")
        elif args.dataset == 'sintel_submission_stride1':
            create_sintel_submission_stride1(model.module, cfg, output_path="output")
        elif args.dataset == 'kitti_submission':
            create_kitti_submission(model.module, cfg)
        elif args.dataset == 'kitti_submission_online':
            create_kitti_submission_online(model.module, cfg)




