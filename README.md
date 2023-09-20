# [VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation](https://arxiv.org/abs/2303.08340)
<!-- ### [Project Page](https://drinkingcoder.github.io/publication/flowformer/)  -->

> VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation  
> [Xiaoyu Shi](https://xiaoyushi97.github.io/), [Zhaoyang Huang](https://drinkingcoder.github.io), [Weikang Bian](https://wkbian.github.io/), [Dasong Li](https://dasongli1.github.io/), [Manyuan Zhang](https://manyuan97.github.io/), Ka Chun Cheung, Simon See, [Hongwei Qin](http://qinhongwei.com/academic/), [Jifeng Dai](https://jifengdai.org/), [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)  
> ICCV 2023

https://github.com/XiaoyuShi97/VideoFlow/assets/25840016/8121acc6-b874-411e-86de-df55f7d386a9


## Requirements
```shell
conda create --name videoflow
conda activate videoflow
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv-python -c pytorch
pip install yacs loguru einops timm==0.4.12 imageio
```

## Models
We provide pretrained [models](https://drive.google.com/drive/folders/16YqDD_IQpzrVWvDHI9xK3kO0MaXnNIGx?usp=sharing). The default path of the models for evaluation is:
```Shell
├── VideoFlow_ckpt
    ├── MOF_sintel.pth
    ├── BOF_sintel.pth
    ├── MOF_things.pth
    ├── BOF_things.pth
    ├── MOF_kitti.pth
    ├── BOF_kitti.pth
```

## Inference & Visualization
Download VideoFlow_ckpt and put it in the root dir. Run the following command:
```shell
python -u inference.py --mode MOF --seq_dir demo_input_images --vis_dir demo_flow_vis
```
If your input only contain three frames, we recommend to use the BOF model:
```shell
python -u inference.py --mode BOF --seq_dir demo_input_images_three_frames --vis_dir demo_flow_vis_three_frames
```

## Data Preparation
To evaluate/train VideoFlow, you will need to download the required datasets. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) (multi-view extension, 20 frames per scene, 14 GB)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/)

By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```


## Training
The script will load the config according to the training stage. The trained model will be saved in a directory in `logs` and `checkpoints`. For example, the following script will load the config `configs/***.py`. The trained model will be saved as `logs/xxxx/final`.
```shell
# Train MOF model
python -u train_MOFNet.py --name MOF-things --stage things --validation sintel
python -u train_MOFNet.py --name MOF-sintel --stage sintel --validation sintel
python -u train_MOFNet.py --name MOF-kitti --stage kitti --validation sintel

# Train BOF model
python -u train_BOFNet.py --name BOF-things --stage things --validation sintel
python -u train_BOFNet.py --name BOF-sintel --stage sintel --validation sintel
python -u train_BOFNet.py --name BOF-kitti --stage kitti --validation sintel
```

## Evaluation
The script will load the config `configs/multiframes_sintel_submission.py` or `configs/sintel_submission.py`. Please change the `_CN.model` in the config file to load corresponding checkpoints.
```shell
# Evaluate MOF_things.pth after C stage
python -u evaluate_MOFNet.py --dataset=sintel
python -u evaluate_MOFNet.py --dataset=things
python -u evaluate_MOFNet.py --dataset=kitti
# To evaluate MOF_sintel.pth, create submission to Sintel bechmark after C+S
python -u evaluate_MOFNet.py --dataset=sintel_submission_stride1
# To evaluate MOF_kitti.pth, create submission to Kitti bechmark after C+S+K
python -u evaluate_MOFNet.py --dataset=kitti_submission
```
Similarly, to evaluate BOF models:
```shell
# Evaluate BOF_things.pth after C stage
python -u evaluate_BOFNet.py --dataset=sintel
python -u evaluate_BOFNet.py --dataset=things
python -u evaluate_BOFNet.py --dataset=kitti
# To evaluate BOF_sintel.pth, create submission to Sintel bechmark after C+S
python -u evaluate_BOFNet.py --dataset=sintel_submission
# To evaluate BOF_kitti.pth, create submission to Kitti bechmark after C+S+K
python -u evaluate_BOFNet.py --dataset=kitti_submission
```

## (Optional & Inference Only) Efficent Implementation
You can optionally use RAFT alternate (efficent) implementation by compiling the provided cuda extension and change the [`corr_fn`](https://github.com/XiaoyuShi97/VideoFlow/blob/main/configs/multiframes_sintel_submission.py#L32) flag to be `efficient` in config files.
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```
Note that this implementation is somewhat slower than all-pairs, but uses significantly less GPU memory during the forward pass. And it does not implement backward function, so do not use it in training.

## License
VideoFlow is released under the Apache License

## Citation
```bibtex
@article{shi2023videoflow,
  title={Videoflow: Exploiting temporal cues for multi-frame optical flow estimation},
  author={Shi, Xiaoyu and Huang, Zhaoyang and Bian, Weikang and Li, Dasong and Zhang, Manyuan and Cheung, Ka Chun and See, Simon and Qin, Hongwei and Dai, Jifeng and Li, Hongsheng},
  journal={arXiv preprint arXiv:2303.08340},
  year={2023}
}
```

## Acknowledgement

In this project, we use parts of codes in:
- [RAFT](https://github.com/princeton-vl/RAFT)
- [GMA](https://github.com/zacjiang/GMA)
- [timm](https://github.com/rwightman/pytorch-image-models)
