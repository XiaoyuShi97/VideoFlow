# VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation
<!-- ### [Project Page](https://drinkingcoder.github.io/publication/flowformer/)  -->

> VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation  
> [Xiaoyu Shi](https://xiaoyushi97.github.io/), [Zhaoyang Huang](https://drinkingcoder.github.io), [Weikang Bian](https://wkbian.github.io/), [Dasong Li](https://dasongli1.github.io/), [Manyuan Zhang](https://manyuan97.github.io/), Ka Chun Cheung, Simon See, [Hongwei Qin](http://qinhongwei.com/academic/), [Jifeng Dai](https://jifengdai.org/), [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)  
> ICCV 2023

<!-- 
<img src="assets/teaser.png"> -->

## TODO List
- [ ] Training Code
- [x] Inference code for VideoFlow (2023-7-26)

## Requirements
```shell
conda create --name videoflow
conda activate videoflow
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
pip install yacs loguru einops timm==0.4.12 imageio
```

## Models
We provide pretrained [models](https://drive.google.com/drive/folders/16YqDD_IQpzrVWvDHI9xK3kO0MaXnNIGx?usp=sharing). The default path of the models for evaluation is:
```Shell
├── VideoFlow_ckpt
    ├── MOF_sintel.pth

```

## Inference & Visualization
Download VideoFlow_ckpt and put it in the root dir. Run the following command:
```shell
python -u inference.py --seq_dir demo_input_images --vis_dir demo_flow_vis
```

<!-- ## Data Preparation
Similar to RAFT, to evaluate/train FlowFormer, you will need to download the required datasets. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)

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

## Requirements
```shell
conda create --name flowformer
conda activate flowformer
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
pip install yacs loguru einops timm==0.4.12 imageio
```

## Training
The script will load the config according to the training stage. The trained model will be saved in a directory in `logs` and `checkpoints`. For example, the following script will load the config `configs/default.py`. The trained model will be saved as `logs/xxxx/final` and `checkpoints/chairs.pth`.
```shell
python -u train_FlowFormer.py --name chairs --stage chairs --validation chairs
```
To finish the entire training schedule, you can run:
```shell
./run_train.sh
```

## Models
We provide [models](https://drive.google.com/drive/folders/1K2dcWxaqOLiQ3PoqRdokrgWsGIf3yBA_?usp=sharing) trained in the four stages. The default path of the models for evaluation is:
```Shell
├── checkpoints
    ├── chairs.pth
    ├── things.pth
    ├── sintel.pth
    ├── kitti.pth
    ├── flowformer-small.pth 
    ├── things_kitti.pth
```
flowformer-small.pth is a small version of our flowformer. things_kitti.pth is the FlowFormer# introduced in our [supplementary](https://drinkingcoder.github.io/publication/flowformer/images/FlowFormer-supp.pdf), used for KITTI training set evaluation.

## Evaluation
The model to be evaluated is assigned by the `_CN.model` in the config file.

Evaluating the model on the Sintel training set and the KITTI training set. The corresponding config file is `configs/things_eval.py`.
```Shell
# with tiling technique
python evaluate_FlowFormer_tile.py --eval sintel_validation
python evaluate_FlowFormer_tile.py --eval kitti_validation --model checkpoints/things_kitti.pth
# without tiling technique
python evaluate_FlowFormer.py --dataset sintel
```
||with tile|w/o tile|
|----|-----|--------|
|clean|0.94|1.01|
|final|2.33|2.40|

Evaluating the small version model. The corresponding config file is `configs/small_things_eval.py`.
```Shell
# with tiling technique
python evaluate_FlowFormer_tile.py --eval sintel_validation --small
# without tiling technique
python evaluate_FlowFormer.py --dataset sintel --small
```
||with tile|w/o tile|
|----|-----|--------|
|clean|1.21|1.32|
|final|2.61|2.68|


Generating the submission for the Sintel and KITTI benchmarks. The corresponding config file is `configs/submission.py`.
```Shell
python evaluate_FlowFormer_tile.py --eval sintel_submission
python evaluate_FlowFormer_tile.py --eval kitti_submission
```
Visualizing the sintel dataset:
```Shell
python visualize_flow.py --eval_type sintel --keep_size
```
Visualizing an image sequence extracted from a video:
```Shell
python visualize_flow.py --eval_type seq
```
The default image sequence format is:
```Shell
├── demo_data
    ├── mihoyo
        ├── 000001.png
        ├── 000002.png
        ├── 000003.png
            .
            .
            .
        ├── 001000.png
``` -->


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