from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = ''
_CN.suffix =''
_CN.gamma = 0.85
_CN.max_flow = 400
_CN.batch_size = 8
_CN.sum_freq = 100
_CN.val_freq = 100000000
_CN.image_size = [432, 960]
_CN.add_noise = False
_CN.use_smoothl1 = False
_CN.critical_params = []

_CN.network = 'MOFNetStack'

_CN.model = 'VideoFlow_ckpt/MOF_sintel.pth'
_CN.input_frames = 5

_CN.restore_ckpt = None

################################################
################################################
_CN.MOFNetStack = CN()
_CN.MOFNetStack.pretrain = True
_CN.MOFNetStack.Tfusion = 'stack'
_CN.MOFNetStack.cnet = 'twins'
_CN.MOFNetStack.fnet = 'twins'
_CN.MOFNetStack.down_ratio = 8
_CN.MOFNetStack.feat_dim = 256
_CN.MOFNetStack.corr_fn = 'default'
_CN.MOFNetStack.corr_levels = 4
_CN.MOFNetStack.mixed_precision = True
_CN.MOFNetStack.context_3D = False

_CN.MOFNetStack.decoder_depth = 32
_CN.MOFNetStack.critical_params = ["cnet", "fnet", "pretrain", 'corr_fn', "Tfusion", "corr_levels", "decoder_depth", "mixed_precision"]

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 12.5e-5
_CN.trainer.adamw_decay = 1e-4
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 90000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'
def get_cfg():
    return _CN.clone()
