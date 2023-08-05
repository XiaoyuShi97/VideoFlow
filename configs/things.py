from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = ''
_CN.suffix =''
_CN.gamma = 0.8
_CN.max_flow = 400
_CN.batch_size = 8
_CN.sum_freq = 100
_CN.val_freq = 100000000
_CN.image_size = [432, 960]
_CN.add_noise = False
_CN.use_smoothl1 = False
_CN.critical_params = []

_CN.network = 'BOFNet'
_CN.mixed_precision = False
_CN.filter_epe = False

_CN.restore_ckpt = None

_CN.BOFNet = CN()
_CN.BOFNet.pretrain = True
_CN.BOFNet.gma = 'GMA-SK2'
_CN.BOFNet.cnet = 'twins'
_CN.BOFNet.fnet = 'twins'
_CN.BOFNet.corr_fn = 'default'
_CN.BOFNet.corr_levels = 4
_CN.BOFNet.mixed_precision = False

_CN.BOFNet.decoder_depth = 12
_CN.BOFNet.critical_params = ["cnet", "fnet", "pretrain", 'corr_fn', "gma", "corr_levels", "decoder_depth", "mixed_precision"]

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 25e-5
_CN.trainer.adamw_decay = 1e-4
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 120000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'
def get_cfg():
    return _CN.clone()
