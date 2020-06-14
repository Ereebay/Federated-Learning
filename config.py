import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.config_name = 'fed_exp'
# Total
__C.Total = edict()
__C.Total.alogrithm = 'fed_mutual'  # fed_mutual/fed_avg/normal
__C.Total.GPU_ID = '0'  # cuda, cpu
__C.Total.node_num = 5  # Number of nodes
__C.Total.R = 50  # Number of rounds:R
__C.Total.E = 5  # Number of local epochs:E
__C.Total.notes = ''  # Note of experiments
__C.Total.data_dir = ''

# Model
__C.Model = edict()
__C.Model.global_model = 'LeNet5'  # LeNet5, MLP, CNN, ResNet18
__C.Model.local_model = 'LeNet5'  # LeNet5, MLP, CNN, ResNet18
__C.Model.catfish = None  # None, LeNet5, MLP, CNN, ResNet18

# Data
__C.Data = edict()
__C.Data.dataset = 'cifar10'  # cifar10, femnist, mnist
__C.Data.batchsize = 128
__C.Data.split = 5
__C.Data.val_ratio = 0.1
__C.Data.all_data = True  # use all data
__C.Data.sampling_mode = 'iid'
__C.Data.equal = True
__C.Data.frac = 0.8
__C.Data.alpha = 1.0

# Optima
__C.Optima = edict()
__C.Optima.optimizer = 'sgd'  # sgd, adam
__C.Optima.local_lr = 0.1  # learning rate
__C.Optima.meme_lr = 0.1
__C.Optima.lr_step = 1  # learning rate decay stop decay
__C.Optima.stop_decay = 50  # round when learning rate stop decay
__C.Optima.momentum = 0.9  # SGD momentum
__C.Optima.alpha = 0.5  # local ratio of data loss
__C.Optima.beta = 0.5  # meme ratio of data loss


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)
