import numpy as np
import os.path as osp
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Config variables for General 

__C.TextDim = 1024
__C.LatentDim = 100
# __C.ImageSizeStageI = 64
# __C.ImageSizeStageII = 256
__C.CUDA = True
__C.GPU_ID = '0'
__C.DatasetName = 'coco'
__C.Workers = 1
__C.ConfigName = ''
__C.Stage = 1
__C.DataDir = ''
__C.ImSize = 64
# Config variables for Training GANS

__C.BatchSize = 64
__C.NumEpochs = 50
__C.LrDecayEpoch = 600
__C.saveModelEpoch = 50
__C.DiscLR = 0.0002
__C.GenLR = 0.0002
__C.KLCoeff = 2
__C.RNum = 4
# Config parameters for mode options for GANS

__C.ConditionDim = 128
__C.DiscInputDim = 64
__C.GenInputDim = 128

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
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)



