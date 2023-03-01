import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx.nn as nn
import math


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho
            w = w.clip(self.clip_min, self.clip_max)
            module.rho.set_value(w)
        if hasattr(module, 'w_gamma'):
            w = module.w_gamma
            w = w.clip(self.clip_min, self.clip_max)
            module.w_gamma.set_value(w)
        if hasattr(module, 'w_beta'):
            w = module.w_beta
            w = w.clip(self.clip_min, self.clip_max)
            module.w_beta.set_value(w)
