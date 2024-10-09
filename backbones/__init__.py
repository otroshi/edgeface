"""
===============================================================================
Author: Anjith George
Institution: Idiap Research Institute, Martigny, Switzerland.

Copyright (C) 2023 Anjith George

This software is distributed under the terms described in the LICENSE file 
located in the parent directory of this source code repository. 

For inquiries, please contact the author at anjith.george@idiap.ch
===============================================================================
"""
from .timmfr import get_timmfrv2, replace_linear_with_lowrank_2

import torch

def get_model(name, **kwargs):

    if name=='edgeface_xs_gamma_06':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_x_small', batchnorm=False), rank_ratio=0.6)
    elif name=='edgeface_xs_q':
        model= get_timmfrv2('edgenext_x_small', batchnorm=False)
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model
    elif  name=='edgeface_xxs':
        return get_timmfrv2('edgenext_xx_small', batchnorm=False)
    elif  name=='edgeface_base':
        return get_timmfrv2('edgenext_base', batchnorm=False)
    elif name=='edgeface_xxs_q':
        model=get_timmfrv2('edgenext_xx_small', batchnorm=False)
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model   
    elif name=='edgeface_s_gamma_05':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_small', batchnorm=False), rank_ratio=0.5)
    else:
        raise ValueError()
