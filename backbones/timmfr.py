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



import timm
import torch
import torch.nn as nn
import math

class LoRaLin(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LoRaLin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        return x

def replace_linear_with_lowrank_recursive_2(model, rank_ratio=0.2):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and 'head' not in name:
            in_features = module.in_features
            out_features = module.out_features
            rank = max(2,int(min(in_features, out_features) * rank_ratio))
            bias=False
            if module.bias is not None:
                bias=True
            lowrank_module = LoRaLin(in_features, out_features, rank, bias)

            setattr(model, name, lowrank_module)
        else:
            replace_linear_with_lowrank_recursive_2(module, rank_ratio)

def replace_linear_with_lowrank_2(model, rank_ratio=0.2):
    replace_linear_with_lowrank_recursive_2(model, rank_ratio)
    return model


        
class TimmFRWrapperV2(nn.Module):
    """
    Wraps timm model
    """
    def __init__(self, model_name='edgenext_x_small', featdim=512, batchnorm=False):
        super().__init__()
        self.featdim = featdim
        self.model_name = model_name
        
        self.model = timm.create_model(self.model_name)
        self.model.reset_classifier(self.featdim)

    def forward(self, x):
        x = self.model(x)
        return x


def get_timmfrv2(model_name, **kwargs):
    """
    Create an instance of TimmFRWrapperV2 with the specified `model_name` and additional arguments passed as `kwargs`.
    """
    return TimmFRWrapperV2(model_name=model_name, **kwargs)
