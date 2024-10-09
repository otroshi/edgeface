"""
===============================================================================
Author: Anjith George
Institution: Idiap Research Institute, Martigny, Switzerland.

Copyright (C) 2024 Anjith George

This software is distributed under the terms described in the LICENSE file 
located in the parent directory of this source code repository. 

For inquiries, please contact the author at anjith.george@idiap.ch
===============================================================================
"""

dependencies = ['torch', 'torchvision', 'timm']

from backbones import get_model
import torch

def edgeface_base(pretrained=True, **kwargs):
    model = get_model('edgeface_base', **kwargs)
    if pretrained:
        checkpoint_url = 'https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface/-/raw/master/checkpoints/edgeface_base.pt'
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint_url, map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model

def edgeface_xs_gamma_06(pretrained=True, **kwargs):
    model = get_model('edgeface_xs_gamma_06', **kwargs)
    if pretrained:
        checkpoint_url = 'https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface/-/raw/master/checkpoints/edgeface_xs_gamma_06.pt'
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint_url, map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model

def edgeface_xs_q(pretrained=True, **kwargs):
    model = get_model('edgeface_xs_q', **kwargs)
    if pretrained:
        checkpoint_url = 'https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface/-/raw/master/checkpoints/edgeface_xs_q.pt'
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint_url, map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model

def edgeface_xxs(pretrained=True, **kwargs):
    model = get_model('edgeface_xxs', **kwargs)
    if pretrained:
        checkpoint_url = 'https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface/-/raw/master/checkpoints/edgeface_xxs.pt'
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint_url, map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model

def edgeface_xxs_q(pretrained=True, **kwargs):
    model = get_model('edgeface_xxs_q', **kwargs)
    if pretrained:
        checkpoint_url = 'https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface/-/raw/master/checkpoints/edgeface_xxs_q.pt'
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint_url, map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model

def edgeface_s_gamma_05(pretrained=True, **kwargs):
    model = get_model('edgeface_s_gamma_05', **kwargs)
    if pretrained:
        checkpoint_url = 'https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface/-/raw/master/checkpoints/edgeface_s_gamma_05.pt'
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint_url, map_location='cpu'
        )
        model.load_state_dict(state_dict)
    return model
