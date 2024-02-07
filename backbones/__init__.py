from .timmfr import get_timmfrv2, replace_linear_with_lowrank_2

def get_model(name, **kwargs):

    if name=='edgeface_xs_gamma_06':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_x_small', batchnorm=False), rank_ratio=0.6)
    elif name=='edgeface_s_gamma_05':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_small', batchnorm=False), rank_ratio=0.5)
    else:
        raise ValueError()
