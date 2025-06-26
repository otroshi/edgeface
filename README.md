

# EdgeFace: Efficient Face Recognition Model for Edge Devices

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-lfw)](https://paperswithcode.com/sota/lightweight-face-recognition-on-lfw?p=edgeface-efficient-face-recognition-model-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-calfw)](https://paperswithcode.com/sota/lightweight-face-recognition-on-calfw?p=edgeface-efficient-face-recognition-model-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-cplfw)](https://paperswithcode.com/sota/lightweight-face-recognition-on-cplfw?p=edgeface-efficient-face-recognition-model-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-cfp-fp)](https://paperswithcode.com/sota/lightweight-face-recognition-on-cfp-fp?p=edgeface-efficient-face-recognition-model-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-agedb-30)](https://paperswithcode.com/sota/lightweight-face-recognition-on-agedb-30?p=edgeface-efficient-face-recognition-model-for)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-ijb-b)](https://paperswithcode.com/sota/lightweight-face-recognition-on-ijb-b?p=edgeface-efficient-face-recognition-model-for)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-ijb-c)](https://paperswithcode.com/sota/lightweight-face-recognition-on-ijb-c?p=edgeface-efficient-face-recognition-model-for)	

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv%3A2307.01838-009d81v2.svg)](https://arxiv.org/abs/2307.01838v2)
[![HF-Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-orange)](https://huggingface.co/spaces/Idiap/EdgeFace)
[![HF-Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-green)](https://huggingface.co/collections/Idiap/edgeface-67f500eded03ecd8be56e63e)


This repository contain inference code and pretrained models to use [**EdgeFace: Efficient Face Recognition Model for Edge Devices**](https://ieeexplore.ieee.org/abstract/document/10388036/), 
which is the **winning entry** in *the compact track of ["EFaR 2023: Efficient Face Recognition Competition"](https://arxiv.org/abs/2308.04168) organised at the IEEE International Joint Conference on Biometrics (IJCB) 2023*. For the complete source code of training and evaluation, please check the [official repository](https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface).


![EdgeFace](assets/edgeface.png)

## Installation
```sh
$ pip install -r requirements.txt
```

## Inference
The following code shows how to use the model for inference:
```python
import torch
from torchvision import transforms
from face_alignment import align
from backbones import get_model

batch_size = 1
# load model
model_name="edgeface_s_gamma_05" # or edgeface_xs_gamma_06
model=get_model(model_name)
checkpoint_path=f'checkpoints/{model_name}.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')) # Load state dict
model.eval() # Call eval() on the model object

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

path = 'path_to_face_image'
aligned = align.get_aligned_face(path) # align face
transformed_input = transform(aligned) # preprocessing
transformed_input = transformed_input.reshape(batch_size, *transformed_input.shape)
# extract embedding
embedding = model(transformed_input)
```



## Pre-trained models
- EdgeFace-s (gamma=0.5): available in [`checkpoints/edgeface_s_gamma_05.pt`](checkpoints/edgeface_s_gamma_05.pt)
- EdgeFace-xs (gamma=0.6): available in [`checkpoints/edgeface_xs_gamma_06.pt`](checkpoints/edgeface_xs_gamma_06.pt)



## Performance
The performance of each model is reported in Table 2 of the [paper](https://arxiv.org/pdf/2307.01838v2.pdf):

![performance](assets/benchmark.png)


## :rocket: New! Using EdgeFace Models via `torch.hub`

### Available Models on `torch.hub`

- `edgeface_base`
- `edgeface_s_gamma_05`
- `edgeface_xs_q`
- `edgeface_xs_gamma_06`
- `edgeface_xxs`
- `edgeface_xxs_q`

**NOTE:** Models with `_q` are quantised and require less storage.

### Loading EdgeFace Models with `torch.hub`

You can load the models using `torch.hub` as follows:

```python
import torch
model = torch.hub.load('otroshi/edgeface', 'edgeface_xs_gamma_06', source='github', pretrained=True)
model.eval()
```

### Performance benchmarks of different variants of EdgeFace

| Model               | MPARAMS| MFLOPs |    LFW(%)    |    CALFW(%)  |   CPLFW(%)   |   CFP-FP(%)  |   AgeDB30(%) |
|:--------------------|-------:|-------:|:-------------|:-------------|:-------------|:-------------|:-------------|
| edgeface_base       |  18.23 |1398.83 | 99.83 ± 0.24 | 96.07 ± 1.03 | 93.75 ± 1.16 | 97.01 ± 0.94 | 97.60 ± 0.70 |
| edgeface_s_gamma_05 |   3.65 | 306.12 | 99.78 ± 0.27 | 95.55 ± 1.05 | 92.48 ± 1.42 | 95.74 ± 1.09 | 97.03 ± 0.85 |
| edgeface_xs_gamma_06|   1.77 | 154.00 | 99.73 ± 0.35 | 95.28 ± 1.37 | 91.58 ± 1.42 | 94.71 ± 1.07 | 96.08 ± 0.95 |
| edgeface_xxs        |   1.24 |  94.72 | 99.57 ± 0.33 | 94.83 ± 0.98 | 90.27 ± 0.93 | 93.63 ± 0.99 | 94.92 ± 1.15 |

## Reference
If you use this repository, please cite the following paper, which is [published](https://ieeexplore.ieee.org/abstract/document/10388036/) in the IEEE Transactions on Biometrics, Behavior, and Identity Science (IEEE T-BIOM). The PDF version of the paper is available as [pre-print on arxiv](https://arxiv.org/pdf/2307.01838v2.pdf). The complete source code for reproducing all experiments in the paper (including training and evaluation) is also publicly available in the [official repository](https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface).


```bibtex
@article{edgeface,
  title={Edgeface: Efficient face recognition model for edge devices},
  author={George, Anjith and Ecabert, Christophe and Shahreza, Hatef Otroshi and Kotwal, Ketan and Marcel, Sebastien},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2024}
}
```
