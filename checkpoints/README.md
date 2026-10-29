## :rocket: Using EdgeFace Models via `torch.hub`

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

### License
EdgeFace checkpoints are released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).
