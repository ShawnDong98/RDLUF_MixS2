import torch
from .duf_mixs2 import DUF_MixS2

def model_generator(opt, device="cuda"):
    if opt.method == 'duf_mixs2':
        model = DUF_MixS2(opt).to(device)
    else:
        print(f'opt.Method {opt.method} is not defined !!!!')
    
    return model