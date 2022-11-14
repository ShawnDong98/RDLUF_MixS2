from architecture import *
from utils import *
import scipy.io as scio
import torch
import os
import numpy as np
from option import opt

from torch_ema import ExponentialMovingAverage

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Intialize mask
Phi_batch_test = init_mask(
    opt.mask_path, 
    opt.input_mask, 
    10, 
    device=device)

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# model
model = model_generator(opt, device)
ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

if opt.pretrained_model_path:
    print(f"===> Loading Checkpoint from {opt.pretrained_model_path}")
    save_state = torch.load(opt.pretrained_model_path, map_location=device)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])

def test(model):
    test_data = LoadTest(opt.test_path)
    test_gt = test_data.to(torch.float32).to(device)
    input_meas = init_meas(test_gt, Phi_batch_test, opt.input_setting)
    model.eval()
    with torch.no_grad():
        with ema.average_parameters():
            model_out, log_dict = model(input_meas, Phi_batch_test)
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    model.train()
    return pred, truth

def main():
    pred, truth = test(model)
    name = opt.outf + 'Test_result.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'truth': truth, 'pred': pred})

if __name__ == '__main__':
    main()