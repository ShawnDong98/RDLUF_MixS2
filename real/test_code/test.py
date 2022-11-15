from architecture import *
from utils import *
seed_everything(
    seed = 3407,
    deterministic = True, 
)
from dataset import dataset
import torch.utils.data as tud
import torch
import torch.nn.functional as F
import time
import datetime
from torch.autograd import Variable
import os
from option import opt
from torch_ema import ExponentialMovingAverage

import losses
from schedulers import get_cosine_schedule_with_warmup

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(path, file_num, height=660):
    HR_HSI = np.zeros((((height,714,file_num))))
    for idx in range(file_num):
        ####  read HrHSI
        path1 = os.path.join(path) + 'scene' + str(idx+1) + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:,:,idx] = data['meas_real'][:height, :]
        HR_HSI[HR_HSI < 0] = 0.0
        HR_HSI[HR_HSI > 1] = 1.0
    return HR_HSI

def load_mask(path):
    ## load mask
    data = sio.loadmat(path)
    mask_3d_shift = data['mask_3d_shift']
    mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=2, keepdims=False)
    mask_3d_shift_s[mask_3d_shift_s == 0] = 1
    mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
    mask_3d_shift_s = torch.FloatTensor(mask_3d_shift_s.copy())
    return mask_3d_shift.unsqueeze(0), mask_3d_shift_s.unsqueeze(0)


HR_HSI = prepare_data(opt.data_path, 5, height=opt.height)
mask_3d_shift, mask_3d_shift_s = load_mask(opt.mask_path)

# model
model = model_generator(opt, device=device)
ema = ExponentialMovingAverage(model.parameters(), decay=0.999)


print(f"===> Loading Checkpoint from {opt.pretrained_model_path}")
save_state = torch.load(opt.pretrained_model_path, map_location=device)
model.load_state_dict(save_state['model'])
ema.load_state_dict(save_state['ema'])

k = 0
save_path = './Results/'
res = []
for j in range(5):
    with torch.no_grad():
        meas = HR_HSI[:,:,j]
        # meas = meas / meas.max() * 0.8
        meas = meas / (meas.max() + 1e-7) * 0.9
        # print(f"meas max: {np.max(meas)} min: {np.min(meas)} mean: {np.mean(meas)}")
        meas = torch.FloatTensor(meas)
        # meas = torch.FloatTensor(meas).unsqueeze(2).permute(2, 0, 1)
        input = meas.unsqueeze(0)
        input = Variable(input)
        input = input.to(device)
        mask_3d_shift = mask_3d_shift.to(device)
        mask_3d_shift_s = mask_3d_shift_s.to(device)
        with ema.average_parameters():
            out, log_dict = model(input, mask_3d_shift)
        result = out
        result = result.clamp(min=0., max=1.)
    k = k + 1
    
    res.append(result.cpu().permute(0,2,3,1).numpy())

if not os.path.exists(save_path):  # Create the model directory if it doesn't exist
    os.makedirs(save_path)
save_file = save_path + f'output.mat'
res = np.concatenate(res, axis=0)
print(res.shape)
sio.savemat(save_file, {'pred':res})