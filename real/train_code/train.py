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
from torchvision.utils import make_grid

import time
import datetime
from torch.autograd import Variable
import os
from option import opt
from torch_ema import ExponentialMovingAverage

from tqdm import tqdm

import losses
from schedulers import get_cosine_schedule_with_warmup
from pprint import pprint




pprint(opt)

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

HR_HSI_test = prepare_data(opt.test_data_path, 5, height=opt.height)
mask_3d_shift_test, mask_3d_shift_s_test = load_mask(opt.test_mask_path)


# load training data
CAVE = prepare_data_cave(opt.data_path_CAVE, 205)
KAIST = prepare_data_KAIST(opt.data_path_KAIST, 30)

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = os.path.join(opt.outf, date_time)
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

logger = gen_log(opt.outf)

# model
model = model_generator(opt, device=device)
ema = ExponentialMovingAverage(model.parameters(), decay=0.999)


# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))

scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(np.floor(opt.trainset_num / opt.batch_size)), 
    num_training_steps=int(np.floor(opt.trainset_num / opt.batch_size)) * opt.max_epoch, 
    eta_min=1e-6)


start_epoch = 0

if opt.resume_ckpt_path:
    logger.info(f"===> Loading Checkpoint from {opt.resume_ckpt_path}")
    save_state = torch.load(opt.resume_ckpt_path)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])
    optimizer.load_state_dict(save_state['optimizer'])
    scheduler.load_state_dict(save_state['scheduler'])
    start_epoch = save_state['epoch']

criterion = losses.CharbonnierLoss().to(device)

def train():
    model.train()
    Dataset = dataset(opt, CAVE, KAIST)
    loader_train = tud.DataLoader(Dataset, num_workers=0, batch_size=opt.batch_size, shuffle=True)
    epoch_loss = 0

    start_time = time.time()
    for i, (input, label, Phi) in enumerate(loader_train):
        input, label, Phi,  = Variable(input), Variable(label), Variable(Phi)
        input, label, Phi = input.to(device), label.to(device), Phi.to(device)

        out, log_dict = model(input, Phi)
        loss = criterion(out, label)

        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        ema.update()
        scheduler.step()

        if i % (50) == 0:
            logger.info('%4d %4d / %4d loss = %.10f time = %s' % (
                epoch + 1, i, len(Dataset) // opt.batch_size, epoch_loss / ((i + 1) * opt.batch_size),
                datetime.datetime.now()))

    elapsed_time = time.time() - start_time
    epoch_loss = epoch_loss / len(Dataset)
    logger.info('epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, epoch_loss, elapsed_time))
    checkpoint(model, ema, optimizer, scheduler, epoch, opt.outf, logger)
    return epoch_loss

def test():
    image_log = {}
    for j in tqdm(range(5)):
        with torch.no_grad():
            meas = HR_HSI_test[:,:,j]
            meas = meas / (meas.max() + 1e-7) * 0.9
            meas = torch.FloatTensor(meas)
            # meas = torch.FloatTensor(meas).unsqueeze(2).permute(2, 0, 1)
            input = meas.unsqueeze(0)
            input = Variable(input)
            input = input.to(device)
            mask_3d_shift_test_ = mask_3d_shift_test.to(device)
            with ema.average_parameters():
                out, log_dict = model(input, mask_3d_shift_test_)
            result = out
            result = result.clamp(min=0., max=1.)
        result = result.squeeze()
        result = result.unsqueeze(1)
        result = torch.flip(result, [2, 3])
        result = make_grid(result, nrows=4) # torch 1.12

        image_log[f'scene{j}'] = [result]

    return image_log


if __name__ == "__main__":
    ## pipline of training
    for epoch in range(start_epoch+1, opt.max_epoch):
        # train_loss = train()
        image_log = test()
        torch.cuda.empty_cache()