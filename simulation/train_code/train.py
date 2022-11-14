from collections import defaultdict
from utils import *
seed_everything(
    seed = 3407,
    deterministic = True, 
)
from architecture import *
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid
from torch_ema import ExponentialMovingAverage

import scipy.io as scio
import time
import gc
import os
import numpy as np
from torch.autograd import Variable
import datetime
from option import opt
from tqdm import tqdm
from pprint import pprint
import seaborn as sns

import losses
from schedulers import get_cosine_schedule_with_warmup


pprint(opt)

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

print(os.environ["CUDA_VISIBLE_DEVICES"])

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# init mask
Phi_batch_train = init_mask(
    opt.mask_path, 
    opt.input_mask, 
    opt.batch_size, 
    device=device)
Phi_batch_test = init_mask(
    opt.mask_path, 
    opt.input_mask, 
    1, 
    device=device)

# dataset
train_set = LoadTraining(opt.data_path, debug=False)
test_data = LoadTest(opt.test_path)

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# model
model = model_generator(opt, device=device)

ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))

scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(np.floor(opt.epoch_sam_num / opt.batch_size)), 
    num_training_steps=int(np.floor(opt.epoch_sam_num / opt.batch_size)) * opt.max_epoch, 
    eta_min=1e-6)

start_epoch = 0

if opt.resume_ckpt_path:
    print(f"===> Loading Checkpoint from {opt.resume_ckpt_path}")
    save_state = torch.load(opt.resume_ckpt_path)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])
    optimizer.load_state_dict(save_state['optimizer'])
    scheduler.load_state_dict(save_state['scheduler'])
    start_epoch = save_state['epoch']


criterion = losses.CharbonnierLoss().to(device)

lrs = []

def train(epoch, logger):
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    train_tqdm = tqdm(range(batch_num))
    for i in train_tqdm:
        gt_batch = shuffle_crop(train_set, opt.batch_size)
        gt = Variable(gt_batch).to(device)
        input_meas = init_meas(gt, Phi_batch_train, opt.input_setting)
        model_out, log_dict = model(input_meas, Phi_batch_train)
        loss = criterion(model_out, gt)
        loss.backward()
        if opt.clip_grad:
            clip_grad_norm_(model.parameters(), max_norm=0.2)
        optimizer.step()
        optimizer.zero_grad()
        ema.update()
        train_tqdm.set_postfix(train_loss=f"{loss.item():.4f}")
        epoch_loss += loss.data
        lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()
        
        
    end = time.time()
    train_loss = epoch_loss / batch_num
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, train_loss, (end - begin)))
    return train_loss

def test(epoch, logger):
    psnr_list, ssim_list = [], []
    test_gt = test_data.to(torch.float32).to(device)
    input_meas = init_meas(test_gt, Phi_batch_test, opt.input_setting)
    model.eval()
    begin = time.time()
    image_log = {}
    stages_x_v = defaultdict(list)
    pred = []
    for k in tqdm(range(test_gt.shape[0])):
        with torch.no_grad():
            with ema.average_parameters():
                out, log_dict = model(input_meas[k].unsqueeze(0), Phi_batch_test)
                model_out = out
                pred.append(model_out)
                for i in range(opt.stage):
                    stages_x_v[f'stage{i}_v'].append(log_dict[f'stage{i}_v'])
                    stages_x_v[f'stage{i}_x'].append(log_dict[f'stage{i}_x'])

        psnr_val = torch_psnr(model_out[0, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[0, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    end = time.time()

    for i in range(opt.stage):
        v = torch.cat(stages_x_v[f'stage{i}_v'], dim=0)
        v = torch.mean(v, dim=1, keepdim=True)
        x = torch.mean(torch.cat(stages_x_v[f'stage{i}_x'], dim=0), dim=1, keepdim=True)

        v = make_grid(v, nrows=2) # torch 1.12
        x = make_grid(x, nrows=2) # torch 1.12


    pred = torch.cat(pred, dim = 0)
    pred = np.transpose(pred.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean,(end - begin)))
    model.train()
    
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean, image_log

def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0
    for epoch in range(start_epoch + 1, opt.max_epoch + 1):
        print(f"==>Epoch{epoch}")
        train_loss = train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean, image_log) = test(epoch, logger)
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 28:
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
                checkpoint(model, ema, optimizer, scheduler, epoch, model_path, logger)
        
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()


