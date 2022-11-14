import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random
from ssim_torch import ssim

def generate_shift_masks(mask_path, batch_size, device):
    mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
    mask_3d_shift = mask['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).to(torch.float32).to(device)
    # Phi_s_batch = torch.sum(Phi_batch**2,1)
    # Phi_s_batch[Phi_s_batch==0] = 1
    # print(Phi_batch.shape, Phi_s_batch.shape)
    return Phi_batch

def LoadTraining(path, debug=False):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    for i in range(len(scene_list) if not debug else 5):
        scene_path = path + scene_list[i]
        scene_num = int(scene_list[i].split('.')[0][5:])
        if scene_num<=205:
            if 'mat' not in scene_path:
                continue
            img_dict = sio.loadmat(scene_path)
            if "img_expand" in img_dict:
                img = img_dict['img_expand'] / 65536.
            elif "img" in img_dict:
                img = img_dict['img'] / 65536.
            img = img.astype(np.float32)
            imgs.append(img)
            print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    return imgs

def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data


# We find that this calculation method is more close to DGSMP's.
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename



def shuffle_crop(train_data, batch_size, crop_size=256, augment=True):
    if augment:
        flag = random.randint(0, 1)
        if flag:
            index = np.random.choice(range(len(train_data)), batch_size)
            processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
            for i in range(batch_size):
                h, w, _ = train_data[index[i]].shape
                x_index = np.random.randint(0, h - crop_size)
                y_index = np.random.randint(0, w - crop_size)
                processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
            gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
            for i in range(gt_batch.shape[0]):
                gt_batch[i] = augment_1(gt_batch[i])
        else:
            gt_batch = []
            processed_data = np.zeros((4, 128, 128, 28), dtype=np.float32)
            for i in range(batch_size):
                sample_list = np.random.randint(0, len(train_data), 4)
                for j in range(4):
                    h, w, _ = train_data[sample_list[j]].shape
                    x_index = np.random.randint(0, h-crop_size//2)
                    y_index = np.random.randint(0, w-crop_size//2)
                    processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]
                generated_sample = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))  # [4,28,128,128]
                gt_batch.append(augment_2(generated_sample))
            gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))

    return gt_batch


def augment_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x

def augment_2(generate_gt):
    c, h, w = generate_gt.shape[1],256,256
    divid_point_h = 128
    divid_point_w = 128
    output_img = torch.zeros(c,h,w)
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img

def shift(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs

def gen_meas_torch(data_batch, Phi_batch):
    [batch_size, nC, H, W] = data_batch.shape
    step = 2
    gt_batch = torch.zeros(batch_size, nC, H, W+step*(nC-1)).to(data_batch.device)
    gt_batch[:,:,:,0:W] = data_batch
    gt_shift_batch = shift(gt_batch)
    meas = torch.sum(Phi_batch*gt_shift_batch, 1)
    meas = meas / nC * 2
    return meas


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def init_mask(mask_path, mask_type, batch_size, device="cuda"):
    if mask_type == 'Phi':
        Phi_batch = generate_shift_masks(mask_path, batch_size, device)
    return Phi_batch

def init_meas(gt, phi, input_setting):
    if input_setting == 'Y':
        input_meas = gen_meas_torch(gt, phi)
    return input_meas


def checkpoint(model, ema, optimizer, scheduler,  epoch, model_path, logger):
    save_dict = {}
    save_dict['model'] = model.state_dict()
    save_dict['ema'] = ema.state_dict()
    save_dict['optimizer'] = optimizer.state_dict()
    save_dict['scheduler'] = scheduler.state_dict()
    save_dict['epoch'] = epoch
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(save_dict, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))



def seed_everything(
    seed = 3407,
    deterministic = False, 
):
    """Set random seed.
    Args:
        seed (int): Seed to be used, default seed 3407, from the paper
        Torch. manual_seed (3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision[J]. arXiv preprint arXiv:2109.08203, 2021.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False