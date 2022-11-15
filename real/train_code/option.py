import argparse
import template

from options import merge_duf_mixs2_opt

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--exp_name', type=str, default="rdluf_mixs2 real", help="name of experiment")
parser.add_argument('--template', default='duf_mixs2',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')
parser.add_argument('--data_path_CAVE', default='../../datasets/cave_1024_28/', type=str,
                        help='path of data')
parser.add_argument('--data_path_KAIST', default='../../datasets/KAIST_CVPR2021/', type=str,
                    help='path of data')
parser.add_argument('--mask_path', default='../../datasets/TSA_real_data/mask_3d_shift.mat', type=str,
                    help='path of mask')
parser.add_argument('--test_data_path', default='../../datasets/TSA_real_data/Measurements/', type=str,
                    help='path of data')
parser.add_argument('--test_mask_path', default='../../datasets/TSA_real_data/mask_3d_shift.mat', type=str,
                    help='path of mask')


# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/rdluf_mixs2_3stage/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='duf_mixs2', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument('--resume_ckpt_path', type=str, default=None, help='resumed checkpoint directory')
parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi',
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask


# Training specifications
parser.add_argument("--height", default=660, type=int, help='cropped patch height')
parser.add_argument("--width", default=660, type=int, help='cropped patch width')
parser.add_argument('--batch_size', type=int, default=1, help='the number of HSIs per batch')
parser.add_argument("--isTrain", default=True, type=bool, help='train or test')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--learning_rate", type=float, default=0.0004)
parser.add_argument("--tune", action='store_true', help='control the max_epoch and milestones')


opt = parser.parse_known_args()[0]

if opt.method == 'duf_mixs2':
    parser = merge_duf_mixs2_opt(parser)

opt = parser.parse_known_args()[0]

template.set_template(opt)

opt.trainset_num = 2500

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False