import os
import math
import argparse
import datetime
import numpy as np
import torch
import imageio
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.utils import save_image
from tqdm import tqdm
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from models.model_dit import MVIF_models
from models.diffusion.gaussian_diffusion import create_diffusion
from dataloader.data_loader_knee import data_loader
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from utils.show import *
from utils.utils import load_checkpoint, calculate_metrics
import warnings
warnings.filterwarnings('ignore')


def main(args):
    seed = args.global_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Create  directories
    model_string_name = args.model.replace("/", "-")  # e.g., MVIF-XL/2 --> MVIF-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{model_string_name}_{args.cur_date}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    save_dir = f"{experiment_dir}/results_test"
    os.makedirs(os.path.join(save_dir, 'GT'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'Pred'), exist_ok=True)

    # Create log file
    log_file = os.path.join(experiment_dir, "testing_log.txt")
    f_log = open(log_file, 'a')
    f_log.write('\n')

    # Create model=======================================================================================================
    os.environ["CUDA_VISIBLE_DEVICES"] = args.test_gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Define and load pretrain VAE model
    diffusion = create_diffusion(timestep_respacing=args.timestep_respacing_test, diffusion_steps=args.diffusion_steps)  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_path, subfolder="sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)     # Freeze vae and text_encoder
    print(f"Load vae_pretrain_state_dict from checkpoint: {args.pretrained_vae_model_path}")

    #  Define and load MVIF_models
    checkpoint_path = os.path.join(checkpoint_dir, args.test_ckpt)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    latent_size = args.image_size // 8
    additional_kwargs = {'num_frames': args.tar_num_frames, 'mode': 'video'}
    model = MVIF_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        **additional_kwargs).to(device)

    if "ema" in checkpoint:  # supports checkpoints from train.py
        print('Using ema ckpt!')
        f_log.write('Using ema ckpt!')
        checkpoint = checkpoint["ema"]
    model_dict = model.state_dict()
    # filter out unnecessary keys
    pretrained_dict = {}
    for k, v in checkpoint.items():
        if k in model_dict:
            pretrained_dict[k] = v
            print('Successfully Load weights from {}'.format(k))
        else:
            print('Ignoring: {}'.format(k))
    print('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Successfully load model at {}!'.format(checkpoint_path))

    # Load Datasets======================================================================================================
    dataset_test = data_loader(args, stage='test')
    loader_test = DataLoader(dataset_test, batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(f"Total number of testing data is {len(loader_test)}.")

    # Testing loops--------------------------------------------------------------------------------------
    print("=======================Start Testing======================")
    f_log.write("=======================%s====================== \n" %(datetime.datetime.now()))
    f_log.write("Timestep_respacing: %s \n" % (args.timestep_respacing_test))
    f_log.write("Test ckpt: %s \n"%(args.test_ckpt))
    f_log.write("Data_path_test: %s \n"%(args.data_path_test))

    vae.eval()
    model.eval()
    diffusion.training = False

    pbar = tqdm((loader_test), total=len(loader_test), desc="\033[1;37mProcessing...]")
    # for i, batch in enumerate(pbar):
    #     video = batch['video']
    #     name = batch['video_name']

    for idx, video_data_test in enumerate(loader_test):
        with torch.no_grad():
            x_test = video_data_test['video'].to(device)
            x_test_name = video_data_test['video_name']

            numpy_array = ((x_test.squeeze(0) * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
            show_frames(numpy_array)

            print(idx, x_test_name)
            # print(x_test.shape, type(x_test))
            x_test_intp = torch.zeros_like(x_test)
            x_test_intp[:, 0, :, :, :] = x_test[:, 0, :, :, :]
            x_test_intp[:, -1, :, :, :] = x_test[:, -1, :, :, :]

            b, _, _, _, _ = x_test_intp.shape
            x_ = rearrange(x_test_intp, 'b f c h w -> (b f) c h w').contiguous()
            latent = vae.encode(x_).latent_dist.sample().mul_(0.18215)
            latent = rearrange(latent, '(b f) c h w -> b f c h w', b=b).contiguous()

            # Get the frames_mask,1代表mask，0代表不mask
            b, f, c, h, w = latent.shape
            mask = torch.ones(b, f, h, w)
            # 将 f 维度的第一帧和最后一帧设置为 0
            mask[:, 0, :, :] = 0  # 第一帧
            mask[:, -1, :, :] = 0  # 最后一帧
            mask = mask.to(device)  # b, f, h ,w

            sample_fn = model.forward
            z = torch.randn(latent.shape, dtype=x_.dtype, device=device)
            z = z.permute(0, 2, 1, 3, 4)  # [b, c, f, h, w]
            samples = diffusion.p_sample_loop(sample_fn, z.shape, z, clip_denoised=False, progress=False, device=device, raw_x=latent.permute(0, 2, 1, 3, 4), mask=mask)  # b, c, f, h, w=[6,4,12,32,32]

            samples = samples.permute(1, 0, 2, 3, 4) * mask + latent.permute(2, 0, 1, 3, 4) * (1 - mask)  # c, b, f, h, w
            samples = samples.permute(1, 2, 0, 3, 4)  # b, f, c, h, w
            samples = rearrange(samples, 'b f c h w -> (b f) c h w') / 0.18215
            decoded_x = vae.decode(samples).sample
            decoded_x = rearrange(decoded_x, '(b f) c h w -> b f c h w', b=b).contiguous()

            x_test[:, :, 0, :, :] = x_test[:, :, 1, :, :]
            x_test[:, :, 2, :, :] = x_test[:, :, 1, :, :]

            decoded_x[:, :, 0, :, :] = decoded_x[:, :, 1, :, :]
            decoded_x[:, :, 2, :, :] = decoded_x[:, :, 1, :, :]

            # save video
            for i, gt_images in enumerate(x_test):
                # print(decoded_x.shape, "!!!")
                gt_image = ((x_test[i] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
                pred_image = ((decoded_x[i] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
                video_gt_save_path = os.path.join(save_dir, 'GT', x_test_name[i])
                video_pred_save_path = os.path.join(save_dir, 'Pred', x_test_name[i])

                # print(video_pred_save_path, pred_image.shape)
                imageio.mimwrite(video_pred_save_path, pred_image, fps=12,  codec='libx264')
                imageio.mimwrite(video_gt_save_path, gt_image, fps=12,  codec='libx264')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../scripts/configs/config_acdc.yaml")  # Knee: config_knee.yaml; ACDC: config_acdc.yaml
    args = parser.parse_args()
    main(OmegaConf.load(args.config))