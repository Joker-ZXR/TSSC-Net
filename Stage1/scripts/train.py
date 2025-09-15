import os
import math
import argparse
import numpy as np
import torch
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
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.utils import (clip_grad_norm_, create_logger, update_ema,
                   requires_grad, cleanup, create_tensorboard,
                   write_tensorboard, setup_distributed, get_experiment_dir)
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import warnings
warnings.filterwarnings('ignore')

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id    ##########
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_PORT'] = '6034'
    os.environ['MASTER_ADDR'] = '127.0.0.1'

    # Setup DDP:
    setup_distributed()
    # dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    # local_rank = rank
    rank = int(os.environ["RANK"])
    # local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = rank
    device = torch.device("cuda", int(args.gpu_id))       ##########

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model.replace("/", "-")  # e.g., MVIF-XL/2 --> MVIF-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{model_string_name}_{args.cur_date}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(os.path.join(experiment_dir, 'runs'))
        val_fold = f"{experiment_dir}/val_pic"
        os.makedirs(val_fold, exist_ok=True)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config_acdc.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        tb_writer = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    additional_kwargs = {'num_frames': args.tar_num_frames, 'mode': 'video'}
    model = MVIF_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        **additional_kwargs)

    # Note that parameter initialization is done within the MVIF constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    # diffusion = create_diffusion(timestep_respacing="", diffusion_steps=args.diffusion_steps)  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_path, subfolder="sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)     # Freeze vae and text_encoder

    # # use pretrained model?
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            logger.info('Using ema ckpt!')
            checkpoint = checkpoint["ema"]

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
                logger.info('Successfully Load weights from {}'.format(k))
            # elif 'x_embedder' in k: # replace model parameter name
            #     pretrained_dict['patch_embedder'] = v
            # elif 't_embedder' in k: # replace model parameter name
            #     pretrained_dict['timestep_embedder'] = v
            else:
                logger.info('Ignoring: {}'.format(k))
        logger.info('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('Successfully load model at {}!'.format(args.pretrained))

    if args.use_compile:
        model = torch.compile(model)

    if args.enable_xformers_memory_efficient_attention:
        logger.info("Using Xformers!")
        model.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        logger.info("Using gradient checkpointing!")
        model.enable_gradient_checkpointing()

    # set distributed training
    model = DDP(model.to(device), device_ids=[int(args.gpu_id)], find_unused_parameters=False)
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup training data:
    dataset = data_loader(args, stage='train')
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=int(args.train_batch_size), shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    logger.info(f"Training Dataset contains {int(len(dataset)*0.9):,} videos ({args.data_path_train})")

    # Setup validation data:
    dataset_val = data_loader(args, stage='val')
    sampler_val = DistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=rank, shuffle=False, seed=args.global_seed)
    loader_val = DataLoader(dataset_val, batch_size=int(args.val_batch_size), shuffle=False, sampler=sampler_val, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    logger.info(f"Validation Dataset contains {int(len(dataset)*0.1):,} videos")


    # Scheduler
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(loader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    dirs = os.path.join(experiment_dir, 'checkpoints', 'latest_epoch_train_model.pth')
    if args.resume_from_checkpoint and os.path.exists(dirs):
        # TODO, need to checkout
        # Get the most recent checkpoint
        logger.info(f"Resuming from checkpoint {dirs}")
        checkpoint = torch.load(dirs, map_location=lambda storage, loc: storage)
        # opt.load_state_dict(checkpoint["opt"])
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        train_steps = checkpoint["train_steps"]

        first_epoch = train_steps // num_update_steps_per_epoch
        resume_step = train_steps % num_update_steps_per_epoch
        # 清理显存缓存
        torch.cuda.empty_cache()
    logger.info(f"Training for {first_epoch} epochs...")
    for epoch in range(first_epoch, num_train_epochs):
        # print(epoch)
        sampler.set_epoch(epoch)
        diffusion = create_diffusion(timestep_respacing=args.timestep_respacing_train, diffusion_steps=args.diffusion_steps)
        diffusion.training=True

        pbar = tqdm((loader), total=len(loader), desc="\033[1;37mProcessing...]")
        for step, video_data in enumerate(pbar):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and os.path.exists(dirs) and epoch == first_epoch and step < resume_step:
                continue

            x = video_data['video'].to(device, non_blocking=True)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                b, _, _, _, _ = x.shape
                x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
                latent = rearrange(latent, '(b f) c h w -> b f c h w', b=b).contiguous()

            # Get the frames_mask,1代表mask，0代表不mask
            b, f, c, h, w = latent.shape
            mask = torch.ones(b, f, h, w)
            # 将 f 维度的第一帧和最后一帧设置为 0
            mask[:, 0, :, :] = 0  # 第一帧
            mask[:, -1, :, :] = 0  # 最后一帧
            mask = mask.to(device)

            t = torch.randint(0, diffusion.num_timesteps, (latent.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, latent, t, mask=mask)
            loss = loss_dict["loss"].mean()
            loss.backward()

            # tqdm
            pbar.set_description(f"Epoch [{epoch + 1}/{num_train_epochs}]")
            pbar.set_postfix(train_cdiff_loss=loss.item())

            if train_steps < args.start_clip_iter:  # if train_steps >= start_clip_iter, will clip gradient
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
            else:
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True)

            opt.step()
            lr_scheduler.step()
            opt.zero_grad()
            update_ema(ema, model.module)

            write_tensorboard(tb_writer, 'Train Loss step', loss, train_steps)
            write_tensorboard(tb_writer, 'Gradient Norm step', gradient_norm, train_steps)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                logger.info(
                    f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                write_tensorboard(tb_writer, 'Train Loss log_every', avg_loss, train_steps)
                write_tensorboard(tb_writer, 'Gradient Norm log_every', gradient_norm, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save ckpt_every-checkpoint:
            if train_steps % args.ckpt_every_iter == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        # "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        # "opt": opt.state_dict(),
                        # "args": args,
                        "train_steps": train_steps
                    }

                    checkpoint_path = f"{checkpoint_dir}/iter_{train_steps:06d}.pth"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        # Save latest_epoch
        if rank == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args,
                "train_steps": train_steps
            }

            checkpoint_path = f"{checkpoint_dir}/latest_epoch_train_model.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved latest_epoch_train_model checkpoint to {checkpoint_path}")

        # Validation
        val_pred_epoch_metric = 0
        best_val_pred_epoch_metric = 100
        # This disables randomized embedding dropout do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
        if (epoch+1) % args.val_interval == 0:
            diffusion = create_diffusion(timestep_respacing=args.timestep_respacing_test, diffusion_steps=args.diffusion_steps)
            diffusion.training=False
            model.eval()  # important!
            vae.eval()

            print(f'=======================Validation Phase======================')
            pbar_val = tqdm((loader_val), total=len(loader_val), desc="\033[1;37mProcessing...]")
            with torch.no_grad():
                for step_val, video_data_val in enumerate(pbar_val):
                    x_val_ = video_data_val['video'].to(device, non_blocking=True)    # [6, 12, 3, 256, 256]

                    # Map input images to latent space + normalize latents:
                    b, _, _, _, _ = x_val_.shape
                    x_val = rearrange(x_val_, 'b f c h w -> (b f) c h w').contiguous()
                    latent_val = vae.encode(x_val).latent_dist.sample().mul_(0.18215)
                    latent_val = rearrange(latent_val, '(b f) c h w -> b f c h w', b=b).contiguous()

                    # Get the frames_mask,1代表mask，0代表不mask
                    b, f, c, h, w = latent_val.shape
                    mask_val = torch.ones(b, f, h, w)
                    # 将 f 维度的第一帧和最后一帧设置为 0
                    mask_val[:, 0, :, :] = 0  # 第一帧
                    mask_val[:, -1, :, :] = 0  # 最后一帧
                    mask_val = mask_val.to(device)    # b, f, h ,w

                    sample_fn = model.forward
                    z = torch.randn(latent_val.shape, dtype=x_val.dtype, device=device)
                    z = z.permute(0, 2, 1, 3, 4)      # [b, c, f, h, w]
                    samples = diffusion.p_sample_loop(sample_fn, z.shape, z, clip_denoised=False, progress=False, device=device,raw_x=latent_val.permute(0, 2, 1, 3, 4) , mask=mask_val)   # b, c, f, h, w=[6,4,12,32,32]

                    samples = samples.permute(1, 0, 2, 3, 4) * mask_val + latent_val.permute(2, 0, 1, 3, 4) * (1 - mask_val)   # c, b, f, h, w
                    samples = samples.permute(1, 2, 0, 3, 4)   # b, f, c, h, w
                    samples = rearrange(samples, 'b f c h w -> (b f) c h w') / 0.18215
                    decoded_x = vae.decode(samples).sample
                    decoded_x = rearrange(decoded_x, '(b f) c h w -> b f c h w', b=b).contiguous()

                    gt_image = (x_val_.detach().cpu().numpy()).astype(np.float32)
                    pred_image = (decoded_x.detach().cpu().numpy()).astype(np.float32)
                    # Cal_metric
                    mae_val = np.mean(np.abs(pred_image - gt_image)) / b
                    mse_val = mean_squared_error(pred_image, gt_image) / b
                    psnr_val = peak_signal_noise_ratio(pred_image, gt_image, data_range=2) / b
                    # ssim_val = structural_similarity(pred_image, gt_image, data_range=2) / b
                    logger.info( f"(Epoch [{epoch + 1}/{num_train_epochs}]; MAE: {mae_val:.4f}, MSE: {mse_val:.4f}, PSNR: {psnr_val:.4f}")
                    write_tensorboard(tb_writer, 'MAE', mae_val, epoch)
                    write_tensorboard(tb_writer, 'MSE', mse_val, epoch)
                    write_tensorboard(tb_writer, 'PSNR', psnr_val, epoch)
                    # val_tqdm
                    pbar_val.set_description(f"Validtion: Epoch [{epoch + 1}/{num_train_epochs}]")
                    pbar_val.set_postfix(val_metric={'MAE': mae_val, 'MSE': mse_val, 'PSNR': psnr_val})

                    # save_val_pic
                    mask_val = F.interpolate(mask_val.float(), size=(x_val_.shape[-2], x_val_.shape[-1]), mode='nearest')
                    mask_val = mask_val.unsqueeze(0).repeat(3, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)

                    x_val_mask = x_val_ * (1 - mask_val)
                    val_pic = torch.cat([x_val_[:3, ], x_val_mask[:3, ], decoded_x[:3, ]], dim=1)
                    save_image(val_pic.reshape(-1, val_pic.shape[-3], val_pic.shape[-2], val_pic.shape[-1]), os.path.join(val_fold, f"Epoch_{epoch+1}.png"), nrow=12, normalize=True, value_range=(-1, 1))

                    # save_best_epoch
                    val_pred_epoch_metric += (mae_val + mse_val) / 2
                print(f"(Epoch [{epoch + 1}/{num_train_epochs}]; MAE: {mae_val:.4f}, MSE: {mse_val:.4f}, PSNR: {psnr_val:.4f}")
                val_pred_epoch_metric = val_pred_epoch_metric / (step_val + 1)
                if val_pred_epoch_metric < best_val_pred_epoch_metric:
                    best_val_pred_epoch_metric = val_pred_epoch_metric
                    if rank == 0:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            # "opt": opt.state_dict(),
                            # "args": args,
                            "train_steps": train_steps,
                            "best_val_pred_epoch_metric": best_val_pred_epoch_metric
                        }

                        checkpoint_path = f"{checkpoint_dir}/best_epoch_train_model.pth"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved best_epoch_train_model checkpoint to {checkpoint_path}")
                    dist.barrier()
        print(f'=============================================================')

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train EnDora-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/config_knee.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))
