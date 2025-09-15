import os
import time
import pywt
import argparse
from models.resmam_x_y_z import *
from dataloader.dataload import loaddata_file
from monai.losses.perceptual import PerceptualLoss
from monai.losses.ssim_loss import SSIMLoss
from monai.metrics import PSNRMetric, SSIMMetric

import torch
from torch.optim import lr_scheduler
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, Dataset, pad_list_data_collate, list_data_collate, decollate_batch


import warnings
warnings.filterwarnings('ignore')

def warmup_rule(start_epoch):
    # learning rate warmup rule
    if start_epoch < 10:
        return 0.01
    elif start_epoch < 20:
        return 0.1
    else:
        return 1.0
class Lambdalr:  # Lr decay scheduler
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        # print(max(0, epoch + self.offset - self.decay_start_epoch))
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def get_args():
    parser = argparse.ArgumentParser(description='Gen_Mamba pipeline for knee datasets')
    parser.add_argument('--indataset_root', type=str, default='../3_Stage1_results/ACDC/Pred',help='Directory of root')
    parser.add_argument('--Gt_root', type=str, default='../3_Stage1_results/ACDC/GT', help='Directory of root')

    parser.add_argument('--checkpoint_dir', type=str, default='../expr/checkpoints',help='Directory for saving network checkpoints')
    parser.add_argument('--log_files_dir', type=str, default='../expr/log', help='Directory for saving log_files')
    parser.add_argument('--runs_dir', type=str, default='../expr/runs', help='Directory for saving to tensorboards')
    parser.add_argument('--save_result_dir', type=str, default='../expr/result', help='Directory for saving log_files')

    parser.add_argument('--train_new_spacing', default=None, type=float, help='spacing in train direction')
    parser.add_argument('--val_new_spacing', default=None, type=float, help='spacing in valid direction')
    parser.add_argument('--test_new_spacing', default=None, type=float, help='spacing in test direction')
    parser.add_argument('--spatial_size', default=[256, 256, 32], type=float, help='spatial_size')
    parser.add_argument('--crop_size', default=(256, 256, None), type=float, help='crop_size')
    parser.add_argument('--RCP_nsamp', default=1, type=float, help='RandCropByPosNegLabeld-num_samples')

    parser.add_argument("--train_batch_size", type=int, default=1, help="size of the test batches")
    parser.add_argument("--val_batch_size", type=int, default=1, help="size of the valid batches")
    parser.add_argument("--val_sw_batch_size", type=int, default=2, help="sw size of the valid batches")
    parser.add_argument("--test_batch_size", type=int, default=1, help="size of the test batches")
    parser.add_argument("--test_sw_batch_size", type=int, default=2, help="sw size of the test batches")

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm', help='Gradient clipping mode. One of ("norm", "value", "agc")')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='plateau', type=str, metavar='SCHEDULER', help='LR scheduler (default: "step", "plateau", "cosine"')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',  help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT', help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT', help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N', help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0, help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 300)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N', help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay_epochs', type=float, default=20, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')

    parser.add_argument("--resume_epochs", type=int, default=0, help="the numbers of resume epoch")
    parser.add_argument("--val_interval", type=int, default=5, help="how mang epochs every eval")
    parser.add_argument("--best_metric", type=int, default=-1, help="best metric")
    parser.add_argument("--best_metric_epoch", type=int, default=-1, help="best metricepoch")
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.8, metavar='RATE', help='LR decay rate (default: 0.1)')

    parser.add_argument("--num_workers", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument('--gpu_id', type=str, default='1', help='which gpu is used')
    parser.add_argument('--seed', type=int, default=3407,help='Seed for random number generator')

    args = parser.parse_args()
    return args

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)

def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Create checkpoint directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.save_result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_files_dir), exist_ok=True)

    # Load Datasets
    args = get_args()
    data_files = loaddata_file(args, 'train')
    train_files, train_transform = data_files["train"]
    val_files, val_transform = data_files["val"]

    print(f"Total number of training data is {len(train_files)}.")
    dataset_train = CacheDataset(data=train_files, transform=train_transform, cache_rate=1.0, num_workers=12)
    # dataset_train = Dataset(data=train_files, transform=train_transform)
    dataloader_train = DataLoader(dataset_train, batch_size=args.train_batch_size, num_workers=12, shuffle=True, drop_last=True)

    print(f"Total number of validation data is {len(val_files)}.")
    dataset_val = CacheDataset(data=val_files, transform=val_transform, cache_rate=1, num_workers=12)
    dataloader_val = DataLoader(dataset_val, batch_size=args.val_batch_size, num_workers=12, shuffle=False)

    # Create model=======================================================================================================
    device = torch.device("cuda:{}".format(int(args.gpu_id)) if torch.cuda.is_available() else "cpu")

    config_mamba = CONFIGS['Res-Mamba-B_16']
    model = TriMamba(config_mamba, input_dim=1, img_size=256, output_dim=1)

    # Load pretrained models/resume train================================================================================
    if args.resume_epochs != 0:
        print('Load the latest path, resume from {} epoch'.format(args.resume_epochs+1))
        checkpoint = torch.load("%s/latest_model.pth" % (args.checkpoint_dir))
        model.load_state_dict(checkpoint)
    model.to(device)

    # Definition loss function
    loss_mae = L1Loss()
    loss_mse = MSELoss()
    loss_SSIM = SSIMLoss(spatial_dims=3)

    # ==================================================================================================================

    def total_variation_3d(volume):
        # volume shape: [B, C, D, H, W]
        d_diff = volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :]
        h_diff = volume[:, :, :, 1:, :] - volume[:, :, :, :-1, :]
        w_diff = volume[:, :, :, :, 1:] - volume[:, :, :, :, :-1]
        tv_loss = torch.mean(d_diff ** 2) + torch.mean(h_diff ** 2) + torch.mean(w_diff ** 2)
        return tv_loss

    def wavelet_loss_3d(fake_volume, real_volume, wavelet='haar', level=3):
        loss = 0.0
        batch_size = fake_volume.shape[0]

        for i in range(batch_size):
            fake_np = fake_volume[i].squeeze().detach().cpu().numpy()
            real_np = real_volume[i].squeeze().cpu().numpy()

            coeffs_fake = pywt.wavedecn(fake_np, wavelet, level=level)
            coeffs_real = pywt.wavedecn(real_np, wavelet, level=level)

            for (cf, cr) in zip(coeffs_fake[1:], coeffs_real[1:]):
                for key in cf.keys():
                    loss += torch.mean(torch.abs(torch.tensor(cf[key] - cr[key])))

        return loss / batch_size
    # ==================================================================================================================

    # Definition optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, eps=1e-08)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=Lambdalr(args.epochs,  args.resume_epochs, args.decay_epochs).step)

    # start_epoch = 0
    # if args.start_epoch is not None:
    #     # a specified start_epoch will always override the resume epoch
    #     start_epoch = args.start_epoch
    # elif args.resume_epochs != 0:
    #     start_epoch = args.resume_epochs
    # if lr_scheduler is not None and start_epoch > 0:
    #     lr_scheduler.step(start_epoch)

    # ----------------------
    #  Training
    # ----------------------
    print('Start training...')
    # Initialize TensorboardX(net: http://localhost:6006/, command:tensorboard --logdir=runs)
    writer = SummaryWriter(log_dir=args.runs_dir, comment='GenMamba')

    # Initialize log_files
    now = time.strftime("%c")
    log_file = open(os.path.join(args.log_files_dir, 'log.txt'), 'a')
    log_file.write('{}\n'.format(str(args)))
    log_file.write('\n')
    log_file.write('--------------Iter_loss(%s)----------------\n' % now)
    prev_time = time.time()

    epoch_loss_values = list()
    best_metric = args.best_metric
    for epoch in range(args.resume_epochs, args.epochs):
        print("=" * 150)
        model.train()
        epoch_loss = 0

        for idx, batch_data in enumerate(dataloader_train):
            step = idx + 1
            optimizer.zero_grad()
            data = batch_data["image"].to(device)
            input = data[:, 0:1, :]
            gt = data[:, 1:, :]
            outputs = model(input)

            loss_L1 = loss_mae(outputs, gt)
            loss_L2 = loss_mse(outputs, gt)
            loss_ssim = loss_SSIM(outputs, gt)
            loss = (loss_L1 + loss_L2) + loss_ssim * 4 + wavelet_loss_3d(outputs, gt) + total_variation_3d(outputs)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_len = len(train_files) // dataloader_train.batch_size

            if (step == 1) or (step == (epoch_len // 2)):
                print(f"[Epoch {epoch + 1}/{args.epochs}] [Batch {step}/{epoch_len}], [train_loss: {loss.item():.4f}], [Lr: {(optimizer.param_groups[0]['lr'])}]")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
                log_file.write(f"\r[Epoch {epoch + 1}/{args.epochs}] [Batch {step}/{epoch_len}], [train_loss: {loss.item():.4f}], [Lr: {(optimizer.param_groups[0]['lr'])}]")
        c_time = time.time() - prev_time
        prev_time = time.time()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} [Average loss: {epoch_loss:.4f}] [Lr: {(optimizer.param_groups[0]['lr'])}] [This epoch took {c_time} s]")
        log_file.write(f"\rEpoch {epoch + 1} [Average loss: {epoch_loss:.4f}] [Lr: {(optimizer.param_groups[0]['lr'])}][This epoch took {c_time} s]")

        # if lr_scheduler is not None:
        #     # step LR for next epoch
        #     lr_scheduler.step(epoch + 1, best_metric)
        scheduler.step()
        if (epoch + 1) % args.val_interval == 0:
            print(f"==================Validation phase===================")
            model.eval()

            with torch.no_grad():
                val_metrics = []
                psnr_metric = PSNRMetric(max_val=1)
                for val_data in dataloader_val:
                    val_images = val_data["image"].to(device)
                    val_input = val_images[:, 0:1, :]
                    val_gt = val_images[:, 1:, :]
                    val_outputs = sliding_window_inference(val_input, args.spatial_size, args.val_sw_batch_size, model)

                    val_metric = psnr_metric(val_outputs, val_gt)
                    val_metrics.append(val_metric.cpu().detach().numpy())
                metric = np.mean(val_metrics)
                print("Validation mean metric:", metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "{}/best_metric_model.pth".format(args.checkpoint_dir))
                    print("Saved new best metric model")
                    log_file.write("\rSaved new best metric model")
                print("Current epoch: {}, Current mean metric: {:.4f}, Best mean metric: {:.4f} at epoch {}".format(epoch + 1, metric, best_metric, best_metric_epoch))
                log_file.write("\rCurrent epoch: {}, Current mean metric: {:.4f}, Best mean metric: {:.4f} at epoch {}".format(epoch + 1, metric, best_metric, best_metric_epoch))
                writer.add_scalar("val_mean_metric", metric, epoch + 1)
                torch.cuda.empty_cache()
        torch.save(model.state_dict(), "{}/latest_model.pth".format(args.checkpoint_dir))
    print(" ")
    print(f"Train completed, Best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    log_file.close()
    
if __name__ == '__main__':
    args = get_args()
    main(args)