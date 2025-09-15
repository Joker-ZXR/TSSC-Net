import os
import time
import nibabel as nib
from dataloader.dataload import loaddata_file

from models.resmam_x_y_z import *
from train import get_args

import torch
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, Dataset, pad_list_data_collate, list_data_collate, decollate_batch

import warnings
warnings.filterwarnings('ignore')

def test(args):
    start = time.time()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.backends.cudnn.benchmark = True

    stage = 'test'
    ori_test_dir = os.path.join(args.indataset_root, "imagesTs")
    os.makedirs(os.path.join(args.save_result_dir, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(args.save_result_dir, 'gt'), exist_ok=True)

    # Load Datasets
    data_files = loaddata_file(args, stage)
    test_files, test_transform = data_files["test"]
    test_ds = CacheDataset(data=test_files, transform=test_transform, cache_rate=1.0, num_workers=12)
    # test_ds = Dataset(data=test_files, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=12, collate_fn=pad_list_data_collate)

    # Create model=======================================================================================================
    device = torch.device("cuda:{}".format(int(args.gpu_id)) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    config_mamba = CONFIGS['Res-Mamba-B_16']
    model = TriMamba(config_mamba, input_dim=1, img_size=256, output_dim=1)

    # ===================================================================================================================

    checkpoint_dir = r'{}/best_metric_model.pth'.format(args.checkpoint_dir)
    # checkpoint_dir = r'{}/latest_model.pth'.format(args.checkpoint_dir)
    checkpoint = torch.load(checkpoint_dir, map_location='cpu')
    # model = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint)
    print('Parameters loading successful!')
    model.to(device)
    model.eval()

    print('Start testing...')
    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            data = batch_data['image'].to(device)

            input = data[:, 0:1, :]
            file_path = input.meta['filename_or_obj'][0]
            print(idx, file_path)
            outputs = sliding_window_inference(input, args.spatial_size, args.val_sw_batch_size, model, mode='gaussian')
            # outputs = model(input)
            output = outputs[0, 0, :, :, :].cpu().detach().numpy()
            gt = data[:, 1:, :][0, 0, :, :, :].cpu().detach().numpy()

            nii = nib.load(file_path)
            nib.save(nib.Nifti1Image(output, affine=nii.affine, header=nii.header), os.path.join(args.save_result_dir, 'pred', file_path.split('/')[-1]))
            nib.save(nib.Nifti1Image(gt, affine=nii.affine, header=nii.header), os.path.join(args.save_result_dir, 'gt', file_path.split('/')[-1]))

        final = time.time()
    times = final - start
    print("Time:", times)

if __name__ == '__main__':
    args = get_args()
    test(args)