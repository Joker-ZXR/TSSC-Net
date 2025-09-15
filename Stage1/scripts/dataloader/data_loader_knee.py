import os
import re
import torch
import decord
import torchvision
import torchvision.transforms as transforms
import numpy as np
import nibabel as nib
from PIL import Image
from einops import rearrange
from typing import Dict, List, Tuple

from .video_transforms import *

class_labels_map = None
cls_sample_cnt = None


def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
            # Filelist.append( filename)
    return Filelist

class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1, **kwargs):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        self.kwargs = kwargs

    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str

class data_loader(torch.utils.data.Dataset):
    """Load the video files

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self, configs, stage):
        self.stage = stage
        self.configs = configs
        self.target_video_len = self.configs.tar_num_frames
        self.v_decoder = DecordInit()

        self.temporal_sample = TemporalRandomCrop(configs.tar_num_frames)
        self.transform_train = transforms.Compose([
                    ToTensorVideo(),
                    CenterCropResizeVideo(configs.image_size),
                    RandomHorizontalFlipVideo(),
                    RandomTemporalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.transform_test = transforms.Compose([
                    ToTensorVideo(),
                    CenterCropResizeVideo(configs.image_size),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])

        if self.stage == 'train' or stage == 'val':
            self.data_path = configs.data_path_train
            len_train = int(0.9 * len(os.listdir(self.data_path)))
            if self.stage == 'train':
                self.video_lists = get_filelist(self.data_path)[:len_train]
                # print(stage, len(self.video_lists), self.video_lists)
            elif self.stage == 'val':
                self.video_lists = get_filelist(self.data_path)[len_train:]
                # print(stage, len(self.video_lists), self.video_lists)
        if self.stage == 'test':
            self.data_path = configs.data_path_test
            self.video_lists = get_filelist(self.data_path)
            # print(stage, len(self.video_lists), self.video_lists)

    def __getitem__(self, index):
        path = self.video_lists[index]
        file_name = os.path.basename(path)

        vframes_gt, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')

        # ==============================================================================================================
        # 解析文件名，提取前缀、数字和扩展名
        # 例如：'20240802-a-t1_mx3d-slice_83.avi'
        pattern = r'(.*?)(\d+)(\.[^.]+)$'
        m = re.match(pattern, file_name)
        if m:
            prefix = m.group(1)  # "20240802-a-t1_mx3d-slice_"
            slice_num = int(m.group(2))  # 83
            ext = m.group(3)  # ".avi"
        else:
            raise ValueError(f"文件名格式不符合要求: {file_name}")

        # 获取path的父目录，用于查找相邻层的视频文件
        parent_dir = os.path.dirname(path)

        # 构造前一层与后一层文件的完整路径
        prev_slice = slice_num - 1
        next_slice = slice_num + 1
        prev_file = os.path.join(parent_dir, f"{prefix}{prev_slice}{ext}")
        next_file = os.path.join(parent_dir, f"{prefix}{next_slice}{ext}")

        # 如果相应的文件不存在，则用当前文件替代
        if not os.path.exists(prev_file):
            # 没有比当前数字小的，拼接的数据应为 [当前, 当前, 下一层]
            prev_file = path
        if not os.path.exists(next_file):
            # 没有比当前数字大的，拼接的数据应为 [上一层, 当前, 当前]
            next_file = path

        # 分别读取三个视频文件
        # 注意：这里读取的视频帧张量形状均为 [T, 3, H, W]（T为帧数）
        vframes_prev, _, _ = torchvision.io.read_video(filename=prev_file, pts_unit='sec', output_format='TCHW')
        vframes_curr, _, _ = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        vframes_next, _, _ = torchvision.io.read_video(filename=next_file, pts_unit='sec', output_format='TCHW')

        # 将 RGB 通道合并成一个通道（转为灰度图像）
        # 采用加权求和转换公式：Gray = 0.299*R + 0.587*G + 0.114*B
        # 注意：这里用切片 [:, i:i+1, :, :] 保证结果形状为 [T, 1, H, W]
        vframes_prev_gray = (0.299 * vframes_prev[:, 0:1, :, :] +
                             0.587 * vframes_prev[:, 1:2, :, :] +
                             0.114 * vframes_prev[:, 2:3, :, :])
        vframes_curr_gray = (0.299 * vframes_curr[:, 0:1, :, :] +
                             0.587 * vframes_curr[:, 1:2, :, :] +
                             0.114 * vframes_curr[:, 2:3, :, :])
        vframes_next_gray = (0.299 * vframes_next[:, 0:1, :, :] +
                             0.587 * vframes_next[:, 1:2, :, :] +
                             0.114 * vframes_next[:, 2:3, :, :])

        # vframes_prev_gray = vframes_prev[:, 0:1, :, :]
        # vframes_curr_gray = vframes_curr[:, 0:1, :, :]
        # vframes_next_gray = vframes_next[:, 0:1, :, :]

        # vframes_prev_gray_ = np.transpose((vframes_prev_gray.detach().cpu().numpy())[:, 0, ], [-1, -2, -3])
        # nib.save(nib.Nifti1Image(vframes_prev_gray_, np.eye(4)), os.path.join(r'/home/I4U/4D_MRI_Generation/xr_Project/Datasets/UIH_Datasets/Knee_Datasets/knee_UIH/Nii/video_pre/knee_video/000', 'prev_{}.nii.gz'.format(prev_slice)))
        #
        # vframes_curr_gray_ = np.transpose((vframes_curr_gray.detach().cpu().numpy())[:, 0, ], [-1, -2, -3])
        # nib.save(nib.Nifti1Image(vframes_curr_gray_, np.eye(4)), os.path.join(r'/home/I4U/4D_MRI_Generation/xr_Project/Datasets/UIH_Datasets/Knee_Datasets/knee_UIH/Nii/video_pre/knee_video/000', 'curr_{}.nii.gz'.format(prev_slice)))
        #
        # vframes_next_gray_ = np.transpose((vframes_next_gray.detach().cpu().numpy())[:, 0, ], [-1, -2, -3])
        # nib.save(nib.Nifti1Image(vframes_next_gray_, np.eye(4)), os.path.join(r'/home/I4U/4D_MRI_Generation/xr_Project/Datasets/UIH_Datasets/Knee_Datasets/knee_UIH/Nii/video_pre/knee_video/000', 'next_{}.nii.gz'.format(prev_slice)))

        # 为了保证三个视频的帧数一致，取最少的帧数
        min_frames = min(vframes_prev_gray.shape[0], vframes_curr_gray.shape[0],vframes_next_gray.shape[0])
        vframes_prev_gray = vframes_prev_gray[:min_frames]
        vframes_curr_gray = vframes_curr_gray[:min_frames]
        vframes_next_gray = vframes_next_gray[:min_frames]

        # 将三个视频拼接为一个新的视频帧张量
        # 新张量形状为 [T, 3, H, W]，其中各通道分别对应上一层、当前层、下一层的灰度数据
        vframes = torch.cat([vframes_prev_gray, vframes_curr_gray, vframes_next_gray], dim=1)
        vframes = vframes.to(torch.uint8)
        total_frames = len(vframes)
        # ==============================================================================================================

        if self.stage == 'train':
            # Sampling video frames with 50% probability
            if random.random() < 0.4:
                # Sample using the full range of frames
                frame_indice = np.linspace(0, total_frames - 1, self.target_video_len, dtype=int)
                # print('total_frames:{}, linspace, {}'.format(total_frames, frame_indice))
            else:
                # Sample using the specified range
                start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
                # assert end_frame_ind - start_frame_ind >= self.target_video_len
                frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)
                # print('total_frames:{}, random, {}'.format(total_frames, frame_indice))
        else:
            if total_frames <= self.target_video_len:
                # 如果帧数不足，重复采样或补全
                frame_indice = np.linspace(0, total_frames - 1, self.target_video_len, dtype=int)
            else:
                # 均匀采样
                frame_indice = np.round(np.linspace(0, total_frames - 1, self.target_video_len)).astype(int)
        # print(total_frames, frame_indice)
        video = vframes[frame_indice]
        video_gt = vframes_gt[frame_indice]

        # 注意：video[0] 的形状为 [3, H, W]，video[0,1:2,:,:] 的形状为 [1, H, W]
        video[0] = video[0, 1:2, :, :].repeat(3, 1, 1)
        video[-1] = video[-1, 1:2, :, :].repeat(3, 1, 1)
        # videotransformer data proprecess

        if self.stage == 'test':
            video = self.transform_test(video)  # T C H W
            video_gt = self.transform_test(video_gt)
        else:
            video = self.transform_train(video)  # T C H W
            video_gt = self.transform_train(video_gt)
        return {'video': video, 'video_name': file_name, 'video_gt': video_gt}


    def __len__(self):
        return len(self.video_lists)


if __name__ == '__main__':
    from tqdm import tqdm
    import imageio
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    config_path = r'../configs/config_knee.yaml'
    config_dict = OmegaConf.load(config_path)
    dataset = data_loader(config_dict, 'train')

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True)

    pbar = tqdm((loader), total=len(loader), desc="\033[1;37mProcessing...]")
    for i, batch in enumerate(pbar):
        video = batch['video']
        print(video.shape)
        print(video.max(), video.min())

