import os
import torch
import decord
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

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
        self.transform = transforms.Compose([
                    ToTensorVideo(),
                    CenterCropResizeVideo(configs.image_size),
                    # RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])

        if self.stage == 'train' or self.stage == 'val':
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
        vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        total_frames = len(vframes)

        if self.stage == 'train':
            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            # assert end_frame_ind - start_frame_ind >= self.target_video_len
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)
        else:
            if total_frames <= self.target_video_len:
                # 如果帧数不足，重复采样或补全
                frame_indice = np.linspace(0, total_frames - 1, self.target_video_len, dtype=int)
            else:
                # 均匀采样
                frame_indice = np.round(np.linspace(0, total_frames - 1, self.target_video_len)).astype(int)
        # print(total_frames, frame_indice)
        video = vframes[frame_indice]
        # videotransformer data proprecess
        video = self.transform(video)  # T C H W
        return {'video': video, 'video_name': file_name}

    def __len__(self):
        return len(self.video_lists)


if __name__ == '__main__':
    from tqdm import tqdm
    import imageio
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    config_path = r'../configs/config_acdc.yaml'
    config_dict = OmegaConf.load(config_path)
    dataset = data_loader(config_dict, 'val')

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
        print(video.shape, video.min(), video.max())
        video_ = ((video[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        print(video_.shape)
