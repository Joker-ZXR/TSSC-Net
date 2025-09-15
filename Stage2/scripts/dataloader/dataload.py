import numpy as np
import tempfile
import shutil
import os
import glob
import json
from torch.utils.data import DataLoader
import monai.transforms as mt



def loaddata_file(args, stage):
    if stage == 'train':
        images = sorted(glob.glob(os.path.join(args.indataset_root, "imagesTr", "*.nii.gz")))
        data_dicts = [{"image": [image_name, image_name.replace("Pred", "GT")], "class": "mri"} for image_name in images]
        # len_train = int(0.9 * len(data_dicts))
        # train_files, val_files = data_dicts[:len_train], data_dicts[len_train:]
        train_files =data_dicts

        images = sorted(glob.glob(os.path.join(args.indataset_root, "imagesTs", "*.nii.gz")))
        val_files = [{"image": [image_name, image_name.replace("Pred", "GT")], "class": "mri"} for image_name in images]

        train_transform = mt.Compose(
            [
                mt.LoadImaged(keys="image"),
                mt.EnsureChannelFirstD(keys="image"),
                # mt.OrientationD(keys="image", axcodes="RAS"),
                # mt.SpacingD(keys="image", pixdim=args.train_new_spacing, mode=("bilinear")),
                mt.ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
                mt.CenterSpatialCropd(keys="image", roi_size=args.crop_size),
                mt.RandSpatialCropd(keys="image", roi_size=args.spatial_size, random_size=False),
                mt.RandRotate90d(keys="image", prob=0.5, max_k=3),
                mt.RandFlipd(keys="image", spatial_axis=-1, prob=0.5),
                mt.RandRotate90d(keys="image", prob=0.5, max_k=3),
                mt.RandZoomd(keys="image", min_zoom=0.9, max_zoom=1.1, prob=0.5),
                # mt.RandAffineD(
                #     keys="image",
                #     spatial_size=(-1, -1, -1),
                #     rotate_range=(0, 0, np.pi / 2),
                #     # scale_range=(0.1, 0.1),
                #     mode=("bilinear"),
                #     prob=0.5),
                mt.ToTensorD(keys=["image"]),
            ])

        val_transform = mt.Compose(
            [
                mt.LoadImageD(keys="image"),
                mt.EnsureChannelFirstD(keys="image"),
                # mt.OrientationD(keys="image", axcodes="RAS"),
                # mt.SpacingD(keys="image", pixdim=args.val_new_spacing, mode=("bilinear")),
                mt.ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
                # mt.CenterSpatialCropd(keys="image", roi_size=args.crop_size),
                # mt.RandSpatialCropd(keys="image", roi_size=args.spatial_size, random_size=False),
                mt.ToTensorD(keys="image"),
            ])
        return {"train": [train_files, train_transform], "val": [val_files, val_transform]}

    elif stage == 'test':
        images = sorted(glob.glob(os.path.join(args.indataset_root, "imagesTs", "*.nii.gz")))
        test_files = [{"image": [image_name, image_name.replace("Pred", "GT")], "class": "mri"} for image_name in images]
        # test_files = [{"image": image_name, "class": "mri"} for image_name in images]
        test_transform = mt.Compose(
            [
                mt.LoadImageD(keys="image"),
                mt.EnsureChannelFirstD(keys="image"),
                # mt.OrientationD(keys="image", axcodes="RAS"),
                # mt.SpacingD(keys="image", pixdim=args.test_new_spacing, mode=("bilinear")),
                mt.ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
                # mt.CenterSpatialCropd(keys="image", roi_size=args.crop_size),
                mt.ToTensorD(keys="image"),
            ])

        return {"test": [test_files, test_transform]}


