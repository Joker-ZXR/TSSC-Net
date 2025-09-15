import cv2
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# 获取第一帧和最后一帧
def show_frames(frames):
    first_frame = frames[0, :]
    last_frame = frames[-1, :]

    # 创建子图并展示
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].set_title("First Frame")
    axes[0].axis('off')  # 去掉坐标轴
    axes[0].set_xticks([])              # 不显示x刻度
    axes[0].set_yticks([])              # 不显示y刻度
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].spines['left'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].imshow(first_frame)


    axes[1].set_title("Last Frame")
    axes[1].axis('off')  # 去掉坐标轴
    axes[1].axis('off')  # 去掉坐标轴
    axes[1].set_xticks([])              # 不显示x刻度
    axes[1].set_yticks([])              # 不显示y刻度
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].imshow(last_frame)
    plt.tight_layout()
    plt.show()

# ======================================================================================================================================================================
import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import re
from natsort import natsorted

def numeric_sort_key(filename):
    """
    从文件名中提取帧号，用于自然排序。
    假设文件名格式为: patient101-1.nii.gz, patient101-2.nii.gz, ..., patient101-12.nii.gz
    """
    base = os.path.basename(filename)
    match = re.search(r'-(\d+)\.nii', base)
    if match:
        return int(match.group(1))
    return 999999


def load_method_data(method_path, patient_name, crop_size=128):
    """
    加载某个方法下指定患者的数据，并对每个 3D 数据在 X 和 Y 方向进行中心裁剪（crop_size x crop_size），
    保持所有 Z 层不变。

    参数：
        method_path: 方法的根目录（例如 'result/TM' 或 'result/method1'）
        patient_name: 患者文件夹名称（例如 'Patient101'）
        crop_size: 裁剪尺寸（例如 128）

    返回：
        形状为 (12, crop_size, crop_size, Z) 的 4D numpy 数组
    """
    patient_folder = os.path.join(method_path, patient_name)
    nii_files = glob.glob(os.path.join(patient_folder, '*.nii.gz'))
    # 使用数字排序保证顺序为 1,2,3,...,10,11,12
    nii_files.sort(key=numeric_sort_key)

    if len(nii_files) != 12:
        print(f"Warning: {patient_folder} 下的 nii 文件数({len(nii_files)})不等于 12")
        return None

    frames_data = []
    for f in nii_files:
        img = nib.load(f)
        data = img.get_fdata()  # 假设数据尺寸为 (256,256,Z)
        data = np.flip(data, axis=0)
        data = np.flip(data, axis=2)
        # data = np.flip(data, axis=0)
        # data = np.transpose(data, (1,0,2))
        # 裁剪中心区域：X、Y方向裁剪到 crop_size x crop_size，Z 保持不变
        x_center = data.shape[0] // 2
        y_center = data.shape[1] // 2
        x_start = x_center - crop_size // 2
        x_end = x_start + crop_size
        y_start = y_center - crop_size // 2
        y_end = y_start + crop_size
        data_cropped = data[x_start:x_end, y_start:y_end, :]
        frames_data.append(data_cropped)

    # 组合为 4D 数组：形状 (12, crop_size, crop_size, Z)
    # stacked = np.stack(frames_data, axis=0)
    return np.stack(frames_data, axis=0)


def visualize_patient_comparison(patient_name, method_paths, output_dir, slice_indices=None, crop_size=128):
    """
    对某个患者进行多方法结果对比展示。

    参数：
        patient_name: 患者文件夹名称（例如 'Patient101'）
        method_paths: 一个列表，每个元素是某个方法的根目录。列表第一个元素作为输入（第一行），
                      后续为不同的对比方法（第二行、第三行……）
        output_dir: 保存结果图像的根目录
        slice_indices: 指定展示的切片索引列表（沿 Z 轴）；若为 None，则遍历所有切片
        crop_size: 裁剪尺寸（例如 128）

    每个生成的图像中，列数为 12（对应 12 帧），行数为方法数量。
    第一行（输入）只显示第 1 帧和第 12 帧，其余帧以灰色填充；其它行显示所有帧。
    """
    # 对每个方法，加载该患者的数据
    method_data_list = []
    for path in method_paths:
        data = load_method_data(path, patient_name, crop_size)
        if data is None:
            print(f"跳过 {patient_name}，因为在 {path} 下数据加载失败。")
            return
        method_data_list.append(data)

    num_methods = len(method_data_list)
    # 获取切片数（假设所有方法数据在 Z 方向层数相同）
    num_slices = method_data_list[0].shape[-1]
    if slice_indices is None:
        slice_indices = range(num_slices)

    # 为该患者创建输出文件夹
    patient_output_dir = os.path.join(output_dir, patient_name)
    os.makedirs(patient_output_dir, exist_ok=True)

    # 定义行标签
    row_labels = ['GroundTruth', 'Input', 'Prediction']

    # 遍历每个切片生成对比图
    for slice_idx in slice_indices:
        # 创建画布：行数 = 方法数，列数 = 12 帧
        fig, axes = plt.subplots(num_methods, 12, figsize=(12,  num_methods))
        fig.patch.set_facecolor('black')  # 整个画布背景设为黑色

        # 若只有一行，确保 axes 为二维数组
        if num_methods == 1:
            axes = np.expand_dims(axes, axis=0)

        # 对每个方法（每一行）进行处理
        for row in range(num_methods):
            # 提取当前方法下该切片的 12 帧图像，列表长度 12，每个图像尺寸 (crop_size, crop_size)
            slice_images = [method_data_list[row][frame, :, :, slice_idx] for frame in range(12)]
            # 对当前方法行，统一归一化：计算全局最小和最大值
            global_min = min(np.min(img) for img in slice_images)
            global_max = max(np.max(img) for img in slice_images)

            # 对当前行的每一列（帧）进行显示
            for col in range(12):
                ax = axes[row, col]
                ax.axis('off')
                # 如果是输入方法（第一行），只显示第一帧和第12帧，其他帧用灰色填充
                if row == 1:
                    if col == 0 or col == 11:
                        ax.imshow(slice_images[col].T, cmap='gray', origin='lower',
                                  vmin=global_min, vmax=global_max)
                    else:
                        gray_val = (global_min + global_max) / 2
                        gray_img = np.full_like(slice_images[col], gray_val)
                        ax.imshow(gray_img.T, cmap='gray', origin='lower',
                                  vmin=global_min, vmax=global_max)
                else:
                    # 对于其它方法，显示全部帧
                    ax.imshow(slice_images[col].T, cmap='gray', origin='lower',
                              vmin=global_min, vmax=global_max)
                # # 在第一列添加行标签
                # if col == 0 and row < len(row_labels):
                #     ax.text(-0.1, 0.5, row_labels[row], transform=ax.transAxes,
                #             fontsize=12, color='white', ha='right', va='center',
                #             rotation=90, weight='bold')

        # 设置子图间隔：wspace, hspace 设得极小，以形成细细的黑色缝隙
        plt.subplots_adjust(wspace=0, hspace=-0.001)   # wspace=0.002, hspace=0)

        # 保存图像，确保背景为黑色
        out_name = f"{patient_name}_slice{slice_idx:03d}.png"
        out_path = os.path.join(patient_output_dir, out_name)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close(fig)


def main(method_paths, output_base_dir):

    os.makedirs(output_base_dir, exist_ok=True)

    # 假设各方法下患者文件夹名称一致，遍历输入数据路径下的所有 Patient* 文件夹
    patient_dirs = glob.glob(os.path.join(method_paths[0], '*'))
    patient_names = [os.path.basename(p) for p in patient_dirs]
    patient_names = natsorted(patient_names)
    # 对每个患者生成对比图
    for patient_name in patient_names:
        print(patient_name)
        visualize_patient_comparison(patient_name, method_paths, output_base_dir, slice_indices=None, crop_size=256)


if __name__ == '__main__':
    # 请在下面填写各个方法对应的数据路径，列表第一个为输入
    method_paths = [
        '../Result/Cardiac/nifti/GT/',  # 输入数据路径
        '../Result/Cardiac/nifti/GT/',
        '../Result/Cardiac/nifti/Pred/'
        # 可继续添加更多方法的路径
    ]
    output_base_dir = '../Result/Cardiac/vis_res'
    main(method_paths, output_base_dir)
