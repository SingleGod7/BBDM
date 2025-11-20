import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from Register import Registers
from datasets.base import ImagePathDataset
from datasets.utils import get_image_paths_from_dir
from PIL import Image
import numpy as np
import cv2
import os


@Registers.datasets.register_with_name('custom_single')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs = ImagePathDataset(image_paths, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.imgs[i]


@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/fdg'))
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/t1'))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]


@Registers.datasets.register_with_name('custom_colorization_LAB')
class CustomColorizationLABDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        img_path = self.image_paths[index]
        image = None
        try:
            image = cv2.imread(img_path)
            if self.to_lab:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        except BaseException as e:
            print(img_path)

        if p:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1).contiguous()

        if self.to_normal:
            image = (image - 127.5) / 127.5
            image.clamp_(-1., 1.)

        L = image[0:1, :, :]
        ab = image[1:, :, :]
        cond = torch.cat((L, L, L), dim=0)
        return image, cond


@Registers.datasets.register_with_name('custom_colorization_RGB')
class CustomColorizationRGBDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        cond_image = image.convert('L')
        cond_image = cond_image.convert('RGB')

        image = transform(image)
        cond_image = transform(cond_image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)
            cond_image = (cond_image - 0.5) * 2.
            cond_image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


@Registers.datasets.register_with_name('custom_inpainting')
class CustomInpaintingDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.
        if index >= self._length:
            index = index - self._length
            p = 1.

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        height, width = self.image_size
        mask_width = random.randint(128, 180)
        mask_height = random.randint(128, 180)
        mask_pos_x = random.randint(0, height - mask_height)
        mask_pos_y = random.randint(0, width - mask_width)
        mask = torch.ones_like(image)
        mask[:, mask_pos_x:mask_pos_x+mask_height, mask_pos_y:mask_pos_y+mask_width] = 0

        cond_image = image * mask

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)

@Registers.datasets.register_with_name('custom_t12fdg')
class MRI2PETDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        """
        初始化切片数据集

        Args:
            data_dir (str): 包含三维数组数据文件的目录路径
            slice_dimension (int): 切片的维度 (0: 深度, 1: 高度, 2: 宽度)
            transform (callable, optional): 可选的数据增强/预处理方法
        """
        if stage == "val":
            stage = "test"
        self.data_path_A = [os.path.join(dataset_config.dataset_path, stage, 't1', d, f) for d in os.listdir(os.path.join(dataset_config.dataset_path, stage, 't1')) for f in os.listdir(os.path.join(dataset_config.dataset_path, stage, 't1', d)) if f.endswith('.npy')] # 假设数据是 .npy 文件
        self.data_path_B = [os.path.join(dataset_config.dataset_path, stage, 'fdg', d, f) for d in os.listdir(os.path.join(dataset_config.dataset_path, stage, 'fdg')) for f in os.listdir(os.path.join(dataset_config.dataset_path, stage, 'fdg', d)) if f.endswith('.npy')] # 假设数据是 .npy 文件
        sorted(self.data_path_A)
        sorted(self.data_path_B)
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.A = 't1'
        self.B = 'fdg'

    def __len__(self):
        """
        返回数据集的总大小 (总切片数量)
        """
        return len(self.data_path_A)

    def __getitem__(self, index):
        slice_A = self.data_path_A[index]
        slice_B = self.data_path_B[index]

        slice_A = np.load(slice_A)
        slice_B = np.load(slice_B)

        if self.A == "t1":
            slice_A[slice_A < 10] = 0
            slice_A = np.clip(slice_A.astype(np.float32), 0, 500) / 500 * 2 - 1
        elif self.A == "t2":
            # slice_A[slice_A < 10] = 0
            slice_A = np.clip(slice_A.astype(np.float32), 0, 700) / 700 * 2 -1
        else:
            raise ValueError("Invalid modality for A. Expected 't1' or 't2'.")

        slice_B = np.clip(slice_B.astype(np.float32), 0, 1.2) / 1.2 * 2 -1
        
        slice_A = np.expand_dims(slice_A.T, axis=0)  # 添加通道维度 (C, H, W)
        slice_B = np.expand_dims(slice_B.T, axis=0)  # 添加通道维度 (C, H, W)

        # slice_A = np.repeat(slice_A, 3, axis=0)
        # slice_B = np.repeat(slice_B, 3, axis=0)
        return (slice_B, os.path.basename(self.data_path_B[index])), (slice_A, os.path.basename(self.data_path_A[index]))

@Registers.datasets.register_with_name('custom_t12psma')
class MRI2PSMADataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        """
        初始化切片数据集

        Args:
            data_dir (str): 包含三维数组数据文件的目录路径
            slice_dimension (int): 切片的维度 (0: 深度, 1: 高度, 2: 宽度)
            transform (callable, optional): 可选的数据增强/预处理方法
        """
        if stage == "val":
            stage = "test"
        self.data_path_A = [os.path.join(dataset_config.dataset_path, stage, 't1', d, f) for d in os.listdir(os.path.join(dataset_config.dataset_path, stage, 't1')) for f in os.listdir(os.path.join(dataset_config.dataset_path, stage, 't1', d)) if f.endswith('.npy')] # 假设数据是 .npy 文件
        self.data_path_B = [os.path.join(dataset_config.dataset_path, stage, 'psma', d, f) for d in os.listdir(os.path.join(dataset_config.dataset_path, stage, 'psma')) for f in os.listdir(os.path.join(dataset_config.dataset_path, stage, 'psma', d)) if f.endswith('.npy')] # 假设数据是 .npy 文件
        sorted(self.data_path_A)
        sorted(self.data_path_B)
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.A = 't1'
        self.B = 'psma'

    def __len__(self):
        """
        返回数据集的总大小 (总切片数量)
        """
        return len(self.data_path_A)

    def __getitem__(self, index):
        slice_A = self.data_path_A[index]
        slice_B = self.data_path_B[index]

        slice_A = np.load(slice_A)
        slice_B = np.load(slice_B)

        if self.A == "t1":
            slice_A[slice_A < 10] = 0
            slice_A = np.clip(slice_A.astype(np.float32), 0, 500) / 500 * 2 - 1
        elif self.A == "t2":
            # slice_A[slice_A < 10] = 0
            slice_A = np.clip(slice_A.astype(np.float32), 0, 700) / 700 * 2 -1
        else:
            raise ValueError("Invalid modality for A. Expected 't1' or 't2'.")

        slice_B = np.clip(slice_B.astype(np.float32), 0, 1.2) / 1.2 * 2 -1
        
        slice_A = np.expand_dims(slice_A.T, axis=0)  # 添加通道维度 (C, H, W)
        slice_B = np.expand_dims(slice_B.T, axis=0)  # 添加通道维度 (C, H, W)

        # slice_A = np.repeat(slice_A, 3, axis=0)
        # slice_B = np.repeat(slice_B, 3, axis=0)
        return (slice_B, os.path.basename(self.data_path_B[index])), (slice_A, os.path.basename(self.data_path_A[index]))
    
@Registers.datasets.register_with_name('custom_t12dota')
class MRI2DOTADataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        """
        初始化切片数据集

        Args:
            data_dir (str): 包含三维数组数据文件的目录路径
            slice_dimension (int): 切片的维度 (0: 深度, 1: 高度, 2: 宽度)
            transform (callable, optional): 可选的数据增强/预处理方法
        """
        if stage == "val":
            stage = "test"
        self.data_path_A = [os.path.join(dataset_config.dataset_path, stage, 't1', d, f) for d in os.listdir(os.path.join(dataset_config.dataset_path, stage, 't1')) for f in os.listdir(os.path.join(dataset_config.dataset_path, stage, 't1', d)) if f.endswith('.npy')] # 假设数据是 .npy 文件
        self.data_path_B = [os.path.join(dataset_config.dataset_path, stage, 'dota', d, f) for d in os.listdir(os.path.join(dataset_config.dataset_path, stage, 'dota')) for f in os.listdir(os.path.join(dataset_config.dataset_path, stage, 'dota', d)) if f.endswith('.npy')] # 假设数据是 .npy 文件
        sorted(self.data_path_A)
        sorted(self.data_path_B)
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.A = 't1'
        self.B = 'dota'

    def __len__(self):
        """
        返回数据集的总大小 (总切片数量)
        """
        return len(self.data_path_A)

    def __getitem__(self, index):
        slice_A = self.data_path_A[index]
        slice_B = self.data_path_B[index]

        slice_A = np.load(slice_A)
        slice_B = np.load(slice_B)

        if self.A == "t1":
            slice_A[slice_A < 10] = 0
            slice_A = np.clip(slice_A.astype(np.float32), 0, 500) / 500 * 2 - 1
        elif self.A == "t2":
            # slice_A[slice_A < 10] = 0
            slice_A = np.clip(slice_A.astype(np.float32), 0, 700) / 700 * 2 -1
        else:
            raise ValueError("Invalid modality for A. Expected 't1' or 't2'.")

        slice_B = np.clip(slice_B.astype(np.float32), 0, 1.2) / 1.2 * 2 -1
        
        slice_A = np.expand_dims(slice_A.T, axis=0)  # 添加通道维度 (C, H, W)
        slice_B = np.expand_dims(slice_B.T, axis=0)  # 添加通道维度 (C, H, W)

        # slice_A = np.repeat(slice_A, 3, axis=0)
        # slice_B = np.repeat(slice_B, 3, axis=0)
        return (slice_B, os.path.basename(self.data_path_B[index])), (slice_A, os.path.basename(self.data_path_A[index]))

@Registers.datasets.register_with_name('stable_analyse')
class StableTestDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        pass

    def __len__(self):
        """
        返回数据集的总大小 (总切片数量)
        """
        return 16

    def __getitem__(self, index):
        slice_A = r"/data/jiangyizhou/renji/control_fdg_2d/test/t1/PMR000079/PMR000079_0317.npy"
        slice_B = r"/data/jiangyizhou/renji/control_fdg_2d/test/fdg/PMR000079/PMR000079_0317.npy"

        slice_A = np.load(slice_A)
        slice_B = np.load(slice_B)

        slice_A[slice_A < 10] = 0
        slice_A = np.clip(slice_A.astype(np.float32), 0, 500) / 500 * 2 - 1

        slice_B = np.clip(slice_B.astype(np.float32), 0, 1.2) / 1.2 * 2 -1
        
        slice_A = np.expand_dims(slice_A.T, axis=0)  # 添加通道维度 (C, H, W)
        slice_B = np.expand_dims(slice_B.T, axis=0)  # 添加通道维度 (C, H, W)

        # slice_A = np.repeat(slice_A, 3, axis=0)
        # slice_B = np.repeat(slice_B, 3, axis=0)
        return (slice_B, f"test_img_{index}"), (slice_A, os.path.basename(f"test_img_{index}"))