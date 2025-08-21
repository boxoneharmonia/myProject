import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as T
from PIL import Image
import os
import glob
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class EventSequenceDataset(Dataset):
    def __init__(self, config, transform=None, is_train=True):
        if is_train:
            if config.task in ['traj','traj_v2']:
                self.root_dir = config.train_root
            elif config.task in ['mlm','mlm_v2']:
                self.root_dir = config.pretrain_root
        else:
            self.root_dir = config.test_root

        self.sequence_length = config.max_seq_len
        self.transform = transform
        self.transform_base = transforms.Compose([
            transforms.ToTensor(), 
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])      

        self.samples = []
        self.data =[]
        sequence_folders = sorted(glob.glob(os.path.join(self.root_dir, '*/')))

        for folder in sequence_folders:
            img_dir = os.path.join(folder, 'img')
            data_dir = os.path.join(folder, 'data')
            pos_image_paths = sorted(glob.glob(os.path.join(img_dir, '*_pos.png')))
            num_images = len(pos_image_paths)
            for start_idx in range(num_images - self.sequence_length + 1):
                self.samples.append((img_dir, start_idx))
                self.data.append((data_dir))

        logger.info(f"Dataset initialized with {len(self.samples)} sequences from {self.root_dir}.")

        self.is_train = is_train
        self.task = config.task

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_dir, start_idx = self.samples[idx]
        data_dir = self.data[idx]
        pos_images = []
        neg_images = []
        traj_list = []
        if self.transform is not None:
            angle = torch.randint(0, 4, (1,)).item() * 90
            flip = torch.randint(0, 3, (1,)).item()
            size_ratio = random.uniform(0.6, 1.0)
            top_ratio = random.random()
            left_ratio = random.random()
            transform_kwargs = {
                'size_ratio': size_ratio,
                'top_ratio': top_ratio,
                'left_ratio': left_ratio,
                'angle': angle,
                'flip': flip
            }
        if self.task in ['traj','traj_v2']:
            csv_path = os.path.join(data_dir, 'trajectory.csv')
            df = pd.read_csv(csv_path)

        for i in range(self.sequence_length):
            frame_idx = start_idx + i
            frame_name = str(frame_idx).zfill(4)
            pos_image_path = os.path.join(img_dir, f'{frame_name}_pos.png')
            neg_image_path = os.path.join(img_dir, f'{frame_name}_neg.png')
            pos_image = Image.open(pos_image_path).convert('L')
            neg_image = Image.open(neg_image_path).convert('L')
            if self.transform is not None:
                pos_image = self.transform(pos_image, **transform_kwargs)
                neg_image = self.transform(neg_image, **transform_kwargs)
            else:
                pos_image = self.transform_base(pos_image)
                neg_image = self.transform_base(neg_image)
            pos_images.append(pos_image)
            neg_images.append(neg_image)

            if self.task in ['traj','traj_v2']:
                data = df.iloc[frame_idx].to_numpy()
                data_tensor = torch.tensor(data).float()
                traj_list.append(data_tensor)

        # revert = torch.rand(1) < 0.5
        x_pos_seq = torch.stack(pos_images) 
        x_neg_seq = torch.stack(neg_images)        
        x_seq = torch.cat([x_pos_seq, x_neg_seq], dim=1)  # (S, 2, 200, 200)

        if self.task in ['traj','traj_v2']:
            traj_seq = torch.stack(traj_list)
            return x_seq, traj_seq
        elif self.task in ['mlm','mlm_v2']:
            return x_seq    
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *args, **kwargs):
        for t in self.transforms:
            image = t(image, *args, **kwargs)
        return image

class ToTensor(object):
    ''' Same as torchvision.ToTensor(), but passes extra arguments. '''
    def __call__(self, image, *args, **kwargs):
        # float tensor [0, 1]
        return T.to_tensor(image)
    
class Rotate(object):
    def __call__(self, image, *args, **kwargs):
        angle = kwargs.get('angle', 0)  # Default angle is 0 if not provided
        image = T.rotate(image, angle)
        return image
    
class Flip(object):
    def __call__(self, image, *args, **kwargs):
        flip = kwargs.get('flip', 0)
        if flip == 1:
            image = T.hflip(image)
        elif flip == 2:
            image = T.vflip(image)
        else:
            pass
        return image
    
class Normalize(object):
    def __call__(self, image, *args, **kwargs):
        image = T.normalize(image, mean=[0.5], std=[0.5])
        return image

class ResizeCrop(object):
    def __call__(self, image, *args, **kwargs):
        original_w, original_h = image.size
        size_ratio = kwargs.get('size_ratio')
        top_ratio = kwargs.get('top_ratio')
        left_ratio = kwargs.get('left_ratio')

        new_w = int(original_w * size_ratio)
        new_h = int(original_h * size_ratio)
        vertical_margin = original_h - new_h
        horizontal_margin = original_w - new_w

        top = int(vertical_margin * top_ratio)
        left = int(horizontal_margin * left_ratio)
        image = T.resized_crop(image, top, left, new_h, new_w, (original_h, original_w), interpolation=T.InterpolationMode.BILINEAR) # type: ignore

        return image

def build_transform(config):
    """ Build a transform pipeline which uses the same random number for all images in the sequence. """
    transforms_list = []
    transforms_list.append(ResizeCrop())
    if config.task in ['mlm','mlm_v2']:
        transforms_list.append(Rotate())
        transforms_list.append(Flip())
    transforms_list.append(ToTensor())
    # transforms_list.append(Normalize())
    return Compose(transforms_list)

def build_dataset(config, is_train=True):
    """ Build the EventSequenceDataset with the specified configuration. """
    transform = None
    if is_train:
        transform = build_transform(config)
    dataset = EventSequenceDataset(config, transform, is_train)
    return dataset

def build_dataloader(config, is_train=True):
    """ Build a DataLoader for the EventSequenceDataset. """
    dataset = build_dataset(config, is_train)
    if is_train:
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers, 
            pin_memory=True,
            drop_last=True)
    else:
        dataloader = DataLoader(
            dataset, batch_size=1,
            shuffle=False,
            num_workers=1, 
            pin_memory=True,
            drop_last=True)
    return dataloader