import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as T
from PIL import Image
import os
import glob
import logging

logger = logging.getLogger(__name__)

class EventSequenceDataset(Dataset):
    def __init__(self, config, transform=None, is_train=True):
        if is_train:
            self.root_dir = config.train_root
        else:
            self.root_dir = config.test_root

        self.sequence_length = config.max_seq_len
        self.transform = transform
        self.transform_base = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])      

        self.samples = []
        self.data =[]
        sequence_folders = sorted(glob.glob(os.path.join(self.root_dir, '*/')))

        for folder in sequence_folders:
            img_dir = os.path.join(folder, 'img')
            pos_image_paths = sorted(glob.glob(os.path.join(img_dir, '*_pos.png')))
            num_images = len(pos_image_paths)
            for start_idx in range(num_images - self.sequence_length + 1):
                self.samples.append((img_dir, start_idx))

                ##self.data.append((csv_path, start_idx))

        logger.info(f"Dataset initialized with {len(self.samples)} sequences from {self.root_dir}.")

        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_dir, start_idx = self.samples[idx]
        pos_images = []
        neg_images = []
        traj_list = []
        if self.transform is not None:
            angle = torch.randint(0, 4, (1,)).item() * 90
            flip = torch.randint(0, 3, (1,)).item()

        for i in range(self.sequence_length):
            frame_idx = start_idx + i
            frame_name = str(frame_idx).zfill(4)
            pos_image_path = os.path.join(img_dir, f'{frame_name}_pos.png')
            neg_image_path = os.path.join(img_dir, f'{frame_name}_neg.png')
            pos_image = Image.open(pos_image_path).convert('RGB') # (3,200,200) (0~255,8bit) -> 标准(-1~1)
            neg_image = Image.open(neg_image_path).convert('RGB')
            if self.transform is not None:
                pos_image = self.transform(pos_image, angle, flip)
                neg_image = self.transform(neg_image, angle, flip)
            else:
                pos_image = self.transform_base(pos_image)
                neg_image = self.transform_base(neg_image)
            pos_images.append(pos_image) # sequence_length(16) 个 (3,200,200)的列表
            neg_images.append(neg_image)

            #用pandas读取csv_path,从frame_idx读取对应行数,保存成（12）一维张量,append添加到traj_list

        x_pos_seq = torch.stack(pos_images) #叠加成(16,3,200,200)
        x_neg_seq = torch.stack(neg_images)

        #再用一次stack组合成(16,12)形状
        
        return x_pos_seq, x_neg_seq
    
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
        image = T.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return image
    
def build_transform():
    """ Build a transform pipeline which uses the same random number for all images in the sequence. """
    transforms_list = []
    transforms_list.append(Rotate())
    transforms_list.append(Flip())
    transforms_list.append(ToTensor())
    transforms_list.append(Normalize())
    return Compose(transforms_list)

def build_dataset(config, is_train=True):
    """ Build the EventSequenceDataset with the specified configuration. """
    transform = None
    if is_train:
        if config.task == 'mlm':
            transform = build_transform()
    dataset = EventSequenceDataset(config, transform, is_train)
    return dataset

def build_dataloader(config, is_train=True):
    """ Build a DataLoader for the EventSequenceDataset. """
    dataset = build_dataset(config, is_train)
    if is_train:
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size,
            shuffle=config.shuffle,
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