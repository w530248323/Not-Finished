import os
import glob
import torch.utils.data
import numpy as np
from PIL import Image
from data_parser import JpegDataset
from torchvision.transforms import *

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG']


def default_loader(path):
    return Image.open(path).convert('RGB')


class VideoFolder(torch.utils.data.Dataset):
    def __init__(self, root, csv_file_input, csv_file_labels, clip_size,
                 nclips, step_size, is_val, transform=None,
                 loader=default_loader):
        self.dataset_object = JpegDataset(csv_file_input, csv_file_labels, root)

        self.csv_data = self.dataset_object.csv_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root
        self.transform = transform
        self.loader = loader

        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val

    def __getitem__(self, index):
        item = self.csv_data[index]
        img_paths = self.get_frame_names(item.path)

        imgs = []
        for img_path in img_paths:
            img = self.loader(img_path)
            img = self.transform(img)
            imgs.append(torch.unsqueeze(img, 0))

        target_idx = self.classes_dict[item.label]

        # format data to torch
        data = torch.cat(imgs)
        data = data.permute(1, 0, 2, 3)
        return (data, target_idx)

    def __len__(self):
        return len(self.csv_data)

    def get_frame_names(self, path):
        frame_names = []
        for ext in IMG_EXTENSIONS:
            frame_names.extend(glob.glob(os.path.join(path, "*" + ext)))
        frame_names = list(sorted(frame_names))
        num_frames = len(frame_names)

        # set number of necessary frames
        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames

        # pick frames
        offset = 0
        if num_frames_necessary > num_frames:
            # Pad last frame if video is shorter than necessary
            frame_names += [frame_names[-1]] * \
                           (num_frames_necessary - num_frames)
        elif num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset.
            diff = (num_frames - num_frames_necessary)
            # temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)
        frame_names = frame_names[offset:num_frames_necessary + offset:self.step_size]
        return frame_names
