import os
import random
import cv2
import numpy as np
import glob
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A

from .util import load_caption_dict, keep_and_drop, load_flo_file, adaptive_weighted_downsample, normalize_for_warping

class UniDataset(Dataset):
    def __init__(self,
                 anno_path,
                 root_dir,
                 local_type_list,
                 resolution,
                 drop_txt_prob,
                 global_type_list,
                 keep_all_cond_prob,
                 drop_all_cond_prob,
                 drop_each_cond_prob,
                 transform=False):

        self.local_type_list = local_type_list
        self.global_type_list = global_type_list
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
        self.transform = transform

        self.sequences = glob.glob(os.path.join(root_dir, '*/*'))
        self.annos = load_caption_dict(anno_path)

        self.video_frames = []
        for video_dir in self.sequences:
            frames = sorted([
                os.path.join(video_dir, f)
                for f in os.listdir(video_dir)
                if f.endswith(('.jpg', '.png')) and f not in ['r1.png', 'r2.png']
            ])
            self.video_frames.extend(frames)

        self.aug_targets ={}
        # Albumentations key remapping
        if 'r1' in self.local_type_list:
            self.aug_targets['r1'] = 'image'
        if 'r2' in self.local_type_list:
            self.aug_targets['r2'] = 'image'
        if 'depth' in self.local_type_list:
            self.aug_targets['depth'] = 'image'

        if self.transform:
            self.augmentation = A.Compose([
                A.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.1, hue=0.1, p=1.0
                ),
            ], additional_targets=self.aug_targets)

    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, index):
        img_path = Path(self.video_frames[index])
        sequence_id = f"{img_path.parent.parent.name}_{img_path.parent.name}"
        anno = self.annos[sequence_id]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        local_files = {}
        for local_type in self.local_type_list:
            if local_type == 'r1':
                local_files['r1'] = img_path.with_name('r1.png')
            elif local_type == 'r2':
                local_files['r2'] = img_path.with_name('r2.png')
            elif local_type == 'depth':
                local_files['depth'] = img_path.parent / 'depth' / img_path.name.replace('.png', '_depth.png')
            elif local_type == 'flow':
                local_files['flow'] = img_path.parent / 'Flow' / img_path.name.replace('.png', '.flo')

        # Prepare inputs for augmentation
        image_inputs = {'image': image}
        for key in local_files.keys():
            path = local_files.get(key, None)
            if key is not 'flow' and path.exists():
                img = cv2.imread(str(path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_inputs[key] = img

        # Apply augmentation
        if self.transform:
            augmented = self.augmentation(**image_inputs)
            # print('aug done')
        else:
            augmented = {k: cv2.resize(v, (self.resolution, self.resolution)) for k, v in image_inputs.items()}

        # Normalize and prepare outputs
        jpg = augmented['image']
        jpg = cv2.resize(jpg, (self.resolution, self.resolution))
        jpg = (jpg.astype(np.float32) / 127.5) - 1.0  # Normalize to [-1, 1]

        local_conditions = []
        for k in ['r1','r2','depth']:
            if k in augmented:
                cond = cv2.resize(augmented[k], (self.resolution, self.resolution))
                cond = cond.astype(np.float32) / 255.0
                local_conditions.append(cond)

        # Handle flow (resize + normalize only)
        flow = None
        if 'flow' in local_files and local_files['flow'].exists():
            flow = load_flo_file(local_files['flow'])
            flow = adaptive_weighted_downsample(flow, target_h=self.resolution, target_w=self.resolution)
            flow = normalize_for_warping(flow)

        # Global conditions (assumed pre-extracted as .npy files)
        global_conditions = []
        for global_type in self.global_type_list:
            global_path = img_path.parent / f"{global_type}.npy"
            if global_path.exists():
                global_conditions.append(np.load(global_path))

        # Drop text or conditions as per policy
        if random.random() < self.drop_txt_prob:
            anno = ""

        local_conditions = keep_and_drop(local_conditions, self.keep_all_cond_prob,
                                         self.drop_all_cond_prob, self.drop_each_cond_prob)
        global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob,
                                          self.drop_all_cond_prob, self.drop_each_cond_prob)

        if len(local_conditions):
            local_conditions = np.concatenate(local_conditions, axis=2)
        if len(global_conditions):
            global_conditions = np.concatenate(global_conditions)

        return {
            'jpg': jpg,
            'txt': anno,
            'local_conditions': local_conditions if len(local_conditions) else None,
            'global_conditions': global_conditions if len(global_conditions) else None,
            'flow': flow
        }
