import os
import random
import cv2
import numpy as np
import glob
from torch.utils.data import Dataset
from pathlib import Path

from .util import *

def load_caption_dict(txt_path):
    caption_dict = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue  # skip empty or malformed lines

            path, caption = line.split(':', 1)
            parts = path.strip().split('/')
            if len(parts) >= 3:
                parent1 = parts[-3].zfill(5)
                parent2 = parts[-2].zfill(4)
                key = f"{parent1}_{parent2}"
                caption_dict[key] = caption.strip()
    return caption_dict


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
                 drop_each_cond_prob):
     
     self.local_type_list = local_type_list
     self.global_type_list = global_type_list
     self.resolution = resolution
     self.drop_txt_prob = drop_txt_prob
     self.keep_all_cond_prob = keep_all_cond_prob
     self.drop_all_cond_prob = drop_all_cond_prob
     self.drop_each_cond_prob = drop_each_cond_prob

     self.sequences = glob.glob(root_dir+'/*/*')
     self.annos = load_caption_dict(anno_path)

     self.video_frames = []
     for video_dir in self.sequences:
        frames = sorted([
                os.path.join(video_dir, f)
                for f in os.listdir(video_dir)
                if f.endswith(('.jpg', '.png')) and f not in ['r1.png','r2.png']
            ])
        self.video_frames.extend(frames)

    def __len__(self):
        return len(self.video_frames)
        
    def __getitem__(self, index):
        img_path = Path(self.video_frames[index])
        parts = os.path.normpath(img_path).split(os.sep)
        sequence_id = f"{parts[-3]}_{parts[-2]}"
        # idx = self.file_ids.index(sequence_id)
        anno = self.annos[sequence_id]
        
        try:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.resolution, self.resolution))
            image = (image.astype(np.float32) / 127.5) - 1.0

        except Exception as e:
            print('error: ',e,img_path)
            raise e  

        global_files = []
        for global_type in self.global_type_list:
            global_files.append(self.global_paths[global_type][index])

        local_files = []
        for local_type in self.local_type_list:
           if local_type =='r1':  
              local_files.append(img_path.with_name('r1.png'))
           if local_type == 'r2':
              local_files.append(img_path.with_name('r2.png'))
           if local_type == 'depth':
              new_path = img_path.parent / 'depth' / img_path.name.replace('.png', '_depth.png')
              local_files.append(new_path)
           if local_type == 'flow':
              local_files.append(img_path.parent / 'Flow' / img_path.name)
        # print(local_files)

        local_conditions = []
        for local_file in local_files: 
            # print(local_file)
            condition = cv2.imread(str(local_file))
            try:    
                condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
                condition = cv2.resize(condition, (self.resolution, self.resolution))
                condition = condition.astype(np.float32) / 255.0
                local_conditions.append(condition)
            except Exception as e:
                print('missing', e,  local_file)
                raise e

        global_conditions = []
        for global_file in global_files:
            condition = np.load(global_file)
            global_conditions.append(condition)

        if random.random() < self.drop_txt_prob:
            anno = ''
        
        local_conditions = keep_and_drop(local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        
        if len(local_conditions) != 0:
            local_conditions = np.concatenate(local_conditions, axis=2)
        if len(global_conditions) != 0:
            global_conditions = np.concatenate(global_conditions)

        return dict(jpg=image, txt=anno, local_conditions=local_conditions, global_conditions=global_conditions)
           