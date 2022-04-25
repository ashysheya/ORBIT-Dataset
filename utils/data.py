# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tv_F
from collections import defaultdict

class DatasetFromClipPaths(Dataset):
    def __init__(self, clip_paths, with_labels):
        super().__init__()
        #TODO currently doesn't support loading of annotations
        self.with_labels = with_labels
        if self.with_labels:
            self.clip_paths, self.clip_labels = clip_paths
        else:
            self.clip_paths = clip_paths
        
        self.normalize_stats = {'mean' : [0.500, 0.436, 0.396], 'std' : [0.145, 0.143, 0.138]} # orbit mean train frame
        
    def __getitem__(self, index):
        clip = []
        for frame_path in self.clip_paths[index]:
            frame = self.load_and_transform_frame(frame_path)
            clip.append(frame)
    
        if self.with_labels:
            return torch.stack(clip, dim=0), self.clip_labels[index]
        else:
            return torch.stack(clip, dim=0)
    
    def load_and_transform_frame(self, frame_path):
        """
        Function to load and transform frame.
        :param frame_path: (str) Path to frame.
        :return: (torch.Tensor) Loaded and transformed frame.
        """
        frame = Image.open(frame_path)
        frame = tv_F.to_tensor(frame)
        return tv_F.normalize(frame, mean=self.normalize_stats['mean'], std=self.normalize_stats['std'])

    def __len__(self):
        return len(self.clip_paths)


class TaskResampler:
    def __init__(self, clip_paths, clip_labels):
        self.clip_dict = defaultdict(defaultdict)
        self.arrange_clips(clip_paths, clip_labels)

    def arrange_clips(self, clip_paths, clip_labels):
        for i, label in enumerate(clip_labels):
            clip_name = '-'.join(clip_paths[i][0].split('-')[:-1])
            l_np = int(label.numpy())
            if clip_name in self.clip_dict[l_np]:
                self.clip_dict[l_np][clip_name].append(i)
            else:
                self.clip_dict[l_np][clip_name] = [i]

    def resample_task(self, clips, clip_labels, batch_size):
        context_batch_size = batch_size*8
        context_idx = torch.zeros_like(clip_labels) - 1.0
        num_classes = len(self.clip_dict)
        num_classes_per_task = num_classes
        chosen_classes = np.random.choice(list(self.clip_dict.keys()), num_classes_per_task, replace=False)

        num_clips_per_class = context_batch_size // num_classes_per_task

        for label in chosen_classes:
            num_videos_per_class = len(self.clip_dict[label])
            # num_context_videos_per_class = np.random.randint(min(3, num_videos_per_class - 1), num_videos_per_class)
            num_context_videos_per_class = num_videos_per_class//2
            if num_clips_per_class < num_context_videos_per_class:
                num_context_videos_per_class = num_clips_per_class
            if num_context_videos_per_class == 0:
                num_context_videos_per_class = 1

            video_idxs = np.random.choice(num_videos_per_class, size=num_context_videos_per_class, replace=False)
            context_idx[clip_labels == label] = 0.0

            all_clips_per_video = [0]
            clip_choosing_dict = {}
            for idx in sorted(video_idxs):
                clip_name = sorted(list(self.clip_dict[label].keys()))[idx]
                clip_idx = np.random.randint(0, len(self.clip_dict[label][clip_name]))
                if num_videos_per_class > 1:
                    context_idx[self.clip_dict[label][clip_name]] = 2.0

                context_idx[np.array(self.clip_dict[label][clip_name])[clip_idx]] = 1.0
                if len(self.clip_dict[label][clip_name]) > 1:
                    clip_choosing_dict[clip_name] = self.clip_dict[label][clip_name][:clip_idx] + self.clip_dict[label][clip_name][clip_idx + 1:]
                    all_clips_per_video += [all_clips_per_video[-1] + len(clip_choosing_dict[clip_name])]

            video_distr = np.random.choice(all_clips_per_video[-1], min(all_clips_per_video[-1], num_clips_per_class - num_context_videos_per_class), replace=False)

            for i, clip_name in enumerate(sorted(clip_choosing_dict)):
                chosen_clips = np.bitwise_and(video_distr < all_clips_per_video[i + 1], video_distr >= all_clips_per_video[i])
                chosen_clips = video_distr[chosen_clips] - all_clips_per_video[i]
                chosen_idxs = np.array(clip_choosing_dict[clip_name])[chosen_clips]
                context_idx[chosen_idxs] = 1.0

        context_clips = clips[context_idx == 1.0]
        context_labels = clip_labels[context_idx == 1.0]

        frame_idx = np.random.randint(0, 8)
        context_clips_sampled = context_clips[:, frame_idx: frame_idx + 1]
        # context_clips_sampled = context_clips

        context_labels_updated = context_labels.clone()
        target_labels = clip_labels[context_idx == 0.0]
        target_labels_updated = target_labels.clone()

        target_size = len(target_labels_updated)
        target_loader = get_clip_loader((clips[context_idx == 0.0], target_labels_updated), batch_size, True)
        return context_clips_sampled, context_labels_updated, target_loader, target_size

    # def resample_task(self, clips, clip_labels, batch_size):
    #     context_batch_size = batch_size*8
    #     context_idx = torch.zeros_like(clip_labels) - 1.0
    #     num_labels = len(self.clip_dict)
    #     # num_classes_per_task = np.random.randint(low=min(5, num_labels), high=num_labels + 1)
    #     num_classes_per_task = num_labels
    #     chosen_classes = np.random.choice(list(self.clip_dict.keys()), num_classes_per_task, replace=False)
    #
    #     for label in chosen_classes:
    #
    #         num_videos = len(self.clip_dict[label])
    #         num_context_videos = num_videos//2
    #
    #         if num_context_videos == 0:
    #             num_context_videos = 1
    #
    #         if num_context_videos > context_batch_size // num_classes_per_task:
    #             num_context_videos = context_batch_size // num_classes_per_task
    #
    #         num_clips_per_video = context_batch_size // (num_classes_per_task * num_context_videos)
    #
    #         video_idxs = np.random.choice(num_videos, num_context_videos, replace=False)
    #         context_idx[clip_labels == label] = 0.0
    #         for idx in video_idxs:
    #             clip_name = list(self.clip_dict[label].keys())[idx]
    #             context_idx[self.clip_dict[label][clip_name]] = 2.0
    #             clips_idx = np.random.choice(len(self.clip_dict[label][clip_name]),
    #                                          min(num_clips_per_video, len(self.clip_dict[label][clip_name])),
    #                                          replace=False)
    #
    #             context_idx[np.array(self.clip_dict[label][clip_name])[clips_idx]] = 1.0
    #
    #     context_clips = clips[context_idx == 1.0]
    #     context_labels = clip_labels[context_idx == 1.0]
    #
    #     frame_idx = np.random.randint(0, 8)
    #     context_clips_sampled = context_clips[:, frame_idx: frame_idx + 1]
    #
    #     context_labels_updated = context_labels.clone()
    #     target_labels = clip_labels[context_idx == 0.0]
    #     target_labels_updated = target_labels.clone()
    #     # for i, label in enumerate(chosen_classes):
    #     #     context_labels_updated[context_labels == label] = i
    #     #     target_labels_updated[target_labels == label] = i
    #
    #     target_loader = get_clip_loader((clips[context_idx == 0.0], target_labels_updated), batch_size, True)
    #     return context_clips_sampled, context_labels_updated, target_loader

# class TaskResampler:
#     def __init__(self, clip_paths, clip_labels):
#         self.clip_dict = defaultdict(defaultdict)
#         self.arrange_clips(clip_paths, clip_labels)

#     def arrange_clips(self, clip_paths, clip_labels):
#         for i, label in enumerate(clip_labels):
#             clip_name = '-'.join(clip_paths[i][0].split('-')[:-1])
#             l_np = int(label.numpy())
#             if clip_name in self.clip_dict[l_np]:
#                 self.clip_dict[l_np][clip_name].append(i)
#             else:
#                 self.clip_dict[l_np][clip_name] = [i]
#         min_num_videos = 100
#         for label in self.clip_dict:
#             if len(self.clip_dict[label]) < min_num_videos:
#                 min_num_videos = len(self.clip_dict[label])
#         self.min_num_videos = min_num_videos

#     def resample_task(self, clips, clip_labels, batch_size):
#         context_batch_size = batch_size*8
#         context_idx = torch.zeros_like(clip_labels) - 1.0
#         num_labels = len(self.clip_dict)
#         # num_classes_per_task = np.random.randint(low=min(5, num_labels), high=num_labels + 1)
#         num_classes_per_task = num_labels
#         chosen_classes = np.random.choice(list(self.clip_dict.keys()), num_classes_per_task, replace=False)

#         num_context_videos = np.random.randint(min(3, self.min_num_videos - 1), self.min_num_videos)

#         if num_context_videos > context_batch_size//num_classes_per_task:
#             num_context_videos = context_batch_size//num_classes_per_task

#         num_clips_per_video = context_batch_size//(num_classes_per_task*num_context_videos)
#         for label in chosen_classes:
#             video_idxs = np.random.choice(len(self.clip_dict[label]), num_context_videos, replace=False)
#             context_idx[clip_labels == label] = 0.0
#             for idx in video_idxs:
#                 clip_name = list(self.clip_dict[label].keys())[idx]
#                 context_idx[self.clip_dict[label][clip_name]] = 2.0
#                 clips_idx = np.random.choice(len(self.clip_dict[label][clip_name]),
#                                              min(num_clips_per_video, len(self.clip_dict[label][clip_name])),
#                                              replace=False)

#                 context_idx[np.array(self.clip_dict[label][clip_name])[clips_idx]] = 1.0

#         context_clips = clips[context_idx == 1.0]
#         context_labels = clip_labels[context_idx == 1.0]

#         frame_idx = np.random.randint(0, 8)
#         # context_clips_sampled = context_clips[:, frame_idx: frame_idx + 1]
#         context_clips_sampled = context_clips

#         context_labels_updated = context_labels.clone()
#         target_labels = clip_labels[context_idx == 0.0]
#         target_labels_updated = target_labels.clone()
#         # for i, label in enumerate(chosen_classes):
#         #     context_labels_updated[context_labels == label] = i
#         #     target_labels_updated[target_labels == label] = i

#         target_loader = get_clip_loader((clips[context_idx == 0.0], target_labels_updated), batch_size, True)
#         return context_clips_sampled, context_labels_updated, target_loader


def get_clip_loader(clips, batch_size, with_labels=False):
    if isinstance(clips[0], np.ndarray):
        clips_dataset = DatasetFromClipPaths(clips, with_labels=with_labels)
        return DataLoader(clips_dataset,
                      batch_size=batch_size,
                      num_workers=4,
                      pin_memory=True,
                      prefetch_factor=8,
                      persistent_workers=True)

    elif isinstance(clips[0], torch.Tensor):
        if with_labels:
            return list(zip(clips[0].split(batch_size), clips[1].split(batch_size)))
        else: 
            return clips.split(batch_size)


def attach_frame_history(frames, history_length):
    
    if isinstance(frames, np.ndarray):
        return attach_frame_history_paths(frames, history_length)
    elif isinstance(frames, torch.Tensor):
        return attach_frame_history_tensor(frames, history_length)

def attach_frame_history_paths(frame_paths, history_length):
    """
    Function to attach the immediate history of history_length frames to each frame in an array of frame paths.
    :param frame_paths: (np.ndarray) Frame paths.
    :param history_length: (int) Number of frames of history to append to each frame.
    :return: (np.ndarray) Frame paths with attached frame history.
    """
    # pad with first frame so that frames 0 to history_length-1 can be evaluated
    frame_paths = np.concatenate([np.repeat(frame_paths[0], history_length-1), frame_paths])
    
    # for each frame path, attach its immediate history of history_length frames
    frame_paths = [ frame_paths ]
    for l in range(1, history_length):
        frame_paths.append( np.roll(frame_paths[0], shift=-l, axis=0) )
    frame_paths_with_history = np.stack(frame_paths, axis=1) # of size num_clips x history_length
    
    if history_length > 1:
        return frames_with_history[:-(history_length-1)] # frames have wrapped around, remove last (history_length - 1) frames
    else:
        return frames_with_history

def attach_frame_history_tensor(frames, history_length):
    """
    Function to attach the immediate history of history_length frames to each frame in a tensor of frame data.
    param frames: (torch.Tensor) Frames.
    :param history_length: (int) Number of frames of history to append to each frame.
    :return: (torch.Tensor) Frames with attached frame history.
    """
    # pad with first frame so that frames 0 to history_length-1 can be evaluated
    frame_0 = frames.narrow(0, 0, 1)
    frames = torch.cat((frame_0.repeat(history_length-1, 1, 1, 1), frames), dim=0)

    # for each frame, attach its immediate history of history_length frames
    frames = [ frames ]
    for l in range(1, history_length):
        frames.append( frames[0].roll(shifts=-l, dims=0) )
    frames_with_history = torch.stack(frames, dim=1) # of size num_clips x history_length
    
    if history_length > 1:
        return frames_with_history[:-(history_length-1)] # frames have wrapped around, remove last (history_length - 1) frames
    else:
        return frames_with_history

def unpack_task(task_dict, device, context_to_device=True, target_to_device=False, preload_clips=False):
   
    context_clips = task_dict['context_clips']
    context_paths = task_dict['context_paths']
    context_labels = task_dict['context_labels']
    context_annotations = task_dict['context_annotations']
    target_clips = task_dict['target_clips']
    target_paths = task_dict['target_paths']
    target_labels = task_dict['target_labels']
    target_annotations = task_dict['target_annotations']
    object_list = task_dict['object_list']

    if context_to_device and isinstance(context_labels, torch.Tensor):
        context_labels = context_labels.to(device)
    if target_to_device and isinstance(target_labels, torch.Tensor):
        target_labels = target_labels.to(device)
  
    if preload_clips:
        return context_clips, context_paths, context_labels, target_clips, target_paths, target_labels, object_list
    else:
        return context_paths, context_paths, context_labels, target_paths, target_paths, target_labels, object_list
