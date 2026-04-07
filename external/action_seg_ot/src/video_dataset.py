import os
import os.path as path

import numpy as np
import torch
from torch.utils.data import Dataset


def parse_action_name(fname, dataset):
    """
    Return the activity class name for a video file.
    Defaults:
      - Breakfast: last underscore token
      - YTI: everything except last underscore token
      - FS/desktop_assembly: single activity -> ''
      - Fallback/custom: strip extension and use full stem to avoid filtering everything out.
    """
    stem = os.path.splitext(fname)[0]
    if dataset == 'Breakfast':
        return fname.split('_')[-1]
    if dataset == 'YTI':  # ignores _idt files in groundTruth automatically
        return '_'.join(fname.split('_')[:-1])
    if dataset in ['FS', 'desktop_assembly']:  # only one activity class
        return ''
    # custom / unknown dataset: use full stem so filtering still works
    return stem


def _normalize_split_entry(name: str) -> str:
    item = str(name or "").strip()
    if not item:
        return ""
    return os.path.basename(item)


def _resolve_split_path(data_dir: str, split: str) -> str:
    candidate = str(split or "").strip()
    if not candidate:
        return ""
    if path.isfile(candidate):
        return candidate
    if path.isabs(candidate):
        return candidate
    return path.join(data_dir, candidate)


class VideoDataset(Dataset):
    def __init__(self, root_dir: str, dataset, n_frames, standardise=True, split: str = None, random=True, n_videos=None, action_class=['all']):
        self.root_dir = root_dir
        self.dataset = dataset
        if self.dataset == 'FSeval':
            self.dataset = 'FS'
            granularity = 'eval'
        else:
            granularity = None
        self.data_dir = path.join(root_dir, self.dataset)
        self.video_fnames = sorted([
            fname for fname in os.listdir(path.join(self.data_dir, 'groundTruth'))
            if not fname.startswith('.')
        ])
        if split:
            split_path = _resolve_split_path(self.data_dir, split)
            if not path.isfile(split_path):
                raise FileNotFoundError(f"Split file not found: {split_path}")
            allowed = set()
            for raw in open(split_path):
                entry = _normalize_split_entry(raw)
                if not entry:
                    continue
                allowed.add(entry)
                if not entry.endswith(".txt"):
                    allowed.add(entry + ".txt")
            self.video_fnames = [fname for fname in self.video_fnames if fname in allowed]
        if self.dataset in ['FS', 'desktop_assembly']:
            action_class = ''
        if action_class != ['all']:
            if type(action_class) is list:
                self.video_fnames = [fname for fname in self.video_fnames if parse_action_name(fname, self.dataset) in action_class]
            else:
                self.video_fnames = [fname for fname in self.video_fnames if parse_action_name(fname, self.dataset) == action_class]
        if n_videos is not None:
            self.video_fnames = self.video_fnames[::max(1, int(len(self.video_fnames) / n_videos))]
        def prep(x):
            i, nm = x.rstrip().split(' ')
            return nm, int(i)
        if granularity is None:  # granularity applies only to 50Salads
            action_mapping = list(map(prep, open(path.join(self.data_dir, 'mapping/mapping.txt'))))
        else:
            action_mapping = list(map(prep, open(path.join(self.data_dir, f'mapping/mapping{granularity}.txt'))))
        self.action_mapping = dict(action_mapping)
        self.n_subactions = len(set(self.action_mapping.keys()))
        self.n_frames = n_frames
        self.standardise = standardise
        self.random = random

    def __len__(self):
        return len(self.video_fnames)
    
    def __getitem__(self, idx):
        video_fname = self.video_fnames[idx]
        gt = [line.rstrip() for line in open(path.join(self.data_dir, 'groundTruth', video_fname))]
        inds, mask = self._partition_and_sample(self.n_frames, len(gt))
        gt = torch.Tensor([self.action_mapping[gt[ind]] for ind in inds]).long()
        action = parse_action_name(video_fname, self.dataset)
        base_name = path.splitext(video_fname)[0]
        feat_fname = path.join(self.data_dir, 'features', action, base_name)
        # fallback: if per-action subdir missing, try flat layout
        if action == '' or (not path.exists(feat_fname + '.txt') and not path.exists(feat_fname + '.npy')):
            feat_fname = path.join(self.data_dir, 'features', base_name)
        tried = [feat_fname + '.txt', feat_fname + '.npy']
        try:
            features = np.loadtxt(tried[0])[inds, :]
        except Exception:
            if path.exists(tried[1]):
                features = np.load(tried[1])[inds, :]
            else:
                raise FileNotFoundError(f"Features not found for {video_fname}; tried {tried}")
        if self.standardise:  # normalize features
            zmask = np.ones(features.shape[0], dtype=bool)
            for rdx, row in enumerate(features):
                if np.sum(row) == 0:
                    zmask[rdx] = False
            z = features[zmask] - np.mean(features[zmask], axis=0)
            std = np.std(features[zmask], axis=0)
            std[std == 0] = 1.0
            z = z / std
            features = np.zeros(features.shape)
            features[zmask] = z
            features = np.nan_to_num(features)
            features /= np.sqrt(features.shape[1])
        # mask = torch.from_numpy(mask * zmask)
        features = torch.from_numpy(features).float()
        return features, mask, gt, video_fname, gt.unique().shape[0]
    
    def _partition_and_sample(self, n_samples, n_frames):
        if n_samples is None:
            indices = np.arange(n_frames)
            mask = np.full(n_frames, 1, dtype=bool)
        elif n_samples < n_frames:
            if self.random:
                boundaries = np.linspace(0, n_frames-1, n_samples+1).astype(int)
                indices = np.random.randint(low=boundaries[:-1], high=boundaries[1:])
            else:
                indices = np.linspace(0, n_frames-1, n_samples).astype(int)
            mask = np.full(n_samples, 1, dtype=bool)
        else:
            indices = np.concatenate((np.arange(n_frames), np.full(n_samples - n_frames, n_frames - 1)))
            mask = np.concatenate((np.full(n_frames, 1, dtype=bool), np.zeros(n_samples - n_frames, dtype=bool)))
        return indices, mask
