import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.dcp import dcp_restore

class UIEBDataset(Dataset):
    def __init__(self, raw_dir, ref_dir, img_size=256):
        self.raw_paths = sorted(os.listdir(raw_dir))
        self.raw_dir = raw_dir
        self.ref_dir = ref_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, idx):
        name = self.raw_paths[idx]

        raw = cv2.imread(os.path.join(self.raw_dir, name))
        ref = cv2.imread(os.path.join(self.ref_dir, name))

        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

        raw = cv2.resize(raw, (self.img_size, self.img_size))
        ref = cv2.resize(ref, (self.img_size, self.img_size))

        raw = raw.astype("float32") / 255.0
        ref = ref.astype("float32") / 255.0

        # ---- Physics ----
        J_phys, t, _ = dcp_restore((raw * 255).astype("uint8"))
        J_phys = J_phys.astype("float32") / 255.0

        # normalize t
        t = t.astype("float32")
        t = cv2.resize(t, (self.img_size, self.img_size))
        t = np.expand_dims(t, axis=2)

        # ---- Stack input ----
        inp = np.concatenate([raw, J_phys, t], axis=2)  # 7 channels

        # ---- To tensor ----
        inp = torch.from_numpy(inp).permute(2,0,1)
        ref = torch.from_numpy(ref).permute(2,0,1)

        return inp, ref