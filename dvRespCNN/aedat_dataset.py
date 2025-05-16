
# aedat_dataset.py

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import tonic.transforms as transforms
from tonic.io import read_aedat4


class AEDATRespirationDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 csv_path: str,
                 sensor_size=(640, 480, 2),
                 frames_per_sample=32):
        """
        Dataset for AEDAT4 event files paired with GT respiration rates.
        Only files with a matching GT RR in the CSV are kept.
        """
        # Load and normalize CSV labels
        df = pd.read_csv(csv_path)
        df["Distance"] = df["Distance"].str.replace(",", ".")
        # Build lookup: (Patient, Distance, Reading) -> GT RR
        self.label_map = {
            (int(r.Patient), r.Distance, int(r.Reading)): float(r["GT RR"])
            for _, r in df.iterrows()
            if pd.notna(r["GT RR"])
        }

        # Gather and filter AEDAT4 files
        all_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".aedat4")
        )
        self.files = []
        for path in all_files:
            basename = os.path.basename(path)
            parts = basename.split("_")
            try:
                patient = int(parts[1])
                distance = parts[2]
                reading = int(parts[-1].replace(".aedat4", ""))
                key = (patient, distance, reading)
                if key in self.label_map:
                    self.files.append(path)
            except Exception:
                # skip files that don't match naming convention
                continue

        # Frame transform
        self.transform = transforms.ToFrame(
            sensor_size=sensor_size,
            n_time_bins=frames_per_sample
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        basename = os.path.basename(path)
        parts = basename.split("_")
        patient = int(parts[1])
        distance = parts[2]
        reading = int(parts[-1].replace(".aedat4", ""))

        # Read raw events and convert to frames
        events = read_aedat4(path)              # numpy structured array
        frames_np = self.transform(events)      # shape (T, H, W) or (T, H, W, C)
        # Convert to torch.Tensor and ensure float type
        frames = torch.from_numpy(frames_np).float()
        # Ensure shape is (C, T, H, W)
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        else:
            # move channel last to first
            frames = frames.permute(3, 0, 1, 2)

        # Retrieve GT RR
        key = (patient, distance, reading)
        gt_rr = self.label_map[key]
        label = torch.tensor(gt_rr, dtype=torch.float32)

        return frames, label
