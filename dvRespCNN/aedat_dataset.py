
# aedat_dataset.py

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import tonic.transforms as transforms
from tonic.io import read_aedat4
import matplotlib.pyplot as plt


class AEDATRespirationDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 csv_path: str,
                 sensor_size=(640, 480, 2),
                 frames_per_sample=900,
                filtered = True,
                plot_labels=False,
                max_time = 30,
                distances = ['0.5m','1m','1.5m','2.0m']):

        """
        Dataset for AEDAT4 event files paired with GT respiration rates.
        Only files with a matching GT RR in the CSV are kept.
        """
        self.sensor_size = sensor_size
        self.frames_per_sample=frames_per_sample
        self.filtered=filtered
        self.max_time=max_time
        self.distances= distances
        # Load and normalize CSV labels
        df = pd.read_csv(csv_path)
        df["Distance"] = df["Distance"].str.replace(",", ".")
        # Build lookup: (Patient, Distance, Reading) -> GT RR
        self.label_map = {
            (int(r.Patient), r.Distance, int(r.Reading)): float(r["GT RR"])
            for _, r in df.iterrows()
            if pd.notna(r["GT RR"]) and 8.0 <= float(r["GT RR"]) <= 20.0
        }


        # Gather and filter AEDAT4 files
        all_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".aedat4")
        )
        self.files = []
        self.labels = []

        for path in all_files:
            basename = os.path.basename(path)
            parts = basename.split("_")
            if self.filtered:
                if not 'filtered' in parts[-1]:
                    continue
            
            if 'filtered.aedat4' in parts:
                #print('removing')
                parts.remove('filtered.aedat4')
                
            if 'filtered2.aedat4' in parts:
                parts.remove('filtered2.aedat4')

            try:
                patient = int(parts[1])
                distance = parts[2].replace(",", ".")
                #print(parts)
                reading = int(parts[-1].replace(".aedat4", ""))
                key = (patient, distance, reading)
                events = read_aedat4(path)  
                if key in self.label_map and distance in self.distances:
                    self.files.append(path)
                    self.labels.append(self.label_map[key])
                else:
                    print(f"Skipping {basename}: No matching GT RR")
            except Exception as e:
                # skip files that don't match naming convention
                print(parts)
                print(f"Failed to read data, skipping. {e}")
                continue
        if plot_labels:
            self.plot_label_distribution()

        # Frame transform
        # self.transform = transforms.ToFrame(
        #     sensor_size=sensor_size,
        #     n_time_bins=frames_per_sample
        # )
        
    def plot_label_distribution(self):
        plt.figure(figsize=(6, 4))
        plt.hist(self.labels, bins=15, color="skyblue", edgecolor="black")
        plt.xlabel("Ground Truth Respiration Rate (bpm)")
        plt.ylabel("Count")
        plt.title("Distribution of Ground Truth RR Labels")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def transform(self, events):
        transform =    transforms.Compose(
            [
                transforms.CropTime(max = events[0][0] + self.max_time*1000000000),
                #transforms.Denoise(filter_time=1000),
                transforms.ToFrame(
                    sensor_size=self.sensor_size,
                    n_time_bins=self.frames_per_sample
                ),
            ]
        )
        return transform(events)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        basename = os.path.basename(path)
        parts = basename.split("_")
        if 'filtered.aedat4' in parts:
            parts.remove('filtered.aedat4')
                
                        
        if 'filtered2.aedat4' in parts:
            parts.remove('filtered2.aedat4')
        patient = int(parts[1])
        distance = parts[2].replace(",", ".")
        reading = int(parts[-1].replace(".aedat4", ""))

        # Read raw events and convert to frames
        
        events = read_aedat4(path)              # numpy structured array
        #print(events)
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
        #print("opening file:" + path + "with gt= " + str(gt_rr))
        return frames, label
