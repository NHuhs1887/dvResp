
# aedat_dataset.py

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import tonic.transforms as transforms
from tonic.io import read_aedat4
import matplotlib.pyplot as plt


def normalize_distance(value):
    """Normalize a distance token to a canonical form, e.g.
    "1m" / "1,0m" / "1.0m" -> "1.0m", "0,5m" -> "0.5m".
    Makes CSV distances ("1m", "2m") match filename distances ("1,0m", "1.0m")."""
    s = str(value).strip().replace(",", ".").lower()
    if s.endswith("m"):
        s = s[:-1]
    try:
        return f"{float(s):.1f}m"
    except ValueError:
        return f"{s}m"


def _clean_parts(basename):
    """Split a filename into underscore parts, dropping the 'filtered' tokens."""
    parts = basename.split("_")
    return [p for p in parts if p not in ("filtered.aedat4", "filtered2.aedat4")]


class AEDATRespirationDataset(Dataset):
    def __init__(self,
                 data_dirs=None,
                 csv_path: str = "../data/ground_truth.csv",
                 data_dir=None,
                 sensor_size=(640, 480, 2),
                 frames_per_sample=900,
                 filtered=True,
                 plot_labels=False,
                 max_time=30,
                 rr_range=(8.0, 30.0)):

        """
        Dataset for AEDAT4 event files paired with GT respiration rates.
        Only files with a matching GT RR in the CSV (within ``rr_range``) are kept.

        ``data_dirs`` maps a dataset name to its recordings directory, e.g.
            {"042025": "../data/042025",
             "042026": "../data/042026"}
        The dataset name is matched against the CSV's ``Dataset`` column so that
        patient numbers that collide across datasets stay distinct.

        A single directory may also be passed (via ``data_dirs`` or the legacy
        ``data_dir`` argument); its dataset name is then inferred from the folder
        basename (e.g. ".../042026" -> "042026"), which matches the convention used
        in the ``Dataset`` column. If the CSV has no ``Dataset`` column the name is
        ignored and matching falls back to (Patient, Distance, Reading).
        """
        if data_dirs is None:
            data_dirs = data_dir
        self.sensor_size = sensor_size
        self.frames_per_sample = frames_per_sample
        self.filtered = filtered
        self.max_time = max_time

        # Load CSV and build lookup: (Dataset, Patient, Distance, Reading) -> GT RR.
        # Read Dataset as a string so names like "042025" keep their leading zero
        # (otherwise pandas parses it as the int 42025 and matching fails).
        header = pd.read_csv(csv_path, nrows=0).columns
        dtype = {"Dataset": str} if "Dataset" in header else None
        df = pd.read_csv(csv_path, dtype=dtype)
        has_dataset = "Dataset" in df.columns
        self.label_map = {}
        for _, r in df.iterrows():
            if pd.isna(r["GT RR"]):
                continue
            try:
                gt = float(r["GT RR"])
            except (ValueError, TypeError):
                continue
            if not (rr_range[0] <= gt <= rr_range[1]):
                continue
            dataset = str(r["Dataset"]) if has_dataset and pd.notna(r["Dataset"]) else None
            key = (dataset, int(r["Patient"]), normalize_distance(r["Distance"]), int(r["Reading"]))
            self.label_map[key] = gt

        # Resolve data_dirs to a {dataset_name: directory} mapping. A single dir
        # gets its name from the folder basename so it lines up with the CSV.
        if isinstance(data_dirs, (str, os.PathLike)):
            d = str(data_dirs)
            name = os.path.basename(os.path.normpath(d)) if has_dataset else None
            data_dirs = {name: d}
        self.data_dirs = data_dirs

        # Gather matching AEDAT4 files across every dataset directory.
        self.files = []
        self.labels = []
        self.keys = []
        for dataset_name, directory in self.data_dirs.items():
            if not os.path.isdir(directory):
                print(f"[AEDATRespirationDataset] directory not found, skipping: {directory}")
                continue
            key_dataset = dataset_name if has_dataset else None
            for fname in sorted(os.listdir(directory)):
                if not fname.endswith(".aedat4"):
                    continue
                parts = _clean_parts(fname)
                if self.filtered and "filtered" not in fname:
                    continue
                try:
                    patient = int(parts[1])
                    distance = normalize_distance(parts[2])
                    reading = int(parts[-1].replace(".aedat4", ""))
                except (ValueError, IndexError):
                    # filename doesn't match the naming convention
                    continue
                key = (key_dataset, patient, distance, reading)
                if key in self.label_map:
                    self.files.append(os.path.join(directory, fname))
                    self.labels.append(self.label_map[key])
                    self.keys.append(key)

        print(f"[AEDATRespirationDataset] matched {len(self.files)} files "
              f"from {len(self.data_dirs)} dataset(s).")
        if plot_labels:
            self.plot_label_distribution()

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
        transform = transforms.Compose(
            [
                transforms.CropTime(max=events[0][0] + self.max_time * 1000000000),
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

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return frames, label
