from pathlib import Path

import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import nibabel as nib


def extract() -> pd.DataFrame:
    """Build a DataFrame mapping subject IDs to NIfTI image paths and diagnosis labels.

    Loads NIfTI files from nifti_output_with_RID/, merges with baseline diagnosis
    from DXSUM.csv, and filters to CN (0) and AD (1) subjects only.
    """
    nifti_dir  = Path(__file__).parent.parent / "data" / "Tau_PET_Images" / "nifti_output_with_RID"
    file_list  = os.listdir(nifti_dir)

    records = []
    for file in file_list:
        if file.endswith("ResatientID.nii.gz") or file.endswith("ResatientID.json"):
            continue
        if not file.endswith(".nii.gz"):
            continue
        subject_id = file[-28:-18]
        records.append({
            "PTID":      subject_id,
            "image":     nifti_dir / file,
            "json_file": file[:-7] + ".json",
        })

    image_df = pd.DataFrame(records)

    dx    = pd.read_csv(Path(__file__).parent.parent / "data" / "DXSUM.csv")
    dx_bl = dx[dx["VISCODE"] == "bl"][["PTID", "DIAGNOSIS"]]
    dx_bl = dx_bl[dx_bl["DIAGNOSIS"].isin([1, 3])].copy()
    dx_bl["label"] = dx_bl["DIAGNOSIS"].map({1: 0, 3: 1})
    dx_bl["PTID"]  = dx_bl["PTID"].astype(str)

    return image_df.merge(dx_bl[["PTID", "label"]], on="PTID", how="inner")


class TauSliceDataset(Dataset):
    """Converts 3D PET volumes into individual 2D axial slices for CNN training.

    Slices 30–65 are extracted per volume to capture medial brain structures
    while avoiding skull and neck noise. Each slice is z-score normalized.
    """

    def __init__(self, df: pd.DataFrame, slice_range: tuple[int, int] = (30, 65)):
        self.samples: list[tuple] = []

        for _, row in df.iterrows():
            img   = nib.load(row["image"]).get_fdata()
            label = row["label"]
            for z in range(slice_range[0], slice_range[1]):
                self.samples.append((img[:, :, z], label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.samples[idx]
        image = (image - image.mean()) / (image.std() + 1e-6)
        return (
            torch.tensor(image, dtype=torch.float32).unsqueeze(0),
            torch.tensor(label, dtype=torch.long),
        )
