from pathlib import Path

import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import nibabel as nib

def extract():
    file_list = os.listdir(Path(__file__).parent.parent / 'data' / 'Tau_PET_Images' / 'nifti_output_with_RID')

    cleaned_file_list = [] # remove unnecessary data
    for file in file_list:
        if file[-18:] == 'ResatientID.nii.gz' or file[-16:] == 'ResatientID.json':
            continue
        cleaned_file_list.append(file)

    # Create records list to be converted to dataframe containing patient ID, image path, and json metadata path
    records = []
    for file in cleaned_file_list:
        if file.endswith('.nii.gz'):
            subject_id = file[-28:-18]
            image_path = json_file = Path(__file__).parent.parent / Path(
                "data/Tau_PET_Images/nifti_output_with_RID/",
                file)
            json_file = file[:-7] + '.json'
            records.append({
                'PTID': subject_id,
                'image': image_path,
                'json_file': json_file
            })

    image_df = pd.DataFrame(records)

    # Build combined dataframe with images and their diagnosis labels
    dx = pd.read_csv(Path(__file__).parent.parent / 'data' / "DXSUM.csv")

    dx_bl = dx[dx["VISCODE"] == "bl"][["PTID", "DIAGNOSIS"]]

    # Create binary diagnosis variable: 0 for Normal (1), 1 for Alzheimer's (3). Ignore MCI (2) for simplicity.
    dx_bl = dx_bl[dx_bl["DIAGNOSIS"].isin([1,3])]
    dx_bl["label"] = dx_bl["DIAGNOSIS"].map({1: 0, 3: 1})

    dx_bl["PTID"] = dx_bl["PTID"].astype(str)

    combined_df = image_df.merge(dx_bl[["PTID", "label"]], on="PTID", how="inner")

    return combined_df

# Class to convert ~600 3-D subject scans into ~17,000 2-D samples
class TauSliceDataset(Dataset):
    # Default slice range is set between 30 and 65 to get medial data (avoid noise in skill/neck)
    def __init__(self, df, slice_range=(30, 65)):
        self.samples = []

        for _, row in df.iterrows():
            img = nib.load(row["image"]).get_fdata() 
            label = row["label"]
            for z in range(slice_range[0], slice_range[1]): 
                slice_2d = img[:, :, z] 
                self.samples.append((slice_2d, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        image = (image - image.mean()) / (image.std() + 1e-6) # Normalize

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return image, label