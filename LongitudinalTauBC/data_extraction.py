import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent)) # Ensure project root is in path
data_dir = Path(__file__).parent.parent / "data"

import pandas as pd

from data.data_extraction import tau_rois

label_map = { # convert data labels to their corresponding time in years
    "y1": 1,
    "y2": 2,
    "y4": 4,
    "4_m12": 1,
    "4_m24": 2,
}

# Helper function to convert VISCODE to years
def viscode_to_years(viscode):
    if viscode in ["bl", "init", "4_bl", "4_init"]:
        return 0
    if viscode in label_map:
        return label_map[viscode]
    return None

def extract_data():
    # Load datasets
    tau = pd.read_csv(data_dir / "UCBERKELEY_TAU_6MM.csv")
    dx  = pd.read_csv(data_dir / "DXSUM.csv")

    # Create new column 'time' from VISCODE and drop rows with missing ROIs
    tau["time"] = tau["VISCODE"].apply(viscode_to_years)
    tau = tau.dropna(subset=tau_rois)

    # Drop missing values in ROIs and collapse duplicates
    tau = tau[["RID", "time"] + tau_rois]
    tau = (tau.groupby(["RID", "time"], as_index=False)[tau_rois].mean())

    # Count timepoints per subject
    tp_counts = tau.groupby("RID")["time"].nunique()

    # Keep subjects with at least 3 timepoints (to ensure more accurate results in how tau progresses)
    valid_rids = tp_counts[tp_counts >= 3].index
    tau = tau[tau["RID"].isin(valid_rids)]
    
    # Keep only subjects with diagnosis of Normal (1) or Alzheimer's (3) at baseline (no MCI = mild cognitive impairment)
    dx_bl = dx[dx["VISCODE"] == "bl"][["RID", "DIAGNOSIS"]]
    dx_bl = dx_bl[dx_bl["DIAGNOSIS"].isin([1, 3])]
    dx_bl["y"] = dx_bl["DIAGNOSIS"].map({1: 0, 3: 1})

    # Merge with diagnosis data to create final dataset
    tau = tau.merge(dx_bl[["RID", "y"]], on="RID", how="inner")
    
    return tau