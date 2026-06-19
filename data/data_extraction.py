import pandas as pd
from pathlib import Path

# Define the data directory to ensure importation works correctly
data_dir = Path(__file__).parent

# Define features to extract
amy_rois = [
    "CTX_ROSTRALANTERIORCINGULATE_SUVR",
    "CTX_CAUDALANTERIORCINGULATE_SUVR", 
    "CTX_MIDDLETEMPORAL_SUVR",
    "CTX_INFERIORTEMPORAL_SUVR",
    "CTX_POSTERIORCINGULATE_SUVR",
    "CTX_PRECUNEUS_SUVR",
    "CTX_SUPERIORFRONTAL_SUVR",
    "CTX_SUPERIORPARIETAL_SUVR",
    "CTX_LATERALOCCIPITAL_SUVR",
    "CTX_FRONTALPOLE_SUVR",
    "CTX_ENTORHINAL_SUVR",
    "CTX_FUSIFORM_SUVR",
    "CTX_INFERIORPARIETAL_SUVR"
]
tau_rois = [
    "CTX_ROSTRALANTERIORCINGULATE_SUVR",
    "CTX_CAUDALANTERIORCINGULATE_SUVR", 
    "CTX_MIDDLETEMPORAL_SUVR",
    "CTX_INFERIORTEMPORAL_SUVR",
    "CTX_POSTERIORCINGULATE_SUVR",
    "CTX_PRECUNEUS_SUVR",
    "CTX_SUPERIORFRONTAL_SUVR",
    "CTX_SUPERIORPARIETAL_SUVR",
    "CTX_LATERALOCCIPITAL_SUVR",
    "CTX_FRONTALPOLE_SUVR",
    "CTX_ENTORHINAL_SUVR",
    "CTX_FUSIFORM_SUVR",
    "CTX_INFERIORPARIETAL_SUVR"
]

def extract_data():
    # Load datasets
    amy = pd.read_csv(data_dir / "UCBERKELEY_AMY_6MM.csv")
    tau = pd.read_csv(data_dir / "UCBERKELEY_TAU_6MM.csv")
    dx  = pd.read_csv(data_dir / "DXSUM.csv")

    # Extract baseline visits
    amy_bl = amy[amy["VISCODE"] == "bl"].copy()
    tau_bl = tau[tau["VISCODE"] == "bl"].copy()
    dx_bl  = dx[dx["VISCODE"] == "bl"].copy()

    # Create binary diagnosis variable: 0 for Normal (1), 1 for Alzheimer's (3). Ignore MCI (2) for simplicity.
    dx_bl_binary = dx_bl[dx_bl["DIAGNOSIS"].isin([1,3])].copy()
    dx_bl_binary["y"] = dx_bl_binary["DIAGNOSIS"].map({1: 0, 3: 1})

    # Select only necessary columns (RID and ROIs)
    amy_bl = amy[amy["VISCODE"] == "bl"][["RID"] + amy_rois].copy()
    tau_bl = tau[tau["VISCODE"] == "bl"][["RID"] + tau_rois].copy()

    # Merge with diagnosis data to create final datasets
    amy_data = amy_bl.merge(dx_bl_binary[["RID","y"]], on="RID", how="inner")
    tau_data = tau_bl.merge(dx_bl_binary[["RID","y"]], on="RID", how="inner")
    combo_data = amy_bl.merge(tau_bl, on="RID", how="inner")
    combo_data = combo_data.merge(dx_bl_binary[["RID","y"]], on="RID", how="inner")

    # Drop rows with missing values
    amy_data.dropna(inplace=True)
    tau_data.dropna(inplace=True)
    combo_data.dropna(inplace=True)

    return amy_data, tau_data, combo_data