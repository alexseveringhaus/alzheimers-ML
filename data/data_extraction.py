import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent

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
    "CTX_INFERIORPARIETAL_SUVR",
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
    "CTX_INFERIORPARIETAL_SUVR",
]


def extract_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and merge amyloid PET, tau PET, and diagnosis data at baseline.

    Returns amyloid-only, tau-only, and combined DataFrames, each with a
    binary label column 'y' (0 = CN, 1 = AD). MCI subjects are excluded.
    """
    amy = pd.read_csv(data_dir / "UCBERKELEY_AMY_6MM.csv")
    tau = pd.read_csv(data_dir / "UCBERKELEY_TAU_6MM.csv")
    dx  = pd.read_csv(data_dir / "DXSUM.csv")

    amy_bl = amy[amy["VISCODE"] == "bl"][["RID"] + amy_rois].copy()
    tau_bl = tau[tau["VISCODE"] == "bl"][["RID"] + tau_rois].copy()

    dx_bl = dx[dx["VISCODE"] == "bl"].copy()
    dx_bl = dx_bl[dx_bl["DIAGNOSIS"].isin([1, 3])].copy()
    dx_bl["y"] = dx_bl["DIAGNOSIS"].map({1: 0, 3: 1})

    amy_data   = amy_bl.merge(dx_bl[["RID", "y"]], on="RID", how="inner").dropna()
    tau_data   = tau_bl.merge(dx_bl[["RID", "y"]], on="RID", how="inner").dropna()
    combo_data = (
        amy_bl.merge(tau_bl, on="RID", how="inner")
        .merge(dx_bl[["RID", "y"]], on="RID", how="inner")
        .dropna()
    )

    return amy_data, tau_data, combo_data
