import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
data_dir = Path(__file__).parent.parent / "data"

import pandas as pd

from data.data_extraction import tau_rois

label_map: dict[str, int] = {
    "y1":    1,
    "y2":    2,
    "y4":    4,
    "4_m12": 1,
    "4_m24": 2,
}


def viscode_to_years(viscode: str) -> int | None:
    if viscode in ["bl", "init", "4_bl", "4_init"]:
        return 0
    return label_map.get(viscode)


def extract_data() -> pd.DataFrame:
    """Load longitudinal tau PET data with diagnosis labels.

    Filters to subjects with ≥3 timepoints and binary diagnosis (CN or AD).
    Returns a DataFrame with columns: RID, time, tau ROIs, y.
    """
    tau = pd.read_csv(data_dir / "UCBERKELEY_TAU_6MM.csv")
    dx  = pd.read_csv(data_dir / "DXSUM.csv")

    tau["time"] = tau["VISCODE"].apply(viscode_to_years)
    tau = tau.dropna(subset=tau_rois + ["time"])

    tau = tau[["RID", "time"] + tau_rois]
    tau = tau.groupby(["RID", "time"], as_index=False)[tau_rois].mean()

    valid_rids = tau.groupby("RID")["time"].nunique()
    tau = tau[tau["RID"].isin(valid_rids[valid_rids >= 3].index)]

    dx_bl = dx[dx["VISCODE"] == "bl"][["RID", "DIAGNOSIS"]]
    dx_bl = dx_bl[dx_bl["DIAGNOSIS"].isin([1, 3])].copy()
    dx_bl["y"] = dx_bl["DIAGNOSIS"].map({1: 0, 3: 1})

    return tau.merge(dx_bl[["RID", "y"]], on="RID", how="inner")
