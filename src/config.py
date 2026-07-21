"""Project-wide constants and paths."""
from pathlib import Path

RANDOM_SEED = 42
N_FOLDS = 5
MIN_GROUP_COUNT = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

TRAIN_PATH = DATA_RAW_DIR / "train.csv"
TEST_PATH = DATA_RAW_DIR / "test.csv"

CSV_READ_KWARGS = {"sep": ";", "decimal": ","}

ID_COL = "id"
TARGET_COL = "target"
WEIGHT_COL = "w"
DATE_COL = "dt"

# The 8 genuinely categorical (text) columns, confirmed programmatically in
# 01_eda.ipynb via the numeric-string regex heuristic (see preprocessing.py).
CATEGORICAL_COLS = [
    "dt",
    "gender",
    "adminarea",
    "city_smart_name",
    "dp_ewb_last_employment_position",
    "addrref",
    "dp_address_unique_regions",
    "period_last_act_ad",
]

REGION_COL = "adminarea"

NON_FEATURE_COLS = [ID_COL, TARGET_COL, WEIGHT_COL]

# Feature-family prefixes used to group columns during EDA (missingness,
# multicollinearity). Order matters only for readability of groupby output.
FEATURE_FAMILY_PREFIXES = [
    "hdb_bki_",
    "bki_",
    "dp_ils_",
    "dp_payoutincomedata_",
    "dp_ewb_",
    "dp_address_",
    "turn_cur_",
    "turn_other_",
    "turn_save_",
    "turn_fdep_",
    "avg_cur_",
    "avg_debet_",
    "avg_credit_",
    "curr_rur_amt_",
    "dda_rur_amt_",
    "loanacc_rur_amt_",
    "express_rur_amt_",
    "total_rur_amt_",
    "by_category__",
    "amount_by_category__",
    "avg_by_category__",
    "transaction_category_",
    "mob_",
    "device_",
    "vert_has_app_",
    "loan",
    "pil",
    "acard",
    "ovrd",
]
