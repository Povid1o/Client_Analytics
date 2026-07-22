"""Leakage-free feature groups for the income model.

Every function in this module uses input features only. Target-based regional
statistics live in ``region_encoding.py`` because they require fold-aware
fitting.
"""
from collections.abc import Iterable

import numpy as np
import pandas as pd


INCOME_ANCHORS = [
    "salary_6to12m_avg",
    "incomeValue",
    "dp_ils_avg_salary_1y",
    "dp_payoutincomedata_payout_avg_3_month",
    "dp_payoutincomedata_payout_avg_6_month",
    "dp_payoutincomedata_payout_avg_prev_year",
]

DEBT_COLUMNS = [
    "hdb_outstand_sum",
    "hdb_other_outstand_sum",
    "hdb_relend_outstand_sum",
    "hdb_bki_active_cc_max_outstand",
    "hdb_bki_other_active_pil_outstanding",
    "hdb_bki_other_active_ip_outstanding",
]

RATIO_PAIRS = {
    "util_cc_active": ("hdb_bki_active_cc_max_outstand", "hdb_bki_active_cc_max_limit"),
    "util_total": ("hdb_outstand_sum", "hdb_bki_total_max_limit"),
    "util_relend": ("hdb_relend_outstand_sum", "hdb_bki_total_max_limit"),
    "limit_cc_share": ("hdb_bki_total_cc_max_limit", "hdb_bki_total_max_limit"),
    "limit_pil_share": ("hdb_bki_total_pil_max_limit", "hdb_bki_total_max_limit"),
    "limit_ip_share": ("hdb_bki_total_ip_max_limit", "hdb_bki_total_max_limit"),
}

FLOW_PAIRS = {
    "cur_avg12": ("turn_cur_cr_avg_v2", "turn_cur_db_avg_v2"),
    "cur_sum12": ("turn_cur_cr_sum_v2", "turn_cur_db_sum_v2"),
    "cur_active": ("turn_cur_cr_avg_act_v2", "turn_cur_db_avg_act_v2"),
    "cur_7avg": ("turn_cur_cr_7avg_avg_v2", "turn_cur_db_7avg_avg_v2"),
    "cur_turn": ("avg_cur_cr_turn", "avg_cur_db_turn"),
}

TREND_PAIRS = {
    "salary_1y_to_2y": ("dp_ils_avg_salary_1y", "dp_ils_avg_salary_2y"),
    "salary_1y_to_3y": ("dp_ils_avg_salary_1y", "dp_ils_avg_salary_3y"),
    "ils_payment_6m_to_12m": ("dp_ils_paymentssum_avg_6m", "dp_ils_paymentssum_avg_12m"),
    "ils_payment_current6m_to_12m": (
        "dp_ils_paymentssum_avg_6m_current",
        "dp_ils_paymentssum_avg_12m",
    ),
    "ils_accpayment_6m_to_12m": ("dp_ils_accpayment_avg_6m", "dp_ils_accpayment_avg_12m"),
    "ils_accpayment_3m_to_12m": ("dp_ils_accpayment_avg_3m", "dp_ils_accpayment_avg_12m"),
    "payout_3m_to_6m": (
        "dp_payoutincomedata_payout_avg_3_month",
        "dp_payoutincomedata_payout_avg_6_month",
    ),
    "spend_3m_to_6m": ("avg_3m_all", "avg_6m_all"),
    "profit_l2m_to_12m": ("profit_income_out_rur_amt_l2m", "profit_income_out_rur_amt_12m"),
}

REGIONAL_BENCHMARKS = ["per_capita_income_rur_amt", "salary_median_in_gex_r1"]

LOG_RANK_COLUMNS = list(
    dict.fromkeys(
        INCOME_ANCHORS
        + DEBT_COLUMNS
        + [column for pair in FLOW_PAIRS.values() for column in pair]
        + ["hdb_bki_total_max_limit", "avg_3m_all", "avg_6m_all"]
    )
)


def _numeric(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def safe_ratio(numerator, denominator, max_abs=1_000.0):
    """Divide finite values, treating zero denominators as missing."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    np.divide(numerator, denominator, out=result, where=valid)
    result[np.abs(result) > max_abs] = np.nan
    return result


def _existing(frame, columns: Iterable[str]):
    return [column for column in columns if column in frame.columns]


def add_anchor_agreement(frame):
    result = frame.copy()
    columns = _existing(result, INCOME_ANCHORS)
    anchors = result[columns].apply(pd.to_numeric, errors="coerce").where(lambda x: x >= 0)
    result["fe_income_anchor_count"] = anchors.notna().sum(axis=1).astype(float)
    result["fe_income_anchor_mean"] = anchors.mean(axis=1)
    result["fe_income_anchor_median"] = anchors.median(axis=1)
    result["fe_income_anchor_std"] = anchors.std(axis=1, ddof=0)
    result["fe_income_anchor_min"] = anchors.min(axis=1)
    result["fe_income_anchor_max"] = anchors.max(axis=1)
    result["fe_income_anchor_cv"] = safe_ratio(
        result["fe_income_anchor_std"], result["fe_income_anchor_mean"]
    )
    result["fe_income_anchor_range_rel"] = safe_ratio(
        result["fe_income_anchor_max"] - result["fe_income_anchor_min"],
        result["fe_income_anchor_median"],
    )
    return result


def add_scale_normalization(frame):
    result = frame if "fe_income_anchor_median" in frame else add_anchor_agreement(frame)
    income = result["fe_income_anchor_median"]
    for column in _existing(result, DEBT_COLUMNS):
        result[f"fe_{column}_to_income"] = safe_ratio(_numeric(result, column), income)
    for name, (numerator, denominator) in RATIO_PAIRS.items():
        if numerator in result and denominator in result:
            result[f"fe_{name}"] = safe_ratio(
                _numeric(result, numerator), _numeric(result, denominator)
            )
    return result


def add_flow_balance(frame):
    result = frame.copy()
    for name, (credit, debit) in FLOW_PAIRS.items():
        if credit not in result or debit not in result:
            continue
        cr = _numeric(result, credit)
        db = _numeric(result, debit)
        result[f"fe_{name}_cr_to_db"] = safe_ratio(cr, db)
        result[f"fe_{name}_balance"] = safe_ratio(cr - db, cr.abs() + db.abs())
    return result


def add_trends(frame):
    result = frame.copy()
    for name, (recent, long_window) in TREND_PAIRS.items():
        if recent in result and long_window in result:
            result[f"fe_trend_{name}"] = safe_ratio(
                _numeric(result, recent), _numeric(result, long_window)
            )
    return result


def add_activity_recency(frame):
    """Express last account activity as a duration relative to the row date."""
    result = frame.copy()
    if "dt" not in result or "period_last_act_ad" not in result:
        return result
    current = pd.to_datetime(result["dt"], errors="coerce")
    last_activity = pd.to_datetime(result["period_last_act_ad"], errors="coerce")
    days = (current - last_activity).dt.days.astype(float)
    result["fe_days_since_last_activity"] = days.clip(lower=0.0, upper=365.0)
    result["fe_missing_last_activity"] = last_activity.isna().astype("int8")
    return result


def add_regional_normalization(frame):
    result = frame if "fe_income_anchor_median" in frame else add_anchor_agreement(frame)
    for benchmark in _existing(result, REGIONAL_BENCHMARKS):
        denominator = _numeric(result, benchmark)
        for anchor in _existing(result, INCOME_ANCHORS):
            result[f"fe_{anchor}_to_{benchmark}"] = safe_ratio(
                _numeric(result, anchor), denominator
            )
    return result


def add_expense_shares(frame):
    result = frame.copy()
    category_columns = [
        column
        for column in result.columns
        if column.startswith("avg_by_category__amount__sum__")
    ]
    if "avg_6m_all" in result:
        total = _numeric(result, "avg_6m_all")
        for column in category_columns:
            result[f"fe_share_{column}"] = safe_ratio(_numeric(result, column), total)

    amount_90d = [
        column for column in result.columns if column.startswith("amount_by_category_90d__")
    ]
    if "avg_3m_all" in result:
        total_3m = _numeric(result, "avg_3m_all")
        for column in amount_90d:
            result[f"fe_share_{column}"] = safe_ratio(_numeric(result, column), total_3m)
    return result


def add_log_rank(frame, rank_columns=None):
    result = frame.copy()
    for column in _existing(result, rank_columns or LOG_RANK_COLUMNS):
        values = _numeric(result, column)
        result[f"fe_log1p_{column}"] = np.sign(values) * np.log1p(np.abs(values))
        result[f"fe_rank_{column}"] = values.rank(method="average", pct=True)
    return result


def select_missing_flag_columns(frame, min_rate=0.01, max_rate=0.95):
    return [
        column
        for column in frame.columns
        if min_rate <= frame[column].isna().mean() <= max_rate
    ]


def add_missing_flags(frame, columns=None):
    result = frame.copy()
    columns = columns or select_missing_flag_columns(result)
    flags = {
        f"fe_missing_{column}": result[column].isna().astype("int8")
        for column in _existing(result, columns)
    }
    return pd.concat([result, pd.DataFrame(flags, index=result.index)], axis=1)


GROUP_BUILDERS = {
    "anchors": add_anchor_agreement,
    "scale": add_scale_normalization,
    "flows": add_flow_balance,
    "trends": add_trends,
    "recency": add_activity_recency,
    "regional_norm": add_regional_normalization,
    "expense_shares": add_expense_shares,
    "log_rank": add_log_rank,
    "missing_flags": add_missing_flags,
}


def add_feature_groups(frame, groups, *, rank_columns=None, missing_columns=None):
    """Add requested groups in a stable order and return a new frame."""
    result = frame.copy()
    for group in groups:
        if group not in GROUP_BUILDERS:
            raise ValueError(f"unknown feature group: {group}")
        if group == "log_rank":
            result = add_log_rank(result, rank_columns=rank_columns)
        elif group == "missing_flags":
            result = add_missing_flags(result, columns=missing_columns)
        else:
            result = GROUP_BUILDERS[group](result)
    return result
