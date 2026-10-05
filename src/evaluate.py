# src/evaluate.py

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import OUTPUT_DIR


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_SIZE = 50

# Existing categories in tickets.csv are useful as a reference signal,
# but they are not treated as perfect ground truth.
CATEGORY_COLUMN = "category"
AI_CATEGORY_COLUMN = "complaint_category"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_tickets_for_review(
    tickets: pd.DataFrame,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Select a reproducible sample for manual evaluation.

    The sample is stratified by existing ticket category when possible,
    so the evaluation does not accidentally contain only the most common
    complaint type.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")

    df = tickets.copy()

    if len(df) <= sample_size:
        return df.reset_index(drop=True)

    if CATEGORY_COLUMN not in df.columns:
        return df.sample(
            n=sample_size,
            random_state=random_state,
        ).reset_index(drop=True)

    # Try proportional/stratified sampling.
    groups = []

    categories = (
        df[CATEGORY_COLUMN]
        .fillna("Unknown")
        .astype(str)
    )

    df["_eval_category"] = categories

    category_counts = df["_eval_category"].value_counts()

    # Allocate approximately proportional sample sizes.
    raw_sizes = (
        category_counts / len(df) * sample_size
    )

    allocations = raw_sizes.round().astype(int)

    # Make sure every sufficiently represented category gets a chance.
    allocations = allocations.clip(
        upper=category_counts
    )

    # Adjust total to exactly sample_size.
    while allocations.sum() < sample_size:
        remaining = (
            category_counts - allocations
        ).sort_values(ascending=False)

        added = False

        for category in remaining.index:
            if allocations[category] < category_counts[category]:
                allocations[category] += 1
                added = True

                if allocations.sum() >= sample_size:
                    break

        if not added:
            break

    while allocations.sum() > sample_size:
        removable = allocations[
            allocations > 1
        ].sort_values(ascending=False)

        if removable.empty:
            break

        allocations[removable.index[0]] -= 1

    for category, n in allocations.items():
        if n <= 0:
            continue

        subset = df[
            df["_eval_category"] == category
        ]

        groups.append(
            subset.sample(
                n=min(n, len(subset)),
                random_state=random_state,
            )
        )

    result = pd.concat(
        groups,
        ignore_index=True,
    )

    result = result.sample(
        frac=1,
        random_state=random_state,
    )

    return (
        result
        .drop(columns=["_eval_category"], errors="ignore")
        .head(sample_size)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Reference-category comparison
# ---------------------------------------------------------------------------

def compare_with_existing_category(
    tickets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare AI classifications with the existing ticket category.

    This is an AGREEMENT measure, not a ground-truth accuracy measure.

    Returns one row per ticket with:
        ticket_id
        existing_category
        ai_category
        agreement
        confidence
    """
    required = {
        "ticket_id",
        CATEGORY_COLUMN,
        AI_CATEGORY_COLUMN,
    }

    missing = required - set(tickets.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    result = tickets[
        [
            "ticket_id",
            CATEGORY_COLUMN,
            AI_CATEGORY_COLUMN,
        ]
        + (
            ["classification_confidence"]
            if "classification_confidence" in tickets.columns
            else []
        )
    ].copy()

    result["existing_category"] = (
        result[CATEGORY_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["ai_category"] = (
        result[AI_CATEGORY_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["agreement"] = (
        result["existing_category"]
        == result["ai_category"]
    )

    result = result.drop(
        columns=[
            CATEGORY_COLUMN,
            AI_CATEGORY_COLUMN,
        ]
    )

    return result


def calculate_reference_agreement(
    tickets: pd.DataFrame,
) -> dict:
    """
    Calculate AI-vs-existing-category agreement.

    IMPORTANT:
    This should be reported as agreement with the intake/agent category,
    not as validated accuracy.
    """
    comparison = compare_with_existing_category(tickets)

    valid = comparison[
        (comparison["existing_category"] != "")
        & (comparison["ai_category"] != "")
    ]

    if valid.empty:
        return {
            "n": 0,
            "agreement": None,
            "disagreement_count": 0,
        }

    agreement = valid["agreement"].mean()

    return {
        "n": int(len(valid)),
        "agreement": round(float(agreement), 4),
        "agreement_pct": round(
            float(agreement * 100),
            2,
        ),
        "disagreement_count": int(
            (~valid["agreement"]).sum()
        ),
    }


# ---------------------------------------------------------------------------
# Manual review sheet
# ---------------------------------------------------------------------------

def create_manual_review_file(
    tickets: pd.DataFrame,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create a CSV designed for human review.

    Reviewer should fill:
        human_category
        correct

    The existing category is shown only as a reference.
    """
    sample = sample_tickets_for_review(
        tickets,
        sample_size=sample_size,
    )

    columns = [
        "ticket_id",
        "created_at",
        "channel",
        "category",
        "complaint_category",
        "classification_confidence",
        "customer_message",
        "agent_notes",
    ]

    available = [
        column
        for column in columns
        if column in sample.columns
    ]

    review = sample[available].copy()

    review["human_category"] = ""
    review["correct"] = ""

    # Put review columns at the end so the sheet is easy to fill.
    ordered_columns = available + [
        "human_category",
        "correct",
    ]

    review = review[ordered_columns]

    if output_path is None:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            OUTPUT_DIR
            / "evaluation_review.csv"
        )

    review.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Manual evaluation file written to: {output_path}"
    )

    return review


# ---------------------------------------------------------------------------
# Human-reviewed evaluation
# ---------------------------------------------------------------------------

def evaluate_manual_review(
    review_df: pd.DataFrame,
) -> dict:
    """
    Calculate evaluation metrics after the reviewer fills the CSV.

    Expected columns:
        complaint_category
        human_category
        correct

    'correct' may contain:
        yes / y / true / 1
        no / n / false / 0
    """
    required = {
        "complaint_category",
        "human_category",
    }

    missing = required - set(review_df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = review_df.copy()

    df["human_category"] = (
        df["human_category"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Only evaluate rows where a human label was actually entered.
    reviewed = df[
        df["human_category"].ne("")
    ].copy()

    if reviewed.empty:
        return {
            "sample_size": int(len(df)),
            "reviewed_count": 0,
            "accuracy": None,
            "error_count": 0,
            "status": "No human-reviewed rows yet.",
        }

    reviewed["ai_category"] = (
        reviewed["complaint_category"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    reviewed["correct_calculated"] = (
        reviewed["ai_category"]
        == reviewed["human_category"]
    )

    error_count = int(
        (~reviewed["correct_calculated"]).sum()
    )

    accuracy = (
        reviewed["correct_calculated"].mean()
    )

    return {
        "sample_size": int(len(df)),
        "reviewed_count": int(len(reviewed)),
        "accuracy": round(
            float(accuracy),
            4,
        ),
        "accuracy_pct": round(
            float(accuracy * 100),
            2,
        ),
        "error_count": error_count,
        "status": "Complete",
    }


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def build_error_analysis(
    review_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Show where the classifier disagreed with human review.

    Useful for the take-home's 'wrong cases' discussion.
    """
    required = {
        "ticket_id",
        "complaint_category",
        "human_category",
    }

    missing = required - set(review_df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = review_df.copy()

    df["human_category"] = (
        df["human_category"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["complaint_category"] = (
        df["complaint_category"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    errors = df[
        df["human_category"].ne("")
        & (
            df["complaint_category"]
            != df["human_category"]
        )
    ].copy()

    if errors.empty:
        return pd.DataFrame(
            columns=[
                "ticket_id",
                "ai_category",
                "human_category",
                "customer_message",
            ]
        )

    output_columns = [
        "ticket_id",
        "complaint_category",
        "human_category",
    ]

    for column in [
        "classification_confidence",
        "customer_message",
        "agent_notes",
    ]:
        if column in errors.columns:
            output_columns.append(column)

    return errors[output_columns].rename(
        columns={
            "complaint_category": "ai_category",
        }
    )


def build_confusion_matrix(
    review_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a simple AI-vs-human category confusion matrix.
    """
    required = {
        "complaint_category",
        "human_category",
    }

    missing = required - set(review_df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = review_df.copy()

    df["human_category"] = (
        df["human_category"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["human_category"].ne("")
    ]

    return pd.crosstab(
        df["human_category"],
        df["complaint_category"],
        margins=True,
    )


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

def build_evaluation_report(
    tickets: pd.DataFrame,
    review_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Produce the complete evaluation report.

    Before manual review:
        - reports AI-vs-existing-category agreement
        - creates a 50-ticket review basis

    After manual review:
        - adds human-reviewed accuracy
        - adds error count
        - adds error analysis
    """
    report = {
        "evaluation_method": (
            "Existing ticket category is treated as a reference signal, "
            "not as perfect ground truth. A manually reviewed sample is "
            "used for validated accuracy."
        ),
        "reference_category_agreement": (
            calculate_reference_agreement(tickets)
        ),
    }

    if review_df is not None:
        report["manual_review"] = (
            evaluate_manual_review(review_df)
        )

        errors = build_error_analysis(review_df)

        report["manual_error_count"] = int(
            len(errors)
        )

        if not errors.empty:
            report["common_errors"] = (
                errors["human_category"]
                .value_counts()
                .head(10)
                .to_dict()
            )
        else:
            report["common_errors"] = {}

    return report


# ---------------------------------------------------------------------------
# Save report
# ---------------------------------------------------------------------------

def save_evaluation_report(
    report: dict,
    filename: str = "evaluation.json",
) -> Path:
    """
    Save evaluation results as JSON.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = OUTPUT_DIR / filename

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return path


# ---------------------------------------------------------------------------
# End-to-end evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    tickets: pd.DataFrame,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict:
    """
    Run the first-stage evaluation.

    Creates:
        output/evaluation_review.csv
        output/evaluation.json

    The JSON initially contains reference-category agreement.
    Human review can then be performed in the CSV.
    """
    review_df = create_manual_review_file(
        tickets,
        sample_size=sample_size,
    )

    report = build_evaluation_report(
        tickets=tickets,
    )

    save_evaluation_report(report)

    return {
        "report": report,
        "review": review_df,
    }