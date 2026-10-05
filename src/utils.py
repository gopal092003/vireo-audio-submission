# src/utils.py

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"


def ensure_directories() -> None:
    """Create directories used by the pipeline if they do not exist."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_csv(filename: str, **kwargs) -> pd.DataFrame:
    """
    Load a CSV from data/raw.

    Example:
        tickets = load_csv("tickets.csv")
    """
    path = RAW_DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    """
    Save a DataFrame to the output directory.
    """
    ensure_directories()

    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)

    return path


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def clean_string(value) -> str:
    """
    Convert a value to a clean string.

    Missing values become an empty string.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_text(value) -> str:
    """
    Basic text normalization for customer messages and agent notes.

    This intentionally does NOT aggressively transform the text because
    complaint classification may depend on wording.
    """
    text = clean_string(value)

    if not text:
        return ""

    # Normalize whitespace without changing the actual words.
    return " ".join(text.split())


def to_numeric(
    series: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    """
    Convert a pandas Series to numeric values.

    Invalid/missing values are replaced with default.
    """
    return pd.to_numeric(series, errors="coerce").fillna(default)


# ---------------------------------------------------------------------------
# Date/time helpers
# ---------------------------------------------------------------------------

def parse_datetime(
    series: pd.Series,
) -> pd.Series:
    """
    Parse ticket timestamps.

    The Vireo README states that exported helpdesk timestamps are displayed
    in IST. We therefore preserve the timestamps as local-naive datetimes
    rather than silently converting them to another timezone.
    """
    return pd.to_datetime(series, errors="coerce")


def add_week_fields(
    df: pd.DataFrame,
    date_column: str = "created_at",
) -> pd.DataFrame:
    """
    Add useful weekly reporting fields.

    Adds:
        week_start
        week_label
    """
    result = df.copy()

    result[date_column] = parse_datetime(result[date_column])

    # Monday is the start of the reporting week.
    result["week_start"] = (
        result[date_column]
        - pd.to_timedelta(result[date_column].dt.weekday, unit="D")
    ).dt.normalize()

    result["week_label"] = result["week_start"].dt.strftime("%Y-%m-%d")

    return result


# ---------------------------------------------------------------------------
# Ticket helpers
# ---------------------------------------------------------------------------

def is_completed_ticket(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Return a boolean mask for tickets that count as completed attendance.

    Vireo policy defines attendance as tickets with status:
        resolved OR closed
    """
    return df["status"].astype(str).str.lower().isin(
        {"resolved", "closed"}
    )


def add_completion_flag(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a boolean 'is_completed' column.
    """
    result = df.copy()
    result["is_completed"] = is_completed_ticket(result)

    return result


def add_first_response_minutes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate first-response time in minutes.

    Requires:
        created_at
        first_response_at
    """
    result = df.copy()

    created = parse_datetime(result["created_at"])
    response = parse_datetime(result["first_response_at"])

    result["first_response_minutes"] = (
        response - created
    ).dt.total_seconds() / 60

    return result


def add_handle_time_minutes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate handle time in minutes.

    Vireo policy defines handle time as:
        first response -> resolution
    """
    result = df.copy()

    response = parse_datetime(result["first_response_at"])
    resolved = parse_datetime(result["resolved_at"])

    result["handle_time_minutes"] = (
        resolved - response
    ).dt.total_seconds() / 60

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def require_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    dataframe_name: str = "DataFrame",
) -> None:
    """
    Raise a useful error when expected columns are missing.
    """
    required = set(required_columns)
    actual = set(df.columns)

    missing = sorted(required - actual)

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: "
            f"{', '.join(missing)}"
        )


def validate_tickets(df: pd.DataFrame) -> None:
    """
    Validate the minimum schema required by the MVP pipeline.
    """
    require_columns(
        df,
        [
            "ticket_id",
            "created_at",
            "resolved_at",
            "status",
            "channel",
            "agent_id",
            "customer_message",
            "agent_notes",
        ],
        dataframe_name="tickets.csv",
    )


def validate_agents(df: pd.DataFrame) -> None:
    """
    Validate the minimum agent roster schema.
    """
    require_columns(
        df,
        [
            "agent_id",
            "name",
            "team",
            "shift",
            "tier",
            "from_date",
            "to_date",
        ],
        dataframe_name="agents.csv",
    )


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def safe_divide(
    numerator,
    denominator,
    default: float = 0.0,
):
    """
    Divide safely without raising ZeroDivisionError.
    """
    if denominator in (0, None) or pd.isna(denominator):
        return default

    return numerator / denominator


def pct_change(
    current: float,
    previous: float,
) -> Optional[float]:
    """
    Calculate percentage change.

    Returns None when the previous value is zero/missing.
    """
    if previous in (0, None) or pd.isna(previous):
        return None

    return ((current - previous) / previous) * 100


def truncate_text(
    text: str,
    max_chars: int = 300,
) -> str:
    """
    Truncate long customer messages for digest output.
    """
    text = clean_text(text)

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3].rstrip() + "..."