# src/analyze.py

from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import (
    add_completion_flag,
    add_first_response_minutes,
    add_handle_time_minutes,
    add_week_fields,
    clean_text,
    pct_change,
    safe_divide,
)


# ---------------------------------------------------------------------------
# Business constants
# ---------------------------------------------------------------------------

# Vireo policy: first-response targets by channel, in minutes.
SLA_MINUTES = {
    "chat": 15,
    "voice": 120,
    "social": 240,
    "email": 480,
}

# Vireo policy: fully loaded cost per contact.
CONTACT_COST_INR = {
    "chat": 210,
    "email": 260,
    "voice": 520,
    "social": 240,
}

TRANSFER_COST_INR = 305
SLA_BREACH_CREDIT_INR = 350


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------

def prepare_tickets(
    tickets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare classified tickets for analysis.

    Expected columns include:
        ticket_id
        created_at
        resolved_at
        first_response_at
        status
        channel
        agent_id
        assigned_team
        transfers
        csat_score
        refund_amount_inr
        complaint_category

    This function does not modify the input DataFrame.
    """
    df = tickets.copy()

    df = add_week_fields(df, "created_at")
    df = add_completion_flag(df)
    df = add_first_response_minutes(df)
    df = add_handle_time_minutes(df)

    # Normalize common categorical fields.
    for column in [
        "status",
        "channel",
        "agent_id",
        "assigned_team",
        "complaint_category",
    ]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()

    # Numeric fields.
    if "transfers" in df.columns:
        df["transfers"] = pd.to_numeric(
            df["transfers"],
            errors="coerce",
        ).fillna(0)

    if "refund_amount_inr" in df.columns:
        df["refund_amount_inr"] = pd.to_numeric(
            df["refund_amount_inr"],
            errors="coerce",
        ).fillna(0)

    if "csat_score" in df.columns:
        # IMPORTANT:
        # Blank CSAT means "no response", not zero.
        df["csat_score"] = pd.to_numeric(
            df["csat_score"],
            errors="coerce",
        )

        # Legacy rows use 0 for no response.
        df.loc[df["csat_score"] == 0, "csat_score"] = pd.NA

    return df


# ---------------------------------------------------------------------------
# SLA analysis
# ---------------------------------------------------------------------------

def add_sla_fields(
    tickets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add SLA target, breach flag and estimated breach credit.

    Vireo's policy defines first-response targets:
        chat    15 min
        voice   120 min
        social  240 min
        email   480 min

    A missed target generates a Rs 350 store credit.
    """
    df = tickets.copy()

    df["sla_target_minutes"] = (
        df["channel"]
        .str.lower()
        .map(SLA_MINUTES)
    )

    df["sla_breach"] = (
        df["first_response_minutes"].notna()
        & df["sla_target_minutes"].notna()
        & (
            df["first_response_minutes"]
            > df["sla_target_minutes"]
        )
    )

    df["sla_breach_credit_inr"] = (
        df["sla_breach"].astype(int)
        * SLA_BREACH_CREDIT_INR
    )

    return df


# ---------------------------------------------------------------------------
# Weekly complaint digest
# ---------------------------------------------------------------------------

def build_weekly_complaint_digest(
    tickets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce weekly complaint-category counts.

    Output:
        week_start
        complaint_category
        ticket_count
        share_of_week
        previous_week_count
        wow_change_pct
    """
    df = prepare_tickets(tickets)

    if "complaint_category" not in df.columns:
        raise ValueError(
            "tickets must contain 'complaint_category'. "
            "Run classify.py first."
        )

    weekly = (
        df.groupby(
            ["week_start", "complaint_category"],
            dropna=False,
        )
        .size()
        .reset_index(name="ticket_count")
    )

    # Total tickets per week.
    weekly_totals = (
        weekly.groupby("week_start")["ticket_count"]
        .sum()
        .rename("weekly_total")
        .reset_index()
    )

    weekly = weekly.merge(
        weekly_totals,
        on="week_start",
        how="left",
    )

    weekly["share_of_week"] = (
        weekly["ticket_count"]
        / weekly["weekly_total"]
        * 100
    )

    # Previous week's count for the same category.
    weekly = weekly.sort_values(
        ["complaint_category", "week_start"]
    )

    weekly["previous_week_count"] = (
        weekly.groupby("complaint_category")["ticket_count"]
        .shift(1)
    )

    weekly["wow_change_pct"] = weekly.apply(
        lambda row: pct_change(
            row["ticket_count"],
            row["previous_week_count"],
        ),
        axis=1,
    )

    return weekly.sort_values(
        ["week_start", "ticket_count"],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_top_complaints(
    tickets: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Return the top complaint categories for each week.
    """
    digest = build_weekly_complaint_digest(tickets)

    return (
        digest.sort_values(
            ["week_start", "ticket_count"],
            ascending=[True, False],
        )
        .groupby("week_start", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Representative complaints
# ---------------------------------------------------------------------------

def get_representative_complaints(
    tickets: pd.DataFrame,
    week_start,
    category: str,
    n: int = 3,
) -> pd.DataFrame:
    """
    Return representative customer messages for a weekly category.

    This is deliberately simple: it selects the longest available messages
    because those usually contain more context for a human digest reviewer.
    """
    df = prepare_tickets(tickets)

    subset = df[
        (df["week_start"] == week_start)
        & (df["complaint_category"] == category)
    ].copy()

    if subset.empty:
        return subset

    subset["message_length"] = (
        subset["customer_message"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    subset = subset.sort_values(
        "message_length",
        ascending=False,
    ).head(n)

    subset["representative_message"] = (
        subset["customer_message"]
        .fillna("")
        .map(lambda x: clean_text(x)[:500])
    )

    return subset[
        [
            "ticket_id",
            "representative_message",
            "channel",
        ]
    ]


# ---------------------------------------------------------------------------
# Agent leaderboard
# ---------------------------------------------------------------------------

def build_agent_leaderboard(
    tickets: pd.DataFrame,
    agents: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build weekly tickets-closed leaderboard.

    IMPORTANT:
    Tier 2 agents are excluded because Vireo's policy explicitly says
    Tier 2 agents must not be compared with Tier 1 on volume metrics.

    The metric is deliberately named tickets_closed rather than
    agent_performance.
    """
    df = prepare_tickets(tickets)

    roster = agents.copy()

    for column in ["agent_id", "name", "team", "shift", "tier"]:
        if column in roster.columns:
            roster[column] = (
                roster[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    # An agent can have multiple roster rows over time.
    # Keep unique identity/team/tier combinations for the MVP.
    roster = roster.drop_duplicates(
        subset=["agent_id", "name", "team", "shift", "tier"]
    )

    # Only completed tickets.
    completed = df[df["is_completed"]].copy()

    # Only tickets with a resolving agent.
    completed = completed[
        completed["agent_id"].ne("")
    ]

    leaderboard = (
        completed.groupby(
            ["week_start", "agent_id"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "tickets_closed"})
    )

    leaderboard = leaderboard.merge(
        roster[
            [
                "agent_id",
                "name",
                "team",
                "shift",
                "tier",
            ]
        ],
        on="agent_id",
        how="left",
    )

    # Exclude Tier 2 from the volume leaderboard.
    leaderboard = leaderboard[
        leaderboard["tier"].str.upper().ne("TIER 2")
    ]

    leaderboard["rank"] = (
        leaderboard
        .groupby("week_start")["tickets_closed"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return leaderboard.sort_values(
        ["week_start", "rank", "name"],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Weekly operational metrics
# ---------------------------------------------------------------------------

def build_weekly_operational_metrics(
    tickets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce additional weekly metrics useful for the business case.

    Includes:
        tickets
        completed tickets
        SLA breaches
        breach credits
        transfers
        transfer cost
        refunds
        CSAT
        average first response
        average handle time
    """
    df = prepare_tickets(tickets)
    df = add_sla_fields(df)

    rows = []

    for week_start, group in df.groupby("week_start"):
        responded = group[
            group["first_response_minutes"].notna()
        ]

        csat = group["csat_score"].dropna()

        transfers = (
            group["transfers"].sum()
            if "transfers" in group.columns
            else 0
        )

        refunds = (
            group["refund_amount_inr"].sum()
            if "refund_amount_inr" in group.columns
            else 0
        )

        rows.append(
            {
                "week_start": week_start,
                "tickets": len(group),
                "completed_tickets": int(
                    group["is_completed"].sum()
                ),
                "sla_breaches": int(
                    group["sla_breach"].sum()
                ),
                "sla_breach_credits_inr": int(
                    group["sla_breach_credit_inr"].sum()
                ),
                "transfers": int(transfers),
                "transfer_cost_inr": int(
                    transfers * TRANSFER_COST_INR
                ),
                "refund_amount_inr": float(refunds),
                "avg_first_response_minutes": (
                    responded["first_response_minutes"].mean()
                    if not responded.empty
                    else None
                ),
                "avg_handle_time_minutes": (
                    group["handle_time_minutes"].dropna().mean()
                    if group["handle_time_minutes"].notna().any()
                    else None
                ),
                "avg_csat": (
                    csat.mean()
                    if not csat.empty
                    else None
                ),
                "csat_responses": int(len(csat)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "week_start"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Complaint summary for a single week
# ---------------------------------------------------------------------------

def summarize_week(
    tickets: pd.DataFrame,
    week_start,
    top_n: int = 5,
) -> dict:
    """
    Generate a compact dictionary suitable for a weekly digest.
    """
    df = prepare_tickets(tickets)
    df = add_sla_fields(df)

    week = df[df["week_start"] == week_start].copy()

    if week.empty:
        return {
            "week_start": str(week_start),
            "ticket_count": 0,
            "top_complaints": [],
            "sla_breaches": 0,
            "avg_csat": None,
        }

    complaints = (
        week["complaint_category"]
        .value_counts()
        .head(top_n)
    )

    top_complaints = [
        {
            "category": category,
            "count": int(count),
            "share_pct": round(
                count / len(week) * 100,
                1,
            ),
        }
        for category, count in complaints.items()
    ]

    csat = week["csat_score"].dropna()

    return {
        "week_start": str(week_start),
        "ticket_count": int(len(week)),
        "completed_tickets": int(
            week["is_completed"].sum()
        ),
        "top_complaints": top_complaints,
        "sla_breaches": int(
            week["sla_breach"].sum()
        ),
        "sla_breach_cost_inr": int(
            week["sla_breach_credit_inr"].sum()
        ),
        "avg_csat": (
            round(float(csat.mean()), 2)
            if not csat.empty
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def run_analysis(
    tickets: pd.DataFrame,
    agents: pd.DataFrame,
) -> dict:
    """
    Run the complete deterministic analysis.

    Returns all major outputs as DataFrames.
    """
    prepared = prepare_tickets(tickets)

    digest = build_weekly_complaint_digest(prepared)

    leaderboard = build_agent_leaderboard(
        prepared,
        agents,
    )

    operational = build_weekly_operational_metrics(
        prepared,
    )

    return {
        "tickets": prepared,
        "weekly_digest": digest,
        "agent_leaderboard": leaderboard,
        "operational_metrics": operational,
    }