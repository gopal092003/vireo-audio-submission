# main.py

import argparse
import os
from pathlib import Path

from src.utils import (
    ensure_directories,
    load_csv,
    save_csv,
    validate_agents,
    validate_tickets,
)
from src.classify import classify_tickets
from src.analyze import run_analysis
from src.evaluate import run_evaluation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "openai/gpt-oss-20b"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    model: str = DEFAULT_MODEL,
    evaluation_sample_size: int = 50,
    skip_classification: bool = False,
    limit: int | None = None,
):
    """
    Run the complete Vireo Audio support-analysis pipeline.

    Pipeline:

        raw CSVs
            ↓
        validation
            ↓
        AI complaint classification
            ↓
        weekly complaint analysis
            ↓
        agent leaderboard
            ↓
        evaluation sample
            ↓
        output files
    """

    print("=" * 70)
    print("VIREO AUDIO — SUPPORT ANALYSIS")
    print("=" * 70)

    ensure_directories()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------

    print("\n[1/5] Loading data...")

    tickets = load_csv("tickets.csv")
    agents = load_csv("agents.csv")

    print(f"  Tickets: {len(tickets):,}")
    print(f"  Agent roster rows: {len(agents):,}")

    validate_tickets(tickets)
    validate_agents(agents)
    if limit is not None:
        tickets = tickets.head(limit).copy()
        print(f"  TEST LIMIT: processing {len(tickets):,} tickets")

    # ------------------------------------------------------------------
    # 2. AI classification
    # ------------------------------------------------------------------

    print("\n[2/5] Classifying complaints...")

    processed_path = (
        Path(__file__).resolve().parent
        / "data"
        / "processed"
        / "tickets_enriched.csv"
    )

    if skip_classification:
        if not processed_path.exists():
            raise FileNotFoundError(
                "skip_classification was requested, but "
                f"{processed_path} does not exist."
            )

        import pandas as pd

        classified = pd.read_csv(processed_path)

        print(
            f"  Loaded existing classifications: "
            f"{len(classified):,} tickets"
        )

    else:
        classified = classify_tickets(
            tickets,
            model=model,
        )

        processed_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        classified.to_csv(
            processed_path,
            index=False,
        )

        print(
            f"  Classified: {len(classified):,} tickets"
        )

    # ------------------------------------------------------------------
    # 3. Analysis
    # ------------------------------------------------------------------

    print("\n[3/5] Building weekly analysis...")

    results = run_analysis(
        tickets=classified,
        agents=agents,
    )

    weekly_digest_path = save_csv(
        results["weekly_digest"],
        "weekly_digest.csv",
    )

    leaderboard_path = save_csv(
        results["agent_leaderboard"],
        "agent_leaderboard.csv",
    )

    operational_path = save_csv(
        results["operational_metrics"],
        "weekly_operational_metrics.csv",
    )

    print(
        f"  Weekly digest: {weekly_digest_path}"
    )

    print(
        f"  Agent leaderboard: {leaderboard_path}"
    )

    print(
        f"  Operational metrics: {operational_path}"
    )

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------

    print("\n[4/5] Preparing evaluation...")

    evaluation = run_evaluation(
        tickets=classified,
        sample_size=evaluation_sample_size,
    )

    report = evaluation["report"]

    agreement = report[
        "reference_category_agreement"
    ]

    if agreement.get("agreement_pct") is not None:
        print(
            "  AI/reference-category agreement: "
            f"{agreement['agreement_pct']:.2f}% "
            f"(n={agreement['n']})"
        )
    else:
        print(
            "  Reference-category agreement unavailable."
        )

    print(
        "  Manual review file: "
        "output/evaluation_review.csv"
    )

    print(
        "  Evaluation report: "
        "output/evaluation.json"
    )

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------

    print("\n[5/5] Pipeline complete.")

    print("\nOutputs:")
    print("  data/processed/tickets_enriched.csv")
    print("  output/weekly_digest.csv")
    print("  output/agent_leaderboard.csv")
    print("  output/weekly_operational_metrics.csv")
    print("  output/evaluation_review.csv")
    print("  output/evaluation.json")

    print("\nNext step:")
    print(
        "  Open output/evaluation_review.csv and manually "
        "review the sampled tickets."
    )

    print("=" * 70)

    return {
        "classified": classified,
        "analysis": results,
        "evaluation": evaluation,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Vireo Audio support ticket classification "
            "and weekly analysis."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Groq model used for complaint classification "
            f"(default: {DEFAULT_MODEL})"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Limit the number of tickets processed "
            "for testing (default: all tickets)"
        ),
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help=(
            "Number of tickets to include in manual evaluation "
            "(default: 50)"
        ),
    )

    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help=(
            "Reuse data/processed/tickets_enriched.csv "
            "instead of making new AI calls."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if not args.skip_classification:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set.\n\n"
                "Set it before running:\n"
                "  Windows PowerShell:\n"
                "    $env:GROQ_API_KEY='your-key'\n\n"
                "  macOS/Linux:\n"
                "    export GROQ_API_KEY='your-key'"
            )

    run_pipeline(
        model=args.model,
        evaluation_sample_size=args.sample_size,
        skip_classification=args.skip_classification,
        limit=args.limit,
    )