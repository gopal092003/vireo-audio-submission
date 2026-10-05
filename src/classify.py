# src/classify.py

import json
import os
import time
from typing import Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Complaint taxonomy
# ---------------------------------------------------------------------------

COMPLAINT_CATEGORIES = [
    "Battery / Charging",
    "Connectivity / Bluetooth",
    "Audio Quality",
    "App / Software",
    "Delivery / Logistics",
    "Warranty / Replacement",
    "Refund / Payment",
    "Product Damage",
    "Setup / Usage",
    "Other",
]


SYSTEM_PROMPT = f"""
You classify Vireo Audio customer support tickets.

Classify the customer's PRIMARY complaint into exactly one category.

Allowed categories:
{json.dumps(COMPLAINT_CATEGORIES, indent=2)}

Rules:
1. Choose exactly one category.
2. Classify the customer's actual problem, not the agent's response.
3. Use the most specific category available.
4. Delivery, tracking, missing delivery and reshipment issues -> Delivery / Logistics.
5. Refunds, payments, duplicate charges, cancellations and price adjustments
   -> Refund / Payment.
6. Warranty claims and replacement requests caused by product faults
   -> Warranty / Replacement.
7. Physical damage to the product itself -> Product Damage.
8. Pairing, Bluetooth, connection drops and device discovery
   -> Connectivity / Bluetooth.
9. Sound, volume, distortion, microphone/audio playback problems
   -> Audio Quality.
10. Mobile app, firmware, software or app-login problems
    -> App / Software.
11. Charging, battery life and battery not holding charge
    -> Battery / Charging.
12. Basic installation, setup or usage questions
    -> Setup / Usage.
13. If none clearly applies -> Other.

Return ONLY valid JSON:

{{
  "category": "<one allowed category>",
  "confidence": <number between 0 and 1>,
  "reason": "<short explanation>"
}}
"""


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def get_groq_client():
    """
    Create a Groq client.

    The API key is read from GROQ_API_KEY.

    Keeping client creation here makes the rest of this module easy to
    replace with another model/provider if needed.
    """
    try:
        from groq import Groq
    except ImportError as exc:
        raise ImportError(
            "Groq package not installed. Run: pip install groq"
        ) from exc

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set."
        )

    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Single-ticket classification
# ---------------------------------------------------------------------------

def build_ticket_text(
    customer_message: str,
    agent_notes: str = "",
) -> str:
    """
    Build the text sent to the classifier.

    Customer message gets priority. Agent notes are included as secondary
    context because they can clarify what issue was actually resolved.
    """
    customer_message = str(customer_message or "").strip()
    agent_notes = str(agent_notes or "").strip()

    return (
        "CUSTOMER MESSAGE:\n"
        f"{customer_message}\n\n"
        "AGENT NOTES:\n"
        f"{agent_notes}"
    )


def classify_ticket(
    customer_message: str,
    agent_notes: str = "",
    model: str = "openai/gpt-oss-20b",
    client=None,
    max_retries: int = 3,
) -> Dict:
    """
    Classify one support ticket.

    Returns:
        {
            "category": ...,
            "confidence": ...,
            "reason": ...
        }
    """
    if client is None:
        client = get_groq_client()

    ticket_text = build_ticket_text(
        customer_message,
        agent_notes,
    )

    if not customer_message.strip():
        return {
            "category": "Other",
            "confidence": 0.0,
            "reason": "No customer message available.",
        }

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": ticket_text,
                    },
                ],
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            return validate_classification(result)

        except Exception as exc:
            if attempt == max_retries - 1:
                return {
                    "category": "Other",
                    "confidence": 0.0,
                    "reason": f"Classification failed: {exc}",
                }

            # Small retry delay.
            time.sleep(1.5 * (attempt + 1))

    return {
        "category": "Other",
        "confidence": 0.0,
        "reason": "Classification failed.",
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_classification(result: Dict) -> Dict:
    """
    Validate and normalize model output.
    """
    category = result.get("category", "Other")

    if category not in COMPLAINT_CATEGORIES:
        category = "Other"

    try:
        confidence = float(result.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    reason = str(result.get("reason", "")).strip()

    return {
        "category": category,
        "confidence": confidence,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------

def classify_tickets(
    tickets: pd.DataFrame,
    model: str = "openai/gpt-oss-20b",
    save_progress_every: int = 50,
) -> pd.DataFrame:
    """
    Classify all tickets in a DataFrame.

    Adds:
        complaint_category
        classification_confidence
        classification_reason

    The function processes tickets sequentially to keep the MVP simple and
    make API usage/cost easy to understand.
    """
    required = {
        "ticket_id",
        "customer_message",
    }

    missing = required - set(tickets.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    client = get_groq_client()

    results: List[Dict] = []

    total = len(tickets)

    for index, row in tickets.iterrows():
        result = classify_ticket(
            customer_message=row.get("customer_message", ""),
            agent_notes=row.get("agent_notes", ""),
            model=model,
            client=client,
        )

        results.append(
            {
                "ticket_id": row["ticket_id"],
                "complaint_category": result["category"],
                "classification_confidence": result["confidence"],
                "classification_reason": result["reason"],
            }
        )

        processed = len(results)

        if processed % save_progress_every == 0:
            print(
                f"Classified {processed}/{total} tickets..."
            )

    classifications = pd.DataFrame(results)

    return tickets.merge(
        classifications,
        on="ticket_id",
        how="left",
    )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def classify_file(
    input_df: pd.DataFrame,
    model: str = "openai/gpt-oss-20b",
) -> pd.DataFrame:
    """
    Convenience wrapper used by main.py.
    """
    return classify_tickets(
        input_df,
        model=model,
    )