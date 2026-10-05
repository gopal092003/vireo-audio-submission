# Vireo Audio — Support Ticket Analyzer

This project analyzes customer support tickets for Vireo Audio, classifies the primary complaint using an LLM, aggregates weekly trends, ranks agents by tickets closed, and produces a lightweight evaluation of model performance.

## 1. What this does

- Classifies each ticket into a standard complaint category
- Groups ticket trends by week for operational reporting
- Builds an agent leaderboard based on resolved ticket volume
- Creates an evaluation sample to review classification quality
- Exports generated reports into the `output/` directory

## 2. Project structure

```text
vireo-audio-support/
├── data/
│   ├── raw/
│   └── processed/
├── output/
├── src/
│   ├── analyze.py
│   ├── classify.py
│   ├── evaluate.py
│   └── utils.py
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
└── .env.example
```

## 3. Requirements

- Python 3.10+
- Groq API key
- `pip` access for installing dependencies

## 4. Setup

```bash
git clone <repo-url>
cd vireo-audio-support
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or .venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

Set your Groq API key before running the pipeline:

```bash
# macOS/Linux
export GROQ_API_KEY="your-key"

# Windows PowerShell
$env:GROQ_API_KEY="your-key"
```

## 5. Run the pipeline

```bash
python main.py
```

Optional arguments:

```bash
python main.py --model openai/gpt-oss-20b
python main.py --sample-size 50
python main.py --skip-classification
```

## 6. Re-run without new AI calls

If the processed file already exists, you can skip classification and reuse prior results:

```bash
python main.py --skip-classification
```

## 7. Outputs

The pipeline writes these outputs into `output/`:

- `tickets_enriched.csv`
- `weekly_digest.csv`
- `agent_leaderboard.csv`
- `weekly_operational_metrics.csv`
- `evaluation_review.csv`
- `evaluation.json`

## 8. How it works

```text
raw ticket data
    ↓
validation
    ↓
Groq-powered complaint classification
    ↓
weekly aggregation and analysis
    ↓
agent leaderboard
    ↓
manual evaluation sample
    ↓
output reports
```

## 9. Important decisions

- Complaint taxonomy is intentionally normalized to a fixed set of categories
- Tier-2 agents are excluded from the volume leaderboard
- Existing labels are treated as a reference, not guaranteed perfect ground truth
- Manual review is used for a reliable evaluation sample

## 10. Known limitations

- The classification model is sensitive to ambiguous or incomplete customer messages
- Weekly trends depend on the quality and completeness of the source ticket data
- Model costs scale with ticket volume and selected Groq model

## 11. AI / cost notes

- Model: `openai/gpt-oss-20b` by default
- Provider: Groq
- API key source: `GROQ_API_KEY`
- Cost depends on ticket volume and model choice; this project keeps the workflow simple and sequential to remain easy to audit and maintain

## 12. Troubleshooting

If you see an error about the API key, make sure the environment variable is set in the same terminal session before running the script.

```bash
python main.py
```

If the package installation fails, upgrade pip and reinstall dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```