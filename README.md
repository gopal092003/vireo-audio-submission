# Vireo Audio — Support Ticket Analyzer

This is an assessment-based project that analyzes Vireo Audio support tickets, classifies the primary complaint using a Groq-hosted model, summarizes weekly trends, ranks agents by tickets closed, and exports generated reports into the project `output/` folder.

## 1. What this does

- Classifies each ticket into a standard complaint category
- Groups ticket trends by week for operational reporting
- Builds an agent leaderboard based on resolved ticket volume
- Creates a human-review evaluation sample for model quality checks
- Saves all generated outputs in the repository `output/` directory

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

For Windows PowerShell, the standard usage is:

```powershell
set GROQ_API_KEY=API_KEY
python main.py --model openai/gpt-oss-20b
```

For a quick smoke test on a small subset of tickets:

```powershell
set GROQ_API_KEY=API_KEY
python main.py --model openai/gpt-oss-20b --limit 10
```

Optional arguments:

```bash
python main.py --model openai/gpt-oss-20b
python main.py --limit 10
python main.py --sample-size 50
python main.py --skip-classification
```

## 6. Re-run without new AI calls

If the processed file already exists, you can skip classification and reuse prior results:

```bash
python main.py --skip-classification
```

## 7. Outputs

The pipeline writes these files into the project `output/` folder:

- `data/processed/tickets_enriched.csv`
- `output/weekly_digest.csv`
- `output/agent_leaderboard.csv`
- `output/weekly_operational_metrics.csv`
- `output/evaluation_review.csv`
- `output/evaluation.json`

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
- For assessment work, generated results are intentionally kept in `output/` so the project can be reviewed directly after each run

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

```powershell
set GROQ_API_KEY=API_KEY
python main.py --model openai/gpt-oss-20b --limit 10
```

If the package installation fails, upgrade pip and reinstall dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 13. Example run

```text
C:\Users\gopal\Submission>set GROQ_API_KEY=API_KEY
C:\Users\gopal\Submission>python main.py --model openai/gpt-oss-20b --limit 10
======================================================================
VIREO AUDIO — SUPPORT ANALYSIS
======================================================================

[1/5] Loading data...
  Tickets: 12,528
  Agent roster rows: 44
  TEST LIMIT: processing 10 tickets

[2/5] Classifying complaints...
  Classified: 14 tickets

[3/5] Building weekly analysis...
  Weekly digest: C:\Users\gopal\Submission\output\weekly_digest.csv
  Agent leaderboard: C:\Users\gopal\Submission\output\agent_leaderboard.csv
  Operational metrics: C:\Users\gopal\Submission\output\weekly_operational_metrics.csv

[4/5] Preparing evaluation...
Manual evaluation file written to: C:\Users\gopal\Submission\output\evaluation_review.csv
  AI/reference-category agreement: 7.14% (n=14)
  Manual review file: output/evaluation_review.csv
  Evaluation report: output/evaluation.json

[5/5] Pipeline complete.

Outputs:
  data/processed/tickets_enriched.csv
  output/weekly_digest.csv
  output/agent_leaderboard.csv
  output/weekly_operational_metrics.csv
  output/evaluation_review.csv
  output/evaluation.json

Next step:
  Open output/evaluation_review.csv and manually review the sampled tickets.
```
