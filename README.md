# European Bank — Churn Analysis

This workspace contains an exploratory analysis and a basic Streamlit dashboard for analyzing customer churn in a European retail bank dataset.

Contents
- `pro_1.py` — analysis script; performs validation, segmentation, summary tables, visualizations, and exports CSV summaries to `outputs/`.
- `streamlit_app.py` — simple Streamlit dashboard for interactive exploration.
- `European_Bank .csv` — dataset (ensure file name matches or is discovered automatically).
- `outputs/` — generated summary CSVs (created after running `pro_1.py`).

Setup
1. Create a virtual environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Run analysis script

```bash
python "d:\Python\New folder\pro_1.py"
```

Run Streamlit dashboard

```bash
streamlit run "d:\Python\New folder\streamlit_app.py"
```

Deliverables
- CSV summaries in `outputs/` after `pro_1.py` finishes.
- Interactive dashboard via Streamlit for segment filtering and KPI inspection.

Notes
- The scripts attempt to auto-detect the dataset filename if it contains spaces or slight variations.
- If you want a different dashboard layout or additional exports (e.g., PDF report), tell me and I can add them.
