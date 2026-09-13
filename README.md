# Project 5 — Industrial Predictive Maintenance & Failure Prevention

Graduation project (TechTrek Advanced Data Science & AI track). Dataset: AI4I 2020
Predictive Maintenance (UCI). This repo currently contains **Member 1's deliverable**
(Data Engineering, Preprocessing & EDA). ML / DL / Deployment members will add their
folders on top of this same structure.

## Repository structure
```
project-5-predictive-maintenance/
├── data/
│   ├── ai4i2020_raw.csv            # as downloaded, unmodified
│   ├── ai4i2020_cleaned.csv        # audited (Raw → Cleaned stage)
│   └── ai4i2020_feature_ready.csv  # encoded + engineered, leakage-safe (hand-off to ML/DL)
├── notebooks/
│   └── 01_eda_preprocessing.ipynb  # full, already-executed EDA & preprocessing notebook
├── figures/                        # 7 exported EDA/preprocessing figures
├── docs/
│   ├── data_dictionary.md
│   └── pipeline_documentation.md
├── requirements.txt
└── README.md   (this file)
```

## How to run it locally (any team member)

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # macOS/Linux:
   source .venv/bin/activate
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Register the kernel and launch Jupyter**
   ```bash
   python -m ipykernel install --user --name project5-venv
   jupyter notebook notebooks/01_eda_preprocessing.ipynb
   ```
   The notebook already has all outputs saved, so you can just read it. To re-run it from
   scratch: `Kernel → Restart & Run All`. It regenerates `data/ai4i2020_cleaned.csv`,
   `data/ai4i2020_feature_ready.csv`, and all 7 PNGs in `figures/`.

5. **Everything else (ML / DL / Streamlit / Docker)** builds on top of
   `data/ai4i2020_feature_ready.csv` — that is the single hand-off file. Do not re-introduce
   `twf, hdf, pwf, osf, rnf` as model inputs (see `docs/pipeline_documentation.md`, Section 8).

## Team workflow (Git/GitHub)
See the step-by-step guide in the chat for creating the GitHub repo, pushing this folder,
and how each teammate should branch/PR their own part.

