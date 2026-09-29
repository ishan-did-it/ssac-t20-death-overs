# Match-Level Validation and the Limits of Bowler Ratings in T20 Death-Over Win Probability

This repository contains the complete, reproducible analytical pipeline for our MIT Sloan Sports Analytics Conference (SSAC) abstract submission. It demonstrates how standard ball-level train/test splits artificially inflate T20 cricket win-probability models due to match-identity leakage, and establishes a rigorous null result for the predictive value of historical bowler economy in death-over scenarios.

## Repository Structure

*   `data/`: Directory for storing raw Cricsheet data and engineered datasets (ignored in version control).
*   `src/parse_cricsheet.py`: Parses raw JSON ball-by-ball Cricsheet data into a tabular format.
*   `src/engineer_features.py`: Generates the contextual scoreboard features (CRR, RRR, wickets lost) and prepares the temporal splits.
*   `src/evaluate_ssac.py`: Executes the final evaluation pipeline, including the leakage audit (GroupKFold vs. KFold), out-of-fold Bayesian shrinkage rating calculations, and the paired match-clustered bootstrap.
*   `requirements.txt`: Python package dependencies.

## Reproducing the Results

To reproduce Table 1 (Leakage Gap by Model Capacity) and Table 2 (Model Comparison) exactly as presented in the abstract, execute the pipeline in the following order:

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download the Raw Data:**
    Download the **Men's T20s** JSON dataset bundle from Cricsheet (`https://cricsheet.org/downloads/t20s_json.zip`). Extract the `.json` files into a new directory named `data/raw/` at the root of this repository.

3.  **Parse the raw event data:**
    ```bash
    python src/parse_cricsheet.py
    ```

4.  **Engineer features and baseline dataset:**
    ```bash
    python src/engineer_features.py
    ```

5.  **Run the strict evaluation audit:**
    ```bash
    python src/evaluate_ssac.py
    ```

> **Note on Reproducibility:** Cricsheet is a living dataset updated periodically as new matches are played. Re-running this pipeline on a freshly downloaded dataset may yield slightly different match counts and point estimates than those reported in the abstract; however, the qualitative findings (monotonic leakage scaling with model capacity, null personnel effect) remain stable across data snapshots.
