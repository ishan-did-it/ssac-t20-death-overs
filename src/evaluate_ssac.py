import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
import warnings
warnings.filterwarnings('ignore')

def rep_gap(death, feats, depth, n_est, reps=8):
    b, m = [], []
    for s in range(reps):
        for name, sp, grp in [('ball', ShuffleSplit(1, test_size=0.2, random_state=s), None),
                              ('match', GroupShuffleSplit(1, test_size=0.2, random_state=s), death['match_id'])]:
            tr, te = next(sp.split(death, groups=grp))
            md1 = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, random_state=s, n_jobs=1)
            md1.fit(death.iloc[tr][feats], death.iloc[tr]['is_win'])
            p = md1.predict_proba(death.iloc[te][feats])[:, 1]
            (b if name == 'ball' else m).append(brier_score_loss(death.iloc[te]['is_win'], p))
    g = np.array(m) - np.array(b)
    return np.mean(b), np.mean(m), g.mean(), g.std(ddof=1) / np.sqrt(reps)

def paired_bootstrap_ci(y_true, p_base, p_rich, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b_base = brier_score_loss(y_true[idx], p_base[idx])
        b_rich = brier_score_loss(y_true[idx], p_rich[idx])
        diffs.append(b_rich - b_base)
    diffs = np.array(diffs)
    return np.mean(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

def run_final_stats():
    print("Loading data...")
    df = pd.read_csv('data/t20_dataset_engineered.csv')
    df_chase = df[df['target_score'] > 0].copy()
    
    # Label sanity
    df_chase['post_ball_score'] = df_chase['current_score'] + df_chase['runs_total']
    match_outcomes = df_chase.groupby('match_id').apply(
        lambda x: 1 if x['post_ball_score'].max() >= x['target_score'].max() else 0
    ).reset_index(name='is_win')
    df_chase = df_chase.merge(match_outcomes, on='match_id')
    
    sort_cols = ['start_date', 'match_id', 'over', 'ball'] if 'start_date' in df_chase.columns else ['match_id', 'over', 'ball']
    df_chase = df_chase.sort_values(by=sort_cols).reset_index(drop=True)

    death = df_chase[df_chase['balls_remaining'] <= 30].reset_index(drop=True)
    feats_scoreboard = ['balls_remaining', 'wickets_lost', 'crr', 'rrr']

    print("\n--- 1. LEAKAGE GAP BY MODEL CAPACITY ---")
    for d, n in [(2, 100), (4, 100), (8, 500)]:
        mean_b, mean_m, gap_mean, gap_se = rep_gap(death, feats_scoreboard, d, n)
        print(f"Depth {d}, Est {n} -> Ball: {mean_b:.5f}, Match: {mean_m:.5f}, Gap: {gap_mean:.5f} ± {gap_se:.5f}")

    print("\n--- 2. MODEL COMPARISON (TABLE 2) ---")
    unique_matches = df_chase['match_id'].unique()
    split_idx = int(len(unique_matches) * 0.8)
    train_matches = unique_matches[:split_idx]
    test_matches = unique_matches[split_idx:]
    
    train_df = df_chase[df_chase['match_id'].isin(train_matches)].copy()
    test_df = df_chase[df_chase['match_id'].isin(test_matches)].copy()
    
    train_death = train_df[train_df['balls_remaining'] <= 30].copy()
    test_death = test_df[test_df['balls_remaining'] <= 30].copy()
    
    # Compute OOF Shrunk Bowler Economy on training matches
    global_runs = train_death['runs_total'].sum()
    global_balls = len(train_death)
    global_econ = (global_runs / global_balls) * 6 if global_balls > 0 else 8.0
    kappa = 60
    
    bowler_stats = train_death.groupby('bowler').agg(
        runs=('runs_total', 'sum'),
        balls=('runs_total', 'count')
    ).reset_index()
    
    bowler_stats['shrunk_econ'] = (bowler_stats['runs'] + kappa * (global_econ / 6)) / (bowler_stats['balls'] + kappa) * 6
    econ_dict = dict(zip(bowler_stats['bowler'], bowler_stats['shrunk_econ']))
    global_default_econ = global_econ
    
    train_death['bowler_death_econ'] = train_death['bowler'].map(econ_dict).fillna(global_default_econ)
    test_death['bowler_death_econ'] = test_death['bowler'].map(econ_dict).fillna(global_default_econ)
    
    y_train = train_death['is_win']
    y_test = test_death['is_win'].values
    
    # (a) LR Naive
    feats_naive = ['balls_remaining', 'rrr']
    lr = LogisticRegression()
    lr.fit(train_death[feats_naive], y_train)
    p_a = lr.predict_proba(test_death[feats_naive])[:, 1]
    b_a = brier_score_loss(y_test, p_a)
    print(f"(a) LR Naive             | Brier: {b_a:.4f}")
    
    # (b) XGB Naive (n_jobs=1 for strict determinism)
    xgb_b = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=1)
    xgb_b.fit(train_death[feats_naive], y_train)
    p_b = xgb_b.predict_proba(test_death[feats_naive])[:, 1]
    b_b = brier_score_loss(y_test, p_b)
    print(f"(b) XGB Naive            | Brier: {b_b:.4f}")
    
    # (c) XGB Scoreboard (n_jobs=1 for strict determinism)
    xgb_c = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=1)
    xgb_c.fit(train_death[feats_scoreboard], y_train)
    p_c = xgb_c.predict_proba(test_death[feats_scoreboard])[:, 1]
    b_c = brier_score_loss(y_test, p_c)
    mean_diff_c, ci_lower_c, ci_upper_c = paired_bootstrap_ci(y_test, p_b, p_c)
    print(f"(c) XGB Scoreboard       | Brier: {b_c:.4f} | Δ vs (b): {mean_diff_c:.4f} (95% CI {ci_lower_c:.4f} to {ci_upper_c:.4f})")
    
    # (d) XGB Personnel (n_jobs=1 for strict determinism)
    feats_pers = ['balls_remaining', 'wickets_lost', 'crr', 'rrr', 'bowler_death_econ']
    xgb_d = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=1)
    xgb_d.fit(train_death[feats_pers], y_train)
    p_d = xgb_d.predict_proba(test_death[feats_pers])[:, 1]
    b_d = brier_score_loss(y_test, p_d)
    mean_diff_d, ci_lower_d, ci_upper_d = paired_boosting_ci_placeholder = paired_bootstrap_ci(y_test, p_c, p_d)
    print(f"(d) XGB Personnel        | Brier: {b_d:.4f} | Δ vs (c): {mean_diff_d:.4f} (95% CI {ci_lower_d:.4f} to {ci_upper_d:.4f})")

    print("\n--- 3. DATASET FACTS FOR ABSTRACT ---")
    print(f"Total Matches [N]:         {len(unique_matches)}")
    print(f"Test Matches [N_test]:     {len(test_matches)}")
    print(f"Test Death Rows [N_rows]:  {len(test_death)}")
    
    if 'start_date' in df_chase.columns:
        print(f"Years:                     {pd.to_datetime(df_chase['start_date']).dt.year.min()} to {pd.to_datetime(df_chase['start_date']).dt.year.max()}")
    else:
        print("Years:                     (No start_date column. Use 'the latest 20% of matches by Cricsheet ID')")

if __name__ == "__main__":
    run_final_stats()