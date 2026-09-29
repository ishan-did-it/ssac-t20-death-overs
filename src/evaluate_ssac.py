import pandas as pd
import numpy as np
import xgboost as xgb
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
            md1 = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, random_state=s, n_jobs=-1)
            md1.fit(death.iloc[tr][feats], death.iloc[tr]['is_win'])
            p = md1.predict_proba(death.iloc[te][feats])[:, 1]
            (b if name == 'ball' else m).append(brier_score_loss(death.iloc[te]['is_win'], p))
    g = np.array(m) - np.array(b)
    return np.mean(b), np.mean(m), g.mean(), g.std(ddof=1) / np.sqrt(reps)

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

    print("\n--- 2. DATASET FACTS FOR ABSTRACT ---")
    unique_matches = df_chase['match_id'].unique()
    split_idx = int(len(unique_matches) * 0.8)
    train_matches = unique_matches[:split_idx]
    test_matches = unique_matches[split_idx:]
    
    test_df = df_chase[df_chase['match_id'].isin(test_matches)]
    test_death = test_df[test_df['balls_remaining'] <= 30]

    print(f"Total Matches [N]:         {len(unique_matches)}")
    print(f"Test Matches [N_test]:     {len(test_matches)}")
    print(f"Test Death Rows [N_rows]:  {len(test_death)}")
    
    if 'start_date' in df_chase.columns:
        print(f"Years:                     {pd.to_datetime(df_chase['start_date']).dt.year.min()} to {pd.to_datetime(df_chase['start_date']).dt.year.max()}")
    else:
        print("Years:                     (No start_date column. Use 'the latest 20% of matches by Cricsheet ID')")

if __name__ == "__main__":
    run_final_stats()