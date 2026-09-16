import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import brier_score_loss

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        bin_mask = binids == i
        if np.sum(bin_mask) > 0:
            bin_probs = y_prob[bin_mask]
            bin_true = y_true[bin_mask]
            
            avg_prob = np.mean(bin_probs)
            emp_freq = np.mean(bin_true)
            
            ece += np.abs(avg_prob - emp_freq) * (np.sum(bin_mask) / len(y_prob))
            
    return ece

def run_calibration_test():
    model = xgb.XGBClassifier()
    model.load_model('data/rigorous_model.json')
    
    df = pd.read_csv('data/t20_dataset_engineered.csv')
    df_chase = df[df['target_score'] > 0].copy()
    
    match_outcomes = df_chase.groupby('match_id').apply(
        lambda x: 1 if x['current_score'].max() >= x['target_score'].max() else 0
    ).reset_index(name='is_win')
    
    df_chase = df_chase.merge(match_outcomes, on='match_id')
    
    features = [
        'balls_remaining', 'wickets_lost', 'crr', 'rrr',
        'bowler_death_econ', 'batter_death_sr'
    ]
    
    df_chase = df_chase.sort_values(by=['match_id', 'over', 'ball'])
    
    unique_matches = df_chase['match_id'].unique()
    split_idx = int(len(unique_matches) * 0.8)
    test_matches = unique_matches[split_idx:]
    
    test_df = df_chase[df_chase['match_id'].isin(test_matches)].copy()
    
    death_mask_test = test_df['balls_remaining'] <= 30
    test_death = test_df[death_mask_test].copy()
    
    X_test_death = test_death[features]
    y_test_death = test_death['is_win']
    
    y_prob = model.predict_proba(X_test_death)[:, 1]
    
    brier = brier_score_loss(y_test_death, y_prob)
    ece = expected_calibration_error(y_test_death.values, y_prob)
    
    print("-" * 50)
    print(f"TEMPORAL DEATH OVERS TEST SAMPLE: {len(test_death)} balls")
    print(f"BRIER SCORE: {brier:.4f}")
    print(f"EXPECTED CALIBRATION ERROR (ECE): {ece:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    run_calibration_test()