import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import xgboost as xgb

print("Loading the contextual alpha dataset...")
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

X = df_chase[features]
y = df_chase['is_win']


print("Applying chronological time-series split to prevent look-ahead bias...")
unique_matches = df_chase['match_id'].unique()


split_idx = int(len(unique_matches) * 0.8)
train_matches = unique_matches[:split_idx]
test_matches = unique_matches[split_idx:]

train_mask = df_chase['match_id'].isin(train_matches)
test_mask = df_chase['match_id'].isin(test_matches)

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]


print("Training the Contextual Engine...")
xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)



death_mask_test = X_test['balls_remaining'] <= 30
X_test_death = X_test[death_mask_test]
y_test_death = y_test[death_mask_test]

preds_death = xgb_model.predict(X_test_death)
acc_death = accuracy_score(y_test_death, preds_death)

print("-" * 50)
print(f"TEMPORAL DEATH OVERS ACCURACY: {acc_death * 100:.2f}%")
print("-" * 50)

xgb_model.save_model('data/rigorous_model.json')
print("Rigorous Contextual Model saved to data/rigorous_model.json.")