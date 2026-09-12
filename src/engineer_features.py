import pandas as pd
print("Loading updated dataset...")
df = pd.read_csv('data/t20_dataset_master.csv')
df['balls_bowled'] = df.groupby(['match_id', 'batting_team'])['is_legal_ball'].cumsum()
df['balls_remaining'] = 120 - df['balls_bowled']
df['balls_remaining'] = df['balls_remaining'].clip(lower=0)
print(df[['match_id', 'batting_team', 'over', 'ball', 'is_legal_ball', 'balls_bowled', 'balls_remaining']].head(15))
df.to_csv('data/t20_dataset_engineered.csv', index=False)
print("Saved fully engineered dataset to data/t20_dataset_engineered.csv")