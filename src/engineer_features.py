import pandas as pd
import numpy as np

print("Loading raw master dataset...")
df = pd.read_csv('data/t20_dataset_master.csv')


print("Calculating match context features...")
df['balls_bowled'] = df.groupby(['match_id', 'batting_team'])['is_legal_ball'].cumsum()
df['balls_remaining'] = (120 - df['balls_bowled']).clip(lower=0)

df['wickets_lost'] = df.groupby(['match_id', 'batting_team'])['is_wicket'].cumsum()

df['crr'] = round((df['current_score'] * 6) / df['balls_bowled'].replace(0, 1), 2)

df['rrr'] = np.where(
    df['target_score'] > 0,
    round(((df['target_score'] - df['current_score']) * 6) / df['balls_remaining'].replace(0, 1), 2),
    -1
)


print("Computing death overs personnel ratings...")
death_mask = df['over'] >= 15
global_death_econ = (df[death_mask]['runs_total'].sum() * 6) / max(df[death_mask]['is_legal_ball'].sum(), 1)
global_death_sr = (df[death_mask]['runs_total'].sum() * 100) / max(df[death_mask]['is_legal_ball'].sum(), 1)
bowler_death = df[death_mask].groupby('bowler').agg(
    runs=('runs_total', 'sum'),
    balls=('is_legal_ball', 'sum')
).reset_index()


M_bowler = 30
bowler_death['bowler_death_econ'] = (
    (bowler_death['runs'] + (global_death_econ / 6) * M_bowler) * 6
) / (bowler_death['balls'] + M_bowler)

batter_death = df[death_mask].groupby('batter').agg(
    runs=('runs_total', 'sum'),
    balls=('is_legal_ball', 'sum')
).reset_index()

M_batter = 30
batter_death['batter_death_sr'] = (
    (batter_death['runs'] + (global_death_sr / 100) * M_batter) * 100
) / (batter_death['balls'] + M_batter)


print("Merging ratings into master dataset...")
df = df.merge(bowler_death[['bowler', 'bowler_death_econ']], on='bowler', how='left')
df['bowler_death_econ'] = df['bowler_death_econ'].fillna(global_death_econ).round(2)

df = df.merge(batter_death[['batter', 'batter_death_sr']], on='batter', how='left')
df['batter_death_sr'] = df['batter_death_sr'].fillna(global_death_sr).round(2)


output_file = 'data/t20_dataset_engineered.csv'
df.to_csv(output_file, index=False)
print("-" * 50)
print(f"Feature engineering complete. Saved to {output_file}")
print(f"Global Death Economy: {global_death_econ:.2f} rpo | Global Death Strike Rate: {global_death_sr:.2f}")
print("-" * 50)