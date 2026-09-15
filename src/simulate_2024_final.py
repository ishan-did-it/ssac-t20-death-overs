import pandas as pd
import xgboost as xgb

print("Initializing Contextual Simulation (Dynamic Pipeline Lookup)...\n")


model = xgb.XGBClassifier()
model.load_model('data/rigorous_model.json')

print("Loading historical ratings from engineered dataset...")
df = pd.read_csv('data/t20_dataset_engineered.csv')


death_mask = df['over'] >= 15
global_death_econ = round((df[death_mask]['runs_total'].sum() * 6) / max(df[death_mask]['is_legal_ball'].sum(), 1), 2)
global_death_sr = round((df[death_mask]['runs_total'].sum() * 100) / max(df[death_mask]['is_legal_ball'].sum(), 1), 2)


bowler_lookup = df.dropna(subset=['bowler', 'bowler_death_econ']).groupby('bowler')['bowler_death_econ'].first().to_dict()
batter_lookup = df.dropna(subset=['batter', 'batter_death_sr']).groupby('batter')['batter_death_sr'].first().to_dict()

def resolve_bowler_econ(bowler_name):
    return bowler_lookup.get(bowler_name, global_death_econ)

def resolve_batter_sr(batter_name):
    return batter_lookup.get(batter_name, global_death_sr)


target_score = 177


match_timeline = [
    (30, 147, 4, 'JJ Bumrah', 'H Klaasen', "Overs 15.0: Bumrah starts death spell"),
    (24, 151, 4, 'HH Pandya', 'H Klaasen', "Overs 16.0: Bumrah restricts to 4 runs"),
    (23, 151, 5, 'HH Pandya', 'DA Miller', "Overs 16.1: Klaasen dismissed by Pandya"),
    (18, 155, 5, 'JJ Bumrah', 'DA Miller', "Overs 17.0: Bumrah returns for 18th over"),
    (14, 156, 6, 'JJ Bumrah', 'M Jansen',  "Overs 17.4: Jansen bowled by Bumrah"),
    (12, 157, 6, 'Arshdeep Singh', 'DA Miller', "Overs 18.0: End of Bumrah's spell (2 runs conceded)"),
    (6, 161, 6, 'HH Pandya', 'DA Miller', "Overs 19.0: Arshdeep concedes only 4 runs"),
    (5, 161, 7, 'HH Pandya', 'K Rabada',  "Overs 19.1: Miller caught on the boundary"),
    (1, 168, 8, 'HH Pandya', 'A Nortje',  "Overs 19.5: Rabada dismissed. Chase sealed.")
]

features = ['balls_remaining', 'wickets_lost', 'crr', 'rrr', 'bowler_death_econ', 'batter_death_sr']

print("-" * 120)
print(f"{'Balls Left':<10} | {'Bowler':<15} | {'Batter':<12} | {'Event':<45} | {'Model Win %':<12}")
print("-" * 120)


for balls_rem, curr_score, wkts, bowler, batter, event in match_timeline:
    balls_bowled = 120 - balls_rem
    crr = round((curr_score * 6) / max(balls_bowled, 1), 2)
    rrr = round(((target_score - curr_score) * 6) / max(balls_rem, 1), 2)
    
    b_econ = resolve_bowler_econ(bowler)
    b_sr = resolve_batter_sr(batter)
    
    input_row = pd.DataFrame([[balls_rem, wkts, crr, rrr, b_econ, b_sr]], columns=features)
    win_prob = model.predict_proba(input_row)[0][1] * 100
    
    print(f"{balls_rem:<10} | {bowler:<15} | {batter:<12} | {event:<45} | {win_prob:>10.2f}%")

print("-" * 120)