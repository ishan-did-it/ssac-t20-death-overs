import pandas as pd
import numpy as np
import xgboost as xgb

def get_empirical_distributions(df):
    death_mask = df['over'] >= 15
    death_df = df[death_mask]
    
    total_balls = len(death_df)
    
    wkt_prob = death_df['is_wicket'].sum() / total_balls
    
    runs_df = death_df[death_df['is_wicket'] == 0]
    run_counts = runs_df['runs_total'].value_counts(normalize=True).to_dict()
    
    return wkt_prob, run_counts

def run_empirical_monte_carlo(model_path, data_path, start_balls, start_score, start_wkts, bowler, batter, target, n_sims=5000):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    df = pd.read_csv(data_path)
    
    wkt_prob, run_dist = get_empirical_distributions(df)
    run_values = list(run_dist.keys())
    run_probs = list(run_dist.values())
    
    death_mask = df['over'] >= 15
    global_econ = round((df[death_mask]['runs_total'].sum() * 6) / max(df[death_mask]['is_legal_ball'].sum(), 1), 2)
    global_sr = round((df[death_mask]['runs_total'].sum() * 100) / max(df[death_mask]['is_legal_ball'].sum(), 1), 2)
    
    bowler_lookup = df.dropna(subset=['bowler', 'bowler_death_econ']).groupby('bowler')['bowler_death_econ'].first().to_dict()
    batter_lookup = df.dropna(subset=['batter', 'batter_death_sr']).groupby('batter')['batter_death_sr'].first().to_dict()
    
    b_econ = bowler_lookup.get(bowler, global_econ)
    b_sr = batter_lookup.get(batter, global_sr)
    
    features = ['balls_remaining', 'wickets_lost', 'crr', 'rrr', 'bowler_death_econ', 'batter_death_sr']
    
    successful_chases = 0
    
    for _ in range(n_sims):
        balls = start_balls
        score = start_score
        wkts = start_wkts
        
        while balls > 0 and wkts < 10 and score < target:
            balls_bowled = 120 - balls
            crr = round((score * 6) / max(balls_bowled, 1), 2)
            rrr = round(((target - score) * 6) / max(balls, 1), 2)
            
            input_row = pd.DataFrame([[balls, wkts, crr, rrr, b_econ, b_sr]], columns=features)
            win_prob = model.predict_proba(input_row)[0][1]
            
            adjusted_wkt_prob = wkt_prob * (b_econ / global_econ)
            
            if np.random.rand() < adjusted_wkt_prob:
                wkts += 1
            else:
                runs = np.random.choice(run_values, p=run_probs)
                score += runs
            
            balls -= 1
            
        if score >= target:
            successful_chases += 1
            
    return (successful_chases / n_sims) * 100

if __name__ == "__main__":
    win_percentage = run_empirical_monte_carlo(
        'data/rigorous_model.json', 
        'data/t20_dataset_engineered.csv', 
        14, 156, 6, 'JJ Bumrah', 'M Jansen', 177, 5000
    )
    print(f"Empirical South Africa Win Probability (5,000 Sims): {win_percentage:.2f}%")