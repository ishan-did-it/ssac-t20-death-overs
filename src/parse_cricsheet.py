import json
import os
import pandas as pd

def parse_cricsheet_json(raw_dir='data/raw', output_csv='data/t20_dataset_master.csv'):
    print("Initializing Phase 1: Contextual Alpha Extraction...")
    
    all_deliveries = []
    
    if not os.path.exists(raw_dir):
        print(f"Error: '{raw_dir}' directory not found. Please ensure raw JSONs are placed there.")
        return

    json_files = [f for f in os.listdir(raw_dir) if f.endswith('.json')]
    print(f"Scanning {len(json_files)} raw match files...")

    for filename in json_files:
        filepath = os.path.join(raw_dir, filename)
        match_id = filename.split('.')[0]
        
        with open(filepath, 'r') as file:
            try:
                match_data = json.load(file)
            except json.JSONDecodeError:
                continue
        innings = match_data.get('innings', [])
        target_score = -1
        
        if len(innings) > 0:
            first_inning_overs = innings[0].get('overs', [])
            first_inning_total = sum(
                deliv.get('runs', {}).get('total', 0) 
                for over in first_inning_overs 
                for deliv in over.get('deliveries', [])
            )
            target_score = first_inning_total + 1

        for inning_idx, inning in enumerate(innings):
            batting_team = inning.get('team')
            overs = inning.get('overs', [])
            
            current_score = 0
            
            for over_data in overs:
                over_num = over_data.get('over', 0)
                deliveries = over_data.get('deliveries', [])
                
                for ball_num, delivery in enumerate(deliveries):
                    batter = delivery.get('batter') or delivery.get('striker')
                    bowler = delivery.get('bowler')
                    non_striker = delivery.get('non_striker')
                    
                    runs_total = delivery.get('runs', {}).get('total', 0)
                    current_score += runs_total
                    
                    wickets = delivery.get('wickets', [])
                    is_wicket = 1 if len(wickets) > 0 else 0
                    
                    extras = delivery.get('extras', {})
                    is_legal_ball = 0 if ('wides' in extras or 'noballs' in extras) else 1
                    all_deliveries.append({
                        'match_id': match_id,
                        'inning': inning_idx + 1,
                        'batting_team': batting_team,
                        'over': over_num,
                        'ball': ball_num + 1,
                        'batter': batter,
                        'bowler': bowler,
                        'non_striker': non_striker,
                        'is_legal_ball': is_legal_ball,
                        'is_wicket': is_wicket,
                        'runs_total': runs_total,
                        'current_score': current_score,
                        'target_score': target_score if inning_idx == 1 else -1
                    })

    df = pd.DataFrame(all_deliveries)
    df.to_csv(output_csv, index=False)
    print("-" * 50)
    print(f"Extraction complete. Saved {len(df)} contextual deliveries to {output_csv}")
    print("-" * 50)

if __name__ == "__main__":
    parse_cricsheet_json()