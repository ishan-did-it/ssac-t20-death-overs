import pandas as pd
import json
import glob
import os
master_data = []
file_list = glob.glob("data/raw/*.json")
for file_path in file_list: 
    with open(file_path, 'r') as f:
        match_data = json.load(f)
    target_score = -1   
    for inning in match_data['innings']:
        current_score = 0
        wickets_lost = 0
        for over in inning['overs']:
            for delivery in over['deliveries']:
                current_score += delivery['runs']['total']
                if 'wickets' in delivery:
                    wickets_lost += len(delivery['wickets'])
             
                row = {

                    'target_score': target_score,
                    'match_id': os.path.basename(file_path).replace('.json', ''),
                    'batting_team': inning['team'],
                    'over': over['over'],
                    'ball': delivery['actual_delivery'],
                    'batter': delivery['batter'],
                    'bowler': delivery['bowler'],
                    'runs_this_ball': delivery['runs']['total'],
                    'current_score': current_score,
                    'wickets_lost': wickets_lost
                }
                master_data.append(row)
        target_score = current_score + 1
df = pd.DataFrame(master_data)
print(f"Total rows extracted: {len(df)}")
df.to_csv('data/t20_dataset_master.csv', index=False)
print("Saved to data/t20_dataset_master.csv!")