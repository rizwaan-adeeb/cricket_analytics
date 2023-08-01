import json
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np 

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
plt.rcParams['text.usetex'] = True

# Get file names
directory = Path("/Users/Rizwaan/Documents/cricket_analytics/cricket_analytics/data/ipl_json")
file_names = [f for f in directory.iterdir() if f.name != "README.txt"]

balls_faced = {i : 0 for i in range(0,11)}
total_balls = 0

# Loop over files
for file_num, file_path in enumerate(tqdm(file_names, desc="Processing files")):
    with open(file_path, 'r') as f:
        data = json.load(f)

        teams_list = data['info']['teams']
        player_list_1 = data['info']['players'][teams_list[0]]
        player_list_2 = data['info']['players'][teams_list[1]]

        if (len(player_list_1)!=11 and len(player_list_2)!=11):
            continue

        if data['innings'][0]['team']==teams_list[0]:
            innings_list_1st = player_list_1
            innings_list_2nd = player_list_2
        else:
            innings_list_1st = player_list_2
            innings_list_2nd = player_list_1

        for over in data['innings'][0]['overs']:
            for ball in over['deliveries']:
                pos = innings_list_1st.index(ball['batter'])
                balls_faced[pos] += 1

                total_balls += 1

        if len(data['innings'])<2:
            continue

        for over in data['innings'][1]['overs']:
            for ball in over['deliveries']:
                pos = innings_list_2nd.index(ball['batter'])
                balls_faced[pos] += 1

                total_balls += 1

for i in balls_faced:
    balls_faced[i] = balls_faced[i]/total_balls * 120

print(balls_faced)
print("Sum = ",np.sum(list(balls_faced.values())))

x_vals = np.array(list(balls_faced.keys())) + 1

fig, ax = plt.subplots(figsize=(14,7))
plt.plot(x_vals, list(balls_faced.values()), ls='', marker='x', color='darkblue')
plt.xticks(x_vals,np.round(x_vals,0))
plt.xlabel("Batting position")
plt.ylabel("Average balls faced")
plt.yticks(np.arange(0, 24, step=2))
plt.tight_layout()
plt.show()

