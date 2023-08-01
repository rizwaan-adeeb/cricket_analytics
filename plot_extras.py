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

# Dicts for wides and no balls, one entry per over
wides = {i: 0 for i in range(20)}
noballs = {i: 0 for i in range(20)}
over_count = {i: 0 for i in range(20)}

# Loop over files
for file_num, file_path in enumerate(tqdm(file_names, desc="Processing files")):
    with open(file_path, 'r') as f:
        data = json.load(f)

    if data['info']['overs'] != 20:
        continue

    for i in range(2):
        if i >= len(data['innings']):
            continue

        overs = data['innings'][i]['overs']
        for j, over in enumerate(overs):
            over_count[j] += 1
            if len(over['deliveries']) <= 6:
                continue

            for delivery in over['deliveries']:
                extras = delivery.get('extras', {})
                if 'wides' in extras:
                    wides[j] += 1
                elif 'noballs' in extras:
                    noballs[j] += 1

wides_arr = np.array(list(wides.values()))
noballs_arr = np.array(list(noballs.values()))
over_count_arr = np.array(list(over_count.values()))

# plot
wides_perover = wides_arr/over_count_arr
noballs_perover = noballs_arr/over_count_arr
over_list = np.array(list(wides.keys()))+1

wides_perover_err = wides_perover * np.sqrt(1/wides_arr + 1/over_count_arr)
noballs_perover_err = noballs_perover * np.sqrt(1/noballs_arr + 1/over_count_arr)

fig, ax = plt.subplots(figsize=(14,7))
#plt.bar(x=over_list,height=wides_perover,width=1,align='center',color='white',label='wides',edgecolor='navy')
#plt.bar(x=over_list,height=noballs_perover,width=1,align='center',color='white',label='no balls',edgecolor='crimson')
plt.errorbar(x=over_list, y=wides_perover, yerr=wides_perover_err, marker='x', ls='', color='darkblue', label='wides')
plt.errorbar(x=over_list, y=noballs_perover, yerr=noballs_perover_err, marker='x', ls='', color='crimson', label='no balls')
plt.xticks(ticks=over_list)
plt.legend(loc='best')
plt.xlabel("Over")
plt.xlim([0.5,20.5])
plt.tight_layout()
#plt.show()
plt.savefig("plots/extras_plot.pdf")
plt.close()


