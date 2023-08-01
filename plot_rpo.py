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
directory = Path("/Users/Rizwaan/Documents/cricket_analytics/data/ipl_json")
file_names = [f for f in directory.iterdir() if f.name != "README.txt"]

# Dicts for over and runs
over_count = {i: 0 for i in range(20)}
run_count = {i: [] for i in range(20)}

# Loop over files
for file_num, file_path in enumerate(tqdm(file_names, desc="Processing files")):
    with open(file_path, 'r') as f:
        data = json.load(f)

    for i in range(2):
        if i >= len(data['innings']):
            continue

        overs = data['innings'][i]['overs']
        for j, over in enumerate(overs):
            tot_runs = 0
            over_count[j] += 1

            for delivery in over['deliveries']:
                runs = delivery.get('runs', {})
                tot_runs += runs['total']

            run_count[j].append(tot_runs)

over_count_arr = np.array(list(over_count.values()))

# Calculate average and uncertainty
rpo_arr = []
rpo_err_arr = []
for over in run_count:
    rpo_arr.append(np.average(run_count[over]))
    rpo_err_arr.append(np.std(run_count[over])/np.sqrt(len(run_count[over])))

# plot
over_list = np.array(list(run_count.keys()))+1

fig, ax = plt.subplots(figsize=(14,7))
plt.errorbar(x=over_list[0:6], y=rpo_arr[0:6], yerr=rpo_err_arr[0:6], marker='x', ls='', color='darkblue', label='Powerplay')
plt.errorbar(x=over_list[6:], y=rpo_arr[6:], yerr=rpo_err_arr[6:], marker='x', ls='', color='darkgreen', label='Non-powerplay')
plt.xticks(ticks=over_list)
plt.axvline(x=6.5,ls='--',color='grey',lw=0.75)
plt.xlabel("Over")
plt.ylabel("Run rate")
plt.xlim([0.5,20.5])
plt.legend(loc='best')
plt.tight_layout()
#plt.show()
plt.savefig("plots/rpo_plot.pdf")
plt.close()


