# cricket_analytics

Uses data from 1024 IPL matches, taken from: https://cricsheet.org/matches/

`score_regressor.py` trains a basic score predicting model, based on the current score, wickets remaining, and balls remaining. The regression algorithm parameters are optimised, and the accuracy of the model is plotted as a function of the innings position.

`plot_extras.py` calculates the average number of wides and no-balls per over, for each over in the innings (i.e. 1-20). It plots the variation of wide and no-ball frequency per over throughout the innings.

`plot_rpo.py` calculates the run rate for each over in the innings and plots it.

`plot_balls_faced.py` calculates the average number of balls faced per batting position and plots it

The json files for each match are saved in the `data/ipl_json/` directory. All plots are saved in the `plots/` directory.
