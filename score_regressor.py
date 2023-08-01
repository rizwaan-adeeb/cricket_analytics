import json
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import xgboost as xgb

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
plt.rcParams['text.usetex'] = True

def process_data(file_names):
    # Create lists of current score, wickets, balls left, total score
    score_list = []
    wickets_list = []
    balls_list = []
    final_score_list = []

    for file_path in tqdm(file_names, desc="Processing files"):
        with open(file_path, 'r') as f:
            data = json.load(f)

        innings_list = data['innings'][0]

        #if len(innings_list['overs']) != 20:
        #    continue

        n_overs = len(innings_list['overs'])

        current_score = 0
        wickets_lost = 0
        balls_gone = 0

        for i in range(n_overs):
            n_deliveries = len(innings_list['overs'][i]['deliveries'])

            for j in range(n_deliveries):
                delivery = innings_list['overs'][i]['deliveries'][j]

                current_score += delivery['runs']['total']
                if 'wickets' in delivery:
                    wickets_lost += 1

                extras = delivery.get('extras', {})
                if 'wides' in extras or 'noballs' in extras:
                    continue
                else:
                    balls_gone += 1

                score_list.append(current_score)
                wickets_list.append(wickets_lost)
                balls_list.append(balls_gone)

        final_score_list = np.append(final_score_list,[current_score]*balls_gone)

    return pd.DataFrame({"Score": score_list, "Wickets": wickets_list, "Balls gone": balls_list, "Final score": final_score_list})

def split_data(df):
    df_train, df_test = train_test_split(df, test_size=0.4)
    X_train = df_train[['Wickets', 'Balls gone', 'Score']]
    X_test = df_test[['Wickets', 'Balls gone', 'Score']]
    Y_train = df_train['Final score']
    Y_test = df_test['Final score']
    return X_train, X_test, Y_train, Y_test, df_test

def train_xgboost_model(X_train, Y_train, do_grid_search=False):
    model = xgb.XGBRegressor()

    if do_grid_search:
        print(":: Grid search ::")
        params = {
            'learning_rate': [0.001, 0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'n_estimators': [50, 100, 200, 500],
        }
        grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train, Y_train)
        best_params = grid_search.best_params_
        print("Best params:", best_params)
    else:
        best_params = {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 100}
        print(":: Using previous best params ::")
        print("Best params:", best_params)

    xgb_model_best = xgb.XGBRegressor(**best_params)
    xgb_model_best.fit(X_train, Y_train)
    return xgb_model_best

def evaluate_model(model, X_test, df_test):
    ave_err_list = []
    mse_list = []

    for i in np.arange(0, 20, 2):
        print(f"Over range: {i} to {i+2}")
        df_test_split = df_test[df_test['Balls gone'].between(i*6, (i+2)*6)]
        Y_predict_split = np.array(model.predict(df_test_split[['Wickets', 'Balls gone', 'Score']]))
        Y_test_split = np.array(df_test_split['Final score'])

        ave_err = np.average(np.abs(Y_predict_split - Y_test_split))
        print("Ave error:", ave_err)

        # Calculate the mean squared error
        mse = mean_squared_error(Y_test_split, Y_predict_split)
        print("Mean Squared Error:", mse)

        ave_err_list.append(ave_err)
        mse_list.append(mse)

    return ave_err_list, mse_list

if __name__ == "__main__":
    do_grid_search = False

    # Get file names
    directory = Path("/Users/Rizwaan/Documents/cricket_analytics/cricket_analytics/data/ipl_json")
    file_names = [f for f in directory.iterdir() if f.name != "README.txt"]

    df = process_data(file_names)
    print(df)

    X_train, X_test, Y_train, Y_test, df_test = split_data(df)
    model = train_xgboost_model(X_train, Y_train, do_grid_search)
    ave_err_list, mse_list = evaluate_model(model, X_test, df_test)

    # Plotting the results
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.plot(np.arange(1, 21, 2), ave_err_list, ls='', marker='x', color='crimson')
    plt.xticks(np.arange(0, 22, 2))
    plt.xlabel("Over")
    plt.ylabel("Average error")
    plt.tight_layout()
    #plt.show()
    plt.savefig("plots/regressor_score_plot.pdf")
    plt.close()

