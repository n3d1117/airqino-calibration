import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def main():
    df = pd.read_csv('../generated_data/merged/smart16-capannori-no2.csv')
    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    X = df['airqino_no2'].values.reshape((-1, 1))
    y = df['arpat_no2'].values

    param = [{'n_estimators': [10, 20, 30, 100], 'max_depth': [None, 2, 5],
              'min_samples_split': [2, 5, 10], 'min_samples_leaf': [2, 5, 10],
              'criterion': ['squared_error'], 'min_weight_fraction_leaf': [0, .1, 1],
              'max_features': ['auto', 'sqrt', 'log2', 1], 'max_leaf_nodes': [None, 50],
              'min_impurity_decrease': [0, .5, 1, 10]
              }]
    grids = GridSearchCV(RandomForestRegressor(), param, cv=3, n_jobs=-1, verbose=2)
    grids.fit(X, y)

    print('Results from Grid Search Random Forest')
    print('\n The best score across ALL searched params:\n', grids.best_score_)
    print('\n The best parameters across ALL searched params:\n', grids.best_params_)

    #  {'criterion': 'squared_error', 'max_depth': 5, 'max_features': 'sqrt', 'max_leaf_nodes': 50, 'min_impurity_decrease': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'min_weight_fraction_leaf': 0, 'n_estimators': 10}


if __name__ == '__main__':
    main()
