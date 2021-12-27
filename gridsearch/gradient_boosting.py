import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


def main():
    df = pd.read_csv('../generated_data/merged/smart25-san-concordio-no2.csv')
    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    X = df['airqino_no2'].values.reshape((-1, 1))
    y = df['arpat_no2'].values

    param = [{'n_estimators': [10, 20, 50, 100], 'max_depth': [3, 5, 10],
              'learning_rate': [.01, .1, .5, 1], 'loss': ['squared_error'],
              'criterion': ['friedman_mse', 'squared_error'],
              'min_samples_split': [2, 5, 15], 'min_samples_leaf': [1, 2, 5],
              'min_impurity_decrease': [0, .1, .5, 1], 'max_features': ['auto', 'sqrt', 'log2'],
              'max_leaf_nodes': [None, 2, 5, 10]}]
    grids = GridSearchCV(GradientBoostingRegressor(), param, cv=3, n_jobs=-1, verbose=2)
    grids.fit(X, y)

    print('Results from Grid Search Gradient Boosting')
    print('\n The best score across ALL searched params:\n', grids.best_score_)
    print('\n The best parameters across ALL searched params:\n', grids.best_params_)

    #  {'criterion': 'friedman_mse', 'learning_rate': 0.01, 'loss': 'squared_error', 'max_depth': 3, 'max_features': 'auto', 'max_leaf_nodes': 2, 'min_impurity_decrease': 0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}


if __name__ == '__main__':
    main()
