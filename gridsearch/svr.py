import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


def main():
    df = pd.read_csv('../generated_data/merged/smart16-capannori-no2.csv')
    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    X = df['airqino_no2'].values.reshape((-1, 1))
    y = df['arpat_no2'].values

    param = [{'kernel': ['rbf'], 'C': [1, 5, 10, 20, 100], 'degree': [2, 3],
              'coef0': [0, .5, 1, 5], 'gamma': ['scale'], 'epsilon': [.1, .3, .5, 1, 5],
              'shrinking': [False, True]}]
    grids = GridSearchCV(SVR(), param, cv=3, n_jobs=-1, verbose=2)
    grids.fit(X, y)

    print('Results from Grid Search SVR Regression')
    print('\n The best score across ALL searched params:\n', grids.best_score_)
    print('\n The best parameters across ALL searched params:\n', grids.best_params_)

    #  {'C': 1, 'coef0': 0, 'degree': 2, 'epsilon': 5, 'gamma': 'scale', 'kernel': 'rbf', 'shrinking': True}


if __name__ == '__main__':
    main()
