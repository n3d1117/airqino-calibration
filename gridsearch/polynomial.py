import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def main():
    df = pd.read_csv('../generated_data/merged/smart16-capannori-no2.csv')
    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    X = df['airqino_no2'].values.reshape((-1, 1))
    y = df['arpat_no2'].values

    def PolynomialRegression(degree=2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

    param_grid = {'polynomialfeatures__degree': [2, 3, 4]}
    grid = GridSearchCV(PolynomialRegression(), param_grid, cv=5, n_jobs=-1, verbose=2)
    grid.fit(X, y)

    print('Results from Grid Search Polynomial Regression')
    print('\n The best score across ALL searched params:\n', grid.best_score_)
    print('\n The best parameters across ALL searched params:\n', grid.best_params_)

    # {'degree': '2'}


if __name__ == '__main__':
    main()
