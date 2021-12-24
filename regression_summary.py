import pandas as pd
from pandas.tseries.offsets import MonthEnd
from prettytable import PrettyTable
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

from utils import get_no2_dataset, Dataset


def avg(scores):
    return '%0.4f' % (sum(scores) / len(scores))


def make_avg(predictor, X, y, n=5):
    r2_scores, rmse_scores = [], []
    for _ in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
        predictor.fit(X_train, y_train)
        r2_scores.append(predictor.score(X_test, y_test))
        rmse_scores.append(mean_squared_error(y_test, predictor.predict(X_test), squared=False))
    return avg(r2_scores), avg(rmse_scores)


def apply_linear_regression(X, y):
    lr = linear_model.LinearRegression()
    return make_avg(lr, X, y)


def apply_huber_regression(X, y):
    huber = linear_model.HuberRegressor(epsilon=3)
    return make_avg(huber, X, y)


def apply_poly_regression(X, y):
    pipe = make_pipeline(PolynomialFeatures(degree=2), linear_model.LinearRegression())
    return make_avg(pipe, X, y)


def apply_random_forest_regression(X, y):
    rf = RandomForestRegressor(n_estimators=10)
    return make_avg(rf, X, y)


def apply_gradient_boosting_regression(X, y):
    params = {'learning_rate': 0.01, 'loss': 'squared_error', 'max_depth': 3, 'min_samples_split': 10,
              'n_estimators': 100}
    gb = GradientBoostingRegressor(**params)
    return make_avg(gb, X, y)


def apply_svr_linear_regression(X, y):
    svr_lin = SVR(kernel='linear', C=10, gamma='scale')
    return make_avg(svr_lin, X, y)  # , make_avg(svr_poly, X, y), make_avg(svr_rbf, X, y)


def apply_svr_polynomial_regression(X, y):
    svr_poly = SVR(kernel='poly', C=10, gamma='scale', degree=2)
    return make_avg(svr_poly, X, y)


def apply_rbf_polynomial_regression(X, y):
    svr_rbf = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    return make_avg(svr_rbf, X, y)


def evaluate(X, y):
    lr_score, lr_rmse = apply_linear_regression(X, y)
    huber_score, huber_rmse = apply_huber_regression(X, y)
    poly_score, poly_rmse = apply_poly_regression(X, y)
    rf_score, rf_rmse = apply_random_forest_regression(X, y)
    gb_score, gb_rmse = apply_gradient_boosting_regression(X, y)
    svr_lin_score, svr_lin_rmse = apply_svr_linear_regression(X, y)
    svr_poly_score, svr_poly_rmse = apply_svr_polynomial_regression(X, y)
    svr_rbf_score, svr_rbf_rmse = apply_rbf_polynomial_regression(X, y)

    r2_scores = [lr_score, huber_score, poly_score, rf_score, gb_score, svr_lin_score, svr_poly_score, svr_rbf_score]
    rmse_scores = [lr_rmse, huber_rmse, poly_rmse, rf_rmse, gb_rmse, svr_lin_rmse, svr_poly_rmse, svr_rbf_rmse]

    return r2_scores, rmse_scores


def print_annual_results(regressors_list, results, title):
    table = PrettyTable()
    table.title = title
    table.add_column('Regressor', regressors_list)
    table.add_column('R²', results[0])
    table.add_column('RMSE', results[1])
    table.align = 'l'
    # table.set_style(MARKDOWN)
    print(table)


def print_monthly_results(regressors_list, month_results, title):
    table = PrettyTable()
    table.title = title
    table.add_column('Regressor', regressors_list)
    for res in month_results:
        table.add_column(res['period'] + ' (' + res['values_count'] + ')', res['results'][0])
    table.align = 'l'
    print(table)


if __name__ == '__main__':

    regressors = [
        'Linear Regression',
        'Huber Regression',
        'Polynomial Regression',
        'Random Forest Regression',
        'Gradient Boosting Regression',
        'SVR Regression (Linear Kernel)',
        'SVR Regression (Polynomial Kernel)',
        'SVR Regression (RBF Kernel)'
    ]

    # NO2
    for d in [Dataset.SMART16, Dataset.SMART24, Dataset.SMART25, Dataset.SMART26]:
        dataset = get_no2_dataset(d)

        X = dataset['airqino_no2'].values.reshape((-1, 1))
        y = dataset['arpat_no2'].values
        results = evaluate(X, y)
        print_annual_results(regressors, results, title='All year - NO2 for {}'.format(d.name))

        month_results = []
        for month in pd.date_range('2020-01-01', '2020-12-31', freq='MS'):
            month_start = month.strftime('%Y-%m-%d')
            month_end = (month + MonthEnd(1)).strftime('%Y-%m-%d')

            month_dataset = dataset.loc[month_start: month_end]
            if month_dataset.empty or len(month_dataset.index) < 30:
                print('too few values for {m1} -> {m2} - NO2 - {d}'.format(m1=month_start, m2=month_end, d=d.name))
                continue

            X = month_dataset['airqino_no2'].values.reshape((-1, 1))
            y = month_dataset['arpat_no2'].values

            results = evaluate(X, y)
            month_results.append({'results': results, 'period': month.strftime('%b'), 'values_count': str(len(y))})

        print_monthly_results(regressors, month_results, title='Monthly results NO2 for {d}'.format(d=d.name))

    # todo: add pm2.5 and pm10 too
