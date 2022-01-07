import pandas as pd
from pandas.tseries.offsets import MonthEnd
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from regression_models import *
from utils import Dataset, get_dataset


def get_regressors():
    return [
        'Linear',
        'Huber',
        'Polynomial',
        'Random Forest',
        'Gradient Boosting',
        'SVR (Linear Kernel)',
        'SVR (Polynomial Kernel)',
        'SVR (RBF Kernel)'
    ]


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
    return make_avg(get_linear_model(), X, y)


def apply_huber_regression(X, y):
    return make_avg(get_huber_model(), X, y)


def apply_poly_regression(X, y):
    return make_avg(get_polynomial_model(), X, y)


def apply_random_forest_regression(X, y):
    return make_avg(get_random_forest_model(), X, y)


def apply_gradient_boosting_regression(X, y):
    return make_avg(get_gradient_boosting_model(), X, y)


def apply_svr_linear_regression(X, y):
    return make_avg(get_svr_linear_model(), X, y)


def apply_svr_polynomial_regression(X, y):
    return make_avg(get_svr_polynomial_model(), X, y)


def apply_rbf_polynomial_regression(X, y):
    return make_avg(get_svr_rbf_model(), X, y)


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


def print_annual_results(results, title):
    table = PrettyTable()
    table.title = title
    table.add_column('Regression Model', get_regressors())
    table.add_column('R²', results[0])
    table.add_column('RMSE', results[1])
    table.align = 'l'
    # table.set_style(MARKDOWN)
    print(table)


def print_monthly_results(month_results, title):
    def construct_table(i):
        t = PrettyTable()
        t.title = '{} | '.format('R²' if i == 0 else 'RMSE') + title
        t.add_column('Regressor', get_regressors())
        for res in month_results:
            t.add_column(res['period'] + ' (' + res['values_count'] + ')', res['results'][i])
        t.align = 'l'
        return t

    # R²
    table = construct_table(0)
    print(table)

    # RMSE
    table = construct_table(1)
    print(table)


def annual_summary(dataset, station, chemical):
    X = dataset['airqino_{}'.format(chemical)].values.reshape((-1, 1))
    y = dataset['arpat_{}'.format(chemical)].values
    results = evaluate(X, y)
    print_annual_results(results, title='All year | {c} | {n}'.format(c=chemical.upper(), n=station))


def monthly_summary(dataset, station, chemical):
    if chemical == 'no2':
        date_range = pd.date_range('2020-01-01', '2020-12-31', freq='MS')
    else:
        date_range = pd.date_range('2020-09-01', '2021-08-31', freq='MS')
    month_results = []
    for month in date_range:
        month_str = month.strftime('%b')
        month_start = month.strftime('%Y-%m-%d')
        month_end = (month + MonthEnd(1)).strftime('%Y-%m-%d')

        month_dataset = dataset.loc[month_start: month_end]
        if month_dataset.empty or len(month_dataset.index) < 30:
            month_results.append({
                'results': [[0 for _ in get_regressors()], [0 for _ in get_regressors()]],
                'period': month_str,
                'values_count': str(len(month_dataset.index))
            })
            continue

        X = month_dataset['airqino_{}'.format(chemical)].values.reshape((-1, 1))
        y = month_dataset['arpat_{}'.format(chemical)].values

        results = evaluate(X, y)
        month_results.append({'results': results, 'period': month_str, 'values_count': str(len(y))})

    print_monthly_results(month_results, title='Monthly results | {c} | {d}'.format(c=chemical.upper(), d=station))


if __name__ == '__main__':
    # SMART16 - NO2
    annual_summary(dataset=get_dataset(Dataset.SMART16_NO2), station='SMART16-CAPANNORI', chemical='no2')
    monthly_summary(dataset=get_dataset(Dataset.SMART16_NO2), station='SMART16-CAPANNORI', chemical='no2')

    # SMART16 - PM2.5
    annual_summary(dataset=get_dataset(Dataset.SMART16_NEW_PM), station='SMART16_new-CAPANNORI', chemical='pm2.5')
    monthly_summary(dataset=get_dataset(Dataset.SMART16_NEW_PM), station='SMART16_new-CAPANNORI', chemical='pm2.5')

    # SMART16 - PM10
    annual_summary(dataset=get_dataset(Dataset.SMART16_NEW_PM), station='SMART16_new-CAPANNORI', chemical='pm10')
    monthly_summary(dataset=get_dataset(Dataset.SMART16_NEW_PM), station='SMART16_new-CAPANNORI', chemical='pm10')
