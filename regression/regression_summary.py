import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.tseries.offsets import MonthEnd
from prettytable import PrettyTable, MARKDOWN
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from regression.regression_models import *
from utils.utils import Dataset, get_dataset


def get_regressors():
    return [
        'Lineare',
        'Lineare robusto (Huber)',
        'Lineare avanzato (Cook)',
        'Ridge',
        'Lasso',
        'Polinomiale (grado 2)',
        'Polinomiale (grado 3)',
        'Random Forest',
        'Gradient Boosting',
        'SVR (Kernel lineare)',
        'SVR (Kernel polinomiale)',
        'SVR (Kernel RBF)',
        'KernelRidge'
    ]


def get_regressors_smaller():
    return [
        'Lineare',
        'Lineare robusto (Huber)',
        'Lineare avanzato (Cook)',
        'Ridge',
        'Lasso',
        'Polinomiale (grado 2)',
        'Polinomiale (grado 3)',
        'Random Forest',
        'Gradient Boosting',
        'SVR (lineare)',
        'SVR (polinomiale)',
        'SVR (RBF)',
        'KernelRidge'
    ]


def avg(scores):
    return '%0.2f' % (sum(scores) / len(scores))


def make_avg(predictor, X, y, n=500):
    r2_scores, rmse_scores = [], []
    for _ in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
        predictor.fit(X_train, y_train)
        r2_scores.append(predictor.score(X_test, y_test))
        rmse_scores.append(mean_squared_error(y_test, predictor.predict(X_test), squared=False))
    return avg(r2_scores), avg(rmse_scores)


def apply_linear_regression(X, y):
    return make_avg(get_linear_model(), X, y)


def apply_cook_regression(X, y):
    m = sm.OLS(y, sm.add_constant(X)).fit()
    infl = m.get_influence()
    soglia = 4 / len(X)
    (a, p) = infl.cooks_distance
    mask = a < soglia
    return make_avg(get_linear_model(), X[mask], y[mask])


def apply_huber_regression(X, y):
    return make_avg(get_huber_model(), X, y)


def apply_lasso_regression(X, y):
    lassoregcv = LassoCV(n_alphas=100, random_state=1)
    model = make_pipeline(StandardScaler(with_mean=False), lassoregcv)
    return make_avg(model, X, y)


def apply_ridge_regression(X, y):
    alpha_range = 10. ** np.arange(-2, 5)
    ridgereg = RidgeCV(alphas=alpha_range, scoring='neg_mean_squared_error')
    model = make_pipeline(StandardScaler(with_mean=False), ridgereg)
    return make_avg(model, X, y)


def apply_poly_regression(X, y):
    return make_avg(get_polynomial_model(), X, y)


def apply_poly_3_regression(X, y):
    m = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), linear_model.LinearRegression(n_jobs=-1))
    return make_avg(m, X, y)


def apply_kridge_regression(X, y):
    kr = make_pipeline(StandardScaler(with_mean=False),
                       KernelRidge(alpha=1, kernel='poly', degree=2, gamma=1, coef0=1))
    return make_avg(kr, X, y)


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
    cook_score, cook_rmse = apply_cook_regression(X, y)

    ridge_score, ridge_rmse = apply_ridge_regression(X, y)
    lasso_score, lasso_rmse = apply_lasso_regression(X, y)
    kridge_score, kridge_rmse = apply_kridge_regression(X, y)

    poly_score, poly_rmse = apply_poly_regression(X, y)
    poly3_score, poly3_rmse = apply_poly_3_regression(X, y)
    rf_score, rf_rmse = apply_random_forest_regression(X, y)
    gb_score, gb_rmse = apply_gradient_boosting_regression(X, y)
    svr_lin_score, svr_lin_rmse = apply_svr_linear_regression(X, y)
    svr_poly_score, svr_poly_rmse = apply_svr_polynomial_regression(X, y)
    svr_rbf_score, svr_rbf_rmse = apply_rbf_polynomial_regression(X, y)

    r2_scores = [lr_score, huber_score, cook_score, ridge_score, lasso_score, poly_score, poly3_score, rf_score,
                 gb_score, svr_lin_score, svr_poly_score,
                 svr_rbf_score, kridge_score]
    rmse_scores = [lr_rmse, huber_rmse, cook_rmse, ridge_rmse, lasso_rmse, poly_rmse, poly3_rmse, rf_rmse, gb_rmse,
                   svr_lin_rmse, svr_poly_rmse,
                   svr_rbf_rmse, kridge_rmse]

    return r2_scores, rmse_scores


def print_annual_results(results, title):
    table = PrettyTable()
    # table.title = title
    table.add_column('Modello di regressione', get_regressors())
    table.add_column('R²', results[0])
    table.add_column('RMSE (µg/m³)', results[1])
    table.align = 'l'
    table.set_style(MARKDOWN)
    print(table)


def print_monthly_results(month_results, title):
    def construct_table(i):
        t = PrettyTable()
        # t.title = '{} | '.format('R²' if i == 0 else 'RMSE (µg/m³)') + title
        t.add_column('Modello', get_regressors_smaller())
        for res in month_results:
            if any(res['results'][i]):
                t.add_column(res['period'], res['results'][i])
        t.align = 'l'
        return t

    # R²
    table = construct_table(0)
    table.set_style(MARKDOWN)
    print(table)

    # RMSE
    table = construct_table(1)
    table.set_style(MARKDOWN)
    print(table)


def annual_summary(dataset, station, chemical):
    X = dataset['airqino_{}'.format(chemical)].values.reshape((-1, 1))
    y = dataset['arpat_{}'.format(chemical)].values
    results = evaluate(X, y)
    print_annual_results(results,
                         title='All year | {n} | {c} (n={l})'.format(c=chemical.upper(), n=station,
                                                                     l=str(len(dataset))))


def monthly_summary(dataset, station, chemical, is_24h=False):
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
        threshold = 5 if is_24h else 70
        if month_dataset.empty or len(month_dataset.index) < threshold:
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

    print_monthly_results(month_results, title='Monthly results | {d} | {c}'.format(c=chemical.upper(), d=station))


if __name__ == '__main__':
    # SMART16 - NO2
    # dataset = get_dataset(Dataset.SMART16_NO2).loc['2020-01-18': '2020-12-31']
    # annual_summary(dataset=dataset, station='SMART16-CAPANNORI', chemical='no2')
    # monthly_summary(dataset=dataset, station='SMART16-CAPANNORI', chemical='no2')

    # SMART16_new - PM2.5
    # annual_summary(dataset=get_dataset(Dataset.SMART16_NEW_PM), station='SMART16_new-CAPANNORI', chemical='pm2.5')
    # monthly_summary(dataset=get_dataset(Dataset.SMART16_NEW_PM), station='SMART16_new-CAPANNORI', chemical='pm2.5')

    # SMART16_new - PM10
    # annual_summary(dataset=get_dataset(Dataset.SMART16_NEW_PM), station='SMART16_new-CAPANNORI', chemical='pm10')
    # monthly_summary(dataset=get_dataset(Dataset.SMART16_NEW_PM), station='SMART16_new-CAPANNORI', chemical='pm10')

    # SMART16_new - PM2.5 - 8H
    # annual_summary(get_dataset(Dataset.SMART16_NEW_PM_12H), 'SMART16_new-CAPANNORI-8h', 'pm2.5')
    # monthly_summary(get_dataset(Dataset.SMART16_NEW_PM_12H), 'SMART16_new-CAPANNORI-12h', 'pm2.5', is_24h=True)

    # SMART16_new - PM10 - 8H
    # annual_summary(get_dataset(Dataset.SMART16_NEW_PM_12H), 'SMART16_new-CAPANNORI-8h', 'pm10')
    monthly_summary(get_dataset(Dataset.SMART16_NEW_PM_12H), 'SMART16_new-CAPANNORI-8h', 'pm10', is_24h=True)
