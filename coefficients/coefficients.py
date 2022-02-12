import numpy as np
import statsmodels.api as sm
from prettytable import PrettyTable
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from utils.utils import get_dataset, Dataset


def avg(scores):
    return '%0.3f' % (sum(scores) / len(scores))


#
# # y = a + b*x + c*x^2
# def make_avg(predictor, X, y, n=1000):
#     r2_scores, rmse_scores, a, b, c = [], [], [], [], []
#     for _ in range(n):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
#         predictor.fit(X_train, y_train)
#         # print(predictor.steps[0][1].get_feature_names_out())
#         a.append(predictor.steps[1][1].intercept_)
#         b.append(predictor.steps[1][1].coef_[0])
#         c.append(predictor.steps[1][1].coef_[1])
#         r2_scores.append(predictor.score(X_test, y_test))
#         rmse_scores.append(mean_squared_error(y_test, predictor.predict(X_test), squared=False))
#     return avg(r2_scores), avg(rmse_scores), avg(a), avg(b), avg(c)
#
#
# def apply_poly_regression(X, y):
#     return make_avg(get_polynomial_model(), X, y)
#
#
# def print_annual_results(results, title):
#     table = PrettyTable()
#     table.title = title
#     table.add_column('Regressor', ['Polynomial'])
#     table.add_column('R²', results[0])
#     table.add_column('RMSE', results[1])
#     table.add_column('a', results[2])
#     table.add_column('b', results[3])
#     table.add_column('c', results[4])
#     table.align = 'l'
#     # table.set_style(MARKDOWN)
#     print(table)
#
#
# def evaluate(X, y):
#     poly_score, poly_rmse, a, b, c = apply_poly_regression(X, y)
#     r2_scores = [poly_score]
#     rmse_scores = [poly_rmse]
#     return r2_scores, rmse_scores, [a], [b], [c]
#
#
# def annual_summary(X, y, station, chemical):
#     results = evaluate(X, y)
#     print_annual_results(results, title='All year | {n} | {c}'.format(c=chemical.upper(), n=station))

def print_results(results, title, regs):
    table = PrettyTable()
    table.title = title
    table.add_column('Regressor', regs)
    table.add_column('R²', results[0])
    table.add_column('RMSE (µg/m³)', results[1])
    table.add_column('a', results[2])
    table.add_column('b', results[3])
    table.add_column('c', results[4])
    table.add_column('d', results[5])
    table.add_column('e', results[6])
    table.align = 'l'
    # table.set_style(MARKDOWN)
    print(table)


if __name__ == '__main__':
    dataset = get_dataset(Dataset.SMART16_NEW_PM_8H)
    X = dataset['airqino_{}'.format('pm2.5')].values.reshape((-1, 1))
    y = dataset['arpat_{}'.format('pm2.5')].values

    linear_r2, linear_a, linear_b, linear_rmse = [], [], [], []
    lasso_r2, lasso_a, lasso_b, lasso_rmse = [], [], [], []
    ridge_r2, ridge_a, ridge_b, ridge_rmse = [], [], [], []
    linear_cook_r2, linear_cook_a, linear_cook_b, linear_cook_rmse = [], [], [], []
    poly2_r2, poly2_a, poly2_b, poly2_c, poly2_rmse = [], [], [], [], []
    poly3_r2, poly3_a, poly3_b, poly3_c, poly3_d, poly3_rmse = [], [], [], [], [], []
    poly4_r2, poly4_a, poly4_b, poly4_c, poly4_d, poly4_e, poly4_rmse = [], [], [], [], [], [], []

    for _ in range(5000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

        # Linear: y = a + b*x
        linreg = LinearRegression()
        linear = make_pipeline(StandardScaler(with_mean=False), linreg)
        linear.fit(X_train, y_train)
        y_pred = linear.predict(X_test)
        linear_r2.append(r2_score(y_test, y_pred))
        linear_rmse.append(mean_squared_error(y_test, y_pred, squared=False))
        linear_a.append(linreg.coef_)
        linear_b.append(linreg.intercept_)

        # Linear Robust (Cook): y = a + b*x
        cook_model_result = sm.OLS(y_train, sm.add_constant(X_train)).fit()
        infl = cook_model_result.get_influence()
        soglia = 4 / len(X)
        (a, p) = infl.cooks_distance
        mask = a < soglia
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_train[mask], y_train[mask],
                                                                            test_size=0.25)
        linreg_cook = LinearRegression()
        linear_cook = make_pipeline(StandardScaler(with_mean=False), linreg_cook)
        linear_cook.fit(X_train_new, y_train_new)
        y_pred = linear_cook.predict(X_test_new)
        linear_cook_r2.append(r2_score(y_test_new, y_pred))
        linear_cook_rmse.append(mean_squared_error(y_test_new, y_pred, squared=False))
        linear_cook_a.append(linreg_cook.coef_)
        linear_cook_b.append(linreg_cook.intercept_)

        # Polynomial 2: y = a + b*x + c*x^2
        polreg = make_pipeline(StandardScaler(with_mean=False), PolynomialFeatures(degree=2, include_bias=False),
                               linear_model.LinearRegression(n_jobs=-1))
        polreg.fit(X_train, y_train)
        y_pred = polreg.predict(X_test)
        poly2_r2.append(r2_score(y_test, y_pred))
        poly2_rmse.append(mean_squared_error(y_test, y_pred, squared=False))
        poly2_a.append(polreg.steps[2][1].intercept_)
        poly2_b.append(polreg.steps[2][1].coef_[0])
        poly2_c.append(polreg.steps[2][1].coef_[1])

        # Polynomial 3: y = a + b*x + c*x^2 + d*x^3
        polreg = make_pipeline(StandardScaler(with_mean=False), PolynomialFeatures(degree=3, include_bias=False),
                               linear_model.LinearRegression(n_jobs=-1))
        polreg.fit(X_train, y_train)
        y_pred = polreg.predict(X_test)
        poly3_r2.append(r2_score(y_test, y_pred))
        poly3_rmse.append(mean_squared_error(y_test, y_pred, squared=False))
        poly3_a.append(polreg.steps[2][1].intercept_)
        poly3_b.append(polreg.steps[2][1].coef_[0])
        poly3_c.append(polreg.steps[2][1].coef_[1])
        poly3_d.append(polreg.steps[2][1].coef_[2])

        # Polynomial 4: y = a + b*x + c*x^2 + d*x^3 + e*x^4
        polreg = make_pipeline(StandardScaler(with_mean=False), PolynomialFeatures(degree=4, include_bias=False),
                               linear_model.LinearRegression(n_jobs=-1))
        polreg.fit(X_train, y_train)
        y_pred = polreg.predict(X_test)
        poly4_r2.append(r2_score(y_test, y_pred))
        poly4_rmse.append(mean_squared_error(y_test, y_pred, squared=False))
        poly4_a.append(polreg.steps[2][1].intercept_)
        poly4_b.append(polreg.steps[2][1].coef_[0])
        poly4_c.append(polreg.steps[2][1].coef_[1])
        poly4_d.append(polreg.steps[2][1].coef_[2])
        poly4_e.append(polreg.steps[2][1].coef_[3])

        # Lasso: y = a + b*x
        lassoregcv = LassoCV(n_alphas=100, random_state=1)
        lasso = make_pipeline(StandardScaler(with_mean=False), lassoregcv)
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)
        lasso_r2.append(r2_score(y_test, y_pred))
        lasso_rmse.append(mean_squared_error(y_test, y_pred, squared=False))
        lasso_a.append(lasso.steps[1][1].coef_)
        lasso_b.append(lasso.steps[1][1].intercept_)

        # Ridge: y = a + b*x
        alpha_range = 10. ** np.arange(-2, 5)
        ridgereg = RidgeCV(alphas=alpha_range, scoring='neg_mean_squared_error')
        ridge = make_pipeline(StandardScaler(with_mean=False), ridgereg)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        ridge_r2.append(r2_score(y_test, y_pred))
        ridge_rmse.append(mean_squared_error(y_test, y_pred, squared=False))
        ridge_a.append(ridge.steps[1][1].coef_)
        ridge_b.append(ridge.steps[1][1].intercept_)

    l_r2, l_rmse, l_a, l_b = avg(linear_r2), avg(linear_rmse), avg(linear_a), avg(linear_b)
    lasso_r2, lasso_rmse, lasso_a, lasso_b = avg(lasso_r2), avg(lasso_rmse), avg(lasso_a), avg(lasso_b)
    ridge_r2, ridge_rmse, ridge_a, ridge_b = avg(ridge_r2), avg(ridge_rmse), avg(ridge_a), avg(ridge_b)
    l_cook_r2, l_cook_rmse, l_cook_a, l_cook_b = avg(linear_cook_r2), avg(linear_cook_rmse), avg(linear_cook_a), avg(
        linear_cook_b)
    poly2_r2, poly2_rmse, poly2_a, poly2_b, poly2_c = avg(poly2_r2), avg(poly2_rmse), avg(poly2_a), avg(poly2_b), avg(
        poly2_c)
    poly3_r2, poly3_rmse, poly3_a, poly3_b, poly3_c, poly3_d = avg(poly3_r2), avg(poly3_rmse), avg(poly3_a), avg(
        poly3_b), avg(poly3_c), avg(poly3_d)
    poly4_r2, poly4_rmse, poly4_a, poly4_b, poly4_c, poly4_d, poly4_e = avg(poly4_r2), avg(poly4_rmse), avg(
        poly4_a), avg(
        poly4_b), avg(poly4_c), avg(poly4_d), avg(poly4_e)

    print_results(
        results=[
            [l_r2, l_cook_r2, lasso_r2, ridge_r2, poly2_r2, poly3_r2, poly4_r2],
            [l_rmse, l_cook_rmse, lasso_rmse, ridge_rmse, poly2_rmse, poly3_rmse, poly4_rmse],
            [l_a, l_cook_a, lasso_a, ridge_a, poly2_a, poly3_a, poly4_a],
            [l_b, l_cook_b, lasso_b, ridge_b, poly2_b, poly3_b, poly4_b],
            ['/', '/', '/', '/', poly2_c, poly3_c, poly4_c],
            ['/', '/', '/', '/', '/', poly3_d, poly4_d],
            ['/', '/', '/', '/', '/', '/', poly4_e],
        ],
        title='All year | SMART16_NEW_PM_8H | PM2.5 | y = a + b*x + c*x^2 + d*x^3 + e*x^4',
        regs=['Linear', 'Linear Robust (Cook)', 'LassoCV', 'RidgeCV', 'Polynomial (d=2)', 'Polynomial (d=3)',
              'Polynomial (d=4)']
    )
