import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from regression.regression_models import get_polynomial_model
from regression.regression_summary import avg
from utils.utils import get_dataset, Dataset

if __name__ == '__main__':
    d = get_dataset(Dataset.SMART16_NEW_PM_8H)
    print(d)

    X = d['airqino_pm2.5'].values.reshape((-1, 1))
    y = d['arpat_pm2.5'].values

    aa, b, c, d, e, f, g, lasso, ridge = [], [], [], [], [], [], [], [], []
    for _ in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

        linreg = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
        linreg.fit(X_train, y_train)
        y_pred = linreg.predict(X_test)
        print()
        print('Linear Regression:')
        # print(" COEFF: ", linreg.coef_)
        # print(" INTERCEPT: ", linreg.intercept_)
        # print(" R-Squared", r2_score(y_test, y_pred))
        # print(" root_mean_squared_error", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        aa.append(r2_score(y_test, y_pred))

        cook_model_result = sm.OLS(y_train, sm.add_constant(X_train)).fit()
        infl = cook_model_result.get_influence()
        soglia = 4 / len(X)
        (a, p) = infl.cooks_distance
        mask = a < soglia
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_train[mask], y_train[mask], test_size=0.3)
        linreg = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
        linreg.fit(X_train_new, y_train_new)
        y_pred = linreg.predict(X_test_new)
        print()
        print('Cook Robust Linear Regression:')
        b.append(r2_score(y_test_new, y_pred))

        polreg = make_pipeline(StandardScaler(with_mean=False), get_polynomial_model())
        polreg.fit(X_train, y_train)
        y_pred = polreg.predict(X_test)
        print()
        print('Polynomial Regression:')
        # print(" COEFF: ", linreg.coef_)
        # print(" INTERCEPT: ", linreg.intercept_)
        # print(" R-Squared", r2_score(y_test, y_pred))
        c.append(r2_score(y_test, y_pred))
        # print(" root_mean_squared_error", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        polreg = make_pipeline(StandardScaler(with_mean=False), PolynomialFeatures(degree=3, include_bias=False),
                               linear_model.LinearRegression(n_jobs=-1))
        polreg.fit(X_train, y_train)
        y_pred = polreg.predict(X_test)
        g.append(r2_score(y_test, y_pred))

        kr = make_pipeline(StandardScaler(with_mean=False),
                           KernelRidge(alpha=1, kernel='poly', degree=2, gamma=1, coef0=1))
        kr.fit(X_train, y_train)
        y_pred = kr.predict(X_test)
        # print(" COEFF: ", kr.dual_coef_)
        # print(" R-Squared", r2_score(y_test, y_pred))
        d.append(r2_score(y_test, y_pred))
        # print(" root_mean_squared_error", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        kr = make_pipeline(StandardScaler(with_mean=False),
                           KernelRidge(alpha=1, kernel='poly', degree=3, gamma=1, coef0=1))
        kr.fit(X_train, y_train)
        y_pred = kr.predict(X_test)
        e.append(r2_score(y_test, y_pred))

        kr = make_pipeline(StandardScaler(with_mean=False),
                           KernelRidge(alpha=1, kernel='poly', degree=4, gamma=1, coef0=1))
        kr.fit(X_train, y_train)
        y_pred = kr.predict(X_test)
        f.append(r2_score(y_test, y_pred))

        lassoregcv = LassoCV(n_alphas=100, random_state=1)
        model = make_pipeline(StandardScaler(with_mean=False), lassoregcv)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        lasso.append(r2_score(y_test, y_pred))

        alpha_range = 10. ** np.arange(-2, 5)
        ridgereg = RidgeCV(alphas=alpha_range, scoring='neg_mean_squared_error')
        model = make_pipeline(StandardScaler(with_mean=False), ridgereg)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        ridge.append(r2_score(y_test, y_pred))

    a1 = avg(aa)
    a2 = avg(b)
    a3 = avg(c)
    a4 = avg(d)
    a5 = avg(e)
    a6 = avg(f)
    a7 = avg(g)
    a8 = avg(lasso)
    a9 = avg(ridge)

    print('linear', a1)
    print('cook', a2)
    print('poly', a3)
    print('poly d3', a7)
    print('kridge', a4)
    print('kridge d3', a5)
    print('kridge d4', a6)
    print('lassoCV', a8)
    print('ridgeCV', a9)
