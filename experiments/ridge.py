import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.utils import get_dataset, Dataset

if __name__ == '__main__':
    d = get_dataset(Dataset.SMART16_NEW_PM_8H)
    print(d)

    X = d['airqino_pm2.5'].values.reshape((-1, 1))
    y = d['arpat_pm2.5'].values

    plt.scatter(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    linreg = LinearRegression()
    model = make_pipeline(StandardScaler(with_mean=False), linreg)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.plot(X_test, y_pred, label='Linear')

    print()
    print('Linear Regression:')
    print(" COEFF: ", linreg.coef_)
    print(" INTERCEPT: ", linreg.intercept_)
    print(" R-Squared", r2_score(y_test, y_pred))
    print(" mean_absolute_error", metrics.mean_absolute_error(y_test, y_pred))
    print(" mean_squared_error", metrics.mean_squared_error(y_test, y_pred))
    print(" root_mean_squared_error", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    lassoregcv = LassoCV(n_alphas=100, random_state=1)
    model = make_pipeline(StandardScaler(with_mean=False), lassoregcv)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.plot(X_test, y_pred, label='Lasso')
    print()
    print('Lasso Regression:')
    print(' Best alpha:', model.steps[1][1].alpha_)
    print(' New Coef:', model.steps[1][1].coef_)
    print(' New Intercept:', model.steps[1][1].intercept_)
    print(" R-Squared", r2_score(y_test, y_pred))
    print(" mean_absolute_error", metrics.mean_absolute_error(y_test, y_pred))
    print(" mean_squared_error", metrics.mean_squared_error(y_test, y_pred))
    print(" root_mean_squared_error", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    alpha_range = 10. ** np.arange(-2, 5)
    ridgereg = RidgeCV(alphas=alpha_range, scoring='neg_mean_squared_error')
    model = make_pipeline(StandardScaler(with_mean=False), ridgereg)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.plot(X_test, y_pred, label='Ridge')
    print()
    print('Ridge Regression:')
    print(' Best alpha:', model.steps[1][1].alpha_)
    print(' New Coef:', model.steps[1][1].coef_)
    print(' New Intercept:', model.steps[1][1].intercept_)
    print(" R-Squared", r2_score(y_test, y_pred))
    print(" mean_absolute_error", metrics.mean_absolute_error(y_test, y_pred))
    print(" mean_squared_error", metrics.mean_squared_error(y_test, y_pred))
    print(" root_mean_squared_error", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    plt.legend()
    plt.show()
